"""File-backed, bounded analysis jobs. Heavy work never publishes from a worker.

One API process owns the supervisor (the supported local uvicorn deployment).
Spawn, not threads, lets cancellation stop native OCR/SAM and network calls too.
Only verified candidates are committed under the supervisor lock.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import queue
import threading
import time
from pathlib import Path
from uuid import uuid4

from app.models.project_store import ProjectStore

ACTIVE = {"running", "publishing"}
STAGES = {"starting": "启动分析进程", "preprocessing": "图像预处理", "ocr": "识别文字",
          "segmentation": "提取视觉对象", "vision": "AI 页面结构理解", "layout": "本地页面布局分析",
          "text_reconstruction": "匹配字体并重建可编辑对象", "validation": "检查内容完整性与重影",
          "pptx": "生成并验证 PPTX", "publishing": "保存已验证结果"}
PAGE_DEADLINE_SECONDS = 600


class TaskConflict(RuntimeError):
    pass


def analysis_worker(root: str, project_id: str, page: int, mode: str, fallback: bool, events) -> None:
    # Abrupt API exit must not leave a billable orphan worker behind.
    def parent_watchdog():
        parent = mp.parent_process()
        while parent is not None:
            if not parent.is_alive():
                os._exit(1)
            time.sleep(1)
    threading.Thread(target=parent_watchdog, daemon=True).start()
    from app.services.reconstruction.pipeline import ReconstructionPipeline
    try:
        pipeline = ReconstructionPipeline(ProjectStore(Path(root)))
        pipeline.analyze_project(project_id, mode, page, fallback,
                                 progress=lambda stage: events.put(("stage", stage)), prepare_only=True)
        events.put(("prepared", str(pipeline.prepared_candidate)))
    except Exception as exc:
        # Never serialize provider exceptions, prompts, paths or secrets to UI.
        code = type(exc).__name__
        events.put(("error", "AI_UNAVAILABLE" if code == "AIUnavailableError" else
                    "QUALITY_REJECTED" if code == "ReconstructionQualityError" else "ANALYSIS_FAILED"))


class AnalysisTasks:
    def __init__(self, store: ProjectStore, *, deadline_seconds=PAGE_DEADLINE_SECONDS,
                 worker=analysis_worker, publisher=None):
        self.store = store
        self.deadline_seconds = deadline_seconds
        self.worker = worker
        self.publisher = publisher
        self.lock = threading.RLock()
        self.live = {}

    def _path(self, project_id, task_id):
        self.store.get(project_id)
        if len(task_id) != 32 or any(c not in "0123456789abcdef" for c in task_id):
            raise FileNotFoundError(task_id)
        return self.store.root / project_id / "tasks" / f"{task_id}.json"

    def _save(self, task):
        path = self._path(task["projectId"], task["id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(task, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)

    def get(self, project_id, task_id):
        with self.lock:
            task = json.loads(self._path(project_id, task_id).read_text(encoding="utf-8"))
            if task["status"] in ACTIVE and task_id not in self.live:
                task.update(status="interrupted", message="服务已重启，上次任务中断；已完成页面仍保留，请重试。", finishedAt=time.time())
                self._save(task)
            task["elapsedSeconds"] = round((task.get("finishedAt") or time.time()) - task["startedAt"])
            task["stageLabel"] = STAGES.get(task["stage"], task["stage"])
            return task

    def latest(self, project_id):
        self.store.get(project_id)
        paths = list((self.store.root / project_id / "tasks").glob("*.json"))
        return self.get(project_id, max(paths, key=lambda p: p.stat().st_mtime_ns).stem) if paths else None

    def submit(self, project_id, page, mode, fallback=False):
        with self.lock:
            previous = self.latest(project_id)
            if previous and previous["status"] in ACTIVE:
                if (previous["page"], previous["mode"], previous["allowFallback"]) == (page, mode, fallback):
                    return previous  # Same page, including another tab: join, never redo.
                raise TaskConflict("本项目已有分析任务，请等待完成或先取消。")
            if self.live:
                raise TaskConflict("已有其他项目正在分析，请先完成或取消，避免重复占用 OCR/模型。")
            record = self.store.get(project_id)
            if not 1 <= page <= len(record["images"]):
                raise ValueError("Page not found")
            if page > 1 and page - 1 not in record.get("approvedPages", []):
                raise TaskConflict("请先确认上一页的重建结果。")
            task = {"id": uuid4().hex, "projectId": project_id, "page": page, "mode": mode,
                    "allowFallback": fallback, "status": "running", "stage": "starting",
                    "startedAt": time.time(), "deadlineSeconds": self.deadline_seconds,
                    "message": "分析已开始；刷新页面可恢复任务。"}
            context = mp.get_context("spawn")
            events = context.Queue()
            process = context.Process(target=self.worker, args=(str(self.store.root), project_id, page, mode, fallback, events), daemon=True)
            process.start()
            self.live[task["id"]] = (process, events)
            try:
                self._save(task)
            except OSError:
                self.live.pop(task["id"], None)
                process.terminate()
                process.join(timeout=3)
                events.close()
                raise
            threading.Thread(target=self._supervise, args=(task, process, events), daemon=True).start()
            return self.get(project_id, task["id"])

    def cancel(self, project_id, task_id):
        with self.lock:
            task = self.get(project_id, task_id)
            if task["status"] in ACTIVE:
                self._finish(task, "cancelled", "已取消分析，原图及已完成页面未改动。")
            return self.get(project_id, task_id)

    def _finish(self, task, status, message, code=None):
        worker = self.live.pop(task["id"], None)
        if worker:
            process, _ = worker
            if process.is_alive():
                process.terminate()
            process.join(timeout=3)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
        task.update(status=status, message=message, finishedAt=time.time(), errorCode=code)
        self._save(task)

    def _supervise(self, task, process, events):
        deadline = time.monotonic() + self.deadline_seconds
        try:
            while True:
                try:
                    event, payload = events.get(timeout=.2)
                except queue.Empty:
                    event, payload = None, None
                with self.lock:
                    if task["id"] not in self.live:
                        return
                    if time.monotonic() >= deadline:
                        self._finish(task, "timed_out", "本页超过总时限，已停止分析；旧结果保留，可重试或选择基础模式。", "PAGE_DEADLINE")
                        return
                    if event == "stage":
                        task["stage"] = payload
                        self._save(task)
                    elif event == "error":
                        self._finish(task, "failed", "分析未完成，旧结果保留。请检查 AI 连接或使用基础模式重试。", payload)
                        return
                    elif event == "prepared":
                        from app.services.reconstruction.pipeline import (
                            ReconstructionPipeline,
                        )
                        task.update(status="publishing", stage="publishing")
                        self._save(task)
                        # Short non-cancellable commit: cancellation/deadline has
                        # already been checked. A killed worker never gets here.
                        (self.publisher or ReconstructionPipeline.publish_candidate)(self.store, Path(payload))
                        self._finish(task, "succeeded", "本页重建完成，请对照原图检查。")
                        return
                    elif event is None and not process.is_alive():
                        self._finish(task, "failed", "分析进程意外退出；旧结果保留，请重试。", "WORKER_EXIT")
                        return
        except Exception:
            with self.lock:
                if task["id"] in self.live:
                    self._finish(task, "failed", "结果保存失败；旧结果保留，请重试。", "PUBLICATION_FAILED")
        finally:
            events.close()
