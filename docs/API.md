# API

## 创建项目

`POST /api/projects`

```json
{ "name": "Image2EditablePPT" }
```

## 上传图片

`POST /api/projects/{id}/images`，multipart 字段名为 `files`，支持 PNG/JPG/JPEG，多文件上传。

## 分析

推荐 `POST /api/projects/{id}/analysis-tasks`，JSON：

```json
{ "page": 1, "mode": "maximum", "allowFallback": false }
```

立即返回 HTTP 202 与任务记录。`page` 从 1 开始，`mode` 为 `fast / standard / high_quality / maximum`。上一页需先确认。相同项目/页面/参数的进行中任务返回同一个编号；其他任务冲突返回 HTTP 409，不重复调用模型。当前 Windows 本地部署使用单个 Uvicorn API 进程，同时只运行一个分析任务。

- `GET /api/projects/{id}/analysis-tasks/{taskId}`：真实阶段、已用时间、总时限、状态、安全错误码。
- `POST /api/projects/{id}/analysis-tasks/{taskId}/cancel`：停止分析子进程；保留源图、候选和已完成页面。任务完成后再次取消不会撤销结果。
- `GET /api/projects/{id}`：项目、上传图片 URL、已保存页面、确认状态、最近任务。不返回本地图片路径或凭据。
- 状态：`running / publishing / succeeded / failed / cancelled / timed_out / interrupted`。
- 阶段：启动 → 预处理 → OCR → 分割 → AI 或本地布局 → 文字/对象重建 → 内容校验 → PPTX 校验 → 保存。
- 分析处理总时限 600 秒，包含 OCR、模型、对象重建和候选 PPT 验证。最后短暂的事务保存不可取消，以免半写入；保存前再次检查取消和截止时间。
- AI 分析共享 180 秒、最多 3 次 HTTP 请求预算（包括 JSON 修复、规划补全、代理路径）；单次最长 90 秒，无同路径自动重试。网络/鉴权失败不再启动新一轮规划请求。已有可用规划时，补全失败可保留 OCR owner，不丢原内容。
- AI 场景请求以流式方式收集完整 JSON，收到片段不代表重建成功，必须完成解析、owner 和 PPT 校验才发布；每个片段检查剩余预算。90 秒是网络无数据超时，不是对持续正常生成的一刀切。取消会停止本地进程和后续请求，不能承诺供应商已接收的请求不计费。
- 子进程仅写候选；父进程验证通过后才发布。取消、超时和失败不覆盖上版结果。服务重启将遗留任务标记 `interrupted`，不自动再次收费运行。

原 `POST /api/projects/{id}/analyze?mode=maximum&page=1` 保留原同步响应格式，但内部也使用同一有时限任务，不能绕过限制。新网页不使用这个长连接接口。

前端仅在 localStorage 保存项目 ID，支持 `/?project={id}`。刷新/新标签页恢复项目与任务；网络中断显示“恢复连接”，不会自动提交新任务。“基础模式”直接使用本地 fast 流程，不再次等待 AI。未点击保存的手动编辑不承诺跨刷新恢复。

## 读取/保存页面

`GET /api/projects/{id}/slides`、`GET /api/projects/{id}/slides/{page}`、`PUT /api/projects/{id}/slides/{page}`。页面编号从 1 开始。

## 导出

`POST /api/projects/{id}/export/pptx` 返回 `downloadUrl` 和 validation；随后 `GET` 同一路径下载 PPTX。

