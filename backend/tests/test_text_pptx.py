from io import BytesIO

import app.main as api
import pytest
from app.services.pptx import text_deck
from app.services.pptx.master_export import MasterExportError, require_master
from app.services.settings.runtime_settings import ProviderSettings, VisionSettings
from fastapi.testclient import TestClient
from pptx import Presentation


def example_plan():
    points = [
        text_deck.SlidePoint(
            title="记录实验条件", body="保存数据版本、运行参数和随机种子。"
        ),
        text_deck.SlidePoint(
            title="复查结果", body="用相同输入重新执行脚本，核对结果。"
        ),
    ]
    return text_deck.DeckPlan(
        slides=[
            text_deck.PlannedSlide(
                title="可复现的实验记录", takeaway="为每次实验保存可核查的上下文。"
            ),
            text_deck.PlannedSlide(
                title="执行前定义记录方式",
                takeaway="明确版本与参数。",
                relationship="sequence",
                points=points,
            ),
            text_deck.PlannedSlide(
                title="复查是交付条件", takeaway="用同一套标准复查。", points=points
            ),
        ]
    )


def test_text_export_uses_native_ppt_master_compiler(monkeypatch):
    try:
        require_master()
    except MasterExportError:
        pytest.skip("PPT Master skill is not installed on this machine")
    calls = []
    plan = example_plan()

    class FakeQwen:
        def __init__(self, vision_settings):
            self.http_client = self

        def _request(self, messages):
            calls.append(messages)
            return plan.model_dump_json()

        def close(self):
            pass

    monkeypatch.setattr(text_deck, "QwenProvider", FakeQwen)
    provider = ProviderSettings(
        enabled=True,
        api_key="test-only",
        base_url="http://example.invalid",
        model="fixture",
    )
    monkeypatch.setattr(
        text_deck,
        "load_vision_settings",
        lambda: VisionSettings(providers={"qwen": provider}),
    )
    monkeypatch.setattr(api, "PUBLIC_SHARED_MODE", False)
    response = TestClient(api.app).post(
        "/api/export/text-pptx",
        json={"content": "如何记录实验结果？", "slideCount": 3, "style": "academic"},
    )
    assert response.status_code == 200, response.text[:200]
    assert len(calls) == 1 and "如何记录实验结果？" in calls[0][-1]["content"]
    assert text_deck._planning_guidance() in calls[0][0]["content"]
    assert response.headers["content-disposition"].endswith('text-ppt-master.pptx"')
    deck = Presentation(BytesIO(response.content))
    assert len(deck.slides) == 3
    for slide, expected in zip(deck.slides, plan.slides):
        assert any(s.has_text_frame and s.text == expected.title for s in slide.shapes)
        assert all(s.shape_type != 13 for s in slide.shapes)

    def point_titles(slide):
        return [
            shape
            for shape in slide.shapes
            if shape.has_text_frame and shape.text in {"记录实验条件", "复查结果"}
        ]

    step_one, step_two = point_titles(deck.slides[1])
    assert step_one.left == step_two.left and step_one.top < step_two.top
    left_point, right_point = point_titles(deck.slides[2])
    assert left_point.left < right_point.left and left_point.top == right_point.top


def test_missing_skill_fails_before_sending_user_content(monkeypatch, tmp_path):
    monkeypatch.setenv("PPT_MASTER_SKILL_DIR", str(tmp_path / "missing"))
    monkeypatch.setattr(api, "PUBLIC_SHARED_MODE", False)
    monkeypatch.setattr(
        text_deck,
        "_plan",
        lambda *_: pytest.fail("Should not send text without compiler"),
    )
    response = TestClient(api.app).post(
        "/api/export/text-pptx", json={"content": "如何记录实验结果？", "slideCount": 3}
    )
    assert response.status_code == 503
    assert "PPT Master" in response.json()["detail"]


def test_missing_planning_guide_fails_before_provider_io(monkeypatch, tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in ("svg_quality_checker.py", "svg_to_pptx.py"):
        (scripts / name).touch()
    monkeypatch.setenv("PPT_MASTER_SKILL_DIR", str(tmp_path))
    monkeypatch.setattr(api, "PUBLIC_SHARED_MODE", False)
    monkeypatch.setattr(
        text_deck,
        "_plan",
        lambda *_: pytest.fail("Should not send text without planner guide"),
    )
    response = TestClient(api.app).post(
        "/api/export/text-pptx", json={"content": "如何记录实验结果？", "slideCount": 3}
    )
    assert response.status_code == 503


def test_blank_brief_cannot_start_generation(monkeypatch):
    monkeypatch.setattr(
        api,
        "generate_text_pptx",
        lambda _: pytest.fail("Should reject the request at validation"),
    )
    assert (
        TestClient(api.app)
        .post("/api/export/text-pptx", json={"content": "     ", "slideCount": 3})
        .status_code
        == 422
    )
