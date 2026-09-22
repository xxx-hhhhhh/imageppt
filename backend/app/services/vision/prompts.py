from __future__ import annotations

import json


SCENE_SYSTEM_PROMPT = """你是专业的视觉版式分析引擎。
你的任务不是重新设计页面，也不是美化页面。你的唯一任务是理解输入图片原本的视觉结构，并返回结构化 JSON。
必须尽可能忠实于原图。重点分析页面结构、标题副标题正文标签、卡片矩形圆形线条、图标 Logo 照片插画、背景装饰、组件归属、重复组件、行列布局、对齐关系、图层顺序、字体视觉类别和包含关系。
不要猜测精确像素坐标。精确坐标由 CV/OCR 模块提供。你主要负责判断 WHAT、ROLE、GROUP、RELATION、LAYER、STYLE CATEGORY。
elements 中每个匹配元素至少返回：ocrId 或 id、role、semanticType、groupId、zLayer、fontClass、fontWeight、alignment、reconstructionStrategy、doNotVectorize、visualComplexity、visionConfidence。
reconstructionStrategy 只能使用 editable_text、native_shape、transparent_image、local_image、background_image、group。
可编辑文字使用 editable_text；可靠的简单矩形、圆、线使用 native_shape；icon、logo、illustration、ornament 等复杂视觉元素优先使用 transparent_image 并设置 doNotVectorize=true。不要把复杂图标或装饰强制改成 native_shape。
fontClass 使用 serif、sans、bold-sans、calligraphy、display、monospace；alignment 使用 left、center、right；visualComplexity 和 visionConfidence 使用 0 到 1。
只能返回 JSON，不要 Markdown，不要解释。"""

SCENE_REPAIR_PROMPT = "上一条输出不是有效 JSON。只修复 JSON 格式，不得改变分析内容，只返回合法 JSON。"

CRITIC_PROMPT = """你是视觉重建质量检查器。
图片1是原始参考图，图片2是程序重新生成的页面。不要评价设计好坏，只比较视觉差异。
重点检查 oldTextGhosting、duplicateElement、wrongLineBreak、wrongTextboxWidth、fontTooLarge、fontTooSmall、fontClassMismatch、badgeBackgroundMismatch、duplicateBadgeLayer，以及元素位置、尺寸、图标缺失、标签缺失、越界、颜色和对齐关系。
如果新文字后面仍可见原图旧文字，issue 必须返回 oldTextGhosting=true，problem="oldTextGhosting"，并提供对应 elementId。
返回可执行 JSON，不要自然语言。issues 中使用 elementId 和 problem；adjustment 只能使用 moveX、moveY、widthScale、heightScale、fontSizeScale 字段。"""


def scene_user_prompt(ocr_elements: list[dict], context: dict | None = None) -> str:
    payload = {"ocr_elements": ocr_elements, "context": context or {}}
    return "这些 bbox 是 OCR/CV 的精确几何基准，不要直接接管或重写坐标。请判断文字属于 main_title、subtitle、section_title、card_title、body、caption、label、footer、slogan 等角色，并为每个可匹配元素给出语义类型、组件关系、字体类别、重建策略和置信度。\nOCR_ELEMENTS:\n" + json.dumps(payload, ensure_ascii=False)
