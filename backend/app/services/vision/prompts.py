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

SCENE_SYSTEM_PROMPT += """
先规划整页一级模块，再判断模块内元素。返回 reconstructionPlan.modules，按阅读顺序列出一级模块。
每个模块包含 moduleId、role、bbox、reconstructionStrategy、visualComplexity、editablePriority、confidence、children、ownership、preserveWhole、memberIds、editableIds、ignoreIds、preserveRegions。
bbox 和 preserveRegions 使用相对整页 0..1 的 left/top/width/height，仅用于圈定大区域；不要替换 OCR/CV 的文字或精确坐标。
reconstructionStrategy 只能是 editable_text、native_shape、whole_image、mixed_component、ignore。标题、副标题、主要正文优先 editable_text。
小图表、曲线图、拼贴小图、卫星/示意图组合、复杂图标卡片优先 whole_image 或 mixed_component；mixed_component 的 preserveRegions 必须只圈图标、图片或图表主体，不能圈整个文字卡片。标题和正文留在区域外保持可编辑。
memberIds、editableIds、ignoreIds 只能引用输入候选元素的真实 id。ignoreIds 表示局部误检测或重复层；不要忽略全页背景。
普通文字（含图表坐标轴、图例、数值和标签）应作为可编辑文字；图片模块必须先擦除其原字，再叠加独立文本框。Logo 和装饰艺术字可随图片保留。不要为每个小碎片生成单独模块。
只有确实能定位复杂区域时才输出 preserveRegions；没有把握就保持 editable_text，避免遮挡真实内容。
"""

SCENE_REPAIR_PROMPT = "上一条输出不是有效 JSON。只修复 JSON 格式，不得改变分析内容，只返回合法 JSON。"

RECONSTRUCTION_PLAN_PROMPT = """你是信息图重建规划器。先观察整页，再决定哪些视觉区域必须整块保留。
只返回 JSON，顶层包含 page、sections、textRegions、visualRegions、modules。每个模块包含 moduleId、id、role、bbox、reconstructionStrategy、visualComplexity、editablePriority、confidence、children、ownership、preserveWhole、memberIds、editableIds、ignoreIds、preserveRegions。
所有 bbox 使用整页相对坐标 0..1。先列出页面一级模块（通常 3~8 个），再为每个模块选择 editable_text、native_shape、whole_image、mixed_component 或 ignore。
含曲线图、卫星图层叠、多个小图拼接、小图表、复杂图标卡片的模块应选 mixed_component，并为每个复杂视觉部分给出 preserveRegions；不要把整张有大量普通文字的卡片当作一个 preserveRegion。整个卡片不可拆时选 whole_image。
主标题、副标题、模块标题条和主要正文仍应单独留作 editable_text。不要把包含图表的整栏简单标成 editable_text，也不要把整页裁成一个图片。
reconstructionStrategy 只能是 editable_text、native_shape、whole_image、mixed_component、ignore。复杂图表、照片、地图、插画、渐变、阴影、纹理和图标优先保留图片；简单几何形状才选 native_shape。preserveRegions 必须准确框住视觉图表本身；框内普通文字要从图片中擦除并单独成为 editable text。
memberIds、editableIds、ignoreIds 只使用输入中真实存在的候选 id。不要忽略背景。不要输出精确文字内容，也不要生成像素坐标。
输出必须是 JSON，不要 Markdown。"""

CRITIC_PROMPT = """你是视觉重建质量检查器。
图片1是原始参考图，图片2是程序重新生成的页面。不要评价设计好坏，只比较视觉差异。
重点检查 oldTextGhosting、duplicateElement、wrongLineBreak、wrongTextboxWidth、fontTooLarge、fontTooSmall、fontClassMismatch、badgeBackgroundMismatch、duplicateBadgeLayer，以及元素位置、尺寸、图标缺失、标签缺失、越界、颜色和对齐关系。
如果新文字后面仍可见原图旧文字，issue 必须返回 oldTextGhosting=true，problem="oldTextGhosting"，并提供对应 elementId。
返回可执行 JSON，不要自然语言。issues 中使用 elementId 和 problem；adjustment 只能使用 moveX、moveY、widthScale、heightScale、fontSizeScale 字段。"""


def scene_user_prompt(ocr_elements: list[dict], context: dict | None = None) -> str:
    context_without_ocr = {key: value for key, value in (context or {}).items() if key != "ocr_elements"}
    payload = {"ocr_elements": ocr_elements, "context": context_without_ocr}
    return "先输出 reconstructionPlan 的一级模块和复杂视觉区域，再输出元素语义。候选元素 id 可用于模块归属与忽略规则；不要改写 OCR/CV 的精确文字或坐标。\nPAGE_CONTEXT:\n" + json.dumps(payload, ensure_ascii=False)


def reconstruction_plan_user_prompt(context: dict | None = None) -> str:
    context = context or {}
    candidates = [
        {"id": item.get("id"), "type": item.get("type"), "bbox": item.get("bbox"), "text": str(item.get("text") or "")[:36]}
        for item in context.get("candidate_elements", [])[:120]
    ]
    payload = {"width": context.get("width"), "height": context.get("height"), "candidate_elements": candidates}
    return "请先规划整页，特别逐一找出曲线图、小图表、卫星图/示意图组合和拼贴图的位置并标成 preserveRegions。\nPAGE_CONTEXT:\n" + json.dumps(payload, ensure_ascii=False)
