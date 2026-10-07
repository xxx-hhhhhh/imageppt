# 架构说明

## 数据流

`Upload → Preprocess → OCR → VLM semantic candidates → SAM/CV segmentation → ObjectFirstBuilder → validated pixel ownership → tight local text patch → independent RGBA/native objects → Background Last → canonical Layout/Scene Graph → Fabric Editor / Preview / PPTXRenderer → Preservation QA → candidate revision commit/rollback`

详细重构差距、规则和验证见 [OBJECT_FIRST.md](OBJECT_FIRST.md)。旧背景策略、planner、router 保留供旧格式测试/迁移使用，不再参与主解析与自动优化。

## 适配器

- `OCRProvider`：PaddleOCR、RapidOCR，后续可接 API/VLM OCR。
- `InpaintingProvider`：OpenCV、LAMA；LAMA 失败自动回落 OpenCV。
- `VLMProvider`：当前保留在 layout 服务的扩展边界，第一版不依赖付费 API。

## 运行时目录

`uploads/{project_id}` 存源图；`outputs/{project_id}/normalized` 存规范化 PNG；`backgrounds` 存底层背景（唯一命名，不覆盖旧版本）；`assets` 存透明资产与源像素 masks（唯一命名）；`slides` 是唯一可编辑对象列表；`revisions/page_N/r_*` 保留 scene、assets、background、preview、qa 和 commit/rollback decision。根目录 `editable.pptx` 和 `validation.json` 为导出结果。

