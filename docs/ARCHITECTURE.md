# 架构说明

## 数据流

`Upload → Preprocess → OCRProvider → LayoutService → InpaintingProvider → Layout JSON → Fabric Editor → PPTXRenderer → Validation`

## 适配器

- `OCRProvider`：PaddleOCR、RapidOCR，后续可接 API/VLM OCR。
- `InpaintingProvider`：OpenCV、LAMA；LAMA 失败自动回落 OpenCV。
- `VLMProvider`：当前保留在 layout 服务的扩展边界，第一版不依赖付费 API。

## 运行时目录

`uploads/{project_id}` 存源图；`outputs/{project_id}/normalized` 存规范化 PNG；`backgrounds` 存去文字背景；`assets` 存独立图片裁剪；`slides` 存 Layout JSON；根目录 `editable.pptx` 和 `validation.json` 为导出结果。

