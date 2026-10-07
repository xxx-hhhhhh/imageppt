# API

## 创建项目

`POST /api/projects`

```json
{ "name": "Image2EditablePPT" }
```

## 上传图片

`POST /api/projects/{id}/images`，multipart 字段名为 `files`，支持 PNG/JPG/JPEG，多文件上传。

## 分析

`POST /api/projects/{id}/analyze?mode=fast|standard|high_quality|maximum`。新页面执行 OCR + 可选 VLM → SAM2/CV → 可靠对象所有权 → 紧文字局部修补 → 背景最后处理。已有 sceneVersion 3.0 页面只进行对象级候选 revision，未改善自动回滚，不再重新擦背景。原图指纹/页面顺序不匹配时保留已编辑页面并警告，不授权重建覆盖。

PUT 返回 canonical Layout（刷新 bbox 和 previewUrl），前端导出前保存当前图。缺少已确认 owner 的图片资产时，导出明确失败，不静默漏图。revision/QA 当前保存在 outputs，不新增数据库或用户系统。

## 读取/保存页面

`GET /api/projects/{id}/slides`、`GET /api/projects/{id}/slides/{page}`、`PUT /api/projects/{id}/slides/{page}`。页面编号从 1 开始。

## 导出

`POST /api/projects/{id}/export/pptx` 返回 `downloadUrl` 和 validation；随后 `GET` 同一路径下载 PPTX。

