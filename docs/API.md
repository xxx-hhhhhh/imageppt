# API

## 创建项目

`POST /api/projects`

```json
{ "name": "Image2EditablePPT" }
```

## 上传图片

`POST /api/projects/{id}/images`，multipart 字段名为 `files`，支持 PNG/JPG/JPEG，多文件上传。

## 分析

`POST /api/projects/{id}/analyze`。同步执行当前版本的预处理、OCR、OpenCV/LAMA fallback、布局分析并保存每页 Layout JSON。

## 读取/保存页面

`GET /api/projects/{id}/slides`、`GET /api/projects/{id}/slides/{page}`、`PUT /api/projects/{id}/slides/{page}`。页面编号从 1 开始。

## 导出

`POST /api/projects/{id}/export/pptx` 返回 `downloadUrl` 和 validation；随后 `GET` 同一路径下载 PPTX。

