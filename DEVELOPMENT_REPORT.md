# 开发报告

## 1. 已完成内容

- 创建统一的 Image2EditablePPT Web 项目。
- 实现图片上传、预处理、OCR、版面分析、OCR mask、背景修复、Layout JSON、Fabric 编辑器、PPTX 导出和质量验证。
- 项目已上传到 Google Drive 的 `图片PPT/Image2EditablePPT` 文件夹。

## 2. 实际项目结构

包含 `frontend/`、`backend/`、`shared/`、`uploads/`、`outputs/`、`temp/`、`scripts/`、`docs/`。上传版本没有包含 `node_modules`、Python 虚拟环境、OCR/VLM 模型、运行时 smoke 输出或本地缓存。

## 3. 使用了哪些参考项目的设计

- `BrainChen/image2ppt`：预处理、OCR/版面分析/PPTX 的分阶段流水线和 outputs 中间产物。
- `ningzimu/image-to-editable-ppt-skill`：对象级重建、文字/几何/复杂视觉元素的取舍、validation 思路。
- `gavin-sparkols/photo-editing`：多图上传、OCR bbox、原始像素坐标编辑器、缩略图、JSON/PPTX 导出。
- `JadeLiu-tech/px-image2pptx`：OCR 引导文字 mask、OpenCV/LAMA inpainting fallback。

只重新实现设计，不直接复制上述仓库源码。

## 4. 实现的 API

已实现：

```text
POST /api/projects
POST /api/projects/{id}/images
POST /api/projects/{id}/analyze
GET  /api/projects/{id}/slides
GET  /api/projects/{id}/slides/{page}
PUT  /api/projects/{id}/slides/{page}
POST /api/projects/{id}/export/pptx
GET  /api/projects/{id}/export/pptx
GET  /api/health
```

## 5. OCR 实现情况

`OCRProvider` 先尝试 PaddleOCR，当前验证环境未安装兼容的 PaddleOCR 包，因此自动使用 RapidOCR fallback。真实 smoke 图片经过 RapidOCR 识别出 2 个文字对象，结果包含 bbox、置信度、字号估计和颜色估计。

## 6. 背景修复实现情况

按 OCR bbox 扩张生成文字 mask，使用 OpenCV Telea inpaint 清除文字。`LamaInpaintingProvider` 已作为可选 adapter，安装不可用时自动回退 OpenCV，不阻塞主流程。

## 7. Web Editor 实现情况

React + TypeScript + Vite + Fabric.js + Zustand + Axios。已验证：上传、OCR、页面缩略图、选择元素、右侧属性编辑、文字内容修改、拖动坐标更新、图形工具、删除/复制/层级按钮、页面删除/上下排序。

## 8. PPTX 导出实现情况

`PPTXRenderer` 按像素比例换算为英寸。文本写入原生 TextFrame，rectangle/roundedRectangle/ellipse/line/arrow 写入原生 Shape 或 connector，复杂视觉元素写入独立图片。清理后的背景作为独立背景对象写入，不把原始截图作为唯一背景再叠透明文字。

## 9. 测试结果

- 后端 pytest：`2 passed`。
- 前端 TypeScript：通过。
- 前端 Vite production build：通过；仅有 Fabric bundle 大于 500 kB 的性能提示。
- FastAPI 真实服务：`http://127.0.0.1:8000/api/health` 返回 `status: ok`。
- 真实 smoke：上传 PNG → RapidOCR → 生成 Layout JSON → 导出 PPTX → python-pptx 重新打开。
- validation：`valid: true`，1 页、2 个文本对象、1 个图片对象、1 个原生 Shape，OCR coverage `1.0`，无越界、NaN 或严重重叠。
- 浏览器验证：网页打开成功，画布显示对象，属性面板可编辑，拖动后 X/Y 从 `108/120` 更新到 `158/150`。

## 10. 当前已知问题

- 当前环境未安装 PaddleOCR，因此默认显示 RapidOCR fallback 提示。
- 复杂渐变、图表和特殊字体仍采用保守的图片/背景保留策略。
- 规则形状检测不是 VLM 级别，复杂页面需要后续接入可选 VLMProvider。
- 前端生产 bundle 有 Fabric.js 体积提示，不影响运行。

## 11. 下一阶段建议

接入可选 VLM 页面理解、改进图片区域检测、增加 PPTX 渲染截图对比、接入更精确的字体测量和多行文本样式，并补充多页复杂科技类页面回归样例。

