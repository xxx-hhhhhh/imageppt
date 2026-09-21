# Image2EditablePPT V2 重建报告

## 目标

当前版本把原先针对单张党建图片的规则，升级为面向普通图片、PPT 截图、信息图、海报、流程图、卡片式版式和技术示意图的通用重建管线。

## 已完成的改造

- 用 Scene Graph 统一描述画布、区域、文本、形状、图片、组合、关系、样式 token、置信度和坐标系统。
- OCR 支持 RapidOCR，保留可替换的 OCR provider；文本分析增加行、段落、文本角色、字体、字号、粗细、颜色和对齐信息。
- Layout provider 支持 PP-Structure V3 的可选接入；当前环境未安装 PaddleOCR 时自动使用 OpenCV 通用回退，不依赖固定颜色、固定数量或固定坐标。
- 增加 OpenCV 通用分割回退、颜色/主题分析、组件分类、包含/邻近/对齐/重复关系、网格和层级分析。
- 增加几何优化：记录 rawBBox 与 refinedBBox，统一使用 source-pixels-left-top 坐标，减少元素偏移。
- 标签组由文本、形状和邻近关系推断，不再按党建图片的红色、位置、数量生成专用标签。文本不会被替换成单独圆点。
- PPTX renderer 保留文本、形状、图片和组合的可编辑对象，并写入字体族、字号、粗细、颜色、对齐和 zIndex。
- 增加 fast、standard、high_quality 三种模式；当前 standard 已实际执行 OCR → Layout → Scene Graph → PPTX。
- 增加重建预览、difference.png、visual_score.json、conversion_report.json 和原图/重建图切换入口。

## 测试与实际转换

### 后端测试

`python -m pytest backend/tests -q`：8 passed，1 warning。

### 前端

TypeScript `tsc -b` 通过，Vite production build 通过。仅保留 Vite 的 bundle 大小提示。

### RapidOCR

`python -c "import rapidocr_onnxruntime; print('rapidocr ok')"`：通过。

### 通用 fixture 转换

已生成并实际转换 5 张不同版式 fixture：simple text、business slide、technology、academic、infographic。结果为 5 页 PPTX，OCR provider 为 RapidOCR；视觉分数约为 0.9448–0.9769，平均约 0.9585。

### 原回归图转换

已实际生成 1 页 PPTX：

- 标签组：5
- 文本对象：28
- 图片对象：6
- 普通 shape：9
- Scene 元素：48
- OCR provider：rapidocr
- 视觉分数：0.9429
- 文本角色包含：main_title、section_title、badge、label_text、body_text、footer、iconWithText
- 字体角色已映射到 SimSun、Microsoft YaHei、KaiTi，并输出字号和粗细层级
- 输出坐标使用源图像素坐标，示例文本与组合均包含 x、y、width、height

## 当前限制

- 本机没有 PaddleOCR/PP-Structure V3，因此复杂表格、曲线图和极端透视版式仍使用 OpenCV 回退。接入 PaddleOCR 后可直接替换 provider，不需要改前端或 Scene Graph 协议。
- VLM provider 已提供可选远程/本地接口，但当前没有配置 VLM API key，因此本次转换没有调用外部模型。
- `difference.png` 当前作为项目级预览文件保存，分页项目的逐页评分仍分别写入 `visual_score_N.json`。
