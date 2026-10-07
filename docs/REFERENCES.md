# 参考项目

本项目在实现前阅读了以下公开仓库。实际代码采用重新设计的目录、数据模型和 API，不直接复制其源码。

## BrainChen/image2ppt

链接：https://github.com/BrainChen/image2ppt

借鉴：图片预处理、OCR/VLM/规则分析的流水线、Slide AST/Layout 中间层、`outputs` 中保留 AST、裁剪资产、日志和 PPTX 的可追踪中间产物。

实际使用：架构思想与输出分层；没有复制源码。仓库声明 MIT，具体版权以其 LICENSE 为准。

2026-10-07 审查：`backend/pipeline.py`、`backend/ast/slide_ast.py`、`backend/agents/image_agent.py`。统一 AST、坐标归一化和阶段产物用于本次 Object First 重构；没有复制代码、图片或版权文件。

## guohuan-xie/image2PPT

链接：https://github.com/guohuan-xie/image2PPT

审查：`src/image2pptx/pipeline.py`、`scene_graph.py`、`sam_segmenter.py`、`mask_postprocess.py`、`vectorize.py`。

借鉴：真实 SAM masks、透明 PNG 对象、保守 native geometry、残余视觉资产、scene graph 和中间 QA 产物。没有复制源码或测试图片。此次检出的仓库未发现 LICENSE 文件，按未明确授权处理，不复用实现。没有继承其双轮去字修复顺序。

## ningzimu/image-to-editable-ppt-skill

链接：https://github.com/ningzimu/image-to-editable-ppt-skill

借鉴：对象级重建原则、文字/几何/复杂视觉元素的取舍、页面级 reconstruction → validation → revision、测量驱动的文字框和源图/结果校验思路。

实际使用：重新实现为本地 FastAPI + React 服务，不依赖 Agent 环境。仓库声明 MIT，具体版权以其 LICENSE 为准。

## gavin-sparkols/photo-editing

链接：https://github.com/gavin-sparkols/photo-editing

借鉴：多图上传、OCR bbox、原始图片坐标、Canvas 拖动/缩放/双击编辑、多选、页面缩略图、顺序调整、JSON/PPTX 导出 API。

实际使用：重新实现为 Fabric.js + Zustand。仓库声明 MIT，具体版权以其 LICENSE 为准；本项目保留致谢和许可证记录。

## JadeLiu-tech/px-image2pptx

链接：https://github.com/JadeLiu-tech/px-image2pptx

借鉴：OCR bbox 扩张 mask、OCR 引导的 text-mask clip、OpenCV/LAMA inpainting fallback、按检测文字恢复原生文本框的原则。

实际使用：实现为 `InpaintingProvider`、OpenCV mask 和可选 LAMA adapter。仓库声明 MIT，具体版权以其 LICENSE 为准。

2026-10-07 审查：`textmask.py`、`pipeline.py`、`inpaint.py`。新实现采用 tight ink AND OCR、局部单次修补，mask 外像素强制还原。没有采用其复杂视觉仍留整页背景的最终装配方式。

## 可选 SAM 依赖

Ultralytics SAM/SAM2 调用遵循[官方接口](https://docs.ultralytics.com/models/sam-2/)。依赖采用 AGPL-3.0 或商业授权；分发/商业闭源使用必须另行审查授权。模型保存在 `%LOCALAPPDATA%/Image2EditablePPT/models`，没有放入 Drive 或 Git。未将 Ultralytics 源码复制进本项目。

