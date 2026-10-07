# 参考项目

本项目在实现前阅读了以下公开仓库。实际代码采用重新设计的目录、数据模型和 API，不直接复制其源码。

## 2026-10-07 Object First 重构补充

新增研究 `guohuan-xie/image2PPT`（https://github.com/guohuan-xie/image2PPT）：借鉴 SAM 分割、轮廓 alpha、复杂区域 PNG、scene graph 与阶段产物。检查的参考版本 `1847650` 没有明确 LICENSE 文件，因此未复制其代码、测试、图像或模型；仅参考设计思想。未采用其先对 SAM 输入进行整块文字清理的顺序。

再次研究 BrainChen/image2ppt 的 Slide AST/统一源坐标和 outputs/assets/intermediate；JadeLiu-tech/px-image2pptx 的 OCR 引导 tight ink mask 与局部修复。实现继续为本项目原创；不依赖参考仓库的 Agent 环境。

新增运行时可选依赖 `ultralytics` 用于真实 SAM2 推理，遵循 AGPL-3.0 或商业许可证，不能视作 MIT。权重由用户在源目录外安装，本仓库没有复制/分发权重；发布应用前需独立审查该依赖的分发许可。

## BrainChen/image2ppt

链接：https://github.com/BrainChen/image2ppt

借鉴：图片预处理、OCR/VLM/规则分析的流水线、Slide AST/Layout 中间层、`outputs` 中保留 AST、裁剪资产、日志和 PPTX 的可追踪中间产物。

实际使用：架构思想与输出分层；没有复制源码。仓库声明 MIT，具体版权以其 LICENSE 为准。

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

