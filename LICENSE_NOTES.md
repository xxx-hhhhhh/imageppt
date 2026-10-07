# 许可证说明

本项目为重新实现的统一 Web 应用，不直接复制四个参考仓库的源码。

参考项目与许可证（以各仓库当前 LICENSE 文件为准）：

- `BrainChen/image2ppt`：MIT
- `ningzimu/image-to-editable-ppt-skill`：MIT
- `gavin-sparkols/photo-editing`：MIT
- `JadeLiu-tech/px-image2pptx`：MIT

本项目自己的代码按本目录中的项目许可约定使用。运行时依赖（FastAPI、React、Fabric.js、python-pptx、RapidOCR 等）各自遵循其上游许可证。项目未复制上述仓库的代码、图片或模型文件，仅吸收公开文档中描述的架构思想，并在 `docs/REFERENCES.md` 中记录来源。

## Object First 重构（2026-10-07）

guohuan-xie/image2PPT 本次检出未见 LICENSE，仅参考公开架构与算法思路，没有复制源码或素材。BrainChen/image2ppt 和 JadeLiu-tech/px-image2pptx 本次检出的 LICENSE 为 MIT，未复制其代码。

可选 `ultralytics` 运行时采用 AGPL-3.0 或商业授权。不要把可选依赖视为 MIT 组件；闭源商业分发前须检查其完整授权条款。权重独立下载至外部模型缓存，不纳入 Git/Google Drive。保留所有原有参考和版权说明。
