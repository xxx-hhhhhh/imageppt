# Object First, Background Last

## 重构前差距（2026-10-07）

- `OptionalSAM2Provider` 没有模型推理，分割结果仅记录在 scene，未成为真正的对象所有权证据。
- layout 的徽章、planner 的整模块 crop 可以压制正文；背景策略独立于最终 renderer，存在擦除了但没有输出 replacement 的风险。
- 背景 mask 和 critic 重洗使用 bbox，容易误伤标题条、浅色承载板；多轮结果直接覆盖，没有候选版本完整性门禁。
- scene、Layout JSON、preview 的策略推导各自独立，QA 主要统计相似度，无法证明重要视觉内容仍存在。

基线 commit：`ef8f334`（其树与原工作版本 `52a2e08` 相同）。未跟踪的用户技能和 AGENTS.md 保持原样，不纳入提交。

## 目标

保留现有 React/Fabric、FastAPI、ProjectStore、Qwen 设置和 python-pptx。OCR 正文/坐标是权威；VLM 只提供角色、模块和语义。所有候选先进入最终 Layout/Scene Graph，再进行像素所有权验证。没有有效 owner 的像素必须原样保留或成为独立 RGBA 残余资产。

底板、标题条、标签背景同样是对象。只有经纯色及几何一致性验证的候选转 native shape，其余由分割/透明 crop 承接。背景在对象资产落盘并验证后生成。LaMa/OpenCV 仅处理已确认 editable text 的紧 mask，一次修补，保护其他像素，并强制把模型在 mask 外的输出还原为原像素。

## 参考审查

- [guohuan-xie/image2PPT](https://github.com/guohuan-xie/image2PPT)：阅读 pipeline、scene_graph、sam_segmenter、mask_postprocess、vectorize。借鉴分割后生成透明 picture node、几何保守 native 化、残余资产和分阶段产物，不采用其先去字/两次 inpaint 的顺序。
- [BrainChen/image2ppt](https://github.com/BrainChen/image2ppt)：阅读 pipeline、Slide AST、image_agent。借鉴统一 AST、裁切资产和 intermediate 分层，不采用重新生成视觉或改写 OCR 文本。
- [JadeLiu-tech/px-image2pptx](https://github.com/JadeLiu-tech/px-image2pptx)：阅读 textmask、pipeline、inpaint。借鉴 ink AND OCR 的局部 mask，不沿用整页背景承载复杂对象的输出方式。

以上均重新实现，不复制源码。SAM 依赖可选，模型和运行时必须在 Drive 外。未知内容优先保真图片，不以全矢量化为目标。
