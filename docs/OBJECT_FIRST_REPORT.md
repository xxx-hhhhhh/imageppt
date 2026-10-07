# Object First 重构实测报告（2026-10-07）

## 1. 范围和结论

继续当前 Image2EditablePPT，没有新建项目、切换框架或改动用户原始图片。保留 React/Fabric/Zustand、FastAPI/ProjectStore、Qwen 设置、OCR、LaMa adapter 和 python-pptx。

已完成安全优先的核心数据流重构。最高优先级规则有代码门禁和回归测试：没有有效 replacement owner 的像素不得删除；不能识别的内容成为独立透明图片或保持原样。尚未完成全量高保真验收，尤其是科研页文字残影、排版及复杂对象的语义级拆分。

运行仓库为本地原有 `google-drive-image2editableppt-web-ai-ocr`。Google Drive 的 `图片PPT/Image2EditablePPT` 已存在，但本次没有覆盖它：提交前发现远端 main 比本地基线多 172 个提交，涉及同一批核心模块，必须先确认采用哪个版本整合。模型、密钥、依赖和真实测试输出保留在本地，不进入 Git/Drive；云盘既有内容没有删除或修改。

## 2. 参考研究

- guohuan-xie/image2PPT，检出 `1847650`：阅读 pipeline、scene_graph、sam_segmenter、mask_postprocess、vectorize。借鉴真实 masks、透明 picture node、保守原生化及残余对象兜底；不采用先去字/两次修复顺序。未见 LICENSE，未复用源码。
- BrainChen/image2ppt，检出 `cba8d15`：阅读 pipeline、Slide AST、image_agent。借鉴统一 AST、坐标和阶段产物；不继承模型改写 OCR 或生成替代视觉。MIT，仅设计参考。
- JadeLiu-tech/px-image2pptx，检出 `6f4ad9f`：阅读 textmask、pipeline、inpaint。借鉴 ink AND OCR 紧 mask、单次局部修补，不把复杂内容留作整页不可移动背景。MIT，仅设计参考。

所有实现重新编写。Ultralytics 是可选 AGPL/商业授权依赖，不能视为 MIT；见 REFERENCES.md 和 LICENSE_NOTES.md。

## 3. 旧 pipeline 的主要问题

分割 adapter 没有真正调用 SAM；planner/徽章整块裁图可以压制正常正文；背景处理独立于最终输出对象，bbox 填充与 critic 重洗可能误伤浅色模块底板；scene/layout/preview 的策略分叉；优化直接覆盖，没有资产完整性/视觉保留门禁。旧结构验证还把 exportedText 直接等同于 recognizedText，没有读回 PPT 正文。

基线 checkpoint：`ef8f334`，树与旧版本 `52a2e08` 相同。旧 .gitignore 的 models/ 曾错误忽略 backend/app/models，本次改为根目录 /models/ 并纳入 ProjectStore 源码，保证提交包含必要存储模块。未跟踪的用户 AGENTS/技能未纳入提交。

## 4. 新架构

原图 → OCR 正文/位置 + 可选 VLM 语义 → SAM2/CV proposals → ObjectFirstBuilder → 可渲染 owner 和像素 lease → 紧文字 mask 单次修补 → native 几何 / RGBA 复杂对象 / 精确残余对象 → 最后生成页面底板 → 统一 Layout/Scene Graph → editor、preview、PPTX。

普通正文来自 OCR，VLM 的 text/style/精确位置不进入最终正文。VLM 只贡献 role、group、semantic type 和分割提示。模块底板也是视觉对象；严格几何/纯色一致性验证成功才 native 化，圆角/复杂对象宁可独立 PNG。

## 5. 文件清单

新增核心：

- backend/app/services/reconstruction/object_first.py
- backend/app/services/reconstruction/revisions.py
- backend/app/services/scene/ownership.py
- backend/app/services/visual_qa/preservation.py
- backend/requirements-segmentation.txt
- backend/tests/test_object_first.py
- scripts/regression_object_first.py、browser_object_first.py、regression_api_revisions.py
- docs/OBJECT_FIRST.md、OBJECT_FIRST_REPORT.md

修改：pipeline、layout/service、segmentation_provider、inpainting/service、PPTX renderer、validation/service、preview analyzer、vision prompts、ProjectStore、main PUT、Pydantic/shared schemas；前端 types、store、EditorCanvas、EditorPage、PageList、styles；相关测试和 README/架构/API/Schema/许可证/参考/开发报告。

没有新增账户、数据库、收费 API 或另一套应用。

## 6. SAM / segmentation

实际使用 Ultralytics SAM2.1 tiny，不是 OpenCV 冒充 SAM。官方权重位于 `%LOCALAPPDATA%/Image2EditablePPT/models/sam2.1_t.pt`，不在仓库/Drive。支持 SAM_MODEL_PATH、SAM_DEVICE、SAM_CPU_THREADS，当前 CPU 4 threads。

VLM 区域 + CV 轮廓 bbox 作为有限数量 prompts，真实 mask 回原图空间并保存 RGBA。相似 mask 去重；紧凑圆形徽章合并内部 artwork，保留真实轮廓和白色内部图案；大模块不吞掉子对象。SAM 缺失/失败明确降级 CV；再以残余轮廓和精确 spill 资产保证不因漏分割而丢内容。

限制：不是 segment-everything；无明确轮廓/语义提示的小对象可能合在残余资产内。三张复杂页均仍有一个覆盖较大区域的 movable 残余资产，不是全部对象语义分离完成。

## 7. Scene Graph / ownership

Layout version 1.2 / sceneVersion 3.0。`elements` 是唯一权威可修改列表，scene_refined 是它的派生快照。

对象含 id/type/role/x/y/width/height/bbox/rotation/zIndex/owner/mask/contour/parent/groupId/source/confidence/reconstructionStrategy/assetPath/editable。原图像素坐标不变；sourceBBox 和源空间 mask 记录提取证据，编辑后刷新 bbox。层级当前是 parent/groupId 元数据，不承诺已输出原生 PPT group。

owner：editable_text、native_shape、movable_image、background、intentional_ignore。文字须有 OCR 正文和紧墨迹证据，native shape 须几何验证，图片须真实存在且可解码；explicitUserIgnore 才允许忽略，VLM ignoreIds 不是删除许可。ledger 只授权已拥有像素。旧无-owner 背景 API 原样复制，重复 bbox reclean 禁用。已拥有资产缺失时 export 失败，不静默漏图。

保存、预览、缩略图及 PPT 都消费这一列表。修复了编辑 text 后 PPT 仍读取旧 lines 的问题，读回 PPTX 检查真实导出正文。前端导出前保存手动修改；较早的保存响应不覆盖后续编辑。

## 8. LaMa 新职责

LaMa/OpenCV adapter 仅处理确认 editable text 的紧墨迹，单次调用；protected mask 保护其他像素，模型在授权范围外的输出强制恢复为原像素。不允许扩大 bbox 擦模块、重复修补或越白越好。

保留 LaMa 接口/设置状态；当前用于真实回归的 Python 环境没有可用 simple-lama-inpainting，实际修补使用 OpenCV。测试用故意“整页擦白”的 adapter 验证 mask 外像素逐字节不变；这不是实际 LaMa 效果验证。

## 9. Revision / QA

每次保留 scene.json、assets、background.png、preview.png、qa.json、decision.json，唯一目录/资产名，不覆盖已有版本。候选只改已知问题对象的有限几何/字号，不换正文、不重生成/重洗整页。严格改善且未退化才 commit，否则返回原图结构。源指纹/页面顺序不匹配也不授权重建覆盖。

新增 editableTextCoverage、objectExtractionCoverage、movableVisualCoverage、missingVisualCount、missingAssetCount、backgroundResidualCount、ghostingCount、duplicateCount、shapeCount、imageAssetCount、visualAreaPreserved、backgroundWhiteArea、unownedVisualPixelCount 及逐对象记录。以实际 source masks + 有效 owner + 合成像素证据测量，不仅看 SSIM。

重要解释：editableTextCoverage 是“已识别 OCR 行中转 native 的比例”，不是原图所有字符准确率；objectExtractionCoverage=1 是源像素所有权覆盖，不等于高保真通过。ghosting 是单次修补后源墨迹残留估计，不是独立 OCR 校验；pixelSimilarity/旧 ssim-like 仍是启发式。字体及阅读语义准确性需要人工核查。

## 10. 四类真实回归

实际 OCR 使用 RapidOCR，四页均真实 SAM2 推理。没有注入假正文。简单案例来自真实党建页标题区域截取，不是完整纯文字整页；这一限制明确保留。

本地报告：`outputs/efb8a00eb72948e2a9b965734b6aac93/object_first_regression.json`。

- 简单标题截取：8 对象；1 文本、0 native shape、6 movable images。editableTextCoverage 1.0，visualAreaPreserved 1.0，missingVisual 0，ghosting 0。
- 当前党建“一总体思路”：51 对象；28 文本、1 native shape、21 movable images。文字覆盖 1.0，视觉保留 0.988，疑似视觉差异 3，残影估计 1。
- 科研迁移学习信息图：142 对象；86 文本、0 native shape、55 movable images。文字覆盖 0.9149，视觉保留 0.9757，疑似视觉差异 4，残影估计 23。
- 图表/图片较多的“取得成效”：87 对象；56 文本、2 native shape、28 movable images。文字覆盖 0.7671，视觉保留 0.987，疑似视觉差异 0，残影估计 10。

四页 missingAssetCount=0、unownedVisualPixelCount=0、objectExtractionCoverage=1、movableVisualCoverage=1。读回 PPTX：171 原生文本框、114 pictures（含每页背景）、3 native shapes。导出文字与最终图当前正文匹配率均为 1.0。低置信度文字保留为视觉，不代表全部普通文字已 editable。

截图检查：党建红绸/建筑/徽章、科研地图/卫星/图表/装饰/浅色承载板仍在；合成仍有重复墨迹和排版误差。圆形徽章透明轮廓/白色 artwork 由专门测试验证。不能据上述覆盖率声称所有重要视觉已完美重建。

## 11. 连续两轮优化

四页各两轮本地候选：全部 rolled_back / no_measured_improvement；对象 IDs、asset refs、数量和 QA 不下降，无 missing assets。没有宣称两轮改善。

另对真实 Qwen + SAM 党建测试项目 `626b2f3fe96d4f0f95735de7a0a6943b` 完成初次标准分析和两轮优化；并通过两个独立 HTTP 请求再次调用 Qwen critic，响应 aiUsed=true，均因未测量到改善回滚。记录：`outputs/626b2f3fe96d4f0f95735de7a0a6943b/api_revisions.json`。故意退化的单测确认资产减少、缺图增多、白化和视觉覆盖下降会拒绝，即使相似度分数上升。

## 12. 实际运行验证

- 基线 Backend pytest：67 passed；最终：85 passed，2 个上游 deprecation warnings。
- 新增/修改核心 Python 文件 Ruff check 通过；可选模型边界 broad exception 有明确安全 fallback 注释。
- frontend npm run build：TypeScript + Vite 通过。Fabric bundle 593 kB，仍有 >500 kB 提示；没有配置独立 ESLint，不把 tsc 冒充 ESLint。
- FastAPI 真实监听 127.0.0.1:8000，网页 127.0.0.1:5173。
- Chromium 完整流程：真实 PNG 上传、PaddleOCR 分析第五页、双击改字、字号/坐标、鼠标拖动/缩放、保存、切页、真实 PPTX 导出、重开检查改字及 text/shape/image 均通过，pageErrors=[]。
- 完整浏览器报告：`outputs/22987f7b0326497aa90830f4bef47323/browser_acceptance.json`。后续显示回归：`outputs/4130a5013e2d451b885b2139f97c829d/browser_acceptance.json`；等待所有图片真正加载后截图，整页可见，没有 asset 404。
- 修复切页 DOM/Fabric 节点冲突、异步 dispose、旧保存响应覆盖新编辑、拖动双重位移、空白缩略图、画布裁切及重复底层绘制。
- 未实际打开桌面 WPS/PowerPoint；只验证 OOXML 和 python-pptx 重开/原生对象类型，桌面兼容与像素级字体渲染仍需人工验收。

## 13. 已知问题 / 下一阶段

普通文字 mask 对抗锯齿、复杂底纹和细小字符仍有残影，不能重复擦白解决。下一阶段应做对象级 source ink 确信度与安全候选资产修订，在 owner/QA 门禁内修复，而不是扩大区域清洗。

进一步提升字体测量/文本框布局、语义级复杂对象分割、模块选择/原生分组、QA 重要区域权重和真正 PPT 渲染对比。混合图片宽高比沿用现有 deck 比例换算，尚未改为逐页等比 letterbox。需要补充完整纯文字整页和更多真实科研页面。

保留原图和失败候选，以便复查。当前优先实现的是不无主删除，而非“全可编辑”或“高保真已完成”。

## 14. 提交 / 推送

重构前 checkpoint：ef8f334。提交前 fetch 发现 origin/main 已到 2989f6b，相较 52a2e08 多 172 个提交、修改 106 文件（+15606 / -301），包括另一轮 ownership、revision、背景对象化和编辑器重构。本地可运行工作树不是远端最新架构。

本次不自动解决如此大范围的版本冲突、不强推 main、不覆盖云盘项目。已验证改动保存为同一仓库的独立安全分支 `codex/object-first-owner-guard-20261007`，不是新建应用或替换远端最新版本。

实现提交：`1380768`；`git push -u origin codex/object-first-owner-guard-20261007` 已成功。分支：[GitHub 验证分支](https://github.com/xxx-hhhhhh/imageppt/tree/codex/object-first-owner-guard-20261007)。远端 main 未修改；未创建 PR。本报告随后以文档提交补记，最终 HEAD 由 git log -1 确认。

下一步需要用户确认：以远端最新 main 为基线审查并迁移必要安全门禁，还是保留此次本地基线分支作为独立评估版本。本报告的 85 项测试只适用于此次本地分支，不是远端最新版的验收结果。
