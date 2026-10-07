# 普通文字可编辑与裁切边缘修复

## 为什么上一版几乎全是图片

上一版把“所选字体的像素轮廓近乎完全相等”和“背景近乎纯色”当作文字恢复前提。真实页存在压缩、抗锯齿、渐变、粗体/中英混排及字体差异，99 条真实 OCR 全部被拒绝，落入图片 Owner。保证了内容留存，却没有满足文字可编辑；不能称为完成。

硬二值 SAM Mask、互补的父子图像孔洞和细碎残留的独立 PNG 造成毛刺。文字恢复后还发现：即便源像素互斥，若同一处修补底色被复制到多张图片，PowerPoint 插值仍会形成白色接缝。后端预览不能代替实际 PPT 渲染。

## 本次代码

- `editable_text.py`：OCR 正文和原坐标不变；可靠 OCR + 可分离字形 + 周围底色证据决定 native text；字体按布局/尺寸和轮廓排序，不再要求完全相同。Windows 字体测量支持宋体、雅黑/粗体、Arial/粗体、楷体。
- 背景使用稳健二次颜色场从文字周围测量，支持局部渐变。Mask 仅包括字形和 3px 邻域中的压缩晕边，不是整块卡片或整页擦白。先持久化文本 Owner 和源 Mask，再修改这些像素。
- 拒绝低置信度、Logo、公式字符串、无法测量底色、冲突边线或无法匹配布局的文字，保留图片 Owner，且不叠同样的新字。
- `visual_assets.py`：高密度照片/图表/卡片保留完整矩形裁切，修复 SAM 内部孔洞；背景可测量的非规则对象使用轮廓简化、4x 超采样 Alpha、边缘去底色。边界不可靠则诚实退回完整局部裁切，不伪造透明抠图。
- `exclusive_ownership.py`：原像素 Owner 互斥；修补字形下的纸面另建 `paperSubstrateMask/BBox`，每个像素只由一张图片承载。文字间纸面岛归入连贯的文字承载对象，不生成几十个 1px 碎片。源素材/旧结果均不删除。
- `preservation_checks.py`：独立重读全部源 Owner 和纸面 Mask，生成“去掉全部 native text”的预览，区分字体替换、可测量羽化边界与不允许的非文字变化。
- `audit_raster_text`：从最终所有 raster 层检查原文字；密集小字的 bbox 中位数会误把纸面识别为字，新增从原图周围独立测量纸面的排除证据，不依赖已声明的删除 Mask。测试故意重叠原截图时必须检出重复。
- Pipeline/API：检测到可靠 OCR 但生成 0 native text 时拒绝成功，返回 409 / `EDITABLE_TEXT_RECONSTRUCTION_INCOMPLETE`；保存候选，保留已发布结果。新增 QA/去文字图纳入事务发布和回滚。

最终 Layout 仍是 editor、preview、PPT 的唯一渲染输入；源 Ownership 是只读出处证明，不是另一套场景。React UI、Qwen 设置、LaMa Adapter、OCR/PPT 接口及存储结构保留。未改 UI。

## LaMa 和自动优化

继续禁用自动 revision / downgrade，不启动 critic。LaMa 配置未改动，未调用 LaMa。本次采用有 Owner 授权的字形局部颜色场修补；试验的局部 Telea 在密集文字下产生斑驳，未作为最终实现发布。任何未知视觉继续保留为 movable image；没有 Owner 禁止删除。

## 回放证据（与新 API 验证区分）

真实已记录的 PaddleOCR/Qwen/SAM2 提案 `606b612355434c89af05c4c309c8b5d8` 经当前核心回放至 `846308e1a75144ef8cbdd9548cac2ecd`，新增 API 调用 0，不是模拟识别。

- 59 个原生文本框，27 个独立 movable image，1 个底层支持对象。原生 Shape 0，不能把本页图表称为可编辑数据图表。
- 未归属源像素 0；重复归属源像素 0；未归属字形纸面 0；重复字形纸面 0；缺失资产 0；源字重复检查 0；允许范围之外的非文字像素变化 0。
- 真正 PowerPoint 只读打开并导出全页及去文字页。去文字页没有之前明显的白色轮廓接缝；浅色底板和装饰均保留。
- 仍有浅色底板局部修补痕迹，复杂主标题/低置信度小字/公式仍在图片里。字体替换会有形态与排版细微差异。不得称为全页文字都可改、所有资产都是真透明或无任何视觉瑕疵。

机读文件在该记录的 `native_recovery.json`；PowerPoint 证据 `powerpoint_native.png`、`powerpoint_text_removed.png`；导出 `editable.pptx`。没有覆盖失败页 `73ab92f7f89b481087140e559b5a0925`。

## 验证与边界

通用自动测试涵盖 JPEG/渐变中文正文、字形局部修改、Logo/低置信度拒绝、整图叠加重复检出、图表孔洞修复、圆形透明羽化/去白边、可操作 409。没有真实页坐标硬编码。真实 API/浏览器完整回归结果见开发报告的新记录。

此轮集中修复当前失败页，未冒充完成四类真实页面回归。WPS 未完成本轮实际打开验证。源像素保留与语义对象提取完整性分开，`semanticObjectCoverage=null`，仍需更多真实输入审查。

## 新 API 与最终浏览器验收

使用用户已授权、保存在本机的 API 配置，真实上传原失败页并从浏览器执行 `maximum / allow_fallback=false`。独立新测试记录 `6b972f8b615e4f2092895a9dba16bd49`；`aiUsed=true`，PaddleOCR，Qwen3-VL-Flash，真实 SAM2。99 条 OCR，83 条置信度 ≥ 0.9，59 条变为 native text，35 张 movable image（Qwen 的本轮语义提案不同于回放），本页原生 Shape 0。

- 所有源像素、字形修补纸面的未归属/重复归属计数均为 0；非文字允许范围外变化 0；缺失资产和旧字重复检查 0。
- 实际浏览器双击改字、字号 36、颜色 `#17365D`、拖动、缩放、保存、确认并导出，全部通过；真实 PPT 验证修改后的原生 run 文字、字号比例和颜色。
- 用 PowerPoint 只读打开**未手动修改**的真实 API 导出并渲染，对比图为 `comparison_native.png`，不是网页截图冒充 PowerPoint。
- 编辑器另发现 DOM/Fabric 都绘制同一文字，且 Fabric/DOM 自动换行不同。最小修正：DOM 唯一可见文字层，Fabric 保留透明选择代理；测量过的单行文字按 PPT 的单行规则显示，不再自动拆行。不改变 UI 外观。
- 最终前端两次连续浏览器回归通过：改字、单行显示、图像拖动/缩放、控制台错误 0、媒体 404 为 0；Fabric 原生文字重复像素 0。初次测试在更新期间碰到“空页解析按钮应启用”的错误等待条件，已改为等待实际项目初始化，不用重试掩盖不稳定检查。
- 自动优化仍暂停，真实 revise 请求返回 409；调用自动 revision 轮数 0。
- Backend：402 passed / 2 deprecation warnings。核心与验收脚本 Ruff 通过，全后端致命错误静态检查通过。TypeScript / Vite build 通过，仍有 604KB bundle 大小警告。

`unmodified_reconstruction.pptx` 是原始重建；`browser_acceptance.pptx` 包含手动编辑回归文字；`native_acceptance.json`、`browser_acceptance.json`、`exclusive_browser.json` 是机读证据。旧用户页和素材未覆盖；新代码同步至原 Drive 项目，依赖/模型/密钥仍在 Drive 外。

本次使用 app-feature-craft / api-architect 保留原 API 与失败回滚契约，react-patterns / ui-ux / vite-react 限定编辑器最小修正；testing-playwright / visual-regression 要求真实双击编辑、可见文字重复检查和截图，不能仅凭后端计数宣布 UI 成功。
