# 开发报告

> 下方原报告记录初版历史状态。最新结果见文末“2026-10-07 Object First 重构验收”；OCR、测试数量和架构结论以该节为准。

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

## 2026-10-07 Object First 重构验收

### 范围、基线与旧流程问题

继续现有项目，基于远端 main `2989f6b`，先保存 checkpoint `febc07d`，在 `codex/object-first-main-integration` 开发。旧本地版本落后远端 172 个提交，因此没有整体合并旧分支的 pipeline。最新主线仍有提前写白底、SAM2 类名占位、bbox/资产声明代替像素归属证明，以及初始 critic 仅靠分数接受候选的问题。

本次完成安全核心数据流与真实分割集成；高保真对象重建尚未全部验收通过。完整架构说明见 `docs/OBJECT_FIRST_INTEGRATION.md`。

### 参考设计与实际代码使用

- `guohuan-xie/image2PPT`：SAM 轮廓与透明复杂资产、scene graph、阶段产物。参考版本 `1847650` 无明确 LICENSE，未复制代码、测试、图片或权重。
- `BrainChen/image2ppt`：统一 AST、源像素坐标、outputs/assets/intermediate 分层；MIT，未复制源码。
- `JadeLiu-tech/px-image2pptx`：OCR 引导的 tight ink mask、局部修复与重建文字；MIT，未复制源码。
- 可选真实 SAM2 推理通过 Ultralytics，AGPL-3.0/商业许可，不能按上述 MIT 项目处理。依赖和权重在源目录外；详见 `LICENSE_NOTES.md`、`docs/REFERENCES.md`。

### 新流程、Scene Graph 与 owner

源图 → OCR + 页面语义 + SAM2/CV 分割 → 现有 Scene/Layout → editable text / native shape / transparent image → 像素归属证据 → 最后生成页面底板 → tight ink 局部修补 → 再次验证归属 → 同一 Layout 的 editor / preview / PPT renderer。

`slides/page_N.json` 是唯一可编辑状态；`scene_final_N.json` 为初始解析审计快照。元素包含已有 id/type/role/groupId/zIndex/confidence，加上 owner/source/bbox/parent/contour/assetPath/editable/reconstructionStrategy；保存时 bbox 从 x/y/width/height 派生。OCR 正文不被 VLM 改写。修复 PPT renderer 优先读取旧 OCR lines 的问题，现在用户编辑的 `text`（包括主动清空）是正文来源。

Owner 采用 editable_text、native_shape、movable_image、background、intentional_ignore 五类。自动流程未授予 intentional_ignore 删除权限。图像须真实可读且 alpha/RGB 与源像素匹配；shape 须有几何/颜色证据；文字须有可靠 OCR 与 tight ink；旋转不明、NaN、缺失资产、低置信度文字不能授权丢弃未知内容。所有未知像素先写为独立透明资产，并重读校验，之后才允许写底板。失败即保留源背景并报错。低对比浅色模块也进入该门禁。

保守兜底仍可能形成跨对象残余裁切和较多图片。新增稀疏碎片聚合，仅在新资产的 alpha/RGB union 与旧资产完全一致时替换 AST 节点；旧资产文件继续保留。

### SAM2 与 LaMa

真实 Ultralytics SAM2 本地 CPU 推理已接入原 scene/planner，CV 提示框 → mask → contour → 原始像素 RGBA PNG。权重为外部 `sam2.1_t.pt`，用 SAM_MODEL_PATH 指定；缺依赖/权重/推理失败明确显示 OpenCV fallback，不假称 SAM 成功。提示覆盖目前依赖 CV，尚无完整语义自动提示。

LaMa 降为授权局部修补工具。`clean_array` 无 owner_mask 返回原图，支持 protected_mask；文字清理限制 tight ink，模型越界输出被裁回授权像素。VLM ghosting 不再直接调用整块 bbox reclean。实际四类回归使用本地 OpenCV 修补。随后从 Drive 启动脚本成功启动本地 LaMa，真实 adapter 推理 1 次成功、0 失败：党建页局部真实裁切的 3507 个授权 ink 像素，protected 区与 mask 外像素完全不变。证据 `outputs/integration-runtime/lama_protected_qa.json`。这不是四类全页 LaMa 质量验收，也未更改用户持久化修图设置；服务不可用仍不阻塞 Web。

### 非破坏式 revision 与 QA

保留主线已有问题区域 revision/事务提交，新增 replacement ownership 下降拒绝；候选 scene、assets、background、preview、QA 存档。原资产 URL 不随拒绝候选失效。碎片数量减少必须有像素替代证明。初始 critic 也接入资产/对象/异常变白/视觉丢失门禁。

指标包含 editableTextCoverage、objectExtractionCoverage、movableVisualCoverage、missingVisualCount、missingAssetCount、backgroundResidualCount、ghostingCount、duplicateCount、shapeCount、imageAssetCount、visualAreaPreserved，并保留 unownedPixelCount 与分区审计。duplicateCount 当前是重复计划视觉像素的告警计数，尚不是完整的语义重复对象统计。

### 四类真实回归与连续两轮

四例均执行真实 PaddleOCR + SAM2、PPT 生成与 python-pptx 重开，未调用付费 VLM，未修改 Qwen 设置。证据在本地 `outputs/object_first_integration_regression.json`；测试原图与运行产物不提交 Git、不复制 Drive。

- 简单页面：真实党建页标题裁切，项目 f88b3cc3cc5b498b922a1ea0dcca0c7d。4 文本、0 shape、47 图片对象（含底板）；editableTextCoverage 100%，visualAreaPreserved 100%，ghostingCount 3。
- 党建复杂页面：f01f132a4f934909943eeb251844ae3e。26 文本、5 shape、108 图片；文字覆盖 89.66%，视觉保留 100%，ghostingCount 1。图标、红绸、建筑、浅色模块底板仍存在。
- 科研信息图：7471f0e018684933957b3575ca46ff8a。91 文本、11 shape、159 图片；文字覆盖 94.06%，视觉保留 99.93%，ghostingCount 6。卫星、地图、山体、地球、浅色承载块仍存在。
- 图表/图片多页面：d685f324bead484991b1dc9bcf63c10f。93 文本、14 shape、132 图片；文字覆盖 93.69%，视觉保留 99.93%，ghostingCount 5。照片、图表、表格、色带、Logo 仍存在。

四例最终 missingAssetCount=0、unownedPixelCount=0、重要 missingVisualCount=0，审计未发现整页单张原图对象。简单例仍有 3 个无有效归属证据的越界文本框，源像素由保守图片 owner 保留；不能据此宣称文字布局完美。

每例连续两轮区域优化均未提交退化候选，公开预览 hash 保持不变。简单页第一轮虽分数上升却出现 1611 个无归属像素，回滚；第二轮无支持修改。党建第一轮出现 3279 个无归属像素及源视觉损失，回滚；第二轮无新目标。科研两轮分别出现 345/35 个无归属像素，回滚。多图页两轮分别出现 170/193 个无归属像素，回滚。这证明 rollback 边界有效，不代表优化已获得改善。

人工查看四张真实预览：重要视觉内容保留，但字体替代、标题/正文位置误差、重复 OCR 叠层和重影仍明显，特别是科研、多图页；高保真验收未通过。随后使用本机 PowerPoint 以只读、不显示窗口的方式打开四个真实回归 PPT 和浏览器下载 PPT，全部成功并输出原生渲染 PNG；文本/shape/图片计数与包结构检查一致，浏览器修改后的中文真实存在于 TextFrame 中。证据 `outputs/integration-runtime/powerpoint/verification.json` 与该目录 PNG。PowerPoint 原生截图复核也显示重影和图片边缘补丁，不能据“文件能打开”宣称视觉完美。WPS 尚未单独验收。

### API、编辑器、Windows 与修改文件

既有项目/上传/解析/页面读取保存/PPT 导出/health API 保留；未增加平行 API、数据库或登录系统。Qwen 设置、LaMa 状态、缩略图与逐页确认继续使用原前端。

修复双击编辑时 Ctrl+A 删除缩放手柄：文本内容与 React 控件分开，文字编辑时不启动对象拖动。真实 Chromium 全流程通过：上传 → PaddleOCR + Qwen standard 解析 → 双击改字 → 36px/#17365d → 拖动 → 手柄缩放 → 保存 → 确认 → 下载 PPT，确认修改后的文字在原生文本框中。无浏览器错误和资产 404。证据项目 `4cfe436331f94392adae01a048633ab3`，`browser_acceptance.json`、`browser_before_edit.png`、`browser_after_edit.png`、`browser_acceptance.pptx` 位于该本地 outputs 目录。截图人工复核通过编辑控件与布局检查，同时可见残余重影，未把视觉重建问题判为通过。另修正页脚固定显示 RapidOCR，改为读取本页实际 OCR metadata。

Windows setup/start/test 保持 PowerShell。Python venv、SAM 权重、前端 node_modules/Vite 执行副本均在 LOCALAPPDATA；不依赖 Drive 文件系统支持 junction。启动前同步前端源文件到外部执行副本。实际完成外部执行副本的 TypeScript/Vite build。

新增：owner_gate.py、requirements-segmentation.txt、test_owner_gate.py、test_fragment_compaction.py、test_segmentation_sam2.py、frontend-runtime.ps1、verify_owner_integration.py、verify_owner_browser.py、verify_revision_preservation.py、OBJECT_FIRST_INTEGRATION.md。

修改：pipeline/planner/white_objectization/text_erasure/layered_background/revision/revision_integrity、SAM provider/scene analyzer、inpainting service、project store、layout schema、PPT renderer、EditorCanvas/EditorPage、相关既有 tests、shared schema、Windows scripts、README、许可与参考文档、.env.example/.gitignore。

### 自动验证、提交与保存

- Backend：386 passed，2 项依赖 deprecation warning（75.45 秒）。
- Frontend：TypeScript 与 Vite production build 通过；外部运行副本 build 也通过。仍有既有 604 KB bundle 提示。
- Python lint：新模块检查与既有工程致命错误规则通过；PowerShell 解析与 git diff --check 通过。
- 后端实际启动 /api/health 返回 status=ok；SAM2 四例为实际推理，非 mock。
- 核心实现 commit：`4ab95d9fbd1cd1fe1774c24a201b6898aae3a4d4`，分支 `codex/object-first-main-integration`；报告与实际 OCR 页脚补充另作提交。
- OCR 页脚修复 commit：`20d0bd1a3f30c48e4c7dece524e28012fbe12db3`。上述实现已 push 到 `https://github.com/xxx-hhhhhh/imageppt/tree/codex/object-first-main-integration`，远端分支 SHA 已核对；未改远端 main。报告验收记录独立追加提交。
- 源码保存：`G:\我的云端硬盘\图片PPT\Image2EditablePPT`，214 个 Git 跟踪源文件/文档逐一复制与 hash 校验一致；未复制 node_modules、venv、OCR/SAM 权重、API Key、真实输入或 outputs。原有项目外内容及 uploads/outputs/temp/work 保留，未删除任何现有用户数据。
- 同步前备份 64 个将被覆盖的既有源文件并校验，位置 `C:\Users\lenovo\AppData\Local\Image2EditablePPT\source-backups\drive-20261007-152601`。原版本可以从该目录恢复。
- 实际从上述 Google Drive 中文路径运行 `scripts/start.ps1 -NoBrowser`，LaMa、Backend、外部前端执行副本健康检查均通过，网站 `http://127.0.0.1:5173/`，Backend `http://127.0.0.1:8000/api/health`。Drive 源目录没有 node_modules、.venv 或模型目录。

下一阶段优先：OCR 字体/行高/位置一致性、文字与复杂图片的双重归属解除、按模块的分割提示和残余合并、LaMa 在线回归、PowerPoint/WPS 实际渲染对比。避免通过减白/加白掩盖这些问题。

## 2026-10-07 Windows 中文路径阻断修复

用户反馈真实 Drive 项目连续四次解析 HTTP 500。复现发现上传、Pillow 预处理、Qwen 页面计划均正常；同一 normalized PNG 存在且可由 Pillow/字节解码读取，但 Windows OpenCV 4.10.0 的 `cv2.imread(filename)` 在 `G:\我的云端硬盘\图片PPT\...` 返回 None，`objectize_on_white` 随后抛 FileNotFoundError。API 捕获为笼统的“页面重建失败，请重试”。此前英文 checkout 完整回归通过与 Drive 启动健康检查，未覆盖 Drive 中文路径完整解析，验收范围存在遗漏。

新增 `backend/app/utils/image_io.py`，采用 Python 文件系统 I/O + OpenCV imdecode/imencode；统一替换 34 个后台模块中 164 个文件读写调用。未 monkey-patch 全局 OpenCV；图像仍采用原 BGR/BGRA、灰度、alpha、16-bit 与 codec 参数。读不到/写不了文件仍返回 None/False，Owner 门禁在资产写入/重读失败时继续拒绝清理，未关闭安全验证。

新增 `test_image_io.py`：中文/空格/emoji 路径、PNG 颜色/alpha/16-bit、JPEG 参数、灰度、缺失/空/损坏文件、写入失败，以及模拟原 OpenCV filename API 全部失效后的真实对象化流程；增加静态检查防止后台再次绕过统一 I/O。既有写入失败测试改为在新 I/O 边界注入故障。浏览器验收脚本增加可选 maximum 模式与可配置等待时间，明确区分此前 standard 验收。

已同步修复源文件并从 Drive 重启实际后台。保留所有用户上传、项目和持久化 Qwen/LaMa 配置；源码覆盖前备份于 `C:\Users\lenovo\AppData\Local\Image2EditablePPT\source-backups\unicode-fix-20261007-160056`。不复制 API Key、模型、venv 或 node_modules。

实际复跑失败项目 `73ab92f7f89b481087140e559b5a0925` 的原图，保持 `mode=maximum&page=1&allow_fallback=false`：HTTP 200，268.7 秒；真实 PaddleOCR + Qwen + SAM2，生成 `slides/page_1.json`、预览与 `editable.pptx`（6,215,132 bytes）。PPT 重开检查：1 页、86 个原生文本、3 个原生 shape、146 个独立图片对象（含页面底板）。Owner gate：unownedPixelCount=0、missingAssetCount=0、missingVisualCount=0；editableTextCoverage=94.95%，visualAreaPreserved=100%。证据为本机 `outputs/integration-runtime/unicode_recovery.json`，真实产物在该 Drive 项目输出目录。

该原项目尚未由用户确认页面，直接 POST export 返回 409“请先逐页确认全部重建结果”，属于既有导出门禁。未将用户项目自动标为通过。重建已实际完成且自动 PPT 文件已生成；浏览器全流程验收在独立测试项目执行。

自动验证：Backend 394 passed、2 个依赖弃用警告；新文件 lint 与后台致命错误检查通过；TypeScript/Vite production build 通过。最高质量重建文件生成阻断已解决，文字重影、字体与位置的高保真问题仍属于前节列出的限制。

Drive 实际浏览器完整验收也通过：独立项目 `2d04cec73cbc41139cdd2566db06e229` 上传同一张失败原图，真实 `maximum/allow_fallback=false` 请求，PaddleOCR + Qwen + SAM2 → 双击改字、字号颜色、拖动缩放 → 保存 → 确认本页 → POST 导出与 GET 下载均 HTTP 200。下载 PPT 重开确认修改后的文字为原生文本，无浏览器错误或资产 404；截图已人工检查。证据在该 Drive 项目的 `browser_acceptance.json`、`browser_before_edit.png`、`browser_after_edit.png`、`browser_acceptance.pptx`。本次不再用英文路径或 standard 模式替代 Drive 最高质量验收。

PowerPoint 以只读方式实际打开用户失败项目重建后的 Drive `editable.pptx`，确认 1 页、86 文本、3 shape、146 图片；中文路径与原生对象可用。修复后运行日志的 findDecoder 中文路径警告数量为 0。

实现 commit `cd223d0763991dcab86d3ffaac623857cedf292e`，仍在 `codex/object-first-main-integration`；验收记录追加提交。用户原项目没有自动标为通过，原图及既有上传均保留。源码与报告继续保存于同一 Google Drive 项目。

