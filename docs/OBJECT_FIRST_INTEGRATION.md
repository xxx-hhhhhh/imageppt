# Object First 集成（2026-10-07）

## Architecture gap analysis

以远端 main `2989f6b` 为基线，先保存 checkpoint `febc07d`。旧本地分支比远端落后 172 个提交，本次没有整体合并其 pipeline，保留最新模块底板、分组、资产归属与局部 revision 实现。

新版仍存在三个关键缺口：SAM2 只是类名占位；解析中提前写白底；文字清理与部分 foreground separation 把 bbox/资产声明当作像素所有权证据。初始 VLM critic 主要以全页分数决定接受，而区域 revision 已有更强门禁。

## 当前数据流

源图与 OCR、页面语义、SAM2/CV 分割进入现有 scene；planner、文字层、native shape、透明图像仍使用现有 Layout JSON。对象化完成后验证像素归属，再生成真正页面底板。后续清理之后再次验证归属，未知像素持久化为透明、独立图片。当前 `slides/page_N.json` 是编辑器、preview、PPT renderer 的唯一可编辑数据源，`scene_final_N.json` 是初始解析的审计快照，不是另一套 renderer 数据。

保留历史字段 `reconstructionSurfaceMode=white_objectized` 以兼容 UI；它不再意味着强制擦白。经页面边缘验证的平坦或单轴平滑渐变底层得以保留，模块浅色底板没有颜色亮度豁免。`legacy` erase-first 模式拒绝执行，防止配置绕过门禁。

## Object ownership

- `editable_text`：普通 OCR 正文不由 VLM 改写；有效文字对象与 tight ink mask 形成替代证据。低置信度 OCR 不授权清理。
- `native_shape`：几何轮廓、填充/描边颜色与原图匹配；仅 bbox 或 native 声明不构成证据。
- `movable_image`：资产真实存在、可读、alpha 与源图像素匹配；空图、缺图、NaN、旋转不明、裁切变换不明不能覆盖未知区域。
- `background`：经页面边缘验证的底层 field；不将浅色模块底板自动算作背景。
- `intentional_ignore`：当前自动流程没有授予此类删除权限。用户主动删除对象仍保持编辑器既有能力。

未知内容最差成为透明 source-pixel 图片；全部替代文件写入并重新读取验证后，才允许生成底板。保留资产使用唯一名称，不覆盖上轮资产。`owner`、`source`、`bbox`、`parent`、`assetPath`、`editable`、`reconstructionStrategy` 与已有 `role`、`groupId`、`confidence`、`zIndex` 位于同一 Layout 元素中。保存时 bbox 从 x/y/width/height 派生，避免用户移动后 AST 坐标过期。

## SAM2 与 LaMa

SAM2 是真实 Ultralytics 本地推理：CV 产生提示框，SAM2 生成轮廓 alpha，输出独立透明 PNG，接入已有 scene/planner。不是模型缺失时仍称作 SAM。依赖或推理失败明确退回 OpenCV。当前提示策略尚不是完整的语义自动分割器，可能遗漏 CV 未提出的对象；ownership fallback 保留这些内容。

模型在 `%LOCALAPPDATA%/Image2EditablePPT/models`，不在源目录或 Drive。使用 `setup.ps1 -EnableSegmentation` 安装可选依赖；`SAM_MODEL_PATH` 可指向外部权重。启动脚本检测本地权重。Ultralytics 的 AGPL-3.0/商业许可需在分发前评估。

LaMa `clean_array` 必须有 owner_mask，支持 protected_mask；未知或受保护像素保留。普通文字、planner 文字清理只改 tight ink。模型输出越界变白也不会提交越界像素。VLM ghosting 不能直接请求整块 bbox 擦除。LaMa 启动失败不阻塞 Web 基础功能。

## 非破坏式 revision 与 QA

继承已有区域修复及事务提交；对带 ownerGate 的页面增加 replacement ownership 下降拒绝。初始 critic 也必须通过资产、视觉丢失、异常变白、对象数量等 integrity gate。候选保存 scene、assets、background、preview、QA；拒绝候选的资产从公开目录移除前已保留独立副本。

新增/统一指标：objectExtractionCoverage、unownedPixelCount、missingAssetCount、visualAreaPreserved、missingVisualCount、shapeCount、imageAssetCount；继续使用现有 editableTextCoverage、movableVisualCoverage、backgroundResidualCount、ghostingCount 与 duplicate 审计。归属完整不等于语义分割完美，也不等于视觉评分满分。

## 参考与范围

借鉴 guohuan-xie/image2PPT 的 SAM 轮廓、透明复杂资产、scene graph 和阶段产物；BrainChen/image2ppt 的统一 AST 与源坐标；JadeLiu-tech/px-image2pptx 的 OCR 引导局部 ink mask。没有复制源码，也没有采用参考实现的先擦除再分割顺序。

此次优先完成安全边界。保守像素保留可能产生较多碎片图片，尚需按模块合并/关联；SAM2 提示覆盖、文本位置/字体与复杂页 ghosting 仍需继续优化。没有宣称已达到四类页面的最终高保真验收。实际运行证据见根目录 DEVELOPMENT_REPORT.md 及本地 outputs/object_first_integration_regression.json。
