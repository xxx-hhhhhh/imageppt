# Object First 互斥归属审计（2026-10-07）

## 本轮目标与停止条件

先恢复“不丢失、不重影、不重复”。停止自动 critic、revision endpoint、service revision loop 和旧版 downgrade。不改 UI 外观，不运行连续自动优化。保留失败页面及原 Layout / 资产 / PPT / 预览；独立测试记录不覆盖用户失败结果。

## Git 基线说明

原工作分支 HEAD 是 `ea51c2f`，已保存在 Git；它能生成文件，但失败页视觉重影明显，不能称为视觉可用版本。

`ef8f334`、`febc07d` 虽然是“working/checkpoint”提交，现有证据不足以证明这张失败页在这些版本符合视觉验收。不能仅根据提交标题宣称最近可用。

新分支 `codex/object-first-exclusive-ownership` 从 `100474e` 建立。该提交是 `2bb16ab`（默认 white objectized 流程）的父提交，是可核实的“擦白重建引入前”源码锚点，不是已经验证视觉合格的历史版本。本分支显式保留 `ea51c2f` 中现有 React/API/PPT 数据契约、中文路径修复、SAM2 adapter 与用户编辑修复，重写生产 pipeline；未将整个应用退回旧依赖或旧 UI。

## 根因

1. 旧 owner gate 对像素掩码取并集。一个像素有 owner 就算保留，没有验证归属唯一性，无法识别图片里的旧字与新文本框同时出现。
2. tight mask 未覆盖的原字边缘被当作“尚未提取的视觉”再次裁成 residual / recovered visual。源文字复活后，新文本框还存在。
3. white_objectization、support recovery、planner transfer、ghost cleanup、residual recovery 轮流修改同一组元素。晚期 CV 回收和 early VLM bbox 重放会重新覆盖归属或层次。
4. VLM 模块计划含重叠补充区域。粗区域如果先抢走源像素，SAM 独立对象随后没有可认领内容，模块底板和标题会被切片。
5. 历史 QA 的 coverage / missingVisualCount 把“源像素仍在任意图片里”误当作正确重建；元素有图片并不证明它不重复、不重影或模块完整。

## 旧页面实际审计

证据页面：`73ab92f7f89b481087140e559b5a0925`，工作成效 / 表格 / 四个图表的失败页。

实际读取每个活动图片资产，在源图坐标重组仅 raster 层，然后在原 OCR footprint 内比较源文字 ink 是否仍可见。不是单纯看 bbox overlap。

检出至少 **10 处 raster/native 重复文字**；活动图片资产缺失 **0**。该检测是保守下界，不代表其余 89 条文字都正确。具体 id、源坐标、文字和残留 ink 比例在 `outputs/exclusive-ownership-regression/audit_before.json`。

旧审计无法证明“错误删除”发生于哪一轮：最终 Layout 不能逆推出所有历史删除调用。会保留源码、旧图、mask 与预览差异作为证据，不把所有渲染差异武断标成删除。旧结果声称 preservation=1 也不能推翻截图中的重影。

## 新生产数据流

源图预处理 → OCR 原文/源坐标 → Qwen 语义及模块提案 + SAM2 mask → 候选 Scene → 验证替代对象 → 互斥源像素 owner label → 复杂视觉透明 PNG / 不可靠文字随图片 → 最后识别 edge-connected 的精确平坦背景 → 候选 preview/PPT/QA → 全部通过才发布。

没有 white-objectization、补白、ghost recovery、suppression 或自动 critic。LaMa adapter 保留，但这条安全流程 **不调用 LaMa**；以后只允许用独立 owner 证据授权的局部 mask，不使用 `owner_mask=mask` 自我授权。

### Owner 规则

- `editable_text`：OCR 原文和位置；背景确为平坦、全部字像素能识别、字体轮廓匹配通过后才生成。不能接受就不叠文字。
- `native_shape`：此阶段仅像素验证通过的纯色、轴对齐矩形。其余形状保留图片，不伪装成 native shape。
- `movable_image`：SAM2 真 mask；或原像素的透明 crop / fallback。PNG 写入并重新解码逐像素相等后才能认领。
- `intentional_background`：最后检测与边缘连通的精确平坦色场。不按“浅色/白色”将卡片或承载块判成背景。
- 无 owner / asset 写入失败 / 重复 claim：禁止 background 发布；候选失败，不覆盖旧结果。

SAM 子对象先分配，父底板和语义模块只认领未被子对象占用的原像素。图片中的文字如果没有可靠 native 替代，保持在图片中，最终 Layout 不再包含第二层同样正文。

### Scene 与审计产物

唯一坐标权威为最终 Layout JSON 的 x/y/width/height；编辑器、preview、PPT 读取该 Layout。`bbox` 为派生值，`ownershipAudit.nodes` 是源像素归属证据，不是第二套可渲染场景。

每个 node 记录 id/type/role/bbox/sourceBBox/mask/parent/zIndex/owner/source/confidence/reconstructionStrategy/assetPath/editable/sourcePixelCount/verified。

产物包括 `ownership_labels_1.png`、`ownership_masks/<run>/...png`、`ownership_audit_1.json`、`raster_text_audit.json`、`scene_final_1.json`、候选 preview/QA/PPT。

候选写入 `candidates/<run>`；重建已有页先保留 `snapshots/<run>` 的旧资产、场景、底板、预览和 QA。资产使用唯一文件名，不删除旧 PNG。发布前验证 PNG、互斥 ownership、raster/native 重复文字和非文字区域像素差；PPT 候选生成也在发布前完成。

## 已知边界

本轮严格字体 gate 会降低编辑性，复杂失败页可能没有原生文字或形状。这是明确的保守 fallback，不能称为“完全可编辑 PPT”。优先确认完整性后，下一阶段才改进字体测量与普通正文可编辑覆盖。

PPT 的简单图形与文字能力由合成算法测试覆盖；真实失败页是否提取这些对象以实际计数为准。源像素互斥和 preview 无损不等于所有语义模块边界正确，SAM 会有局部 mask / 模块拆分问题，仍需人工查看单页与移动后的效果。

最终真实测试数据、PowerPoint 打开/渲染及浏览器验收将记录在本轮报告，不借用上一轮的成功记录冒充当前验证。

## 本轮单页结果

真实 Qwen/PaddleOCR/SAM2 初始解析分别运行于独立测试记录 `982e8ea4ed4949739582c99d8bd9a566`（209.58 秒）与 `606b612355434c89af05c4c309c8b5d8`（135.38 秒）。这是开发中两次独立初始解析验证，不是自动 revision。源图都是当前失败页，没有模型结果造假或测试图特判。

第一次发现粗模块会占走 SAM 像素，只生成 7 张大图片，未作为最终结果。第二次 SAM 子对象优先，生成 **49 个独立透明/裁切图片**和 **1 个底层支持对象**。99 条 OCR 文字均由图片保留；本页 **原生文字 0、原生 Shape 0**。这是视觉止损版本，尚未完成普通文字可编辑化。

PowerPoint 实际只读打开、导出 PNG 后发现 hard alpha mask 插值造成细白接缝。使用同一次真实 OCR/Qwen/SAM 提案离线重放到 `56cfb1790cb44643888fbb5880070fee`，没有新增 API 调用、没有 critic，也没有覆盖任何旧结果。只在已经生成可靠 Owner 的相邻资产下方添加 2px 的局部颜色延续底材；它不复制另一对象的原字，不认领另一对象的源像素。源像素标签仍然互斥；底材必须低于原 Owner 图层，不得覆盖它。

最终结果：

- PNG 预览与源图逐像素相等，差异像素 0。
- 逐个重新读取 source masks：无 Owner 源像素 0，重复 Owner 源像素 0。
- raster/native 重复文字 0（本页不存在 native text，本指标不能被解释为字体恢复成功）。
- 缺失/损坏活动资产 0，重复原视觉 claim 0。
- 浏览器真实读取保存的 API Layout；图像拖动、缩放通过，控制台错误和媒体 404 均为 0。
- PowerPoint 实际打开最终文件：1 页，50 个 picture 对象。没有声称 WPS 也完成本轮实测。
- PowerPoint 渲染仍有微小边缘插值差异，RGB 平均绝对误差由 4.3259 降到 4.2371（0–255 范围）。该值不是“保留百分比”，也不证明语义模块完美。
- 自动优化 / 降级入口返回 409，自动 revision 轮数 0。

对比图：`outputs/exclusive-ownership-regression/comparison_final.png`，第三栏是实际 PowerPoint 渲染，不是后端预览。机读证据：`audit_before.json`、`summary.json`、`seam_replay.json`、`final_acceptance.json`，完整源区域 Owner 见最终记录的 `ownership_audit_1.json`。

目前未观察到旧文字重影、重要内容消失或重复对象；仍有透明分割细边差异与模块切分粒度不足。尤其 **99 条普通文字不能直接改**，编辑性问题仍未解决。根据用户指令在这里停止，不继续自动优化或改 UI。

最终工程验证：pytest **395 passed / 2 warnings**，新核心与脚本 lint 通过，TypeScript + Vite build 通过（bundle 大小警告）。Drive 实际服务重启后 health=200、revise/downgrade=409、PPT 下载=200。原失败 Layout/PPT/预览/源图与备份 SHA-256 全部一致。
