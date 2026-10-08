# 字体与混合字号修复（2026-10-08）

## 根因

1. PPT Renderer 给 `Font._element` 调用了不存在的 `get_or_add_rPr()`，随后被宽泛异常处理吞掉。中文字体 `a:ea`、复杂文字字体 `a:cs` 没有写入，PowerPoint 可能使用主题字体替代。只有 `font.name` 的西文字体设置不够。
2. 字体候选缺少宋体粗体等组合。旧二值轮廓匹配会偏向较粗字形；整块缩放比较又掩盖了字距差异。
3. 一个 OCR 行只能得到一个字号。真实页面的 `8门，覆盖学生3560人次` 已被 PaddleOCR 完整识别，但整行字形匹配失败，全部退回图片。
4. 拆出细小标点后，字体 advance bbox 比真实墨迹宽，旧尺寸筛选仍会拒绝标点。

## 实现

- `pptx/renderer.py` 直接在真实 `a:rPr` 上设置 latin/ea/cs，不再吞掉字体写入错误；字距从像素转换到百分之一磅。
- `typography/font_raster.py` 是字体匹配、Preview、Preservation QA 共用的测量器。支持安装字体、粗体模拟、抗锯齿、真实字形偏移和字距。
- `editable_text.py` 增加宋体粗体、黑体、仿宋、Times New Roman 等候选，并比较抗锯齿灰度、密度、真实墨迹尺寸；含中文的文本排除无中文字形的西文字体文件。标点不再被 advance 宽度误拒。
- `mixed_text.py` 用原图墨迹投影和动态规划对应 **OCR 原字符顺序**，合并断开的笔画，识别字号突变并分段。一次只认不出的片段不会阻断后文。保留 `sourceOcrIds`、`sourceCharRange` 和原文，不让 VLM 改写正文。
- Owner Builder 仍然先持久化成功文本 Owner/Mask，才局部修补旧字；失败片段仍由 movable image 保存。没有整块擦白，也没有自动 revision。
- Editor 原有文字层增加字距读取，不改版 UI。QA 区分原生文本框数量与完整可编辑 OCR 行数，避免分段后虚增覆盖率。

## 真实回归

未对真实图坐标或正文硬编码算法。通用单元测试另用不同正文 `9项，服务教师204人`。

真实原图取自此前失败页面的只读本地备份，未修改用户上传文件。一次新浏览器完整上传/maximum 解析使用实际 PaddleOCR、用户已配置的 Qwen3-VL-Flash API、SAM2，记录：

`outputs/af9632add0024a4ca62e29cb5e6cafdd`

- 99 条 OCR；66 个原生文本框，表示 61 条完整 OCR 行；23 个 movable image。本页 native Shape 0。
- 底部 OCR 正文全部恢复为 6 个可编辑字号分段：`8`、`门`、`，`、`覆盖学生`、`3560`、`人次`；字号分别 56/28/29/27/49/28 像素。
- `typography_acceptance.json` 验证每个 native text 的正文、latin/ea/cs 字体与 Scene 一致。完整 OCR 行覆盖率 61/99，字符覆盖率约 63.06%，不是全页全部可编辑。
- 真实浏览器：双击改字、字号/颜色、拖动、缩放、保存和下载 PPT 通过。补充独立图片拖动/缩放通过；DOM/Fabric 双绘字像素 0、资产 404 和 JS 错误 0。
- 图片拖动脚本最初误点到文字下方的纸面 carrier 中心，修正为 DOM 命中检测可见图片点后通过；没有为测试修改产品选择行为。
- PowerPoint 只读打开 `unmodified_reconstruction.pptx` 并实际渲染 `powerpoint_native.png`，另渲染去 native text 的诊断 PPT。未退出用户 PowerPoint 或修改用户正在打开的文稿。
- `comparison_native.png` 是原图与真实 PowerPoint 渲染的左右对照，不是用后端预览冒充 PPT。
- Ownership：源像素未归属/重复、修补纸面未归属/重复、禁止范围非文字变化、缺失资产、残留原字与新字重复的检查均为 0。此结果不代表字体或语义分割完全等同原图。
- Backend：408 passed，2 项依赖弃用警告；Ruff 致命错误检查与 diff check 通过。
- Frontend：TypeScript + Vite build 通过；604KB bundle 警告仍在。
- 自动优化调用/轮数均为 0。

## 已知限制与保存

字体匹配是已安装字体中的近似选择，截图不能证明原始字体身份。实际 PPT 对比仍有部分字重、间距和基线差异；不能宣称字体 100% 相同。艺术标题和 38 条尚不能可靠重建的 OCR 行保留图片。复杂抠图未在这次重做；未做四类页面和 WPS 本轮验收。

本轮 Google Drive 挂载暂不可访问，先在原本地 Git 仓库完成代码、测试和结果；尚未同步 Drive。没有在另一个目录新建源码项目。密钥、模型、依赖均保持在本地缓存，不复制到 Drive 或 Git。
