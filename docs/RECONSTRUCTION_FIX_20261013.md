# 2026-10-13：文字完整性、对象拆分与真实导出质量修复

本次直接修改现有项目，基线 commit 为 `ecb3f2c`，工作分支为 `codex/text-integrity-quality`。没有改版 UI，没有启动自动优化，没有覆盖用户上传的 `editable (9).pptx` 或旧项目记录。测试产物使用新的诊断记录。

## 根因及修改位置

- `backend/app/services/reconstruction/mixed_text.py`：旧逻辑先按字形高度拆字，把英语大小写、升部降部误判成不同字号；失败后又提交部分字符。现在先尝试完整行，按参考字形校正高度，真正混排才分段；分段必须完整成功。保留 OCR 原文，不用 VLM 改正文。
- `editable_text.py`：用稳健的纸面样本初始化拟合；分离 OCR 框内但正文未包含的行首圆点，包含 JPEG 导致的断裂小圆点；补充等线字体候选；无效字体候选不能挤掉有效候选。文字擦除仍只在已经确认的字形局部进行。
- `exclusive_ownership.py`：整行预检后才注册 Owner；相邻文字晕边只有在局部纸面一致时才能共享修补范围，源像素仍只归一个 Owner。递归读取 Scene Graph 子模块，先分配小对象再分配承载底板；修正越过右边缘的归一化 bbox 被当成像素坐标的问题。
- 同文件：已清除可靠文字后的严格纯色矩形允许恢复 native shape；原文字的纸面承载有独立证据，不允许整块擦白。
- `segmentation/segmentation_provider.py`：CV 提案读取卡片内部的嵌套轮廓，降低小对象面积门槛；OCR 文本区域不再作为图像对象送入 SAM；SAM 结果去重。仍调用真正的本地 Ultralytics SAM2，未安装或失败时保留 CV fallback。
- `pptx/renderer.py`：取消最小 5pt 导致小字被放大的限制，按原坐标比例导出；将预览使用的宋体合成粗体描边写入原生文字格式，解决 PowerPoint 字重与预览不一致。文字仍是可修改的文本框。
- `typography/font_raster.py`：等线字体编辑与后端渲染保持一致。
- `validation/service.py`：实际重新打开 PPT，比较逐页文本内容与数量；删除“exportedText = recognizedText”的假校验。源 OCR 覆盖率与最终 Layout 导出完整性分开报告，用户手动改字不会被错判为导出丢字。
- `validation/reconstruction_quality.py`（新增）：统计完整可编辑行、字符覆盖率、部分行、最大图片 Owner 源像素占比及未恢复文本。语义对象覆盖率等未实测指标返回 null，不再填写 100% 或 0 缺失。
- `visual_qa/analyzer.py`：SSIM 改为局部均值/方差/协方差计算，不再把 MSE 的变换命名为 SSIM。
- `reconstruction/pipeline.py`：接入真实质量指标，保留问题列表；综合分不高于可编辑字符覆盖率，明确提示“部分可编辑”；导出内容校验失败时禁止发布候选。继续暂停自动 revision。

## 真实回归证据

### 科研遥感信息图（用户 editable (9).pptx 对应原图）

旧记录：`b669c14c592e4d669c37a42fc8fd16a7`。

修复后、相同真实 OCR/VLM 证据、本地重新运行 SAM2：`a6d40a93d71046c690f06e1ca77a0004`。

- 完整可编辑 OCR 行从 **25/52 → 37/52**；旧版的 37 个文本框不等于 37 条完整文字，其中有大量碎片。
- 可编辑字符覆盖率从 **47.12% → 62.36%**。
- 不完整的“部分字符可编辑、部分仍在图片”行数变为 **0**。
- 37 个原生文本框、50 个独立图片 Owner；最大图片 Owner 实际拥有的源像素从约 40.89% 降至 **10.75%**。这是源像素面积，不是图片 bbox 面积，也不是语义提取完成率。
- 缺失资产、原文字与新文字重复检查、源像素未归属/重复归属、禁止范围内非文字变化均为 0。检查不等于所有视觉细节无瑕疵。
- 已用真实 PowerPoint 打开并按原图尺寸导出 `powerpoint_native.png`，人工检查标题、表述、卫星、羽流图、浅色承载块、底部结论。

完整在线回归：`9f3812a293994557a3102626a39a9c33`。浏览器真实上传 → PaddleOCR → 本地 SAM2 → 用户配置的 Qwen3-VL-Plus → 编辑 → 保存 → 导出。aiUsed=true，未 mock Provider。37/52 条完整可编辑文字；VLM 本次模块提案不同，独立图片 Owner 为 44。不存在图片请求 404 或浏览器异常。

该在线回归会在自己的新测试记录中改字，原始导出备份为 `unmodified_reconstruction.pptx`，用户比较请优先使用上面未手动修改的诊断记录。

### 党建复杂图表页

旧记录：`af9632add0024a4ca62e29cb5e6cafdd`。

修复后真实证据重放并重新运行 SAM2：`f080ef85df1c4daba3edc1e7e9562cdd`。

- 完整可编辑 OCR 行 **61/99 → 64/99**，原生文本框 69 个；多字号行允许多个完整的文本段。
- 底部 `8门，覆盖学生3560人次` 的 **13/13** 字符均进入原生文本框，包含大小数字、标点和后文。
- 字符覆盖率 **63.42%**，部分恢复的行数 0；独立图片 Owner 72 个，最大 Owner 的源像素占比 7.54%。对象数量增加不自动代表语义拆分正确。
- 同样通过 Owner/资产/原字重复/非文字内容保护检查；实际 PowerPoint 渲染保存在该目录的 `powerpoint_native.png`。
- 大标题、红绸、建筑、表格和四个图表在渲染中保留；字号、字体和细节仍有近似。

## 测试与可复现命令

新增 `backend/tests/test_reconstruction_integrity.py`，扩充 `test_typography_recovery.py`。覆盖普通英文整行、混合字号及后文完整性、原生宋体描边、嵌套模块、文本排除分割、文本卡片 native shape、实际 PPT 文本丢失检测、小字号、真实 SSIM、未知质量指标。

```powershell
$env:PYTHONPATH='backend'
python -m pytest backend/tests -q
python scripts/verify_native_recovery.py outputs/b669c14c592e4d669c37a42fc8fd16a7 --refresh-segmentation
python scripts/verify_owner_browser.py outputs/b669c14c592e4d669c37a42fc8fd16a7/original.png --mode maximum
.\scripts\verify-powerpoint-render.ps1 -ProjectPath outputs/a6d40a93d71046c690f06e1ca77a0004
```

真实浏览器验收遵循项目 testing-playwright 规范：双击编辑、字号/颜色、拖动、缩放、保存及下载均实际执行。另有独立图片拖动/缩放检查，检查 DOM/Fabric 是否重复绘字。没有修改 UI 文件。TypeScript 和 Vite build 通过；仍有约 604KB bundle 警告。Ruff F 类检查通过。

最终 Backend pytest：**420 passed**（2 个依赖弃用警告），耗时约 107 秒。最新服务的真实导出接口再次调用成功，重新打开的文本校验 valid=true。图片拖动验收最初选择了页面等宽底层承载图，x=0 正确受画布边界限制，原断言因此失败；脚本改为选择有移动空间的独立对象，不移除位移验证。

## 尚未解决，不能宣称全部可编辑

1. 科研图仍有 **15/52** 条文字未完整转为原生文本，包含核心判断/底部整句及部分小字、艺术字；党建页仍有 35/99 条。保留图片确保内容不丢，而不是隐藏这些失败。
2. 安装字体匹配属于近似恢复，不保证识别原始字体。科研图 `transfer learning` 等中英混排的字形与原图仍有差别，字体/字重/基线不是完全一致。
3. 宋体测量描边以原生文字轮廓导出。PowerPoint 中仍能直接改字，但单独修改文字填充色时，可能需要同时修改“文本轮廓”颜色；网页改色会同步输出轮廓颜色。
4. 两张复杂真实页的 native Shape 仍为 **0**。已验证纯色文字卡片可以原生化；渐变、带纹理的底板和复杂箭头仍是图片，不以错误矢量化凑数量。
5. 复杂对象分割有所改善，但 SAM 的边缘仍可能不精确，部分图像仍带局部底色。没有声称每个图表/卫星都已达到人工作图级的分组与抠图。
6. 本轮为两张真实页 + 通用合成算法测试，不冒充四套独立真实页面。实测 PowerPoint，未实测 WPS。本轮自动优化次数为 0。
7. Google Drive 当前未挂载；修改及结果位于现有本地 checkout，尚未同步到 Drive。不上传密钥、模型、虚拟环境或 node_modules。

## 使用

现有网站仍为 `http://127.0.0.1:5173/`，后端已重启加载修改。刷新后重新解析，旧 PPT 不会自动更新。也可直接打开上述新诊断记录中的 `editable.pptx`。
