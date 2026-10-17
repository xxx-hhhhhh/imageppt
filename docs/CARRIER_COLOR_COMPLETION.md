# 底板内部缺口补色（2026-10-17）

## 结果与范围

接续 `969ac06`，保留已完成的模块交互、轮廓和 ownership 逻辑。使用 `outputs/e17b87059d2f4024886e05bf7501698c` 保存的 OCR/Qwen/SAM 证据离线重放，新副本为 `outputs/932a1de896c849d7a2332c506ead2a7d`。没有新增收费 API 请求，也没有覆盖原两页项目或旧测试结果。

对照旧副本的全部 15 个带 support mask 的底板，3 个资产的内部透明缺口得到补齐：

- 左下大底板 `visual_142`：134,147 → 0 像素。
- 照片下方条带 `visual_134`：23,017 → 0 像素。
- 左上子底板 `visual_138`：7 → 0 像素。

合计 157,171 个内部透明像素归零。其余 12 个底板原来已无内部透明孔洞。所有底板的原有可见 RGBA 像素、support 轮廓及轮廓外像素与旧副本逐像素一致。这里统计的是已识别底板的内部透明像素，不等于整页所有视觉问题均解决。

## 实现

在现有 `complete_local_floor` 中保留严格渐变拟合；失败后，使用同一底板的原始可见像素提供近似颜色。采样排除已生成的 substrate、文字 owner、透明区隐藏 RGB、饱和边框和高频纹理。平滑浅色区域使用 3×3 邻域；仅剩窄边缘时，允许至少 16 个具有局部一致性、总体颜色范围受限的真实浅色样本。

每个断开的 support 连通区域独立取样，避免向没有底色证据的孤立区域借用其他区域颜色。多尺度颜色扩散补齐大洞，只写入轮廓内、透明且已有可靠上层 owner 的像素。不会改变原有像素，也不会复制照片或文字图案。缺乏可靠浅色证据的其他输入仍会明确记录 unresolved，不保证任意深色或照片纹理背景均可恢复。

每个底板新增 `localCarrierRepair`：严格拟合像素数、近似补色像素数、采样数、剩余缺口数及补色 mask 路径，供后续审计。

## 验证

- Backend：473 passed，2 个既有第三方弃用警告。
- 新增 6 个通用用例：大洞、曲线外轮廓、稀疏浅色条带、断开区域无样本、样本所有权及未归属洞保护、单像素边缘、孤立浅色噪点等；补色测试文件合计 14 项通过。
- 定向 Ruff E9/F/I、`git diff --check` 通过。未改前端源代码。
- 离线重建保留 88 个原生文本框、70 个可移动图片；无原可编辑行退化，PPT 原生文本与 Layout 一致。
- 缺失资产、检测到的旧字重复、无主/重复源像素、非文字内部意外改动均为 0。
- Chromium 实际加载新项目，将其他对象在测试标签页内移到画布外，截取三个裸露底板；无媒体错误、JS 错误或写入请求。截图等待全部 DOM 对象和图片解码完成；首轮过早截图不作验收证据。
- PowerPoint COM 只读打开并渲染完整 PPT，以及三个独立底板的诊断 PPT。仅关闭本次打开的诊断演示文稿。

验证入口：

```powershell
python scripts/verify_carrier_completion.py outputs/e17b87059d2f4024886e05bf7501698c outputs/932a1de896c849d7a2332c506ead2a7d --output temp/carrier-verified --browser
.\scripts\verify-powerpoint-render.ps1 -ProjectPath outputs/932a1de896c849d7a2332c506ead2a7d
```

证据：`temp/carrier-verified/result.json`、`*-comparison.png`、`*-browser-exposed.png`、`browser-full.png`，以及各独立底板目录的 `powerpoint_native.png`。完整 PPT 位于新测试副本的 `editable.pptx`。

## 剩余边界

补齐后的渐变是与周围协调的近似，不是被遮挡原图的精确恢复。原有浅色纹理、局部残边及尚未独立提取的内容仍在原资产中；本次没有重绘它们。整组移开后原位置残边、10 行 OCR 未完整原生化、手柄遮挡和原生 PPT 分组等上一轮未完成项仍保留。

此前 `969ac06` 已按用户要求推送 GitHub；本轮补色修改保留在本地，尚未再次推送。
