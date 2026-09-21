# V3 AI Reconstruction Report

## 当前工作区继承情况

继续使用原项目 `google-drive-image2editableppt-web-ai-ocr`，未新建项目、未重置 Git、未覆盖已有正确修改。项目没有 Git 元数据，因此本轮通过当前文件状态核对；`EditorPage.tsx` 已确认仍保留上传、Canvas、页面列表、属性面板、OCR 和 PPTX 导出功能。

## AI 设置

- 网页提供“AI 设置”入口、启用开关、Provider、API Key、Base URL、模型、标准/高精度模式和连接测试。
- 设置保存到 Windows `%LOCALAPPDATA%/Image2EditablePPT/settings.json`，采用 `{ "vision": { ... } }` 结构，同时兼容旧的扁平格式。
- API 返回 `apiKeyMasked`，不返回完整密钥；空 API Key 更新会保留原本地密钥。
- `.env`、项目 JSON、README、日志和输出报告不写入真实 API Key。
- 本轮升级为多 Provider：自动免费、Gemini Flash、Gemini Flash-Lite、Qwen、OpenRouter Free、Custom OpenAI-compatible 和本地模式；各 Provider 独立 Key、启停、脱敏显示和清除操作。
- `VisionRouter` 按配置和 `freePreferred` 路由，并把 `requestedProvider`、`usedProvider`、`usedModel`、`fallbackCount` 写入 routing 元数据。

## Qwen 接入

`backend/app/services/vision/` 提供统一接口、Gemini 官方 `google-genai` provider、Qwen/OpenRouter/Custom OpenAI-compatible provider、工厂、VisionRouter、提示词、JSON 修复和 schema。`POST /api/settings/vision/test` 与 `test-all` 在有配置时实际调用 API；当前环境没有可用的真实 Provider 配置，因此实际示例按安全 fallback 执行，未伪造连接成功。

## Scene Analyzer / Fusion / Router

Scene Analyzer 已在 ReconstructionPipeline 中调用，并将 OCR、CV、分割和视觉语义融合为 `scene_raw.json` / `scene_refined.json`。新增 Reconstruction Router，根据文本、形状置信度、复杂度和背景类型选择 `editable_text`、`native_shape`、`transparent_image`、`local_image`、`background_image` 或 `group`。

## Typography

文本保留 OCR 行结构；字体类别包括 serif、sans、bold-sans、calligraphy、display 等，并通过系统字体 registry 做可用性匹配。字号按角色和 bbox 估计，文本框默认零边距，正文默认顶部对齐，支持有限范围 Text Fit。当前回归结果字体类别为宋体、黑体、微软雅黑和楷体的分层组合，而非所有文字强制同一字体。

## Background / Shape

底色分析使用内部区域、边缘排除、主色/中位色/方差，并区分 solid、gradient、texture 和 complex。椭圆检测增加 circularity、面积填充率和内部一致性约束；回归页没有出现原图不存在的大红圆。

## Visual QA

高精度模式支持重建预览、视觉评分和 Qwen critic 闭环；当前无 Key 时保持 OCR+CV fallback。回归页实际评分为 `0.9500`，并生成 `visual_score.json`、`visual_validation.json`。前端新增“原图 / 重建 / 对比”模式和原图-重建滑块。

## UI / 实际验证

- 网站：`http://127.0.0.1:5173/`
- 后端：`http://127.0.0.1:8000/`
- `GET /api/settings/vision`、`PUT /api/settings/vision`、`POST /api/settings/vision/test`、`GET /api/vision/status` 已接通。
- 实际生成了回归 PPTX；1 张党建回归图和此前 5 张通用 fixture 图已完成转换，共 6 张实际转换图片。当前回归页布局 JSON 包含 28 个文本、6 个图片、10 个 Shape、6 个分组；最终 PPTX 结构校验包含 28 个文本、7 个图片、10 个 Shape。

## 测试

- Backend：24 passed
- Frontend TypeScript：通过
- Frontend Vite build：通过；仅有 bundle 大小提示
- 实际 PPTX：通过，包含文本对象、图片对象、Shape、分组和坐标验证

## 已知问题

1. 当前机器未配置真实 Qwen API Key，因此 Qwen 实际连接与 critic 调用标记为 SKIPPED，系统使用 OCR+CV fallback。
2. PPStructure、PaddleOCR 等可选组件在当前环境不可用时会产生 warning，但不阻断转换；RapidOCR fallback 已验证。
3. 本轮 CUA 原生 UI 自动化服务不可用，因此 UI 采用前端 build、HTTP API 和产物检查完成验证；代码路径已保留真实上传、分析、对比和导出操作。
