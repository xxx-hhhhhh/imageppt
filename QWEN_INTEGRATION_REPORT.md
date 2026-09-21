# Qwen3-VL-Flash 集成报告

## 结果

- Qwen integration code：PASS
- Mock / offline tests：PASS，13 passed
- Real API：SKIPPED，当前环境未配置 `QWEN_API_KEY` 与 `QWEN_BASE_URL`
- Backend：PASS
- Frontend typecheck：PASS
- Frontend build：PASS
- PPT export：PASS，已用 OCR + OpenCV fallback 实际生成 1 页 PPTX
- 本次党建回归图视觉评分：0.9500，PPTX validation valid，OCR coverage 1.0，越界 0，重叠告警 0
- 回归图对象统计：28 个文本、7 个图片、10 个普通 shape、6 个椭圆、6 个组件组
- 文本字体分布：SimSun、SimHei、Microsoft YaHei、KaiTi；28 个文本对象均保留原始单行 `lines` 信息

## 集成方式

Qwen3-VL-Flash 通过 OpenAI-compatible API 接入。图片在本地读取并编码为按 MIME 自动判断的 PNG/JPEG/WEBP data URL。Qwen 只负责页面语义、角色、组件、关系、重复组件和图层判断；OCR/OpenCV 继续负责文字内容、精确 bbox、颜色、形状和像素坐标。

Fusion Engine 不采用 Qwen bbox，而是把语义字段合并回 OCR/CV 元素，并写入 `visionConfidence`、`ocrConfidence`、`cvConfidence`、`finalConfidence` 和 `reconstructionStrategy`。

高精度模式支持一次 Visual Critic，最多模式支持三轮。每轮调整都会经过安全校验：位置变化不超过页面尺寸 10%，字号变化不超过 30%，尺寸变化不超过 20%。

## 配置

```env
VISION_PROVIDER=qwen
QWEN_API_KEY=用户自己的Key
QWEN_BASE_URL=https://{WorkspaceId}.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen3-vl-flash
QWEN_ENABLE_THINKING=false
QWEN_TIMEOUT=120
QWEN_MAX_RETRIES=2
```

真实 Key 不写入源码、`.env.example`、日志或报告。`.env` 已在 `.gitignore` 中。

## 已验证接口

- `GET /api/vision/status`：返回 provider、model、configured，不暴露 Key。
- `POST /api/vision/test`：使用本地 8×8 PNG 测试图片；未配置时返回明确的 fallback 状态。

## 离线回退

无 Key、401/403、超时、网络错误、非法 JSON 或服务端不接受 thinking 参数时，主流程不会崩溃。SceneAnalyzer 使用 OCR + OpenCV Scene Graph，PPTX 仍然可以导出。
