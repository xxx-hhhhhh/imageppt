# Image2EditablePPT

Image2EditablePPT 是一个 Windows 优先的图片转对象级可编辑 PowerPoint Web 项目。它将图片先转成统一的 Layout JSON，再在浏览器中用 Fabric.js 编辑，最后用 `python-pptx` 将文字、简单图形和图片分别写成 PPTX 对象。

## 1. 项目介绍

目标流程：

```text
图片 → 预处理 → OCR → 版面分析 → OCR mask → 背景修复 → Layout JSON
     → Qwen3-VL-Flash 页面理解 → Fusion Scene Graph → React/Fabric 编辑器
     → 保存 Layout JSON → PPTX Renderer → Visual QA/Critic → editable.pptx
```

项目不会把整张原图当成唯一 PPT 背景来伪装成可编辑结果。导出时会写入清理后的背景图、原生文本框、可编辑 PowerPoint Shape，以及复杂图片/插图的独立图片对象。

## 2. 功能

- 上传 JPG、JPEG、PNG，支持一次上传多页
- PaddleOCR 优先，RapidOCR fallback；OCR Provider 可扩展
- Qwen3-VL-Flash 可选页面理解：Qwen 只判断角色、组件、关系和图层，OCR/OpenCV 保留文字、颜色和精确坐标
- OCR bbox、字号、文字颜色、置信度写入 Layout JSON
- OpenCV mask + inpaint，LAMA 可选且失败自动 fallback
- 规则版面分析：文本、矩形、圆角矩形、椭圆、线条、图片区域
- Fabric.js 画布：选择、拖动、缩放、双击改文字、多选、复制、删除、层级调整
- 文字：字体、字号、粗体、斜体、颜色、对齐
- 图形：rectangle、roundedRectangle、ellipse、line、arrow
- 左侧缩略图、页面切换、删除和上下排序
- Layout JSON 保存
- 对象级 PPTX 导出
- Structural / OCR coverage / Layout validation，输出 `validation.json`

## 3. 系统架构

- `frontend/`：React + TypeScript + Vite + Fabric.js + Zustand + Axios
- `backend/`：FastAPI API、OCR、OpenCV、Layout、PPTX Renderer、Validation
- `shared/`：跨前后端的 Layout JSON schema 与样例
- `uploads/`：本地运行时上传的原图，已加入 gitignore
- `outputs/`：Layout、背景修复图、独立图片资产、PPTX、validation.json
- `temp/`：项目元数据和中间状态，已加入 gitignore

## 4. 技术栈

React、TypeScript、Vite、Fabric.js、Zustand、Axios、FastAPI、Pillow、OpenCV、PaddleOCR（优先）、RapidOCR（fallback）、python-pptx、pytest。

## 5. 文件目录

```text
Image2EditablePPT/
├── frontend/src/{components,editor,pages,services,stores,types,utils}
├── backend/app/{api,models,schemas,services,utils}
├── shared/layout-schema.json
├── uploads/ outputs/ temp/
├── scripts/{setup.ps1,start.ps1,test.ps1}
├── docs/{ARCHITECTURE,API,LAYOUT_SCHEMA,DEVELOPMENT,REFERENCES}.md
└── DEVELOPMENT_REPORT.md
```

## 6. Windows 安装

在项目根目录执行：

```powershell
.\scripts\setup.ps1
```

脚本会把 Python 虚拟环境放在 `%LOCALAPPDATA%\Image2EditablePPT\venv`，不会把 venv、`node_modules` 或模型文件写入 Google Drive 项目目录。PaddleOCR 与 LAMA 为可选增强依赖，当前 Python/Windows 没有兼容 wheel 时会保留 RapidOCR/OpenCV 主流程。

## 7. 启动

```powershell
.\scripts\start.ps1
```

默认地址：`http://127.0.0.1:5173`，后端健康检查：`http://127.0.0.1:8000/api/health`。

## 8.1 Qwen3-VL-Flash 配置

复制 `.env.example` 为 `.env`，只设置用户自己的 Alibaba Cloud Model Studio Workspace 配置：

```env
VISION_PROVIDER=qwen
QWEN_API_KEY=用户自己的Key
QWEN_BASE_URL=https://{WorkspaceId}.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen3-vl-flash
QWEN_ENABLE_THINKING=false
QWEN_TIMEOUT=120
QWEN_MAX_RETRIES=2
```

不要把真实 Key 写入 README、源码、日志或 Google Drive 文件。没有 Key 时，系统自动回退到 OCR + OpenCV。连接状态可通过 `GET /api/vision/status` 查看，连接测试使用 `POST /api/vision/test`。

网页顶部的“AI设置 / API Key”可以选择自动免费、Gemini 2.5 Flash、Gemini 2.5 Flash-Lite、Qwen3-VL-Flash、OpenRouter Free、自定义 OpenAI-compatible 或本地模式，并分别填写、保存和测试各 Provider。API Key 只保存到 Windows `%LOCALAPPDATA%/Image2EditablePPT/settings.json`，不会写入项目目录；接口只返回掩码后的 Key。自动免费模式按已配置且允许自动使用的 Provider 路由，并记录每页实际使用模型和 fallback 次数。

## 9. 使用方法

1. 上传一张或多张 PNG/JPG/JPEG。
2. 点击“OCR解析”，等待页面缩略图和对象生成。
3. 在中间画布拖动、缩放对象，双击文字编辑内容。
4. 在右侧属性面板修改文字、颜色、位置、尺寸或图形样式。
5. 点击“保存当前页面”写回 Layout JSON。
6. 点击“导出PPT”，得到对象级 `editable.pptx`。

## 10. API

详见 [`docs/API.md`](docs/API.md)。核心接口：

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
GET  /api/vision/status
POST /api/vision/test
```

## 11. Layout JSON

所有坐标使用原始图片像素。`text` 会导出为原生文本框；简单几何导出为原生 Shape；复杂视觉元素导出为独立图片；`zIndex` 决定写入顺序。完整字段见 [`docs/LAYOUT_SCHEMA.md`](docs/LAYOUT_SCHEMA.md) 与 [`shared/layout-schema.json`](shared/layout-schema.json)。

## 12. PPT 导出机制

PPT 页面尺寸按首个页面的像素比例建立。像素坐标通过 `slide_width / pixel_width` 与 `slide_height / pixel_height` 转成英寸。背景、文字、形状、图片按 `zIndex` 写入。文字使用真实 `TextFrame`，形状使用 `MSO_SHAPE`/connector，图片使用 `add_picture`。导出后立即重新打开 PPTX 并生成 validation report。

## 13. 已知限制

- PaddleOCR 在部分 Windows/Python 版本没有匹配的 PaddlePaddle wheel，默认可用 RapidOCR fallback。
- 规则版面分析无法可靠还原复杂渐变、图表、装饰插画和特殊字体；这些区域倾向保留为独立图片。
- OpenCV inpaint 只适合文字周围纹理可推断的区域，复杂人像/结构化背景需要安装 LAMA 或后续接入 VLM。
- PPT 中的图片和修复背景保持独立对象，但复杂背景的完全像素级复原不作保证。

## 14. 后续计划

接入可选 VLMProvider、改进形状/图片分割、支持更精确字体测量和多行文本样式、增加页面级视觉对比报告、补充 PPTX 渲染截图回归测试。

## 14. 开源项目参考

四个参考项目及借鉴范围记录在 [`docs/REFERENCES.md`](docs/REFERENCES.md)。本项目优先重新实现，未复制其源码或运行时模型。
