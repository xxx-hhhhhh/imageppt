# 测试样例说明

当前仓库的自动测试在 `backend/tests/test_core.py` 中生成最小可复现图片，并覆盖：

- Case 1：文字识别与文本框导出
- Case 2：文字 + 矩形/椭圆 Shape
- Case 3：独立图片对象与原生文本框共存
- Case 4：背景清理、Layout JSON 与 validation

运行时可将任意 PNG/JPG/JPEG 放入项目的 `uploads/`，通过 Web 页面上传，不需要把样例图片或 OCR 模型放进 Google Drive 项目目录。

