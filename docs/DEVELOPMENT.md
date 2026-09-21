# 开发说明

后端启动：

```powershell
$py = "$env:LOCALAPPDATA\Image2EditablePPT\venv\Scripts\python.exe"
& $py -m uvicorn app.main:app --app-dir backend --reload --host 127.0.0.1 --port 8000
```

前端启动：

```powershell
cd frontend
pnpm dev --host 127.0.0.1
```

测试：`.\scripts\test.ps1`。运行时依赖不要安装到项目目录，模型缓存也应使用 Paddle/ONNX 默认的用户缓存位置。

