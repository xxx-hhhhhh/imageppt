# Layout JSON

示例：

```json
{
  "version": "1.0",
  "slide": { "width": 1920, "height": 1080 },
  "source": "slide.png",
  "backgroundUrl": "/media/backgrounds/project/page_1.png",
  "elements": [
    {
      "id": "text_001",
      "type": "text",
      "x": 120,
      "y": 80,
      "width": 600,
      "height": 80,
      "rotation": 0,
      "zIndex": 20,
      "text": "项目介绍",
      "style": { "fontFamily": "Microsoft YaHei", "fontSize": 42, "fontWeight": 700, "color": "#17365D", "align": "left" },
      "confidence": 0.98
    }
  ]
}
```

坐标统一是原始图片像素。`background` 是清理过文字的独立背景对象，不等同于把原始截图贴到 PPT 后再叠透明文字。

