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

## Object First AST 字段

`sceneVersion=2.1-owner-evidence` 表示已通过像素归属门禁。`owner` 有 editable_text/native_shape/movable_image/background/intentional_ignore 五种，但自动流程不授权 intentional_ignore 删除。元素包含已有 role/groupId/zIndex/confidence，以及 source、parent、assetPath、editable、reconstructionStrategy；分割元素可带 contour，真实轮廓 alpha 保存在 PNG 中。

`bbox` 是 x/y/width/height 的派生视图，保存时自动刷新，不能另行覆盖主坐标。编辑器、preview、PPT 导出都读取当前 `slides/page_N.json`；早期 VLM scene、scene_final 与 revision scene 均为调试/审计快照。

`metadata.ownerGate` 记录未归属像素、真实资产检查及回收信息；fragment 聚合必须验证原 alpha/RGB 的完整 union 后才停用旧节点，原资产文件仍保留。页面真正底板由 pageOwner 和 border-anchored field 表示，浅色模块底板不能以背景名义忽略。

