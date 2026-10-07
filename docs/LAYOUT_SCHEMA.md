# Layout JSON

Object First 使用 Layout `1.2` / sceneVersion `3.0`，`elements` 是唯一可修改的图结构；scene_refined 是它的派生快照，不是另一份权威数据。

每个对象增加 owner、role、bbox、parent/groupId、mask、contour、source、confidence、reconstructionStrategy、assetPath、editable。x/y/width/height 仍为原图像素，保存时重新计算 bbox。mask 是源图空间的证据文件，sourceBBox 保存初始提取区域；不是用户移动后重新生成的分割。

owner 为 editable_text / native_shape / movable_image / background / intentional_ignore。最后一种仅在用户明确忽略时允许消除像素，VLM ignore 建议不构成许可。图片 asset 必须存在且能解码，native shape 必须通过几何验证，文字必须是 OCR 正文且有可靠墨迹 mask，才可申请删除对应旧像素。

previewUrl 指向由同一份图合成的缩略图；背景不再承担页面视觉内容。revision 保留 scene/assets/background/preview/qa/decision，候选不改变原图、已接受资产或背景。

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

