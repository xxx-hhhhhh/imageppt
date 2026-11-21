"""Native runs in a single PowerPoint text frame for a measured OCR line."""
from pptx.util import Pt

from app.services.typography.font_raster import font_face


def write_measured_runs(paragraph, runs, sx):
    from app.services.pptx.renderer import _rgb, _set_font_family, font_px_to_pt
    metrics = []
    for item in runs:
        style = item['style']
        face = font_face(style.get('fontFile','msyh.ttc'), round(style['fontSize']), int(style.get('fontWeight',400)))
        metrics.append((face, face.getmetrics()[0]))
    for index, (item, (face, _)) in enumerate(zip(runs, metrics)):
        style = item['style']
        run = paragraph.add_run()
        run.text = item['text']
        font = run.font
        _set_font_family(font, style.get('fontFamily','Microsoft YaHei'))
        # Use the paragraph's shared baseline. PowerPoint's script baseline
        # formatting silently shrinks runs, even with an explicit font size.
        font.size = Pt(font_px_to_pt(style['fontSize'], sx))
        font.bold = int(style.get('fontWeight',400)) >= 600
        font.italic = style.get('fontStyle') == 'italic'
        font.color.rgb = _rgb(style.get('color'))
        spacing = style.get('letterSpacing',0)
        if index+1 < len(runs) and item['text']:
            spacing = (runs[index+1]['x']-item['x']-face.getlength(item['text']))/len(item['text'])
        font._element.set('spc',str(round(spacing*sx*72*100)))
        font._element.set('baseline','0')
