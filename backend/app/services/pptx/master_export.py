"""Export the saved editor scene through the local PPT Master compiler."""
from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree as ET

from PIL import Image
from pptx import Presentation
from pptx.oxml.ns import qn

from app.services.reconstruction.line_text import measured_line_runs
from app.services.validation.service import save_validation, validate_pptx

LOG = logging.getLogger(__name__)
MASTER_FILENAME = 'editable-ppt-master.pptx'


class MasterExportError(RuntimeError):
    pass


def _skill_dir():
    configured = os.getenv('PPT_MASTER_SKILL_DIR')
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(os.getenv('CODEX_HOME') or Path.home()/'.codex')/'skills'/'ppt-master'


def _num(value):
    return f'{float(value):.4f}'.rstrip('0').rstrip('.')


def _color(value, default):
    return value.upper() if isinstance(value, str) and re.fullmatch(r'#[0-9a-fA-F]{6}', value) else default


def _font(style):
    return {'font-family': style.get('fontFamily') or 'Microsoft YaHei',
            'font-size': _num(style.get('fontSize') or 24),
            'font-weight': 'bold' if int(style.get('fontWeight') or 400)>=600 else 'normal',
            'font-style': 'italic' if style.get('fontStyle')=='italic' else 'normal',
            'fill': _color(style.get('color'), '#111827'),
            **({'letter-spacing': _num(style['letterSpacing'])} if style.get('letterSpacing') else {})}


def _asset(project_dir, src):
    match = re.fullmatch(r'/media/(assets|backgrounds)/([a-fA-F0-9]+)/([^/\\?#]+)(?:\?[^#]*)?', src or '')
    if not match:
        raise MasterExportError('图片资源无法用于 PPT Master 导出，请检查页面或使用标准导出。')
    category, owner, name = match.groups()
    if owner != project_dir.name or name in ('.', '..'):
        raise MasterExportError('图片资源不属于当前项目，请检查页面或使用标准导出。')
    root = project_dir.resolve()
    path = (root/category/name).resolve()
    if not path.is_relative_to(root/category) or not path.is_file():
        raise MasterExportError('项目图片缺失，请重新加载项目后重试。')
    return path


def _save_image(source, target, crop):
    with Image.open(source) as image:
        image = image.convert('RGBA')
        if crop:
            left, top, right, bottom = (float(crop.get(k, 0)) for k in ('left','top','right','bottom'))
            w, h = image.size
            box = (round(w*left), round(h*top), round(w*(1-right)), round(h*(1-bottom)))
            if not (0<=box[0]<box[2]<=w and 0<=box[1]<box[3]<=h):
                raise MasterExportError('图片裁剪范围无效，请调整后重试。')
            image = image.crop(box)
        image.save(target)


def _write_page(layout, project_dir, work, index):
    size = layout['slide']
    root = ET.Element('svg', {'xmlns':'http://www.w3.org/2000/svg', 'xmlns:xlink':'http://www.w3.org/1999/xlink',
        'viewBox':f'0 0 {_num(size["width"])} {_num(size["height"])}', 'lang':'zh-CN',
        'font-family':'Microsoft YaHei','data-pptx-page-role':'content'})
    for n, element in enumerate(sorted(layout.get('elements', []), key=lambda e:e.get('zIndex',0))):
        meta = element.get('metadata') or {}
        kind = element['type']
        if kind=='group' or any(meta.get(k) for k in ('suppressed','suppressRender','ownedBy')):
            continue
        x,y,w,h = (float(element[k]) for k in ('x','y','width','height'))
        if w<=0 or h<=0:
            if kind not in ('line','arrow'):
                raise MasterExportError('有对象的尺寸无效，请调整后重试。')
        style = element.get('style') or {}
        attrs = {'id':f'p{index}-object-{n}', 'opacity':_num(style.get('opacity',1))}
        box = dict(zip(('x','y','width','height'),map(_num,(x,y,w,h))))
        rotation = float(element.get('rotation') or 0)
        if rotation:
            attrs['transform'] = f'rotate({_num(rotation)} {_num(x+w/2)} {_num(y+h/2)})'
        if kind in ('background','image'):
            target = work/'images'/f'{index}-{n}.png'
            _save_image(_asset(project_dir, element.get('src')), target, element.get('crop') or {})
            ET.SubElement(root, 'image', {**attrs, **box, 'xlink:href':f'../images/{target.name}', 'preserveAspectRatio':'none'})
        elif kind=='text':
            body = element.get('text') or ''
            if not body.strip():
                continue
            align = style.get('align','left')
            anchor = {'left':'start','center':'middle','right':'end'}.get(align,'start')
            left = x+{'start':0,'middle':w/2,'end':w}[anchor]
            runs = measured_line_runs(element)
            font_size = float(style.get('fontSize') or 24)
            baseline = max((r['y']+r['style']['fontSize'] for r in runs),default=y+font_size)
            node = ET.SubElement(root,'text',{**attrs,**_font(style),'x':_num(left),'y':_num(baseline),
                'text-anchor':anchor,'xml:space':'preserve'})
            if runs:
                for run in runs:
                    ET.SubElement(node,'tspan',_font(run['style'])).text = run['text']
            else:
                first,*rest = body.split('\n')
                node.text = first
                for line in rest:
                    ET.SubElement(node,'tspan',{'x':_num(left),'dy':_num(font_size*float(style.get('lineSpacing') or 1.12))}).text = line
        else:
            paint = {'fill':_color(style.get('fill'),'#DCE6F1'), 'stroke':_color(style.get('stroke'),'#17365D'),
                     'stroke-width':_num(style.get('strokeWidth',1))}
            if kind=='ellipse':
                ET.SubElement(root,'ellipse',{**attrs,**paint,'cx':_num(x+w/2),'cy':_num(y+h/2),'rx':_num(w/2),'ry':_num(h/2)})
            elif kind in ('rectangle','roundedRectangle'):
                if kind=='roundedRectangle': box['rx'] = _num(min(18,w/2,h/2))
                ET.SubElement(root,'rect',{**attrs,**paint,**box})
            elif kind in ('line','arrow'):
                ET.SubElement(root,'line',{**attrs,'stroke':paint['stroke'],'stroke-width':paint['stroke-width'],
                    'x1':_num(x),'y1':_num(y),'x2':_num(x+w),'y2':_num(y+h)})
            else:
                raise MasterExportError('当前页面包含 PPT Master 不支持的对象，请使用标准导出。')
    ET.ElementTree(root).write(work/'svg_output'/f'{index:03d}_page.svg',encoding='utf-8')


def _restore_literal_bullets(pptx, layouts):
    """Preserve OCR's actual bullet character instead of a generated list marker."""
    deck = Presentation(pptx)
    changed = False
    for slide, layout in zip(deck.slides, layouts):
        source = [e['text'] for e in sorted(layout['elements'],key=lambda e:e.get('zIndex',0))
                  if e['type']=='text' and (e.get('text') or '').strip()
                  and not any((e.get('metadata') or {}).get(k) for k in ('suppressed','suppressRender','ownedBy'))]
        shapes = [s for s in slide.shapes if s.has_text_frame and s.text.strip()]
        if len(source) != len(shapes):
            continue  # The ordinary export validator catches mismatched text frames.
        for original, shape in zip(source, shapes):
            if original == shape.text:
                continue
            lines = original.split('\n')
            paragraphs = shape.text_frame.paragraphs
            if len(lines) != len(paragraphs):
                continue
            for line, paragraph in zip(lines, paragraphs):
                match = re.fullmatch(r'(\s*[·•●▪■◆◇◦‣][ \t]*)(.+)', line)
                properties = paragraph._p.find(qn('a:pPr'))
                if not match or match[2] != paragraph.text or properties is None:
                    continue
                bullet = properties.find(qn('a:buChar'))
                if bullet is None or not paragraph.runs:
                    continue
                # Keep it in the same editable frame and inherit adjacent text style.
                paragraph.runs[0].text = match[1] + paragraph.runs[0].text
                for child in list(properties):
                    if child.tag.split('}')[-1].startswith('bu'):
                        properties.remove(child)
                properties.attrib.pop('marL',None)
                properties.attrib.pop('indent',None)
                changed = True
    if changed:
        deck.save(pptx)


def require_master():
    scripts = _skill_dir()/'scripts'
    checker = scripts/'svg_quality_checker.py'
    exporter = scripts/'svg_to_pptx.py'
    if not checker.is_file() or not exporter.is_file():
        raise MasterExportError('服务器未找到 PPT Master，请安装后重试，或使用标准导出。')
    return checker, exporter


def _run_master_script(script: Path, work: Path, args: list[str]) -> None:
    result = subprocess.run(
        [sys.executable, '-X', 'utf8', str(script), str(work), *args],
        cwd=work, capture_output=True, text=True, encoding='utf-8', errors='replace',
        timeout=90, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
    )
    if result.returncode:
        LOG.warning('PPT Master %s failed (exit %d)', script.name, result.returncode)
        raise MasterExportError('PPT Master 无法导出当前页面，请使用标准导出。')


def export_master(project_dir: Path, layouts, *, check_first_five: bool = False):
    checker, exporter = require_master()
    try:
        with TemporaryDirectory(prefix='i2e-ppt-master-',dir=project_dir) as temp:
            work = Path(temp).resolve()
            (work/'svg_output').mkdir()
            (work/'images').mkdir()
            check_args = ['--quick-generate', '--canonical-authoring', '--json', '--stage']
            for i, layout in enumerate(layouts,1):
                _write_page(layout,project_dir,work,i)
                if check_first_five and len(layouts) >= 7 and i == 5:
                    _run_master_script(checker, work, [*check_args, 'early'])
            candidate = work/MASTER_FILENAME
            _run_master_script(checker, work, [*check_args, 'final'])
            _run_master_script(exporter, work, ['--quick-generate', '--no-notes', '--transition', 'none', '-o', str(candidate)])
            _restore_literal_bullets(candidate,layouts)
            report = validate_pptx(candidate,layouts)
            if not report['valid']:
                LOG.warning('PPT Master export validation failed: %s',report['warnings'])
                raise MasterExportError('PPT Master 导出的文字或页数校验未通过，请使用标准导出。')
            save_validation(report,work/'validation.json')
            destination = project_dir/MASTER_FILENAME
            candidate.replace(destination)
            (work/'validation.json').replace(project_dir/'validation-ppt-master.json')
            return destination,report
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        LOG.warning('PPT Master export could not finish: %s',exc)
        raise MasterExportError('PPT Master 导出未完成，请重试或使用标准导出。') from exc
