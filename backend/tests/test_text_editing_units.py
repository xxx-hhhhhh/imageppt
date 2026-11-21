import copy

import numpy as np
from app.services.reconstruction import mixed_text


def candidate(text, x=0, size=20, color='#123456'):
    return ({'id': text, 'text': text, 'style': {'fontSize': size, 'color': color},
             'metadata': {'rawOCRBBox': [x, 10, x+len(text)*20, 30],
                          'fontMeasurement': {'fontAlphaError': .1, 'fontMaskSimilarity': .9,
                                              'fontDimensionError': 0}}},
            np.zeros((40, 200), bool), np.zeros((40, 200, 3), np.uint8))


def test_validated_whole_line_replaces_individual_characters(monkeypatch):
    monkeypatch.setattr(mixed_text, '_line_paper', lambda *args: None)
    monkeypatch.setattr(mixed_text, 'restore_text_candidate',
                        lambda source, item, *args, **kwargs: candidate(item['text']))
    item = {'id': 'ocr', 'text': '连续观测'}
    result = mixed_text.coalesce_text_runs(None, item, [candidate(t, i*20) for i, t in enumerate(item['text'])])
    assert len(result) == 1
    assert result[0][0]['text'] == item['text']
    assert result[0][0]['id'] == 'ocr'


def test_failed_whole_fit_can_still_join_punctuation_to_its_word(monkeypatch):
    monkeypatch.setattr(mixed_text, '_line_paper', lambda *args: None)
    monkeypatch.setattr(mixed_text, 'restore_text_candidate',
                        lambda source, item, *args, **kwargs: candidate(item['text']) if item['text']=='正文。' else None)
    parts = [candidate('标题：', size=32), candidate('正文', 60), candidate('。', 100)]
    result = mixed_text.coalesce_text_runs(None, {'id': 'ocr', 'text': '标题：正文。'}, parts)
    assert [p[0]['text'] for p in result] == ['标题：', '正文。']
    assert len({p[0]['id'] for p in result}) == 2


def test_true_colour_and_size_contrasts_are_not_flattened(monkeypatch):
    monkeypatch.setattr(mixed_text, '_line_paper', lambda *args: None)
    monkeypatch.setattr(mixed_text, 'restore_text_candidate',
                        lambda source, item, *args, **kwargs: candidate(item['text']))
    for tail in [candidate('正文', 40, size=32), candidate('正文', 40, color='#EE2222')]:
        parts = [candidate('标题'), tail]
        result = mixed_text.coalesce_text_runs(None, {'id': 'ocr', 'text': '标题正文'}, copy.deepcopy(parts))
        assert len(result) == 2
        assert [p[0]['style'] for p in result] == [p[0]['style'] for p in parts]


def packed_fixture():
    from app.services.reconstruction.line_text import pack_text_line
    parts = []
    for text, x, y, size in [('t', 10, 10, 24), ('1', 20, 18, 16)]:
        part = candidate(text, x, size)
        part[0].update(type='text', x=x, y=y, width=25, height=32)
        part[0]['style'].update(fontFamily='Arial', fontFile='arial.ttf', fontWeight=400)
        part[0]['metadata'].update(fontMatchingVersion=2, exclusiveFontVerified=True,
                                   reconstructionStrategy='editable_text')
        parts.append(part)
    item = {'id':'line', 'text':'t1', 'x':10, 'y':10, 'width':35, 'height':40}
    return pack_text_line(item, parts)[0][0], parts


def test_mixed_line_is_one_textbox_in_ppt_and_edits_do_not_export_old_runs(tmp_path):
    from app.services.pptx.renderer import PPTXRenderer
    from pptx import Presentation
    packed, _ = packed_fixture()
    layout = {'slide':{'width':200,'height':100}, 'elements':[packed]}
    path, report = PPTXRenderer().render_project('line', [layout], output_dir=tmp_path)
    deck = Presentation(path)
    assert len(deck.slides[0].shapes)==1
    shape = deck.slides[0].shapes[0]
    assert shape.text=='t1'
    assert len(shape.text_frame.paragraphs[0].runs)==2
    runs = shape.text_frame.paragraphs[0].runs
    assert abs(runs[0].font.size.pt/runs[1].font.size.pt-1.5)<.01
    assert all(run.font._element.get('baseline')=='0' for run in runs)
    assert report['valid']
    packed['text']='edited whole line'
    path, report = PPTXRenderer().render_project('line', [layout], output_dir=tmp_path)
    assert Presentation(path).slides[0].shapes[0].text=='edited whole line'
    assert report['valid']


def test_packed_preview_keeps_original_glyph_positions_and_scales_as_one():
    from app.services.reconstruction.line_text import measured_line_runs
    from app.services.typography.font_raster import paint_measured_text
    from PIL import Image
    packed, parts = packed_fixture()
    expected, actual = Image.new('RGBA',(200,100)), Image.new('RGBA',(200,100))
    for part, _, _ in parts:
        paint_measured_text(expected,part)
    paint_measured_text(actual,packed)
    assert np.array_equal(np.array(expected),np.array(actual))
    packed['x']+=30
    assert measured_line_runs(packed)[0]['x']==40
    packed['style']['color']='#FF0000'
    assert measured_line_runs(packed)==[]
