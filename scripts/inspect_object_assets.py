"""Inspect real movable PNGs on checkerboard; never changes project assets."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('project', type=Path)
    parser.add_argument('--compare', type=Path)
    args = parser.parse_args()
    root = args.project
    layout = json.loads((root/'slides/page_1.json').read_text(encoding='utf8'))
    images = [e for e in layout['elements'] if e.get('owner') == 'movable_image']
    print(json.dumps({'images': len(images), 'policies': dict(Counter(e.get('metadata', {}).get('visualAssetPolicy', e.get('source')) for e in images))}))
    cell_w, cell_h = 300, 200
    sheet = Image.new('RGB', (cell_w*4, cell_h*((len(images)+3)//4)), 'white')
    draw = ImageDraw.Draw(sheet)
    for index, e in enumerate(images):
        x, y = index % 4 * cell_w, index // 4 * cell_h
        for yy in range(0, 165, 12):
            for xx in range(0, cell_w, 12):
                draw.rectangle((x+xx, y+yy, x+xx+11, y+yy+11), fill='#8b9aa8' if (xx//12+yy//12)%2 else '#c5cdd5')
        asset = Image.open(root/'assets'/e['src'].split('/')[-1]).convert('RGBA')
        asset.thumbnail((cell_w-12, 155))
        sheet.paste(asset, (x+(cell_w-asset.width)//2, y+(165-asset.height)//2), asset)
        label = f"{index}: {e['id'].split('_visual_')[-1]} {e['width']}x{e['height']}"
        draw.text((x+3, y+167), label, fill='black')
        draw.text((x+3, y+181), str(e.get('metadata', {}).get('visualAssetPolicy', e.get('source')))[:42], fill='black')
    sheet.save(root/'asset_contact_sheet.png')
    (root/'asset_contact_index.json').write_text(json.dumps(images, ensure_ascii=False, indent=2), encoding='utf8')
    print(root/'asset_contact_sheet.png')
    if args.compare:
        compare_assets(args.compare,root,images)


def compare_assets(before_root,after_root,after):
    """Match by spatial overlap, not filenames or hardcoded example coordinates."""
    before=json.loads((before_root/'slides/page_1.json').read_text(encoding='utf8'))['elements']
    before=[e for e in before if e.get('owner')=='movable_image']
    def bbox(e):
        return (e['x'],e['y'],e['x']+e['width'],e['y']+e['height'])
    def similarity(a,b):
        ax,ay,bx,by=bbox(a)
        cx,cy,dx,dy=bbox(b)
        intersection=max(0,min(bx,dx)-max(ax,cx))*max(0,min(by,dy)-max(ay,cy))
        return intersection/max(1,(bx-ax)*(by-ay)+(dx-cx)*(dy-cy)-intersection)
    def load(root,e):
        return Image.open(root/'assets'/e['src'].split('/')[-1]).convert('RGBA')
    pairs=[]
    for e in after:
        if min(e['width'],e['height'])<35 or e['width']>320 or e['height']>200:
            continue
        old=max(before,key=lambda b:similarity(e,b))
        if similarity(e,old)<.6:
            continue
        old_im,new_im=load(before_root,old),load(after_root,e)
        old_alpha=np.asarray(old_im)[:,:,3]
        new_alpha=np.asarray(new_im)[:,:,3]
        change=float(np.mean(old_alpha==255)-np.mean(new_alpha==255))
        if change>.07:
            pairs.append((change,old,e,old_im,new_im))
    pairs.sort(key=lambda p:p[0],reverse=True)
    pairs=pairs[:6]
    comparison=Image.new('RGB',(880,60+220*len(pairs)),'#eef2f6')
    draw=ImageDraw.Draw(comparison)
    face=Path('C:/Windows/Fonts/msyh.ttc')
    font=ImageFont.truetype(str(face),18) if face.is_file() else ImageFont.load_default()
    draw.text((20,18),'修改前：实际导出的独立图片',fill='black',font=font)
    draw.text((460,18),'修改后：按轮廓提取的独立图片',fill='black',font=font)
    rows=[]
    for row,(_,old,new,old_im,new_im) in enumerate(pairs):
        top=60+row*220
        for col,im in enumerate((old_im,new_im)):
            left=col*440
            for y in range(top,top+190,16):
                for x in range(left,left+440,16):
                    draw.rectangle((x,y,x+15,y+15),fill='#637282' if ((x-left)//16+(y-top)//16)%2 else '#9cabb9')
            im=im.copy()
            im.thumbnail((400,175))
            comparison.paste(im,(left+(440-im.width)//2,top+(190-im.height)//2),im)
        draw.text((10,top+196),old['id'],fill='black')
        draw.text((450,top+196),new['id'],fill='black')
        rows.append({'before':old['id'],'after':new['id'],'bboxIoU':similarity(old,new)})
    comparison.save(after_root/'object_before_after.png')
    (after_root/'object_comparison.json').write_text(json.dumps(rows,indent=2),encoding='utf8')


if __name__ == '__main__':
    main()
