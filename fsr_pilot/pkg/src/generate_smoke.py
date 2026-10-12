"""Generate four NEW procedural smoke-test scenes. Not research results or a benchmark.
Requires Pillow. Generates input-only manifest and separate annotations. Targets are
circle-plus-diagonal symbols; distractors vary central strokes. The reference is fixed.
Use these to test plumbing only; design and freeze a richer independent E1 corpus later.
"""
from __future__ import annotations
import json, random, hashlib
from pathlib import Path
from PIL import Image, ImageDraw

def glyph(draw, x, y, kind, radius=10):
    draw.ellipse((x-radius,y-radius,x+radius,y+radius),outline='black',width=2)
    if kind=='target': draw.line((x-6,y+6,x+6,y-6),fill='black',width=2)
    elif kind=='horizontal': draw.line((x-6,y,x+6,y),fill='black',width=2)
    elif kind=='vertical': draw.line((x,y-6,x,y+6),fill='black',width=2)

def main():
    out=Path('examples'); out.mkdir(exist_ok=True)
    ref=Image.new('RGB',(48,48),'white'); glyph(ImageDraw.Draw(ref),24,24,'target')
    ref.save(out/'reference.png')
    samples=[]; annotations=[]
    for sid in range(4):
        rng=random.Random(82000+sid); width=height=768; cell=192
        image=Image.new('RGB',(width,height),'white'); d=ImageDraw.Draw(image)
        regions=[]; targets=[]; counter=0
        for row in range(4):
            for col in range(4):
                rid=f'R{row*4+col+1:03d}'; x0,y0=col*cell,row*cell
                regions.append({'region_id':rid,'x0':x0,'y0':y0,'width':cell,'height':cell})
                # Identical markers across all E1 loads. This is not a marks ablation.
                d.rectangle((x0,y0,x0+cell-1,y0+cell-1),outline=(180,180,180),width=1)
                d.text((x0+5,y0+5),rid,fill='black')
                positions=[(x0+48+48*a,y0+48+48*b) for a in range(3) for b in range(3)]
                rng.shuffle(positions)
                count=0 if sid==0 else rng.choice([0,1,2,4])
                for k,(x,y) in enumerate(positions[:count+2]):
                    kind='target' if k<count else rng.choice(['horizontal','vertical','empty'])
                    glyph(d,x,y,kind)
                    if kind=='target':
                        counter+=1; targets.append({'instance_id':f'T{counter:04d}','owner':rid,
                                                   'label':'target','point':[x,y]})
        image_path=out/f'smoke_{sid:02d}.png'; image.save(image_path)
        sample_id=f'smoke_{sid:02d}'
        samples.append({'sample_id':sample_id,'source_id':sample_id,'split':'smoke_only',
                        'image_path':str(image_path),'reference_path':'examples/reference.png',
                        'width':width,'height':height,'regions':regions})
        annotations.append({'sample_id':sample_id,'source_id':sample_id,'tau':8.0,'targets':targets})
    (out/'samples.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in samples))
    (out/'annotations_SMOKE_ONLY.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in annotations))
    print('Generated 4 procedural smoke scenes; no model inference.')
if __name__=='__main__': main()
