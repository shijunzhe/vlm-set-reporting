"""FSR pilot v2 stimuli: one family, easy recognition, guarded layouts, a single load dose.
1024x1024 canvas (below every tested vendor's resize limit), visible 4x4 registry (256 px regions).
Target: filled triangle. Distractors: hollow circle and hollow square (shape-distinct, so recognition is
not the bottleneck). Glyph side 22 px; centres on a jittered 5x5 slot lattice per region, at least 20 px
from every region border (no glyph crosses a border, so no cross-region matching and no cut glyphs).
Dose: total targets n in {16, 64, 192, 384}; 6 scenes per dose. Matching tolerance tau = 11 px (half a glyph).
Ground truth is written to a separate file and never enters prompts.
"""
import json, random, hashlib
from pathlib import Path
from PIL import Image, ImageDraw

OUT = Path(__file__).parent / 'data_v2'
W = H = 1024; CELL = 256; G = 22; TAU = 11.0
OFFS = [36, 76, 116, 156, 196]          # slot centres within a region (label sits in the top-left 35x17 px)
DOSES = [16, 64, 192, 384]; PER = 6

def draw_glyph(d, x, y, kind):
    h = G / 2
    if kind == 'target':
        d.polygon([(x, y - h), (x - h, y + h), (x + h, y + h)], fill='black')
    elif kind == 'circle':
        d.ellipse((x - h, y - h, x + h, y + h), outline='black', width=2)
    else:
        d.rectangle((x - h + 1, y - h + 1, x + h - 1, y + h - 1), outline='black', width=2)

def overlay(d):
    regs = []
    for row in range(4):
        for col in range(4):
            rid = f'R{row*4+col+1:03d}'; x0, y0 = col * CELL, row * CELL
            regs.append({'region_id': rid, 'x0': x0, 'y0': y0, 'width': CELL, 'height': CELL})
            d.rectangle((x0, y0, x0 + CELL - 1, y0 + CELL - 1), outline=(180, 180, 180), width=1)
            d.text((x0 + 5, y0 + 5), rid, fill='black')
    return regs

def scene(dose, idx):
    rng = random.Random(int(hashlib.sha256(f'fsr-v2:{dose}:{idx}'.encode()).hexdigest()[:12], 16))
    counts = [0] * 16
    for _ in range(dose):                      # random allocation, cap 24 of 25 slots per region
        free = [i for i in range(16) if counts[i] < 24]; counts[rng.choice(free)] += 1
    im = Image.new('RGB', (W, H), 'white'); d = ImageDraw.Draw(im)
    regs = overlay(d); targets = []
    for i, r in enumerate(regs):
        slots = [(r['x0'] + ox + rng.randint(-4, 4), r['y0'] + oy + rng.randint(-4, 4)) for ox in OFFS for oy in OFFS]
        rng.shuffle(slots)
        nt = counts[i]; nd = min(25 - nt, max(1, nt // 2) if dose < 384 else 25 - nt)
        for k, (x, y) in enumerate(slots[:nt + nd]):
            if k < nt:
                draw_glyph(d, x, y, 'target'); targets.append({'instance_id': f'T{len(targets)+1:04d}', 'owner': r['region_id'], 'label': 'target', 'point': [x, y]})
            else:
                draw_glyph(d, x, y, rng.choice(['circle', 'square']))
    return im, regs, targets

def masked(im, keep):
    m = Image.new('RGB', (W, H), 'white'); overlay(ImageDraw.Draw(m))
    x0, y0 = keep['x0'], keep['y0']; m.paste(im.crop((x0, y0, x0 + CELL, y0 + CELL)), (x0, y0)); return m

def main():
    OUT.mkdir(exist_ok=True); (OUT / 'masked').mkdir(exist_ok=True)
    ref = Image.new('RGB', (48, 48), 'white'); d = ImageDraw.Draw(ref)
    d.polygon([(24, 13), (13, 35), (35, 35)], fill='black'); ref.save(OUT / 'reference.png')
    samples, anns = [], []
    for dose in DOSES:
        for i in range(PER):
            sid = f'v2_n{dose:03d}_{i}'; im, regs, tg = scene(dose, i)
            im.save(OUT / f'{sid}.png')
            for r in regs: masked(im, r).save(OUT / 'masked' / f'{sid}_{r["region_id"]}.png')
            samples.append({'sample_id': sid, 'source_id': sid, 'dose': dose, 'image_path': f'data_v2/{sid}.png',
                            'reference_path': 'data_v2/reference.png', 'width': W, 'height': H, 'regions': regs})
            anns.append({'sample_id': sid, 'source_id': sid, 'tau': TAU, 'targets': tg})
            assert len(tg) == dose
    (OUT / 'samples.jsonl').write_text(''.join(json.dumps(s) + '\n' for s in samples))
    (OUT / 'annotations_DEV.jsonl').write_text(''.join(json.dumps(a) + '\n' for a in anns))
    print(len(samples), 'scenes')

if __name__ == '__main__':
    main()
