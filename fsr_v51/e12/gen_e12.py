"""v5-E12 stimuli (development/calibration data for E3; no model calls).

Same visual family as the v2 pilot (filled-triangle target, hollow circle/square distractors,
22 px glyphs, visible 4x4 registry on a 1024x1024 canvas), but glyph centres are continuous
random positions instead of a 5x5 slot lattice, so target count no longer maps to a near-full
lattice. Design constraints:
  * minimum distance between any two glyph centres: 36 px, checked on the final rounded coordinates
    (so an 18 px residual tolerance never pairs a point with two targets);
  * every glyph centre at least 20 px from every region border (b_tau = 0 for every tau + delta <= 19,
    i.e. tau = 11 with the whole delta grid up to 8);
  * no glyph overlaps the region label box in the top-left corner.
Doses: n in {96, 192} targets, 6 scenes each, n/2 distractors. Two focal regions per scene are
chosen by a public hash of the scene id (no labels). Ground truth is written to separate files.
"""
import hashlib, json, math, random
from pathlib import Path
from PIL import Image, ImageDraw

OUT = Path(__file__).parent / 'data_e12'
W = H = 1024; CELL = 256; G = 22
MIN_D = 36.0; GUARD = 20; LABEL_X, LABEL_Y = 52, 34      # centre exclusion box for the label corner
DOSES = [96, 192]; PER = 6; CAP = 22                      # at most 22 glyphs per region


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
            rid = f'R{row * 4 + col + 1:03d}'; x0, y0 = col * CELL, row * CELL
            regs.append({'region_id': rid, 'x0': x0, 'y0': y0, 'width': CELL, 'height': CELL})
            d.rectangle((x0, y0, x0 + CELL - 1, y0 + CELL - 1), outline=(180, 180, 180), width=1)
            d.text((x0 + 5, y0 + 5), rid, fill='black')
    return regs


def sample_region(rng, x0, y0, k, placed):
    pts = []
    for _ in range(k):
        for _try in range(20000):
            x = round(rng.uniform(x0 + GUARD, x0 + CELL - GUARD), 1); y = round(rng.uniform(y0 + GUARD, y0 + CELL - GUARD), 1)
            if x < x0 + LABEL_X and y < y0 + LABEL_Y: continue
            if all(math.hypot(x - a, y - b) >= MIN_D for a, b in placed + pts):
                pts.append((x, y)); break
        else:
            return None
    return pts


def scene(dose, idx):
    for attempt in range(200):
        rng = random.Random(int(hashlib.sha256(f'fsr-e12:{dose}:{idx}:{attempt}'.encode()).hexdigest()[:12], 16))
        nt = [0] * 16; nd = [0] * 16
        for _ in range(dose):
            free = [i for i in range(16) if nt[i] + nd[i] < CAP]; nt[rng.choice(free)] += 1
        for _ in range(dose // 2):
            free = [i for i in range(16) if nt[i] + nd[i] < CAP]; nd[rng.choice(free)] += 1
        im = Image.new('RGB', (W, H), 'white'); d = ImageDraw.Draw(im); regs = overlay(d)
        targets, distractors, placed, ok = [], [], [], True
        for i, r in enumerate(regs):
            pts = sample_region(rng, r['x0'], r['y0'], nt[i] + nd[i], placed)
            if pts is None: ok = False; break
            placed += pts; rng.shuffle(pts)
            for k, (x, y) in enumerate(pts):
                if k < nt[i]:
                    draw_glyph(d, x, y, 'target')
                    targets.append({'instance_id': f'T{len(targets) + 1:04d}', 'owner': r['region_id'], 'label': 'target', 'point': [x, y]})
                else:
                    kind = rng.choice(['circle', 'square']); draw_glyph(d, x, y, kind)
                    distractors.append({'owner': r['region_id'], 'kind': kind, 'point': [x, y]})
        if ok:
            return im, regs, targets, distractors, attempt
    raise RuntimeError('could not place glyphs')


def masked(im, keep):
    m = Image.new('RGB', (W, H), 'white'); overlay(ImageDraw.Draw(m))
    x0, y0 = keep['x0'], keep['y0']; m.paste(im.crop((x0, y0, x0 + CELL, y0 + CELL)), (x0, y0)); return m


def focal_regions(sid):
    h = hashlib.sha256(f'e12-focal:{sid}'.encode()).hexdigest()
    order = sorted(range(16), key=lambda i: hashlib.sha256(f'{h}:{i}'.encode()).hexdigest())
    return [f'R{order[0] + 1:03d}', f'R{order[1] + 1:03d}']


def main():
    OUT.mkdir(exist_ok=True); (OUT / 'masked').mkdir(exist_ok=True)
    ref = Image.new('RGB', (48, 48), 'white'); d = ImageDraw.Draw(ref)
    d.polygon([(24, 13), (13, 35), (35, 35)], fill='black'); ref.save(OUT / 'reference.png')
    samples, anns = [], []
    for dose in DOSES:
        for i in range(PER):
            sid = f'e12_n{dose:03d}_{i}'; im, regs, tg, ds, att = scene(dose, i)
            im.save(OUT / f'{sid}.png'); foc = focal_regions(sid)
            for r in regs:
                if r['region_id'] in foc: masked(im, r).save(OUT / 'masked' / f'{sid}_{r["region_id"]}.png')
            samples.append({'sample_id': sid, 'dose': dose, 'image_path': f'data_e12/{sid}.png', 'reference_path': 'data_e12/reference.png',
                            'width': W, 'height': H, 'regions': regs, 'focal_regions': foc, 'placement_attempt': att})
            anns.append({'sample_id': sid, 'tau': 11.0, 'targets': tg, 'distractors': ds})
            assert len(tg) == dose and len(ds) == dose // 2
            allp = [t['point'] for t in tg] + [d_['point'] for d_ in ds]
            assert min(math.hypot(p[0] - q[0], p[1] - q[1]) for i, p in enumerate(allp) for q in allp[i + 1:]) >= MIN_D
            assert min(min(p[0] % CELL, CELL - p[0] % CELL, p[1] % CELL, CELL - p[1] % CELL) for p in allp) >= GUARD
    (OUT / 'samples.jsonl').write_text(''.join(json.dumps(s) + '\n' for s in samples))
    (OUT / 'annotations_DEV.jsonl').write_text(''.join(json.dumps(a) + '\n' for a in anns))
    print(len(samples), 'scenes')


if __name__ == '__main__':
    main()
