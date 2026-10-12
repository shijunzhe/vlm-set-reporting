"""FSR pilot development layouts (12 scenes, 3 prespecified families x 4), plus masked-region images.
Same glyph vocabulary and visible 4x4 registry overlay as the package smoke generator.
Families: A separated (guarded lattice, >= 48 px from region borders), B crowded (jittered, near borders,
up to 9 targets per region), C background-heavy (many distractors and clutter strokes, few targets).
Ground truth goes to a separate annotations file and never into prompts.
"""
import json, random, hashlib
from pathlib import Path
from PIL import Image, ImageDraw

OUT = Path(__file__).parent / 'data'
W = H = 768; CELL = 192; TAU = 8.0

def glyph(d, x, y, kind, r=10):
    d.ellipse((x - r, y - r, x + r, y + r), outline='black', width=2)
    k = int(r * 0.6)
    if kind == 'target': d.line((x - k, y + k, x + k, y - k), fill='black', width=2)
    elif kind == 'horizontal': d.line((x - k, y, x + k, y), fill='black', width=2)
    elif kind == 'vertical': d.line((x, y - k, x, y + k), fill='black', width=2)
    elif kind == 'anti': d.line((x - k, y - k, x + k, y + k), fill='black', width=2)

def overlay(d):
    regions = []
    for row in range(4):
        for col in range(4):
            rid = f'R{row*4+col+1:03d}'; x0, y0 = col * CELL, row * CELL
            regions.append({'region_id': rid, 'x0': x0, 'y0': y0, 'width': CELL, 'height': CELL})
            d.rectangle((x0, y0, x0 + CELL - 1, y0 + CELL - 1), outline=(180, 180, 180), width=1)
            d.text((x0 + 5, y0 + 5), rid, fill='black')
    return regions

def owner_of(x, y):
    return f'R{int(y // CELL) * 4 + int(x // CELL) + 1:03d}'

def place(rng, n, existing, mind, lo, hi_x, hi_y, tries=4000):
    pts = []
    for _ in range(tries):
        if len(pts) == n: break
        x, y = rng.uniform(lo[0], hi_x), rng.uniform(lo[1], hi_y)
        if x % CELL < 16 and y % CELL < 22 and False: continue
        if all((x - a) ** 2 + (y - b) ** 2 >= mind ** 2 for a, b in existing + pts):
            # keep clear of the region label text in each cell's top-left corner
            if (x % CELL) < 40 and (y % CELL) < 22: continue
            pts.append((round(x), round(y)))
    return pts

def scene(fam, idx):
    rng = random.Random(int(hashlib.sha256(f'fsr-pilot-v1:{fam}:{idx}'.encode()).hexdigest()[:12], 16))
    im = Image.new('RGB', (W, H), 'white'); d = ImageDraw.Draw(im)
    targets, distract = [], []
    if fam == 'A':
        for row in range(4):
            for col in range(4):
                x0, y0 = col * CELL, row * CELL
                pos = [(x0 + 48 + 48 * a, y0 + 48 + 48 * b) for a in range(3) for b in range(3)]
                rng.shuffle(pos); c = rng.choice([0, 0, 1, 2, 3, 4])
                targets += pos[:c]; distract += pos[c:c + rng.choice([1, 2, 3])]
        radius = lambda: 10
    elif fam == 'B':
        occupied = []
        for row in range(4):
            for col in range(4):
                c = rng.choice([0, 2, 4, 6, 9])
                # centres may lie anywhere in the region, including within 12 px of its border
                pts = place(rng, c, occupied, 26, (col * CELL + 4, row * CELL + 4), col * CELL + CELL - 4, row * CELL + CELL - 4)
                occupied += pts; targets += pts
        distract = place(rng, 40, targets, 26, (12, 12), W - 12, H - 12)
        radius = lambda: rng.choice([8, 9, 10])
    else:
        pts = place(rng, rng.choice([3, 5, 7]), [], 40, (20, 20), W - 20, H - 20)
        targets = pts
        distract = place(rng, 70, targets, 26, (12, 12), W - 12, H - 12)
        for _ in range(60):  # clutter strokes
            x, y = rng.uniform(0, W), rng.uniform(0, H); L = rng.uniform(20, 120)
            if rng.random() < .5: d.line((x, y, x + L, y), fill=(60, 60, 60), width=1)
            else: d.line((x, y, x, y + L), fill=(60, 60, 60), width=1)
        radius = lambda: 10
    regions = overlay(d)
    ann = []
    for i, (x, y) in enumerate(targets):
        glyph(d, x, y, 'target', radius()); ann.append({'instance_id': f'T{i+1:04d}', 'owner': owner_of(x, y), 'label': 'target', 'point': [x, y]})
    for (x, y) in distract:
        glyph(d, x, y, rng.choice(['horizontal', 'vertical', 'anti', 'empty']), radius())
    return im, regions, ann

def masked(im, keep):
    """White out every region except `keep`, then redraw the identical registry overlay."""
    m = Image.new('RGB', (W, H), 'white'); x0, y0 = keep['x0'], keep['y0']
    overlay(ImageDraw.Draw(m))   # overlay first, then the kept region's own pixels (incl. its label) on top
    m.paste(im.crop((x0, y0, x0 + CELL, y0 + CELL)), (x0, y0)); return m

def main():
    OUT.mkdir(exist_ok=True); (OUT / 'masked').mkdir(exist_ok=True)
    ref = Image.new('RGB', (48, 48), 'white'); glyph(ImageDraw.Draw(ref), 24, 24, 'target'); ref.save(OUT / 'reference.png')
    samples, anns = [], []
    for fam in 'ABC':
        for i in range(4):
            sid = f'dev_{fam}{i}'; im, regions, ann = scene(fam, i)
            p = OUT / f'{sid}.png'; im.save(p)
            for r in regions:
                masked(im, r).save(OUT / 'masked' / f'{sid}_{r["region_id"]}.png')
            samples.append({'sample_id': sid, 'source_id': sid, 'family': fam, 'split': 'pilot_dev',
                            'image_path': f'data/{sid}.png', 'reference_path': 'data/reference.png',
                            'width': W, 'height': H, 'regions': regions})
            anns.append({'sample_id': sid, 'source_id': sid, 'tau': TAU, 'targets': ann})
    (OUT / 'samples.jsonl').write_text(''.join(json.dumps(s) + '\n' for s in samples))
    (OUT / 'annotations_DEV.jsonl').write_text(''.join(json.dumps(a) + '\n' for a in anns))
    for a, s in zip(anns, samples):
        print(s['sample_id'], s['family'], len(a['targets']))

if __name__ == '__main__':
    main()
