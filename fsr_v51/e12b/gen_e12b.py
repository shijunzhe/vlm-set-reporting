"""v5.2-E12b stimuli (no model calls). Same visual family as E12 (filled-triangle target, hollow circle/square
distractors, 1024x1024 canvas, 4x4 responsibility registry), new seeds.

Blocks
  A   24 scenes (12 with n=96, 12 with n=192), glyph 22 px; each rendered twice: grid ON (E12 overlay: region lines
      and region labels) and grid OFF (blank background). Glyph positions are identical in the two renderings.
      B1 reuses 12 A scenes (grid OFF) with 2x2 and 8x8 registries (text only).
  B2  four new cells, grid ON, 6 scenes each:
        c024 / c048: 6 targets in each of 4 / 8 occupied regions (chosen by seed), the rest of the regions hold no
                     target; n/2 distractors spread over all 16 regions;
        g14 / g32:   n = 96, glyph 14 / 32 px, reference symbol drawn at the same glyph size.
Constraints (asserted on the final rounded coordinates)
  * minimum centre distance MIN_D(G) = max(36, G + 14) px (edge gap at least 14 px; >= 2 x 18 px residual tolerance);
  * every centre at least 20 px from every 4x4 region border (b_tau = 0 for tau + delta <= 19);
  * the label corner is kept free in BOTH renderings so that the two images differ only by the overlay.
Ground truth goes to separate files.
"""
import hashlib, json, math, random
from pathlib import Path
from PIL import Image, ImageDraw

OUT = Path(__file__).parent / 'data_e12b'
W = H = 1024; CELL = 256; GUARD = 20; LABEL_X, LABEL_Y = 52, 34


def min_d(G): return max(36.0, G + 14.0)


def draw_glyph(d, x, y, kind, G):
    h = G / 2
    if kind == 'target':
        d.polygon([(x, y - h), (x - h, y + h), (x + h, y + h)], fill='black')
    elif kind == 'circle':
        d.ellipse((x - h, y - h, x + h, y + h), outline='black', width=2)
    else:
        d.rectangle((x - h + 1, y - h + 1, x + h - 1, y + h - 1), outline='black', width=2)


def registry(k):
    """k x k registry over the canvas, row-major IDs R001..."""
    c = W // k
    return [{'region_id': f'R{r * k + q + 1:03d}', 'x0': q * c, 'y0': r * c, 'width': c, 'height': c} for r in range(k) for q in range(k)]


def overlay(d):
    for r in registry(4):
        x0, y0 = r['x0'], r['y0']
        d.rectangle((x0, y0, x0 + CELL - 1, y0 + CELL - 1), outline=(180, 180, 180), width=1)
        d.text((x0 + 5, y0 + 5), r['region_id'], fill='black')


def sample_region(rng, x0, y0, k, placed, G):
    pts, md = [], min_d(G); lx, ly = LABEL_X + max(0, (G - 22) / 2), LABEL_Y + max(0, (G - 22) / 2)
    for _ in range(k):
        for _try in range(20000):
            x = round(rng.uniform(x0 + GUARD, x0 + CELL - GUARD), 1); y = round(rng.uniform(y0 + GUARD, y0 + CELL - GUARD), 1)
            if x < x0 + lx and y < y0 + ly: continue
            if all(math.hypot(x - a, y - b) >= md for a, b in placed + pts):
                pts.append((x, y)); break
        else:
            return None
    return pts


def layout(tag, n, G=22, occupied=None, cap=22):
    """Glyph positions only (rendering is separate). occupied: number of regions that hold targets (6 each)."""
    for attempt in range(400):
        rng = random.Random(int(hashlib.sha256(f'fsr-e12b:{tag}:{attempt}'.encode()).hexdigest()[:12], 16))
        nt = [0] * 16; nd = [0] * 16
        if occupied:
            for i in rng.sample(range(16), occupied): nt[i] = n // occupied
        else:
            for _ in range(n):
                free = [i for i in range(16) if nt[i] + nd[i] < cap]; nt[rng.choice(free)] += 1
        for _ in range(n // 2):
            free = [i for i in range(16) if nt[i] + nd[i] < cap]; nd[rng.choice(free)] += 1
        regs = registry(4); targets, distractors, placed, ok = [], [], [], True
        for i, r in enumerate(regs):
            pts = sample_region(rng, r['x0'], r['y0'], nt[i] + nd[i], placed, G)
            if pts is None: ok = False; break
            placed += pts; rng.shuffle(pts)
            for k, (x, y) in enumerate(pts):
                if k < nt[i]:
                    targets.append({'instance_id': f'T{len(targets) + 1:04d}', 'owner': r['region_id'], 'label': 'target', 'point': [x, y]})
                else:
                    distractors.append({'owner': r['region_id'], 'kind': rng.choice(['circle', 'square']), 'point': [x, y]})
        if ok:
            return targets, distractors, attempt
    raise RuntimeError(f'could not place glyphs for {tag}')


def render(targets, distractors, G, grid):
    im = Image.new('RGB', (W, H), 'white'); d = ImageDraw.Draw(im)
    if grid: overlay(d)
    for t in targets: draw_glyph(d, *t['point'], 'target', G)
    for s in distractors: draw_glyph(d, *s['point'], s['kind'], G)
    return im


def reference(G):
    s = int(G + 26); im = Image.new('RGB', (s, s), 'white'); d = ImageDraw.Draw(im); c = s / 2; h = G / 2
    d.polygon([(c, c - h), (c - h, c + h), (c + h, c + h)], fill='black'); return im


def check(tg, ds, G):
    allp = [t['point'] for t in tg] + [s['point'] for s in ds]
    assert min(math.hypot(p[0] - q[0], p[1] - q[1]) for i, p in enumerate(allp) for q in allp[i + 1:]) >= min_d(G)
    assert min(min(p[0] % CELL, CELL - p[0] % CELL, p[1] % CELL, CELL - p[1] % CELL) for p in allp) >= GUARD


def main():
    OUT.mkdir(exist_ok=True)
    for G in (14, 22, 32): reference(G).save(OUT / f'reference_g{G}.png')
    samples, anns = [], []
    def add(sid, block, cell, n, G, grids, **kw):
        tg, ds, att = layout(sid, n, G, **kw); check(tg, ds, G)
        assert len(tg) == n and len(ds) == n // 2
        paths = {}
        for g in grids:
            p = f'data_e12b/{sid}_grid{g}.png'; render(tg, ds, G, g == 'on').save(OUT / Path(p).name); paths[g] = p
        samples.append({'sample_id': sid, 'block': block, 'cell': cell, 'dose': n, 'glyph': G, 'image_paths': paths,
                        'reference_path': f'data_e12b/reference_g{G}.png', 'width': W, 'height': H,
                        'regions': registry(4), 'placement_attempt': att})
        anns.append({'sample_id': sid, 'tau': 11.0, 'targets': tg, 'distractors': ds})
    for n in (96, 192):
        for i in range(12): add(f'e12b_A_n{n:03d}_{i:02d}', 'A', f'n{n}', n, 22, ('on', 'off'))
    for occ, n in ((4, 24), (8, 48)):
        for i in range(6): add(f'e12b_B2_c{n:03d}_{i}', 'B2', f'c{n:03d}', n, 22, ('on',), occupied=occ)
    for G in (14, 32):
        for i in range(6): add(f'e12b_B2_g{G}_{i}', 'B2', f'g{G}', 96, G, ('on',), cap=22 if G < 30 else 12)
    (OUT / 'samples.jsonl').write_text(''.join(json.dumps(s) + '\n' for s in samples))
    (OUT / 'annotations_DEV.jsonl').write_text(''.join(json.dumps(a) + '\n' for a in anns))
    print(len(samples), 'scenes;', sum(len(s['image_paths']) for s in samples), 'images')


if __name__ == '__main__':
    main()
