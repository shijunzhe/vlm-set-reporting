"""B1: readout channel, causal and component tests (SCP 100 images + PixMo-Count 100 images). No model calls.
Arms (SCP):
  a  one-sided wording ('Do not miss any'), counts first        b  two-sided wording, counts first
  c  two-sided, list first                                      d  two-sided, list only (no count field)
  e  d + localisation instruction (no marks)                     f/g/h  d + marker instruction + edge / grid / random marks
  i  d + spatial scan order                                      j  d + grouped-by-type order
  k  d + 'There are about N shapes' (N = true x m, m in {0.7,0.85,1,1.15,1.3}, two independent draws: k1, k2)
  l  Gemini only: d with the pixel contract (others: native contract)
  e-h are run twice (rep 0, 1).
PixMo: a, c, d, i, k1, k2 with the class name.
Usage: python -I build_b1.py [--pilot]"""
import json, os, sys, random, argparse
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bcommon as C
ap = argparse.ArgumentParser(); ap.add_argument('--pilot', action='store_true'); A = ap.parse_args()
R = C.ROOT; rng = random.Random(20261011)
SCP_IMG = '/home/claude/release/test_images'; SCP_GT = '/home/claude/release/ground_truth'
TYPES = 'circle, triangle, square, star, pentagon'
ONE = 'Be very careful and systematic. Count every single shape. Do not miss any.'
TWO = 'Locate every shape carefully. Count systematically; do not miss any and do not invent shapes that are not there.'
LOC = "Use the image borders and nearby shapes as reference points to estimate each shape's coordinate as accurately as possible."
MARK = ("Faint semi-transparent magenta '+' markers are drawn in the image as visual reference points. They are reference markers "
        "only, NOT shapes to count. Use them to estimate each shape's coordinate as accurately as possible.")
SCAN = 'List the shapes in reading order: scan the image row by row from top to bottom, and within a row from left to right.'
GROUP = 'List all circles first, then all triangles, then squares, then stars, then pentagons.'
EX_SHAPE = {'grid': '{"type": "circle", "x": 402, "y": 311}', 'pixel': '{"type": "circle", "x": 412, "y": 318}'}
COUNTS = '"counts_by_type": {"circle": 5, "triangle": 3, "square": 0, "star": 2, "pentagon": 4}'


def scp_system(arm, mode, w, h):
    shapes = '"shapes": [' + EX_SHAPE[mode] + ', ...]'
    if arm in ('a', 'b'): fmt = '{' + COUNTS + ', ' + shapes + '}'
    elif arm == 'c': fmt = '{' + shapes + ', ' + COUNTS + '}'
    else: fmt = '{' + shapes + '}'
    extra = {'e': LOC, 'f': MARK, 'g': MARK, 'h': MARK, 'i': SCAN, 'j': GROUP}.get(arm, '')
    wording = ONE if arm == 'a' else TWO
    s = ('You are a precise visual counting and localization assistant.\n\nGiven an image containing geometric shapes, identify '
         'every shape.\n' + C.coord_sentence(mode, w, h) + '\n' + (extra + '\n' if extra else '') +
         '\nRespond with ONLY a JSON object (no markdown, no commentary) in this format:\n' + fmt + '\n\n'
         f'Shape types to recognize: {TYPES}.\n' + wording)
    if arm == 'c': s += '\nWrite the full shapes list first; after it, give counts_by_type for the shapes you listed.'
    return s


def scp_user(arm, hint=None):
    u = {'a': 'Count every geometric shape in this image. Output counts by type and each shape\'s center. Respond with JSON only.',
         'b': 'Count every geometric shape in this image. Output counts by type and each shape\'s center. Respond with JSON only.',
         'c': 'List every geometric shape in this image with its center, then counts by type. Respond with JSON only.'}.get(
        arm, 'List every geometric shape in this image with its center. Respond with JSON only.')
    if hint is not None: u += f' Hint: there are about {hint} shapes in this image.'
    return u


def marks(layout, gt, seed):
    if layout == 'g': return [(100 * i, 100 * j) for i in range(1, 11) for j in range(1, 11)]
    if layout == 'f':
        pts = []
        for t in np.linspace(20, 1004, 25):
            pts += [(t, 10), (t, 1014), (10, t), (1014, t)]
        return pts
    r = random.Random(seed); pts = []; S = gt['shapes']; tries = 0
    while len(pts) < 100 and tries < 200000:
        tries += 1; x, y = r.uniform(12, 1012), r.uniform(12, 1012)
        if all((x - s['cx']) ** 2 + (y - s['cy']) ** 2 > (s['radius'] + 12) ** 2 for s in S): pts.append((x, y))
    return pts


reqs = []; meta_items = {}
ids = list(range(100)) if not A.pilot else [0, 17, 42, 63, 88]
os.makedirs(f'{R}/data/scp', exist_ok=True)
for i in ids:
    gt = json.load(open(f'{SCP_GT}/gt_{i:03d}.json')); n = gt['total_shapes']
    base = Image.open(f'{SCP_IMG}/test_{i:03d}.png').convert('RGB')
    paths = {'none': f'data/scp/{i:03d}.png'}
    if not os.path.exists(f'{R}/{paths["none"]}'): base.save(f'{R}/{paths["none"]}')
    for lay in ('f', 'g', 'h'):
        p = f'data/scp/{i:03d}_{lay}.png'; paths[lay] = p
        if not os.path.exists(f'{R}/{p}'): C.draw_markers(base, marks(lay, gt, 1000 + i)).save(f'{R}/{p}')
    mults = [rng.choice([0.7, 0.85, 1.0, 1.15, 1.3]) for _ in range(2)]
    meta_items[i] = dict(n=n, mult_k1=mults[0], mult_k2=mults[1])
    for m in C.MODELS:
        mode = C.MODE[m]
        arms = [('a', 0), ('b', 0), ('c', 0), ('d', 0), ('i', 0), ('j', 0), ('k1', 0), ('k2', 0)] + [(x, r) for x in 'efgh' for r in (0, 1)]
        if m == 'gem38': arms.append(('l', 0))
        for arm, rep in arms:
            a0 = arm[0]; md = 'pixel' if arm == 'l' else mode
            img = paths.get(a0, paths['none']) if a0 in 'fgh' else paths['none']
            hint = round(n * mults[int(arm[1]) - 1]) if a0 == 'k' else None
            sysarm = 'd' if arm in ('k1', 'k2', 'l') else a0
            reqs.append(dict(request_id=f'scp|{i:03d}|{m}|{arm}|r{rep}', model_key=m, setting='min', images=[img],
                             system=scp_system(sysarm, md, 1024, 1024), user=scp_user(sysarm, hint),
                             meta=dict(task='scp', item=i, arm=arm, rep=rep, mode=md, w=1024, h=1024, sent=[1024, 1024], n=n, hint=hint)))
# ---------------- SCP-dense (120-200 smaller shapes; generated with the released SCP generator, seed 2026)
SD = f'{R}/data/scpd_gen'
os.makedirs(f'{R}/data/scpd', exist_ok=True)
for i in (list(range(50)) if not A.pilot else [0, 1]):
    gt = json.load(open(f'{SD}/ground_truth/gt_{i:03d}.json')); n = gt['total_shapes']
    p = f'data/scpd/{i:03d}.png'
    if not os.path.exists(f'{R}/{p}'): Image.open(f'{SD}/test_images/test_{i:03d}.png').convert('RGB').save(f'{R}/{p}')
    mults = [rng.choice([0.7, 0.85, 1.0, 1.15, 1.3]) for _ in range(2)]
    meta_items[f'scpd_{i}'] = dict(n=n, mult_k1=mults[0], mult_k2=mults[1])
    for m in C.MODELS:
        mode = C.MODE[m]
        for arm in ('a', 'b', 'c', 'd', 'i', 'j', 'k1', 'k2'):
            hint = round(n * mults[int(arm[1]) - 1]) if arm[0] == 'k' else None
            sysarm = 'd' if arm[0] == 'k' else arm
            reqs.append(dict(request_id=f'scpd|{i:03d}|{m}|{arm}|r0', model_key=m, setting='min', images=[p],
                             system=scp_system(sysarm, mode, 1024, 1024), user=scp_user(sysarm, hint),
                             meta=dict(task='scpd', item=i, arm=arm, rep=0, mode=mode, w=1024, h=1024, sent=[1024, 1024], n=n, hint=hint)))
# ---------------- PixMo-Count
PX = '/home/claude/x2/arr/data_e4/pixmo_count'
man = json.load(open(f'{PX}/manifest.json'))['items']
by = {}
for it in man: by.setdefault(it['gt_count'], []).append(it)
r2 = random.Random(7); pick = []
cnts = sorted(by)
while len(pick) < (100 if not A.pilot else 5):
    for c in cnts:
        if by[c] and len(pick) < (100 if not A.pilot else 5): pick.append(by[c].pop(r2.randrange(len(by[c]))))
os.makedirs(f'{R}/data/pixmo', exist_ok=True)


def px_system(arm, mode, w, h, label):
    pts = '"points": [' + ('{"x": 402, "y": 311}' if mode == 'grid' else '{"x": 412, "y": 318}') + ', ...]'
    fmt = {'a': '{"count": <integer>, ' + pts + '}', 'c': '{' + pts + ', "count": <integer>}'}.get(arm, '{' + pts + '}')
    wording = (f'Be very careful and systematic. Count every single instance of "{label}". Do not miss any.' if arm == 'a' else
               f'Locate every instance of "{label}" carefully. Do not miss any and do not include anything that is not an instance of "{label}".')
    extra = (f'List the instances in reading order: scan the image from top to bottom, and within a row from left to right.\n' if arm == 'i' else '')
    s = (f'You are a precise visual counting and localization assistant.\n\nThe target class is "{label}". Find every instance of '
         'this class in the image.\n' + C.coord_sentence(mode, w, h) + '\n' + extra +
         '\nRespond with ONLY a JSON object (no markdown, no commentary) in this format:\n' + fmt + '\n'
         'Each point is the center of one instance.\n' + wording)
    if arm == 'c': s += '\nWrite the full points list first; after it, give the count of the points you listed.'
    return s


for it in pick:
    im = Image.open(f'{PX}/{it["file"]}').convert('RGB'); W, H = im.size
    meta_items[f'px_{it["item_id"]}'] = dict(n=it['gt_count'], label=it['label'])
    mults = [rng.choice([0.7, 0.85, 1.0, 1.15, 1.3]) for _ in range(2)]
    for m in C.MODELS:
        mode = C.MODE[m]; sent = C.prepare(im, m)
        p = f'data/pixmo/{it["item_id"]}_{"a" if m == "sonnet55" else "g"}.png'
        if not os.path.exists(f'{R}/{p}'): sent.save(f'{R}/{p}')
        for arm in ('a', 'c', 'd', 'i', 'k1', 'k2'):
            n = it['gt_count']; hint = max(1, round(n * mults[int(arm[1]) - 1])) if arm[0] == 'k' else None
            sysarm = 'd' if arm[0] == 'k' else arm
            lb = it['label']
            u = {'a': f'Count every instance of "{lb}" in this image and give each one\'s center. Respond with JSON only.',
                 'c': f'List every instance of "{lb}" in this image with its center, then the count. Respond with JSON only.'}.get(
                sysarm, f'List every instance of "{lb}" in this image with its center. Respond with JSON only.')
            if hint is not None: u += f' Hint: there are about {hint} instances of "{lb}" in this image.'
            reqs.append(dict(request_id=f'px|{it["item_id"]}|{m}|{arm}|r0', model_key=m, setting='min', images=[p],
                             system=px_system(sysarm, mode, *sent.size, it['label']), user=u,
                             meta=dict(task='pixmo', item=it['item_id'], arm=arm, rep=0, mode=mode, w=W, h=H, sent=list(sent.size), n=n, hint=hint, label=it['label'])))
sub = 'b1pilot' if A.pilot else 'b1'
C.write_manifest(f'{R}/runs/{sub}/requests_main.jsonl', reqs)
json.dump(meta_items, open(f'{R}/runs/{sub}/items.json', 'w'), indent=1)
from collections import Counter
print(sub, len(reqs), Counter(q['model_key'] for q in reqs), Counter(q['meta']['task'] for q in reqs))
