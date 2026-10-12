"""B1': applicability condition of the readout principle on FSC-147 (dense, small objects). No model calls.
100 images from the E4 FSC subset, stratified into size tertiles by exemplar size relative to the image.
Arms: cf = two-sided wording, count first; lo = list only; tl = 2x2 tiles (10% overlap), each crop upsampled 2x (then capped
by the provider rule), list only, four separate calls, merged in code (points from different crops within tau in an overlap
zone are merged). tau = half the mean exemplar short side (task input, as in E3).
Usage: python -I build_b1p.py [--pilot]"""
import json, os, sys, random, argparse
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bcommon as C
ap = argparse.ArgumentParser(); ap.add_argument('--pilot', action='store_true'); A = ap.parse_args()
R = C.ROOT; FS = '/home/claude/x2/arr/data_e4/fsc147'
man = json.load(open(f'{FS}/manifest.json'))['items']; ann = json.load(open(f'{FS}/annotation_FSC147_384.json'))
OVL = 0.10
CROPS = [(0, 0, 0.5 + OVL / 2, 0.5 + OVL / 2), (0.5 - OVL / 2, 0, 1, 0.5 + OVL / 2), (0, 0.5 - OVL / 2, 0.5 + OVL / 2, 1), (0.5 - OVL / 2, 0.5 - OVL / 2, 1, 1)]
rows = []
for it in man:
    a = ann.get(it['item_id']);
    if a is None: continue
    im = Image.open(f'{FS}/{it["file"]}'); W, H = im.size
    ex = json.loads(a['box_examples_coordinates']) if isinstance(a['box_examples_coordinates'], str) else a['box_examples_coordinates']
    sides = [(min(abs(b[2][0] - b[0][0]), abs(b[2][1] - b[0][1])), max(abs(b[2][0] - b[0][0]), abs(b[2][1] - b[0][1]))) for b in ex]
    tau = 0.5 * float(np.mean([s[0] for s in sides])); rel = float(np.mean([np.sqrt(s[0] * s[1]) for s in sides])) / np.sqrt(W * H)
    rows.append(dict(item=it['item_id'], file=it['file'], label=it['label'], n=it['gt_count'], W=W, H=H, tau=tau, rel_size=rel,
                     points=[[p['x'], p['y']] for p in it['gt_points']]))
rows.sort(key=lambda r: r['rel_size']); k = len(rows) // 3
strata = [rows[:k], rows[k:2 * k], rows[2 * k:]]
rng = random.Random(11); pick = []
per = [33, 33, 34] if not A.pilot else [2, 2, 1]
for s, (lst, cnt) in enumerate(zip(strata, per)):
    for r in rng.sample(lst, cnt): r['stratum'] = s; pick.append(r)
os.makedirs(f'{R}/data/fsc', exist_ok=True)


def system(arm, mode, w, h, label):
    pts = '"points": [' + ('{"x": 402, "y": 311}' if mode == 'grid' else '{"x": 412, "y": 318}') + ', ...]'
    fmt = '{"count": <integer>, ' + pts + '}' if arm == 'cf' else '{' + pts + '}'
    where = 'this image' if arm != 'tl' else 'this image, which is one crop of a larger picture (count instances whose center lies in the crop)'
    return (f'You are a precise visual counting and localization assistant.\n\nThe target class is "{label}". Find every instance of '
            f'this class in {where}.\n' + C.coord_sentence(mode, w, h) +
            '\n\nRespond with ONLY a JSON object (no markdown, no commentary) in this format:\n' + fmt + '\nEach point is the center of '
            f'one instance.\nLocate every instance of "{label}" carefully. Do not miss any and do not include anything that is not an '
            f'instance of "{label}".')


reqs = []
for r in pick:
    im = Image.open(f'{FS}/{r["file"]}').convert('RGB')
    for m in C.MODELS:
        mode = C.MODE[m]; tag = 'a' if m == 'sonnet55' else 'g'
        sent = C.prepare(im, m); p = f'data/fsc/{r["item"].split(".")[0]}_{tag}.png'
        if not os.path.exists(f'{R}/{p}'): sent.save(f'{R}/{p}')
        for arm in ('cf', 'lo'):
            u = (f'Count every instance of "{r["label"]}" and give each one\'s center. Respond with JSON only.' if arm == 'cf' else
                 f'List every instance of "{r["label"]}" with its center. Respond with JSON only.')
            reqs.append(dict(request_id=f'fsc|{r["item"]}|{m}|{arm}|c0', model_key=m, setting='min', images=[p], system=system(arm, mode, *sent.size, r['label']),
                             user=u, meta=dict(task='fsc', item=r['item'], arm=arm, crop=None, mode=mode, w=r['W'], h=r['H'], sent=list(sent.size))))
        for ci, c in enumerate(CROPS):
            cr = im.crop((round(c[0] * r['W']), round(c[1] * r['H']), round(c[2] * r['W']), round(c[3] * r['H'])))
            cr = cr.resize((cr.width * 2, cr.height * 2), Image.LANCZOS); cs = C.prepare(cr, m)
            p = f'data/fsc/{r["item"].split(".")[0]}_c{ci}_{tag}.png'
            if not os.path.exists(f'{R}/{p}'): cs.save(f'{R}/{p}')
            reqs.append(dict(request_id=f'fsc|{r["item"]}|{m}|tl|c{ci}', model_key=m, setting='min', images=[p], system=system('tl', mode, *cs.size, r['label']),
                             user=f'List every instance of "{r["label"]}" in this crop with its center. Respond with JSON only.',
                             meta=dict(task='fsc', item=r['item'], arm='tl', crop=list(c), mode=mode, w=r['W'], h=r['H'], sent=list(cs.size))))
sub = 'b1ppilot' if A.pilot else 'b1p'
C.write_manifest(f'{R}/runs/{sub}/requests_main.jsonl', reqs)
json.dump({r['item']: r for r in pick}, open(f'{R}/runs/{sub}/items.json', 'w'))
print(sub, len(reqs), [len([r for r in pick if r['stratum'] == s]) for s in range(3)], [round(np.median([r['rel_size'] for r in st]), 4) for st in strata])
