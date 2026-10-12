"""B5: effective resolution. ScreenSpot-Pro, 100 items, the model's own second-stage zoom window, sent at two other sizes:
up = native crop upsampled 2x (then capped by the provider rule), down = crop shown at the full-view scale. Same prompt as
the existing zoom stage (native size already run). No model calls here. Usage: python -I build_b5.py [--pilot]"""
import json, os, sys, random, argparse, re
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bcommon as C
ap = argparse.ArgumentParser(); ap.add_argument('--pilot', action='store_true'); A = ap.parse_args()
R = C.ROOT; GUI = '/home/claude/gui'; rng = random.Random(55)
QB = {}
for l in open(f'{GUI}/runs/main/requests_B.jsonl'):
    q = json.loads(l)
    if q['meta']['arm'] == 'B4' and q['meta']['item'].startswith('sspro'): QB[(q['meta']['item'], q['model_key'])] = q
QA = {}
for l in open(f'{GUI}/runs/main/requests_A.jsonl'):
    q = json.loads(l)
    if q['meta'].get('arm') == 'click' and q['meta']['rep'] == 1: QA[(q['meta']['item'], q['model_key'])] = q
CI = {c['id']: c for c in json.load(open(f'{GUI}/data/click_items_prepped.json'))}
ids = sorted({k[0] for k in QB if all((k[0], m) in QB for m in C.MODELS)})
ids = rng.sample(ids, 100 if not A.pilot else 3)
reqs = []; os.makedirs(f'{R}/data/b5', exist_ok=True)
for iid in ids:
    c = CI[iid]; full = Image.open(f'{GUI}/{c["image"]}').convert('RGB'); W, H = full.size
    for m in C.MODELS:
        q = QB[(iid, m)]; w = q['meta']['window']; fs = QA[(iid, m)]['meta']['sizes'][0]
        nat = full.crop((round(w[0] * W), round(w[1] * H), round(w[2] * W), round(w[3] * H)))
        variants = {'up': C.prepare(nat.resize((nat.width * 2, nat.height * 2), Image.LANCZOS), m),
                    'down': nat.resize((max(1, round((w[2] - w[0]) * fs[0])), max(1, round((w[3] - w[1]) * fs[1]))), Image.LANCZOS)}
        for v, im in variants.items():
            p = f'data/b5/{iid}_{m}_{v}.png'; im.save(f'{R}/{p}')
            user = q['user']
            if q['meta']['mode'] == 'pixel': user = re.sub(r'crop \d+x\d+', f'crop {im.width}x{im.height}', user)
            reqs.append(dict(request_id=f'b5|{iid}|{m}|{v}', model_key=m, setting='min', images=[p], system=q['system'], user=user,
                             meta=dict(task='b5', item=iid, cond=v, window=w, mode=q['meta']['mode'], sent=[im.width, im.height],
                                       native_sent=q['meta']['sizes'][0], full_sent=fs, bbox=c['bbox_norm'], width=W, height=H)))
sub = 'b5pilot' if A.pilot else 'b5'
C.write_manifest(f'{R}/runs/{sub}/requests_main.jsonl', reqs); print(sub, len(reqs))
