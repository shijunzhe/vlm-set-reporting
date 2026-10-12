"""B2: draft conditioning (copy weight omega), composition cost and cross-view confirmation. No model calls here.
GroundCUA 30 screens + SKU-110K 30 images x 3 models x 4 native crops (6% overlap) x 2 arms:
  blind : list every element whose box lies in the crop (separate call per crop, original pixels)
  draft : same crop and pixels plus a draft list (the model's own full-view first draft restricted to the crop) with random
          edits: 2 true records deleted, 2 false boxes injected (no overlap with any true box), 1 true record shifted by
          0.6 of its width; the model returns the complete corrected list.
Truth is used here only to choose which draft records to delete / shift and where to inject (a manipulation), and later in
scoring. Usage: python -I build_b2.py [--pilot]"""
import json, os, sys, random, argparse
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bcommon as C
ap = argparse.ArgumentParser(); ap.add_argument('--pilot', action='store_true'); A = ap.parse_args()
R = C.ROOT; rng = random.Random(22)
sys.path.insert(0, '/home/claude/gui/code'); import fsrgui as G
GUI = '/home/claude/gui'; CROPS = G.CROPS
P = {i['id']: i for i in json.load(open(f'{GUI}/data/parse_items.json'))}
QA, RA = {}, {}
for l in open(f'{GUI}/runs/main/requests_A.jsonl'): q = json.loads(l); QA[q['request_id']] = q
for m in C.MODELS:
    for l in open(f'{GUI}/runs/main/responses_A_{m}.jsonl'):
        r = json.loads(l)
        if r['status'] == 'completed': RA[r['request_id']] = r
def sc(q): return [None if q['meta']['mode'] == 'grid' else tuple(s) for s in q['meta']['sizes']]
E3 = '/home/claude/fsr_v51/e3'; sys.path.insert(0, E3)
import e3lib as L, analyze_e3 as A0
from score_e3 import load_items
items3 = load_items(); rows3 = A0.load_rows('runs_main/scored.jsonl')
d3 = {(x['item'], x['model'], x['rep']): x for x in rows3 if x['block'] in ('MAIN', 'AGG') and x['cand'] == 'L-min' and x['dataset'] == 'sku110k'}


def enc(b, mode, sw, sh):  # b normalized to crop
    if mode == 'grid': return dict(x1=round(b[0] * 1000), y1=round(b[1] * 1000), x2=round(b[2] * 1000), y2=round(b[3] * 1000))
    return dict(x1=round(b[0] * sw), y1=round(b[1] * sh), x2=round(b[2] * sw), y2=round(b[3] * sh))


def to_crop(b, c):
    w, h = c[2] - c[0], c[3] - c[1]; return [(b[0] - c[0]) / w, (b[1] - c[1]) / h, (b[2] - c[0]) / w, (b[3] - c[1]) / h]


def inside(b, c): cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2; return c[0] <= cx < c[2] and c[1] <= cy < c[3]


def system(kind, arm, mode, sw, sh):
    what = ('every visible UI element (buttons, icons, menu items, tabs, links, input fields, checkboxes, labels, list rows)' if kind == 'gui'
            else 'every retail product on the shelves (one box per product facing)')
    coord = ('Give boxes as integers on a 0-1000 grid of THIS crop: x1, y1 top-left and x2, y2 bottom-right, (0,0) = top-left of the crop.' if mode == 'grid'
             else f'The crop is {sw}x{sh} pixels. Give boxes as integer pixel coordinates in this crop: x1, y1 top-left and x2, y2 bottom-right.')
    s = (f'The image is one crop of a larger {"screenshot" if kind == "gui" else "shelf photo"}. Find {what} whose box center lies in this crop.\n'
         + coord + '\nRespond with ONLY a JSON object (no markdown, no commentary): {"elements": [{"x1": .., "y1": .., "x2": .., "y2": ..}, ...]}\n'
         'Do not miss any and do not include anything that is not such an element.')
    if arm == 'draft':
        s += ('\nA draft list from an earlier pass is given in the user message, in the same coordinates. The draft may contain mistakes '
              'and may miss elements. Return the complete, corrected list of all elements in this crop (not only changes).')
    return s


reqs = []; meta = {}
gui_ids = []
for fam in ('libreoffice', 'onlyoffice', 'ide'):
    ids = sorted(i for i, it in P.items() if it['source'] == 'groundcua' and it['group'] == fam)
    gui_ids += rng.sample(ids, 10 if not A.pilot else 1)
sku_ids = sorted({k[0] for k in d3 if all((k[0], m, 0) in d3 for m in C.MODELS)})
sku_ids = rng.sample(sku_ids, 30 if not A.pilot else 2)
os.makedirs(f'{R}/data/b2', exist_ok=True)
for kind, ids in (('gui', gui_ids), ('sku', sku_ids)):
    for iid in ids:
        if kind == 'gui':
            it = P[iid]; img = Image.open(f'{GUI}/{it["src_image"]}').convert('RGB'); gt = it['targets']
        else:
            it = items3[iid]; img = Image.open(f'{E3}/{it["image_path"]}').convert('RGB'); W, H = img.size
            gt = [[t['box'][0] / W, t['box'][1] / H, t['box'][2] / W, t['box'][3] / H] for t in it['targets']]
        W, H = img.size
        for m in C.MODELS:
            if kind == 'gui':
                k = f'{iid}|{m}|D1'
                if k not in RA: continue
                draft = [b for b, n in G.decode_list(RA[k]['raw_text'], sc(QA[k])[0])[0]]
            else:
                x = d3.get((iid, m, 0))
                if x is None: continue
                draft = [[r['box'][0] / W, r['box'][1] / H, r['box'][2] / W, r['box'][3] / H] for r in A0.recs_of(x) if r['box']]
            tpd = {a for a, b in G.lex_match(draft, gt, 0.5)}
            for ci, c in enumerate(CROPS):
                crop = G.crop_native(img, c); sent = C.prepare(crop, m); sw, sh = sent.size
                p = f'data/b2/{kind}_{iid}_{"a" if m == "sonnet55" else "g"}_c{ci}.png'
                if not os.path.exists(f'{R}/{p}'): sent.save(f'{R}/{p}')
                inc = [j for j, b in enumerate(draft) if inside(b, c)]
                tp_in = [j for j in inc if j in tpd]; rr = random.Random(hash((iid, m, ci)) % 10 ** 6)
                dele = rr.sample(tp_in, min(2, len(tp_in))); rest_tp = [j for j in tp_in if j not in dele]
                shf = rr.sample(rest_tp, 1) if rest_tp else []
                kept = [to_crop(draft[j], c) for j in inc if j not in dele]
                if shf:
                    j = shf[0]; b = to_crop(draft[j], c); w = b[2] - b[0]; d = 0.6 * w if b[2] + 0.6 * w <= 1 else -0.6 * w
                    sb = [b[0] + d, b[1], b[2] + d, b[3]]; kept[[to_crop(draft[q], c) for q in inc if q not in dele].index(b)] = sb
                # inject 2 false boxes of typical size away from every true box
                gtc = [to_crop(g, c) for g in gt]; sizes = [(b[2] - b[0], b[3] - b[1]) for b in kept] or [(0.05, 0.03)]
                inj = []
                for _ in range(400):
                    if len(inj) == 2: break
                    w_, h_ = sizes[rr.randrange(len(sizes))]; x0 = rr.uniform(0, max(0.001, 1 - w_)); y0 = rr.uniform(0, max(0.001, 1 - h_))
                    b = [x0, y0, x0 + w_, y0 + h_]
                    if all(G.iou(b, g) == 0 for g in gtc) and all(G.iou(b, q) == 0 for q in inj): inj.append(b)
                draft_c = kept + inj; rr.shuffle(draft_c)
                key = f'{kind}|{iid}|{m}|c{ci}'
                meta[key] = dict(crop=list(c), deleted=[draft[j] for j in dele], shifted=([draft[shf[0]]] if shf else []), shift_frac=0.6,
                                 injected=[[c[0] + b[0] * (c[2] - c[0]), c[1] + b[1] * (c[3] - c[1]), c[0] + b[2] * (c[2] - c[0]), c[1] + b[3] * (c[3] - c[1])] for b in inj],
                                 n_draft=len(draft_c), mode=C.MODE[m], sent=[sw, sh])
                for arm in ('blind', 'draft'):
                    u = 'List every element in this crop. Respond with JSON only.'
                    if arm == 'draft': u = 'Draft list: ' + json.dumps({'elements': [enc(b, C.MODE[m], sw, sh) for b in draft_c]}) + '\nReturn the complete corrected list. Respond with JSON only.'
                    reqs.append(dict(request_id=f'{key}|{arm}', model_key=m, setting='min', images=[p], system=system(kind, arm, C.MODE[m], sw, sh), user=u,
                                     meta=dict(task='b2', kind=kind, item=iid, crop=ci, arm=arm, mode=C.MODE[m], sent=[sw, sh])))
sub = 'b2pilot' if A.pilot else 'b2'
C.write_manifest(f'{R}/runs/{sub}/requests_main.jsonl', reqs); json.dump(dict(meta=meta, gui_ids=gui_ids, sku_ids=sku_ids), open(f'{R}/runs/{sub}/items.json', 'w'))
from collections import Counter
print(sub, len(reqs), Counter((q['meta']['kind'], q['model_key']) for q in reqs))
