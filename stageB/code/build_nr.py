"""EXP-NR: same-response readout on natural multi-class images. No model calls here.
Question: does the gap between the list count and the stated count, seen on dense synthetic scenes, appear on photographs and
documents with several classes, and does the list-first protocol of Section 5 (list first, counts after, row-by-row scan,
allowed class names, declared alias map, degeneracy fallback) lower per-type count error there?
Images (selected with the annotations for the property under study: several classes and many objects; disjoint from E3):
  DOTA v1.0 val: per image, the 1024x1024 window (stride 512) with the most classes among objects at least half inside and not
                 marked difficult; kept if >= 3 classes and 30 to 300 objects (all 30 such images; E3 source images excluded).
  DocLayNet v1.2 test shards 0 and 3: pages with >= 4 classes and >= 20 elements, at most 2 pages per document, 50 pages,
                 seeded (E3 pages excluded).
Arms per image and model (setting 'min', cap 16,384): cf = counts first, then the list (two-sided wording);
  lf = list first, counts for the listed objects after, row-by-row scan, allowed class names. Gemini 3.8 also lf at its
  default setting with the same cap (default16k), the shipped configuration.
Records are class + centre point. Output: runs/nr/requests_main.jsonl, runs/nr/items.json, data/nr/<ds>/*.png
Usage: python -I build_nr.py"""
import hashlib, io, json, os, random, sys
from collections import defaultdict
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bcommon as C
R = C.ROOT; RAW = '/home/claude/e3data'; E3 = '/home/claude/fsr_v51/e3/data'
DOTA = ['plane', 'ship', 'storage-tank', 'baseball-diamond', 'tennis-court', 'basketball-court', 'ground-track-field', 'harbor',
        'bridge', 'large-vehicle', 'small-vehicle', 'helicopter', 'roundabout', 'soccer-ball-field', 'swimming-pool']
DLN = ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title']
DESC = {'dota': ('aerial image', 'object', 'objects'), 'doclaynet': ('document page', 'layout element', 'layout elements')}


def seed(s): return int(hashlib.sha256(s.encode()).hexdigest()[:12], 16)


def dota_items():
    e3 = {json.loads(l)['meta']['source_image'][:-4] for l in open(f'{E3}/dota/items.jsonl')}
    lab = f'{RAW}/dota/DOTAv1/labels/val_original'; img = f'{RAW}/dota/DOTAv1/images/val'; out = []
    for f in sorted(os.listdir(lab)):
        name = f[:-4]
        if name in e3: continue
        p = os.path.join(img, name + '.jpg')
        if not os.path.exists(p): p = os.path.join(img, name + '.png')
        if not os.path.exists(p): continue
        W0, H0 = Image.open(p).size; ww, wh = min(1024, W0), min(1024, H0); objs = []
        for line in open(os.path.join(lab, f)):
            t = line.split()
            if len(t) < 10: continue
            xs = [float(v) for v in t[0:8:2]]; ys = [float(v) for v in t[1:8:2]]
            objs.append(([min(xs), min(ys), max(xs), max(ys)], t[8], int(t[9])))
        best = None
        for x0 in sorted(set(list(range(0, max(1, W0 - ww + 1), 512)) + [W0 - ww])):
            for y0 in sorted(set(list(range(0, max(1, H0 - wh + 1), 512)) + [H0 - wh])):
                tg, ig = [], []
                for b, cl, diff in objs:
                    area = max(1e-6, (b[2] - b[0]) * (b[3] - b[1])); cc = [max(b[0], x0), max(b[1], y0), min(b[2], x0 + ww), min(b[3], y0 + wh)]
                    if cc[2] <= cc[0] or cc[3] <= cc[1]: continue
                    cb = [round(cc[0] - x0, 1), round(cc[1] - y0, 1), round(cc[2] - x0, 1), round(cc[3] - y0, 1)]
                    if diff or (cc[2] - cc[0]) * (cc[3] - cc[1]) / area < 0.5 or cl not in DOTA: ig.append(cb)
                    else: tg.append(dict(label=cl, box=cb))
                k = len({t['label'] for t in tg})
                if 30 <= len(tg) <= 300 and (best is None or (k, len(tg)) > (best[0], len(best[3]))): best = (k, x0, y0, tg, ig)
        if best and best[0] >= 3:
            k, x0, y0, tg, ig = best; d = f'{R}/data/nr/dota'; os.makedirs(d, exist_ok=True)
            Image.open(p).convert('RGB').crop((x0, y0, x0 + ww, y0 + wh)).save(f'{d}/{name}.png')
            out.append(dict(id=f'dota_{name}', dataset='dota', image=f'data/nr/dota/{name}.png', width=ww, height=wh, labels=DOTA,
                            targets=tg, ignore=ig, n=len(tg), n_classes=k, meta=dict(source_image=os.path.basename(p), window=[x0, y0, ww, wh])))
    return out


def dln_items():
    import pyarrow.parquet as pq
    e3 = {json.loads(l)['id'][4:] for l in open(f'{E3}/doclaynet/items.jsonl')}
    rows = []
    for shard in ('test-00000.parquet', 'test-00003.parquet'):
        pf = pq.ParquetFile(f'{RAW}/doclaynet/{shard}')
        for g in range(pf.num_row_groups):
            for i, r in enumerate(pf.read_row_group(g, columns=['bboxes', 'category_id', 'metadata']).to_pylist()):
                if r['metadata']['page_hash'][:16] in e3: continue
                if len(set(r['category_id'])) >= 4 and 20 <= len(r['bboxes']) <= 300:
                    rows.append(dict(shard=shard, g=g, i=i, bboxes=r['bboxes'], cats=r['category_id'], meta=r['metadata']))
    byd = defaultdict(list)
    for r in rows: byd[r['meta']['original_filename']].append(r)
    rng = random.Random(seed('nr-doclaynet')); pool = []
    for doc in sorted(byd):
        v = sorted(byd[doc], key=lambda r: r['meta']['page_hash']); rng.shuffle(v); pool += v[:2]
    rng.shuffle(pool); sel = pool[:50]; out = []; d = f'{R}/data/nr/doclaynet'; os.makedirs(d, exist_ok=True)
    for r in sel:
        t = pq.ParquetFile(f"{RAW}/doclaynet/{r['shard']}").read_row_group(r['g'], columns=['image'])
        im = Image.open(io.BytesIO(t.column('image')[r['i']].as_py()['bytes'])).convert('RGB'); W, H = im.size
        name = r['meta']['page_hash'][:16]; im.save(f'{d}/{name}.png')
        sx, sy = W / r['meta']['coco_width'], H / r['meta']['coco_height']
        tg = [dict(label=DLN[c - 1], box=[round(max(0, b[0]) * sx, 1), round(max(0, b[1]) * sy, 1), round(min(W, (b[0] + b[2]) * sx), 1), round(min(H, (b[1] + b[3]) * sy), 1)])
              for b, c in zip(r['bboxes'], r['cats'])]
        out.append(dict(id=f'dln_{name}', dataset='doclaynet', image=f'data/nr/doclaynet/{name}.png', width=W, height=H, labels=DLN,
                        targets=tg, ignore=[], n=len(tg), n_classes=len({x['label'] for x in tg}),
                        meta=dict(doc_category=r['meta']['doc_category'], original_filename=r['meta']['original_filename'], page_no=r['meta']['page_no'])))
    return out


def system(ds, arm, mode, w, h, labels):
    kind, obj, objs = DESC[ds]
    ex = '{"type": "%s", "x": %d, "y": %d}' % (labels[0], 402 if mode == 'grid' else round(0.402 * w), 311 if mode == 'grid' else round(0.311 * h))
    counts = '"counts_by_type": {' + ', '.join(f'"{l}": {n}' for l, n in zip(labels, [3, 0, 2] + [0] * (len(labels) - 3))) + '}'
    lst = '"objects": [' + ex + ', ...]'
    fmt = '{' + (counts + ', ' + lst if arm == 'cf' else lst + ', ' + counts) + '}'
    names = ', '.join(labels)
    s = (f'You are a precise visual counting and localization assistant.\n\nGiven an {kind}, identify every {obj} of the types listed below.\n'
         + C.coord_sentence(mode, w, h) + ' Give the centre of each ' + obj + '.\n')
    if arm == 'lf':
        s += (f'List the {objs} in reading order: scan the image row by row from top to bottom, and within a row from left to right.\n'
              f'The "type" value must be exactly one of: {names}. Do not use any other type name.\n')
    s += ('\nRespond with ONLY a JSON object (no markdown, no commentary) in this format:\n' + fmt + '\n'
          '"counts_by_type" has one entry for every type, 0 if there is none.\n\n'
          f'Types to recognize: {names}.\nLocate every {obj} carefully. Count systematically; do not miss any and do not invent {objs} that are not there.')
    if arm == 'lf': s += f'\nWrite the full objects list first; after it, give counts_by_type for the {objs} you listed.'
    return s


def user(ds, arm):
    kind, obj, objs = DESC[ds]
    if arm == 'cf': return f'Count every {obj} of the listed types in this {kind}. Output counts by type and each {obj}\'s centre. Respond with JSON only.'
    return f'List every {obj} of the listed types in this {kind} with its centre, then counts by type. Respond with JSON only.'


if __name__ == '__main__':
    items = dota_items() + dln_items()
    os.makedirs(f'{R}/runs/nr', exist_ok=True)
    json.dump(items, open(f'{R}/runs/nr/items.json', 'w'))
    reqs = []
    for it in items:
        for m in C.MODELS:
            mode = C.MODE[m]; im = Image.open(os.path.join(R, it['image'])); sent = C.prepare(im, m).size
            assert sent == im.size, 'image would be resized'
            arms = [('cf', 'min'), ('lf', 'min')] + ([('lf', 'default16k')] if m == 'gem38' else [])
            for arm, st in arms:
                reqs.append(dict(request_id=f"nr|{it['id']}|{m}|{arm}|{st}", model_key=m, setting=st, images=[it['image']],
                                 system=system(it['dataset'], arm, mode, it['width'], it['height'], it['labels']), user=user(it['dataset'], arm),
                                 meta=dict(task='nr', item=it['id'], dataset=it['dataset'], arm=arm, setting=st, mode=mode, w=it['width'], h=it['height'],
                                           sent=list(sent), n=it['n'])))
    C.write_manifest(f'{R}/runs/nr/requests_main.jsonl', reqs)
    from collections import Counter
    print(len(items), Counter(i['dataset'] for i in items), len(reqs), Counter(q['model_key'] for q in reqs))
