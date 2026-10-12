"""E3 data preparation (no model calls). Four datasets, 60 sources each, 20 per count tertile (seeded), n <= 300.
Unified item format (data/<ds>/items.jsonl):
  id, dataset, image_path, width, height, task ('point' | 'box'), labels (allowed label set), targets [{label, point | box}],
  ignore [box] (regions where unmatched predictions are not counted), n, stratum, tau_point (uniform per image),
  reference_path (FSC only: crop of the first exemplar box), meta
Sources
  FSC-147     test split, images not used in earlier work (e3data/fsc), 384-px-high images as released; points
  SKU-110K    test split; images resized to max side 1600; boxes, single label
  DOTA v1.0   official val split (test labels are not public), original labels with difficult flag; one 1024x1024 window
              per image at a seeded position independent of the labels; objects with < 50% of their hull area inside the
              window, and difficult objects, become ignore regions; hull boxes + 15 class labels
  DocLayNet   v1.2 test shards 0 and 3, one page per document; 1025x1025 page images; boxes + 11 class labels
tau_point = 0.5 x (median short side of the target boxes) for box tasks, 0.5 x (mean short side of the 3 exemplars) for FSC.
"""
import csv, hashlib, io, json, os, random, statistics, sys
from collections import defaultdict
from pathlib import Path
from PIL import Image

OUT = Path('/home/claude/fsr_v51/e3/data'); RAW = Path('/home/claude/e3data')
CAP, PER = 300, 20
DOTA_CLASSES = ['plane', 'ship', 'storage-tank', 'baseball-diamond', 'tennis-court', 'basketball-court', 'ground-track-field', 'harbor',
                'bridge', 'large-vehicle', 'small-vehicle', 'helicopter', 'roundabout', 'soccer-ball-field', 'swimming-pool']
DLN_CLASSES = ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title']


def seed(s): return int(hashlib.sha256(s.encode()).hexdigest()[:12], 16)


def stratified(pool, key, tag):
    """pool: list of candidates with count key(x) in [1, CAP]; returns 60 items, 20 per count tertile, seeded."""
    pool = sorted(pool, key=key); t1, t2 = len(pool) // 3, 2 * len(pool) // 3
    rng = random.Random(seed(tag)); out = []
    for k, st in enumerate([pool[:t1], pool[t1:t2], pool[t2:]]):
        st = st[:]; rng.shuffle(st); out += [(k, x) for x in st[:PER]]
    return out, [key(pool[t1]), key(pool[t2])]


def short_side_tau(boxes):
    return round(0.5 * statistics.median(min(b[2] - b[0], b[3] - b[1]) for b in boxes), 2) if boxes else 5.0


def write(ds, items, bounds, source):
    d = OUT / ds; (d / 'images').mkdir(parents=True, exist_ok=True)
    (d / 'items.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in items))
    (d / 'meta.json').write_text(json.dumps(dict(dataset=ds, n_items=len(items), tertile_bounds=bounds, source=source), indent=1))
    print(ds, len(items), 'tertiles', bounds, 'n range', min(x['n'] for x in items), max(x['n'] for x in items))


def prep_fsc():
    sel = json.load(open(RAW / 'fsc/selection.json'))['items']; ann = json.load(open('/home/claude/fsc/annotation_FSC147_384.json'))
    d = OUT / 'fsc147'; (d / 'images').mkdir(parents=True, exist_ok=True); items = []
    for s in sel:
        f = s['file']; a = ann[f]; im = Image.open(RAW / 'fsc/images' / f).convert('RGB'); W, H = im.size
        ex = [[min(p[0] for p in b), min(p[1] for p in b), max(p[0] for p in b), max(p[1] for p in b)] for b in a['box_examples_coordinates']]
        x0, y0, x1, y1 = [int(round(v)) for v in ex[0]]
        im.save(d / 'images' / f.replace('.jpg', '.png')); im.crop((x0, y0, x1, y1)).save(d / 'images' / f.replace('.jpg', '_ref.png'))
        tau = round(max(3.0, 0.5 * statistics.mean(min(b[2] - b[0], b[3] - b[1]) for b in ex)), 2)
        items.append(dict(id='fsc_' + f[:-4], dataset='fsc147', image_path=f'data/fsc147/images/{f[:-4]}.png', width=W, height=H, task='point',
                          labels=[s['cls'] or 'object'], targets=[dict(label=s['cls'] or 'object', point=[float(p[0]), float(p[1])]) for p in a['points']],
                          ignore=[], n=len(a['points']), stratum=s['stratum'], tau_point=tau, reference_path=f'data/fsc147/images/{f[:-4]}_ref.png',
                          meta=dict(cls=s['cls'], exemplars=ex)))
    write('fsc147', items, json.load(open(RAW / 'fsc/selection.json'))['tertile_bounds'], 'FSC-147 test split via huggingface isentropic/FSC147')


def prep_sku():
    boxes = defaultdict(list); size = {}
    for r in csv.reader(open(RAW / 'sku/SKU110K_fixed/annotations/annotations_test.csv')):
        if len(r) < 8: continue
        boxes[r[0]].append([float(r[1]), float(r[2]), float(r[3]), float(r[4])]); size[r[0]] = (int(r[6]), int(r[7]))
    pool = [f for f in boxes if 1 <= len(boxes[f]) <= CAP and (RAW / 'sku/SKU110K_fixed/images' / f).exists()]
    sel, bounds = stratified(pool, lambda f: len(boxes[f]), 'e3-sku'); d = OUT / 'sku110k'; (d / 'images').mkdir(parents=True, exist_ok=True); items = []
    for k, f in sel:
        im = Image.open(RAW / 'sku/SKU110K_fixed/images' / f).convert('RGB'); W0, H0 = im.size; s = 1600 / max(W0, H0)
        im = im.resize((round(W0 * s), round(H0 * s)), Image.LANCZOS); W, H = im.size; im.save(d / 'images' / f.replace('.jpg', '.png'))
        bx = [[round(b[0] * s, 1), round(b[1] * s, 1), round(b[2] * s, 1), round(b[3] * s, 1)] for b in boxes[f]]
        items.append(dict(id='sku_' + f[:-4], dataset='sku110k', image_path=f'data/sku110k/images/{f[:-4]}.png', width=W, height=H, task='box',
                          labels=['product'], targets=[dict(label='product', box=b) for b in bx], ignore=[], n=len(bx), stratum=k,
                          tau_point=short_side_tau(bx), meta=dict(scale=s, original_size=[W0, H0])))
    write('sku110k', items, bounds, 'SKU-110K test split, official tarball (academic use)')


def prep_dota():
    lab = RAW / 'dota/DOTAv1/labels/val_original'; img = RAW / 'dota/DOTAv1/images/val'; cands = []
    for f in sorted(os.listdir(lab)):
        p = img / f.replace('.txt', '.jpg')
        if not p.exists(): p = img / f.replace('.txt', '.png')
        if not p.exists(): continue
        W0, H0 = Image.open(p).size; rng = random.Random(seed('e3-dota-window:' + f))
        ww, wh = min(1024, W0), min(1024, H0); x0 = rng.randint(0, W0 - ww); y0 = rng.randint(0, H0 - wh)
        tg, ig = [], []
        for line in open(lab / f):
            t = line.split()
            if len(t) < 10: continue
            xs = [float(v) for v in t[0:8:2]]; ys = [float(v) for v in t[1:8:2]]; cls_, diff = t[8], int(t[9])
            b = [min(xs), min(ys), max(xs), max(ys)]; area = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
            c = [max(b[0], x0), max(b[1], y0), min(b[2], x0 + ww), min(b[3], y0 + wh)]
            if c[2] <= c[0] or c[3] <= c[1]: continue
            frac = (c[2] - c[0]) * (c[3] - c[1]) / area; cb = [round(c[0] - x0, 1), round(c[1] - y0, 1), round(c[2] - x0, 1), round(c[3] - y0, 1)]
            if diff or frac < 0.5 or cls_ not in DOTA_CLASSES: ig.append(cb)
            else: tg.append(dict(label=cls_, box=cb))
        if 1 <= len(tg) <= CAP: cands.append(dict(f=f, p=str(p), win=[x0, y0, ww, wh], tg=tg, ig=ig))
    sel, bounds = stratified(cands, lambda c: len(c['tg']), 'e3-dota'); d = OUT / 'dota'; (d / 'images').mkdir(parents=True, exist_ok=True); items = []
    for k, c in sel:
        x0, y0, ww, wh = c['win']; name = c['f'][:-4]
        Image.open(c['p']).convert('RGB').crop((x0, y0, x0 + ww, y0 + wh)).save(d / 'images' / f'{name}.png')
        items.append(dict(id='dota_' + name, dataset='dota', image_path=f'data/dota/images/{name}.png', width=ww, height=wh, task='box',
                          labels=DOTA_CLASSES, targets=c['tg'], ignore=c['ig'], n=len(c['tg']), stratum=k, tau_point=short_side_tau([t['box'] for t in c['tg']]),
                          meta=dict(window=c['win'], source_image=Path(c['p']).name)))
    write('dota', items, bounds, 'DOTA v1.0 official val split, original labels (Ultralytics DOTAv1.zip), academic use')


def prep_doclaynet():
    import pyarrow.parquet as pq
    rows = []
    for shard in ('test-00000.parquet', 'test-00003.parquet'):
        pf = pq.ParquetFile(RAW / 'doclaynet' / shard)
        for g in range(pf.num_row_groups):
            t = pf.read_row_group(g, columns=['bboxes', 'category_id', 'metadata']).to_pylist()
            for i, r in enumerate(t): rows.append(dict(shard=shard, g=g, i=i, bboxes=r['bboxes'], cats=r['category_id'], meta=r['metadata']))
    byd = defaultdict(list)
    for r in rows:
        if 1 <= len(r['bboxes']) <= CAP: byd[r['meta']['original_filename']].append(r)
    pool = [random.Random(seed('e3-dln-page:' + k)).choice(v) for k, v in sorted(byd.items())]     # one page per document
    sel, bounds = stratified(pool, lambda r: len(r['bboxes']), 'e3-doclaynet'); d = OUT / 'doclaynet'; (d / 'images').mkdir(parents=True, exist_ok=True); items = []
    need = defaultdict(list)
    for k, r in sel: need[(r['shard'], r['g'])].append((k, r))
    for (shard, g), lst in need.items():
        t = pq.ParquetFile(RAW / 'doclaynet' / shard).read_row_group(g, columns=['image'])
        for k, r in lst:
            im = Image.open(io.BytesIO(t.column('image')[r['i']].as_py()['bytes'])).convert('RGB'); W, H = im.size
            name = f"{r['meta']['page_hash'][:16]}"; im.save(d / 'images' / f'{name}.png')
            sx, sy = W / r['meta']['coco_width'], H / r['meta']['coco_height']
            tg = [dict(label=DLN_CLASSES[c - 1], box=[round(max(0, b[0]) * sx, 1), round(max(0, b[1]) * sy, 1), round(min(W, (b[0] + b[2]) * sx), 1), round(min(H, (b[1] + b[3]) * sy), 1)])
                  for b, c in zip(r['bboxes'], r['cats'])]
            items.append(dict(id='dln_' + name, dataset='doclaynet', image_path=f'data/doclaynet/images/{name}.png', width=W, height=H, task='box',
                              labels=DLN_CLASSES, targets=tg, ignore=[], n=len(tg), stratum=k, tau_point=short_side_tau([t_['box'] for t_ in tg]),
                              meta=dict(doc_category=r['meta']['doc_category'], original_filename=r['meta']['original_filename'], page_no=r['meta']['page_no'])))
    items.sort(key=lambda x: (x['stratum'], x['id']))
    write('doclaynet', items, bounds, 'DocLayNet v1.2 test split (docling-project/DocLayNet-v1.2), CDLA-Permissive-1.0')


if __name__ == '__main__':
    for fn in sys.argv[1:] or ['fsc', 'sku', 'dota', 'doclaynet']: globals()['prep_' + fn]()
