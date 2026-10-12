"""E3 manifests (no model calls).
Per dataset x model (60 sources):
  MAIN   4 candidates (L-min, R-min, L-def, R-def); the screening pool (10 per count tertile, seeded) gets 2 calls per
         candidate, the other 30 sources 1 call                                   -> 4 x (30 x 2 + 30) = 360
  AGG    the fixed interface L-min is completed to 3 calls on every source        -> 30 + 60 = 90
  SPLIT  SKU-110K only: L-min restricted to each 2x2 quadrant, 1 call per quadrant -> 240 per model
Repairs are built from the MAIN responses afterwards (build_e3_repair.py).
SMOKE  3 sources per dataset (one per tertile), 4 candidates, 1 call; used only to check the pipeline.
"""
import hashlib, json, random, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import e3lib as L

ROOT = Path(__file__).parent
MODELS = ['gpt61', 'sonnet55', 'gem38']
DATASETS = ['fsc147', 'sku110k', 'dota', 'doclaynet']
CANDS = [('L', 'min'), ('R', 'min'), ('L', 'default'), ('R', 'default')]


def items(ds): return [json.loads(l) for l in (ROOT / f'data/{ds}/items.jsonl').read_text().splitlines() if l.strip()]


def pool(ds, its):
    rng = random.Random(int(hashlib.sha256(f'e3-screen:{ds}'.encode()).hexdigest()[:12], 16)); out = set()
    for k in range(3):
        st = sorted(i['id'] for i in its if i['stratum'] == k); rng.shuffle(st); out |= set(st[:10])
    return out


def req(item, m, arm, setting, rep, block, quadrant=None):
    if quadrant is None: sysm, user, reg = L.build_prompt(item, arm)
    else: sysm, user, reg = L.subset_prompt(item, quadrant)
    images = ([item['reference_path']] if item.get('reference_path') else []) + [item['image_path']]
    qtag = '' if quadrant is None else ':q' + '_'.join('%.2f' % v for v in quadrant)
    rid = f"E3:{block}:{item['id']}:{m}:{arm}-{setting}{qtag}:r{rep}"
    return dict(request_id=rid, block=block, dataset=item['dataset'], item_id=item['id'], model_key=m, arm=arm, setting=setting, cand=f'{arm}-{setting}',
                rep=rep, quadrant=quadrant, images=images, requested=[r['region_id'] for r in reg], system=sysm, user=user,
                prompt_sha256=hashlib.sha256((sysm + '\n---\n' + user).encode()).hexdigest())


def build():
    main, smoke = [], []
    for ds in DATASETS:
        its = items(ds); scr = pool(ds, its)
        smoke_ids = {sorted(i['id'] for i in its if i['stratum'] == k and i['id'] in scr)[0] for k in range(3)}
        for m in MODELS:
            for it in its:
                reps = (0, 1) if it['id'] in scr else (0,)
                for arm, st in CANDS:
                    for r in reps: main.append(req(it, m, arm, st, r, 'MAIN'))
                for r in ((2,) if it['id'] in scr else (1, 2)): main.append(req(it, m, 'L', 'min', r, 'AGG'))
                if ds == 'sku110k':
                    for q in ((0, 0, .5, .5), (.5, 0, 1, .5), (0, .5, .5, 1), (.5, .5, 1, 1)): main.append(req(it, m, 'L', 'min', 0, 'SPLIT', q))
                if it['id'] in smoke_ids:
                    for arm, st in CANDS: smoke.append(req(it, m, arm, st, 0, 'SMOKE'))
    assert len({q['request_id'] for q in main}) == len(main)
    rng = random.Random(20261011); rng.shuffle(main)
    (ROOT / 'runs').mkdir(exist_ok=True)
    for name, rows in (('requests_main.jsonl', main), ('requests_smoke.jsonl', smoke)):
        p = ROOT / 'runs' / name; p.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        print(name, len(rows), hashlib.sha256(p.read_bytes()).hexdigest()[:16], Counter(r['block'] for r in rows), Counter(r['model_key'] for r in rows))
    (ROOT / 'runs/screening_pool.json').write_text(json.dumps({ds: sorted(pool(ds, items(ds))) for ds in DATASETS}, indent=1))


if __name__ == '__main__':
    build()
