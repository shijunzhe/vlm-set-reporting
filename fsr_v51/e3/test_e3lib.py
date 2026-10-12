"""Checks for e3lib: prompts and examples valid for every item and arm; perfect reports decode to E = 0 under every metric;
dropping / shifting records changes E as expected; the floor never exceeds E_a + E_b (theorem); ignore regions."""
import json, random, sys
sys.path.insert(0, '.')
import e3lib as L
items = [json.loads(l) for ds in ('fsc147', 'sku110k', 'dota', 'doclaynet') for l in open(f'data/{ds}/items.jsonl')]
res = dict(prompts=0, perfect_ok=0, drop_ok=0, floor_viol=0, floor_cases=0)
rng = random.Random(1)
def report(item, arm, recs):
    W, H = item['width'], item['height']
    def o(r):
        if item['task'] == 'point': return {'x': round(r['cx'] / W, 6), 'y': round(r['cy'] / H, 6)}
        b = r['box']; d = {'x1': round(b[0] / W, 6), 'y1': round(b[1] / H, 6), 'x2': round(b[2] / W, 6), 'y2': round(b[3] / H, 6)}
        if L.multi_label(item): d['label'] = r['label']
        return d
    if arm == 'L': return json.dumps({'status': 'found', 'objects': [o(r) for r in recs]})
    reg = L.registry(W, H); lines = []
    for g in reg:
        inside = [r for r in recs if g['x0'] <= r['cx'] < g['x0'] + g['width'] and g['y0'] <= r['cy'] < g['y0'] + g['height']]
        lines.append(json.dumps({'region_id': g['region_id'], 'status': 'found' if inside else 'empty', 'objects': [o(r) for r in inside]}))
    return '\n'.join(lines)
def truth_recs(item):
    out = []
    for t in item['targets']:
        if 'point' in t: out.append(dict(label=t['label'], cx=t['point'][0], cy=t['point'][1], box=None))
        else:
            b = [min(max(v, 0), lim) for v, lim in zip(t['box'], (item['width'], item['height'], item['width'], item['height']))]
            if b[2] - b[0] < 0.5 or b[3] - b[1] < 0.5: b = [b[0], b[1], b[0] + 1, b[1] + 1]
            out.append(dict(label=t['label'], cx=(b[0] + b[2]) / 2, cy=(b[1] + b[3]) / 2, box=b))
    return out
metrics = lambda item: ['point'] + ([('box', .5), ('box', .75)] if item['task'] == 'box' else [])
for item in items:
    for arm in ('L', 'R'):
        s, u, reg = L.build_prompt(item, arm); res['prompts'] += 1
        tr = truth_recs(item); recs, au = L.decode(item, arm, report(item, arm, tr))
        ok = all(L.error(item, recs, m)[0] == 0 for m in metrics(item)) and au['salvaged'] == 0 and au['invalid'] == 0
        res['perfect_ok'] += int(ok)
        if not ok: print('PERFECT FAIL', item['id'], arm, [L.error(item, recs, m) for m in metrics(item)], au, len(tr), len(recs))
        k = max(1, len(tr) // 5); dropped = tr[k:]
        r2, _ = L.decode(item, arm, report(item, arm, dropped))
        res['drop_ok'] += int(all(L.error(item, r2, m)[0] == k for m in metrics(item)))
    tr = truth_recs(item)
    for _ in range(3):
        def noisy():
            out = []
            for r in tr:
                if rng.random() < .2: continue
                dx, dy = rng.gauss(0, item['tau_point'] / 2), rng.gauss(0, item['tau_point'] / 2)
                if r['box'] is None: out.append(dict(r, cx=min(item['width'] - 1, max(0, r['cx'] + dx)), cy=min(item['height'] - 1, max(0, r['cy'] + dy))))
                else:
                    b = r['box']; nb = [b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy]
                    out.append(dict(r, box=nb, cx=(nb[0] + nb[2]) / 2, cy=(nb[1] + nb[3]) / 2))
            return out
        a, b = noisy(), noisy()
        it2 = dict(item, ignore=[])                      # theorem applies without ignore-region removal
        for m in metrics(item):
            res['floor_cases'] += 1
            res['floor_viol'] += int(L.disagreement(it2, a, b, m) > L.error(it2, a, m)[0] + L.error(it2, b, m)[0])
# ignore region: a spurious prediction inside an ignore box is not counted
d = next(i for i in items if i['ignore'])
tr = truth_recs(d); bx = d['ignore'][0]; extra = dict(label=d['labels'][0], cx=(bx[0] + bx[2]) / 2, cy=(bx[1] + bx[3]) / 2, box=bx)
res['ignore_ok'] = L.error(d, tr + [extra], ('box', .5))[0] == 0
print(res); json.dump(res, open('checks_e3lib.json', 'w'), indent=1)
ok = res['perfect_ok'] == res['prompts'] and res['drop_ok'] == res['prompts'] and res['floor_viol'] == 0 and res['ignore_ok']
print('ALL PASS' if ok else 'FAIL')
