"""Sample the confirmation set (no model calls): 600 ScreenSpot-Pro items not used in the pilot (the 200 sspro items of
data/click_items.json), stratified by application file and, within application, by (group, ui_type) in proportion.
Seeded. Writes data/conf_items.json. Run with python -I from /home/claude/gui."""
import json, glob, random, collections, os
N = 600; rng = random.Random('gui-conf-20261013')
used = {c['id'] for c in json.load(open('data/click_items.json')) if c['source'] == 'sspro'}
A = []
for f in sorted(glob.glob('raw/sspro/annotations/*.json')):
    for a in json.load(open(f)): a['_file'] = os.path.basename(f)[:-5]; A.append(a)
pool = [a for a in A if 'sspro_' + a['id'] not in used]
assert len(pool) == len(A) - len(used), (len(pool), len(A), len(used))
by = collections.defaultdict(list)
for a in pool: by[a['_file']].append(a)
alloc = {k: round(N * len(v) / len(pool)) for k, v in by.items()}
while sum(alloc.values()) > N: alloc[max(alloc, key=lambda k: alloc[k] - N * len(by[k]) / len(pool))] -= 1
while sum(alloc.values()) < N: alloc[min(alloc, key=lambda k: alloc[k] - N * len(by[k]) / len(pool))] += 1
out = []
for f in sorted(by):
    g = collections.defaultdict(list)
    for a in by[f]: g[(a['group'], a['ui_type'])].append(a)
    q = alloc[f]; sub = {k: round(q * len(v) / len(by[f])) for k, v in g.items()}
    while sum(sub.values()) > q: sub[max(sub, key=lambda k: sub[k] - q * len(g[k]) / len(by[f]))] -= 1
    while sum(sub.values()) < q: sub[min(sub, key=lambda k: sub[k] - q * len(g[k]) / len(by[f]))] += 1
    for k in sorted(g):
        for a in rng.sample(sorted(g[k], key=lambda x: x['id']), min(sub[k], len(g[k]))):
            x1, y1, x2, y2 = a['bbox']
            out.append(dict(id='sspro_' + a['id'], source='sspro', image='raw/sspro/images/' + a['img_filename'], instruction=a['instruction'],
                            bbox=[x1, y1, x2, y2], img_size=a['img_size'], platform=a['platform'], group=a['group'], app=a['application'], ui_type=a['ui_type']))
assert len(out) == N and not ({o['id'] for o in out} & used) and len({o['image'] for o in out}) == N
json.dump(out, open('data/conf_items.json', 'w'), indent=1)
print(len(out), collections.Counter(o['platform'] for o in out), collections.Counter(o['ui_type'] for o in out))
print(sorted(collections.Counter(o['app'] for o in out).items()))
missing = [o['image'] for o in out if not os.path.exists(o['image'])]; print('images to download', len(missing))
