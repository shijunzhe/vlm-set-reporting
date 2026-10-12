"""Sample click-grounding items (no model calls). ScreenSpot-Pro: 200 items stratified by platform (windows 90, macos 80,
linux 30) and, within platform, by group in proportion; icon/text kept in proportion by sorting within strata.
ScreenSpot-v2: 30 desktop + 30 web items. Seeded; writes data/click_items.json. Run with python -I from /home/claude/gui."""
import json, glob, random, collections
rng = random.Random('gui-click-v1')
A = []
for f in sorted(glob.glob('raw/sspro/annotations/*.json')): A += json.load(open(f))
quota = {'windows': 90, 'macos': 80, 'linux': 30}
out = []
for plat, q in quota.items():
    P = [a for a in A if a['platform'] == plat]; g = collections.defaultdict(list)
    for a in P: g[(a['group'], a['ui_type'])].append(a)
    tot = len(P); alloc = {k: max(1, round(q * len(v) / tot)) for k, v in g.items()}
    while sum(alloc.values()) > q: alloc[max(alloc, key=alloc.get)] -= 1
    while sum(alloc.values()) < q: alloc[max(g, key=lambda k: len(g[k]) - alloc[k])] += 1
    for k, v in sorted(g.items()):
        for a in rng.sample(v, min(alloc[k], len(v))):
            x1, y1, x2, y2 = a['bbox']
            out.append(dict(id='sspro_' + a['id'], source='sspro', image='raw/sspro/images/' + a['img_filename'], instruction=a['instruction'],
                            bbox=[x1, y1, x2, y2], img_size=a['img_size'], platform=plat, group=a['group'], app=a['application'], ui_type=a['ui_type']))
for f, n, kind in (('screenspot_desktop_v2.json', 30, 'desktop'), ('screenspot_web_v2.json', 30, 'web')):
    D = json.load(open('raw/ssv2/' + f))
    for i, a in enumerate(rng.sample(D, n)):
        x, y, w, h = a['bbox']
        out.append(dict(id=f'ssv2_{kind}_{i}', source='ssv2', image='raw/ssv2/images/' + a['img_filename'], instruction=a['instruction'],
                        bbox=[x, y, x + w, y + h], img_size=None, platform=kind, group=a['data_source'], app=a['data_source'], ui_type=a['data_type']))
json.dump(out, open('data/click_items.json', 'w'), indent=1)
print(len(out), collections.Counter((o['source'], o['platform']) for o in out), collections.Counter(o['group'] for o in out if o['source'] == 'sspro'))
