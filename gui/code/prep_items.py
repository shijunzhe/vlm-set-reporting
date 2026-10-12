"""Build parse items (GroundCUA 20 per family + ScreenParse 60) with normalized ground-truth boxes, and render model-facing
views: full view (long side <= 2048) and four native-resolution 2x2 crops. Click items get a full view. Run with python -I
from /home/claude/gui; writes data/parse_items.json, data/click_items_prepped.json and data/views/."""
import json, os, sys, random, collections
sys.path.insert(0, 'code'); import fsrgui as G
rng = random.Random('gui-prep-v1')
P = json.load(open('raw/groundcua/picked.json')); items = []
by = collections.defaultdict(list)
for p in P: by[p['family']].append(p)
for fam, L in sorted(by.items()):
    for p in L[:20]:
        im = G.load(f"raw/groundcua/images/{p['hash']}.png"); W, H = im.size
        T = [x['bbox'] for x in json.load(open(f"raw/groundcua/data/{p['hash']}.json"))]
        T = [[max(0, b[0]) / W, max(0, b[1]) / H, min(W, b[2]) / W, min(H, b[3]) / H] for b in T]
        T = [b for b in T if b[2] > b[0] and b[3] > b[1]]
        items.append(dict(id='gc_' + p['hash'][:12], source='groundcua', src_image=f"raw/groundcua/images/{p['hash']}.png", width=W, height=H,
                          group=fam, app=p['app'], targets=T, n=len(T)))
for s in json.load(open('data/parse_screenparse.json')):
    W, H = s['width'], s['height']
    T = [[max(0, t['box'][0]) / W, max(0, t['box'][1]) / H, min(W, t['box'][2]) / W, min(H, t['box'][3]) / H] for t in s['targets']]
    T = [b for b in T if b[2] > b[0] and b[3] > b[1]]
    items.append(dict(id=s['id'], source='screenparse', src_image=s['image'], width=W, height=H, group='vp' + s['group'], app=s['url'], targets=T, n=len(T)))
for it in items:
    im = G.load(it['src_image'])
    it['view'] = G.save(G.fit(im), f"data/views/{it['id']}/full.png")
    it['crops'] = [G.save(G.crop_norm(im, c), f"data/views/{it['id']}/crop{k + 1}.png") for k, c in enumerate(G.CROPS)]
json.dump(items, open('data/parse_items.json', 'w'), indent=1)
C = json.load(open('data/click_items.json'))
for c in C:
    im = G.load(c['image']); W, H = im.size; c['width'], c['height'] = W, H
    b = c['bbox']; c['bbox_norm'] = [b[0] / W, b[1] / H, b[2] / W, b[3] / H]
    c['view'] = G.save(G.fit(im), f"data/views/{c['id']}/full.png")
json.dump(C, open('data/click_items_prepped.json', 'w'), indent=1)
print(len(items), collections.Counter(i['group'] for i in items), len(C))
