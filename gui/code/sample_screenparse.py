"""Sample 60 ScreenParse v2 screens from shard 000000 (no model calls), 20 per viewport group: 1440x900, 1920x1080, and
smaller viewports (1366x768, 1280x800, 1024x768, 768x1024). Keep screens with 8 to 60 annotated elements. Writes PNGs and
data/parse_screenparse.json (boxes converted to [x1,y1,x2,y2]). Run with python -I from /home/claude/gui."""
import json, random, io, os
import pyarrow.parquet as pq
from PIL import Image
rng = random.Random('gui-parse-sp-v1')
t = pq.ParquetFile('raw/screenparse/000000.parquet').read().to_pydict()
grp = lambda w, h: '1440' if (w, h) == (1440, 900) else ('1920' if (w, h) == (1920, 1080) else 'small')
idx = {g: [i for i in range(len(t['id'])) if grp(t['width'][i], t['height'][i]) == g and 8 <= t['num_elements'][i] <= 60] for g in ('1440', '1920', 'small')}
os.makedirs('data/img/screenparse', exist_ok=True); out = []
for g, L in idx.items():
    for i in rng.sample(L, 20):
        im = Image.open(io.BytesIO(t['image'][i]['bytes'])).convert('RGB'); p = f"data/img/screenparse/{t['id'][i]}.png"; im.save(p)
        T = [dict(box=[x, y, x + w, y + h], label=l, text=(tx or '')[:80], interactable=bool(it))
             for (x, y, w, h), l, tx, it in zip(t['bboxes'][i], t['labels'][i], t['texts'][i], t['interactable'][i])]
        out.append(dict(id='sp_' + t['id'][i], source='screenparse', image=p, width=im.width, height=im.height, viewport=[t['width'][i], t['height'][i]],
                        group=g, url=t['url'][i], targets=T, n=len(T)))
json.dump(out, open('data/parse_screenparse.json', 'w'), indent=1)
import collections; print(len(out), collections.Counter(o['group'] for o in out), collections.Counter((o['width'], o['height']) for o in out).most_common(6))
print('n', sorted(o['n'] for o in out))
