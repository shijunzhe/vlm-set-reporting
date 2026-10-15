"""Stage Z2: two-stage zoom around each model's SECOND click (C2), the same request as B4 (which zooms around C1).
Adds the missing comparator for the click alarm after zoom: the same model's two zoomed points (B4 vs B4c2).
No model calls here. Usage: python3 -I code/build_zoom_c2.py   (writes runs/main/requests_Z2.jsonl)"""
import collections, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir('/home/claude/gui')
import build_gui as B
import fsrgui as G
SUB = 'main'
P, C = B.subset_items(SUB)
A = B.load_resp(SUB, 'A'); Areq = B.load_req(SUB, 'A'); R = []; miss = 0; nopt = 0
for m in B.MODELS:
    Pm = G.prompts(B.MODE[m])
    for c in C:
        rid = f"{c['id']}|{m}|C2"; r = A.get(rid)
        if r is None: miss += 1; continue
        p, _ = G.decode_click(r['raw_text'], B.scale_of(m, Areq[rid]['meta']['sizes'][0]))
        if p is None: nopt += 1; continue                         # no second click: nothing to zoom around
        w4 = G.window(p[0], p[1], 1 / 3)
        f4, s4 = B.view(m, c['image'], f"{SUB}/{m}/{c['id']}/zoom_c2", crop=w4)
        R.append(B.req(f"{c['id']}|{m}|B4c2", m, [f4], Pm['zoom'], G.size_note(B.MODE[m], [s4], ['crop']) + f'Platform: {c["platform"]}. Instruction: "{c["instruction"]}". Return the click point in this crop.',
                       [s4], item=c['id'], arm='B4c2', window=w4, click=p))
with open(f'runs/{SUB}/requests_Z2.jsonl', 'w') as f:
    for q in R: f.write(json.dumps(q) + '\n')
print('requests', len(R), 'missing C2', miss, 'no point', nopt, collections.Counter(q['model_key'] for q in R))
