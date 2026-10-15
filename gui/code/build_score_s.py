"""Stage S: score-only verification of each model's zoomed click point (the final point of two-stage zoom), by every model.
For each of the 260 click items of the alarm analysis, each candidate (the zoomed first-click point of GPT-6.1, Sonnet 5.5 and
Gemini 3.8) is shown to each judge model as a native crop (window 0.2 of the screenshot) with a marker at the candidate, the
instruction and the platform. The judge returns a verdict and a probability that the marked click hits the intended element; it
does not propose another point and is not told which model produced the candidate. Truth is used only in scoring.
No model calls here. Usage: python3 -I code/build_score_s.py   (writes runs/main/requests_S.jsonl)"""
import collections, json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir('/home/claude/gui')
import build_gui as B
import fsrgui as G
SUB = 'main'
SYSTEM = ('You are checking a click proposed by a computer-use agent. The image is a zoomed crop of a screenshot at native resolution; '
          'the red circle with a cross marks the proposed click. Judge whether clicking at the center of the marker would hit the UI element '
          'the instruction refers to. Do not propose another point.\n\n'
          'Output contract: one JSON object {"decision": "correct" or "uncertain" or "wrong", "p_hit": a number from 0 to 1}, where p_hit is your '
          'probability that clicking at the center of the marker hits the intended element. Give your best estimate even when unsure. '
          'No prose, no code fence.')
P, C = B.subset_items(SUB); CI = {c['id']: c for c in C}
A = B.load_resp(SUB, 'A'); Areq = B.load_req(SUB, 'A'); Bz = B.load_resp(SUB, 'B'); Breq = B.load_req(SUB, 'B')
def zoomed(cid, m):
    """final point of two-stage zoom (as in alarm_curves.zpt): zoomed point if found, else the first click"""
    r1 = A.get(f'{cid}|{m}|C1')
    if r1 is None: return None
    p1, _ = G.decode_click(r1['raw_text'], B.scale_of(m, Areq[f'{cid}|{m}|C1']['meta']['sizes'][0]))
    k = f'{cid}|{m}|B4'; r4 = Bz.get(k)
    if r4 is None or p1 is None: return p1
    q4 = Breq[k]; p, _ = G.decode_click(r4['raw_text'], B.scale_of(m, q4['meta']['sizes'][0])); w = q4['meta']['window']
    return [w[0] + p[0] * (w[2] - w[0]), w[1] + p[1] * (w[3] - w[1])] if p is not None else p1
items = [c['id'] for c in C if all(f"{c['id']}|{m}|{s}" in A for m in B.MODELS for s in ('C1', 'C2'))]
R = []; nopt = 0
for cid in items:
    c = CI[cid]
    for k in B.MODELS:
        p = zoomed(cid, k)
        if p is None: nopt += 1; continue
        w = G.window(p[0], p[1], 0.2)
        for j in B.MODELS:
            f, s = B.view(j, c['image'], f"{SUB}/{j}/{cid}/score_{k}", crop=w, marker=p)
            R.append(B.req(f"{cid}|{j}|S|{k}", j, [f], SYSTEM, G.size_note(B.MODE[j], [s], ['crop']) + f'Platform: {c["platform"]}. Instruction: "{c["instruction"]}". Is the marked click correct?',
                           [s], item=cid, arm='S', judge=j, candidate_model=k, window=w, click=p))
with open(f'runs/{SUB}/requests_S.jsonl', 'w') as f:
    for q in R: f.write(json.dumps(q) + '\n')
print('items', len(items), 'requests', len(R), 'no point', nopt, collections.Counter(q['model_key'] for q in R))
