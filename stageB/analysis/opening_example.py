"""Opening example and how common it is (Figure 1, Section 1). No model calls.
For the list-first responses of the readout-confirmation run (SCP-dense, 50 images, lowest setting): a list is *exact* when it has one
record per target, every record has a valid class after the alias map, and a one-to-one matching pairs every record with a target of the
same class within 20 px. For exact lists we check whether the per-type counts stated in the same response equal the true counts.
Output: out/opening_example.json"""
import json, sys, collections
from pathlib import Path
sys.path.insert(0, '/home/claude/stageB/code')
import bcommon as C
from degen import T5
OUT = Path(__file__).parent / 'out'
SYN = {'diamond': 'square', 'rhombus': 'square', 'rotated square': 'square'}
Q = C.load_requests('rc16', 'main'); R = C.load_responses('rc16', 'main'); res = {'per_image': {}, 'summary': {}}
for m in ('gpt61', 'sonnet55'):
    exact = wrong = 0
    for rid, q in sorted(Q.items()):
        if q['model_key'] != m: continue
        r = R[rid]; mt = q['meta']; i = mt['item']
        recs, dec, _ = C.decode_shapes(r['raw_text'], mt['mode'], 1024, 1024, 1024, 1024)
        recs = [dict(x, type=SYN.get(x['type'], x['type'])) for x in recs]; typed = [x for x in recs if x['type'] in T5]
        G = json.load(open(f'{C.ROOT}/data/scpd_gen/ground_truth/gt_{i:03d}.json'))['shapes']
        tp = len(C.match_typed(typed, G, 20)); gc = collections.Counter(s['type'] for s in G)
        st = {t: int(dec.get(t, 0) or 0) for t in T5} if isinstance(dec, dict) else None
        is_exact = tp == len(G) == len(typed) == len(recs)
        st_wrong = st is not None and any(st[t] != gc[t] for t in T5)
        exact += is_exact; wrong += is_exact and st_wrong
        res['per_image'][f'{m}|{i}'] = dict(n=len(G), records=len(recs), valid=len(typed), matched=tp, stated_total=sum(st.values()) if st else None,
                                            stated_per_type_error=sum(abs(st[t] - gc[t]) for t in T5) if st else None, exact_list=is_exact)
    res['summary'][m] = dict(images=50, exact_lists=exact, exact_lists_with_wrong_stated_counts=wrong)
json.dump(res, open(OUT / 'opening_example.json', 'w'), indent=1)
print(res['summary'], res['per_image']['gpt61|28'], res['per_image']['gpt61|14'])
