"""Per-cell, per-candidate quality table (rep 0, 60 sources, primary metric): mean set error, mean per-image F1, FP, FN,
mean output tokens, cost per call, share of calls whose output reached 16k tokens. Writes runs/table_v2.json."""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import e3lib as L
import analyze_e3 as A0
from score_e3 import load_items, PRICE
items = load_items(); rows = A0.load_rows('runs_main/scored.jsonl')
extra = Path(__file__).parent / 'runs_ctrl16k/scored.jsonl'
if extra.exists(): rows += A0.load_rows('runs_ctrl16k/scored.jsonl')
acc = defaultdict(lambda: defaultdict(list))
for x in rows:
    if x['block'] not in ('MAIN', 'CTRL16K') or x['rep'] != 0: continue
    it = items[x['item']]; pm = A0.primary(it); E, F1 = x[f'E_{pm}'], x[f'F1_{pm}']          # stored scores (full-precision coordinates)
    tp = it['n'] if F1 >= 1 else F1 * E / (2 * (1 - F1))                                        # from E = cnt + n - 2TP and F1 = 2TP/(n + cnt)
    a = acc[(x['dataset'], x['model'], x['cand'])]
    a['E'].append(E); a['F1'].append(F1); a['FP'].append(E - it['n'] + tp); a['FN'].append(it['n'] - tp)
    a['out'].append(x['out_tok']); a['cost'].append((x['in_tok'] * PRICE[x['model']][0] + x['out_tok'] * PRICE[x['model']][1]) / 1e6); a['cap'].append(int(x['out_tok'] >= 16000))
out = {f'{k[0]}|{k[1]}|{k[2]}': {f: round(float(np.mean(v)), 4 if f in ('F1', 'cost', 'cap') else 2) for f, v in a.items()} | {'n_calls': len(a['E'])} for k, a in sorted(acc.items())}
(Path(__file__).parent / 'runs/table_v2.json').write_text(json.dumps(out, indent=1))
for k, v in out.items(): print(k, v)
