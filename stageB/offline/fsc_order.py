"""Why did the FSC order effect reverse between model generations? Same 100 FSC-147 images, same 384 px canvas.
Old: E4 (Sonnet 4.6, GPT-5.4, Gemini 2.5, Qwen3-VL, Gemma 4): counts_first vs list_first.
New: B1' (GPT-6.1, Sonnet 5.5, Gemini 3.8): count-first (cf) vs list-only (lo).
Per model: MAE of the written count under count-first (gist path), MAE of the list without a count before it
(enumeration path), each split into |bias| and spread of the signed error; point-level F1 of the enumeration list
(maximum matching within tau). The two-path model says the enumeration path wins when it is reliable; prediction:
across models the order effect (MAE_list - MAE_written) falls as enumeration F1 rises. No model calls.
Usage: python -I fsc_order.py"""
import json, sys, ast
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, '/home/claude/stageB/code')
import bcommon as C
items = json.load(open('/home/claude/stageB/runs/b1p/items.json'))
E4 = '/home/claude/x2/arr/results/e4/fsc147'
def f1(P, it):
    G = [dict(type='o', cx=x, cy=y) for x, y in it['points']]; P = [dict(type='o', cx=x, cy=y) for x, y in P]
    m = C.match_typed(P, G, float(it['tau']), typed=False); return 2 * len(m) / (len(P) + len(G)) if (P or G) else 1.0
rows = {}
def flag(P, k, written):
    """label-free degeneracy of the list (same rule as analyze_diag): length vs the written count of the count-first call
    of the same image, or > 10% repeated coordinates."""
    seen = set(); rep = 0
    for x, y in P:
        key = (round(x), round(y)); rep += key in seen; seen.add(key)
    return bool((written and (k > 1.5 * written or k < 0.5 * written)) or (P and rep / len(P) > 0.1))
for v in ('claude', 'gpt', 'gemini', 'qwen3vl', 'gemma4'):
    R = [json.loads(l) for l in open(f'{E4}/{v}_run0.jsonl')]
    by = {}
    for r in R:
        if r['item_id'] not in items: continue
        by.setdefault(r['item_id'], {})[r['order']] = r
    w, l, F, FL = [], [], [], []
    for iid, d in by.items():
        if 'counts_first' not in d or 'list_first' not in d: continue
        a, b = d['counts_first'], d['list_first']
        if a['status'] != 'success' or not isinstance(a.get('declared_count'), (int, float)) or b['status'] != 'success': continue
        n = int(items[iid]['n']); pts = b['points'] if isinstance(b['points'], list) else ast.literal_eval(b['points'])
        P = []
        for p in pts:
            try: P.append((float(p['x']), float(p['y'])) if isinstance(p, dict) else (float(p[0]), float(p[1])))
            except Exception: pass
        W, H = b['canvas']
        if P and (max(x for x, _ in P) > W * 1.05 or max(y for _, y in P) > H * 1.05):   # 0-1000 normalised output (seen for Qwen)
            P = [(x * W / 1000, y * H / 1000) for x, y in P]
        w.append(a['declared_count'] - n); l.append(int(b['list_count']) - n); F.append(f1(P, items[iid])); FL.append(flag(P, int(b['list_count']), a['declared_count']))
    rows['old|' + R[0]['model']] = (np.array(w, float), np.array(l, float), np.array(F), np.array(FL))
Q = C.load_requests('b1p', 'main'); RS = C.load_responses('b1p', 'main')
for m in C.MODELS:
    w, l, F, FL = [], [], [], []
    for iid, it in items.items():
        rc = RS.get(f'fsc|{iid}|{m}|cf|c0'); rl = RS.get(f'fsc|{iid}|{m}|lo|c0')
        if not rc or not rl or rc['status'] != 'completed' or rl['status'] != 'completed': continue
        qc = Q[f'fsc|{iid}|{m}|cf|c0']['meta']; ql = Q[f'fsc|{iid}|{m}|lo|c0']['meta']
        _, dec, _ = C.decode_shapes(rc['raw_text'], qc['mode'], it['W'], it['H'], *qc['sent'], list_key='points', count_key='count')
        recs, _, _ = C.decode_shapes(rl['raw_text'], ql['mode'], it['W'], it['H'], *ql['sent'], list_key='points', count_key='count')
        try: dec = int(dec)
        except Exception: continue
        n = int(it['n']); P = [(x['cx'], x['cy']) for x in recs]; w.append(dec - n); l.append(len(recs) - n); F.append(f1(P, it)); FL.append(flag(P, len(recs), dec))
    rows['new|' + m] = (np.array(w, float), np.array(l, float), np.array(F), np.array(FL))
out = {}
for k, (w, l, F, FL) in rows.items():
    ok = ~FL
    out[k] = dict(n=len(w), mae_written=round(float(np.mean(abs(w))), 2), mae_list=round(float(np.mean(abs(l))), 2),
                  order_effect_rel=round(float((np.mean(abs(l)) - np.mean(abs(w))) / np.mean(abs(w))) * 100, 1),
                  bias_written=round(float(np.mean(w)), 2), bias_list=round(float(np.mean(l)), 2), sd_written=round(float(np.std(w)), 2), sd_list=round(float(np.std(l)), 2),
                  enum_F1=round(float(np.mean(F)), 3), flagged_share=round(float(FL.mean()), 3),
                  enum_F1_unflagged=round(float(np.mean(F[ok])), 3) if ok.any() else None,
                  order_effect_unflagged=round(float((np.mean(abs(l[ok])) - np.mean(abs(w[ok]))) / np.mean(abs(w[ok]))) * 100, 1) if ok.sum() > 5 else None,
                  mae_diagnosed=round(float(np.mean(np.where(FL, abs(w), abs(l)))), 2))
ks = list(out); rho = spearmanr([out[k]['enum_F1'] for k in ks], [out[k]['order_effect_rel'] for k in ks])
out['_spearman_F1_vs_order_effect'] = dict(rho=round(float(rho.correlation), 3), p=round(float(rho.pvalue), 4), n=len(ks))
ks2 = [k for k in ks if not k.startswith('_') and out[k]['order_effect_unflagged'] is not None]
rho2 = spearmanr([out[k]['enum_F1_unflagged'] for k in ks2], [out[k]['order_effect_unflagged'] for k in ks2])
out['_spearman_F1_vs_order_effect_unflagged'] = dict(rho=round(float(rho2.correlation), 3), p=round(float(rho2.pvalue), 4), n=len(ks2))
ks3 = [k for k in ks2 if 'qwen' not in k]   # Qwen's E4 points are in no consistent frame, so its F1 is not measurable
rho3 = spearmanr([out[k]['enum_F1_unflagged'] for k in ks3], [out[k]['order_effect_unflagged'] for k in ks3])
out['_spearman_unflagged_without_qwen'] = dict(rho=round(float(rho3.correlation), 3), p=round(float(rho3.pvalue), 4), n=len(ks3))
json.dump(out, open('/home/claude/stageB/offline/fsc_order.json', 'w'), indent=1)
for k, v in out.items(): print(k, v)
