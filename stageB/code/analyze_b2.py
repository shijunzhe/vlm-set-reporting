"""B2 analysis: copy weight omega, separate-call crops vs full view vs one-call tiles, cross-view confirmation.
Usage: python -I analyze_b2.py <subset>"""
import json, os, sys, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bcommon as C
sys.path.insert(0, '/home/claude/gui/code'); import fsrgui as G
SUB = sys.argv[1] if len(sys.argv) > 1 else 'b2'
Q = C.load_requests(SUB, 'main'); RS = C.load_responses(SUB, 'main'); M = json.load(open(f'{C.ROOT}/runs/{SUB}/items.json')); meta = M['meta']
GUI = '/home/claude/gui'
P = {i['id']: i for i in json.load(open(f'{GUI}/data/parse_items.json'))}
QA, RA = {}, {}
for l in open(f'{GUI}/runs/main/requests_A.jsonl'): q = json.loads(l); QA[q['request_id']] = q
for m in C.MODELS:
    for l in open(f'{GUI}/runs/main/responses_A_{m}.jsonl'):
        r = json.loads(l)
        if r['status'] == 'completed': RA[r['request_id']] = r
def sc(q): return [None if q['meta']['mode'] == 'grid' else tuple(s) for s in q['meta']['sizes']]
E3 = '/home/claude/fsr_v51/e3'; sys.path.insert(0, E3)
import analyze_e3 as A0
from score_e3 import load_items
items3 = load_items(); rows3 = A0.load_rows('runs_main/scored.jsonl')
d3 = {(x['item'], x['model'], x['rep']): x for x in rows3 if x['block'] in ('MAIN', 'AGG') and x['cand'] == 'L-min' and x['dataset'] == 'sku110k'}


IRREG = collections.Counter(); TOTAL = collections.Counter()
def decode(raw, mode, sent, model=None):
    bx, irr = C.boxes_positional(raw); out = []
    if model: IRREG[model] += irr; TOTAL[model] += len(bx)
    for x1, y1, x2, y2 in bx:
        if mode == 'grid': b = [x1 / 1000, y1 / 1000, x2 / 1000, y2 / 1000]
        else: b = [x1 / sent[0], y1 / sent[1], x2 / sent[0], y2 / sent[1]]
        if b[2] > b[0] and b[3] > b[1]: out.append(b)
    return out


def to_full(b, c): w, h = c[2] - c[0], c[3] - c[1]; return [c[0] + b[0] * w, c[1] + b[1] * h, c[0] + b[2] * w, c[1] + b[3] * h]


def merge(parts):
    out = []
    for ci, P_ in parts:
        for b in P_:
            if not any(cj != ci and G.iou(b, q) >= 0.5 for cj, q in out): out.append((ci, b))
    return [b for _, b in out]


def gt_of(kind, iid):
    if kind == 'gui': return P[iid]['targets']
    it = items3[iid]; W, H = it['width'], it['height']
    return [[t['box'][0] / W, t['box'][1] / H, t['box'][2] / W, t['box'][3] / H] for t in it['targets']]


def drafts(kind, iid, m):
    if kind == 'gui':
        out = []
        for j in (1, 2):
            k = f'{iid}|{m}|D{j}'; out.append([b for b, n in G.decode_list(RA[k]['raw_text'], sc(QA[k])[0])[0]] if k in RA else None)
        k = f'{iid}|{m}|T1'
        t1 = [b for b, n in G.decode_tiles(RA[k]['raw_text'], sc(QA[k]) if sc(QA[k])[0] else None)[0]] if k in RA else None
        return out[0], out[1], t1
    it = items3[iid]; W, H = it['width'], it['height']
    def rep(r):
        x = d3.get((iid, m, r)); return [[b['box'][0] / W, b['box'][1] / H, b['box'][2] / W, b['box'][3] / H] for b in A0.recs_of(x) if b['box']] if x else None
    return rep(0), rep(1), None


spend = collections.Counter(); out = dict(cells={})
dec = {}
for rid, q in Q.items():
    r = RS.get(rid)
    if r is None or r['status'] != 'completed': continue
    spend[q['model_key']] += C.cost(q['model_key'], r); mt = q['meta']
    key = rid.rsplit('|', 1)[0]; c = meta[key]['crop']
    dec[rid] = [to_full(b, c) for b in decode(r['raw_text'], mt['mode'], mt['sent'], q['model_key'])]
for kind in ('gui', 'sku'):
    ids = M['gui_ids'] if kind == 'gui' else M['sku_ids']
    for m in C.MODELS:
        om = collections.defaultdict(list); scores = collections.defaultdict(list)
        for iid in ids:
            gt = gt_of(kind, iid); D1, D2, T1 = drafts(kind, iid, m)
            parts_b, parts_d = [], []
            for ci in range(4):
                key = f'{kind}|{iid}|{m}|c{ci}'
                if key not in meta or f'{key}|blind' not in dec or f'{key}|draft' not in dec: continue
                B = dec[f'{key}|blind']; D = dec[f'{key}|draft']; mk = meta[key]
                parts_b.append((ci, B)); parts_d.append((ci, D))
                for d in mk['deleted']:
                    gi = max(range(len(gt)), key=lambda j: G.iou(d, gt[j])) if gt else None
                    if gi is None or G.iou(d, gt[gi]) < 0.5: continue
                    om['del_rec_draft'].append(int(any(G.iou(b, gt[gi]) >= 0.5 for b in D))); om['del_rec_blind'].append(int(any(G.iou(b, gt[gi]) >= 0.5 for b in B)))
                for inj in mk['injected']:
                    om['inj_keep_draft'].append(int(any(G.iou(b, inj) >= 0.5 for b in D))); om['inj_keep_blind'].append(int(any(G.iou(b, inj) >= 0.5 for b in B)))
                for s0 in mk['shifted']:
                    w = s0[2] - s0[0]; cfull = mk['crop']; d = 0.6 * w if (s0[2] - cfull[0]) / (cfull[2] - cfull[0]) + 0.6 * w / (cfull[2] - cfull[0]) <= 1 else -0.6 * w
                    sh = [s0[0] + d, s0[1], s0[2] + d, s0[3]]
                    cand = [b for b in D if max(G.iou(b, s0), G.iou(b, sh)) >= 0.3]
                    if cand:
                        b = max(cand, key=lambda b: max(G.iou(b, s0), G.iou(b, sh)))
                        om['shift_frac'].append(((b[0] + b[2]) / 2 - (s0[0] + s0[2]) / 2) / d)
            if len(parts_b) < 4 or D1 is None: continue
            MB = merge(parts_b); MD = merge(parts_d)
            for name, S in (('full_D1', D1), ('crops_blind', MB), ('crops_draft', MD), ('tiles_onecall', T1),
                            ('inter_D1_D2', G.construct([[(b, '') for b in D1], [(b, '') for b in D2]], 'inter') if D2 is not None else None),
                            ('inter_D1_crops', G.construct([[(b, '') for b in D1], [(b, '') for b in MB]], 'inter'))):
                if S is None: continue
                s = G.score_boxes(S, gt); scores[name].append(s)
        cell = {}
        for k in ('del_rec_draft', 'del_rec_blind', 'inj_keep_draft', 'inj_keep_blind', 'shift_frac'):
            if om[k]: cell[k] = round(float(np.mean(om[k])), 3)
        cell['n_deleted'] = len(om['del_rec_draft']); cell['n_injected'] = len(om['inj_keep_draft'])
        if om['del_rec_blind'] and np.mean(om['del_rec_blind']) > 0:
            cell['omega_from_deletions'] = round(1 - float(np.mean(om['del_rec_draft'])) / float(np.mean(om['del_rec_blind'])), 3)
        if om['inj_keep_draft']: cell['omega_from_injections'] = round(float(np.mean(om['inj_keep_draft'])) - float(np.mean(om['inj_keep_blind'])), 3)
        for name, L_ in scores.items():
            cell[name] = dict(n=len(L_), E=round(float(np.mean([s['E'] for s in L_])), 2), F1=round(float(np.mean([s['f1'] for s in L_])), 4),
                              recall=round(float(np.sum([s['tp'] for s in L_]) / max(1, np.sum([s['tp'] + s['fn'] for s in L_]))), 4),
                              precision=round(float(np.sum([s['tp'] for s in L_]) / max(1, np.sum([s['tp'] + s['fp'] for s in L_]))), 4),
                              FP=round(float(np.mean([s['fp'] for s in L_])), 2), FN=round(float(np.mean([s['fn'] for s in L_])), 2))
        rngb = np.random.default_rng(0); paired = {}
        for a1, a2 in (('full_D1', 'inter_D1_crops'), ('full_D1', 'crops_blind'), ('tiles_onecall', 'crops_blind'), ('inter_D1_D2', 'inter_D1_crops'), ('crops_blind', 'crops_draft')):
            A = np.array([s['E'] for s in scores.get(a1, [])], float); B = np.array([s['E'] for s in scores.get(a2, [])], float)
            if len(A) != len(B) or len(A) < 3: continue
            bs = [(B[i].mean() - A[i].mean()) / max(1e-9, A[i].mean()) for i in (rngb.integers(0, len(A), len(A)) for _ in range(4000))]
            paired[f'{a2}_vs_{a1}'] = dict(rel=round(float((B.mean() - A.mean()) / max(1e-9, A.mean())) * 100, 1), ci=[round(float(np.percentile(bs, 2.5)) * 100, 1), round(float(np.percentile(bs, 97.5)) * 100, 1)], n=len(A))
        cell['paired'] = paired
        out['cells'][f'{kind}|{m}'] = cell
out['cost'] = {k: round(v, 2) for k, v in spend.items()}
out['key_drift_share'] = {k: round(IRREG[k] / max(1, TOTAL[k]), 3) for k in TOTAL}
pred = json.load(open(f'{C.ROOT}/offline/P2_view_planner_predictions.json'))['cells']
out['P2_check'] = {}
for m in C.MODELS:
    c = out['cells'].get(f'gui|{m}', {})
    if 'full_D1' in c and 'crops_blind' in c:
        obs = c['crops_blind']['recall'] - c['full_D1']['recall']; p = pred[f'{m}|groundcua']
        out['P2_check'][m] = dict(predicted=p['predicted_recall_change'], observed=round(obs, 4), sign_match=(obs > 0) == (p['predicted_recall_change'] > 0))
json.dump(out, open(f'{C.ROOT}/runs/{SUB}/b2_analysis.json', 'w'), indent=1)
print(json.dumps(out, indent=1)[:6000])
