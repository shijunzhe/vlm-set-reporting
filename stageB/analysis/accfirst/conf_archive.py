"""Per-item archive of the 600-item confirmation (protocol v3, runs/conf_v1). Read-only over the saved responses; makes no
model calls and does not change any rule. Every final point is recomputed with the frozen functions in gui/code/decision.py
and the detailed reconstruction of the comparison step is asserted to give the same final point as decision.c_rule.
Usage: python3 accfirst/conf_archive.py --subset conf_v1 --out <dir>
Writes <dir>/per_item.jsonl (full record per item) and <dir>/per_item.csv (one flat row per item)."""
import argparse, collections, csv, itertools, json, os, sys
from pathlib import Path
BASE = os.environ.get('FSR_BASE', '/home/claude'); GUI = BASE + '/gui'
sys.path.insert(0, GUI + '/code')
cwd = os.getcwd(); import fusion_stages as F; import gl_stages as GLS; import decision as DC; os.chdir(cwd)
G, B = F.G, F.B
ap = argparse.ArgumentParser(); ap.add_argument('--subset', required=True); ap.add_argument('--out', required=True); a = ap.parse_args()
OUTDIR = Path(a.out).resolve(); OUTDIR.mkdir(parents=True, exist_ok=True)
os.chdir(GUI)
C = F.items_of('main' if a.subset == 'main' else 'conf')
if a.subset == 'main':
    keep = {json.loads(l)['meta']['item'] for l in open('runs/main/requests_S.jsonl')}; C = {k: v for k, v in C.items() if k in keep}
items = sorted(C); run = DC.Run(GUI, a.subset, G, B)
hit = lambda p, c: int(p is not None and G.hit(p, C[c]['bbox_norm']))
r6 = lambda p: None if p is None else [round(p[0], 6), round(p[1], 6)]


def cmp_detail(cfg, cid):
    """same logic as decision.c_rule, but returns the pairwise verdicts and wins as well"""
    A, sc, cls, flags = DC.structure(run, cfg, cid, C[cid])
    if not cls: return dict(point=None, pairs=[], wins={}, compared=[], s_choice=None, path='no_output')
    S = max(cls, key=lambda cl: DC.s_key(sc, cl)); comp = DC.compared(sc, cls)
    out = dict(compared=[cl['rep']['id'] for cl in comp], s_choice=S['rep']['id'], pairs=[], wins={})
    if len(comp) < 2: out.update(point=S['rep']['point'], chosen=S['rep']['id'], path='single_cluster' if len(cls) == 1 else 'single_compared'); return out
    st = f'CMPA_{cfg}'; Q, R, _ = run.stage(st); wins = collections.Counter()
    for x, y in itertools.combinations(comp, 2):
        verdict = {}
        for o, (u, v) in (('ab', (x, y)), ('ba', (y, x))):
            for rid in (f'{cid}|{DC.GEM}|{st}|{u["rep"]["id"]}|{v["rep"]["id"]}|{o}', f'{cid}|{DC.GEM}|{st}|{v["rep"]["id"]}|{u["rep"]["id"]}|{o}'):
                s = run.status(st, rid)
                if s == 'not_built': continue
                if s == 'completed':
                    q = Q[rid]; rv = G.first_json(R[rid]['raw_text']); lab = rv.get('preferred_id') if isinstance(rv, dict) else None
                    verdict[q['meta']['order']] = q['meta']['labels'].get(lab)
                else: verdict[o] = f'<{s}>'
                break
        agree = verdict.get('ab') is not None and not str(verdict.get('ab')).startswith('<') and verdict.get('ab') == verdict.get('ba')
        if agree: wins[verdict['ab']] += 1
        out['pairs'].append(dict(a=x['rep']['id'], b=y['rep']['id'], ab=verdict.get('ab'), ba=verdict.get('ba'), win=verdict['ab'] if agree else None))
    out['wins'] = dict(wins)
    if not wins: out.update(point=S['rep']['point'], chosen=S['rep']['id'], path='no_wins_S'); return out
    top = max(wins.values()); best = [cl for cl in comp if wins.get(cl['rep']['id'], 0) == top]
    ch = max(best, key=lambda cl: DC.s_key(sc, cl)); out.update(point=ch['rep']['point'], chosen=ch['rep']['id'], path='most_wins' if len(best) == 1 else 'tie_S_key')
    return out


def glf_trace(cid):
    L = {n: run.stage(f'GLF{n}')[:2] for n in (1, 2, 3)}; tr = []
    for n in (1, 2, 3):
        Q, R = L[n]; k = f'{cid}|{GLS.MOD}|GLF{n}'
        if k not in Q: break
        if k not in R: tr.append(dict(round=n, action='no_response')); break
        fr = Q[k]['meta']['frame']
        if n == 3:
            p, _ = G.decode_click(R[k]['raw_text'], None); tr.append(dict(round=n, action='click' if p is not None else 'unparsed', frame=fr)); break
        act, val = GLS.decode(R[k]['raw_text'], fr)
        tr.append(dict(round=n, action=act or 'unparsed', value=r6(val) if act == 'click' else ([round(v, 6) for v in val] if val else None), frame=fr))
        if act != 'crop': break
    return tr


EV = {}
rows, recs = [], []
for cid in items:
    c = C[cid]; rec = dict(item=cid, app=c.get('app'), platform=c.get('platform'), group=c.get('group'), ui_type=c.get('ui_type'),
                           instruction=c['instruction'], image=c['image'], width=c['width'], height=c['height'], bbox_px=c['bbox'],
                           bbox_norm=[round(v, 6) for v in c['bbox_norm']])
    finals = {}
    for fl, fn in (('GPT-min-zoom', lambda: DC.single_zoom(run, 'M', 'gpt61', cid)), ('GPT-default-zoom', lambda: DC.single_zoom(run, 'D', 'gpt61', cid)),
                   ('Sonnet-min-zoom', lambda: DC.single_zoom(run, 'M', 'sonnet55', cid)), ('Sonnet-default-zoom', lambda: DC.single_zoom(run, 'D', 'sonnet55', cid)),
                   ('GLF', lambda: DC.glf(run, cid, GLS.final_point, C)), ('GPTx3', lambda: DC.gpt_x3(run, cid, c)),
                   ('M-S', lambda: DC.s_rule(run, 'M', cid, c)), ('M-C', lambda: DC.c_rule(run, 'M', cid, c)),
                   ('D-S', lambda: DC.s_rule(run, 'D', cid, c)), ('D-C', lambda: DC.c_rule(run, 'D', cid, c))):
        p, f = fn(); finals[fl] = dict(point=r6(p), hit=hit(p, cid), flags=sorted(f))
    rec['final'] = finals
    for cfg in ('M', 'D'):
        A, sc, cls, flags = DC.structure(run, cfg, cid, c)
        cl_of = {i: k for k, cl in enumerate(cls) for i in cl['members']}
        cands = [dict(id=x['id'], model=x['model'], setting=x['setting'], parent=x['parent'], point=r6(x['point']), hit=hit(x['point'], cid),
                      p_hit=sc.get(x['id']), cluster=cl_of[x['id']]) for x in A]
        cl_out = [dict(index=k, members=cl['members'], size=cl['size'], rep=cl['rep']['id'], rep_point=r6(cl['rep']['point']),
                       rep_hit=hit(cl['rep']['point'], cid), s_key=[DC.scval(sc, cl['rep']), cl['size'], -cl['rank']]) for k, cl in enumerate(cls)]
        cd = cmp_detail(cfg, cid)
        ref = finals[f'{cfg}-C']['point']; assert r6(cd['point']) == ref, ('cmp reconstruction differs', cid, cfg)
        s_pt = finals[f'{cfg}-S']['point']
        events = {}
        for rule in ('S', 'C'):
            fl = f'{cfg}-{rule}'
            if finals[fl]['hit']: ev = 'correct'
            elif not any(x['hit'] for x in cands): ev = 'no_correct_candidate'
            elif not any(x['rep_hit'] for x in cl_out): ev = 'representative_lost_correct'
            else: ev = 'wrong_choice_among_representatives'
            events[rule] = ev
        rec[cfg] = dict(candidates=cands, clusters=cl_out, comparison=dict((k, v) for k, v in cd.items() if k != 'point'), events=events, flags=sorted(flags))
    rec['GLF_trace'] = glf_trace(cid)
    gp = finals['GLF']['point']
    rec['GLF_near_D_candidate'] = bool(gp is not None and any(DC.dist(gp, x['point'], c) <= DC.DELTA for x in rec['D']['candidates']))
    recs.append(rec)
    row = dict(item=cid, app=rec['app'], platform=rec['platform'], group=rec['group'], ui_type=rec['ui_type'], width=c['width'], height=c['height'],
               bbox_px=' '.join(map(str, c['bbox'])))
    for fl, v in finals.items():
        row[f'{fl}_x'], row[f'{fl}_y'] = (v['point'] or [None, None]); row[f'{fl}_hit'] = v['hit']
    for cfg in ('M', 'D'):
        d = rec[cfg]
        for x in d['candidates']:
            tag = {'gpt61': 'gpt', 'sonnet55': 'sonnet', 'gem38': 'gemini'}[x['model']]
            row[f'{cfg}_{tag}_x'], row[f'{cfg}_{tag}_y'] = x['point']; row[f'{cfg}_{tag}_hit'] = x['hit']; row[f'{cfg}_{tag}_phit'] = x['p_hit']; row[f'{cfg}_{tag}_cluster'] = x['cluster']
        row[f'{cfg}_n_clusters'] = len(d['clusters']); row[f'{cfg}_reps_hit'] = ''.join(str(cl['rep_hit']) for cl in d['clusters'])
        row[f'{cfg}_cmp_path'] = d['comparison'].get('path'); row[f'{cfg}_cmp_wins'] = json.dumps(d['comparison'].get('wins', {}))
        row[f'{cfg}-S_event'] = d['events']['S']; row[f'{cfg}-C_event'] = d['events']['C']
    row['GLF_rounds'] = '>'.join(t['action'] for t in rec['GLF_trace']); row['GLF_near_D_candidate'] = int(rec['GLF_near_D_candidate'])
    rows.append(row)

with open(OUTDIR / 'per_item.jsonl', 'w') as f:
    for r in recs: f.write(json.dumps(r, ensure_ascii=False) + '\n')
cols = []
for r in rows:
    for k in r:
        if k not in cols: cols.append(k)
with open(OUTDIR / 'per_item.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); [w.writerow(r) for r in rows]
# summary for cross-checking against conf_eval_<subset>.json
summ = {fl: sum(1 - r['final'][fl]['hit'] for r in recs) for fl in recs[0]['final']}
ev = {f'{cfg}-{rule}': dict(collections.Counter(r[cfg]['events'][rule] for r in recs)) for cfg in ('M', 'D') for rule in ('S', 'C')}
x = collections.Counter()
for r in recs:
    g, d = r['final']['GLF']['hit'], r['final']['D-C']['hit']
    if g and not d: x['GLF_right_DC_wrong:' + r['D']['events']['C']] += 1
    if d and not g: x['DC_right_GLF_wrong:GLF_near_D_candidate=' + str(r['GLF_near_D_candidate'])] += 1
print(json.dumps(dict(errors=summ, events=ev, glf_cross=dict(x)), indent=1))
json.dump(dict(errors=summ, events=ev, glf_cross=dict(x)), open(OUTDIR / 'summary.json', 'w'), indent=1)
