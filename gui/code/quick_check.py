"""Quick decode/score/cost check of a stage's responses (labels used only for scoring)."""
import json, sys, collections, os
import numpy as np
sys.path.insert(0, 'code'); import fsrgui as G
sys.path.insert(0, '/home/claude/fsr_v51/e3'); from score_e3 import PRICE
subset, stage = sys.argv[1], sys.argv[2]
P = {i['id']: i for i in json.load(open('data/parse_items.json'))}; C = {c['id']: c for c in json.load(open('data/click_items_prepped.json'))}
for m in ['gpt61', 'sonnet55', 'gem38']:
    f = f'runs/{subset}/responses_{stage}_{m}.jsonl'
    if not os.path.exists(f): continue
    R = [json.loads(l) for l in open(f)]
    cost = sum((r['usage']['input_tokens'] * PRICE[m][0] + r['usage']['output_tokens'] * PRICE[m][1]) / 1e6 for r in R if r['status'] == 'completed')
    by = collections.defaultdict(list)
    RQ = {json.loads(l)['request_id']: json.loads(l) for l in open(f'runs/{subset}/requests_{stage}.jsonl')}
    for r in R:
        mt = RQ[r['request_id']]['meta']; arm = mt['arm']; u = r.get('usage', {})
        sc = [None if mt.get('mode', 'grid') == 'grid' else tuple(s) for s in mt.get('sizes', [None])]
        row = dict(fin=r.get('finish_reason'), inp=u.get('input_tokens'), out=u.get('output_tokens'))
        if arm in ('draft', 'tiles'):
            recs, au = G.decode_list(r['raw_text'], sc[0]) if arm == 'draft' else G.decode_tiles(r['raw_text'], sc if sc[0] else None); it = P[mt['item']]
            s = G.score_boxes([b for b, n in recs], it['targets']); row.update(n_pred=len(recs), n=it['n'], E=s['E'], f1=s['f1'], f1c=s['f1_c'], json=au['json_ok'], resc=au['rescaled'], inval=au['invalid'])
        elif arm == 'click':
            p, au = G.decode_click(r['raw_text'], sc[0]); c = C[mt['item']]; row.update(hit=int(G.hit(p, c['bbox_norm'])), none=int(p is None), json=au['json_ok'], resc=au.get('rescaled', 0))
        elif arm in ('A3audit', 'A4audit'):
            e, au = G.decode_edits(r['raw_text'], mt['n_draft'], arm == 'A4audit', sc[0], sc[1:] if arm == 'A4audit' and sc[0] else None); row.update(n_edits=len(e), prop=au['proposed'], inval=au['invalid'], json=au['json_ok'],
                                                                                               ops=dict(collections.Counter(x['op'] for x in e)), n_new=sum(len(x['new']) for x in e))
        elif arm in ('A3verify', 'A4verify'):
            d, au = G.decode_decisions(r['raw_text'], mt['n_changes']); row.update(dec=dict(collections.Counter(d)), json=au['json_ok'], miss=au['missing'])
        elif arm == 'B3':
            d, p, au = G.decode_vclick(r['raw_text'], sc[0]); row.update(dec=d, has_pt=int(p is not None), json=au['json_ok'])
        elif arm == 'B4':
            p, au = G.decode_click(r['raw_text'], sc[0]); row.update(found=int(p is not None), json=au['json_ok'])
        by[arm].append(row)
    print(f'== {m} cost ${cost:.3f} calls {len(R)}')
    for arm, rows in by.items():
        keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float)) and rows[0][k] is not None]
        print(' ', arm, len(rows), {k: round(float(np.mean([r[k] for r in rows if r.get(k) is not None])), 3) for k in keys},
              'fin', dict(collections.Counter(str(r['fin']) for r in rows)))
        if arm in ('A3audit', 'A4audit', 'A3verify', 'A4verify', 'B3'): print('    ', [r.get('ops') or r.get('dec') for r in rows][:8])
