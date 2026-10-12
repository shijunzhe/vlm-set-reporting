"""Task adapters: the task specification (classes, alias map, matching and association rules), the decoders used in the
experiments, and the stored-response index used by ReplayClient. Labels are read only by score(), matched() and _gt(),
which evaluation code calls; pipeline.run never calls them."""
import collections, json, math, os, sys
import numpy as np
from pipeline import Report, degeneracy_flags

HOME = '/home/claude'                         # tools/set_root.py rewrites this root to the repository checkout
sys.path.insert(0, f'{HOME}/stageB/code'); sys.path.insert(0, f'{HOME}/gui/code')
import bcommon as C
import fsrgui as G
MODELS = ('gpt61', 'sonnet55', 'gem38')


def _load_jsonl(path):
    out = {}
    if os.path.exists(path):
        for l in open(path):
            if l.strip():
                r = json.loads(l)
                if r.get('status') == 'completed' or r['request_id'] not in out: out[r['request_id']] = r
    return out


def _merge_views(parts, same):
    """Union of per-view reports with duplicates removed: a record is dropped if it is associated (same()) with a record
    already kept from another view. parts: [(view, records)]."""
    out = []
    for v, P in parts:
        for p in P:
            if not any(w != v and same(p, q) for w, q in out): out.append((v, p))
    return [p for _, p in out]


def _inside(box, view):
    return view[0] <= box[0] and view[1] <= box[1] and box[2] <= view[2] and box[3] <= view[3]


class Base:
    """Shared parts of the set and count adapters."""

    def empty_report(self, req): return Report([], [], None, [], {'no_response': True, 'uninformative': True}, req.view)

    def same(self, item, p, q): raise NotImplementedError

    def merge_views(self, item, reps):
        """One report from the reports of several views of the same call (Step 5)."""
        if len(reps) == 1: return list(reps[0].Z)
        return _merge_views([(r.view, r.Z) for r in reps], lambda p, q: self.same(item, p, q))

    def usable(self, item, reports, cfg):
        """One merged report per call. A call without a readable answer after recovery counts as an empty report
        (default 'count_as_empty', the evaluated construction) or is dropped from support counting ('exclude')."""
        rule = cfg['construction'].get('unreadable_call', {}).get('default', 'count_as_empty')
        out = []
        for rk in reports:
            if rule == 'exclude' and all(r.status.get('uninformative') for r in rk): continue
            out.append(self.merge_views(item, rk))
        return out or [[]]

    def confirm(self, item, S, merged, views, cfg):
        raise NotImplementedError('cross-view confirmation is not configured for this task in this release')


# ------------------------------------------------------------------------------------------------ dense counting (SCP-dense)
class CountSCPDense(Base):
    """Per-type counting on SCP-dense (five classes). Stored responses: RC16 list-first runs (lowest setting for GPT-6.1 and
    Sonnet 5.5; Gemini 3.8 at its lowest setting in rc16 and at its default setting in rc16c, both with a 16k cap)."""
    kind, task = 'count', 'scp_dense_count'
    T5 = ['circle', 'square', 'triangle', 'pentagon', 'star']
    ALIAS = {'diamond': 'square', 'rhombus': 'square', 'rotated square': 'square'}
    TOL = 20

    def __init__(self, gemini_setting='default'):
        root = f'{HOME}/stageB/runs'
        self.index = {}
        for sub in ('rc16', 'rc16c'):
            Q = {json.loads(l)['request_id']: json.loads(l) for l in open(f'{root}/{sub}/requests_main.jsonl')}
            R = {}
            for m in MODELS: R.update(_load_jsonl(f'{root}/{sub}/responses_main_{m}.jsonl'))
            for rid, q in Q.items():
                self.index[(q['meta']['item'], q['model_key'], 'lowest' if sub == 'rc16' else 'default')] = R.get(rid)
        self.gemini_setting = gemini_setting
        self.items = list(range(50))

    def setting(self, model): return self.gemini_setting if model == 'gem38' else 'lowest'

    def lookup(self, req):
        if req.kind in ('first', 'resend') and req.view is None and req.rep == 0:
            return self.index.get((req.item, req.model, self.setting(req.model)))
        return None                                                  # no follow-ups, views or repeated calls were stored

    def calls_for(self, cfg, item): return cfg['construction']['count']['calls']

    def same(self, item, p, q):
        return p['type'] == q['type'] and math.hypot(p['cx'] - q['cx'], p['cy'] - q['cy']) <= self.TOL

    def decode(self, req, call, cfg):
        mode = 'grid' if req.model in ('gpt61', 'gem38') else 'pixel'
        recs, dec, au = C.decode_shapes(call.raw, mode, 1024, 1024, 1024, 1024)
        recs = [dict(x, type=self.ALIAS.get(x['type'], x['type'])) for x in recs]
        Z = [x for x in recs if x['type'] in self.T5]
        d = None
        if isinstance(dec, dict):                                    # the alias map applies to stated counts too
            d = {t: 0 for t in self.T5}
            for k, v in dec.items():
                t = self.ALIAS.get(str(k).lower().strip(), str(k).lower().strip())
                if t in d:
                    try: d[t] += int(v or 0)
                    except (TypeError, ValueError): pass
        fl = degeneracy_flags(recs, len(recs) - len(Z), sum(d.values()) if d else None, call.finish_reason, cfg,
                              lambda r: (round(r['cx']), round(r['cy'])))
        return Report(recs, Z, d, fl, {'json_ok': au['json_ok'], 'salvaged': au['salvaged'], 'no_records_recovered': not recs,
                                       'uninformative': not recs and d is None, 'unknown_class': len(recs) - len(Z)}, req.view)

    def construct(self, item, reports, cfg, status):
        return self.usable(item, reports, cfg)[0]                    # one call; its views merged

    def read(self, S, reports, cfg, status):
        lc = collections.Counter(x['type'] for x in S); lc = {t: lc.get(t, 0) for t in self.T5}
        rk = reports[0]; flagged = any(r.flags for r in rk)
        if flagged and len(rk) == 1 and rk[0].d is not None:
            return dict(rk[0].d), 'count_from_statement'
        if flagged and len(rk) > 1:
            status['stated_counts_not_combined_across_views'] = True   # overlapping views would double-count
        return lc, ('list_count_flagged' if flagged else 'list_count')

    def disagreement(self, item, a, b):
        return sum(abs(a[1][t] - b[1][t]) for t in self.T5)        # per-type difference of the two answers

    # --- evaluation only
    def score(self, item, y_hat):
        g = json.load(open(f'{HOME}/stageB/data/scpd_gen/ground_truth/gt_{item:03d}.json'))['shapes']
        gc = collections.Counter(s['type'] for s in g)
        return sum(abs(y_hat[t] - gc[t]) for t in self.T5)


# ------------------------------------------------------------------------------------------------ E3 sets (boxes and points)
class SetE3(Base):
    """Detection and point sets on the E3 sources (list interface, lowest setting). Stored: three calls per image and model;
    follow-ups (previous answer attached) for a sample of flagged first calls (runs_repair, action CHANGE)."""
    kind, task = 'set', 'e3_set'

    def __init__(self, dataset):
        E3 = f'{HOME}/fsr_v51/e3'; sys.path.insert(0, E3); cwd = os.getcwd(); os.chdir(E3)
        import e3lib as L, analyze_e3 as A0, offline_agg2, aggregate_v2 as AG          # offline_agg2 installs the release correspondence
        from score_e3 import load_items
        self.L, self.A0, self.AG = L, A0, AG
        self.items, self.dec_items = load_items(), load_items('')
        self.R, self.F = {}, {}
        for m in MODELS:
            for rid, r in _load_jsonl(f'{E3}/runs_main/responses_{m}.jsonl').items():
                p = rid.split(':')
                if p[1] in ('MAIN', 'AGG') and p[4] == 'L-min': self.R[(p[2], p[3], int(p[5][1:]))] = r
            for rid, r in _load_jsonl(f'{E3}/runs_repair/responses_{m}.jsonl').items():
                if rid.startswith('E3:REPAIR:CHANGE:'):
                    p = rid.split(':'); self.F[(p[5], p[6], int(p[8][1:]))] = r
        os.chdir(cwd)
        self.dataset = dataset
        self.images = sorted(i for i, it in self.items.items() if it['dataset'] == dataset
                             and all((i, m, k) in self.R for m in MODELS for k in (0, 1, 2)))

    def lookup(self, req):
        if req.view is not None: return None
        if req.kind in ('first', 'resend'): return self.R.get((req.item, req.model, req.rep))
        if req.kind == 'followup': return self.F.get((req.item, req.model, req.rep))
        return None

    def calls_for(self, cfg, item):
        reg = cfg['task_family_regime'][self.dataset]
        return cfg['construction']['set_error_over_reporting_families' if reg == 'over' else 'set_error_under_reporting_families']['calls']

    def same(self, item, p, q):
        it = self.items[item]
        if it['task'] == 'point' or p['box'] is None: return p['label'] == q['label'] and math.hypot(p['cx'] - q['cx'], p['cy'] - q['cy']) <= it['tau_point']
        return p['label'] == q['label'] and self.L.iou_matrix([p['box']], [q['box']])[0, 0] >= 0.5

    def decode(self, req, call, cfg):
        it = self.dec_items[req.item]
        recs, au = self.L.decode(it, 'L', call.raw)
        Z = [r for r in recs if it['task'] == 'point' or r['box'] is not None]
        allrecs = recs + [None] * au['invalid']
        fl = degeneracy_flags(allrecs, au['invalid'], None, call.finish_reason, cfg,
                              lambda r: (round(r['cx']), round(r['cy'])) if r else object())
        unread = not recs and (not (call.raw or '').strip() or au['salvaged'] > 0)
        return Report(allrecs, Z, None, fl, {'salvaged': au['salvaged'], 'invalid': au['invalid'],
                                            'no_records_recovered': not recs, 'uninformative': unread}, req.view)

    def construct(self, item, reports, cfg, status):
        use = self.usable(item, reports, cfg); K = len(use)
        if K == 1: return list(use[0])
        cl = self.AG.build_clusters(self.items[item], use)
        return [self.AG.fuse(c[1]) for c in cl if len(c[2]) == K]           # unanimity, averaged coordinates

    def read(self, S, reports, cfg, status): return S, 'set'

    def disagreement(self, item, a, b):
        """Records of the two answers not associated with each other (the label-free disagreement of Proposition 5)."""
        cl = self.AG.build_clusters(self.items[item], [a[0], b[0]])
        return sum(len(c[1]) for c in cl if len(c[2]) == 1)

    def score(self, item, S):
        it = self.items[item]; E, r, c = self.L.error(it, S, self.A0.metric_of(it, self.A0.primary(it))); return E


# ------------------------------------------------------------------------------------------------ screen parsing
class SetGUI(Base):
    """Screen parsing (GroundCUA, ScreenParse): class-agnostic boxes in normalised coordinates; stored drafts D1 to D3."""
    kind, task = 'set', 'gui_parse'

    def __init__(self, source):
        self.source = source
        root = f'{HOME}/gui'
        self.items = {i['id']: i for i in json.load(open(f'{root}/data/parse_items.json'))}
        self.Q = {json.loads(l)['request_id']: json.loads(l) for l in open(f'{root}/runs/main/requests_A.jsonl')}
        self.R = {}
        for m in MODELS: self.R.update(_load_jsonl(f'{root}/runs/main/responses_A_{m}.jsonl'))
        self.images = sorted(i for i, it in self.items.items() if it['source'] == source and all(
            f'{i}|{m}|D{j}' in self.R and self.R[f'{i}|{m}|D{j}']['status'] == 'completed' for m in MODELS for j in (1, 2, 3)))

    def lookup(self, req):
        if req.view is None and req.kind in ('first', 'resend'): return self.R.get(f'{req.item}|{req.model}|D{req.rep + 1}')
        return None

    def calls_for(self, cfg, item):
        reg = cfg['task_family_regime'][self.source]
        return cfg['construction']['set_error_over_reporting_families' if reg == 'over' else 'set_error_under_reporting_families']['calls']

    def same(self, item, p, q): return G.iou(p[0], q[0]) >= 0.5

    def decode(self, req, call, cfg):
        q = self.Q.get(f'{req.item}|{req.model}|D{req.rep + 1}')
        scale = None if (q is None or q['meta']['mode'] == 'grid') else tuple(q['meta']['sizes'][0])
        recs, au = G.decode_list(call.raw, scale)
        allrecs = recs + [None] * au['invalid']
        fl = degeneracy_flags(allrecs, au['invalid'], None, call.finish_reason, cfg,
                              lambda r: tuple(round(v, 3) for v in r[0]) if r else object())
        return Report(allrecs, recs, None, fl, {'salvaged': au['salvaged'], 'invalid': au['invalid'], 'no_records_recovered': not recs,
                                                'uninformative': not recs and not au['json_ok']}, req.view)

    def construct(self, item, reports, cfg, status):
        use = self.usable(item, reports, cfg); K = len(use)
        if K == 1: return [b for b, _ in use[0]]
        return [list(c['fused']) for c in G.clusters(use) if len(c['sup']) == K]

    def read(self, S, reports, cfg, status): return S, 'set'

    def disagreement(self, item, a, b):
        cl = G.clusters([[(x, '') for x in a[0]], [(x, '') for x in b[0]]])
        return sum(len(c['members']) for c in cl if len(c['sup']) == 1)

    def score(self, item, S): return G.score_boxes(S, self.items[item]['targets'])['E']


# ------------------------------------------------------------------------------------------------ multi-view reports (B2 crops)
class MultiViewBoxes(Base):
    """Reports from an external view planner: four native-resolution crops (6% overlap), one call per crop, and up to three
    whole-image calls. Sources: GroundCUA screens and SKU-110K images of the cross-view experiment (stageB/runs/b2).
    All boxes in normalised image coordinates, class-agnostic, as in that experiment."""
    kind, task = 'set', 'multiview_boxes'

    def __init__(self, kind):
        self.k = kind; root = f'{HOME}/stageB/runs/b2'
        M = json.load(open(f'{root}/items.json')); self.meta = M['meta']
        self.ids = M['gui_ids'] if kind == 'gui' else M['sku_ids']
        self.Q = {json.loads(l)['request_id']: json.loads(l) for l in open(f'{root}/requests_main.jsonl')}
        self.R = {}
        for m in MODELS: self.R.update(_load_jsonl(f'{root}/responses_main_{m}.jsonl'))
        self.full = SetGUI('groundcua') if kind == 'gui' else SetE3('sku110k')

    def views(self, item, model):
        return [tuple(self.meta[f'{self.k}|{item}|{model}|c{ci}']['crop']) for ci in range(4) if f'{self.k}|{item}|{model}|c{ci}' in self.meta]

    def _crop_key(self, req):
        for ci in range(4):
            key = f'{self.k}|{req.item}|{req.model}|c{ci}'
            if key in self.meta and tuple(self.meta[key]['crop']) == tuple(req.view): return key
        return None

    def lookup(self, req):
        if req.kind not in ('first', 'resend', 'view'): return None    # no follow-ups were stored for this experiment
        if req.view is None:
            if self.k == 'gui': return self.full.lookup(req)
            return self.full.R.get((req.item, req.model, req.rep))
        key = self._crop_key(req)
        return self.R.get(f'{key}|blind') if key else None

    def calls_for(self, cfg, item): return 1

    def same(self, item, p, q): return G.iou(p, q) >= 0.5

    def decode(self, req, call, cfg):
        if req.view is None:
            if self.k == 'gui':
                rep = self.full.decode(req, call, cfg); Z = [b for b, _ in rep.Z]; fl, st = rep.flags, rep.status
            else:
                rep = self.full.decode(req, call, cfg); it = self.full.dec_items[req.item]; W, H = it['width'], it['height']
                Z = [[r['box'][0] / W, r['box'][1] / H, r['box'][2] / W, r['box'][3] / H] for r in rep.Z if r['box'] is not None]
                fl, st = rep.flags, rep.status
            return Report(Z, Z, None, fl, st, None)
        q = self.Q[f'{self._crop_key(req)}|blind']['meta']; v = req.view; w, h = v[2] - v[0], v[3] - v[1]
        bx, _ = C.boxes_positional(call.raw); Z = []
        for x1, y1, x2, y2 in bx:
            b = [x1 / 1000, y1 / 1000, x2 / 1000, y2 / 1000] if q['mode'] == 'grid' else [x1 / q['sent'][0], y1 / q['sent'][1], x2 / q['sent'][0], y2 / q['sent'][1]]
            if b[2] > b[0] and b[3] > b[1]: Z.append([v[0] + b[0] * w, v[1] + b[1] * h, v[0] + b[2] * w, v[1] + b[3] * h])
        fl = degeneracy_flags(Z, 0, None, call.finish_reason, cfg, lambda r: tuple(round(x, 3) for x in r))
        return Report(Z, Z, None, fl, {'no_records_recovered': not Z, 'uninformative': not Z and not (call.raw or '').strip()}, req.view)

    def construct(self, item, reports, cfg, status):
        use = self.usable(item, reports, cfg); K = len(use)
        if K == 1: return list(use[0])
        return [list(c['fused']) for c in G.clusters([[(b, '') for b in P] for P in use]) if len(c['sup']) == K]

    def confirm(self, item, S, merged, views, cfg):
        """Cross-view confirmation of S by the merged view report. Confirmed records take averaged coordinates.
        coverage_aware (shipped): a record lying wholly inside some view is kept only if a view record is associated
        with it; a record lying wholly inside no view is kept, because no view could have reported it. plain: every
        record needs an associated view record."""
        rule = cfg['views']['sets_and_counts'].get('confirmation_rule', 'coverage_aware')
        out = []
        for c in G.clusters([[(b, '') for b in S], [(b, '') for b in merged]]):
            if 0 not in c['sup']: continue
            if len(c['sup']) == 2: out.append(list(c['fused'])); continue
            b = c['members'][0]
            if rule == 'coverage_aware' and not any(_inside(b, v) for v in views): out.append(b)
        return out

    @staticmethod
    def confirm_deletion_only(full, merged):
        cl = G.clusters([[(b, '') for b in full], [(b, '') for b in merged]])
        return [c['members'][0] for c in cl if 0 in c['sup'] and len(c['sup']) == 2]

    def read(self, S, reports, cfg, status): return S, 'set'

    def disagreement(self, item, a, b):
        cl = G.clusters([[(x, '') for x in a[0]], [(x, '') for x in b[0]]])
        return sum(len(c['members']) for c in cl if len(c['sup']) == 1)

    # --- evaluation only
    def _gt(self, item):
        if self.k == 'gui': return self.full.items[item]['targets']
        it = self.full.items[item]; W, H = it['width'], it['height']
        return [[t['box'][0] / W, t['box'][1] / H, t['box'][2] / W, t['box'][3] / H] for t in it['targets']]

    def matched(self, item, S): return G.score_boxes(S, self._gt(item))['tp']

    def score(self, item, S): return G.score_boxes(S, self._gt(item))['E']


# ------------------------------------------------------------------------------------------------ clicks
class Click:
    """Click grounding (ScreenSpot-Pro, ScreenSpot-v2): first click on the whole screenshot (C1) and the second click on the
    crop around it (B4)."""
    kind, task = 'click', 'click'

    def __init__(self, source='sspro'):
        root = f'{HOME}/gui'
        self.items = {c['id']: c for c in json.load(open(f'{root}/data/click_items_prepped.json')) if c['source'] == source}
        self.QA = {json.loads(l)['request_id']: json.loads(l) for l in open(f'{root}/runs/main/requests_A.jsonl')}
        self.QB = {json.loads(l)['request_id']: json.loads(l) for l in open(f'{root}/runs/main/requests_B.jsonl')}
        self.RA, self.RB = {}, {}
        for m in MODELS:
            self.RA.update(_load_jsonl(f'{root}/runs/main/responses_A_{m}.jsonl')); self.RB.update(_load_jsonl(f'{root}/runs/main/responses_B_{m}.jsonl'))

    def lookup(self, req):
        if req.kind in ('first', 'resend'): return self.RA.get(f'{req.item}|{req.model}|C1')
        if req.kind in ('zoom', 'zoom_resend'):
            k = f'{req.item}|{req.model}|B4'; q = self.QB.get(k)
            if q is None or any(abs(a - b) > 1e-6 for a, b in zip(q['meta']['window'], req.view)): return None
            return self.RB.get(k)
        return None

    def window(self, p, frac): return G.window(p[0], p[1], frac)

    def decode_point(self, item, call, window=None):
        r = call.request; key = f'{item}|{r.model}|' + ('B4' if window else 'C1'); q = (self.QB if window else self.QA).get(key)
        scale = None if (q is None or q['meta']['mode'] == 'grid') else tuple(q['meta']['sizes'][0])
        p, _ = G.decode_click(call.raw, scale)
        if p is None or window is None: return p
        w = window; return [w[0] + p[0] * (w[2] - w[0]), w[1] + p[1] * (w[3] - w[1])]

    def distance(self, item, p, q):
        if p is None or q is None: return None
        c = self.items[item]; W, H = c['width'], c['height']
        return math.hypot((p[0] - q[0]) * W, (p[1] - q[1]) * H) / max(W, H)

    def score(self, item, p): return int(G.hit(p, self.items[item]['bbox_norm']))
