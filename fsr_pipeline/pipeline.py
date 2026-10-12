"""Default reporting pipeline (paper Section 5, Algorithm 1) as one executable entry point.

    result = run(adapter, item, model, cfg, client, views=None, partner=None)

Every branch reads only (i) the task adapter (task specification: classes, alias map, matching rule, loss),
(ii) the shipped configuration cfg (config/default_pipeline.json), (iii) the current responses, obtained through
`client`, and optionally (iv) views supplied by an external view planner. Labels are never read here; scoring
lives in the adapters' `score` method and is called only by evaluation code (replay.py).

The client decides where responses come from. ReplayClient (clients.py) serves stored responses and never calls a
model; a live client would send the same Request objects to an API. No live client is shipped in this package.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

CONFIG = Path(__file__).resolve().parent.parent / 'config' / 'default_pipeline.json'
PRICE = {'gpt61': (2, 10), 'sonnet55': (2, 10), 'gem38': (0.75, 3.75)}     # USD per million input / output tokens


def load_config(path=CONFIG):
    return json.load(open(path))


@dataclass
class Request:
    """One call. kind: first | resend | followup | view | zoom. view: None (whole image) or a crop (x1, y1, x2, y2),
    normalised to the image. rep: index of a repeated call of the same interface."""
    task: str
    item: str
    model: str
    kind: str = 'first'
    rep: int = 0
    view: tuple = None
    previous: str = None          # previous answer attached to a follow-up
    point: tuple = None           # first-stage click for a zoom request


@dataclass
class Call:
    request: Request
    status: str                   # completed | transport_failed | unavailable (replay only: no stored response)
    raw: str = ''
    finish_reason: str = None
    cost: float = 0.0


@dataclass
class Report:
    """Decoded response: audit records, delivered records, stated counts, status."""
    Z_all: list
    Z: list
    d: dict = None
    flags: list = field(default_factory=list)
    status: dict = field(default_factory=dict)
    view: tuple = None


@dataclass
class Result:
    S: list
    y_hat: object
    branch: str
    status: dict
    alarm: object
    cost: float
    calls: list


def call_cost(model, usage):
    if not usage or model not in PRICE: return 0.0
    return ((usage.get('input_tokens') or 0) * PRICE[model][0] + (usage.get('output_tokens') or 0) * PRICE[model][1]) / 1e6


def degeneracy_flags(records, n_invalid, stated_total, finish_reason, cfg, coord_key):
    """The four list-level flags of Section 5, thresholds from cfg. records: every record recovered from the list,
    including those without a valid class; n_invalid: how many of them lack a valid class (or could not be read)."""
    th = cfg['degeneracy_flags']; n = len(records); fl = []
    if n and n_invalid / n > th['missing_class_share_gt']: fl.append('missing_class')
    lo, hi = th['list_length_outside_times_stated_total']
    if stated_total and not (lo * stated_total <= n <= hi * stated_total): fl.append('length')
    if n:
        seen, rep = set(), 0
        for r in records:
            k = coord_key(r); rep += k in seen; seen.add(k)
        if rep / n > th['duplicate_coordinate_share_gt']: fl.append('repeat')
    if th['finish_reason_length'] and str(finish_reason) in ('length', 'max_tokens', 'MAX_TOKENS', 'FinishReason.MAX_TOKENS'):
        fl.append('truncated')
    return fl


def _dispatch(adapter, req, client, cfg, calls, status):
    """Step 1 to 3 for one request: send, decode, bounded recovery (at most one extra call)."""
    c = client.call(req); calls.append(c)
    if c.status == 'transport_failed':                              # L1: resend the identical request once
        c = client.call(Request(**{**req.__dict__, 'kind': 'resend'})); calls.append(c)
        if c.status != 'completed':
            status.setdefault('transport_failed', []).append(req.rep)
            return adapter.empty_report(req)
    if c.status != 'completed':
        status.setdefault('no_response', []).append(req.rep)
        return adapter.empty_report(req)
    rep = adapter.decode(req, c, cfg)
    if rep.flags or rep.status.get('uninformative'):
        status.setdefault('flagged', []).append({'rep': req.rep, 'view': req.view, 'flags': list(rep.flags) or ['no_readable_answer']})
        if cfg['recovery']['max_recovery_calls_per_call'] >= 1:
            f = client.call(Request(**{**req.__dict__, 'kind': 'followup', 'previous': c.raw})); calls.append(f)
            if f.status == 'completed':
                rep2 = adapter.decode(req, f, cfg)
                if not rep2.flags and not rep2.status.get('uninformative'):
                    status.setdefault('recovered', []).append(req.rep); return rep2
                status.setdefault('recovery_still_flagged', []).append(req.rep)
            else:
                status.setdefault('recovery_unavailable', []).append(req.rep)
    return rep


def run(adapter, item, model, cfg, client, views=None, partner=None, confirm_with_views=None, calls_override=None):
    """Algorithm 1.
    views: optional crops from an external view planner; the report of each call is then the merge of its views.
    confirm_with_views: optional crops whose merged report confirms the whole-image result (cross-view confirmation).
    partner: optional second model; its answer is compared with the user's (the alarm).
    calls_override: number of calls K instead of the configured one (used only to replay comparisons)."""
    calls, status = [], {}
    if adapter.kind == 'click':
        return _run_click(adapter, item, model, cfg, client, partner, calls, status)
    K = calls_override or adapter.calls_for(cfg, item)
    view_list = views or [None]
    reports = []                                                    # reports[k] = list over views of Report
    for k in range(K):                                              # Steps 1 to 4
        reports.append([_dispatch(adapter, Request(adapter.task, item, model, 'first' if v is None else 'view', k, v), client, cfg, calls, status)
                        for v in view_list])
    if all(not r.Z for rk in reports for r in rk):
        status['empty_first_pass'] = True
    unread = [k for k, rk in enumerate(reports) if all(r.status.get('uninformative') for r in rk)]
    if unread and K > 1: status['calls_without_readable_answer'] = unread
    if views: status['views'] = len(view_list)
    S = adapter.construct(item, reports, cfg, status)                 # Step 5: merge views, then the construction of the loss
    if confirm_with_views:                                          # Step 5, cross-view confirmation
        vr = [_dispatch(adapter, Request(adapter.task, item, model, 'view', 0, v), client, cfg, calls, status) for v in confirm_with_views]
        S = adapter.confirm(item, S, adapter.merge_views(item, vr), confirm_with_views, cfg)
        status['confirmed_by_views'] = len(confirm_with_views)
    y_hat, branch = adapter.read(S, reports, cfg, status)             # Step 6: read from the final set, or the declared fallback
    alarm = None
    if partner is not None:                                         # Step 8: alarm from a second model's answer
        pr = [[_dispatch(adapter, Request(adapter.task, item, partner, 'first' if v is None else 'view', 0, v), client, cfg, calls, status) for v in view_list]]
        Sp = adapter.construct(item, pr, cfg, {}); yp, _ = adapter.read(Sp, pr, cfg, {})
        alarm = adapter.disagreement(item, (S, y_hat), (Sp, yp))
    return Result(S, y_hat, branch, status, alarm, sum(c.cost for c in calls), calls)


def label_rule(errors_default, errors_alternative, min_images=3):
    """Step 7 (optional labels). Paired per-image losses of the shipped default and of one menu configuration on the same
    labelled images. Switch only if there are at least min_images images and the mean advantage minus one standard error
    over images is positive."""
    import math
    n = len(errors_default)
    if n < min_images: return 'default', None
    adv = [a - b for a, b in zip(errors_default, errors_alternative)]
    m = sum(adv) / n; sd = math.sqrt(sum((x - m) ** 2 for x in adv) / (n - 1)) if n > 1 else float('inf')
    return ('alternative' if m - sd / math.sqrt(n) > 0 else 'default'), m - sd / math.sqrt(n)


def _run_click(adapter, item, model, cfg, client, partner, calls, status):
    zc = cfg['views']['clicks']

    def send(req):
        c = client.call(req); calls.append(c)
        if c.status == 'transport_failed':                          # L1: resend once
            c = client.call(Request(**{**req.__dict__, 'kind': 'resend' if req.kind == 'first' else req.kind + '_resend'})); calls.append(c)
        return c

    def two_stage(m):
        c = send(Request(adapter.task, item, m))
        p1 = adapter.decode_point(item, c) if c.status == 'completed' else None
        if p1 is None:
            status.setdefault('no_first_point', []).append(m); return None
        if not zc['two_stage_zoom']: return p1
        w = adapter.window(p1, zc['crop_fraction_of_each_side'])
        z = send(Request(adapter.task, item, m, 'zoom', view=w, point=tuple(p1)))
        p2 = adapter.decode_point(item, z, window=w) if z.status == 'completed' else None
        if p2 is None:
            status.setdefault('no_point_in_crop', []).append(m); return p1
        return p2
    p = two_stage(model)
    alarm = None
    if partner is not None:
        q = two_stage(partner)
        alarm = adapter.distance(item, p, q)                         # None when either point is missing: treated as an alarm
    return Result(p, p, 'zoomed' if p is not None else 'none', status, alarm, sum(c.cost for c in calls), calls)
