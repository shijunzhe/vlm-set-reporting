"""Replay the default pipeline on stored responses and on boundary cases. No model calls.
Usage: python3 -I fsr_pipeline/replay.py  ->  stageB/analysis/out/pipeline_replay.json
Checks that the single entry point (pipeline.run) reproduces the component analyses of the paper from raw responses, and
records how every branch behaves on boundary cases (empty, unparseable, partial JSON, no stated counts, degenerate list,
transport failure, missing click)."""
import copy, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import run, load_config
from clients import ReplayClient
import adapters as A

HOME = A.HOME
OUT = Path(f'{HOME}/stageB/analysis/out'); OUT.mkdir(parents=True, exist_ok=True)
CFG = load_config()
NOREC = copy.deepcopy(CFG); NOREC['recovery']['max_recovery_calls_per_call'] = 0
EXCL = copy.deepcopy(NOREC); EXCL['construction']['unreadable_call']['default'] = 'exclude'
MODELS = ['gpt61', 'sonnet55', 'gem38']
res = {'about': __doc__.strip().splitlines()[0], 'count': {}, 'sets': {}, 'clicks': {}, 'multiview': {}, 'boundary': {}}


def branches(results):
    out = {}
    for r in results:
        out[r.branch] = out.get(r.branch, 0) + 1
    return out


def status_counts(results):
    out = {}
    for r in results:
        for k in r.status:
            out[k] = out.get(k, 0) + 1
    return out


# ------------------------------------------------------------------ 1. dense counts (Table 7 full method; Table 9)
for gem in ('lowest', 'default'):
    ad = A.CountSCPDense(gemini_setting=gem)
    for m in MODELS:
        if gem == 'default' and m != 'gem38': continue
        R = [run(ad, i, m, CFG, ReplayClient(ad)) for i in ad.items]
        err = [ad.score(i, r.y_hat) for i, r in zip(ad.items, R)]
        res['count'][f'{m}|{"lowest" if m != "gem38" else gem}'] = dict(
            n=len(R), per_type_error=round(float(np.mean(err)), 3), branches=branches(R), status=status_counts(R),
            cost_per_image=round(float(np.mean([r.cost for r in R])), 5))
rc = json.load(open(OUT / 'rc16_setting.json'))['levels']; ro = json.load(open(OUT / 'readout_effects.json'))
res['count']['reference'] = {'gem38|lowest': rc['min16k']['readout'], 'gem38|default': rc['default16k']['readout'],
                             'gpt61|lowest': ro['gpt61']['full_method']['b'] if 'full_method' in ro['gpt61'] else None,
                             'sonnet55|lowest': ro['sonnet55']['full_method']['b'] if 'full_method' in ro['sonnet55'] else None}

# ------------------------------------------------------------------ 2. sets (Table 13 single and unanimity cells)
ct = json.load(open(OUT / 'construction_table.json'))['cells']
srcs = [('E3', ds) for ds in ('doclaynet', 'dota', 'fsc147', 'sku110k')] + [('GUI', s) for s in ('groundcua', 'screenparse')]
for fam, ds in srcs:
    ad = A.SetE3(ds) if fam == 'E3' else A.SetGUI(ds)
    over = CFG['task_family_regime'][ds] == 'over'
    for m in MODELS:
        cell = {}
        for name, cfg in (('no_recovery', NOREC), ('no_recovery_unreadable_excluded', EXCL), ('with_stored_recovery', CFG)):
            R = [run(ad, i, m, cfg, ReplayClient(ad)) for i in ad.images]
            E = [ad.score(i, r.S) for i, r in zip(ad.images, R)]
            cell[name] = dict(n=len(R), E=round(float(np.mean(E)), 3), status=status_counts(R), cost_per_image=round(float(np.mean([r.cost for r in R])), 5))
        ref = ct[f'{fam}|{ds}|{m}']['unan3_fus' if over else 'single']['E']
        cell['reference_component_E'] = ref; cell['construction'] = 'unan3_fus' if over else 'single'
        res['sets'][f'{fam}|{ds}|{m}'] = cell
    print(fam, ds, {m: (res['sets'][f'{fam}|{ds}|{m}']['no_recovery']['E'], res['sets'][f'{fam}|{ds}|{m}']['reference_component_E']) for m in MODELS}, flush=True)

# ------------------------------------------------------------------ 3. clicks: two-stage zoom and after-zoom cross-model alarm
ad = A.Click('sspro'); partner = CFG['alarm']['partner_rule']; delta = CFG['alarm']['click_threshold_fraction_of_longer_side']
for m in MODELS:
    ids = sorted(ad.items)
    R = [run(ad, i, m, CFG, ReplayClient(ad), partner=partner[m]) for i in ids]
    hits = [ad.score(i, r.S) for i, r in zip(ids, R)]
    errs = [1 - h for h in hits]
    deferred = [r.alarm is None or r.alarm > delta for r in R]
    caught = sum(1 for e, d in zip(errs, deferred) if e and d)
    acted = [h for h, d in zip(hits, deferred) if not d]
    res['clicks'][m] = dict(n=len(R), hit_rate=round(float(np.mean(hits)), 4), status=status_counts(R),
                            after_zoom_alarm=dict(partner=partner[m], delta=delta, errors_caught=round(caught / max(1, sum(errs)), 3),
                                                  coverage=round(1 - float(np.mean(deferred)), 3), miss_rate_when_acting=round(1 - float(np.mean(acted)), 4) if acted else None),
                            cost_per_item_with_partner=round(float(np.mean([r.cost for r in R])), 5))
ca = json.load(open(f'{HOME}/gui/runs/main/click_analysis.json'))['summary']
res['clicks']['reference_hit_rate_B4'] = {m: ca[f'{m}|all_sspro']['B4'] for m in MODELS}
res['clicks']['reference_n'] = {m: ca[f'{m}|all_sspro']['n'] for m in MODELS}

# ------------------------------------------------------------------ 4. multi-view reports from a view planner (four crops)
PLAIN = copy.deepcopy(CFG); PLAIN['views']['sets_and_counts']['confirmation_rule'] = 'plain'
b2 = json.load(open(f'{HOME}/stageB/runs/b2/b2_analysis.json'))['cells']
def paired_ci(a, b, rng):
    a = np.array(a, float); b = np.array(b, float); bs = []
    for _ in range(4000):
        s = rng.integers(0, len(a), len(a)); bs.append((b[s].mean() - a[s].mean()) / a[s].mean())
    return dict(rel=round(float((b.mean() - a.mean()) / a.mean() * 100), 1), ci=[round(float(np.percentile(bs, 2.5) * 100), 1), round(float(np.percentile(bs, 97.5) * 100), 1)])
for kind in ('gui', 'sku'):
    ad = A.MultiViewBoxes(kind)
    for m in MODELS:
        rows = {k: [] for k in ('full', 'views_merged', 'confirm_coverage_aware', 'confirm_plain', 'second_call', 'unanimity3', 'deletion_only')}; dr = dc = 0
        for i in ad.ids:
            V = ad.views(i, m)
            if len(V) < 4: continue
            rf = run(ad, i, m, CFG, ReplayClient(ad))                                   # one whole-image call
            rv = run(ad, i, m, CFG, ReplayClient(ad), views=V)                          # the four views, merged
            rc = run(ad, i, m, CFG, ReplayClient(ad), confirm_with_views=V)             # whole image confirmed by the views (shipped rule)
            rp = run(ad, i, m, PLAIN, ReplayClient(ad), confirm_with_views=V)           # same, plain rule
            r2 = run(ad, i, m, CFG, ReplayClient(ad), calls_override=2)                 # confirmed by a second whole-image call
            r3 = run(ad, i, m, CFG, ReplayClient(ad), calls_override=3)                 # three-call unanimity
            if any(c.status == 'unavailable' and c.request.kind != 'followup' for r in (rf, rv, rc, r2, r3) for c in r.calls): continue
            for k, r in (('full', rf), ('views_merged', rv), ('confirm_coverage_aware', rc), ('confirm_plain', rp), ('second_call', r2), ('unanimity3', r3)):
                rows[k].append(ad.score(i, r.S))
            kept = ad.confirm_deletion_only(rf.S, rv.S); rows['deletion_only'].append(ad.score(i, kept))
            dr += ad.matched(i, rf.S) - ad.matched(i, kept); dc += len(rf.S) - len(kept)
        cell = {k: round(float(np.mean(v)), 2) for k, v in rows.items()}; cell['n'] = len(rows['full'])
        rng = np.random.default_rng(20261012)
        cell['paired'] = {f'{k}_vs_full': paired_ci(rows['full'], rows[k], rng) for k in ('views_merged', 'confirm_coverage_aware', 'confirm_plain', 'second_call', 'unanimity3', 'deletion_only')}
        cell['paired']['confirm_coverage_aware_vs_second_call'] = paired_ci(rows['second_call'], rows['confirm_coverage_aware'], rng)
        cell['paired']['confirm_coverage_aware_vs_unanimity3'] = paired_ci(rows['unanimity3'], rows['confirm_coverage_aware'], rng)
        cell['p1_unconfirmed'] = round(dr / dc, 3) if dc else None
        cell['selection_gain_per_image'] = round(float(np.mean(rows['full']) - np.mean(rows['deletion_only'])), 2)
        ref = b2[f'{kind}|{m}']; cell['reference_b2'] = dict(full_D1=ref['full_D1']['E'], crops_blind=ref['crops_blind']['E'], inter_D1_crops=ref['inter_D1_crops']['E'], inter_D1_D2=ref['inter_D1_D2']['E'])
        res['multiview'][f'{kind}|{m}'] = cell
        print(kind, m, {k: cell[k] for k in rows}, flush=True)

# ------------------------------------------------------------------ 5. boundary cases (synthetic responses through the same code)
ad = A.CountSCPDense(); i = 0; m = 'gpt61'
good = ad.index[(0, 'gpt61', 'lowest')]['raw_text']
cases = {
    'empty_response': {'raw_text': ''},
    'refusal_text': {'raw_text': 'I cannot help with that.'},
    'unparseable_partial_json': {'raw_text': good[: len(good) // 2], 'finish_reason': 'length'},
    'no_stated_counts': {'raw_text': json.dumps({'shapes': json.loads(good)['shapes']})},
    'degenerate_list_with_counts': {'raw_text': json.dumps({'shapes': [{'x': 10 * k, 'y': 10} for k in range(40)] + json.loads(good)['shapes'][:20],
                                                            'counts_by_type': json.loads(good)['counts_by_type']})},
    'degenerate_list_without_counts': {'raw_text': json.dumps({'shapes': [{'x': 10 * k, 'y': 10} for k in range(40)]})},
}
key = lambda kind: ('scp_dense_count', i, m, kind, 0, None)
for name, resp in cases.items():
    r = run(ad, i, m, CFG, ReplayClient(ad, {key('first'): resp}))
    res['boundary'][f'count:{name}'] = dict(y_hat_total=sum(r.y_hat.values()), branch=r.branch, status=r.status, calls=[c.request.kind + ':' + c.status for c in r.calls])
r = run(ad, i, m, CFG, ReplayClient(ad, {key('first'): 'transport_failed'}))
res['boundary']['count:transport_failure_then_resend'] = dict(branch=r.branch, status=r.status, calls=[c.request.kind + ':' + c.status for c in r.calls])
r = run(ad, i, m, CFG, ReplayClient(ad, {key('first'): 'transport_failed', key('resend'): 'transport_failed'}))
res['boundary']['count:transport_failure_twice'] = dict(y_hat_total=sum(r.y_hat.values()), branch=r.branch, status=r.status, calls=[c.request.kind + ':' + c.status for c in r.calls])
# sets: all three calls empty; one call unparseable
ad = A.SetE3('sku110k'); i = ad.images[0]
ov = {('e3_set', i, 'gpt61', 'first', k, None): {'raw_text': ''} for k in range(3)}
r = run(ad, i, 'gpt61', CFG, ReplayClient(ad, ov))
res['boundary']['set:all_calls_empty'] = dict(n_records=len(r.S), status=r.status, calls=[c.request.kind + ':' + c.status for c in r.calls])
r = run(ad, i, 'gpt61', CFG, ReplayClient(ad, {('e3_set', i, 'gpt61', 'first', 1, None): {'raw_text': '{"objects": [ {"x1": '}}))
res['boundary']['set:one_call_unparseable'] = dict(n_records=len(r.S), status=r.status, calls=[c.request.kind + ':' + c.status for c in r.calls])
# clicks: no first point; no point in the crop
ad = A.Click('sspro'); i = sorted(ad.items)[0]
r = run(ad, i, 'gem38', CFG, ReplayClient(ad, {('click', i, 'gem38', 'first', 0, None): {'raw_text': 'not found'}}), partner='sonnet55')
res['boundary']['click:no_first_point'] = dict(point=r.S, hit=ad.score(i, r.S), alarm=r.alarm, status=r.status, calls=[c.request.kind + ':' + c.status for c in r.calls])
c1 = ReplayClient(ad); p1 = run(ad, i, 'gem38', CFG, c1).calls[1].request.view
r = run(ad, i, 'gem38', CFG, ReplayClient(ad, {('click', i, 'gem38', 'zoom', 0, p1): {'raw_text': '{}'}}))
res['boundary']['click:no_point_in_crop'] = dict(point=r.S, status=r.status, calls=[c.request.kind + ':' + c.status for c in r.calls])

# ------------------------------------------------------------------ 6. apparent size of objects in the whole-image call
# tokens across the shorter object side, from each model's input geometry measured in the companion paper:
# 32 px tokens on the image as sent (GPT-6.1 calibration in the companion paper's release), 28 px for Sonnet 5.5, a fixed grid of about
# 1,090 tokens for Gemini 3.8. Images sent: Sonnet <= 1,568 px long side and 1.15 MP; others <= 2,048 px long side.
def sent_scale(m, W, H):
    return min(1.0, 1568 / max(W, H), (1_150_000 / (W * H)) ** 0.5) if m == 'sonnet55' else min(1.0, 2048 / max(W, H))
def tokens_across(m, side, W, H):
    if m == 'gem38': return side / (W * H / 1090) ** 0.5
    return side * sent_scale(m, W, H) / (32 if m == 'gpt61' else 28)
S_STAR = CFG['views']['sets_and_counts']['companion_paper_design_value_S_star']
size = {}
gts = [json.load(open(f'{HOME}/stageB/data/scpd_gen/ground_truth/gt_{i:03d}.json')) for i in range(50)]
for m in MODELS:
    S = [tokens_across(m, 2 * s['radius'], 1024, 1024) for g in gts for s in g['shapes']]
    size[f'scp_dense|{m}'] = dict(median=round(float(np.median(S)), 2), share_below_S_star=round(float(np.mean(np.array(S) < S_STAR)), 3))
for ds in ('doclaynet', 'dota', 'sku110k'):
    ad = A.SetE3(ds)
    for m in MODELS:
        S = []
        for i in ad.images:
            it = ad.items[i]; W, H = it['width'], it['height']
            S += [tokens_across(m, min(t['box'][2] - t['box'][0], t['box'][3] - t['box'][1]), W, H) for t in it['targets'] if t.get('box')]
        size[f'{ds}|{m}'] = dict(median=round(float(np.median(S)), 2), share_below_S_star=round(float(np.mean(np.array(S) < S_STAR)), 3))
res['apparent_size'] = size

json.dump(res, open(OUT / 'pipeline_replay.json', 'w'), indent=1, default=str)
print(json.dumps({k: v for k, v in res.items() if k in ('count', 'clicks', 'boundary')}, indent=1, default=str))
