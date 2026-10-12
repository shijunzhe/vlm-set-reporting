"""Generates the data figures of the FSR paper from saved responses, labels and analysis outputs. No model calls.
fig_example.pdf   Figure 3: SCP-dense image 28, GPT-6.1 list-first response (RC16): truth vs list vs stated count per type
fig_density.pdf   Figure 3: same-response readout gap across object density (analysis/out/readout_density.json)
fig_alarm.pdf     Figure 4: click alarm risk-coverage (analysis/out/alarm_curves.json)
fig_select.pdf    Figure 5: (a) deletion threshold check (intersection_decomp.json), (b) oracle vs selectors (prep1_v2.json)
Run: python3 -I make_figures.py  (from any directory)"""
import json, collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/claude/stageB')
OUT = ROOT / 'analysis' / 'out'
plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'legend.fontsize': 7,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
MODEL = {'gpt61': 'GPT-6.1', 'sonnet55': 'Sonnet 5.5', 'gem38': 'Gemini 3.8'}
COL = {'gpt61': '#1f77b4', 'sonnet55': '#d62728', 'gem38': '#2ca02c'}
T5 = ['circle', 'triangle', 'square', 'star', 'pentagon']


def fig_example():
    gt = json.load(open(ROOT / 'data/scpd_gen/ground_truth/gt_028.json'))
    resp = None
    for line in open(ROOT / 'runs/rc16/responses_main_gpt61.jsonl'):
        r = json.loads(line)
        if r['request_id'] == 'scpd|028|gpt61|v516|r0':
            resp = json.loads(r['raw_text'])
    assert resp is not None
    truth = collections.Counter(s['type'] for s in gt['shapes'])
    listed = collections.Counter(s['type'] for s in resp['shapes'])
    stated = resp['counts_by_type']
    img = Image.open(ROOT / 'data/scpd/028.png').convert('RGB')
    W, H = img.size
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(6.9, 2.75), gridspec_kw={'width_ratios': [1, 1.25]})
    a0.imshow(img)
    xs = [s['x'] / 1000 * W for s in resp['shapes']]; ys = [s['y'] / 1000 * H for s in resp['shapes']]
    a0.scatter(xs, ys, s=5, c='black', marker='x', linewidths=0.6)
    a0.set_xticks([]); a0.set_yticks([])
    for sp in a0.spines.values(): sp.set_visible(True)
    a0.set_title(f'SCP-dense image 28: {gt["total_shapes"]} shapes; x = records in GPT-6.1\'s list ({len(resp["shapes"])})')
    x = np.arange(len(T5)); w = 0.27
    a1.bar(x - w, [truth[t] for t in T5], w, label=f'truth (total {sum(truth.values())})', color='#bbbbbb', edgecolor='black', linewidth=0.4)
    a1.bar(x, [listed[t] for t in T5], w, label=f'list count (total {sum(listed.values())})', color='#1f77b4')
    a1.bar(x + w, [stated[t] for t in T5], w, label=f'stated count in the same response (total {sum(stated.values())})', color='#ff7f0e')
    a1.set_xticks(x); a1.set_xticklabels(T5); a1.set_ylabel('count'); a1.set_ylim(0, 56)
    el = sum(abs(listed[t] - truth[t]) for t in T5); es = sum(abs(stated[t] - truth[t]) for t in T5)
    a1.set_title(f'Per-type count error: list {el}, stated {es}')
    a1.legend(loc='upper left', frameon=False, ncol=1)
    fig.tight_layout(); fig.savefig(HERE / 'fig_example.pdf', bbox_inches='tight'); plt.close(fig)
    return dict(truth=dict(truth), listed=dict(listed), stated=stated, err_list=el, err_stated=es)


def fig_density():
    d = json.load(open(OUT / 'readout_density.json'))
    order = [('px', 'PixMo\n(6.0)'), ('fsc', 'FSC-147\n(43.8)'), ('scp', 'SCP\n(47.6)'), ('scpd', 'SCP-dense\n(164.3)')]
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.1), sharey=False)
    for ax, m in zip(axes, MODEL):
        st = [d[f'{t}|{m}']['stated'] for t, _ in order]; cd = [d[f'{t}|{m}']['code_raw'] for t, _ in order]
        x = np.arange(len(order)); w = 0.38
        ax.bar(x - w / 2, st, w, color='#ff7f0e', label='stated count')
        ax.bar(x + w / 2, cd, w, color='#1f77b4', label='list count')
        for i, (t, _) in enumerate(order):
            v = d[f'{t}|{m}']; lo, hi = v['ci_raw']; sig = lo > 0 or hi < 0
            ax.text(i, max(st[i], cd[i]) * 1.04 + 0.5, f"{v['rel_raw']:+.0f}%" + ('' if sig else '\nn.s.'), ha='center', va='bottom', fontsize=6)
        ax.set_xticks(x); ax.set_xticklabels([o[1] for o in order], fontsize=6)
        ax.set_title(MODEL[m]); ax.set_ylim(0, max(max(st), max(cd)) * 1.35 + 1)
    axes[0].set_ylabel('count error (same response)'); axes[0].legend(frameon=False, loc='upper left')
    fig.tight_layout(); fig.savefig(HERE / 'fig_density.pdf', bbox_inches='tight'); plt.close(fig)


def fig_alarm():
    d = json.load(open(OUT / 'alarm_curves.json'))
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.15))
    sty = {'same': ('same model, unzoomed', '#7f7f7f', 'o'), 'cross': ('cross model, unzoomed', '#1f77b4', 's'),
           'cross_after_zoom': ('cross model, after zoom', '#d62728', '^')}
    for ax, m in zip(axes, MODEL):
        v = d['models'][m]
        for k, (lab, c, mk) in sty.items():
            pts = sorted(v[k]['points'], key=lambda p: p['coverage'])
            ax.plot([p['coverage'] * 100 for p in pts], [p['risk_when_acting'] * 100 for p in pts], marker=mk, ms=3, color=c, lw=1, label=lab)
            base = 100 * (1 - v[k]['base_accuracy'])
            ax.scatter([100], [base], color=c, marker=mk, s=14, facecolors='none')
        ax.set_title(f'{MODEL[m]} (partner {MODEL[v["partner"]]})'); ax.set_xlabel('coverage (%)')
        ax.set_xlim(55, 101)
    axes[0].set_ylabel('miss rate when acting (%)'); axes[0].legend(frameon=False, loc='upper left', fontsize=6)
    fig.tight_layout(); fig.savefig(HERE / 'fig_alarm.pdf', bbox_inches='tight'); plt.close(fig)


def fig_select():
    dec = json.load(open(OUT / 'intersection_decomp.json')); pr = json.load(open(OUT / 'prep1_v2.json'))
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(6.9, 2.4), gridspec_kw={'width_ratios': [1, 1.5]})
    for k, v in dec.items():
        if not isinstance(v, dict) or 'p1_marginal' not in v: continue
        same = k.endswith('|same'); gui = k.startswith('GUI')
        a0.scatter(v['p1_marginal'], v['selection'], s=12, marker='o' if same else 's', facecolors='none' if gui else ('#555555' if same else '#1f77b4'),
                   edgecolors='#555555' if same else '#1f77b4', linewidths=0.7)
    a0.axvline(0.5, color='k', lw=0.6, ls='--'); a0.axhline(0, color='k', lw=0.5)
    a0.set_yscale('symlog', linthresh=1); a0.set_xlabel(r'marginal matching rate $p_1$ of deleted records'); a0.set_ylabel('selection gain in $E$ per image')
    a0.set_title('(a) Deletion helps iff $p_1 < 1/2$ (54/54 pairs)')
    from matplotlib.lines import Line2D
    a0.legend(handles=[Line2D([], [], marker='o', ls='', color='#555555', mfc='#555555', ms=4, label='same-model partner'),
                       Line2D([], [], marker='s', ls='', color='#1f77b4', mfc='#1f77b4', ms=4, label='cross-model partner'),
                       Line2D([], [], marker='s', ls='', color='#1f77b4', mfc='none', ms=4, label='open markers: screen sources')], frameon=False, fontsize=6, loc='lower left')
    cells = [k for k in pr if not k.startswith('_')]
    lab = [k.replace('|', '\n').replace('gpt61', 'GPT').replace('sonnet55', 'Son.').replace('gem38', 'Gem.').replace('doclaynet', 'DLN').replace('fsc147', 'FSC').replace('sku110k', 'SKU').replace('dota', 'DOTA') for k in cells]
    x = np.arange(len(cells)); w = 0.27
    a1.bar(x - w, [100 * pr[k]['oracle_constrained'] for k in cells], w, color='#bbbbbb', edgecolor='black', linewidth=0.4, label='constrained oracle')
    a1.bar(x, [100 * pr[k]['crossfit_oracle_logit'] for k in cells], w, color='#1f77b4', label='label-fitted selector (cross-fit)')
    a1.bar(x + w, [100 * pr[k]['labels10_oracle_logit_median'] for k in cells], w, color='#ff7f0e', label='selector fitted on 10 labels')
    a1.axhline(0, color='k', lw=0.5); a1.set_xticks(x); a1.set_xticklabels(lab, fontsize=5.5)
    a1.set_ylabel('reduction of $E$ vs one call (%)'); a1.set_ylim(-12, 100); a1.legend(frameon=False, fontsize=6, loc='upper center', ncol=3)
    a1.set_title('(b) Selection among three calls of one model (E3)')
    fig.tight_layout(); fig.savefig(HERE / 'fig_select.pdf', bbox_inches='tight'); plt.close(fig)


def _click_example(cid='sspro_photoshop_windows_39', m='gem38'):
    import sys as _s
    _s.path.insert(0, '/home/claude/gui/code'); import fsrgui as G
    GUI = '/home/claude/gui'
    c = [x for x in json.load(open(f'{GUI}/data/click_items_prepped.json')) if x['id'] == cid][0]
    QA = {}; QB = {}
    for l in open(f'{GUI}/runs/main/requests_A.jsonl'):
        q = json.loads(l)
        if q['request_id'].startswith(cid + '|' + m): QA[q['request_id']] = q
    for l in open(f'{GUI}/runs/main/requests_B.jsonl'):
        q = json.loads(l)
        if q['request_id'].startswith(cid + '|' + m): QB[q['request_id']] = q
    RA = {}; RB = {}
    for l in open(f'{GUI}/runs/main/responses_A_{m}.jsonl'):
        r = json.loads(l)
        if r['request_id'] in QA: RA[r['request_id']] = r
    for l in open(f'{GUI}/runs/main/responses_B_{m}.jsonl'):
        r = json.loads(l)
        if r['request_id'] in QB: RB[r['request_id']] = r
    scc = lambda q: None if q['meta']['mode'] == 'grid' else tuple(q['meta']['sizes'][0])
    k = lambda s: f'{cid}|{m}|{s}'
    p1 = G.decode_click(RA[k('C1')]['raw_text'], scc(QA[k('C1')]))[0]
    q4 = QB[k('B4')]; pt, _ = G.decode_click(RB[k('B4')]['raw_text'], scc(q4)); w = q4['meta']['window']
    p4 = [w[0] + pt[0] * (w[2] - w[0]), w[1] + pt[1] * (w[3] - w[1])]
    return c, p1, p4, w


def fig_teaser():
    gt = json.load(open(ROOT / 'data/scpd_gen/ground_truth/gt_028.json'))
    resp = None
    for line in open(ROOT / 'runs/rc16/responses_main_gpt61.jsonl'):
        r = json.loads(line)
        if r['request_id'] == 'scpd|028|gpt61|v516|r0': resp = json.loads(r['raw_text'])
    truth = collections.Counter(s['type'] for s in gt['shapes']); listed = collections.Counter(s['type'] for s in resp['shapes']); stated = resp['counts_by_type']
    fig = plt.figure(figsize=(6.9, 2.15))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.25, 0.95], wspace=0.3)
    a1 = fig.add_subplot(gs[0]); a2 = fig.add_subplot(gs[1]); a3 = fig.add_subplot(gs[2])
    x = np.arange(len(T5)); w = 0.27
    a1.bar(x - w, [truth[t] for t in T5], w, color='#bbbbbb', edgecolor='black', linewidth=0.3, label=f'truth: {sum(truth.values())}')
    a1.bar(x, [listed[t] for t in T5], w, color='#1f77b4', label=f'list count: {sum(listed.values())}')
    a1.bar(x + w, [stated[t] for t in T5], w, color='#ff7f0e', label=f'stated count: {sum(stated.values())}')
    a1.set_xticks(x); a1.set_xticklabels(['circ.', 'tri.', 'sq.', 'star', 'pent.'], fontsize=6); a1.set_ylim(0, 62); a1.set_ylabel('count', fontsize=6)
    a1.legend(frameon=False, fontsize=5.5, loc='upper left', ncol=1); a1.set_title('(a) Repair: count the list', fontsize=7, loc='left')
    c, p1, p4, win = _click_example()
    im = Image.open('/home/claude/gui/' + c['image']).convert('RGB'); Wc, Hc = im.size
    bb = c['bbox_norm']; cx, cy = (bb[0] + bb[2]) / 2 * Wc, (bb[1] + bb[3]) / 2 * Hc; r = 40
    x0 = max(0, int(cx - 2 * r)); y0 = max(0, int(cy - r)); x1 = x0 + 3 * r; y1 = y0 + 2 * r
    crop = im.crop((x0, y0, x1, y1))
    a2.imshow(crop, extent=(x0, x1, y1, y0))
    from matplotlib.patches import Rectangle
    a2.add_patch(Rectangle((bb[0] * Wc, bb[1] * Hc), (bb[2] - bb[0]) * Wc, (bb[3] - bb[1]) * Hc, fill=False, ec='#2ca02c', lw=1.0))
    a2.scatter([p1[0] * Wc], [p1[1] * Hc], marker='x', c='#d62728', s=25, linewidths=1.2, label='one click (miss)')
    a2.scatter([p4[0] * Wc], [p4[1] * Hc], marker='o', facecolors='none', edgecolors='#1f77b4', s=25, linewidths=1.2, label='after zoom (hit)')
    a2.set_xlim(x0, x1); a2.set_ylim(y1, y0); a2.set_xticks([]); a2.set_yticks([])
    a2.legend(frameon=False, fontsize=5.5, loc='upper center', bbox_to_anchor=(0.5, -0.04), ncol=2); a2.set_title('(b) Repair: zoom in', fontsize=7, loc='left')
    al = json.load(open(OUT / 'alarm_curves.json')); xs = np.arange(3); vals_s = []; vals_c = []
    for mm in MODEL:
        v = al['models'][mm]; g = lambda kk: [p for p in v[kk]['points'] if abs(p['tau'] - 0.015) < 1e-9][0]['errors_caught'] * 100
        vals_s.append(g('same')); vals_c.append(g('cross'))
    a3.bar(xs - 0.2, vals_s, 0.4, color='#7f7f7f', label='2nd call, same model'); a3.bar(xs + 0.2, vals_c, 0.4, color='#1f77b4', label='2nd model')
    for i in range(3):
        a3.text(xs[i] - 0.2, vals_s[i] + 1.5, f'{vals_s[i]:.0f}', ha='center', fontsize=5.5); a3.text(xs[i] + 0.2, vals_c[i] + 1.5, f'{vals_c[i]:.0f}', ha='center', fontsize=5.5)
    a3.set_xticks(xs); a3.set_xticklabels(['GPT', 'Sonnet', 'Gemini'], fontsize=6); a3.set_ylim(0, 95); a3.set_ylabel('errors caught (%)', fontsize=6)
    a3.legend(frameon=False, fontsize=5.5, loc='upper left'); a3.set_title('(c) Detect: a second model', fontsize=7, loc='left')
    fig.savefig(HERE / 'fig_teaser.pdf', bbox_inches='tight'); plt.close(fig)
    return dict(click=c['id'], p1=p1, p4=p4)


if __name__ == '__main__':
    info = fig_example(); print('example', info)
    try:
        print('teaser', fig_teaser())
    except FileNotFoundError as e:
        print('teaser kept as shipped (needs the ScreenSpot-Pro screenshot):', e)
    fig_density(); fig_alarm(); fig_select(); print('ok')
