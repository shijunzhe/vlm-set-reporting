"""Generates the data figures of the FSR paper from saved responses, labels and analysis outputs. No model calls.
fig_teaser.pdf    Figure 1: (a) count readout, (b) zoom example, (c) 600-item click accuracy against cost (accfirst outputs)
fig_example.pdf   Figure 3: SCP-dense image 28, GPT-6.1 list-first response (RC16): truth vs list vs stated count per type
fig_density.pdf   Figure 4: same-response readout change across object density (analysis/out/readout_density.json)
fig_select.pdf    Figure 5: (a) where the base/partner pairs fall relative to p1 = 1/2 (intersection_decomp.json),
                  (b) oracle vs selectors (prep1_v2.json)
fig_alarm.pdf     Figure 6: click alarm risk-coverage, same and cross-model, before and after zoom (analysis/out/alarm_curves.json)
Every figure is drawn at the text width of the paper (6.3 in), so LaTeX does not rescale it, with one type scale:
8 pt for axis labels and panel titles, 7 pt for ticks, legends and annotations.
Run: python3 -I make_figures.py  (from any directory)"""
import json, collections
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/claude/stageB')
OUT = ROOT / 'analysis' / 'out'
W = 6.3                                     # text width of the paper, inches
SANS = ['Liberation Sans', 'Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': SANS, 'mathtext.fontset': 'custom',
                     'mathtext.rm': SANS[0], 'mathtext.it': SANS[0] + ':italic', 'mathtext.bf': SANS[0] + ':bold',
                     'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'legend.fontsize': 7,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
                     'ytick.major.width': 0.6, 'lines.linewidth': 1.0, 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})
MODEL = {'gpt61': 'GPT-6.1', 'sonnet55': 'Sonnet 5.5', 'gem38': 'Gemini 3.8'}
T5 = ['circle', 'triangle', 'square', 'star', 'pentagon']
# one role per colour, in every figure
C_LIST = '#1f5a96'      # list count; FSR and the cross-model partner (the recommended choice)
C_STATED = '#e07b00'    # stated count only
C_TRUTH = '#c8c8c8'     # ground truth
C_BASE = '#7a7a7a'      # baselines; same-model second call
C_SEL1 = '#1b9e77'      # label-fitted selectors
C_SEL2 = '#9fd8c4'
C_MISS = '#b2182b'      # a miss
C_HIT = '#1f5a96'       # a hit
C_TARGET = '#2e9e44'    # target box


def half_up(x, nd=1):
    q = Decimal(1).scaleb(-nd)
    return str(Decimal(str(x)).quantize(q, rounding=ROUND_HALF_UP))


def signed(x, nd=1):
    s = half_up(x, nd)
    return ('+' + s if not s.startswith('-') else '−' + s[1:]) if float(s) != 0 else s


def panel_titles(fig, axes, titles, pad=0.012):
    """Panel labels on one baseline, left-aligned with each panel."""
    fig.canvas.draw()
    top = max(ax.get_position().y1 for ax in axes)
    for ax, t in zip(axes, titles):
        fig.text(ax.get_position().x0, top + pad, t, ha='left', va='bottom', fontsize=8)


def save(fig, name):
    fig.savefig(HERE / name, dpi=600)
    plt.close(fig)


def _scp_response():
    gt = json.load(open(ROOT / 'data/scpd_gen/ground_truth/gt_028.json'))
    resp = None
    for line in open(ROOT / 'runs/rc16/responses_main_gpt61.jsonl'):
        r = json.loads(line)
        if r['request_id'] == 'scpd|028|gpt61|v516|r0':
            resp = json.loads(r['raw_text'])
    assert resp is not None
    return gt, resp


def fig_example():
    gt, resp = _scp_response()
    truth = collections.Counter(s['type'] for s in gt['shapes'])
    listed = collections.Counter(s['type'] for s in resp['shapes'])
    stated = resp['counts_by_type']
    img = Image.open(ROOT / 'data/scpd/028.png').convert('RGB')
    Wi, Hi = img.size
    CC = {'circle': '#6a3d9a', 'triangle': '#2e9e44', 'square': '#8c564b', 'star': '#d6604d', 'pentagon': '#1f9fbf'}
    X0, Y0, X1, Y1 = 0, 0, 420, 420                      # crop shown in panel (b), in image pixels
    fig = plt.figure(figsize=(W, 2.75))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.42], wspace=0.08, left=0.01, right=0.99, bottom=0.15, top=0.88)
    a0, a1, a2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    a0.imshow(img, interpolation='none'); a0.set_xticks([]); a0.set_yticks([])
    for sp in a0.spines.values(): sp.set_visible(True); sp.set_linewidth(0.6)
    a0.add_patch(Rectangle((X0, Y0), X1 - X0, Y1 - Y0, fill=False, ec=C_MISS, lw=1.0))
    a1.imshow(img.crop((X0, Y0, X1, Y1)), interpolation='none'); a1.set_xticks([]); a1.set_yticks([])
    for sp in a1.spines.values(): sp.set_visible(True); sp.set_edgecolor(C_MISS); sp.set_linewidth(0.8)
    for t in T5:
        pts = [(sh['x'] / 1000 * Wi - X0, sh['y'] / 1000 * Hi - Y0) for sh in resp['shapes'] if sh['type'] == t]
        pts = [(u, v) for u, v in pts if 0 <= u < X1 - X0 and 0 <= v < Y1 - Y0]
        a1.scatter([u for u, _ in pts], [v for _, v in pts], s=10, color=CC[t], marker='o', linewidths=0.3, edgecolors='white', label=t)
    a1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=3, handletextpad=0.1, columnspacing=0.6, fontsize=6.5, markerscale=1.1)
    a2.axis('off')
    first, last = resp['shapes'][0], resp['shapes'][-1]
    js = lambda sh: '{"type":"%s","x":%d,"y":%d}' % (sh['type'], sh['x'], sh['y'])
    cb = resp['counts_by_type']
    txt = ('{"shapes": [\n  ' + js(first) + ',\n  ... %d more records ...,\n  ' % (len(resp['shapes']) - 2) + js(last) + '],\n'
           ' "counts_by_type": {\n  ' + ', '.join(f'"{t}":{cb[t]}' for t in T5[:3]) + ',\n  ' + ', '.join(f'"{t}":{cb[t]}' for t in T5[3:]) + '}}')
    a2.text(0.02, 0.97, txt, family='monospace', fontsize=6.3, va='top', ha='left', transform=a2.transAxes,
            bbox=dict(boxstyle='round,pad=0.4', fc='#f5f5f5', ec='#aaaaaa', lw=0.5))
    el = sum(abs(listed[t] - truth[t]) for t in T5); es = sum(abs(stated[t] - truth[t]) for t in T5)
    rows = [['', 'truth', 'list count', 'stated count']] + [[t, str(truth[t]), str(listed[t]), str(stated[t])] for t in T5] + \
           [['total', str(sum(truth.values())), str(sum(listed.values())), str(sum(stated.values()))],
            ['per-type error', '', str(el), str(es)]]
    tb = a2.table(cellText=rows, cellLoc='center', colLoc='center', bbox=[0.02, -0.08, 0.96, 0.56], edges='horizontal', colWidths=[0.3, 0.2, 0.24, 0.26])
    tb.auto_set_font_size(False); tb.set_fontsize(7)
    for (r, c), cell in tb.get_celld().items():
        cell.set_linewidth(0.4)
        if c == 2 and r > 0: cell.get_text().set_color(C_LIST)
        if c == 3 and r > 0: cell.get_text().set_color(C_STATED)
        if r == 0: cell.get_text().set_fontweight('bold')
    panel_titles(fig, [a0, a1, a2], [f'(a) Scene, {gt["total_shapes"]} shapes', '(b) Listed records in the red box', '(c) One response: list, then counts'])
    save(fig, 'fig_example.pdf')
    return dict(truth=dict(truth), listed=dict(listed), stated=stated, err_list=el, err_stated=es)


def fig_density():
    d = json.load(open(OUT / 'readout_density.json'))
    order = [('px', 'PixMo'), ('fsc', 'FSC-147'), ('scp', 'SCP'), ('scpd', 'SCP-dense')]
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.15), sharey=True, gridspec_kw=dict(left=0.1, right=0.995, bottom=0.2, top=0.86, wspace=0.08))
    for ax, m in zip(axes, MODEL):
        x = np.arange(len(order))
        for i, (t, _) in enumerate(order):
            v = d[f'{t}|{m}']; lo, hi = v['ci_raw']; sig = lo > 0 or hi < 0; r = v['rel_raw']
            col = (C_LIST if r < 0 else C_STATED) if sig else '#d9d9d9'
            ax.bar(i, r, 0.62, color=col, edgecolor='none')
            ax.errorbar(i, r, yerr=[[r - lo], [hi - r]], fmt='none', ecolor='#333333', elinewidth=0.6, capsize=1.8)
            y = hi + 9 if r >= 0 else lo - 12
            ax.text(i, y, signed(r) + '%' + ('' if sig else '\nn.s.'), ha='center', va='bottom' if r >= 0 else 'top', fontsize=6.5)
        ax.axhline(0, color='#333333', lw=0.6)
        ax.set_xticks(x); ax.set_xticklabels([f"{lab}\n{d[f'{t}|{m}']['mean_objects']}" for t, lab in order])
        ax.set_ylim(-140, 290)
    axes[0].set_ylabel('list vs. stated count:\nchange of error (%)')
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:+.0f}'.replace('-', '−') if v else '0'))
    panel_titles(fig, axes, [f'({c}) {MODEL[m]}' for c, m in zip('abc', MODEL)])
    save(fig, 'fig_density.pdf')


def fig_alarm():
    d = json.load(open(OUT / 'alarm_curves.json'))
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.35), gridspec_kw=dict(left=0.075, right=0.995, bottom=0.31, top=0.88, wspace=0.28))
    sty = {'same': ('same model, unzoomed', C_BASE, 'o', '-', True), 'cross': ('cross-model, unzoomed', C_LIST, 'o', '-', True),
           'same_after_zoom': ('same model, after zoom', C_BASE, 's', '--', False), 'cross_after_zoom': ('cross-model, after zoom', C_LIST, 's', '--', False)}
    for ax, m in zip(axes, MODEL):
        v = d['models'][m]
        for k, (lab, c, mk, ls, filled) in sty.items():
            pts = sorted(v[k]['points'], key=lambda p: p['coverage'])
            ax.plot([p['coverage'] * 100 for p in pts], [p['risk_when_acting'] * 100 for p in pts], marker=mk, ms=3.2, color=c, lw=1, ls=ls,
                    mfc=c if filled else 'white', mew=0.8, label=lab)
            p0 = [p for p in pts if abs(p['tau'] - 0.015) < 1e-9][0]                       # shipped default delta
            ax.scatter([p0['coverage'] * 100], [p0['risk_when_acting'] * 100], s=46, facecolors='none', edgecolors='black', linewidths=0.7, zorder=4)
        for k, filled in (('same', True), ('same_after_zoom', False)):                       # acting on every item
            base = 100 * (1 - v[k]['base_accuracy'])
            ax.scatter([100], [base], marker='D', s=16, facecolors='black' if filled else 'white', edgecolors='black', linewidths=0.8, zorder=5)
        ax.set_xlabel('acceptance coverage (%)'); ax.set_xlim(55, 102)
    axes[0].set_ylabel('miss rate when acting (%)')
    handles = [Line2D([], [], color=C_BASE, marker='o', ms=3.2, label='same model, unzoomed'), Line2D([], [], color=C_LIST, marker='o', ms=3.2, label='cross-model, unzoomed'),
               Line2D([], [], color=C_BASE, marker='s', ms=3.2, ls='--', mfc='white', label='same model, after zoom'),
               Line2D([], [], color=C_LIST, marker='s', ms=3.2, ls='--', mfc='white', label='cross-model, after zoom'),
               Line2D([], [], color='black', marker='o', ms=6, ls='', mfc='none', mew=0.7, label=r'$\delta=1.5\%$ (default)'),
               Line2D([], [], color='black', marker='D', ms=3.6, ls='', label='act on every item (filled: unzoomed)')]
    fig.legend(handles=handles, loc='lower center', ncol=3, bbox_to_anchor=(0.53, 0.0), columnspacing=1.2, handlelength=2.2)
    panel_titles(fig, axes, [f'({c}) {MODEL[m]}' for c, m in zip('abc', MODEL)])
    save(fig, 'fig_alarm.pdf')


def fig_select():
    dec = json.load(open(OUT / 'intersection_decomp.json')); pr = json.load(open(OUT / 'prep1_v2.json'))
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(W, 2.4), gridspec_kw=dict(width_ratios=[1, 1.45], left=0.085, right=0.995, bottom=0.21, top=0.88, wspace=0.26))
    for k, v in dec.items():
        if not isinstance(v, dict) or 'p1_marginal' not in v: continue
        same = k.endswith('|same'); gui = k.startswith('GUI')
        c = C_BASE if same else C_LIST
        a0.scatter(v['p1_marginal'], v['selection'], s=13, marker='o' if same else 's', facecolors='white' if gui else c, edgecolors=c, linewidths=0.7, zorder=3)
    a0.axvline(0.5, color='black', lw=0.6, ls='--'); a0.axhline(0, color='black', lw=0.5)
    ys = [v['selection'] for v in dec.values() if isinstance(v, dict) and 'p1_marginal' in v]
    a0.set_yscale('symlog', linthresh=1); a0.set_ylim(min(ys) * 1.6, max(ys) * 1.6)
    a0.set_yticks([-10, -1, 0, 1, 10, 100])
    a0.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'.replace('-', '−')))
    a0.set_xlabel(r'marginal matching rate $p_1$ of deleted records'); a0.set_ylabel(r'selection gain (reduction of $E$ per image)')
    a0.text(0.51, 0.99, r'$p_1=1/2$', transform=a0.get_xaxis_transform(), fontsize=6.5, ha='left', va='top')
    a0.legend(handles=[Line2D([], [], marker='o', ls='', color=C_BASE, ms=4, label='same-model partner'),
                       Line2D([], [], marker='s', ls='', color=C_LIST, ms=4, label='cross-model partner'),
                       Line2D([], [], marker='s', ls='', color=C_LIST, mfc='white', ms=4, label='open: screen sources')],
              loc='lower left', ncol=1, handletextpad=0.2, fontsize=6.5, borderaxespad=0.2)
    cells = [k for k in pr if not k.startswith('_')]
    lab = [k.split('|')[1].replace('gpt61', 'GPT').replace('sonnet55', 'Son.').replace('gem38', 'Gem.') for k in cells]
    dsn = {'doclaynet': 'DocLayNet', 'dota': 'DOTA', 'fsc147': 'FSC-147', 'sku110k': 'SKU-110K'}
    x = np.arange(len(cells)); w = 0.27
    a1.bar(x - w, [100 * pr[k]['oracle_constrained'] for k in cells], w, color='white', edgecolor='#444444', linewidth=0.6, label='constrained oracle')
    a1.bar(x, [100 * pr[k]['crossfit_oracle_logit'] for k in cells], w, color=C_SEL1, label='cross-fitted logistic selector')
    a1.bar(x + w, [100 * pr[k]['labels10_oracle_logit_median'] for k in cells], w, color=C_SEL2, label='logistic selector, 10 labelled images')
    a1.axhline(0, color='black', lw=0.5); a1.set_xticks(x); a1.set_xticklabels(lab, fontsize=6.5)
    for g in range(0, len(cells), 3):
        a1.text(g + 1, -0.16, dsn[cells[g].split('|')[0]], transform=a1.get_xaxis_transform(), ha='center', va='top', fontsize=7)
        if g: a1.axvline(g - 0.5, color='#cccccc', lw=0.5)
    a1.set_ylabel(r'reduction of $E$ vs. one call (%)'); a1.set_ylim(-12, 120); a1.set_yticks([0, 20, 40, 60, 80, 100])
    a1.legend(loc='upper right', ncol=1, fontsize=6.5)
    panel_titles(fig, [a0, a1], [r'(a) Where the 54 pairs fall relative to $p_1=1/2$', '(b) Selection among three calls of one model'])
    save(fig, 'fig_select.pdf')


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
    gt, resp = _scp_response()
    truth = collections.Counter(s['type'] for s in gt['shapes']); listed = collections.Counter(s['type'] for s in resp['shapes']); stated = resp['counts_by_type']
    fig = plt.figure(figsize=(W, 2.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.1, 1.05], wspace=0.3, left=0.065, right=0.985, bottom=0.2, top=0.86)
    a1 = fig.add_subplot(gs[0]); a2 = fig.add_subplot(gs[1]); a3 = fig.add_subplot(gs[2])
    x = np.arange(len(T5)); w = 0.27
    a1.bar(x - w, [truth[t] for t in T5], w, color=C_TRUTH, label=f'truth ({sum(truth.values())})')
    a1.bar(x, [listed[t] for t in T5], w, color=C_LIST, label=f'list count ({sum(listed.values())})')
    a1.bar(x + w, [stated[t] for t in T5], w, color=C_STATED, label=f'stated count ({sum(stated.values())})')
    a1.set_xticks(x); a1.set_xticklabels(['circle', 'triangle', 'square', 'star', 'pentagon'], rotation=30, ha='right', rotation_mode='anchor')
    a1.set_ylim(0, 66); a1.set_ylabel('objects per class')
    a1.legend(loc='upper left', ncol=1, handlelength=1.2, borderaxespad=0.1)
    c, p1, p4, win = _click_example('sspro_excel_macos_14')
    im = Image.open('/home/claude/gui/' + c['image']).convert('RGB'); Wc, Hc = im.size
    bb = c['bbox_norm']
    xs = [p1[0] * Wc, p4[0] * Wc, bb[0] * Wc, bb[2] * Wc]; ys = [p1[1] * Hc, p4[1] * Hc, bb[1] * Hc, bb[3] * Hc]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    hw = max(max(xs) - min(xs), (max(ys) - min(ys)) * 1.5) * 0.8 + 30; hh = hw / 1.5
    x0 = max(0, int(cx - hw)); y0 = max(0, int(cy - hh)); x1 = min(Wc, int(x0 + 2 * hw)); y1 = min(Hc, int(y0 + 2 * hh))
    a2.imshow(im.crop((x0, y0, x1, y1)), extent=(x0, x1, y1, y0), interpolation='none')
    a2.add_patch(Rectangle((bb[0] * Wc, bb[1] * Hc), (bb[2] - bb[0]) * Wc, (bb[3] - bb[1]) * Hc, fill=False, ec=C_TARGET, lw=1.2))
    a2.scatter([p1[0] * Wc], [p1[1] * Hc], marker='x', c=C_MISS, s=34, linewidths=1.5, label='one click (miss)', zorder=4)
    a2.scatter([p4[0] * Wc], [p4[1] * Hc], marker='o', facecolors='none', edgecolors=C_HIT, s=34, linewidths=1.5, label='after zoom (hit)', zorder=4)
    a2.set_xlim(x0, x1); a2.set_ylim(y1, y0); a2.set_xticks([]); a2.set_yticks([])
    for sp in a2.spines.values(): sp.set_visible(True); sp.set_linewidth(0.6)
    a2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=2, handletextpad=0.2, columnspacing=0.8)
    AC = ROOT / 'analysis' / 'accfirst'
    ev = json.load(open(AC / 'conf_eval_conf_v1.json'))['flows']; dep = json.load(open(AC / 'deploy_costs_conf_v1.json'))
    ev = dict(ev, **{'vote-3': dict(errors=json.load(open(AC / 'conf_vote_conf_v1.json'))['vote']['errors'])})
    pts = [('GPT-default-zoom', 'GPT-6.1 zoom', C_BASE, (5, -3)), ('GPTx3', 'GPT-6.1\nzoom ×3, vote', C_BASE, (-20, 6)),
           ('GLF', 'crop-first', C_BASE, (5, -3)), ('vote-3', 'three models,\ncluster vote', C_BASE, (-66, 1)),
           ('D-C', 'FSR GUI-grounding\ntier', C_LIST, (-70, -6))]
    for k, lab, col, off in pts:
        xx, yy = dep[k]['usd_per_item'] * 100, ev[k]['errors'] / 6
        a3.scatter([xx], [yy], s=24, color=col, zorder=3)
        arrow = dict(arrowstyle='-', color='#999999', lw=0.5, shrinkA=0, shrinkB=3) if k == 'vote-3' else None
        a3.annotate(lab, (xx, yy), xytext=off, textcoords='offset points', fontsize=7, color='black' if col == C_BASE else C_LIST, arrowprops=arrow)
    a3.set_xlim(0.5, 3.1); a3.set_ylim(8, 15); a3.set_xlabel('cost per item (US cents)'); a3.set_ylabel('miss rate (%)')
    panel_titles(fig, [a1, a2, a3], ['(a) Repair the readout: count the list', '(b) Repair the view: zoom in', '(c) Choose among models'])
    save(fig, 'fig_teaser.pdf')
    return dict(click=c['id'], p1=p1, p4=p4)


if __name__ == '__main__':
    info = fig_example(); print('example', info)
    try:
        print('teaser', fig_teaser())
    except FileNotFoundError as e:
        print('teaser kept as shipped (needs the ScreenSpot-Pro screenshot):', e)
    fig_density(); fig_alarm(); fig_select(); print('ok')
