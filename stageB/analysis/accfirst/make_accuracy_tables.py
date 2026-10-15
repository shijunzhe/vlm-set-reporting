"""LaTeX tables for the click-choice confirmation (600 held-out ScreenSpot-Pro items) and the E3 leave-one-source-out
configuration choice. Reads only saved outputs and saved responses; no model calls.
Inputs: accfirst/conf_eval_conf_v1.json, conf_singles_conf_v1.json, deploy costs from gui/runs/conf_v1, out/e3_loso_config.json.
The paper's table directory is FSR_TEX/tables (environment variable; reproduce.sh sets it to paper/).
Writes: paper tables t9_choose.tex, t10_config.tex and accfirst/deploy_costs_conf_v1.json."""
import collections, glob, json, os, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; AN = HERE.parent
_repo_paper = HERE.parents[2] / 'paper'
TEX = Path(os.environ.get('FSR_TEX') or (_repo_paper if (_repo_paper / 'main.tex').exists() else '/home/claude/stageB/paper/tex')) / 'tables'
sys.path.insert(0, '/home/claude/fsr_v51/e3'); from score_e3 import PRICE
RUN = '/home/claude/gui/runs/conf_v1'
ev = json.load(open(HERE / 'conf_eval_conf_v1.json'))
items = sorted({json.loads(l)['request_id'].split('|')[0] for l in open(f'{RUN}/responses_A_gpt61.jsonl')})
assert len(items) == 600
calls = collections.defaultdict(lambda: collections.defaultdict(float)); usd = collections.defaultdict(lambda: collections.defaultdict(float))
cost = collections.defaultdict(dict)
for f in glob.glob(f'{RUN}/responses_*_*.jsonl'):
    st = os.path.basename(f)[len('responses_'):].rsplit('_', 1)[0]
    for l in open(f):
        r = json.loads(l)
        if r['status'] != 'completed': continue
        m = r['request_id'].split('|')[1]; u = r['usage']
        cost[st][r['request_id']] = (u['input_tokens'] * PRICE[m][0] + u['output_tokens'] * PRICE[m][1]) / 1e6
ALL = ('gpt61', 'sonnet55', 'gem38')
def parts(st, models, tags=None):
    return [(st, m, tags) for m in models]
DEPLOY = {
    'GPT-min-zoom': [('A', 'gpt61', ['C1']), ('B', 'gpt61', ['B4'])],
    'GPT-default-zoom': [('DA', 'gpt61', ['C1d']), ('DB', 'gpt61', ['B4d'])],
    'Sonnet-min-zoom': [('A', 'sonnet55', ['C1']), ('B', 'sonnet55', ['B4'])],
    'Gemini-min-zoom': [('A', 'gem38', ['C1']), ('B', 'gem38', ['B4'])],
    'Sonnet-default-zoom': [('DA', 'sonnet55', ['C1d']), ('DB', 'sonnet55', ['B4d'])],
    'GPTx3': [('DA', 'gpt61', ['C1d', 'C1d1', 'C1d2']), ('DB', 'gpt61', ['B4d', 'B4d1', 'B4d2'])],
    'GLF': [('GLF1', 'gpt61', None), ('GLF2', 'gpt61', None), ('GLF3', 'gpt61', None)],
    'M-S': [('A', m, ['C1']) for m in ALL] + [('B', m, ['B4']) for m in ALL] + [('S', 'gem38', ['S|' + m for m in ALL])],
    'D-S': [('A', 'gem38', ['C1']), ('B', 'gem38', ['B4'])] + [('DA', m, ['C1d']) for m in ('gpt61', 'sonnet55')] +
           [('DB', m, ['B4d']) for m in ('gpt61', 'sonnet55')] + [('S', 'gem38', ['S|gem38']), ('SD', 'gem38', None)]}
DEPLOY['M-C'] = DEPLOY['M-S'] + [('CMPA_M', 'gem38', None)]; DEPLOY['D-C'] = DEPLOY['D-S'] + [('CMPA_D', 'gem38', None)]
dep = {}
for fl, P in DEPLOY.items():
    pu = collections.defaultdict(float); pc = collections.defaultdict(int)
    for st, m, tags in P:
        for rid, c in cost.get(st, {}).items():
            p = rid.split('|')
            if p[1] != m: continue
            if tags is not None and p[2] not in tags and '|'.join(p[2:4]) not in tags: continue
            pu[p[0]] += c; pc[p[0]] += 1
    dep[fl] = dict(usd_per_item=round(float(np.mean([pu[i] for i in items])), 5), calls_per_item=round(float(np.mean([pc[i] for i in items])), 2))
json.dump(dep, open(HERE / 'deploy_costs_conf_v1.json', 'w'), indent=1)
print(json.dumps(dep, indent=1))

# ------------------------------------------------------------------ Table: click choice on 600 held-out items
cmp = {}
for x in [ev['H1']] + ev['H1b'] + ev['H2'] + ev['H3']: cmp[(x['base'], x['flow'])] = x
fmt_ci = lambda c: f'$[{c[0]:.2f}, {c[1]:.2f}]$'
NAME = [('GPT-min-zoom', 'GPT-6.1, lowest, two-stage zoom'), ('Sonnet-min-zoom', 'Sonnet 5.5, lowest, two-stage zoom'), ('Gemini-min-zoom', 'Gemini 3.8, lowest, two-stage zoom'),
        ('Sonnet-default-zoom', 'Sonnet 5.5, default, two-stage zoom'), ('GPT-default-zoom', 'GPT-6.1, default, two-stage zoom (reference)'),
        ('GPTx3', 'GPT-6.1, three default zooms, cluster vote'), ('GLF', 'Crop-first baseline, GUI-Lens style (ours)'),
        ('M-S', 'Three models, lowest; judge score'), ('M-C', 'Three models, lowest; score and comparison'),
        ('D-S', 'Three models, D settings; judge score'), ('D-C', 'Three models, D settings; score, comparison (Alg.~\\ref{alg:clicks})')]
sing = json.load(open(HERE / 'conf_singles_conf_v1.json'))
err = {k: v['errors'] for k, v in sing.items()}; nopt = {k: v['no_point'] for k, v in sing.items()}
for k, v in ev['flows'].items(): assert err[k] == v['errors'] and nopt[k] == v['no_output'], k
assert err['Sonnet-min-zoom'] == ev['flows']['Sonnet-default-zoom']['errors'] + cmp[('Sonnet-min-zoom', 'Sonnet-default-zoom')]['fixed'] - cmp[('Sonnet-min-zoom', 'Sonnet-default-zoom')]['broken']
L = [r'\begin{table*}[tp]', r'\centering\small\setlength{\tabcolsep}{3pt}',
     r'\caption{Choosing one click among several calls, on 600 ScreenSpot-Pro items that share no item and no screenshot with the items used to design the method. Every item is in the denominator; a flow that returns no point counts as a miss, which happened only in single-model flows (column No point). Calls and cost: what the flow needs per item, at list prices. Lower block: paired comparisons, as the difference in miss rate in percentage points with a 95\% paired bootstrap interval over screenshots, items repaired and broken, and the exact McNemar $p$ (Holm-adjusted over the two comparisons marked $\dagger$). D settings: GPT-6.1 and Sonnet 5.5 default, Gemini 3.8 lowest; M: all three lowest.}\label{tab:choose}',
     r'\begin{tabular}{@{}lrrrrr@{}}\toprule',
     r'Flow & Misses & No point & Miss rate & Calls & Cost / item \\\midrule']
for k, nm in NAME:
    d = dep[k]
    L.append(f'{nm} & {err[k]} & {nopt[k]} & {100 * err[k] / 600:.2f}\\% & {d["calls_per_item"]:.2f} & \\${d["usd_per_item"]:.4f} \\\\')
L += [r'\midrule', r'Comparison & Difference & 95\% interval & Repaired / broken & \multicolumn{2}{r}{$p$} \\\midrule']
COMP = {'D-C': [('GPT-default-zoom', '', ''), ('GPTx3', '', '$\\dagger$'), ('GLF', '', '$\\dagger$'), ('D-S', '', '')],
        'M-C': [('M-S', '', '')], 'GPT-default-zoom': [('GPT-min-zoom', '', '')], 'Sonnet-default-zoom': [('Sonnet-min-zoom', '', '')]}
SHORT = {'GPT-default-zoom': 'GPT-6.1 default zoom', 'GPTx3': 'GPT-6.1 $\\times$3', 'GLF': 'crop-first baseline', 'D-S': 'D, judge score',
         'M-S': 'M, judge score', 'GPT-min-zoom': 'GPT-6.1 lowest zoom', 'Sonnet-min-zoom': 'Sonnet 5.5 lowest zoom'}
FLOWN = {'D-C': 'D, score and comparison', 'M-C': 'M, score and comparison', 'GPT-default-zoom': 'GPT-6.1 default zoom', 'Sonnet-default-zoom': 'Sonnet 5.5 default zoom'}
for fl in ('D-C', 'M-C', 'GPT-default-zoom', 'Sonnet-default-zoom'):
    for base, _, mark in COMP[fl]:
        x = cmp[(base, fl)]; p = x.get('p_holm', x['p_exact'])
        if p < 0.001:
            mant, ex = f'{p:.0e}'.split('e'); ps = f'${mant}\\times10^{{{int(ex)}}}$'
        else: ps = f'{p:.3f}'
        L.append(f'{FLOWN[fl]} vs {SHORT[base]}{mark} & ${x["diff_pp"]:+.2f}$ & {fmt_ci(x["ci95"])} & {x["fixed"]} / {x["broken"]} & \\multicolumn{{2}}{{r}}{{{ps}}} \\\\')
L += [r'\bottomrule\end{tabular}', r'\end{table*}']
open(TEX / 't9_choose.tex', 'w').write('\n'.join(L) + '\n')

# ------------------------------------------------------------------ Table: E3 configuration chosen on the other sources (F1)
lo = json.load(open(AN / 'out/e3_loso_config.json'))
MN = {'gpt61': 'GPT', 'sonnet55': 'Sonnet', 'gem38': 'Gemini'}
DN = {'doclaynet': 'DocLayNet', 'dota': 'DOTA', 'fsc147': 'FSC-147', 'sku110k': 'SKU-110K'}
CN = {'L-min': 'list, low', 'R-min': 'regions, low', 'L-default': 'list, def.', 'R-default': 'regions, def.'}
s = lo['summary']['per_model_F1']; sE = lo['summary']['per_model_E']
mm = lo['menu_means']; DSS = sorted(DN); CANDS = ['L-min', 'R-min', 'L-default', 'R-default']
allpick = {m: max(CANDS, key=lambda c: (np.mean([mm[f'{d}|{m}|{c}']['F1'] for d in DSS]), -CANDS.index(c))) for m in MN}
same = sum(r['chosen'] == allpick[r['model']] for r in lo['cells']['per_model_F1'])

L = [r'\begin{table}[t]', r'\centering\small\setlength{\tabcolsep}{2.5pt}',
     (r'\caption{Single-call configuration chosen on the other three detection sources and applied to the fourth (F1, primary scoring). '
      r'Reference: list interface at the lowest setting (low); def.: default setting. Change in F1 points with 95\% paired bootstrap interval over images. '
      f'Macro F1 {100 * s["macro_ref"]:.1f} (reference) and {100 * s["macro_chosen"]:.1f} (chosen); the best configuration per cell in hindsight gives {100 * s["macro_hindsight"]:.1f}. '
      f'The table evaluates the selection procedure. The shipped configuration (Table~\\ref{{tab:defaults}}) is chosen by the same rule on all four sources and equals the choice shown here in {same} of 12 cells; Table~\\ref{{tab:config_menu}} gives the four fixed configurations.' + r'}\label{tab:config}'),
     r'\begin{tabular}{@{}llll@{}}\toprule', r'Source & Model & Chosen & Change [95\% CI] \\\midrule']
for r in lo['cells']['per_model_F1']:
    mark = {'better': '$^+$', 'worse': '$^-$', 'ns': ''}[r['result']]
    L.append(f'{DN[r["source"]]} & {MN[r["model"]]} & {CN[r["chosen"]]} & ${100 * r["diff"]:+.1f}$ $[{100 * r["ci95"][0]:+.1f}, {100 * r["ci95"][1]:+.1f}]${mark} \\\\')
L += [r'\bottomrule\end{tabular}', r'\end{table}']
open(TEX / 't10_config.tex', 'w').write('\n'.join(L) + '\n')
# supplementary: the four fixed configurations, macro over the four sources
L = [r'\begin{table}[h]', r'\centering\small',
     r'\caption{The four fixed single-call configurations on the four detection sources (first repetition; mean over sources of per-source mean F1 at the primary scoring, and mean cost per image at list prices). low: lowest setting; def.: default setting. The shipped F1 configuration per model is the highest F1 of its column (bold); for Gemini 3.8 the two default-setting configurations differ by 0.2 points.}\label{tab:config_menu}',
     r'\begin{tabular}{@{}lrrrrrr@{}}\toprule', r' & \multicolumn{2}{c}{GPT-6.1} & \multicolumn{2}{c}{Sonnet 5.5} & \multicolumn{2}{c}{Gemini 3.8} \\', r'Configuration & F1 & Cost & F1 & Cost & F1 & Cost \\\midrule']
for c in CANDS:
    row = [CN[c]]
    for m in MN:
        f1 = round(100 * np.mean([mm[f'{d}|{m}|{c}']['F1'] for d in DSS]) + 1e-9, 1); co = np.mean([mm[f'{d}|{m}|{c}']['cost'] for d in DSS])
        row += [(f'\\textbf{{{f1:.1f}}}' if allpick[m] == c else f'{f1:.1f}'), f'\\${co:.4f}']
    L.append(' & '.join(row) + r' \\')
L += [r'\bottomrule\end{tabular}', r'\end{table}']
open(TEX / 'tA_config_menu.tex', 'w').write('\n'.join(L) + '\n')
print(open(TEX / 'tA_config_menu.tex').read(), 'same', same)
print(open(TEX / 't9_choose.tex').read()); print(open(TEX / 't10_config.tex').read())
