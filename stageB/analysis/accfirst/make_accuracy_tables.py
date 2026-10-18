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
vt = json.load(open(HERE / 'conf_vote_conf_v1.json'))  # plain three-model vote (accfirst/conf_vote.py)
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
DEPLOY['vote-3'] = [('A', 'gem38', ['C1']), ('B', 'gem38', ['B4'])] + [('DA', m, ['C1d']) for m in ('gpt61', 'sonnet55')] + \
                  [('DB', m, ['B4d']) for m in ('gpt61', 'sonnet55')]
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
for x in vt['paired']: cmp[(x['base'].replace('vote-3-models', 'vote-3'), x['flow'].replace('vote-3-models', 'vote-3'))] = x
from decimal import Decimal, ROUND_HALF_UP
usd4 = lambda x: str(Decimal(str(x)).quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP))
fmt_ci = lambda c: '[' + ', '.join(f'{v:.2f}'.replace('-', '$-$') for v in c) + ']'
def mark(t, c, lower_better=True):
    good = c[1] < 0 if lower_better else c[0] > 0; bad = c[0] > 0 if lower_better else c[1] < 0
    return ('{\\bfseries\\boldmath ' + t + '}') if good else (('\\underline{' + t + '}') if bad else t)
SIGNOTE = (r'\textbf{Bold}: the paired 95\% interval excludes 0 in the favourable direction; '
           r'\underline{underlined}: it excludes 0 in the unfavourable direction.')
NAME = [('GPT-min-zoom', 'GPT-6.1 zoom, lowest'), ('Sonnet-min-zoom', 'Sonnet 5.5 zoom, lowest'), ('Gemini-min-zoom', 'Gemini 3.8 zoom, lowest'),
        ('Sonnet-default-zoom', 'Sonnet 5.5 zoom, default'), ('GPT-default-zoom', 'GPT-6.1 zoom, default'),
        ('GPTx3', 'GPT-6.1 zoom $\\times$3, default; cluster vote'), ('GLF', 'Crop-first, GUI-Lens style (our implementation)'), ('vote-3', 'Three models; cluster vote'),
        ('M-S', 'Three models, all lowest; score'), ('M-C', 'Three models, all lowest; score and comparison'),
        ('D-S', 'Three models; score'), ('D-C', 'GUI-grounding tier (Alg.~\\ref{alg:clicks}): three models; score and comparison')]
NM = {'D-C': 'GUI-grounding tier', 'GPT-default-zoom': 'GPT-6.1 zoom, default', 'GPTx3': 'GPT-6.1 zoom $\\times$3, vote', 'GLF': 'crop-first baseline',
      'vote-3': 'three models, cluster vote', 'D-S': 'three models, score', 'M-C': 'All lowest, score and comparison', 'M-S': 'all lowest, score',
      'GPT-min-zoom': 'GPT-6.1 zoom, lowest', 'Sonnet-default-zoom': 'Sonnet 5.5 zoom, default', 'Sonnet-min-zoom': 'Sonnet 5.5 zoom, lowest'}
sing = json.load(open(HERE / 'conf_singles_conf_v1.json'))
err = {k: v['errors'] for k, v in sing.items()}; nopt = {k: v['no_point'] for k, v in sing.items()}
for k, v in ev['flows'].items(): assert err[k] == v['errors'] and nopt[k] == v['no_output'], k
err['vote-3'], nopt['vote-3'] = vt['vote']['errors'], vt['vote']['no_output']
assert err['Sonnet-min-zoom'] == ev['flows']['Sonnet-default-zoom']['errors'] + cmp[('Sonnet-min-zoom', 'Sonnet-default-zoom')]['fixed'] - cmp[('Sonnet-min-zoom', 'Sonnet-default-zoom')]['broken']
L = [r'\begin{table*}[tp]', r'\centering\footnotesize\setlength{\tabcolsep}{4pt}',
     r'\caption{Choosing one click among several calls, on 600 ScreenSpot-Pro items that share no item or screenshot with the 260 click items of the GUI experiment. '
     r'(a) Every flow, with every item in the denominator; a flow that returns no point counts as a miss, which happened only in single-model flows (column No point). Calls and cost: what the flow needs per item, every attempt counted, at list prices. '
     r"The three-model flows use GPT-6.1 and Sonnet 5.5 at the default setting and Gemini 3.8 at the lowest, unless marked ``all lowest''. "
     r'Cluster vote: the largest cluster of points within $\delta$ wins, ties go to the earliest model (GPT-6.1, Sonnet 5.5, Gemini 3.8), and the earliest point of the winning cluster is returned; computed from the stored calls, no scorer. '
     r'(b) Paired comparisons: $\Delta$ is the miss rate of the first flow minus that of the second, in percentage points (negative favours the first), with a 95\% paired bootstrap interval over screenshots, items repaired and broken by the first flow, and the exact McNemar $p$ (Holm-adjusted over the two comparisons marked $\dagger$). ' + SIGNOTE + r'}\label{tab:choose}',
     r'\begin{tabular}{@{}lrrrrr@{}}\toprule',
     r'\multicolumn{6}{@{}l}{\emph{(a) Flows}} \\',
     r'Flow & Misses & No point & Miss rate (\%) & Calls & Cost / item (\$) \\\midrule']
for k, nm in NAME:
    d = dep[k]
    L.append(f'{nm} & {err[k]} & {nopt[k]} & {100 * err[k] / 600:.2f} & {d["calls_per_item"]:.2f} & {usd4(d["usd_per_item"])} \\\\')
L += [r'\bottomrule\end{tabular}', '', r'\medskip', '',
      r'\begin{tabular}{@{}lrrrr@{}}\toprule', r'\multicolumn{5}{@{}l}{\emph{(b) Paired comparisons}} \\',
      r'First flow vs.\ second flow & $\Delta$ (pp) & 95\% interval & Repaired / broken & $p$ \\\midrule']
COMP = {'D-C': [('GPT-default-zoom', ''), ('GPTx3', '$^\\dagger$'), ('GLF', '$^\\dagger$'), ('vote-3', ''), ('D-S', '')],
        'M-C': [('M-S', '')], 'GPT-default-zoom': [('GPT-min-zoom', '')], 'Sonnet-default-zoom': [('Sonnet-min-zoom', '')]}
for fl in ('D-C', 'M-C', 'GPT-default-zoom', 'Sonnet-default-zoom'):
    for base, dg in COMP[fl]:
        x = cmp[(base, fl)]; p = x.get('p_holm', x['p_exact'])
        if p < 0.001:
            mant, ex = f'{p:.0e}'.split('e'); ps = f'${mant}\\times10^{{{int(ex)}}}$'
        else: ps = f'{p:.3f}'
        dpp = f'{x["diff_pp"]:+.2f}'.replace('-', '$-$').replace('+', '$+$')
        lc = lambda t: t[0].lower() + t[1:] if t.startswith(('Three', 'Crop')) else t
        L.append(f'{NM[fl]} vs.\\ {lc(NM[base])}{dg} & {mark(dpp, x["ci95"])} & {fmt_ci(x["ci95"])} & {x["fixed"]} / {x["broken"]} & {ps} \\\\')
L += [r'\bottomrule\end{tabular}', r'\end{table*}']
open(TEX / 't9_choose.tex', 'w').write('\n'.join(L) + '\n')

# ------------------------------------------------------------------ Table: E3 configuration chosen on the other sources (F1)
lo = json.load(open(AN / 'out/e3_loso_config.json'))
MN = {'gpt61': 'GPT-6.1', 'sonnet55': 'Sonnet 5.5', 'gem38': 'Gemini 3.8'}
DN = {'doclaynet': 'DocLayNet', 'dota': 'DOTA', 'fsc147': 'FSC-147', 'sku110k': 'SKU-110K'}
CN = {'L-min': 'list, lowest', 'R-min': 'regions, lowest', 'L-default': 'list, default', 'R-default': 'regions, default'}
s = lo['summary']['per_model_F1']; sE = lo['summary']['per_model_E']
mm = lo['menu_means']; DSS = sorted(DN); CANDS = ['L-min', 'R-min', 'L-default', 'R-default']
allpick = {m: max(CANDS, key=lambda c: (np.mean([mm[f'{d}|{m}|{c}']['F1'] for d in DSS]), -CANDS.index(c))) for m in MN}
same = sum(r['chosen'] == allpick[r['model']] for r in lo['cells']['per_model_F1'])
sgn = lambda v: f'{v:+.1f}'.replace('-', '$-$').replace('+', '$+$')
L = [r'\begin{table*}[tp]', r'\centering\footnotesize\setlength{\tabcolsep}{5pt}',
     (r'\caption{Single-call configuration chosen on the other three detection sources and applied to the fourth. F1 at the primary scoring of Table~\ref{tab:contract} (boxes: same class, IoU $\ge0.75$; FSC-147: point tolerance), in \%. '
      r'Reference: list interface at the lowest setting. Change: chosen minus reference, in F1 points, with a 95\% paired bootstrap interval over images. ' + SIGNOTE + ' '
      f'Macro F1, the mean over the 12 cells: {100 * s["macro_ref"]:.1f} (reference) and {100 * s["macro_chosen"]:.1f} (chosen); the best configuration of each cell chosen with its own labels gives {100 * s["macro_hindsight"]:.1f}. '
      f'The shipped configuration (Table~\\ref{{tab:defaults}}) is chosen by the same rule on all four sources (Table~\\ref{{tab:config_menu}}).' + r'}\label{tab:config}'),
     r'\begin{tabular}{@{}llllrr@{}}\toprule', r'& & \multicolumn{2}{c}{Chosen} & & \\\cmidrule(lr){3-4}', r'Source & Model & Interface & Setting & Change & 95\% CI \\\midrule']
for r in lo['cells']['per_model_F1']:
    c = [100 * r['ci95'][0], 100 * r['ci95'][1]]
    t = sgn(100 * r['diff']); t = {'better': '{\\bfseries\\boldmath ' + t + '}', 'worse': '\\underline{' + t + '}', 'ns': t}[r['result']]
    cs = '[' + ', '.join(f'{v:.1f}'.replace('-', '$-$') for v in c) + ']'
    iface, sett = CN[r['chosen']].split(', ')
    L.append(f'{DN[r["source"]]} & {MN[r["model"]]} & {iface} & {sett} & {t} & {cs} \\\\')
L += [r'\bottomrule\end{tabular}', r'\end{table*}']
open(TEX / 't10_config.tex', 'w').write('\n'.join(L) + '\n')
# supplementary: the four fixed configurations, macro over the four sources
L = [r'\begin{table}[H]', r'\centering\footnotesize\setlength{\tabcolsep}{4pt}',
     r'\caption{The four fixed single-call configurations on the four detection sources: F1 (\%, primary scoring of Table~\ref{tab:contract}), the mean over sources of the per-source mean F1 of the first of the three stored calls, and cost per image (\$, list prices). The shipped F1 configuration of each model is the highest F1 of its column (bold); for Gemini 3.8 the two default-setting configurations differ by less than 0.2 points.}\label{tab:config_menu}',
     r'\begin{tabular}{@{}lrrrrrr@{}}\toprule', r' & \multicolumn{2}{c}{GPT-6.1} & \multicolumn{2}{c}{Sonnet 5.5} & \multicolumn{2}{c}{Gemini 3.8} \\\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}', r'Configuration & F1 & Cost & F1 & Cost & F1 & Cost \\\midrule']
for c in CANDS:
    row = [CN[c]]
    for m in MN:
        f1 = round(100 * np.mean([mm[f'{d}|{m}|{c}']['F1'] for d in DSS]) + 1e-9, 1); co = np.mean([mm[f'{d}|{m}|{c}']['cost'] for d in DSS])
        row += [(f'\\textbf{{{f1:.1f}}}' if allpick[m] == c else f'{f1:.1f}'), f'{co:.4f}']
    L.append(' & '.join(row) + r' \\')
L += [r'\bottomrule\end{tabular}', r'\end{table}']
open(TEX / 'tA_config_menu.tex', 'w').write('\n'.join(L) + '\n')
print(open(TEX / 'tA_config_menu.tex').read(), 'same', same)
print(open(TEX / 't9_choose.tex').read()); print(open(TEX / 't10_config.tex').read())
