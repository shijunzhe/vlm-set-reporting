"""Writes the LaTeX tables of the FSR paper from analysis/out/*.json. No model calls, no hand-typed numbers.
Run after analysis/run_all.sh:  python3 -I make_tables.py"""
import json
import numpy as np, collections
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = Path('/home/claude/stageB/analysis/out')
TAB = HERE / 'tables'; TAB.mkdir(exist_ok=True)
M = ['gpt61', 'sonnet55', 'gem38']
MN = {'gpt61': 'GPT-6.1', 'sonnet55': 'Sonnet 5.5', 'gem38': 'Gemini 3.8', 'same': 'same model'}
SRC = [('E3', 'doclaynet', 'DocLayNet'), ('E3', 'dota', 'DOTA'), ('E3', 'fsc147', 'FSC-147'), ('E3', 'sku110k', 'SKU-110K'),
       ('GUI', 'groundcua', 'GroundCUA'), ('GUI', 'screenparse', 'ScreenParse')]


def j(name): return json.load(open(OUT / f'{name}.json'))


def num(x, nd=2):
    s = f'{x:.{nd}f}'
    return s.replace('-', '$-$')


def pct(x, nd=1, sign=True):
    s = f'{x:+.{nd}f}' if sign else f'{x:.{nd}f}'
    if float(s) == 0: s = f'{0:.{nd}f}'
    return s.replace('-', '$-$').replace('+', '$+$') + '\\%'


def ci(c, nd=1):
    return '[' + ', '.join(f'{v:.{nd}f}'.replace('-', '$-$') for v in c) + ']'


def sig(c): return c[0] > 0 or c[1] < 0


def hu(x, nd=1):
    """Round half up on the decimal value (a stored 0.4375 prints as 43.8)."""
    from decimal import Decimal, ROUND_HALF_UP
    return str(Decimal(repr(round(x, 10))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)).replace('-', '$-$')


def mark(t, c, lower_better=True):
    """Significance convention used in every generated table: bold if the paired 95% interval excludes 0 in the
    favourable direction, underlined if it excludes 0 in the unfavourable direction."""
    good = c[1] < 0 if lower_better else c[0] > 0
    bad = c[0] > 0 if lower_better else c[1] < 0
    if good: return '{\\bfseries\\boldmath ' + t + '}'
    if bad: return '\\underline{' + t + '}'
    return t


def H(*lines):
    """A header cell of several centred lines."""
    return '\\begin{tabular}[b]{@{}c@{}}' + '\\\\'.join(('{}' + x) if x.startswith('[') else x for x in lines) + '\\end{tabular}'


SIGNOTE = ('\\textbf{Bold}: the paired 95\\% interval excludes 0 in the favourable direction; '
           '\\underline{underlined}: it excludes 0 in the unfavourable direction.')


def write(name, s):
    (TAB / f'{name}.tex').write_text(s)


# ---------------------------------------------------------------- tables, in the order of the main script
def t1():
    d = j('readout_effects')
    rows = [('Pure readout, count-first', 'stated $\\to$ raw list count', 'pure_readout_old_protocol'),
            ('Pure readout, list-first', 'stated $\\to$ raw list count', 'pure_readout_new_protocol'),
            ('Alias map, count-first', 'raw $\\to$ mapped list count', 'normalization_old'),
            ('Alias map, list-first', 'raw $\\to$ mapped list count', 'normalization_new'),
            ('Protocol', 'mapped list count, count-first $\\to$ list-first', 'protocol_fixed_readout'),
            ('Full method', 'count-first stated $\\to$ readout', 'full_method')]
    L = []
    for lab, comp, k in rows:
        cells = []
        for m in M:
            v = d[m][k]
            if v['rel'] == 0.0 and v['ci'] == [0.0, 0.0]:
                cells.append(f"{num(v['a'])} $\\to$ {num(v['b'])} (no change)")
            else:
                n = f" ($n={v['n']}$)" if v['n'] != 50 else ''
                cells.append(f"\\begin{{tabular}}{{@{{}}c@{{}}}}{num(v['a'])} $\\to$ {num(v['b'])}{n}\\\\ {mark(pct(v['rel']) + ' ' + ci(v['ci']), v['ci'])}\\end{{tabular}}")
        L.append(f"{lab} & {comp} & " + ' & '.join(cells) + ' \\\\')
    tot = ' & '.join(f"{num(d[m]['totals_same_response_new']['list_mae'])} vs.\\ {num(d[m]['totals_same_response_new']['declared_mae'])}" for m in M)
    L.append('\\midrule')
    L.append(f'Totals, list-first & list length vs.\\ stated total & {tot} \\\\')
    s = ('\\begin{table*}[tp]\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\n'
         '\\caption{Readout effects kept separate (SCP-dense, 50 images, lowest setting, one call; per-type count error, lower is better; paired image bootstrap 95\\% intervals). '
         'The \\emph{raw} list count counts the records of each class before the alias map, the \\emph{mapped} list count after it. '
         '\\emph{Pure readout}: stated count $\\to$ raw list count of the same response. \\emph{Alias map}: raw $\\to$ mapped list count of the same list. '
         '\\emph{Protocol}: mapped list count of a count-first response $\\to$ that of a list-first response. '
         '\\emph{Full method}: stated count of a count-first response $\\to$ readout of Section~\\ref{sec:method} on a list-first response, the effect of the reporting and decoding protocol. '
         '\\emph{Totals}: mean absolute difference between the list length and the true total, and between the stated total and the true total. ' + SIGNOTE + '}\\label{tab:readout}\n'
         '\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{2.2cm}>{\\raggedright\\arraybackslash}p{2.4cm}ccc@{}}\\toprule\n'
         'Effect & Comparison & GPT-6.1 & Sonnet 5.5 & Gemini 3.8 (lowest) \\\\\\midrule\n' + '\n'.join(L) +
         '\n\\bottomrule\\end{tabular}\\end{table*}\n')
    write('t1_readout', s)


def t2():
    d = j('rc16_setting'); lv = d['levels']; c = d['contrasts']
    names = {'min16k': 'lowest, 16k', 'default16k': 'default, 16k', 'default32k': 'default, 32k'}
    L = []
    for k in ['min16k', 'default16k', 'default32k']:
        v = lv[k]
        L.append(f"{names[k]} & {num(v['readout'])} & {num(v['code'])} & {num(v['E'])} & {num(v['n_unknown'])} & {num(v['E_retain'])} & {v['flagged']*100:.0f} & {v['strict_fail']*100:.0f} & {v['trunc']*100:.0f} & {v['think_share']*100:.0f} & {v['out_tok']:,.0f} & {v['cost_total']:.2f} \\\\")
    a = c['readout: default16k vs min16k']; b = c['readout: default32k vs default16k']
    s = ('\\begin{table*}[tp]\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\n'
         '\\caption{Gemini 3.8 on SCP-dense under the list-first protocol (50 images, one call each). At a fixed 16k output cap, the default reasoning setting changes the count readout error by '
         f"{pct(a['rel'])} {ci(a['ci'])}; raising the cap to 32k changes it by a further {pct(b['rel'])} {ci(b['ci'])} (not significant). "
         'Readout error: per-type count error of the Step-6 readout (stated count if the list is flagged, list count otherwise). List-count error: the same for the list count alone. '
         'Set error $E$ scores the delivered records, those whose class is valid after the alias map (equal class, 20 px); unknown-class records are excluded from the delivered set and counted in the next column; '
         'the column after it is a sensitivity that keeps them and scores them as unmatched. '
         'Flagged: share of lists with at least one degeneracy flag. JSON fail: share of responses that do not parse as the requested object (records are then recovered by the declared salvage step). Hit cap: share of responses that stopped at the output cap. Reasoning: share of calls with reasoning tokens. Output tokens: mean per call. Cost: all 50 calls, counting every attempted call. '
         'The setting comparison is cap-matched, not cost-matched: output tokens and cost rise with reasoning.}\\label{tab:setting}\n'
         '\\begin{tabular}{@{}lrrrrrrrrrrr@{}}\\toprule\n'
         + ' & '.join([H('Setting,', 'cap'), H('Readout', 'error'), H('List-count', 'error'), H('Set', 'error $E$'), H('Unknown', 'records'), H('$E$, unknown', 'kept'),
                       H('Flagged', '(\\%)'), H('JSON', 'fail (\\%)'), H('Hit', 'cap (\\%)'), H('Reasoning', '(\\%)'), H('Output', 'tokens'), H('Cost', '(\\$)')]) + ' \\\\\\midrule\n'
         + '\n'.join(L) + '\n\\bottomrule\\end{tabular}\\end{table*}\n')
    write('t2_setting', s)


def t3():
    d = j('intersection_decomp'); pr = j('p1_predict')['pairs']
    L = []; allrows = []
    for fam, src, lab in SRC:
        rows = [(k, v) for k, v in d.items() if isinstance(v, dict) and k.startswith(f'{fam}|{src}|')]
        allrows += rows
        th = np.mean([pr[k]['sign_correct_threshold'] for k, _ in rows])
        helps = sum(v['selection'] > 0 for _, v in rows)
        gpos = sum(v['geometry'] > 0 for _, v in rows); gneg = sum(v['geometry'] < 0 for _, v in rows)
        flips = sum((v['total'] > 0) != (v['selection'] > 0) for _, v in rows if v['selection'] != 0)
        p1 = [v['p1_marginal'] for _, v in rows]
        sel = [v['selection'] for _, v in rows]; geo = [v['geometry'] for _, v in rows]
        L.append(f"{lab} & {100 * th:.0f} & {helps} & {min(p1):.2f} to {max(p1):.2f} & {num(min(sel),1)} to {num(max(sel),1)} & {gpos} / {gneg} & {num(min(geo),1)} to {num(max(geo),1)} & {flips} \\\\")
    th = np.mean([pr[k]['sign_correct_threshold'] for k, _ in allrows])
    helps = sum(v['selection'] > 0 for _, v in allrows)
    gpos = sum(v['geometry'] > 0 for _, v in allrows); gneg = sum(v['geometry'] < 0 for _, v in allrows)
    flips = sum((v['total'] > 0) != (v['selection'] > 0) for _, v in allrows if v['selection'] != 0)
    p1 = [v['p1_marginal'] for _, v in allrows]; sel = [v['selection'] for _, v in allrows]; geo = [v['geometry'] for _, v in allrows]
    L.append('\\midrule')
    L.append(f"All {len(allrows)} pairs & {100 * th:.0f} & {helps} & {min(p1):.2f} to {max(p1):.2f} & {num(min(sel),1)} to {num(max(sel),1)} & {gpos} / {gneg} & {num(min(geo),1)} to {num(max(geo),1)} & {flips} \\\\")
    s = ('\\begin{table*}[tp]\\centering\\footnotesize\\setlength{\\tabcolsep}{4pt}\n'
         '\\caption{Confirmation by a partner call, split into selection and geometry (Propositions~\\ref{prop:identity} and \\ref{prop:geometry}). Each source has 9 base/partner pairs (3 base models $\\times$ same-model or one of two other models), 60 images, one call each; selection, geometry and their total are per-image reductions of $E$ (positive is better). '
         'Both identities hold exactly on every image of every pair. \\emph{Sign from 5 labels}: $p_1$ is estimated on 5 labelled images and Proposition~\\ref{thm:intersect} predicts whether deletion lowers $E$ on the other 55; share of correct predictions over 200 draws per pair, averaged over the pairs (Section~\\ref{sec:res:set}). '
         '\\emph{Deletion helps}: pairs whose selection term is positive. \\emph{Geometry $+$ / $-$}: pairs whose geometry term is positive / negative. \\emph{Sign flips}: pairs in which the total with averaged coordinates has the opposite sign of the deletion-only selection term. Per-pair values are in Table~\\ref{tab:intersect_full}.}\\label{tab:intersect}\n'
         '\\begin{tabular}{@{}lrrrrrrr@{}}\\toprule\n'
         'Source & ' + ' & '.join([H('Sign from', '5 labels (\\%)'), H('Deletion', 'helps'), H('$p_1$', 'range'), H('Selection', 'range'), H('Geometry', '$+$ / $-$'), H('Geometry', 'range'), H('Sign', 'flips')]) + ' \\\\\\midrule\n'
         + '\n'.join(L) + '\n\\bottomrule\\end{tabular}\\end{table*}\n')
    write('t3_intersect', s)
    # full appendix table
    L = []
    for fam, src, lab in SRC:
        for b in M:
            for p in ['same'] + [x for x in M if x != b]:
                k = f'{fam}|{src}|{b}|{p}'
                if k not in d: continue
                v = d[k]
                L.append(f"{lab} & {MN[b]} & {MN[p]} & {v['deleted_scored_per_img']:.2f} & {v['matches_lost_per_img']:.2f} & {v['p1_marginal']:.3f} & {num(v['selection'])} & {num(v['geometry'])} & {num(v['total'])} \\\\")
        L.append('\\midrule')
    L = L[:-1]
    s = ('{\\footnotesize\\setlength{\\tabcolsep}{4pt}\\begin{longtable}{@{}lllrrrrrr@{}}\n'
         '\\caption{Selection and geometry terms for every base/partner pair (60 images per source, one call each; data of Table~\\ref{tab:intersect}). $\\Delta c$: deleted scored records per image; $\\Delta r$: matches lost per image; $p_1=\\Delta r/\\Delta c$, the marginal matching rate of the deleted records. Selection, geometry and total are per-image reductions of $E$ (positive is better); selection plus geometry equals the total on every row (Propositions~\\ref{prop:identity} and \\ref{prop:geometry}).}\\ltlabel{tab:intersect_full}\\\\\\toprule\n'
         'Source & Base & Partner & $\\Delta c$ & $\\Delta r$ & $p_1$ & Selection & Geometry & Total \\\\\\midrule\\endfirsthead\n'
         '\\multicolumn{9}{@{}l}{\\emph{Table~\\ref{tab:intersect_full}, continued}}\\\\\\toprule Source & Base & Partner & $\\Delta c$ & $\\Delta r$ & $p_1$ & Selection & Geometry & Total \\\\\\midrule\\endhead\n'
         '\\midrule\\multicolumn{9}{r}{\\emph{continued on the next page}}\\\\\\endfoot\n\\bottomrule\\endlastfoot\n'
         + '\n'.join(L) + '\n\\end{longtable}\\ltlabelafter{tab:intersect_full}}\n')
    write('tA_intersect_full', s)


def t4():
    d = j('prep1_v2')
    L = []
    for _, src, lab in SRC[:4]:
        for m in M:
            v = d[f'{src}|{m}']
            g = v['gap_crossfit_oracle_logit_ci']
            L.append(f"{lab} & {MN[m]} & {v['oracle_constrained']*100:.1f} & {v['oracle_unrestricted_atoms']*100:.1f} & "
                     f"{num(v['crossfit_oracle_logit']*100,1)} & {num(v['crossfit_oracle_gbt']*100,1)} & {num(v['insample_oracle_logit']*100,1)} & {num(v['insample_oracle_gbt']*100,1)} & {num(v['labels10_oracle_logit_median']*100,1)} & "
                     f"{(v['oracle_constrained']-v['crossfit_oracle_logit'])*100:.1f} {ci([x*100 for x in g])} \\\\")
    s_ = d['_summary']
    tot = sum(d[f'{src}|{m}']['cases_unrestricted_ne_constrained'] for _, src, _ in SRC[:4] for m in M)
    gaps = sorted((d[f'{src}|{m}']['oracle_constrained'] - d[f'{src}|{m}']['crossfit_oracle_logit']) * 100 for _, src, _ in SRC[:4] for m in M)
    med_gap = (gaps[5] + gaps[6]) / 2
    ig = sorted(d[f'{src}|{m}']['insample_oracle_gbt'] * 100 for _, src, _ in SRC[:4] for m in M); med_ig = (ig[5] + ig[6]) / 2
    L.append('\\midrule')
    L.append(f"\\multicolumn{{2}}{{@{{}}l}}{{Median of 12 cells}} & {s_['oracle_constrained']['median']*100:.1f} & {s_['oracle_unrestricted_atoms']['median']*100:.1f} & "
             f"{s_['crossfit_oracle_logit']['median']*100:.1f} & {s_['crossfit_oracle_gbt']['median']*100:.1f} & {s_['insample_oracle_logit']['median']*100:.1f} & {med_ig:.1f} & {s_['labels10_oracle_logit_median']['median']*100:.1f} & {med_gap:.1f} \\\\")
    s = ('\\begin{table*}[tp]\\centering\\footnotesize\\setlength{\\tabcolsep}{2pt}\n'
         '\\caption{Selection among three calls of one model (detection sources, list interface, lowest setting, 60 images per cell; reduction of $E$ relative to the first call, in \\%, higher is better). '
         'The constrained oracle optimizes over the same feasible family as the selectors (Proposition~\\ref{prop:oracle}); the unrestricted value lets two members of one cluster match two targets and is not attainable. '
         f'The two oracles differ on {tot} of 720 images. '
         'Selectors are logistic or boosted-tree record gates on output features; they read no labels at inference but are fitted with labels: cross-fitted (5 folds by image, persisted), in-sample, or on 10 labelled images (median over 40 draws, each evaluated on the images it did not use). '
         'Gap: constrained oracle minus cross-fitted logistic, with a 95\\% image-bootstrap interval for the fitted selector; it mixes information absent from the features with selector and estimation shortfall (Proposition~\\ref{prop:order}).}\\label{tab:select}\n'
         '\\begin{tabular}{@{}llrrrrrrr@{\\hspace{8pt}}r@{}}\\toprule\n'
         '& & \\multicolumn{2}{c}{Oracles} & \\multicolumn{5}{c}{Selectors fitted with labels} & \\\\\\cmidrule(lr){3-4}\\cmidrule(lr){5-9}\n'
         'Source & Model & ' + ' & '.join(['Constrained', 'Unrestricted', H('Cross-fit', 'logistic'), H('Cross-fit', 'trees'), H('In-sample', 'logistic'), H('In-sample', 'trees'), H('10 labels', 'logistic'), H('Gap', '[95\\% CI]')]) + ' \\\\\\midrule\n'
         + '\n'.join(L) + '\n\\bottomrule\\end{tabular}\\end{table*}\n')
    write('t4_select', s)


CONSTR = {'unan3_fus': 'unanimity of 3 calls, averaged', 'maj3_fus': 'majority of 3 calls, averaged', 'union3': 'union of 3 calls',
          'x3_maj_fus': 'majority of 3 models, averaged'}


def constr_name(c):
    if c in CONSTR: return CONSTR[c]
    kind, partner = c.split('|')
    base = {'x2_union': 'union with', 'x2_inter_fus': 'intersection with'}[kind]
    return f"{base} {MN[partner]}" + (', averaged' if kind.endswith('_fus') else '')


def t5():
    d = j('construction_table'); dt = d['default_transfer']; sm = d['default_transfer_summary']
    single = collections.defaultdict(dict)
    for k, v in dt.items():
        pool, fam, src, loss, m = k.split('|'); single[(fam, src, loss)][m] = v['single']

    def cell(pool, fam, src, loss, m):
        v = dt[f'{pool}|{fam}|{src}|{loss}|{m}']
        rel = (v['value'] - v['single']) / v['single'] * 100
        s = single[(fam, src, loss)]; strongest = (min(s, key=s.get) if loss == 'E' else max(s, key=s.get)) == m
        t = mark(pct(rel, 0).replace('\\%', ''), v['ci'], lower_better=(loss == 'E'))
        if strongest: t = t + '$^{\\star}$'
        return t
    L = []
    for fam, src, lab in SRC:
        row = [lab]
        for pool, loss in [('same_model_only', 'E'), ('same_model_only', 'F1'), ('all', 'E'), ('all', 'F1')]:
            row += [cell(pool, fam, src, loss, m) for m in M]
        L.append(' & '.join(row) + ' \\\\')
    tot = ['Better / worse / n.s.']
    for k in ['same_model_only|E', 'same_model_only|F1', 'all|E', 'all|F1']:
        tot.append(f"\\multicolumn{{3}}{{c}}{{{sm[k]['better']} / {sm[k]['worse']} / {sm[k]['ns']}}}")
    L.append('\\midrule'); L.append(' & '.join(tot) + ' \\\\')
    s = ('\\begin{table*}[tp]\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\n'
         '\\caption{Defaults evaluated by leave-one-source-out transfer: for each source, the construction is the one that is best on the other five sources (same model only, or any construction including cross-model ones), and it is evaluated on the held-out source against a single call of the user\'s model. '
         'Entries are relative changes of the loss in \\% (for $E$, negative is better; for F1, positive is better). ' + SIGNOTE + ' '
         '$^{\\star}$: the user\'s model is the strongest single model on that source and loss; every significantly worse cross-model entry carries a $\\star$. GPT, Son., Gem.: GPT-6.1, Sonnet 5.5, Gemini 3.8. Costs, recall and the chosen construction per cell are in Table~\\ref{tab:transfer_full}.}\\label{tab:transfer}\n'
         '\\begin{tabular}{@{}l' + 'rrr' * 4 + '@{}}\\toprule\n'
         '& \\multicolumn{6}{c}{Same model only} & \\multicolumn{6}{c}{Any construction (incl.\\ cross-model)} \\\\\\cmidrule(lr){2-7}\\cmidrule(l){8-13}\n'
         '& \\multicolumn{3}{c}{$E$} & \\multicolumn{3}{c}{F1} & \\multicolumn{3}{c}{$E$} & \\multicolumn{3}{c}{F1} \\\\\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\\cmidrule(l){11-13}\n'
         'Source' + ' & GPT & Son. & Gem.' * 4 + ' \\\\\\midrule\n' + '\n'.join(L) + '\n\\bottomrule\\end{tabular}\\end{table*}\n')
    write('t5_transfer', s)
    # appendix full
    L = []
    for pool, loss in [('same_model_only', 'E'), ('same_model_only', 'F1'), ('all', 'E'), ('all', 'F1')]:
        ptxt = 'Same-model constructions only' if pool.startswith('same') else 'Any construction, including cross-model ones'
        ltxt = 'set error $E$ per image (lower is better)' if loss == 'E' else 'F1 in \\% (higher is better)'
        L.append('\\multicolumn{8}{@{}l}{\\emph{' + ptxt + '; ' + ltxt + '}} \\\\')
        for fam, src, lab in SRC:
            for m in M:
                v = dt[f'{pool}|{fam}|{src}|{loss}|{m}']; o = v['other_metrics']
                if loss == 'E':
                    a, b, c = num(v['single'], 2), num(v['value'], 2), v['ci']; cs = ci(c, 2)
                else:
                    a, b, c = hu(v['single'] * 100), hu(v['value'] * 100), [x * 100 for x in v['ci']]; cs = ci(c, 1)
                L.append(f"\\quad {lab} & {MN[m]} & {constr_name(v['applied'])} & {a} & {b} & {mark(cs, c, lower_better=(loss == 'E'))} & {hu(o['recall']*100)} & {o['cost_per_image']:.4f} \\\\")
        L.append('\\midrule')
    L = L[:-1]
    s = ('{\\footnotesize\\setlength{\\tabcolsep}{4pt}\\begin{longtable}{@{}ll>{\\raggedright\\arraybackslash}p{4.2cm}rrrrr@{}}\n' +
         '\\caption{Leave-one-source-out transfer, every cell (data of Table~\\ref{tab:transfer}). Default applied: the construction chosen on the other five sources; ``averaged\'\' means the coordinates of the kept records are averaged over the calls that report them. '
         'Single and Default: loss of one call of the model and of the default applied. 95\\% CI: paired image bootstrap of the change, default minus single (negative is better for $E$, positive for F1). ' + SIGNOTE + ' '
         'Recall (\\%) and cost per image (\\$, every call counted) are those of the default applied.}\\ltlabel{tab:transfer_full}\\\\\\toprule\n'
         'Source & Model & Default applied & Single & Default & 95\\% CI & Recall & Cost \\\\\\midrule\\endfirsthead\n'
         '\\multicolumn{8}{@{}l}{\\emph{Table~\\ref{tab:transfer_full}, continued}}\\\\\\toprule Source & Model & Default applied & Single & Default & 95\\% CI & Recall & Cost \\\\\\midrule\\endhead\n'
         '\\midrule\\multicolumn{8}{r}{\\emph{continued on the next page}}\\\\\\endfoot\n\\bottomrule\\endlastfoot\n'
         + '\n'.join(L) + '\n\\end{longtable}\\ltlabelafter{tab:transfer_full}}\n')
    write('tA_transfer_full', s)


def t6():
    d = j('alarm_curves')
    L = []
    for m in M:
        v = d['models'][m]
        for k, lab in [('same', 'same model, unzoomed'), ('cross', 'cross-model, unzoomed'), ('same_after_zoom', 'same model, after zoom'), ('cross_after_zoom', 'cross-model, after zoom')]:
            p = [x for x in v[k]['points'] if abs(x['tau'] - 0.015) < 1e-9][0]
            L.append(f"{MN[m] if k=='same' else ''} & {MN[v['partner']] if k=='same' else ''} & {lab} & {100*(1-v[k]['base_accuracy']):.1f} & {p['coverage']*100:.1f} & {p['risk_when_acting']*100:.1f} & {p['errors_caught']*100:.0f} \\\\")
        if m != M[-1]: L.append('\\midrule')
    s = ('\\begin{table*}[tp]\\centering\\footnotesize\\setlength{\\tabcolsep}{4pt}\n'
         '\\caption{Disagreement alarms for clicks (ScreenSpot-Pro and ScreenSpot-v2, 260 items; partner set by a fixed rule as the cheapest other model; point estimates). The alarm compares the acted-on point with a second point for the same item and fires when the two are farther apart than $\\delta=1.5\\%$ of the longer image side; in this risk-coverage analysis the item is then set aside instead of acted on (in Algorithm~\\ref{alg:clicks} it triggers a comparison instead). Unzoomed alarms compare first clicks; after zoom, the acted-on point is the zoomed first click, and the second point is the same model\'s zoomed second click or the partner\'s zoomed first click. '
         'Miss rate, all items: miss rate when acting on every item. Acceptance coverage: share of items acted on. Miss rate when acting: share of acted-on items whose point is outside the target. Errors caught: share of the acted-on prediction\'s misses that the alarm sets aside. Other values of $\\delta$ are in Figure~\\ref{fig:alarm}, and miss rates at matched acceptance coverage in Table~\\ref{tab:alarm_matched}.}\\label{tab:alarm}\n'
         '\\begin{tabular}{@{}lllrrrr@{}}\\toprule\n'
         'Model & Partner & Alarm & ' + ' & '.join([H('Miss rate,', 'all items (\\%) $\\downarrow$'), H('Acceptance', 'coverage (\\%) $\\uparrow$'), H('Miss rate when', 'acting (\\%) $\\downarrow$'), H('Errors', 'caught (\\%) $\\uparrow$')]) + ' \\\\\\midrule\n'
         + '\n'.join(L) + '\n\\bottomrule\\end{tabular}\\end{table*}\n')
    write('t6_alarm', s)
    # appendix: matched acceptance coverage
    L = []
    for m in M:
        first = True
        for key, lab in [('matched_coverage', 'unzoomed'), ('matched_coverage_after_zoom', 'after zoom')]:
            seen = set(); rows = []
            for r in d['models'][m][key]:
                kk = (round(r['coverage'], 4), round(r['risk_same'], 4), round(r['risk_cross'], 4))
                if kk in seen: continue
                seen.add(kk); rows.append(r)
            if not rows:
                L.append(f"{MN[m] if first else ''} & {lab} & \\multicolumn{{3}}{{l}}{{the two curves do not overlap}} \\\\"); first = False
            for i, r in enumerate(rows):
                L.append(f"{MN[m] if first else ''} & {lab if i == 0 else ''} & {r['coverage']*100:.1f} & {r['risk_same']*100:.1f} & {r['risk_cross']*100:.1f} \\\\"); first = False
        if m != M[-1]: L.append('\\midrule')
    s = ('\\begin{table}[H]\\centering\\footnotesize\\setlength{\\tabcolsep}{4pt}\n'
         '\\caption{Click alarms at matched acceptance coverage (260 items; data of Table~\\ref{tab:alarm} and Figure~\\ref{fig:alarm}; point estimates without intervals). Rows are the acceptance coverages that the same-model and the cross-model alarm both reach on the grid of $\\delta$; miss rates are those of the acted-on items, lower is better.}\\label{tab:alarm_matched}\n'
         '\\begin{tabular}{@{}llrrr@{}}\\toprule\n'
         'Model & Points & ' + ' & '.join([H('Acceptance', 'coverage (\\%)'), H('Miss rate,', 'same model (\\%)'), H('Miss rate,', 'cross-model (\\%)')]) + ' \\\\\\midrule\n'
         + '\n'.join(L) + '\n\\bottomrule\\end{tabular}\\end{table}\n')
    write('tA_alarm_matched', s)


def tA_density():
    d = j('readout_density')
    names = [('px', 'PixMo-Count', 'total'), ('fsc', 'FSC-147', 'total'), ('scp', 'SCP', 'per type'), ('scpd', 'SCP-dense', 'per type')]
    L = []
    for t, lab, kind in names:
        for m in M:
            v = d[f'{t}|{m}']
            if t in ('scp', 'scpd'):
                mapped = f"{num(v['code'])} & {mark(pct(v['rel']).replace(chr(92) + '%', ''), v['ci'])} & {ci(v['ci'])}"
            else:
                mapped = '\\multicolumn{3}{c}{not applicable (one class)}'
            L.append(f"{(chr(92) + 'begin{tabular}[t]{@{}l@{}}' + lab + chr(92)*2 + kind + ', ' + str(v['mean_objects']) + ' obj.' + chr(92) + 'end{tabular}') if m=='gpt61' else ''} & {MN[m]} & {v['n']} & {num(v['stated'])} & {num(v['code_raw'])} & {mark(pct(v['rel_raw']).replace(chr(92) + '%', ''), v['ci_raw'])} & {ci(v['ci_raw'])} & {mapped} \\\\")
        L.append('\\midrule')
    L = L[:-1]
    s = ('\\begin{table}[H]\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\n'
         '\\caption{Same-response readout across datasets (count-first prompts, lowest setting, one call; data of Figure~\\ref{fig:density}). Count error of the stated count and of the list count in the same response, without and with the declared alias map (lower is better; total error for one-class datasets, per-type error otherwise). '
         'Change: relative change from the stated count to the list count, with its paired image-bootstrap 95\\% interval; this is a separate run from Table~\\ref{tab:readout}, so shared rows can differ in the last digit of an interval, and the two list-count columns use separate draws. ' + SIGNOTE + ' Under each dataset: the error measure and the mean number of objects per image.}\\label{tab:density}\n'
         '\\begin{tabular}{@{}llrrrrrrrr@{}}\\toprule\n'
         '& & & & \\multicolumn{3}{c}{Raw list count} & \\multicolumn{3}{c}{Mapped list count} \\\\\\cmidrule(lr){5-7}\\cmidrule(l){8-10}\n'
         'Dataset & Model & $n$ & Stated & Error & Change (\\%) & 95\\% CI & Error & Change (\\%) & 95\\% CI \\\\\\midrule\n'
         + '\n'.join(L) + '\n\\bottomrule\\end{tabular}\\end{table}\n')
    write('tA_density', s)


def t7_common():
    d = j('common_practice'); al = j('alarm_curves')
    def cell(v, nd=2, pc=False):
        return f"{v*100:.1f}\\%" if pc else num(v, nd)
    def rel(b): return mark(pct(b['rel'], 1), b['ci'])
    L = []
    L.append('\\multicolumn{5}{@{}p{15.2cm}@{}}{\\emph{Repair the readout: dense counting (SCP-dense, 50 images, per-type count error; cost per image)}} \\\\')
    for m in M:
        v = d['dense_count'][m]; c = v['cost']
        L.append(f"\\quad {MN[m]} & {num(v['naive'])} (\\${c['naive']:.3f}) & {num(v['self_consistency'])} (\\${c['self_consistency']:.3f}), {rel(v['self_consistency_vs_naive'])} & n/a & {num(v['ours'])} (\\${c['ours']:.3f}), {rel(v['ours_vs_naive'])} \\\\")
    L.append('\\midrule')
    mlist = ', '.join(MN[m] for m in M[:-1]) + ' and ' + MN[M[-1]]
    ns = ', '.join(str(d['clicks'][m]['n']) for m in M[:-1]) + ' and ' + str(d['clicks'][M[-1]]['n'])
    L.append(f'\\multicolumn{{5}}{{@{{}}p{{15.2cm}}@{{}}}}{{\\emph{{Repair the view: clicks (ScreenSpot-Pro items with all four calls stored: {ns} items for {mlist}; miss rate; cost per item)}}}} \\\\')
    for m in M:
        v = d['clicks'][m]; c = v['cost']
        L.append(f"\\quad {MN[m]} & {cell(v['naive'],pc=True)} (\\${c['naive']:.4f}) & {cell(v['self_consistency'],pc=True)} (\\${c['self_consistency']:.4f}), {rel(v['self_consistency_vs_naive'])} & {cell(v['gated_zoom'],pc=True)} (\\${c['gated_zoom']:.4f}), {rel(v['gated_zoom_vs_naive'])} & {cell(v['ours'],pc=True)} (\\${c['ours']:.4f}), {rel(v['ours_vs_naive'])} \\\\")
    L.append('\\midrule')
    sets = d['sets']
    over = [k for k in sets if 'fsc147' not in k]
    import statistics
    med = lambda name: statistics.median([sets[k][f'{name}|E']['rel'] for k in over])
    cnt = lambda name: (sum(sets[k][f'{name}|E']['ci'][1] < 0 for k in over), sum(sets[k][f'{name}|E']['ci'][0] > 0 for k in over))
    mv, bf, ou, vb = cnt('majority_vote'), cnt('box_fusion'), cnt('ours'), cnt('ours_vs_box_fusion')
    L.append(f'\\multicolumn{{5}}{{@{{}}p{{15.2cm}}@{{}}}}{{\\emph{{Repair the set: detection and screen parsing ({len(over)} over-reporting cells, set error $E$, 3 calls each; cells better / worse than one call, median change)}}}} \\\\')
    L.append(f"\\quad {len(over)} cells & one call & majority vote: {mv[0]} / {mv[1]}, {pct(med('majority_vote'),1)} & box fusion: {bf[0]} / {bf[1]}, {pct(med('box_fusion'),1)} & {ou[0]} / {ou[1]}, {pct(med('ours'),1)}; against box fusion: {vb[0]} / {vb[1]} \\\\")
    L.append('\\midrule')
    L.append('\\multicolumn{5}{@{}p{15.2cm}@{}}{\\emph{Detect errors: clicks, share of a single call\'s misses caught by a disagreement alarm ($\\delta=1.5\\%$; Section~\\ref{sec:res:persist})}} \\\\')
    for m in M:
        v = al['models'][m]
        g = lambda k: [x for x in v[k]['points'] if abs(x['tau'] - 0.015) < 1e-9][0]
        L.append(f"\\quad {MN[m]} & n/a & same model: {g('same')['errors_caught']*100:.0f}\\% & n/a & second model: {g('cross')['errors_caught']*100:.0f}\\% \\\\")
    s = ('\\begin{table*}[tp]\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\n'
         '\\caption{Against common practice, from stored calls only. Columns give one call, the usual remedies (self-consistency: median of three stated counts, or a vote of three clicks; uncertainty-gated zoom: zoom only when two clicks disagree; majority vote and equal-weight box fusion over three calls), and our default. '
         'Relative changes are against one call, lower is better. ' + SIGNOTE + ' Our count readout costs about as much as one call and our click protocol less than a three-call vote. '
         'For Gemini 3.8, ``ours\'\' is the list-first protocol at its lowest setting; the method runs Gemini 3.8 at its default setting, which at the same 16k cap gives a readout error of 24.52 (Section~\\ref{sec:res:degen}). '
         'For F1 our default is one call; over the 15 over-reporting cells, majority vote and box fusion change F1 by a median of $-0.3\\%$ and $+0.2\\%$. Per-cell set errors are in the released analysis outputs.}\\label{tab:common}\n'
         '\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{1.8cm}>{\\raggedright\\arraybackslash}p{2.6cm}>{\\raggedright\\arraybackslash}p{3.5cm}>{\\raggedright\\arraybackslash}p{3.6cm}>{\\raggedright\\arraybackslash}p{3.65cm}@{}}\\toprule\n'
         '& One call & Common remedy & Second common remedy & Ours (Section~\\ref{sec:method}) \\\\\\midrule\n'
         + '\n'.join(L) + '\n\\bottomrule\\end{tabular}\\end{table*}\n')
    write('t7_common', s)


def t8_natural():
    d = j('natural_readout'); run = d['_run']; L = []
    for ds, lab in (('dota', 'DOTA'), ('doclaynet', 'DocLayNet')):
        for i, m in enumerate(M):
            v = d[f'{ds}|{m}']; pc = v['pure_readout_cf']; fp = v['full_protocol']
            L.append(f"{lab if i == 0 else ''} & {MN[m]} & {100 * v['list_recall_lf']:.0f} & {num(v['readout_gap_cf'], 1)} & {num(pc['a'], 1)} $\\to$ {num(pc['b'], 1)} & {mark(pct(pc['rel']).replace(chr(92) + '%', '') + ' ' + ci(pc['ci']), pc['ci'])} & "
                     f"{num(fp['b'], 1)} & {mark(pct(fp['rel']).replace(chr(92) + '%', '') + ' ' + ci(fp['ci']), fp['ci'])} \\\\")
        g = d[f'{ds}|gem38']['default16k_readout_vs_cf_stated']
        L.append(f" & Gemini 3.8, default & n/a & n/a & n/a & n/a & {num(g['b'], 1)} & {mark(pct(g['rel']).replace(chr(92) + '%', '') + ' ' + ci(g['ci']), g['ci'])} \\\\")
        if ds == 'dota': L.append('\\midrule')
    s = ('\\begin{table*}[tp]\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\n'
         f"\\caption{{Readout on aerial and document images with several classes: {run['images']['dota']} DOTA windows ({run['mean_objects']['dota']:.0f} objects and {run['mean_classes']['dota']:.1f} classes per image on average) and {run['images']['doclaynet']} DocLayNet pages ({run['mean_objects']['doclaynet']:.0f} elements, {run['mean_classes']['doclaynet']:.1f} classes); one call per image and arm, lowest setting unless stated, 16k cap. "
         'Per-type count error, lower is better; paired image bootstrap 95\\% intervals. Difficult DOTA objects and objects less than half inside the window are not targets, so records on them are counted like any other record, for every readout alike. \\emph{List recall}: share of true objects matched by a record centre inside a true box of the same class (list-first arm). \\emph{Readout gap}: mean per-type difference between the stated count and the list count of the same count-first response, which bounds how much switching readouts can change the error. '
         '\\emph{Same response}: stated count $\\to$ list count of the count-first response. \\emph{Full method}: readout of the list-first response (list count, or the stated count if the list is flagged) against the stated count of the count-first response; the Gemini 3.8 default row is the shipped configuration. ' + SIGNOTE + '}\\label{tab:natural}\n'
         '\\begin{tabular}{@{}llrrrrrr@{}}\\toprule\n'
         '& & & & \\multicolumn{2}{c}{Same response} & \\multicolumn{2}{c}{Full method} \\\\\\cmidrule(lr){5-6}\\cmidrule(l){7-8}\n'
         'Images & Model & ' + ' & '.join([H('List', 'recall (\\%)'), H('Readout', 'gap'), 'Error', 'Change, \\% [95\\% CI]', 'Error', 'Change, \\% [95\\% CI]']) + ' \\\\\\midrule\n'
         + '\n'.join(L) + '\n\\bottomrule\\end{tabular}\\end{table*}\n')
    write('t8_natural', s)



if __name__ == '__main__':
    t1(); t2(); t3(); t4(); t5(); t6(); tA_density(); t7_common(); t8_natural()
    print(sorted(p.name for p in TAB.iterdir()))
