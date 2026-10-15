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


def write(name, s):
    (TAB / f'{name}.tex').write_text(s)


# ---------------------------------------------------------------- Table 7: readout effects
def t1():
    d = j('readout_effects')
    rows = [('Pure readout, count-first', 'same response', 'pure_readout_old_protocol'),
            ('Pure readout, list-first', 'same response', 'pure_readout_new_protocol'),
            ('Alias map', 'same list', 'normalization_new'),
            ('Protocol', 'mapped list count', 'protocol_fixed_readout'),
            ('Full method', 'count-first stated $\\to$ readout', 'full_method')]
    L = []
    for lab, comp, k in rows:
        cells = []
        for m in M:
            v = d[m][k]
            if v['rel'] == 0.0 and v['ci'] == [0.0, 0.0]:
                cells.append(f"{num(v['a'])} $\\to$ {num(v['b'])} (no change)")
            else:
                n = f" (n={v['n']})" if v['n'] != 50 else ''
                cells.append(f"\\begin{{tabular}}{{@{{}}c@{{}}}}{num(v['a'])} $\\to$ {num(v['b'])}{n}\\\\ {pct(v['rel'])} {ci(v['ci'])}\\end{{tabular}}")
        L.append(f"{lab} & {comp} & " + ' & '.join(cells) + ' \\\\')
    tot = ' & '.join(f"{num(d[m]['totals_same_response_new']['list_mae'])} vs.\\ {num(d[m]['totals_same_response_new']['declared_mae'])}" for m in M)
    L.append('\\midrule')
    L.append(f'Totals, list-first & $|$list length$-n|$ vs.\\ $|$stated total$-n|$ & {tot} \\\\')
    s = ('\\begin{table*}[tp]\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\n'
         '\\caption{Readout effects kept separate (SCP-dense, 50 images, lowest setting, one call; per-type count error; paired image bootstrap 95\\% intervals). '
         '\\emph{Pure readout} compares the stated count with the list count of the list in the same response, without the alias map. '
         '\\emph{Alias map} applies the declared map to the same list. \\emph{Protocol} compares count-first and list-first lists under the same mapped readout. '
         '\\emph{Full method} compares the stated count of a count-first response with the readout of Section~\\ref{sec:method} on a list-first response. '
         'This is the effect of the reporting and decoding protocol.}\\label{tab:readout}\n'
         '\\begin{tabular}{@{}p{2.3cm}p{2.5cm}ccc@{}}\\toprule\n'
         'Effect & Comparison & GPT-6.1 & Sonnet 5.5 & Gemini 3.8 (lowest) \\\\\\midrule\n' + '\n'.join(L) +
         '\n\\bottomrule\\end{tabular}\\end{table*}\n')
    write('t1_readout', s)


# ---------------------------------------------------------------- Table 9: Gemini setting and cap
def t2():
    d = j('rc16_setting'); lv = d['levels']; c = d['contrasts']
    names = {'min16k': 'lowest, 16k', 'default16k': 'default, 16k', 'default32k': 'default, 32k'}
    L = []
    for k in ['min16k', 'default16k', 'default32k']:
        v = lv[k]
        L.append(f"{names[k]} & {num(v['readout'])} & {num(v['code'])} & {num(v['E'])} & {num(v['n_unknown'])} & {num(v['E_retain'])} & {v['flagged']*100:.0f}\\% & {v['strict_fail']*100:.0f}\\% & {v['trunc']*100:.0f}\\% & {v['think_share']*100:.0f}\\% & {v['out_tok']:.0f} & \\${v['cost_total']:.2f} \\\\")
    a = c['readout: default16k vs min16k']; b = c['readout: default32k vs default16k']
    s = ('\\begin{table*}[tp]\\centering\\footnotesize\\setlength{\\tabcolsep}{2.5pt}\n'
         '\\caption{Gemini 3.8 on SCP-dense under the list-first protocol (50 images, one call each). Changing the reasoning setting at a fixed 16k output cap lowers the count readout error by '
         f"{pct(a['rel'])} {ci(a['ci'])}; raising the cap to 32k changes it by a further {pct(b['rel'])} {ci(b['ci'])} (not significant). "
         'Readout error: per-type count error of the Step-6 readout (stated count if the list is flagged, list count otherwise). List-count error: the same for the list count alone. '
         'Set error $E$ scores the delivered records, those whose class is valid after the alias map (equal class, 20 px); unknown-class records are excluded from the delivered set and reported in the next column; '
         'the column after it is a sensitivity that keeps them and scores them as unmatched. '
         'Flagged lists: share with at least one degeneracy flag. JSON fail: share of responses that do not parse as the requested object (records are then recovered by the declared salvage step). '
         'The setting comparison is cap-matched, not cost-matched: output tokens and cost rise with reasoning. Cost counts every attempted call.}\\label{tab:setting}\n'
         '\\begin{tabular}{@{}lccccccccccc@{}}\\toprule\n'
         '\\shortstack{Setting,\\\\cap} & \\shortstack{Readout\\\\error} & \\shortstack{List-count\\\\error} & \\shortstack{Set\\\\error $E$} & \\shortstack{Unknown\\\\records} & \\shortstack{$E$, un-\\\\known kept} & Flagged & \\shortstack{JSON\\\\fail} & \\shortstack{Trun-\\\\cated} & \\shortstack{Reasoning\\\\share} & \\shortstack{Output\\\\tokens} & Cost \\\\\\midrule\n'
         + '\n'.join(L) + '\n\\bottomrule\\end{tabular}\\end{table*}\n')
    write('t2_setting', s)


# ---------------------------------------------------------------- Table 10: intersection decomposition
def t3():
    d = j('intersection_decomp'); pr = j('p1_predict')['pairs']
    # compact: by source
    L = []
    for fam, src, lab in SRC:
        rows = [(k, v) for k, v in d.items() if isinstance(v, dict) and k.startswith(f'{fam}|{src}|')]
        th = np.mean([pr[k]['sign_correct_threshold'] for k, _ in rows])
        helps = sum(v['selection'] > 0 for _, v in rows)
        gpos = sum(v['geometry'] > 0 for _, v in rows); gneg = sum(v['geometry'] < 0 for _, v in rows)
        flips = sum((v['total'] > 0) != (v['selection'] > 0) for _, v in rows if v['selection'] != 0)
        p1 = [v['p1_marginal'] for _, v in rows]
        sel = [v['selection'] for _, v in rows]; geo = [v['geometry'] for _, v in rows]
        L.append(f"{lab} & {len(rows)} & {100 * th:.0f}\\% & {helps} & {min(p1):.2f} to {max(p1):.2f} & {num(min(sel),1)} to {num(max(sel),1)} & {gpos} / {gneg} & {num(min(geo),1)} to {num(max(geo),1)} & {flips} \\\\")
    s = ('\\begin{table*}[tp]\\centering\\small\\setlength{\\tabcolsep}{4pt}\n'
         '\\caption{Confirmation by a partner call, split into selection and geometry (Propositions~\\ref{prop:identity} and \\ref{prop:geometry}). Each source has 9 base/partner pairs (3 base models $\\times$ same-model or one of two other models), 60 images, one call each; values are per-image reductions of $E$ (positive is better). '
         'Both identities hold exactly on every image of every pair. \\emph{Sign from 5 labels}: $p_1$ is estimated on 5 labelled images and Proposition~\\ref{thm:intersect} is used to predict whether deletion lowers $E$ on the other 55; the share of correct predictions over 200 draws per pair, averaged over the pairs (Section~\\ref{sec:res:set}). Sign flips: pairs in which the fused total has the opposite sign of the deletion-only selection term. Full per-pair values are in Appendix~\\ref{app:tables}.}\\label{tab:intersect}\n'
         '\\resizebox{\\textwidth}{!}{\\begin{tabular}{@{}lcccccccc@{}}\\toprule\n'
         'Source & Pairs & Sign from 5 labels & Deletion helps & $p_1$ range & Selection range & Geometry $+$ / $-$ & Geometry range & Sign flips \\\\\\midrule\n'
         + '\n'.join(L) + '\n\\bottomrule\\end{tabular}}\\end{table*}\n')
    write('t3_intersect', s)
    # full appendix table
    L = []
    for fam, src, lab in SRC:
        for b in M:
            for p in [x for x in M if x != b][:0] + ['same'] + [x for x in M if x != b]:
                k = f'{fam}|{src}|{b}|{p}'
                if k not in d: continue
                v = d[k]
                L.append(f"{lab} & {MN[b]} & {MN[p]} & {v['deleted_scored_per_img']:.2f} & {v['matches_lost_per_img']:.2f} & {v['p1_marginal']:.3f} & {num(v['selection'])} & {num(v['geometry'])} & {num(v['total'])} \\\\")
        L.append('\\midrule')
    L = L[:-1]
    s = ('\\begin{longtable}{@{}lllcccccc@{}}\n'
         '\\caption{Selection and geometry terms for every base/partner pair (per image, reduction of $E$; positive is better). $\\Delta c$: deleted scored records; $\\Delta r$: matches lost.}\\label{tab:intersect_full}\\\\\\toprule\n'
         'Source & Base & Partner & $\\Delta c$ & $\\Delta r$ & $p_1$ & Selection & Geometry & Total \\\\\\midrule\\endfirsthead\n'
         '\\toprule Source & Base & Partner & $\\Delta c$ & $\\Delta r$ & $p_1$ & Selection & Geometry & Total \\\\\\midrule\\endhead\n'
         + '\n'.join(L) + '\n\\bottomrule\\end{longtable}\n')
    write('tA_intersect_full', s)


# ---------------------------------------------------------------- Table 12: selection
def t4():
    d = j('prep1_v2')
    L = []
    for _, src, lab in SRC[:4]:
        for m in M:
            v = d[f'{src}|{m}']
            g = v['gap_crossfit_oracle_logit_ci']
            L.append(f"{lab} & {MN[m]} & {v['oracle_constrained']*100:.1f} & {v['oracle_unrestricted_atoms']*100:.1f} & {v['cases_unrestricted_ne_constrained']} & "
                     f"{num(v['crossfit_oracle_logit']*100,1)} & {num(v['crossfit_oracle_gbt']*100,1)} & {num(v['insample_oracle_logit']*100,1)} & {num(v['labels10_oracle_logit_median']*100,1)} & "
                     f"{(v['oracle_constrained']-v['crossfit_oracle_logit'])*100:.1f} {ci([x*100 for x in g])} \\\\")
    s_ = d['_summary']
    tot = sum(d[f'{src}|{m}']['cases_unrestricted_ne_constrained'] for _, src, _ in SRC[:4] for m in M)
    gaps = sorted((d[f'{src}|{m}']['oracle_constrained'] - d[f'{src}|{m}']['crossfit_oracle_logit']) * 100 for _, src, _ in SRC[:4] for m in M)
    med_gap = (gaps[5] + gaps[6]) / 2
    L.append('\\midrule')
    L.append(f"\\multicolumn{{2}}{{@{{}}l}}{{Median of 12 cells}} & {s_['oracle_constrained']['median']*100:.1f} & {s_['oracle_unrestricted_atoms']['median']*100:.1f} & {tot} (total) & "
             f"{s_['crossfit_oracle_logit']['median']*100:.1f} & {s_['crossfit_oracle_gbt']['median']*100:.1f} & {s_['insample_oracle_logit']['median']*100:.1f} & {s_['labels10_oracle_logit_median']['median']*100:.1f} & {med_gap:.1f} \\\\")
    s = ('\\begin{table*}[tp]\\centering\\small\\setlength{\\tabcolsep}{3.5pt}\n'
         '\\caption{Selection among three calls of one model (E3, list interface, lowest setting, 60 images per cell; reduction of $E$ relative to the first call, in \\%). '
         'The constrained oracle optimizes over the same feasible family as the selectors (Proposition~\\ref{prop:oracle}); the unrestricted value lets two members of one cluster match two targets and is not attainable. '
         'Selectors are logistic or boosted-tree record gates on output features; they read no labels at inference but are \\emph{fitted} with labels (5-fold cross-fitting by image with persisted folds; ``10 labels\'\' is the median over 40 draws of 10 labelled images, each evaluated on the images it did not use). '
         'The gap (oracle minus cross-fitted logistic, with an evaluation interval) mixes information absent from the features with selector and estimation shortfall (Proposition~\\ref{prop:order}); it is not an information limit.}\\label{tab:select}\n'
         '\\resizebox{\\textwidth}{!}{\\begin{tabular}{@{}llccccccccc@{}}\\toprule\n'
         'Source & Model & Oracle & Unrestricted & Differ & Cross-fit logistic & Cross-fit trees & In-sample logistic & 10 labels & Gap \\\\\\midrule\n'
         + '\n'.join(L) + '\n\\bottomrule\\end{tabular}}\\end{table*}\n')
    write('t4_select', s)


# ---------------------------------------------------------------- Table 13: transfer
def t5():
    d = j('construction_table'); dt = d['default_transfer']; sm = d['default_transfer_summary']
    single = collections.defaultdict(dict)
    for k, v in dt.items():
        pool, fam, src, loss, m = k.split('|'); single[(fam, src, loss)][m] = v['single']

    def cell(pool, fam, src, loss, m):
        v = dt[f'{pool}|{fam}|{src}|{loss}|{m}']
        rel = (v['value'] - v['single']) / v['single'] * 100
        lo, hi = v['ci']
        better = (hi < 0) if loss == 'E' else (lo > 0); worse = (lo > 0) if loss == 'E' else (hi < 0)
        s = single[(fam, src, loss)]; strongest = (min(s, key=s.get) if loss == 'E' else max(s, key=s.get)) == m
        t = pct(rel, 0)
        if better: t = '\\textbf{' + t + '}'
        elif worse: t = '\\underline{' + t + '}'
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
    s = ('\\begin{table*}[tp]\\centering\\small\\setlength{\\tabcolsep}{2.6pt}\n'
         '\\caption{Defaults evaluated by leave-one-source-out transfer: for each source, the construction is the one that is best on the other five sources (same model only, or any construction including cross-model ones), and it is evaluated on the evaluated source against a single call of the user\'s model. '
         'Entries are relative changes of the loss (for $E$, negative is better; for F1, positive is better). \\textbf{Bold}: paired 95\\% interval excludes 0 in the favourable direction; \\underline{underlined}: significantly worse. '
         '$^{\\star}$: the user\'s model is the strongest single model on that source and loss. Every significantly worse cross-model entry carries a $\\star$. Costs, recall and the chosen construction per cell are in Appendix~\\ref{app:tables}.}\\label{tab:transfer}\n'
         '\\resizebox{\\textwidth}{!}{\\begin{tabular}{@{}l' + 'ccc' * 4 + '@{}}\\toprule\n'
         '& \\multicolumn{6}{c}{Same model only} & \\multicolumn{6}{c}{Any construction (incl.\\ cross-model)} \\\\\\cmidrule(lr){2-7}\\cmidrule(l){8-13}\n'
         '& \\multicolumn{3}{c}{$E$} & \\multicolumn{3}{c}{F1} & \\multicolumn{3}{c}{$E$} & \\multicolumn{3}{c}{F1} \\\\\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\\cmidrule(l){11-13}\n'
         'Source' + ' & GPT & Son. & Gem.' * 4 + ' \\\\\\midrule\n' + '\n'.join(L) + '\n\\bottomrule\\end{tabular}}\\end{table*}\n')
    write('t5_transfer', s)
    # appendix full
    CN = lambda c: c.replace('_', '\\_').replace('|', ' w/ ').replace('gpt61', 'GPT').replace('sonnet55', 'Sonnet').replace('gem38', 'Gemini')
    L = []
    for pool, loss in [('same_model_only', 'E'), ('same_model_only', 'F1'), ('all', 'E'), ('all', 'F1')]:
        for fam, src, lab in SRC:
            for m in M:
                v = dt[f'{pool}|{fam}|{src}|{loss}|{m}']; o = v['other_metrics']
                nd = 2 if loss == 'E' else 3
                L.append(f"{'same' if pool.startswith('same') else 'any'} & {loss} & {lab} & {MN[m]} & {CN(v['applied'])} & {num(v['single'],nd)} $\\to$ {num(v['value'],nd)} & {ci(v['ci'], 3 if loss=='F1' else 2)} & {o['recall']:.3f} & \\${o['cost_per_image']:.4f} \\\\")
        L.append('\\midrule')
    L = L[:-1]
    s = ('{\\small\\setlength{\\tabcolsep}{3pt}\\begin{longtable}{@{}llllllccc@{}}\n'
         '\\caption{Leave-one-source-out transfer, every cell. Construction names: unan3 / maj3 / union3 = three same-model calls kept by unanimity / majority / union; x2\\_inter, x2\\_union = two-model intersection or union with the named partner; x3\\_maj = three-model majority; \\_fus = fused coordinates, \\_del = deletion only. Interval: paired 95\\% bootstrap of the change. Cost per image counts every call.}\\label{tab:transfer_full}\\\\\\toprule\n'
         'Pool & Loss & Source & Model & Default applied & Single $\\to$ default & 95\\% CI & Recall & Cost/img \\\\\\midrule\\endfirsthead\n'
         '\\toprule Pool & Loss & Source & Model & Default applied & Single $\\to$ default & 95\\% CI & Recall & Cost/img \\\\\\midrule\\endhead\n'
         + '\n'.join(L) + '\n\\bottomrule\\end{longtable}}\n')
    write('tA_transfer_full', s)


# ---------------------------------------------------------------- Table 11: alarms
def t6():
    d = j('alarm_curves')
    L = []
    for m in M:
        v = d['models'][m]
        for k, lab in [('same', 'same model, unzoomed'), ('cross', 'cross model, unzoomed'), ('same_after_zoom', 'same model, after zoom'), ('cross_after_zoom', 'cross model, after zoom')]:
            p = [x for x in v[k]['points'] if abs(x['tau'] - 0.015) < 1e-9][0]
            L.append(f"{MN[m] if k=='same' else ''} & {MN[v['partner']] if k=='same' else ''} & {lab} & {100*(1-v[k]['base_accuracy']):.1f}\\% & {p['coverage']*100:.1f}\\% & {p['risk_when_acting']*100:.1f}\\% & {p['errors_caught']*100:.0f}\\% \\\\")
        if m != M[-1]: L.append('\\midrule')
    mc = []
    for m in M:
        seen = set(); parts = []
        for r in d['models'][m]['matched_coverage']:
            key = (round(r['coverage'], 4), round(r['risk_same'], 4), round(r['risk_cross'], 4))
            if key in seen: continue
            seen.add(key); parts.append(f"{r['coverage']*100:.1f}\\%: {r['risk_same']*100:.1f} vs.\\ {r['risk_cross']*100:.1f}")
        mc.append(f"{MN[m]} ({'; '.join(parts)})")
    mz = []
    for m in M:
        seen = set(); parts = []
        for r in d['models'][m]['matched_coverage_after_zoom']:
            key = (round(r['coverage'], 4), round(r['risk_same'], 4), round(r['risk_cross'], 4))
            if key in seen: continue
            seen.add(key); parts.append(f"{r['coverage']*100:.1f}\\%: {r['risk_same']*100:.1f} vs.\\ {r['risk_cross']*100:.1f}")
        mz.append(f"{MN[m]} ({'; '.join(parts) if parts else 'no overlap'})")
    s = ('\\begin{table*}[tp]\\centering\\small\\setlength{\\tabcolsep}{4pt}\n'
         '\\caption{Disagreement alarms for clicks (ScreenSpot-Pro and ScreenSpot-v2, 260 items; partner set by a fixed rule as the cheapest other model). The alarm compares the acted-on point with a second point for the same item and fires when the two are farther apart than $\\delta=1.5\\%$ of the longer image side; the item is then deferred (sent for review or another step) instead of acted on. Unzoomed alarms compare first clicks; after zoom, the acted-on point is the zoomed first click, and the second point is the same model\'s zoomed second click or the partner\'s zoomed first click. '
         'Coverage: share of items acted on; miss rate: share of acted-on items whose point is outside the target; errors caught: share of the acted-on prediction\'s errors (the first click unzoomed, the zoomed first click after zoom) that the alarm defers. These deferrals measure risk and coverage; the automatic tier of Algorithm~\\ref{alg:clicks} defers nothing. Other $\\delta$ values are in Figure~\\ref{fig:alarm}. '
         'Where the same-model and cross-model curves overlap, miss rates at matched coverage (same vs.\\ cross, in \\%, point estimates without intervals) are, unzoomed: ' + '; '.join(mc) + '; after zoom: ' + '; '.join(mz) + '.}\\label{tab:alarm}\n'
         '\\resizebox{\\textwidth}{!}{\\begin{tabular}{@{}lllcccc@{}}\\toprule\n'
         'Model & Partner & Alarm & Error rate, no alarm & Coverage & Miss rate when acting & Errors caught \\\\\\midrule\n'
         + '\n'.join(L) + '\n\\bottomrule\\end{tabular}}\\end{table*}\n')
    write('t6_alarm', s)


# ---------------------------------------------------------------- Appendix: readout across density
def tA_density():
    d = j('readout_density')
    names = [('px', 'PixMo-Count', 'total'), ('fsc', 'FSC-147', 'total'), ('scp', 'SCP', 'per type'), ('scpd', 'SCP-dense', 'per type')]
    L = []
    for t, lab, kind in names:
        for m in M:
            v = d[f'{t}|{m}']
            mapped = f"{num(v['code'])}, {pct(v['rel'])} {ci(v['ci'])}" if t in ('scp', 'scpd') else 'same'
            L.append(f"{lab if m=='gpt61' else ''} & {v['mean_objects'] if m=='gpt61' else ''} & {kind if m=='gpt61' else ''} & {MN[m]} & {v['n']} & {num(v['stated'])} & {num(v['code_raw'])}, {pct(v['rel_raw'])} {ci(v['ci_raw'])} & {mapped} \\\\")
        L.append('\\midrule')
    L = L[:-1]
    s = ('\\begin{table}[h]\\centering\\small\\setlength{\\tabcolsep}{4pt}\n'
         '\\caption{Same-response readout across datasets (count-first prompts, lowest setting, one call; data of Figure~\\ref{fig:density}). List count without and with the declared alias map; paired image bootstrap (a separate run from Table~\\ref{tab:readout}, so shared rows can differ in the last digit of an interval).}\\label{tab:density}\n'
         '\\resizebox{\\textwidth}{!}{\\begin{tabular}{@{}lclllccc@{}}\\toprule\n'
         'Dataset & Objects & Error & Model & $n$ & Stated & List count (unmapped) & List count (alias map) \\\\\\midrule\n'
         + '\n'.join(L) + '\n\\bottomrule\\end{tabular}}\\end{table}\n')
    write('tA_density', s)


# ---------------------------------------------------------------- Table: against common practice
def t7_common():
    d = j('common_practice'); al = j('alarm_curves')
    def cell(v, nd=2, pc=False):
        x = f"{v*100:.1f}\\%" if pc else num(v, nd)
        return x
    def rel(b):
        t = pct(b['rel'], 1)
        if b['ci'][1] < 0: t = '\\textbf{' + t + '}'
        elif b['ci'][0] > 0: t = '\\underline{' + t + '}'
        return t
    L = []
    L.append('\\multicolumn{5}{@{}l}{\\emph{Repair the readout: dense counting (SCP-dense, 50 images, per-type count error; cost per image)}} \\\\')
    for m in M:
        v = d['dense_count'][m]; c = v['cost']
        L.append(f"\\quad {MN[m]} & {num(v['naive'])} (\\${c['naive']:.3f}) & {num(v['self_consistency'])} (\\${c['self_consistency']:.3f}), {rel(v['self_consistency_vs_naive'])} & -- & {num(v['ours'])} (\\${c['ours']:.3f}), {rel(v['ours_vs_naive'])} \\\\")
    L.append('\\midrule')
    L.append('\\multicolumn{5}{@{}l}{\\emph{Repair the view: clicks (ScreenSpot-Pro items with all four calls stored, click error; cost per item)}} \\\\')
    for m in M:
        v = d['clicks'][m]; c = v['cost']
        L.append(f"\\quad {MN[m]} ($n={v['n']}$) & {cell(v['naive'],pc=True)} (\\${c['naive']:.4f}) & {cell(v['self_consistency'],pc=True)} (\\${c['self_consistency']:.4f}), {rel(v['self_consistency_vs_naive'])} & {cell(v['gated_zoom'],pc=True)} (\\${c['gated_zoom']:.4f}), {rel(v['gated_zoom_vs_naive'])} & {cell(v['ours'],pc=True)} (\\${c['ours']:.4f}), {rel(v['ours_vs_naive'])} \\\\")
    L.append('\\midrule')
    ss = d['sets_summary']; sets = d['sets']
    over = [k for k in sets if 'fsc147' not in k]
    import statistics
    med = lambda name: statistics.median([sets[k][f'{name}|E']['rel'] for k in over])
    cnt = lambda name: (sum(sets[k][f'{name}|E']['ci'][1] < 0 for k in over), sum(sets[k][f'{name}|E']['ci'][0] > 0 for k in over))
    mv, bf, ou = cnt('majority_vote'), cnt('box_fusion'), cnt('ours')
    L.append('\\multicolumn{5}{@{}l}{\\emph{Repair the set: detection and screen parsing (15 over-reporting cells, set error $E$, 3 calls each; better / worse vs.\\ one call, median change)}} \\\\')
    L.append(f"\\quad 15 cells & one call & majority vote: {mv[0]} / {mv[1]}, {pct(med('majority_vote'),1)} & box fusion: {bf[0]} / {bf[1]}, {pct(med('box_fusion'),1)} & {ou[0]} / {ou[1]}, {pct(med('ours'),1)} \\\\")
    L.append(f"\\quad vs.\\ box fusion & & & & better in {ss['ours_vs_box_fusion|E']['better']} of 18 cells, worse in {ss['ours_vs_box_fusion|E']['worse']} \\\\")
    L.append('\\midrule')
    L.append('\\multicolumn{5}{@{}l}{\\emph{Detect errors: clicks, share of a single call\'s errors caught by a disagreement alarm ($\\delta=1.5\\%$)}} \\\\')
    for m in M:
        v = al['models'][m]
        g = lambda k: [x for x in v[k]['points'] if abs(x['tau'] - 0.015) < 1e-9][0]
        L.append(f"\\quad {MN[m]} & -- & second call, same model: {g('same')['errors_caught']*100:.0f}\\% & -- & second model: {g('cross')['errors_caught']*100:.0f}\\% \\\\")
    s = ('\\begin{table*}[tp]\\centering\\small\\setlength{\\tabcolsep}{4pt}\n'
         '\\caption{Against common practice, from stored calls only. Columns give the common baseline, the usual remedies (self-consistency: median of three stated counts, or a vote of three clicks; uncertainty-gated zoom: zoom only when two clicks disagree; majority vote and equal-weight box fusion over three calls), and our default. '
         'Relative changes are against one call (\\textbf{bold}: paired 95\\% interval below 0; \\underline{underlined}: above 0). Our count readout costs about as much as one call and our click protocol less than a three-call vote. '
         'For Gemini 3.8, ``ours\'\' is the list-first protocol at its lowest setting; the method runs Gemini at its default setting, which at the same 16k cap gives a readout error of 24.52 (Table~\\ref{tab:setting}). '
         'For F1 our default is one call; over the 15 over-reporting cells, majority vote and box fusion change F1 by a median of $-0.3\\%$ and $+0.2\\%$. Per-cell values are in \\texttt{stageB/analysis/out/common\\_practice.json}.}\\label{tab:common}\n'
         '\\resizebox{\\textwidth}{!}{\\begin{tabular}{@{}lllll@{}}\\toprule\n'
         '& One call & Common remedy & Second common remedy & Ours (Section~\\ref{sec:method}) \\\\\\midrule\n'
         + '\n'.join(L) + '\n\\bottomrule\\end{tabular}}\\end{table*}\n')
    write('t7_common', s.replace(' -- ', ' n/a '))


# ---------------------------------------------------------------- Table: natural multi-class readout (EXP-NR)
def t8_natural():
    d = j('natural_readout'); run = d['_run']; L = []
    for ds, lab in (('dota', 'DOTA'), ('doclaynet', 'DocLayNet')):
        for m in M:
            v = d[f'{ds}|{m}']; pc = v['pure_readout_cf']; fp = v['full_protocol']
            L.append(f"{lab} & {MN[m]} & {100 * v['list_recall_lf']:.0f}\\% & {num(v['readout_gap_cf'], 1)} & {num(pc['a'], 1)} $\\to$ {num(pc['b'], 1)} & {pct(pc['rel'])} {ci(pc['ci'])} & "
                     f"{num(fp['b'], 1)} & {pct(fp['rel'])} {ci(fp['ci'])} \\\\")
        g = d[f'{ds}|gem38']['default16k_readout_vs_cf_stated']
        L.append(f"{lab} & Gemini 3.8, default & & & & & {num(g['b'], 1)} & {pct(g['rel'])} {ci(g['ci'])} \\\\")
    s = ('\\begin{table*}[tp]\\centering\\small\\setlength{\\tabcolsep}{4pt}\n'
         f"\\caption{{Readout on aerial and document images with several classes: {run['images']['dota']} DOTA windows ({run['mean_objects']['dota']:.0f} objects and {run['mean_classes']['dota']:.1f} classes per image on average) and {run['images']['doclaynet']} DocLayNet pages ({run['mean_objects']['doclaynet']:.0f} elements, {run['mean_classes']['doclaynet']:.1f} classes); one call per image and arm, lowest setting unless stated, 16k cap. "
         'Per-type count error; paired image bootstrap. Difficult DOTA objects and objects less than half inside the window are not targets, so records on them are counted like any other record, for every readout alike. \\emph{List recall}: share of true objects matched by a record centre inside a true box of the same class (list-first arm). \\emph{Readout gap}: mean per-type difference between the stated count and the list count of the same count-first response, which bounds how much switching readouts can change the error. '
         '\\emph{Same response}: stated count $\\to$ list count of the count-first response. \\emph{Protocol}: readout of the list-first response (list count, or the stated count if the list is flagged) against the stated count of the count-first response; the Gemini default row is the shipped configuration.}\\label{tab:natural}\n'
         '\\resizebox{\\textwidth}{!}{\\begin{tabular}{@{}llcccccc@{}}\\toprule\n'
         'Images & Model & List recall & Readout gap & Same response & Change & Protocol readout & Change vs.\\ stated count \\\\\\midrule\n'
         + '\n'.join(L) + '\n\\bottomrule\\end{tabular}}\\end{table*}\n')
    write('t8_natural', s)


if __name__ == '__main__':
    t1(); t2(); t3(); t4(); t5(); t6(); tA_density(); t7_common(); t8_natural()
    print(sorted(p.name for p in TAB.iterdir()))
