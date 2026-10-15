"""Builds every main table of the manuscript draft from the saved analysis outputs (out/*.json), which run_all.sh regenerates from
saved responses. No model calls. Output: out/main_tables.md"""
import json
from pathlib import Path
import numpy as np
O = Path(__file__).parent / 'out'; J = lambda n: json.load(open(O / n))
MN = {'gpt61': 'GPT-6.1', 'sonnet55': 'Sonnet 5.5', 'gem38': 'Gemini 3.8'}
L = []; w = L.append
def ci(x): return f"{x['rel']:+.1f}% [{x['ci'][0]:+.1f}, {x['ci'][1]:+.1f}]"

# T1 readout effects
R = J('readout_effects.json')
w('## Table 7. Readout effects (SCP-dense 50, min, per-type count error; paired image bootstrap)\n')
w('| Effect | Comparison | ' + ' | '.join(MN.values()) + ' |'); w('|---|---|---|---|---|')
for key, lab, cmp in (('pure_readout_old_protocol', 'Pure readout', 'same response, declared -> code count (count-first protocol)'),
                      ('pure_readout_new_protocol', 'Pure readout', 'same response, declared -> code count (list-first protocol)'),
                      ('normalization_new', 'Alias normalization', 'same list, before -> after alias map'),
                      ('protocol_fixed_readout', 'Protocol', 'mapped code count, count-first list -> list-first list'),
                      ('full_method', 'Full method', 'count-first declared -> list-first readout of draft 3.4')):
    w(f'| {lab} | {cmp} | ' + ' | '.join(f"{R[m][key]['a']} -> {R[m][key]['b']}, {ci(R[m][key])}" + (f" (n={R[m][key]['n']})" if R[m][key]['n'] != 50 else '') for m in MN) + ' |')
w('| Totals | same response, |list length - n| vs |declared total - n| | ' + ' | '.join(f"{R[m]['totals_same_response_new']['list_mae']} vs {R[m]['totals_same_response_new']['declared_mae']}" for m in MN) + ' |\n')

# T2 setting
S = J('rc16_setting.json')
w('## Table 9. Gemini 3.8 setting and output cap (list-first protocol, SCP-dense 50)\n')
w('| Configuration | Count readout error | Code count error | Set error E | Flagged lists | Strict-JSON failures | Truncated | Thinking share | Output tokens | Cost (50 calls) |'); w('|---|---|---|---|---|---|---|---|---|---|')
for k in ('min16k', 'default16k', 'default32k'):
    v = S['levels'][k]; w(f"| {k} | {v['readout']} | {v['code']} | {v['E']} | {v['flagged']:.0%} | {v['strict_fail']:.0%} | {v['trunc']:.0%} | {v['think_share']:.0%} | {v['out_tok']:.0f} | ${v['cost_total']} |")
for k in ('readout: default16k vs min16k', 'readout: default32k vs default16k', 'readout: default32k vs min16k'):
    w(f"\n{k}: {ci(S['contrasts'][k])}")
w('')

# T3 intersection decomposition
I = J('intersection_decomp.json'); s = I['_summary']
w('## Table 10. Intersection = selection + geometry (per image, E reduction; positive = better)\n')
w(f"Identity checks exact on every image: selection {s['identity_selection_all_exact']}, geometry {s['identity_geometry_all_exact']} ({s['pairs']} base/partner pairs x 60 images). "
  f"Geometry term positive in {s['geometry_positive_pairs']} pairs, negative in {s['geometry_negative_pairs']}. Sign of the fused total differs from the deletion-only sign in: {', '.join(s['sign_flips_due_to_geometry'])}.\n")
w('| Source | Base | Partner | Deleted scored / img | Matches lost / img | p1 (marginal) | Selection | Geometry | Total | Takeover |'); w('|---|---|---|---|---|---|---|---|---|---|')
for k, v in I.items():
    if k.startswith('_'): continue
    f, ds, a, b = k.split('|')
    w(f"| {f} {ds} | {MN[a]} | {'same model' if b == 'same' else MN[b]} | {v['deleted_scored_per_img']} | {v['matches_lost_per_img']} | {v['p1_marginal']} | {v['selection']:+.2f} | {v['geometry']:+.2f} | {v['total']:+.2f} | {v['takeover']:+.2f} |")
w('')

# T4 PREP-1 v2
P = J('prep1_v2.json')
w('## Table 12. Label-fitted output-feature selectors vs the constrained oracle (E3 sources, list prompt, lowest setting, K = 3; reduction of E relative to call 0)\n')
w('| Cell | Oracle (constrained) | Oracle (unrestricted atoms) | Images where they differ | Cross-fit logistic | Cross-fit boosted trees | In-sample logistic | 10 labels (median of 40) | Empirical gap (cross-fit logistic) [eval. CI] |'); w('|---|---|---|---|---|---|---|---|---|')
for k, v in P.items():
    if k.startswith('_'): continue
    ds, m = k.split('|')
    w(f"| {ds} {MN[m]} | {v['oracle_constrained']:.1%} | {v['oracle_unrestricted_atoms']:.1%} | {v['cases_unrestricted_ne_constrained']} | {v['crossfit_oracle_logit']:.1%} | {v['crossfit_oracle_gbt']:.1%} | {v['insample_oracle_logit']:.1%} | {v['labels10_oracle_logit_median']:.1%} | {v['gap_crossfit_oracle_logit']:.1%} [{v['gap_crossfit_oracle_logit_ci'][0]:.1%}, {v['gap_crossfit_oracle_logit_ci'][1]:.1%}] |")
sm = P['_summary']
w(f"| **median** | {sm['oracle_constrained']['median']:.1%} | {sm['oracle_unrestricted_atoms']['median']:.1%} | {sm['cases_unrestricted_ne_constrained_total']} (total) | {sm['crossfit_oracle_logit']['median']:.1%} | {sm['crossfit_oracle_gbt']['median']:.1%} | {sm['insample_oracle_logit']['median']:.1%} | {sm['labels10_oracle_logit_median']['median']:.1%} | {sm['gap_crossfit_oracle_logit']['median']:.1%} |\n")
w('The gap mixes information not in the features with selector and estimation shortfall (Prop. 6); it is not an information limit.\n')

# T5 defaults
C = J('construction_table.json')
w('## Table 13. Deployable defaults by leave-one-source-out transfer (change vs single call; significant = paired CI excludes 0)\n')
w('| Pool | Loss | Better | Worse | Not significant |'); w('|---|---|---|---|---|')
for k, v in C['default_transfer_summary'].items():
    pool, loss = k.split('|'); w(f"| {pool} | {loss} | {v['better']} | {v['worse']} | {v['ns']} |")
w('')
w('| Pool | Loss | Source | Model | Default applied | Single -> default (loss) | 95% CI of change | E | F1 | Recall | Cost / image | Cell best (reference) |'); w('|---|---|---|---|---|---|---|---|---|---|---|---|')
for k, v in C['default_transfer'].items():
    pool, fam, ds, loss, m = k.split('|'); om = v['other_metrics']
    w(f"| {pool} | {loss} | {fam} {ds} | {MN[m]} | {v['applied']} | {v['single']} -> {v['value']} | [{v['ci'][0]}, {v['ci'][1]}] | {om['E']} | {om['F1']} | {om['recall']} | ${om['cost_per_image']} | {v['cell_best_reference']} ({v['cell_best_value']}) |")
w('')

# T6 alarms
A = J('alarm_curves.json')
w('## Table 11. Click alarms (260 items; partner set by a fixed rule: cheapest other model)\n')
w('| Model | Partner | Alarm | tau | Coverage | Miss rate when acting | Errors caught |'); w('|---|---|---|---|---|---|---|')
for m, v in A['models'].items():
    for nm, key in (('same model, unzoomed', 'same'), ('cross model, unzoomed', 'cross'), ('cross model, after zoom', 'cross_after_zoom')):
        for p in v[key]['points']:
            if p['tau'] in (0.005, 0.015, 0.05):
                w(f"| {MN[m]} | {MN[v['partner']]} | {nm} | {p['tau']:.1%} | {p['coverage']:.1%} | {p['risk_when_acting']:.1%} | {p['errors_caught']:.0%} |")
w('\nMatched coverage (interpolated over the overlap of the two curves):\n')
for m, v in A['models'].items():
    seen = []; parts = []
    for x in v['matched_coverage']:
        if x['risk_same'] is None or x['risk_cross'] is None or x['coverage'] in seen: continue
        seen.append(x['coverage']); parts.append(f"coverage {x['coverage']:.1%}: same {x['risk_same']:.1%} vs cross {x['risk_cross']:.1%}")
    w(f"- {MN[m]}: " + '; '.join(parts) + ' (point estimates)')
(O / 'main_tables.md').write_text('\n'.join(L) + '\n'); print('written', O / 'main_tables.md', len(L), 'lines')
