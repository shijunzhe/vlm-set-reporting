## Table 7. Readout effects (SCP-dense 50, min, per-type count error; paired image bootstrap)

| Effect | Comparison | GPT-6.1 | Sonnet 5.5 | Gemini 3.8 |
|---|---|---|---|---|
| Pure readout | same response, declared -> code count (count-first protocol) | 10.84 -> 2.64, -75.6% [-83.2, -65.8] | 35.02 -> 7.8, -77.7% [-91.3, -58.4] | 45.68 -> 129.04, +182.5% [+163.3, +202.3] |
| Pure readout | same response, declared -> code count (list-first protocol) | 10.1 -> 1.6, -84.2% [-88.2, -79.6] | 18.52 -> 5.42, -70.7% [-76.9, -64.8] | 46.556 -> 95.044, +104.2% [+72.7, +139.5] (n=45) |
| Alias normalization | same list, before -> after alias map | 1.6 -> 1.6, +0.0% [+0.0, +0.0] | 5.42 -> 0.42, -92.3% [-97.0, -86.2] | 96.78 -> 96.78, +0.0% [+0.0, +0.0] |
| Protocol | mapped code count, count-first list -> list-first list | 2.64 -> 1.6, -39.4% [-57.6, -13.5] | 5.42 -> 0.42, -92.3% [-97.7, -44.4] | 129.04 -> 96.78, -25.0% [-32.7, -17.6] |
| Full method | count-first declared -> list-first readout of draft 3.4 | 10.84 -> 1.6, -85.2% [-89.1, -80.3] | 35.02 -> 0.42, -98.8% [-99.5, -98.1] | 45.68 -> 52.8, +15.6% [-5.3, +39.4] |
| Totals | same response, |list length - n| vs |declared total - n| | 0.04 vs 7.78 | 0.02 vs 10.48 | 109.38 vs 46.111 |

## Table 9. Gemini 3.8 setting and output cap (list-first protocol, SCP-dense 50)

| Configuration | Count readout error | Code count error | Set error E | Flagged lists | Strict-JSON failures | Truncated | Thinking share | Output tokens | Cost (50 calls) |
|---|---|---|---|---|---|---|---|---|---|
| min16k | 52.8 | 96.78 | 103.46 | 90% | 72% | 4% | 0% | 4796 | $0.952 |
| default16k | 24.52 | 25.9 | 34.58 | 12% | 70% | 8% | 100% | 8831 | $1.709 |
| default32k | 15.44 | 20.14 | 28.26 | 6% | 78% | 0% | 98% | 9487 | $1.832 |

readout: default16k vs min16k: -53.6% [-72.4, -29.5]

readout: default32k vs default16k: -37.0% [-62.4, +7.7]

readout: default32k vs min16k: -70.8% [-79.5, -57.9]

## Table 10. Intersection = selection + geometry (per image, E reduction; positive = better)

Identity checks exact on every image: selection True, geometry True (54 base/partner pairs x 60 images). Geometry term positive in 37 pairs, negative in 13. Sign of the fused total differs from the deletion-only sign in: E3|doclaynet|gpt61|sonnet55, GUI|groundcua|sonnet55|same.

| Source | Base | Partner | Deleted scored / img | Matches lost / img | p1 (marginal) | Selection | Geometry | Total | Takeover |
|---|---|---|---|---|---|---|---|---|---|
| E3 doclaynet | GPT-6.1 | same model | 0.917 | 0.483 | 0.5273 | -0.05 | -0.30 | -0.35 | +0.00 |
| E3 doclaynet | GPT-6.1 | Sonnet 5.5 | 2.633 | 0.917 | 0.3481 | +0.80 | -1.10 | -0.30 | +0.00 |
| E3 doclaynet | GPT-6.1 | Gemini 3.8 | 8.083 | 5.15 | 0.6371 | -2.22 | +0.43 | -1.78 | +0.00 |
| E3 doclaynet | Sonnet 5.5 | GPT-6.1 | 3.033 | 0.45 | 0.1484 | +2.13 | +2.77 | +4.90 | +0.00 |
| E3 doclaynet | Sonnet 5.5 | same model | 1.333 | 0.317 | 0.2375 | +0.70 | +0.10 | +0.80 | +0.00 |
| E3 doclaynet | Sonnet 5.5 | Gemini 3.8 | 8.3 | 3.967 | 0.4779 | +0.37 | +3.07 | +3.43 | +0.00 |
| E3 doclaynet | Gemini 3.8 | GPT-6.1 | 2.117 | 0.55 | 0.2598 | +1.02 | +0.47 | +1.48 | +0.00 |
| E3 doclaynet | Gemini 3.8 | Sonnet 5.5 | 1.933 | 0.217 | 0.1121 | +1.50 | +0.00 | +1.50 | +0.00 |
| E3 doclaynet | Gemini 3.8 | same model | 3.483 | 2.033 | 0.5837 | -0.58 | +0.30 | -0.28 | +0.00 |
| E3 dota | GPT-6.1 | same model | 7.067 | 0.433 | 0.0613 | +6.20 | +0.50 | +6.70 | +0.00 |
| E3 dota | GPT-6.1 | Sonnet 5.5 | 29.333 | 6.533 | 0.2227 | +16.27 | -0.65 | +15.62 | +0.00 |
| E3 dota | GPT-6.1 | Gemini 3.8 | 27.083 | 5.717 | 0.2111 | +15.65 | -1.50 | +14.15 | +0.00 |
| E3 dota | Sonnet 5.5 | GPT-6.1 | 6.2 | 0.317 | 0.0511 | +5.57 | +2.92 | +8.48 | -0.03 |
| E3 dota | Sonnet 5.5 | same model | 6.15 | 0.867 | 0.1409 | +4.42 | +0.05 | +4.47 | +0.00 |
| E3 dota | Sonnet 5.5 | Gemini 3.8 | 10.3 | 2.283 | 0.2217 | +5.73 | +1.25 | +6.98 | -0.03 |
| E3 dota | Gemini 3.8 | GPT-6.1 | 7.6 | 0.683 | 0.0899 | +6.23 | +3.90 | +10.13 | +0.00 |
| E3 dota | Gemini 3.8 | Sonnet 5.5 | 13.933 | 2.5 | 0.1794 | +8.93 | +1.17 | +10.10 | +0.00 |
| E3 dota | Gemini 3.8 | same model | 11.25 | 1.867 | 0.1659 | +7.52 | +0.82 | +8.33 | +0.00 |
| E3 fsc147 | GPT-6.1 | same model | 2.783 | 1.767 | 0.6347 | -0.75 | +0.27 | -0.48 | -0.13 |
| E3 fsc147 | GPT-6.1 | Sonnet 5.5 | 22.6 | 20.05 | 0.8872 | -17.50 | +0.00 | -17.50 | -0.20 |
| E3 fsc147 | GPT-6.1 | Gemini 3.8 | 27.283 | 24.517 | 0.8986 | -21.75 | +0.10 | -21.65 | +0.00 |
| E3 fsc147 | Sonnet 5.5 | GPT-6.1 | 4.533 | 2.1 | 0.4632 | +0.33 | +0.37 | +0.70 | -0.17 |
| E3 fsc147 | Sonnet 5.5 | same model | 7.35 | 4.817 | 0.6553 | -2.28 | +0.10 | -2.18 | -0.07 |
| E3 fsc147 | Sonnet 5.5 | Gemini 3.8 | 11.233 | 8.3 | 0.7389 | -5.37 | +0.03 | -5.33 | -0.10 |
| E3 fsc147 | Gemini 3.8 | GPT-6.1 | 0.817 | 0.317 | 0.3878 | +0.18 | +0.10 | +0.28 | -0.07 |
| E3 fsc147 | Gemini 3.8 | Sonnet 5.5 | 2.833 | 2.283 | 0.8059 | -1.73 | +0.13 | -1.60 | +0.00 |
| E3 fsc147 | Gemini 3.8 | same model | 3.533 | 3.117 | 0.8821 | -2.70 | +0.07 | -2.63 | +0.00 |
| E3 sku110k | GPT-6.1 | same model | 33.917 | 0.517 | 0.0152 | +32.88 | +5.83 | +38.72 | +0.00 |
| E3 sku110k | GPT-6.1 | Sonnet 5.5 | 153.817 | 43.05 | 0.2799 | +67.72 | -14.07 | +53.65 | +0.00 |
| E3 sku110k | GPT-6.1 | Gemini 3.8 | 113.817 | 22.833 | 0.2006 | +68.15 | +1.17 | +69.32 | +0.00 |
| E3 sku110k | Sonnet 5.5 | GPT-6.1 | 59.233 | 0.233 | 0.0039 | +58.77 | +13.83 | +72.60 | -0.33 |
| E3 sku110k | Sonnet 5.5 | same model | 61.95 | 2.95 | 0.0476 | +56.05 | +0.87 | +56.92 | -0.27 |
| E3 sku110k | Sonnet 5.5 | Gemini 3.8 | 73.15 | 5.567 | 0.0761 | +62.02 | +9.47 | +71.48 | -0.33 |
| E3 sku110k | Gemini 3.8 | GPT-6.1 | 13.067 | 0.4 | 0.0306 | +12.27 | +9.67 | +21.93 | +0.00 |
| E3 sku110k | Gemini 3.8 | Sonnet 5.5 | 66.983 | 25.867 | 0.3862 | +15.25 | -10.10 | +5.15 | +0.00 |
| E3 sku110k | Gemini 3.8 | same model | 36.883 | 13.017 | 0.3529 | +10.85 | +1.43 | +12.28 | +0.00 |
| GUI groundcua | GPT-6.1 | same model | 5.85 | 1.867 | 0.3191 | +2.12 | -0.67 | +1.45 | +0.00 |
| GUI groundcua | GPT-6.1 | Sonnet 5.5 | 34.583 | 4.0 | 0.1157 | +26.58 | +13.03 | +39.62 | +0.00 |
| GUI groundcua | GPT-6.1 | Gemini 3.8 | 43.417 | 17.017 | 0.3919 | +9.38 | -6.30 | +3.08 | +0.00 |
| GUI groundcua | Sonnet 5.5 | GPT-6.1 | 13.817 | 8.067 | 0.5838 | -2.32 | -6.50 | -8.82 | +0.00 |
| GUI groundcua | Sonnet 5.5 | same model | 5.167 | 3.017 | 0.5839 | -0.87 | +1.20 | +0.33 | +0.00 |
| GUI groundcua | Sonnet 5.5 | Gemini 3.8 | 38.9 | 27.833 | 0.7155 | -16.77 | -7.73 | -24.50 | +0.00 |
| GUI groundcua | Gemini 3.8 | GPT-6.1 | 26.317 | 1.833 | 0.0697 | +22.65 | +9.60 | +32.25 | +0.00 |
| GUI groundcua | Gemini 3.8 | Sonnet 5.5 | 42.567 | 2.45 | 0.0576 | +37.67 | +15.43 | +53.10 | -0.03 |
| GUI groundcua | Gemini 3.8 | same model | 30.983 | 6.917 | 0.2232 | +17.15 | -0.93 | +16.22 | +0.00 |
| GUI screenparse | GPT-6.1 | same model | 0.383 | 0.083 | 0.2174 | +0.22 | +0.07 | +0.28 | +0.00 |
| GUI screenparse | GPT-6.1 | Sonnet 5.5 | 4.9 | 1.4 | 0.2857 | +2.10 | +0.93 | +3.03 | +0.00 |
| GUI screenparse | GPT-6.1 | Gemini 3.8 | 4.117 | 1.25 | 0.3036 | +1.62 | +0.03 | +1.65 | +0.00 |
| GUI screenparse | Sonnet 5.5 | GPT-6.1 | 1.8 | 0.633 | 0.3519 | +0.53 | -0.47 | +0.07 | +0.00 |
| GUI screenparse | Sonnet 5.5 | same model | 0.8 | 0.3 | 0.375 | +0.20 | +0.00 | +0.20 | +0.00 |
| GUI screenparse | Sonnet 5.5 | Gemini 3.8 | 2.833 | 1.417 | 0.5 | +0.00 | -0.37 | -0.37 | +0.00 |
| GUI screenparse | Gemini 3.8 | GPT-6.1 | 1.567 | 0.3 | 0.1915 | +0.97 | +0.47 | +1.43 | +0.00 |
| GUI screenparse | Gemini 3.8 | Sonnet 5.5 | 3.383 | 0.933 | 0.2759 | +1.52 | +0.87 | +2.38 | +0.00 |
| GUI screenparse | Gemini 3.8 | same model | 2.167 | 0.983 | 0.4538 | +0.20 | +0.00 | +0.20 | +0.00 |

## Table 12. Label-fitted output-feature selectors vs the constrained oracle (E3 sources, list prompt, lowest setting, K = 3; reduction of E relative to call 0)

| Cell | Oracle (constrained) | Oracle (unrestricted atoms) | Images where they differ | Cross-fit logistic | Cross-fit boosted trees | In-sample logistic | 10 labels (median of 40) | Empirical gap (cross-fit logistic) [eval. CI] |
|---|---|---|---|---|---|---|---|---|
| doclaynet GPT-6.1 | 63.2% | 63.2% | 0 | -8.4% | -7.4% | 1.6% | -7.7% | 71.6% [60.1%, 87.4%] |
| doclaynet Sonnet 5.5 | 59.8% | 59.8% | 0 | 17.3% | 14.8% | 22.6% | 16.6% | 42.5% [34.2%, 54.0%] |
| doclaynet Gemini 3.8 | 45.2% | 45.2% | 0 | 5.9% | 1.3% | 6.9% | 3.2% | 39.3% [30.1%, 50.3%] |
| dota GPT-6.1 | 74.3% | 74.4% | 1 | 39.9% | 39.6% | 40.4% | 39.6% | 34.4% [25.1%, 42.6%] |
| dota Sonnet 5.5 | 41.9% | 41.9% | 1 | 23.3% | 22.6% | 24.1% | 21.3% | 18.6% [13.7%, 25.1%] |
| dota Gemini 3.8 | 49.3% | 49.3% | 0 | 26.8% | 24.6% | 27.5% | 20.5% | 22.5% [16.7%, 30.1%] |
| fsc147 GPT-6.1 | 16.7% | 19.0% | 12 | 0.2% | -0.7% | 1.6% | -0.7% | 16.5% [7.6%, 31.7%] |
| fsc147 Sonnet 5.5 | 16.6% | 17.3% | 9 | 1.1% | 1.0% | 1.2% | 0.8% | 15.5% [10.6%, 21.4%] |
| fsc147 Gemini 3.8 | 6.0% | 6.2% | 2 | 3.9% | 4.1% | 4.2% | 2.0% | 2.1% [0.8%, 4.0%] |
| sku110k GPT-6.1 | 71.8% | 71.8% | 0 | 42.0% | 39.9% | 42.2% | 40.7% | 29.8% [27.2%, 32.7%] |
| sku110k Sonnet 5.5 | 45.3% | 45.3% | 0 | 34.9% | 34.8% | 35.1% | 34.6% | 10.3% [8.4%, 12.6%] |
| sku110k Gemini 3.8 | 44.6% | 44.6% | 0 | 13.9% | 14.3% | 14.3% | 9.8% | 30.6% [26.4%, 35.6%] |
| **median** | 45.2% | 45.2% | 25 (total) | 15.6% | 14.6% | 18.4% | 13.2% | 26.1% |

The gap mixes information not in the features with selector and estimation shortfall (Prop. 6); it is not an information limit.

## Table 13. Deployable defaults by leave-one-source-out transfer (change vs single call; significant = paired CI excludes 0)

| Pool | Loss | Better | Worse | Not significant |
|---|---|---|---|---|
| same_model_only | E | 9 | 2 | 7 |
| same_model_only | F1 | 2 | 4 | 12 |
| all | E | 12 | 2 | 4 |
| all | F1 | 10 | 3 | 5 |

| Pool | Loss | Source | Model | Default applied | Single -> default (loss) | 95% CI of change | E | F1 | Recall | Cost / image | Cell best (reference) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| same_model_only | E | E3 doclaynet | GPT-6.1 | unan3_fus | 8.5667 -> 9.15 | [-0.35, 1.65] | 9.15 | 0.6825 | 0.6774 | $0.03228 | x3_maj_fus (7.417) |
| same_model_only | E | E3 doclaynet | Sonnet 5.5 | unan3_fus | 13.7667 -> 12.3333 | [-2.25, -0.7] | 12.333 | 0.5759 | 0.5711 | $0.03835 | x3_maj_fus (7.383) |
| same_model_only | E | E3 doclaynet | Gemini 3.8 | unan3_fus | 11.8333 -> 12.1833 | [-0.6333, 1.4167] | 12.183 | 0.2631 | 0.2296 | $0.01317 | x3_maj_fus (7.417) |
| same_model_only | E | E3 dota | GPT-6.1 | unan3_fus | 45.0 -> 34.9833 | [-16.7338, -4.9667] | 34.983 | 0.4088 | 0.4644 | $0.07851 | x3_unan_del (28.383) |
| same_model_only | E | E3 dota | Sonnet 5.5 | unan3_fus | 37.8667 -> 31.95 | [-8.7004, -3.7662] | 31.95 | 0.333 | 0.3209 | $0.04723 | x3_unan_fus (29.017) |
| same_model_only | E | E3 dota | Gemini 3.8 | unan3_fus | 40.9833 -> 30.8 | [-16.3833, -5.3833] | 30.8 | 0.2964 | 0.2577 | $0.01781 | x3_unan_fus (29.367) |
| same_model_only | E | E3 fsc147 | GPT-6.1 | unan3_fus | 30.1167 -> 30.9 | [-0.1004, 1.7833] | 30.9 | 0.6761 | 0.6373 | $0.03339 | maj3_fus (30.017) |
| same_model_only | E | E3 fsc147 | Sonnet 5.5 | unan3_fus | 48.3167 -> 51.7333 | [2.0667, 4.8333] | 51.733 | 0.3522 | 0.2919 | $0.01874 | x2_union|gpt61 (31.483) |
| same_model_only | E | E3 fsc147 | Gemini 3.8 | unan3_fus | 52.05 -> 54.6333 | [0.4167, 5.4167] | 54.633 | 0.2186 | 0.1893 | $0.00965 | x2_union|gpt61 (30.367) |
| same_model_only | E | E3 sku110k | GPT-6.1 | unan3_fus | 198.35 -> 148.3167 | [-58.6667, -41.8] | 148.317 | 0.5066 | 0.5294 | $0.21213 | x2_inter_fus|gem38 (129.033) |
| same_model_only | E | E3 sku110k | Sonnet 5.5 | unan3_fus | 217.3 -> 148.6667 | [-80.4342, -57.8167] | 148.667 | 0.1299 | 0.0836 | $0.12584 | x3_unan_fus (137.95) |
| same_model_only | E | E3 sku110k | Gemini 3.8 | unan3_fus | 150.9667 -> 136.4833 | [-23.9, -6.1833] | 136.483 | 0.2359 | 0.1919 | $0.05839 | x2_inter_fus|gpt61 (129.033) |
| same_model_only | E | GUI groundcua | GPT-6.1 | unan3_fus | 92.5833 -> 89.25 | [-7.4, 0.5] | 89.25 | 0.4679 | 0.4923 | $0.092 | x2_inter_fus|sonnet55 (52.967) |
| same_model_only | E | GUI groundcua | Sonnet 5.5 | unan3_fus | 44.15 -> 43.6667 | [-3.1337, 2.4333] | 43.667 | 0.7116 | 0.6798 | $0.09992 | maj3_fus (42.05) |
| same_model_only | E | GUI groundcua | Gemini 3.8 | unan3_fus | 121.75 -> 95.5833 | [-34.9004, -17.9496] | 95.583 | 0.1792 | 0.1443 | $0.0438 | x2_inter_fus|sonnet55 (68.65) |
| same_model_only | E | GUI screenparse | GPT-6.1 | unan3_fus | 16.2333 -> 15.7833 | [-0.8333, -0.1] | 15.783 | 0.5648 | 0.5974 | $0.02919 | x2_inter_fus|sonnet55 (13.2) |
| same_model_only | E | GUI screenparse | Sonnet 5.5 | unan3_fus | 13.2667 -> 12.9667 | [-0.75, 0.1667] | 12.967 | 0.613 | 0.5969 | $0.03175 | x2_inter_del|gpt61 (12.733) |
| same_model_only | E | GUI screenparse | Gemini 3.8 | unan3_fus | 16.0167 -> 15.5333 | [-1.2833, 0.2] | 15.533 | 0.5021 | 0.4766 | $0.01089 | x3_unan_fus (13.55) |
| same_model_only | F1 | E3 doclaynet | GPT-6.1 | union3 | 0.7102 -> 0.6857 | [-0.0486, -0.0033] | 9.4 | 0.6857 | 0.7081 | $0.03228 | x3_maj_fus (0.7206) |
| same_model_only | F1 | E3 doclaynet | Sonnet 5.5 | union3 | 0.5729 -> 0.5685 | [-0.0208, 0.0135] | 15.15 | 0.5685 | 0.6288 | $0.03835 | x3_maj_fus (0.7212) |
| same_model_only | F1 | E3 doclaynet | Gemini 3.8 | union3 | 0.3842 -> 0.459 | [0.0276, 0.1288] | 12.8 | 0.459 | 0.4843 | $0.01317 | x3_maj_fus (0.7206) |
| same_model_only | F1 | E3 dota | GPT-6.1 | union3 | 0.3945 -> 0.3889 | [-0.0236, 0.0116] | 60.433 | 0.3889 | 0.5094 | $0.07851 | x3_maj_fus (0.4187) |
| same_model_only | F1 | E3 dota | Sonnet 5.5 | union3 | 0.3335 -> 0.3293 | [-0.0219, 0.0143] | 45.867 | 0.3293 | 0.3759 | $0.04723 | x3_maj_fus (0.4182) |
| same_model_only | F1 | E3 dota | Gemini 3.8 | union3 | 0.3124 -> 0.319 | [-0.0247, 0.044] | 49.817 | 0.319 | 0.3613 | $0.01781 | x3_maj_fus (0.4106) |
| same_model_only | F1 | E3 fsc147 | GPT-6.1 | maj3_fus | 0.7067 -> 0.6953 | [-0.0436, 0.0079] | 30.017 | 0.6953 | 0.6703 | $0.03339 | x2_union|sonnet55 (0.7165) |
| same_model_only | F1 | E3 fsc147 | Sonnet 5.5 | maj3_fus | 0.4524 -> 0.4252 | [-0.0461, -0.0084] | 49.0 | 0.4252 | 0.3547 | $0.01874 | x2_union|gpt61 (0.7165) |
| same_model_only | F1 | E3 fsc147 | Gemini 3.8 | maj3_fus | 0.2733 -> 0.2773 | [-0.0336, 0.0423] | 52.767 | 0.2773 | 0.2452 | $0.00965 | x3_union (0.715) |
| same_model_only | F1 | E3 sku110k | GPT-6.1 | union3 | 0.4367 -> 0.4023 | [-0.0465, -0.0235] | 254.633 | 0.4023 | 0.5409 | $0.21213 | unan3_fus (0.5066) |
| same_model_only | F1 | E3 sku110k | Sonnet 5.5 | union3 | 0.1278 -> 0.1295 | [-0.0074, 0.0105] | 296.483 | 0.1295 | 0.1566 | $0.12584 | x3_maj_fus (0.4212) |
| same_model_only | F1 | E3 sku110k | Gemini 3.8 | union3 | 0.3542 -> 0.4019 | [0.0224, 0.0798] | 166.4 | 0.4019 | 0.3904 | $0.05839 | x2_union|gpt61 (0.4294) |
| same_model_only | F1 | GUI groundcua | GPT-6.1 | union3 | 0.4831 -> 0.4624 | [-0.0317, -0.0104] | 100.833 | 0.4624 | 0.5271 | $0.092 | x2_inter_fus|sonnet55 (0.6242) |
| same_model_only | F1 | GUI groundcua | Sonnet 5.5 | union3 | 0.7174 -> 0.7301 | [-0.0016, 0.0289] | 43.833 | 0.7301 | 0.7452 | $0.09992 | union2 (0.7324) |
| same_model_only | F1 | GUI groundcua | Gemini 3.8 | union3 | 0.256 -> 0.2864 | [-0.0035, 0.0666] | 159.233 | 0.2864 | 0.3741 | $0.0438 | x2_union|sonnet55 (0.5574) |
| same_model_only | F1 | GUI screenparse | GPT-6.1 | union3 | 0.5638 -> 0.5662 | [-0.0042, 0.0089] | 16.367 | 0.5662 | 0.6182 | $0.02919 | x3_maj_fus (0.5962) |
| same_model_only | F1 | GUI screenparse | Sonnet 5.5 | union3 | 0.6148 -> 0.6145 | [-0.0089, 0.01] | 13.6 | 0.6145 | 0.6291 | $0.03175 | union2 (0.6165) |
| same_model_only | F1 | GUI screenparse | Gemini 3.8 | union3 | 0.5357 -> 0.5466 | [-0.0072, 0.0307] | 16.7 | 0.5466 | 0.5879 | $0.01089 | x3_maj_fus (0.5962) |
| all | E | E3 doclaynet | GPT-6.1 | x2_inter_fus|sonnet55 | 8.5667 -> 8.8667 | [-1.2333, 1.7667] | 8.867 | 0.6836 | 0.6512 | $0.02362 | x3_maj_fus (7.417) |
| all | E | E3 doclaynet | Sonnet 5.5 | x2_inter_fus|gpt61 | 13.7667 -> 8.8667 | [-6.6333, -3.3667] | 8.867 | 0.6836 | 0.6512 | $0.02362 | x3_maj_fus (7.383) |
| all | E | E3 doclaynet | Gemini 3.8 | x2_inter_fus|gpt61 | 11.8333 -> 10.35 | [-2.6667, -0.35] | 10.35 | 0.4056 | 0.3571 | $0.01508 | x3_maj_fus (7.417) |
| all | E | E3 dota | GPT-6.1 | x3_maj_fus | 45.0 -> 31.4333 | [-21.9013, -7.2654] | 31.433 | 0.4187 | 0.4096 | $0.04815 | x3_unan_del (28.383) |
| all | E | E3 dota | Sonnet 5.5 | x3_maj_fus | 37.8667 -> 31.4667 | [-10.4008, -3.15] | 31.467 | 0.4182 | 0.4091 | $0.04815 | x3_unan_fus (29.017) |
| all | E | E3 dota | Gemini 3.8 | x3_maj_fus | 40.9833 -> 31.85 | [-13.8338, -5.0162] | 31.85 | 0.4106 | 0.4005 | $0.04815 | x3_unan_fus (29.367) |
| all | E | E3 fsc147 | GPT-6.1 | x2_inter_fus|sonnet55 | 30.1167 -> 47.6167 | [11.2825, 24.4833] | 47.617 | 0.4305 | 0.3561 | $0.01747 | maj3_fus (30.017) |
| all | E | E3 fsc147 | Sonnet 5.5 | x2_inter_fus|gpt61 | 48.3167 -> 47.6167 | [-2.2837, 0.75] | 47.617 | 0.4305 | 0.3561 | $0.01747 | x2_union|gpt61 (31.483) |
| all | E | E3 fsc147 | Gemini 3.8 | x2_inter_fus|gpt61 | 52.05 -> 51.7667 | [-0.8, 0.0833] | 51.767 | 0.2707 | 0.2344 | $0.01419 | x2_union|gpt61 (30.367) |
| all | E | E3 sku110k | GPT-6.1 | x3_maj_fus | 198.35 -> 138.55 | [-72.3, -47.7] | 138.55 | 0.4245 | 0.3816 | $0.13525 | x2_inter_fus|gem38 (129.033) |
| all | E | E3 sku110k | Sonnet 5.5 | x3_maj_fus | 217.3 -> 139.4333 | [-92.0179, -64.6667] | 139.433 | 0.4212 | 0.3791 | $0.13525 | x3_unan_fus (137.95) |
| all | E | E3 sku110k | Gemini 3.8 | x3_maj_fus | 150.9667 -> 139.1333 | [-17.2, -6.7833] | 139.133 | 0.4218 | 0.3785 | $0.13525 | x2_inter_fus|gpt61 (129.033) |
| all | E | GUI groundcua | GPT-6.1 | x3_maj_fus | 92.5833 -> 68.75 | [-29.7013, -18.2667] | 68.75 | 0.5653 | 0.5574 | $0.0787 | x2_inter_fus|sonnet55 (52.967) |
| all | E | GUI groundcua | Sonnet 5.5 | x3_maj_fus | 44.15 -> 67.0167 | [17.9167, 27.9004] | 67.017 | 0.5766 | 0.5687 | $0.0787 | maj3_fus (42.05) |
| all | E | GUI groundcua | Gemini 3.8 | x3_maj_fus | 121.75 -> 71.0333 | [-59.0504, -42.6996] | 71.033 | 0.5509 | 0.5431 | $0.0787 | x2_inter_fus|sonnet55 (68.65) |
| all | E | GUI screenparse | GPT-6.1 | x3_maj_fus | 16.2333 -> 14.0167 | [-3.1833, -1.2833] | 14.017 | 0.5962 | 0.6129 | $0.02412 | x2_inter_fus|sonnet55 (13.2) |
| all | E | GUI screenparse | Sonnet 5.5 | x3_maj_fus | 13.2667 -> 13.8167 | [-0.5833, 1.7] | 13.817 | 0.6005 | 0.6167 | $0.02412 | x2_inter_del|gpt61 (12.733) |
| all | E | GUI screenparse | Gemini 3.8 | x3_maj_fus | 16.0167 -> 14.0167 | [-2.85, -1.2833] | 14.017 | 0.5962 | 0.6128 | $0.02412 | x3_unan_fus (13.55) |
| all | F1 | E3 doclaynet | GPT-6.1 | x2_union|sonnet55 | 0.7102 -> 0.6597 | [-0.1006, 0.0006] | 11.8 | 0.6597 | 0.7284 | $0.02362 | x3_maj_fus (0.7206) |
| all | F1 | E3 doclaynet | Sonnet 5.5 | x2_union|gpt61 | 0.5729 -> 0.6597 | [0.0557, 0.1205] | 11.8 | 0.6597 | 0.7284 | $0.02362 | x3_maj_fus (0.7212) |
| all | F1 | E3 doclaynet | Gemini 3.8 | x2_union|gpt61 | 0.3842 -> 0.6742 | [0.2149, 0.368] | 9.15 | 0.6742 | 0.7359 | $0.01508 | x3_maj_fus (0.7206) |
| all | F1 | E3 dota | GPT-6.1 | x2_union|sonnet55 | 0.3945 -> 0.3694 | [-0.0518, 0.0033] | 51.217 | 0.3694 | 0.4911 | $0.04189 | x3_maj_fus (0.4187) |
| all | F1 | E3 dota | Sonnet 5.5 | x2_union|gpt61 | 0.3335 -> 0.3694 | [-0.0248, 0.0954] | 51.217 | 0.3694 | 0.4911 | $0.04189 | x3_maj_fus (0.4182) |
| all | F1 | E3 dota | Gemini 3.8 | x2_union|gpt61 | 0.3124 -> 0.3656 | [0.0006, 0.1072] | 52.733 | 0.3656 | 0.4884 | $0.03237 | x3_maj_fus (0.4106) |
| all | F1 | E3 fsc147 | GPT-6.1 | x3_maj_fus | 0.7067 -> 0.4578 | [-0.328, -0.1733] | 45.583 | 0.4578 | 0.3895 | $0.02071 | x2_union|sonnet55 (0.7165) |
| all | F1 | E3 fsc147 | Sonnet 5.5 | x3_maj_fus | 0.4524 -> 0.4567 | [-0.0233, 0.0341] | 45.667 | 0.4567 | 0.3884 | $0.02071 | x2_union|gpt61 (0.7165) |
| all | F1 | E3 fsc147 | Gemini 3.8 | x3_maj_fus | 0.2733 -> 0.458 | [0.1214, 0.2519] | 45.583 | 0.458 | 0.3895 | $0.02071 | x3_union (0.715) |
| all | F1 | E3 sku110k | GPT-6.1 | x2_union|sonnet55 | 0.4367 -> 0.3393 | [-0.116, -0.0794] | 271.183 | 0.3393 | 0.4608 | $0.11551 | unan3_fus (0.5066) |
| all | F1 | E3 sku110k | Sonnet 5.5 | x2_union|gpt61 | 0.1278 -> 0.3393 | [0.1815, 0.2437] | 271.183 | 0.3393 | 0.4608 | $0.11551 | x3_maj_fus (0.4212) |
| all | F1 | E3 sku110k | Gemini 3.8 | x2_union|gpt61 | 0.3542 -> 0.4294 | [0.0352, 0.1208] | 209.45 | 0.4294 | 0.5183 | $0.08983 | x2_union|gpt61 (0.4294) |
| all | F1 | GUI groundcua | GPT-6.1 | x2_union|sonnet55 | 0.4831 -> 0.5967 | [0.0906, 0.1369] | 79.033 | 0.5967 | 0.7005 | $0.06401 | x2_inter_fus|sonnet55 (0.6242) |
| all | F1 | GUI groundcua | Sonnet 5.5 | x2_union|gpt61 | 0.7174 -> 0.5967 | [-0.1434, -0.0978] | 79.033 | 0.5967 | 0.7005 | $0.06401 | union2 (0.7324) |
| all | F1 | GUI groundcua | Gemini 3.8 | x2_union|gpt61 | 0.256 -> 0.412 | [0.1118, 0.2047] | 123.067 | 0.412 | 0.5034 | $0.04541 | x2_union|sonnet55 (0.5574) |
| all | F1 | GUI screenparse | GPT-6.1 | x3_maj_fus | 0.5638 -> 0.5962 | [0.0112, 0.0546] | 14.017 | 0.5962 | 0.6129 | $0.02412 | x3_maj_fus (0.5962) |
| all | F1 | GUI screenparse | Sonnet 5.5 | x3_maj_fus | 0.6148 -> 0.6005 | [-0.0414, 0.0126] | 13.817 | 0.6005 | 0.6167 | $0.02412 | union2 (0.6165) |
| all | F1 | GUI screenparse | Gemini 3.8 | x3_maj_fus | 0.5357 -> 0.5962 | [0.0401, 0.0838] | 14.017 | 0.5962 | 0.6128 | $0.02412 | x3_maj_fus (0.5962) |

## Table 11. Click alarms (260 items; partner set by a fixed rule: cheapest other model)

| Model | Partner | Alarm | tau | Coverage | Miss rate when acting | Errors caught |
|---|---|---|---|---|---|---|
| GPT-6.1 | Gemini 3.8 | same model, unzoomed | 0.5% | 93.1% | 7.0% | 26% |
| GPT-6.1 | Gemini 3.8 | same model, unzoomed | 1.5% | 95.8% | 7.6% | 17% |
| GPT-6.1 | Gemini 3.8 | same model, unzoomed | 5.0% | 97.7% | 7.5% | 17% |
| GPT-6.1 | Gemini 3.8 | cross model, unzoomed | 0.5% | 65.4% | 3.5% | 74% |
| GPT-6.1 | Gemini 3.8 | cross model, unzoomed | 1.5% | 85.8% | 4.0% | 61% |
| GPT-6.1 | Gemini 3.8 | cross model, unzoomed | 5.0% | 92.3% | 5.4% | 43% |
| GPT-6.1 | Gemini 3.8 | cross model, after zoom | 0.5% | 77.7% | 2.5% | 75% |
| GPT-6.1 | Gemini 3.8 | cross model, after zoom | 1.5% | 87.7% | 2.6% | 70% |
| GPT-6.1 | Gemini 3.8 | cross model, after zoom | 5.0% | 92.3% | 3.3% | 60% |
| Sonnet 5.5 | Gemini 3.8 | same model, unzoomed | 0.5% | 87.7% | 10.5% | 47% |
| Sonnet 5.5 | Gemini 3.8 | same model, unzoomed | 1.5% | 91.1% | 12.2% | 36% |
| Sonnet 5.5 | Gemini 3.8 | same model, unzoomed | 5.0% | 93.1% | 13.6% | 27% |
| Sonnet 5.5 | Gemini 3.8 | cross model, unzoomed | 0.5% | 58.9% | 3.3% | 89% |
| Sonnet 5.5 | Gemini 3.8 | cross model, unzoomed | 1.5% | 78.5% | 6.4% | 71% |
| Sonnet 5.5 | Gemini 3.8 | cross model, unzoomed | 5.0% | 86.2% | 8.9% | 56% |
| Sonnet 5.5 | Gemini 3.8 | cross model, after zoom | 0.5% | 77.7% | 1.5% | 89% |
| Sonnet 5.5 | Gemini 3.8 | cross model, after zoom | 1.5% | 85.4% | 2.7% | 78% |
| Sonnet 5.5 | Gemini 3.8 | cross model, after zoom | 5.0% | 89.6% | 3.4% | 70% |
| Gemini 3.8 | Sonnet 5.5 | same model, unzoomed | 0.5% | 78.1% | 19.2% | 45% |
| Gemini 3.8 | Sonnet 5.5 | same model, unzoomed | 1.5% | 90.4% | 22.1% | 27% |
| Gemini 3.8 | Sonnet 5.5 | same model, unzoomed | 5.0% | 95.8% | 24.1% | 15% |
| Gemini 3.8 | Sonnet 5.5 | cross model, unzoomed | 0.5% | 58.9% | 9.2% | 80% |
| Gemini 3.8 | Sonnet 5.5 | cross model, unzoomed | 1.5% | 78.5% | 16.7% | 52% |
| Gemini 3.8 | Sonnet 5.5 | cross model, unzoomed | 5.0% | 86.2% | 19.2% | 39% |
| Gemini 3.8 | Sonnet 5.5 | cross model, after zoom | 0.5% | 77.7% | 2.5% | 80% |
| Gemini 3.8 | Sonnet 5.5 | cross model, after zoom | 1.5% | 85.4% | 3.6% | 68% |
| Gemini 3.8 | Sonnet 5.5 | cross model, after zoom | 5.0% | 89.6% | 5.1% | 52% |

Matched coverage (interpolated over the overlap of the two curves):

- GPT-6.1: coverage 93.1%: same 7.0% vs cross 5.8% (point estimates)
- Sonnet 5.5: coverage 87.7%: same 10.5% vs cross 10.1%; coverage 88.5%: same 11.3% vs cross 10.6%; coverage 89.2%: same 12.1% vs cross 11.2% (point estimates)
- Gemini 3.8: coverage 78.1%: same 19.2% vs cross 16.5%; coverage 83.7%: same 20.5% vs cross 18.8%; coverage 89.2%: same 21.9% vs cross 20.7% (point estimates)
