#!/usr/bin/env bash
# Recompute every table and figure of the paper from the shipped responses. No model calls, no API keys.
# Optional: bash reproduce.sh --sources <dir>   first rebuilds the DOTA v1.0, SKU-110K and FSC-147 labels from the official
# downloads (thirdparty/SOURCES.md), so that the analyses that score against them are rerun too.
set -e
cd "$(dirname "$0")"
python3 tools/set_root.py
if [ "$1" = "--sources" ] && [ -n "$2" ]; then python3 -I thirdparty/tp.py rebuild --sources "$2"; fi
python3 -I thirdparty/tp.py status
cp -r stageB/analysis/out stageB/analysis/out_shipped 2>/dev/null || true
rm -rf stageB/analysis/accfirst_shipped; cp -r stageB/analysis/accfirst stageB/analysis/accfirst_shipped
python3 -I tools/check_protocol_hashes.py
bash stageB/analysis/run_all.sh
python3 -I paper/make_tables.py
python3 -I paper/figures/make_figures.py || echo "figures: Figure 1 panel (b) needs the ScreenSpot-Pro screenshot; see README"
# compare regenerated analysis outputs with the shipped ones
python3 - <<'PY'
import filecmp, os
diff = []
for a, b in (('stageB/analysis/out', 'stageB/analysis/out_shipped'), ('stageB/analysis/accfirst', 'stageB/analysis/accfirst_shipped'),
             ('stageB/analysis/accfirst/archive_conf_v1', 'stageB/analysis/accfirst_shipped/archive_conf_v1')):
    diff += [os.path.join(a, f) for f in sorted(os.listdir(b)) if os.path.isfile(os.path.join(b, f)) and not f.endswith('.py')
             and not filecmp.cmp(os.path.join(a, f), os.path.join(b, f), shallow=False)]
print('all analysis outputs identical to the shipped ones' if not diff else f'differs: {diff}')
PY
