#!/bin/sh
# Regenerates every main table from saved responses and labels. No model calls.
# Scripts in the second group score against DOTA v1.0, SKU-110K or FSC-147 labels, which are not redistributed.
# They run only after `python3 -I thirdparty/tp.py rebuild --sources <dir>` (see thirdparty/SOURCES.md);
# otherwise their shipped outputs in out/ are kept.
set -e
cd "$(dirname "$0")"
# --- group 1: needs no third-party labels beyond those shipped
python3 -I readout_effects.py      # Table 7
python3 -I rc16_setting.py         # Table 9
python3 -I alarm_curves.py         # Table 11
python3 -I readout_density.py      # Figure 4 (same-response readout across density)
python3 -I common_practice.py      # Table 6: against common practice
python3 -I opening_example.py      # Figure 1 example and exact-list counts
# --- group 2: needs the rebuilt third-party labels
if python3 -I ../../thirdparty/tp.py check; then
  python3 -I intersection_decomp.py  # Table 10 (exact identity checks)
  python3 -I prep1_v2.py             # Table 12 (persisted folds, stable seeds)
  python3 -I construction_table.py   # Table 13
  python3 -I natural_readout.py      # Table 8: readout on natural multi-class images (EXP-NR)
  python3 -I p1_predict.py           # deletion predicted from a few labelled images (Table 10, Section 7)
  python3 -I ../../fsr_pipeline/replay.py   # Section 5: the entry point replayed on stored calls and boundary cases
else
  echo "third-party labels not rebuilt: kept the shipped outputs of intersection_decomp, prep1_v2, construction_table,"
  echo "natural_readout, p1_predict and pipeline replay. Run: python3 -I thirdparty/tp.py rebuild --sources <dir>"
fi
python3 -I main_tables.py
