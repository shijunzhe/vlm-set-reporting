#!/bin/sh
# Regenerates every main table from saved responses and labels. No model calls.
# Scripts in the second group score against DOTA v1.0, SKU-110K or FSC-147 labels, which are not redistributed.
# They run only after `python3 -I thirdparty/tp.py rebuild --sources <dir>` (see thirdparty/SOURCES.md);
# otherwise their shipped outputs in out/ are kept.
set -e
cd "$(dirname "$0")"
# --- group 1: needs no third-party labels beyond those shipped
python3 -I readout_effects.py      # Table 8
python3 -I rc16_setting.py         # Table 10
python3 -I alarm_curves.py         # Table 12
python3 -I readout_density.py      # Figure 4 (same-response readout across density)
python3 -I common_practice.py      # Table 7: against common practice
python3 -I opening_example.py      # Figure 1 example and exact-list counts
# --- clicks from several models (Section 7.6) and configuration transfer (Section 7.9); ScreenSpot-Pro labels are shipped
python3 -I e3_loso_config.py                                   # Table 16 and the configuration menu (leave-one-source-out)
python3 -I accfirst/conf_eval.py --subset conf_v1              # Table 13: the 600-item evaluation, paired tests
python3 -I accfirst/conf_eval.py --subset main                 # the same frozen rules on the 260 click items of the GUI experiment
python3 -I accfirst/conf_report_extras.py --subset conf_v1     # comparator accuracy, clusters, costs
python3 -I accfirst/conf_coverage.py                           # rescue rates and items with no correct candidate (600 items)
python3 -I accfirst/conf_coverage.py --subset main --items main   # the same rates on the 260 click items of the GUI experiment
python3 -I accfirst/conf_singles.py                            # single-model flows and items with no point
python3 -I accfirst/conf_vote.py --subset conf_v1              # plain three-model vote (Table 13 row and paired test)
python3 -I accfirst/conf_vote.py --subset main                 # the same vote on the 260 click items of the GUI experiment
python3 -I accfirst/make_accuracy_tables.py                    # deploy costs; Tables 13, 16 and the configuration menu
python3 -I accfirst/d4_eval.py                                 # the four-source measurement
python3 -I accfirst/conf_archive.py --subset conf_v1 --out accfirst/archive_conf_v1   # per-item table
# --- group 2: needs the rebuilt third-party labels
if python3 -I ../../thirdparty/tp.py check; then
  python3 -I intersection_decomp.py  # Table 11 (exact identity checks)
  python3 -I prep1_v2.py             # Table 14 (persisted folds, stable seeds)
  python3 -I construction_table.py   # Table 15
  python3 -I natural_readout.py      # Table 9: readout on natural multi-class images (EXP-NR)
  python3 -I p1_predict.py           # deletion predicted from a few labelled images (Table 11, Section 7)
  python3 -I ../../fsr_pipeline/replay.py   # Section 5: the entry point replayed on stored calls and boundary cases
else
  echo "third-party labels not rebuilt: kept the shipped outputs of intersection_decomp, prep1_v2, construction_table,"
  echo "natural_readout, p1_predict and pipeline replay. Run: python3 -I thirdparty/tp.py rebuild --sources <dir>"
fi
python3 -I main_tables.py
