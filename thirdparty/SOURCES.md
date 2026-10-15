# Third-party data: sources, licences and rebuild

No third-party image is included in this repository. Third-party labels are handled in two ways.

## Labels kept in the repository (permissive licences)

The item lists and the labels of the items we used are included, with attribution, under the licence of each source.
Download the images from the source.

| Dataset | Used in | Source | Licence |
|---|---|---|---|
| DocLayNet v1.2 | detection experiment (E3), natural readout (EXP-NR) | https://huggingface.co/datasets/docling-project/DocLayNet-v1.2 | CDLA-Permissive-1.0 |
| ScreenSpot-Pro | clicks, comparator, resolution and end-to-end runs; the 600-item click evaluation | https://huggingface.co/datasets/likaixin/ScreenSpot-Pro | MIT |
| ScreenSpot-v2 | clicks | https://huggingface.co/datasets/OS-Copilot/ScreenSpot-v2 | Apache-2.0 |
| GroundCUA | screen element sets | https://huggingface.co/datasets/ServiceNow/GroundCUA | MIT |
| ScreenParse | screen element sets | https://huggingface.co/datasets/docling-project/screenparse | CC BY 4.0 |
| PixMo-Count | readout across density | https://huggingface.co/datasets/allenai/pixmo-count | ODC-BY 1.0 |

Check each dataset card for the current terms before reuse.

## Labels not redistributed (rebuilt from the official downloads)

The terms of these datasets do not clearly allow redistribution, so their labels are not in the repository.
Every value that came from them is replaced by a placeholder such as

```json
{"__tp__": {"src": "dota", "label_file": "P0300.txt", "window": [388, 0, 1024, 839], "field": "targets"}}
```

The placeholder names the official file and the transformation we applied: the window, the scale and the rounding.
What stays in the repository is item metadata: the image name, the window or scale, the class name and the number of objects.
Regions we added ourselves are also kept, namely the ignore regions from our label audit (`items_corr.jsonl`).

| Dataset | Files with placeholders | Official source | File needed |
|---|---|---|---|
| DOTA v1.0 (val) | `fsr_v51/e3/data/dota/*.jsonl`, `stageB/runs/nr/items.json` (DOTA entries) | https://captain-whu.github.io/DOTA/dataset.html; the same labels are in Ultralytics' `DOTAv1.zip` (about 2.0 GB, includes images): https://github.com/ultralytics/assets/releases/download/v0.0.0/DOTAv1.zip | v1.0 val labels in the original format (`P0003.txt`, ...) |
| SKU-110K | `fsr_v51/e3/data/sku110k/*.jsonl` | https://github.com/eg4000/SKU110K_CVPR19; archive (about 12.2 GB, includes images): http://trax-geometry.s3.amazonaws.com/cvpr_challenge/SKU110K_fixed.tar.gz | `SKU110K_fixed/annotations/annotations_test.csv` |
| FSC-147 | `fsr_v51/e3/data/fsc147/*.jsonl`, `stageB/runs/b1p/items.json`, `stageB/runs/b1ppilot/items.json` | https://github.com/cvlab-stonybrook/LearningToCountEverything | `data/annotation_FSC147_384.json` (45.6 MB), https://raw.githubusercontent.com/cvlab-stonybrook/LearningToCountEverything/master/data/annotation_FSC147_384.json |

Place the files in one directory:

```
<sources>/dota/DOTAv1/labels/val_original/P0003.txt ...      (layout of DOTAv1.zip)
   or     <sources>/dota/labelTxt/P0003.txt ...              (official v1.0 val labelTxt files)
<sources>/sku110k/SKU110K_fixed/annotations/annotations_test.csv
<sources>/fsc147/annotation_FSC147_384.json
```

Then run:

```bash
python3 -I thirdparty/tp.py rebuild --sources <sources>   # fill every placeholder; write a file only if its SHA-256 matches
python3 -I thirdparty/tp.py status                        # 'rebuilt' or 'redacted' for each of the 12 files
bash reproduce.sh --sources <sources>                     # rebuild, then rerun every analysis
```

`thirdparty/manifest.json` stores two SHA-256 digests for each file, one of the redacted file and one of the file used for
the paper. A file is written only when the rebuilt bytes match the second digest. So a rebuilt file is byte-identical to
the one we scored against, or it is not written.

The manifest also stores the digest of every official file that was read: 90 DOTA label files, the SKU-110K csv and
the FSC-147 json. If a rebuild fails, the script names the official files that differ from the ones we used.
We checked that the FSC-147 json on GitHub has the stored digest, as of October 2026.

Without the rebuild, `reproduce.sh` keeps the shipped outputs of the analyses that score against these labels:
`intersection_decomp`, `prep1_v2`, `construction_table`, `natural_readout`, `p1_predict` and the pipeline replay.
It reruns everything else.

Maintainers: `python3 -I thirdparty/tp.py redact --sources <sources>` replaces the labels with placeholders and writes the manifest.
It first checks that every file round-trips byte for byte, and it writes nothing if any file fails.

## Citations

- DOTA: G.-S. Xia et al., "DOTA: A Large-scale Dataset for Object Detection in Aerial Images", CVPR 2018.
- SKU-110K: E. Goldman et al., "Precise Detection in Densely Packed Scenes", CVPR 2019.
- FSC-147: V. Ranjan et al., "Learning To Count Everything", CVPR 2021.
- DocLayNet: B. Pfitzmann et al., "DocLayNet: A Large Human-Annotated Dataset for Document-Layout Analysis", KDD 2022.
- ScreenSpot-Pro: K. Li et al., "ScreenSpot-Pro: GUI Grounding for Professional High-Resolution Computer Use", 2025.
- ScreenSpot-v2: Z. Wu et al., "OS-ATLAS: A Foundation Action Model for Generalist GUI Agents", 2024.
- PixMo-Count: M. Deitke et al., "Molmo and PixMo", 2024.
- GroundCUA: A. Feizi et al., "Grounding Computer Use Agents on Human Demonstrations", ICLR 2026.
- ScreenParse: A. S. Gurbuz et al., "ScreenParse: Moving Beyond Sparse Grounding with Complete Screen Parsing Supervision", ICML 2026.
