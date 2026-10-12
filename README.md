# From Seeing to Reporting: A Theory of Reliable Visual Set Generation

Code, every model request and response, and the labels we may redistribute, for the paper by Junzhe Shi, Yuan Gan and Shida Jiang (arXiv link to be added). Labels of DOTA v1.0, SKU-110K and FSC-147 are rebuilt from the official downloads by `thirdparty/tp.py` (see `thirdparty/SOURCES.md`). The paper PDF is in `paper/FSR_arxiv.pdf`.

Vision-language models are asked to report sets of objects: to count them, box them, parse a screen, or pick a click target. A typical answer has a *list*, one record (class and location) per object, and *stated counts*, the numbers the model writes for each class. The two need not agree. In one response to a dense synthetic scene with 173 shapes, GPT-6.1 lists 173 records that match every shape, yet the counts it states in the same response add up to 188. The paper treats reporting as a chain from responses to records, sets and readouts. It separates three problems:

* repairing a report;
* detecting that a report is wrong;
* judging which report is more accurate.

It derives defaults for each problem and tests them in 46,785 calls on 1,146 images with 53,875 annotated targets.

## What the results mean for practice

| If you see this | Do this | Do not do this |
|---|---|---|
| The stated count disagrees with the model's own list | Read the count from the list (the list count). If the list is degenerate (missing classes, repeats, wrong length, truncated), use the stated count instead | Ask for the total first |
| Class names outside your vocabulary ("diamond" for a rotated square) | List the allowed names and map declared aliases in code | Rely on the prompt alone |
| Small objects are missed, or clicks are slightly off target | Decompose into views where every object lies inside some view (one-object overlap, one call per view); use two-stage zoom on original pixels | Tile without a one-object overlap; pack tiles into one call |
| Two calls of the same model agree, and you want more | Another call of the same interface will add little. Confirm with another view or model where it rejects more false records than true ones (Proposition 5) | Read agreement as correctness; buy more calls of the same model; same-view self-review on screens |
| Over-reporting tasks (detection, layout, screen parsing) | Keep a record only if all three calls report it, and fuse coordinates | Delete records for count objectives |
| You want an alarm for wrong outputs | Use disagreement with a second model | Treat agreement as correctness |
| Several models are available | Use cross-model disagreement as an alarm; choose cross-model confirmation from the menu with a few labels or a family-level ranking | An equal-weight vote of all models on points, screens, clicks or counts when one model is strongest (worse or no better there; it helped on DOTA and SKU-110K) |

The shipped configuration of the default pipeline (Algorithm 1 and Table 3 of the paper) is `config/default_pipeline.json`. The degeneracy flags are in `stageB/code/degen.py`. The readout rule is Step 6 of the method (Section 5 of the paper):
- use the stated counts when the list is flagged and stated counts exist;
- otherwise use the list count, the number of records of each class in the list.

## Run the default pipeline (no API calls)

`fsr_pipeline/` implements Algorithm 1 of the paper as one entry point, `pipeline.run(adapter, item, model, cfg, client, views=None, partner=None)`. Every branch reads only three things:

- the task adapter (`adapters.py`: classes, alias map, matching and association rules, decoders);
- the shipped configuration (`config/default_pipeline.json`);
- the responses returned by a client.

Optional inputs are a list of views from an external view planner and a partner model for the alarm. `ReplayClient` (`clients.py`) serves the stored responses of the paper and never calls a model. A request with no stored response, such as a follow-up that was never sent, returns `unavailable`, and the pipeline continues on its documented branch. `LiveClient` is a hook for live use and is not implemented in this release.

```bash
python3 tools/set_root.py
python3 -I thirdparty/tp.py rebuild --sources <dir>    # the replay scores detection sets against DOTA, SKU-110K and FSC-147 labels
cd stageB/analysis && python3 -I ../../fsr_pipeline/replay.py
```

`replay.py` writes `stageB/analysis/out/pipeline_replay.json`. It replays the entry point on stored calls and reproduces the component results:

- count readout 1.60 / 0.42 / 24.52;
- click hit rates 90.0% / 87.0% / 87.5%;
- shipped-construction set error in all 18 transfer cells, within 0.07 per image.

It also covers three other cases:

- **Multi-view reports.** Reports from four crops (the cross-view experiment), merged across views and used to confirm the whole-image report.
- **Stored follow-ups.** The follow-ups stored for flagged detection calls.
- **Boundary cases.** Empty, unreadable or truncated responses, missing stated counts, degenerate lists, repeated transport failures, an unreadable call among three, and missing click points.

## Reproduce every table and figure (no API calls)

```bash
pip install -r requirements.txt
bash reproduce.sh                       # level 1: no third-party downloads
bash reproduce.sh --sources <dir>       # level 2: rebuild the DOTA, SKU-110K and FSC-147 labels first, then rerun everything
```

Labels from DOTA v1.0, SKU-110K and FSC-147 are not in this repository (see [Data and licences](#data-and-licences)). `thirdparty/SOURCES.md` gives the download links and the directory layout for `<dir>`.

`reproduce.sh` runs the following steps:

1. **Fix paths.** `tools/set_root.py` points every script at this checkout. The code was written with absolute paths, and the tool rewrites them.
2. **Rebuild third-party labels (level 2 only).** `thirdparty/tp.py rebuild` fills the placeholders from the official files. It writes a file only when its SHA-256 matches the file used for the paper.
3. **Recompute analyses.** It reruns the analyses in `stageB/analysis/run_all.sh` on the shipped responses and labels.
   - **Level 1** reruns the analyses that need no rebuilt labels: Tables 6, 7, 9, 11 and 15 and Figures 1, 3, 4 and 6.
   - It keeps the shipped outputs of the analyses that score against the rebuilt labels: Tables 8, 10, 12, 13, 16 and 17, Figure 5 and the pipeline replay.
   - **Level 2** reruns all of them.
4. **Rebuild the paper outputs.** It regenerates the LaTeX tables (`paper/make_tables.py`) and the figures (`paper/figures/make_figures.py`).
5. **Check the results.** It compares the regenerated analysis outputs with the shipped ones. With the pinned packages they match byte for byte.
   - Random draws use seeds derived from SHA-256 digests.
   - Cross-fitting folds are stored in `stageB/analysis/out/prep1_v2_folds.json`.

Panel (b) of Figure 1 needs one ScreenSpot-Pro screenshot, which is not redistributed. Without it, the shipped figure is kept.

To build the paper, run `cd paper && latexmk -pdf main.tex`.

## Where each result comes from

| Paper result | Script (`stageB/analysis/`) | Output (`stageB/analysis/out/`) |
|---|---|---|
| Opening example and exact-list counts (Figures 1a and 3) | `opening_example.py` | `opening_example.json` |
| Readout effects (Tables 7 and 15, Figures 3 and 4) | `readout_effects.py`, `readout_density.py` | `readout_effects.json`, `readout_density.json` |
| Gemini 3.8 setting and output cap, with the keep-unknown sensitivity (Table 9) | `rc16_setting.py` | `rc16_setting.json` |
| Selection and geometry identities, intersection threshold (Tables 10 and 16, Figure 5a) | `intersection_decomp.py` | `intersection_decomp.json` |
| Constrained oracle and label-fitted selectors (Table 12, Figure 5b) | `prep1_v2.py` | `prep1_v2.json`, `prep1_v2_folds.json` |
| Leave-one-source-out construction transfer and request-level costs (Tables 13 and 17) | `construction_table.py` | `construction_table.json` |
| Click alarms (Table 11, Figure 6) | `alarm_curves.py` | `alarm_curves.json` |
| Against common practice (Table 6) | `common_practice.py` | `common_practice.json` |
| Readout on natural multi-class images (Table 8) | `natural_readout.py` | `natural_readout.json` |
| Deletion predicted from a few labelled images (Table 10, Section 7) | `p1_predict.py` | `p1_predict.json` |
| All main tables in Markdown | `main_tables.py` | `main_tables.md` |
| Default pipeline replayed on stored calls, multi-view reports, boundary cases (Section 5) | `../../fsr_pipeline/replay.py` | `pipeline_replay.json` |

Shared objects (records, clusters, matching, oracle) are in `stageB/analysis/fsr_core.py`. Table numbers follow the arXiv version.

## Repository layout

```
stageB/code/            request builders, decoder (bcommon.py), degeneracy flags (degen.py), runner, SCP generator
stageB/analysis/        analysis scripts, run_all.sh, and their outputs (out/)
stageB/runs/            requests, raw responses and registries of the readout, tiling, review, comparator,
                        resolution, end-to-end, reasoning, open-model, grid and readout-confirmation experiments
stageB/data/            SCP and SCP-dense images and labels (synthetic, generated by stageB/code/scp_generate.py)
stageB/xmodel, prep1, offline/   offline analyses of stored calls (cross-model merges, selection, mechanism checks)
release/ground_truth/   SCP labels (100 images)
fsr_v51/e3/             detection experiment: scoring and correspondence code, item lists and labels (see thirdparty/), all calls
fsr_v51/e12, e12b/      controlled-scene experiments (images, labels, calls)
fsr_pilot/              pilot experiments (scenes, prompts, calls)
gui/                    screen experiments: code, item lists and labels, all calls (screenshots not included)
paper/                  LaTeX source, table and figure scripts, figures, and FSR_arxiv.pdf
config/                 shipped configuration of the default pipeline (default_pipeline.json)
fsr_pipeline/           executable entry point of the default pipeline, replay client, replay harness
tools/set_root.py       path setup used by reproduce.sh
thirdparty/             rebuild of DOTA, SKU-110K and FSC-147 labels from the official downloads (tp.py, manifest.json, SOURCES.md)
```

Each response file is JSON Lines. Every line holds:
- the request id;
- the status;
- the raw response text;
- token usage, including reasoning tokens where the API reports them.

The matching request files hold the prompt, the image reference and the metadata.

## Data and licences

* **Code:** MIT (`LICENSE`).
* **Synthetic images and labels, and all model responses:** CC BY 4.0 (`LICENSE-DATA`).
* **Third-party images are not redistributed.** Download them from the original sources (`thirdparty/SOURCES.md`).
* **Third-party labels with permissive licences are included with attribution.** These are DocLayNet v1.2 (CDLA-Permissive-1.0), ScreenSpot-Pro (MIT), ScreenSpot-v2 (Apache-2.0), GroundCUA (MIT), ScreenParse (CC BY 4.0) and PixMo-Count (ODC-BY 1.0). They cover the item lists and the labels of the items we used, and remain under those licences.
* **Labels from DOTA v1.0, SKU-110K and FSC-147 are not redistributed.** Their files hold placeholders and item metadata: image name, window or scale, class name and object count. `python3 -I thirdparty/tp.py rebuild --sources <dir>` restores the byte-identical files from the official downloads.
* **Model calls:** rerunning them needs vendor API keys and costs money. The registry in Appendix C of the paper lists every experiment, its calls and its cost (about $693 in total).

## Citation

```bibtex
@misc{shi2026seeing,
  title  = {From Seeing to Reporting: A Theory of Reliable Visual Set Generation},
  author = {Shi, Junzhe and Gan, Yuan and Jiang, Shida},
  year   = {2026},
  note   = {arXiv preprint, identifier to be added}
}
```

The companion paper on what a single call can see is *Why VLMs Miss Small Objects, and When Zooming In Is Safe* (arXiv:2610.09313, https://github.com/shijunzhe/vlm-small-objects).
