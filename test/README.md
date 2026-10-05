# test/ — model testing and evaluation

## What is here

| file | what it is |
|---|---|
| `HoloQPI_Evaluation.ipynb` | the notebook to open and run |
| `test_outputs/` | created on the first run: `predictions/`, `figures/`, `tables/` |

## Run it

```bash
cd test
jupyter lab HoloQPI_Evaluation.ipynb      # or: jupyter notebook
```

Then edit **section 1** and run all cells. Nothing else needs changing.

Requirements are the project's own (`environment.yml` / `requirements.txt`) plus `jupyter`,
`pandas` and `matplotlib`:

```bash
pip install jupyterlab pandas matplotlib
```

## What it does

Loads the real `best_model.pt` checkpoints and runs them through the project's own
preprocessing, measurement chain and metric classes — everything is imported from
`holoqpi/`, nothing is reimplemented. For each field it reports:

- the raw hologram, and the 900 px crop the network actually sees (1024 → 900 px onto the
  phase grid, origin moved by `data.crop_offset_px`)
- predicted **quantitative phase** (rad), with the reference and a signed error map
- predicted **amplitude**, for an arm whose amplitude head is on, against both the
  reconstruction reference and the thin-phase assumption A = 1
- predicted **segmentation**, with the reference and a boundary overlay
- **per-cell measurements** — projected area, circularity, integrated phase, optical
  volume, dry mass — matched to reference cells by IoU, with MAPE, bias and Bland–Altman
- phase, segmentation and detection metrics per sample and in aggregate
- a model-to-model comparison table and bar chart
- grids across samples and across models
- saved `.npz` predictions, `.csv` tables, `run_metadata.json` and `.png` figures

## Defaults

- **Models**: arm A (the recommended configuration), arm KA (the 3.36 M compact model), and
  arm D1 (the only one of the three with an amplitude head). Change `MODELS` to load others;
  any `config/v2/*.yaml` works as long as `runs/<experiment_name>_<modality>/best_model.pt`
  exists.
- **Split**: `"test"`. This is the honest default — it is the split the study reports. If no
  field of that split is present the notebook says so loudly and falls back to everything it
  can find, with a warning that training fields give optimistic numbers.
- **Device**: `auto` — CUDA if available, otherwise CPU. A 900 × 900 forward pass is about
  0.7 s on CPU and well under 0.1 s on a GPU, so the whole notebook runs on CPU in minutes.

## Your own data

Section 16 covers this. The short version: only a hologram is required.

```python
s = load_sample(hologram_path="/path/to/my_hologram.tif")
p = run_model(REGISTRY[PRIMARY_MODEL], s)
cells, labels = cells_for(p["phase"], p["mask"])
```

Every reference — phase, mask, amplitude — is optional, and whatever is missing is skipped
rather than faked.

Three things that will bite you, in order of likelihood:

1. **Size.** The loader crops 1024 → 900 (origin moved by `data.crop_offset_px`) and never resamples, because interpolated
   fringes no longer encode optical path length. A hologram smaller than 900 px fails.
2. **Modality.** An off-axis checkpoint on an in-line hologram produces confident nonsense.
   Set `MODALITY` and load `config/v2/g_baseline_gabor.yaml` for in-line data.
3. **Absolute units.** λ, α and the pixel pitch live in `config/base.yaml` and are specific
   to this instrument. Relative metrics are invariant to them; absolute picograms and µm²
   are not.

## Notebook numbers versus the study's results

The notebook scores whatever fields it finds. Per-field variation is large, so numbers from a
few fields will differ from the study's 113-field results. Quote
`results_for_manuscript/` for results. Use this notebook to inspect individual fields and to
run new data.
