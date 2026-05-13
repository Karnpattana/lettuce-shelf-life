# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project State

Phase 0–9 complete + Phase 9b (texture model) + Phase 9c (deployment) complete.  
**Live app:** https://lettuce-shelf-life-3m4f7pz9jqt7afybkuavvv.streamlit.app  
See `DEVLOG.md` for full decision log, known issues, and per-phase acceptance criteria.

## Commands

```bash
# Activate venv (Windows)
.venv\Scripts\activate

# Re-segment all 2,920 images (overwrites data/segmented/)
python -m src.segment

# Re-extract features → data/features.csv
python -m src.features.extract

# Run Streamlit demo
streamlit run app.py

# Execute a notebook non-interactively
.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace notebooks/<name>.ipynb

# Run automated test suite (60 tests, ~3 sec)
pytest tests/ -v

# Quick inference smoke-test
python -c "from src.inference import predict; print(predict('data/raw/COS02_A_D0_E_side.jpg', 'COS'))"
```

## Tests

60 automated tests in `tests/` — run with `pytest tests/ -v`

| ไฟล์ | ครอบคลุม |
|------|---------|
| `test_grade.py` | `day_to_grade()` boundary ทุกจุด, `predict_shelf_life()` status transitions |
| `test_feature_contract.py` | FEATURE_COLS ครบ/ถูกลำดับ, extractor คืน finite values, empty mask → NaN |
| `test_inference_golden.py` | golden regression: COS D0/D4/D8, GOK D0/D6 — skip ถ้าไม่มี data/raw/ |
| `test_segment_edges.py` | black/white/red image → empty mask, green blob, low_confidence flag |
| `test_no_plant_leakage.py` | GroupKFold ไม่มี plant_id ซ้ำ, dataset integrity (varieties, day range) |

## Architecture

The pipeline is **image → predicted_day (float) → grade + shelf life**. All thresholds live in one place (`src/grade.py`) so changing a boundary propagates everywhere automatically.

```
image file
  └─ src/preprocess.py        load_image → resize_with_padding(512×512) → denoise
  └─ src/segment.py           HSV mask [H=10°–90°, V>25] → opening → largest component → binary_fill_holes
  └─ src/features/color.py    Lab* stats + pct_green/yellow/brown  (thresholds in src/features/__init__.py)
  └─ src/features/texture.py  GLCM on L* channel (contrast, correlation, energy, homogeneity)
  └─ src/model.py             XGBoost regressor — FEATURE_COLS, trained with GroupKFold(5) by plant_id
  └─ src/grade.py             THRESHOLDS dict → day_to_grade, predict_shelf_life
  └─ src/variety_classifier.py  predict_variety() — XGBoost, 92.4% GroupKFold CV
  └─ src/inference.py         predict() — full pipeline, returns grade + shelf_life + variety + flags
  └─ app.py                   Streamlit UI
```

### Key constants (do not duplicate)

| Constant | File | Value |
|----------|------|-------|
| `THRESHOLDS` | `src/grade.py` | A/B=1.2, B/C=3.6, C/D=5.6 (calibrated Phase 5) |
| `MARKETABILITY_DAY` | `src/grade.py` | `THRESHOLDS['B'][1]` = 3.6 |
| `UNUSABLE_DAY` | `src/grade.py` | `THRESHOLDS['C'][1]` = 5.6 |
| `FEATURE_COLS` | `src/model.py` | 14 features (color × 9 + variety_enc + texture × 4) |
| Lab* pixel thresholds | `src/features/__init__.py` | GREEN/YELLOW/BROWN_LAB_RANGE |

### inference.py output dict

```python
{
  "variety":             "COS"|"GOK",
  "variety_confidence":  float,        # 0.0–1.0
  "variety_source":      "auto"|"user",
  "predicted_day":       float,        # 0.0–8.0
  "grade":               "A"|"B"|"C"|"D",
  "shelf_life": {
      "days_to_marketability_limit": float,
      "days_to_unusable":            float,
      "status": "fresh"|"good"|"warning"|"expired",
  },
  "low_confidence":    bool,         # True if area_ratio < 5%
  "gok_extrapolation": bool,         # True if GOK predicted_day > 6.5 (no D8 training data)
  "area_ratio":        float,
  "features":          dict,         # raw feature values
}
```

## Critical constraints

- **Do not retrain `models/xgb_model_texture.json`** unless explicitly asked — it is the final model (Phase 9b) refit on all 2,920 images with 14 features.
- **`models/xgb_model.json` (old 10-feature model) is kept as backup only** — it is INCOMPATIBLE with current `FEATURE_COLS` (14 features) and will crash with `feature shape mismatch` if passed to `predict()` without also reverting `FEATURE_COLS`.
- **Always use `GroupKFold` grouped by `plant_id`** for any new CV — splitting by image causes data leakage (multiple images per plant per day).
- **GOK has no D8 images** — the model is extrapolating for GOK when predicted_day > 6.5. The `gok_extrapolation` flag handles this.
- **Shelf life constants must derive from `THRESHOLDS`**, not be hardcoded independently. If `THRESHOLDS` changes, shelf life boundaries update automatically.
- `data/raw/` and `data/segmented/` are gitignored — never assume they are present in CI.
- **Use `opencv-python-headless`** (not `opencv-python`) in `requirements.txt` — headless variant works on Streamlit Cloud and other server environments.
- **`models/variety_classifier.pkl` must stay tracked in git** — `.gitignore` was updated to allow `.pkl` in `models/`.

## Dataset

- 2,920 images: COS 1,480 | GOK 1,440
- Filename pattern: `{COS|GOK}{NN}_{A|B}_D{day}_{M|E}_{top|side}.jpg`
- `plant_id` is the numeric part of the filename (e.g. `COS02` → plant_id=2)
- D3-M, D4-E, D8-M temperature values in `metadata.csv` are placeholders (not recorded); temperature is not a model feature so this has no effect.

## Notebooks (in order)

| Notebook | Purpose |
|----------|---------|
| `00_segment_check.ipynb` | Visual QC of segmentation masks |
| `01_feature_check.ipynb` | Feature trend sanity check |
| `02_eda.ipynb` | EDA — view/variety comparison, outliers |
| `03_model.ipynb` | XGBoost CV, feature importance |
| `04_grade.ipynb` | Threshold calibration |
| `05_inference_check.ipynb` | End-to-end pipeline test incl. holdout |
