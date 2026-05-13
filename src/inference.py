from pathlib import Path

import numpy as np

from src.features.color import extract_color
from src.features.texture import extract_texture
from src.grade import day_to_grade, predict_shelf_life, _apply_thresholds
from src.model import FEATURE_COLS, load_model
from src.preprocess import preprocess_pipeline
from src.segment import MIN_AREA_RATIO, segment_lettuce
from src.variety_classifier import predict_variety

DEFAULT_MODEL_PATH = Path("models/xgb_model.json")


def predict(
    image_path: str | Path,
    variety: str | None = None,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    thresholds: tuple[float, float, float] | None = None,
) -> dict:
    """
    รับภาพ 1 ภาพ → คืน grade, shelf_life, และ variety

    Parameters
    ----------
    image_path : path ไปยังภาพ raw (.jpg)
    variety    : 'COS', 'GOK', หรือ None (auto-detect)
    model_path : path ไปยัง xgb_model.json
    thresholds : (t1, t2, t3) A/B, B/C, C/D boundary
                 ถ้า None → ใช้ THRESHOLDS default จาก grade.py

    Returns
    -------
    dict:
        variety            : 'COS' / 'GOK'
        variety_confidence : float 0-1
        variety_source     : 'auto' / 'user'
        predicted_day      : float
        grade              : 'A' / 'B' / 'C' / 'D'
        shelf_life         : dict (days_to_marketability_limit, days_to_unusable, status)
        low_confidence     : bool  (True ถ้า area_ratio < 5%)
        gok_extrapolation  : bool
        area_ratio         : float
        features           : dict ของ feature values ที่ใช้ predict
    """
    # 1. Preprocess
    img = preprocess_pipeline(str(image_path))

    # 2. Segment
    mask, cropped = segment_lettuce(img)
    area_ratio = float(np.sum(mask > 0)) / mask.size
    low_confidence = area_ratio < MIN_AREA_RATIO

    # 3. Extract features
    color_feats = extract_color(cropped, mask)
    texture_feats = extract_texture(cropped, mask)
    raw_feats = {**color_feats, **texture_feats, "area_ratio": area_ratio}

    # 4. Variety: auto-detect or user override
    if variety is not None:
        variety_upper = variety.upper()
        if variety_upper not in ("COS", "GOK"):
            raise ValueError(f"variety ต้องเป็น 'COS' หรือ 'GOK' ไม่ใช่ '{variety}'")
        variety_final = variety_upper
        variety_confidence = 1.0
        variety_source = "user"
    else:
        variety_final, variety_confidence = predict_variety(raw_feats)
        variety_source = "auto"

    # 5. Predict day
    variety_enc = 1 if variety_final == "GOK" else 0
    all_feats = {**raw_feats, "variety_enc": variety_enc}
    feature_vector = np.array([[all_feats[col] for col in FEATURE_COLS]], dtype=np.float32)

    model = load_model(model_path)
    predicted_day = float(model.predict(feature_vector)[0])
    predicted_day = float(np.clip(predicted_day, 0.0, 8.0))

    # 6. Map to grade
    if thresholds is not None:
        t1, t2, t3 = thresholds
        grade = str(_apply_thresholds(np.array([predicted_day]), t1, t2, t3)[0])
    else:
        grade = str(day_to_grade(predicted_day))

    gok_extrapolation = (variety_final == "GOK") and (predicted_day > 6.5)

    return {
        "variety":             variety_final,
        "variety_confidence":  round(variety_confidence, 4),
        "variety_source":      variety_source,
        "predicted_day":       round(predicted_day, 2),
        "grade":               grade,
        "shelf_life":          predict_shelf_life(predicted_day),
        "low_confidence":      low_confidence,
        "gok_extrapolation":   gok_extrapolation,
        "area_ratio":          round(area_ratio, 4),
        "features":            {k: round(v, 4) for k, v in all_feats.items()},
    }
