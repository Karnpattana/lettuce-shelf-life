import joblib
import numpy as np
from pathlib import Path

MODEL_PATH = Path(__file__).parent.parent / "models" / "variety_classifier.pkl"

VARIETY_FEATURES = [
    "L_mean", "L_std", "a_mean", "a_std", "b_mean", "b_std",
    "pct_green", "pct_yellow", "pct_brown",
    "contrast", "correlation", "energy", "homogeneity",
    "area_ratio",
]

_bundle = None


def _load():
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def predict_variety(features: dict) -> tuple[str, float]:
    """
    รับ features dict → return (variety, confidence)
    variety: 'COS' หรือ 'GOK'
    confidence: 0.0–1.0

    XGBoost classifier trained with GroupKFold(5) by plant_id.
    CV accuracy ~92% — main discriminants: energy, homogeneity, a_std, b_mean.
    Color features (a_mean, pct_green) are day-confounded so accuracy ceiling is ~92%.
    """
    bundle = _load()
    pipeline = bundle["pipeline"]
    classes = bundle["classes"]  # ['COS', 'GOK']
    feat_list = bundle.get("features", VARIETY_FEATURES)

    X = np.array([[features[f] for f in feat_list]])
    pred_enc = int(pipeline.predict(X)[0])
    proba = float(pipeline.predict_proba(X)[0].max())
    return str(classes[pred_enc]), proba
