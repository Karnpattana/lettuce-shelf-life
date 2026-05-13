"""
Test #3 — Golden image regression
ยืนยันว่า predict() คืน grade และ predicted_day ที่ถูกต้องบนรูปที่รู้ผลอยู่แล้ว

ค่า golden ยึดจากโมเดล xgb_model_texture.json (14 features, Phase 9b)
  MAE=0.481 วัน, R²=0.904, Grade accuracy=80.55%

ถ้า test เหล่านี้พัง หมายความว่า:
  - model ถูก retrain หรือเปลี่ยน → ต้องอัปเดตค่า golden
  - pipeline มี regression (feature order เปลี่ยน, preprocess เปลี่ยน ฯลฯ)

Skip อัตโนมัติถ้าไม่มี data/raw/ (เช่น CI ที่ไม่มีข้อมูล)
"""
from pathlib import Path

import pytest

DATA_DIR = Path("data/raw")
MODEL_PATH = Path("models/xgb_model_texture.json")

pytestmark = pytest.mark.skipif(
    not DATA_DIR.exists() or not MODEL_PATH.exists(),
    reason="data/raw/ หรือ models/xgb_model_texture.json ไม่มีอยู่ (ข้าม CI)",
)

# tolerance สำหรับ predicted_day (± วัน)
DAY_TOL = 0.5


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def run(image_name: str, variety: str) -> dict:
    from src.inference import predict
    return predict(DATA_DIR / image_name, variety)


# ---------------------------------------------------------------------------
# COS — วันที่ 0 (ต้องได้เกรด A, status fresh)
# ---------------------------------------------------------------------------

def test_cos_d0_grade_and_status():
    out = run("COS01_A_D0_E_side.jpg", "COS")
    assert out["grade"] == "A", f"D0 ควรได้เกรด A ได้ {out['grade']}"
    assert out["shelf_life"]["status"] == "fresh"
    assert out["low_confidence"] is False


def test_cos_d0_predicted_day_near_zero():
    """golden: 0.42 — ยอมรับ ± 0.5 วัน"""
    out = run("COS01_A_D0_E_side.jpg", "COS")
    assert abs(out["predicted_day"] - 0.42) <= DAY_TOL, (
        f"predicted_day={out['predicted_day']} เบี่ยงเกิน {DAY_TOL} วัน จาก golden=0.42"
    )


# ---------------------------------------------------------------------------
# COS — วันที่ 4 (ต้องได้เกรด C, status warning)
# ---------------------------------------------------------------------------

def test_cos_d4_grade_and_status():
    out = run("COS01_A_D4_E_side.jpg", "COS")
    assert out["grade"] == "C", f"D4 ควรได้เกรด C ได้ {out['grade']}"
    assert out["shelf_life"]["status"] == "warning"
    assert out["low_confidence"] is False


def test_cos_d4_predicted_day():
    """golden: 4.15"""
    out = run("COS01_A_D4_E_side.jpg", "COS")
    assert abs(out["predicted_day"] - 4.15) <= DAY_TOL, (
        f"predicted_day={out['predicted_day']} เบี่ยงเกิน {DAY_TOL} วัน จาก golden=4.15"
    )


# ---------------------------------------------------------------------------
# COS — วันที่ 8 (ต้องได้เกรด D, status expired)
# ---------------------------------------------------------------------------

def test_cos_d8_grade_and_status():
    out = run("COS01_A_D8_M_side.jpg", "COS")
    assert out["grade"] == "D", f"D8 ควรได้เกรด D ได้ {out['grade']}"
    assert out["shelf_life"]["status"] == "expired"
    assert out["shelf_life"]["days_to_unusable"] == 0.0
    assert out["low_confidence"] is False


def test_cos_d8_predicted_day():
    """golden: 7.61"""
    out = run("COS01_A_D8_M_side.jpg", "COS")
    assert abs(out["predicted_day"] - 7.61) <= DAY_TOL, (
        f"predicted_day={out['predicted_day']} เบี่ยงเกิน {DAY_TOL} วัน จาก golden=7.61"
    )


# ---------------------------------------------------------------------------
# GOK — วันที่ 0 (ต้องได้เกรด A, status fresh)
# ---------------------------------------------------------------------------

def test_gok_d0_grade_and_status():
    out = run("GOK01_A_D0_E_side.jpg", "GOK")
    assert out["grade"] == "A", f"GOK D0 ควรได้เกรด A ได้ {out['grade']}"
    assert out["shelf_life"]["status"] == "fresh"
    assert out["gok_extrapolation"] is False


def test_gok_d0_predicted_day():
    """golden: 0.10"""
    out = run("GOK01_A_D0_E_side.jpg", "GOK")
    assert abs(out["predicted_day"] - 0.10) <= DAY_TOL, (
        f"predicted_day={out['predicted_day']} เบี่ยงเกิน {DAY_TOL} วัน จาก golden=0.10"
    )


# ---------------------------------------------------------------------------
# GOK — วันที่ 6 (ตรวจ gok_extrapolation flag)
# ---------------------------------------------------------------------------

def test_gok_d6_grade():
    out = run("GOK01_A_D6_E_side.jpg", "GOK")
    assert out["grade"] in ("C", "D"), (
        f"GOK D6 ควรได้เกรด C หรือ D (predicted_day={out['predicted_day']})"
    )


def test_gok_d6_predicted_day():
    """golden: 5.69"""
    out = run("GOK01_A_D6_E_side.jpg", "GOK")
    assert abs(out["predicted_day"] - 5.69) <= DAY_TOL, (
        f"predicted_day={out['predicted_day']} เบี่ยงเกิน {DAY_TOL} วัน จาก golden=5.69"
    )


def test_gok_d6_extrapolation_flag():
    out = run("GOK01_A_D6_E_side.jpg", "GOK")
    if out["predicted_day"] <= 6.5:
        assert out["gok_extrapolation"] is False
    else:
        assert out["gok_extrapolation"] is True


# ---------------------------------------------------------------------------
# Output schema — ทุก predict() ต้องมี key ครบ
# ---------------------------------------------------------------------------

def test_output_schema_complete():
    out = run("COS01_A_D0_E_side.jpg", "COS")
    required_keys = {
        "variety", "variety_confidence", "variety_source",
        "predicted_day", "grade", "shelf_life",
        "low_confidence", "gok_extrapolation", "area_ratio", "features",
    }
    missing = required_keys - out.keys()
    assert not missing, f"output dict ขาด key: {missing}"

    shelf_keys = {"days_to_marketability_limit", "days_to_unusable", "status"}
    missing_shelf = shelf_keys - out["shelf_life"].keys()
    assert not missing_shelf, f"shelf_life dict ขาด key: {missing_shelf}"
