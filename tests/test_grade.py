"""
Test #1 — Grade boundary และ shelf life status transitions
ทดสอบ day_to_grade() และ predict_shelf_life() ใน src/grade.py
"""
import numpy as np
import pytest

from src.grade import (
    FRESH_DAY,
    MARKETABILITY_DAY,
    THRESHOLDS,
    UNUSABLE_DAY,
    day_to_grade,
    predict_shelf_life,
)


# ---------------------------------------------------------------------------
# day_to_grade — boundary tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("day,expected", [
    # เขต A: [0.0, 1.2)
    (0.0,  "A"),
    (0.5,  "A"),
    (1.19, "A"),
    # ขอบ A→B: 1.2 ต้องเป็น B
    (1.2,  "B"),
    # เขต B: [1.2, 3.6)
    (2.0,  "B"),
    (3.59, "B"),
    # ขอบ B→C: 3.6 ต้องเป็น C
    (3.6,  "C"),
    # เขต C: [3.6, 5.6)
    (4.5,  "C"),
    (5.59, "C"),
    # ขอบ C→D: 5.6 ต้องเป็น D
    (5.6,  "D"),
    # เขต D: [5.6, ∞)
    (7.0,  "D"),
    (8.0,  "D"),
    (9.5,  "D"),
    (20.0, "D"),  # ค่าสูงมาก ต้องไม่ crash
])
def test_day_to_grade_boundaries(day: float, expected: str):
    assert day_to_grade(day) == expected


def test_day_to_grade_scalar_returns_string():
    result = day_to_grade(2.0)
    assert isinstance(result, (str, np.str_))


def test_day_to_grade_array_returns_array():
    days = np.array([0.5, 2.0, 4.0, 7.0])
    result = day_to_grade(days)
    assert list(result) == ["A", "B", "C", "D"]


def test_day_to_grade_consistent_with_thresholds():
    """THRESHOLDS dict ต้อง sync กับผลลัพธ์จริงของฟังก์ชัน"""
    for grade, (lo, hi) in THRESHOLDS.items():
        mid = (lo + hi) / 2
        assert day_to_grade(mid) == grade, (
            f"predicted_day={mid} ควรได้เกรด {grade} ตาม THRESHOLDS แต่ได้ {day_to_grade(mid)}"
        )


# ---------------------------------------------------------------------------
# predict_shelf_life — status transitions
# ---------------------------------------------------------------------------

def test_shelf_life_fresh():
    out = predict_shelf_life(0.5)
    assert out["status"] == "fresh"
    assert out["days_to_marketability_limit"] == round(MARKETABILITY_DAY - 0.5, 1)
    assert out["days_to_unusable"] == round(UNUSABLE_DAY - 0.5, 1)


def test_shelf_life_good():
    out = predict_shelf_life(2.0)
    assert out["status"] == "good"
    assert out["days_to_marketability_limit"] == round(MARKETABILITY_DAY - 2.0, 1)
    assert out["days_to_unusable"] > 0


def test_shelf_life_warning():
    out = predict_shelf_life(4.0)
    assert out["status"] == "warning"
    assert out["days_to_marketability_limit"] == 0.0  # เลยขีดแล้ว
    assert out["days_to_unusable"] > 0


def test_shelf_life_expired():
    out = predict_shelf_life(7.0)
    assert out["status"] == "expired"
    assert out["days_to_marketability_limit"] == 0.0
    assert out["days_to_unusable"] == 0.0


def test_shelf_life_clamp_no_negative():
    """days_to_* ต้องไม่ติดลบแม้ predicted_day จะสูงมาก"""
    out = predict_shelf_life(99.0)
    assert out["days_to_marketability_limit"] == 0.0
    assert out["days_to_unusable"] == 0.0


@pytest.mark.parametrize("day,expected_status", [
    (FRESH_DAY - 0.01, "fresh"),   # ก่อนขอบ A/B → fresh
    (FRESH_DAY,        "good"),    # ขอบ A/B พอดี → good (< ไม่ใช่ <=)
    (MARKETABILITY_DAY - 0.01, "good"),
    (MARKETABILITY_DAY,        "warning"),
    (UNUSABLE_DAY - 0.01,      "warning"),
    (UNUSABLE_DAY,             "expired"),
])
def test_shelf_life_status_at_boundaries(day: float, expected_status: str):
    out = predict_shelf_life(day)
    assert out["status"] == expected_status, (
        f"predicted_day={day} ควรได้ status='{expected_status}' แต่ได้ '{out['status']}'"
    )


def test_shelf_life_returns_required_keys():
    out = predict_shelf_life(3.0)
    assert "days_to_marketability_limit" in out
    assert "days_to_unusable" in out
    assert "status" in out
