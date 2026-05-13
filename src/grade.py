import numpy as np
import pandas as pd

# Grade boundary (day-based): A=D0-1, B=D2-3, C=D4-5, D=D6-8
# ปรับได้โดยเปลี่ยน THRESHOLDS เท่านั้น ไม่ต้อง retrain
THRESHOLDS = {
    'A': (0.0, 1.2),   # predicted_day < 1.2
    'B': (1.2, 3.6),   # 1.2 <= predicted_day < 3.6
    'C': (3.6, 5.6),   # 3.6 <= predicted_day < 5.6
    'D': (5.6, 9.0),   # predicted_day >= 5.6
}

# Shelf life thresholds — ดึงจาก THRESHOLDS เพื่อให้ sync กับ grade boundaries เสมอ
# Kader et al. (1973) OVQ scale: OVQ≤5 = marketability limit, OVQ≤3 = unusable
MARKETABILITY_DAY = THRESHOLDS['B'][1]   # B→C boundary = 3.6
UNUSABLE_DAY      = THRESHOLDS['C'][1]   # C→D boundary = 5.6
FRESH_DAY         = THRESHOLDS['A'][1]   # A→B boundary = 1.2
GRADE_ORDER = ['A', 'B', 'C', 'D']


def day_to_grade(predicted_day: float | np.ndarray) -> str | np.ndarray:
    """Map predicted day value(s) → grade letter A/B/C/D"""
    scalar = np.isscalar(predicted_day)
    arr = np.atleast_1d(np.asarray(predicted_day, dtype=float))
    result = np.empty(len(arr), dtype='U1')
    for grade, (lo, hi) in THRESHOLDS.items():
        mask = (arr >= lo) & (arr < hi)
        result[mask] = grade
    # ครอบขอบบน
    result[arr >= THRESHOLDS['D'][0]] = 'D'
    return result[0] if scalar else result


def true_day_to_grade(day: int | np.ndarray) -> str | np.ndarray:
    """Map actual day → grade (ใช้ boundary เดียวกัน)"""
    return day_to_grade(day)


def evaluate_grades(df: pd.DataFrame, oof_pred: np.ndarray) -> dict:
    """
    คำนวณ grade accuracy และ confusion matrix

    Parameters
    ----------
    df       : features DataFrame (ต้องมีคอลัมน์ 'day', 'variety')
    oof_pred : OOF predicted day values

    Returns
    -------
    dict: accuracy, per_variety_accuracy, confusion_matrix DataFrame
    """
    true_grade = true_day_to_grade(df['day'].values)
    pred_grade = day_to_grade(oof_pred)

    overall_acc = float(np.mean(true_grade == pred_grade))

    per_var = {}
    for var in ['COS', 'GOK']:
        mask = df['variety'].values == var
        per_var[var] = float(np.mean(true_grade[mask] == pred_grade[mask]))

    # Confusion matrix
    cm = pd.crosstab(
        pd.Series(true_grade, name='True'),
        pd.Series(pred_grade, name='Predicted'),
        rownames=['True'], colnames=['Predicted'],
    ).reindex(index=GRADE_ORDER, columns=GRADE_ORDER, fill_value=0)

    return {
        'overall_accuracy': overall_acc,
        'per_variety': per_var,
        'confusion_matrix': cm,
    }


def calibrate_thresholds(
    df: pd.DataFrame,
    oof_pred: np.ndarray,
    step: float = 0.1,
) -> dict:
    """
    Grid search หา thresholds ที่ให้ overall accuracy สูงสุด
    ปรับแค่ 3 boundary points: t1 (A/B), t2 (B/C), t3 (C/D)
    """
    best_acc = 0.0
    best_t = (1.5, 3.5, 5.5)
    true_grade = true_day_to_grade(df['day'].values)

    for t1 in np.arange(0.5, 2.5, step):
        for t2 in np.arange(t1 + 1.0, 4.5, step):
            for t3 in np.arange(t2 + 1.0, 6.5, step):
                pred = _apply_thresholds(oof_pred, t1, t2, t3)
                acc = float(np.mean(pred == true_grade))
                if acc > best_acc:
                    best_acc = acc
                    best_t = (t1, t2, t3)

    return {
        'best_t1': round(best_t[0], 2),  # A/B boundary
        'best_t2': round(best_t[1], 2),  # B/C boundary
        'best_t3': round(best_t[2], 2),  # C/D boundary
        'best_accuracy': round(best_acc, 4),
    }


def predict_shelf_life(predicted_day: float) -> dict:
    """
    คำนวณ shelf life จาก predicted_day
    อ้างอิง Kader et al. (1973) OVQ scale:
      - Marketability limit (OVQ ≤ 5) = B→C boundary = MARKETABILITY_DAY
      - Unusable threshold  (OVQ ≤ 3) = C→D boundary = UNUSABLE_DAY
    Status ดึงจาก THRESHOLDS เพื่อ sync กับ grade A/B/C/D เสมอ
    """
    days_to_marketability = round(max(0.0, MARKETABILITY_DAY - predicted_day), 1)
    days_to_unusable      = round(max(0.0, UNUSABLE_DAY - predicted_day), 1)

    if predicted_day < FRESH_DAY:
        status = "fresh"
    elif predicted_day < MARKETABILITY_DAY:
        status = "good"
    elif predicted_day < UNUSABLE_DAY:
        status = "warning"
    else:
        status = "expired"

    return {
        "days_to_marketability_limit": days_to_marketability,
        "days_to_unusable":            days_to_unusable,
        "status":                      status,
    }


def _apply_thresholds(pred_day, t1, t2, t3):
    arr = np.asarray(pred_day, dtype=float)
    result = np.empty(len(arr), dtype='U1')
    result[arr < t1] = 'A'
    result[(arr >= t1) & (arr < t2)] = 'B'
    result[(arr >= t2) & (arr < t3)] = 'C'
    result[arr >= t3] = 'D'
    return result
