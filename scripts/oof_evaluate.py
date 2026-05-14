"""
OOF (Out-of-Fold) Evaluation
รัน GroupKFold(5) เหมือนตอน train ทุกอย่าง — แต่ละภาพถูก predict
ด้วย fold ที่มันเป็น val (โมเดลนั้นไม่เคยเห็นภาพนั้นเลย)

ผลที่ได้คือ metric ที่น่าเชื่อได้จริง ≠ batch_eval.csv
ซึ่งรันบนภาพชุดเดิมที่ train (in-sample)

รันด้วย:
    python scripts/oof_evaluate.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import FEATURE_COLS, TARGET, prepare_features
from src.grade import day_to_grade, GRADE_ORDER

OUTPUT_CSV = Path("results/oof_predictions.csv")

GRADE_ORDER_LIST = GRADE_ORDER  # ['A','B','C','D']


def grade_diff(pred_grade: str, true_grade: str) -> int:
    try:
        return GRADE_ORDER_LIST.index(pred_grade) - GRADE_ORDER_LIST.index(true_grade)
    except ValueError:
        return 0


def main():
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    df = prepare_features("data/features.csv")
    X = df[FEATURE_COLS].values
    y = df[TARGET].values
    groups = df["plant_id"].values

    gkf = GroupKFold(n_splits=5)
    oof_pred = np.zeros(len(y))
    fold_col = np.zeros(len(y), dtype=int)

    print(f"รัน GroupKFold(5) บน {len(df)} ภาพ, {df['plant_id'].nunique()} plants")
    print()

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        m = XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbosity=0,
        )
        m.fit(X[train_idx], y[train_idx])
        pred = m.predict(X[val_idx])
        oof_pred[val_idx] = pred
        fold_col[val_idx] = fold + 1

        mae = mean_absolute_error(y[val_idx], pred)
        rmse = np.sqrt(mean_squared_error(y[val_idx], pred))
        r2 = r2_score(y[val_idx], pred)
        val_plants = df.iloc[val_idx]["plant_id"].nunique()
        print(f"Fold {fold+1}: val={len(val_idx)} imgs ({val_plants} plants) | MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")

    # สร้าง output dataframe
    out = df[["filename", "variety", "plant_id", "day", "session", "view", "ab_group"]].copy() if "filename" in df.columns else df[["variety", "plant_id", "day"]].copy()

    out = df.copy()
    out["predicted_day"] = np.round(oof_pred, 4)
    out["fold"] = fold_col
    out["true_grade"] = out["day"].apply(lambda d: str(day_to_grade(d)))
    out["pred_grade"] = out["predicted_day"].apply(lambda d: str(day_to_grade(d)))
    out["grade_correct"] = out["pred_grade"] == out["true_grade"]
    out["grade_diff"] = out.apply(lambda r: grade_diff(r["pred_grade"], r["true_grade"]), axis=1)
    out["grade_adjacent"] = out["grade_diff"].abs() <= 1
    out["error"] = np.round(oof_pred - y, 4)
    out["abs_error"] = out["error"].abs()
    out["sq_error"] = np.round(out["error"] ** 2, 6)

    out.to_csv(OUTPUT_CSV, index=False)

    # ── สรุปผล OOF ──────────────────────────────────────────────
    mae_oof  = mean_absolute_error(y, oof_pred)
    rmse_oof = np.sqrt(mean_squared_error(y, oof_pred))
    r2_oof   = r2_score(y, oof_pred)
    grade_acc = out["grade_correct"].mean() * 100
    adj_acc   = out["grade_adjacent"].mean() * 100

    print()
    print("=" * 55)
    print("OOF Summary (แต่ละภาพ predict ด้วย fold ที่ไม่เคยเห็นมัน)")
    print("=" * 55)
    print(f"MAE:               {mae_oof:.4f} วัน")
    print(f"RMSE:              {rmse_oof:.4f} วัน")
    print(f"R2:                {r2_oof:.4f}")
    print(f"Grade accuracy:    {grade_acc:.2f}%")
    print(f"Adjacent accuracy: {adj_acc:.2f}%")
    print()

    # breakdown by variety
    print("-- Breakdown by variety --")
    for var in ["COS", "GOK"]:
        sub = out[out["variety"] == var]
        print(f"  {var}: MAE={sub['abs_error'].mean():.4f}  Grade acc={sub['grade_correct'].mean()*100:.2f}%  n={len(sub)}")

    # breakdown by day
    print()
    print("-- MAE by true day --")
    for d in sorted(out["day"].unique()):
        sub = out[out["day"] == d]
        print(f"  D{int(d)}: MAE={sub['abs_error'].mean():.4f}  n={len(sub)}")

    print()
    print(f"บันทึกแล้วที่: {OUTPUT_CSV} ({len(out)} แถว, {len(out.columns)} คอลัมน์)")


if __name__ == "__main__":
    main()
