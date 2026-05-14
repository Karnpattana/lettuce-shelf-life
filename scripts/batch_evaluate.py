"""
Batch evaluation — รัน predict() บนทุกภาพใน data/raw/
บันทึกผลทุกคอลัมน์ลง results/batch_eval.csv

รันด้วย:
    python scripts/batch_evaluate.py

ใช้เวลาประมาณ 15–20 นาที (2,920 ภาพ)
บันทึก checkpoint ทุก 100 ภาพ — ถ้า crash รัน script เดิมซ้ำได้
"""
import re
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# เพิ่ม root ไปยัง path เพื่อ import src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.grade import GRADE_ORDER, day_to_grade
from src.inference import predict

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RAW_DIR    = Path("data/raw")
OUTPUT_CSV = Path("results/batch_eval.csv")
CHECKPOINT = Path("results/_batch_eval_checkpoint.csv")

FILENAME_RE = re.compile(
    r'^(COS|GOK)(\d{2})_([AB])_D(\d+)_([ME])_(top|side)\.jpg$',
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def parse_filename(name: str) -> dict | None:
    m = FILENAME_RE.match(name)
    if not m:
        return None
    return {
        "variety":   m.group(1).upper(),
        "plant_id":  int(m.group(2)),
        "ab_group":  m.group(3).upper(),
        "true_day":  int(m.group(4)),
        "session":   m.group(5).upper(),
        "view":      m.group(6).lower(),
    }


def grade_diff(pred_grade: str, true_grade: str) -> int:
    """จำนวน grade ที่ห่างกัน (signed): + = over-estimate, - = under-estimate"""
    try:
        return GRADE_ORDER.index(pred_grade) - GRADE_ORDER.index(true_grade)
    except ValueError:
        return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    image_files = sorted(RAW_DIR.glob("*.jpg"))
    if not image_files:
        print(f"ไม่พบภาพใน {RAW_DIR}")
        sys.exit(1)

    print(f"พบ {len(image_files)} ภาพ — เริ่ม batch evaluation")
    print(f"บันทึกผลที่: {OUTPUT_CSV}")

    # โหลด checkpoint ถ้ามี
    done_files: set = set()
    rows: list[dict] = []
    if CHECKPOINT.exists():
        df_ck = pd.read_csv(CHECKPOINT)
        done_files = set(df_ck["filename"].tolist())
        rows = df_ck.to_dict("records")
        print(f"  โหลด checkpoint: {len(done_files)} ภาพทำแล้ว")

    errors: list[str] = []

    for img_path in tqdm(image_files, desc="Evaluating", unit="img"):
        if img_path.name in done_files:
            continue

        meta = parse_filename(img_path.name)
        if meta is None:
            errors.append(f"parse error: {img_path.name}")
            continue

        try:
            out = predict(img_path, variety=meta["variety"])
        except Exception as e:
            errors.append(f"{img_path.name}: {e}")
            rows.append({"filename": img_path.name, "error": str(e), **meta})
            continue

        sl    = out["shelf_life"]
        feats = out["features"]
        true_grade = str(day_to_grade(meta["true_day"]))
        pred_grade = out["grade"]
        err        = round(out["predicted_day"] - meta["true_day"], 4)
        gdiff      = grade_diff(pred_grade, true_grade)

        row = {
            # ── ข้อมูลจากชื่อไฟล์ ──────────────────────────────
            "filename":    img_path.name,
            "variety":     meta["variety"],
            "plant_id":    meta["plant_id"],
            "ab_group":    meta["ab_group"],
            "true_day":    meta["true_day"],
            "session":     meta["session"],
            "view":        meta["view"],

            # ── ผลโมเดลหลัก ─────────────────────────────────────
            "predicted_day":   out["predicted_day"],
            "true_grade":      true_grade,
            "pred_grade":      pred_grade,
            "grade_correct":   pred_grade == true_grade,
            "grade_adjacent":  abs(gdiff) <= 1,
            "grade_diff":      gdiff,          # + = over, - = under

            # ── error ────────────────────────────────────────────
            "error":       err,                # predicted - true (signed)
            "abs_error":   abs(err),
            "sq_error":    round(err ** 2, 6),

            # ── shelf life ───────────────────────────────────────
            "status":                      sl["status"],
            "days_to_marketability_limit": sl["days_to_marketability_limit"],
            "days_to_unusable":            sl["days_to_unusable"],

            # ── variety detection ────────────────────────────────
            "variety_detected":   out["variety"],
            "variety_confidence": out["variety_confidence"],
            "variety_source":     out["variety_source"],
            "variety_correct":    out["variety"] == meta["variety"],

            # ── flags ────────────────────────────────────────────
            "low_confidence":   out["low_confidence"],
            "gok_extrapolation": out["gok_extrapolation"],
            "area_ratio":       out["area_ratio"],

            # ── features (15 ตัว รวม b_std) ─────────────────────
            "feat_a_mean":       feats.get("a_mean"),
            "feat_a_std":        feats.get("a_std"),
            "feat_L_mean":       feats.get("L_mean"),
            "feat_L_std":        feats.get("L_std"),
            "feat_b_mean":       feats.get("b_mean"),
            "feat_b_std":        feats.get("b_std"),
            "feat_pct_green":    feats.get("pct_green"),
            "feat_pct_yellow":   feats.get("pct_yellow"),
            "feat_pct_brown":    feats.get("pct_brown"),
            "feat_area_ratio":   feats.get("area_ratio"),
            "feat_variety_enc":  feats.get("variety_enc"),
            "feat_contrast":     feats.get("contrast"),
            "feat_correlation":  feats.get("correlation"),
            "feat_energy":       feats.get("energy"),
            "feat_homogeneity":  feats.get("homogeneity"),

            "run_error": None,
        }

        rows.append(row)
        done_files.add(img_path.name)

        # checkpoint ทุก 100 ภาพ — เขียน temp แล้ว rename (atomic, ป้องกัน lock)
        if len(rows) % 100 == 0:
            tmp = CHECKPOINT.with_suffix(".tmp")
            try:
                pd.DataFrame(rows).to_csv(tmp, index=False)
                tmp.replace(CHECKPOINT)
            except Exception:
                pass  # ข้ามถ้าเขียนไม่ได้ — จะลองรอบหน้า

    # บันทึก final
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    CHECKPOINT.unlink(missing_ok=True)

    # ── สรุปผล ──────────────────────────────────────────────────
    valid = df[df["grade_correct"].notna()]
    print(f"\n{'='*50}")
    print(f"ภาพทั้งหมด:        {len(df)}")
    print(f"Error (parse/run): {len(errors)}")
    print(f"Grade accuracy:    {valid['grade_correct'].mean()*100:.2f}%")
    print(f"Adjacent accuracy: {valid['grade_adjacent'].mean()*100:.2f}%")
    print(f"MAE:               {valid['abs_error'].mean():.4f} วัน")
    print(f"RMSE:              {(valid['sq_error'].mean()**0.5):.4f} วัน")
    print(f"\nบันทึกแล้วที่: {OUTPUT_CSV}")
    print(f"จำนวนคอลัมน์: {len(df.columns)}")

    if errors:
        print(f"\nภาพที่มีปัญหา ({len(errors)} ภาพ):")
        for e in errors[:10]:
            print(f"  {e}")


if __name__ == "__main__":
    main()
