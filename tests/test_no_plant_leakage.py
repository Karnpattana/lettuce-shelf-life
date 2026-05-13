"""
Test #5 — GroupKFold data leakage guard
ยืนยันว่าการ split ด้วย GroupKFold by plant_id ไม่มี plant_id
ซ้ำกันระหว่าง train fold กับ val fold เลย

ถ้า test นี้พัง หมายความว่า CV setup มี data leakage
→ metric ที่รายงาน (MAE, R²) สูงเกินจริง → กรรมการ defense จะตั้งคำถาม

Skip ถ้า data/features.csv ไม่มี (CI ที่ไม่มีข้อมูล)
"""
from pathlib import Path

import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

from src.model import FEATURE_COLS, TARGET

FEATURES_CSV = Path("data/features.csv")

pytestmark = pytest.mark.skipif(
    not FEATURES_CSV.exists(),
    reason="data/features.csv ไม่มีอยู่ (ข้าม CI)",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def features_df():
    df = pd.read_csv(FEATURES_CSV)
    df["variety_enc"] = (df["variety"] == "GOK").astype(int)
    return df


# ---------------------------------------------------------------------------
# Test #5a — ไม่มี plant_id ซ้ำระหว่าง train/val fold
# ---------------------------------------------------------------------------

def test_groupkfold_no_plant_id_overlap(features_df):
    """
    ทุก fold ต้องไม่มี plant_id เดียวกันอยู่ทั้งใน train และ val
    นี่คือ invariant หลักที่ป้องกัน data leakage
    """
    X = features_df[FEATURE_COLS].values
    y = features_df[TARGET].values
    groups = features_df["plant_id"].values

    gkf = GroupKFold(n_splits=5)
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        train_plants = set(groups[train_idx])
        val_plants   = set(groups[val_idx])
        overlap = train_plants & val_plants
        assert not overlap, (
            f"Fold {fold_idx + 1}: plant_id {overlap} ปรากฏอยู่ทั้งใน train และ val\n"
            f"→ data leakage ทำให้ metric สูงเกินจริง"
        )


# ---------------------------------------------------------------------------
# Test #5b — ทุก plant_id ถูก assign ลง val fold ครบทุกตัว (ไม่มีตกหล่น)
# ---------------------------------------------------------------------------

def test_groupkfold_all_plants_validated(features_df):
    """ทุก plant_id ต้องได้เป็น val อย่างน้อยหนึ่งครั้งใน 5 folds"""
    X = features_df[FEATURE_COLS].values
    y = features_df[TARGET].values
    groups = features_df["plant_id"].values

    all_plants = set(groups)
    validated_plants: set = set()

    gkf = GroupKFold(n_splits=5)
    for _, val_idx in gkf.split(X, y, groups):
        validated_plants |= set(groups[val_idx])

    never_validated = all_plants - validated_plants
    assert not never_validated, (
        f"plant_id ต่อไปนี้ไม่เคยได้เป็น val fold เลย: {never_validated}\n"
        f"→ ไม่มี OOF prediction สำหรับ plant เหล่านี้"
    )


# ---------------------------------------------------------------------------
# Test #5c — จำนวน fold ถูกต้อง (5 folds ตาม CLAUDE.md)
# ---------------------------------------------------------------------------

def test_groupkfold_produces_five_folds(features_df):
    X = features_df[FEATURE_COLS].values
    y = features_df[TARGET].values
    groups = features_df["plant_id"].values

    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(X, y, groups))
    assert len(folds) == 5, f"ต้องได้ 5 folds ได้ {len(folds)}"


# ---------------------------------------------------------------------------
# Test #5d — ข้อมูลมี variety ทั้งสอง (COS + GOK) และ day ครบ
# ---------------------------------------------------------------------------

def test_dataset_has_both_varieties(features_df):
    varieties = set(features_df["variety"].unique())
    assert "COS" in varieties, "dataset ต้องมี variety COS"
    assert "GOK" in varieties, "dataset ต้องมี variety GOK"


def test_dataset_day_range(features_df):
    """COS ต้องมีวัน 0–8, GOK ต้องมีวัน 0–6 (ไม่มี D8 ตาม CLAUDE.md)"""
    cos_days = set(features_df[features_df["variety"] == "COS"]["day"].unique())
    gok_days = set(features_df[features_df["variety"] == "GOK"]["day"].unique())

    assert 0 in cos_days and 8 in cos_days, f"COS ควรมีวัน 0 และ 8 ได้: {sorted(cos_days)}"
    assert 0 in gok_days, f"GOK ควรมีวัน 0 ได้: {sorted(gok_days)}"
    assert 8 not in gok_days, (
        f"GOK ไม่ควรมีวัน 8 (ตาม CLAUDE.md ไม่มี D8 training data) ได้: {sorted(gok_days)}"
    )


# ---------------------------------------------------------------------------
# Test #5e — FEATURE_COLS ทุกตัวมีอยู่ใน features.csv (ไม่มี key หาย)
# ---------------------------------------------------------------------------

def test_feature_cols_exist_in_csv(features_df):
    """ยืนยันว่า features.csv มีทุก column ที่ model ต้องการ"""
    missing = [col for col in FEATURE_COLS if col not in features_df.columns]
    assert not missing, (
        f"FEATURE_COLS มี column ที่ไม่อยู่ใน features.csv: {missing}\n"
        f"columns ที่มีใน CSV: {features_df.columns.tolist()}"
    )
