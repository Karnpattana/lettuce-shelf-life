"""
Test #2 — Feature contract: ยืนยันว่า feature ที่ extractor ผลิตได้
ครอบคลุม FEATURE_COLS ทุกตัวและไม่มี key หายไป

วิธีทำงาน: สร้าง synthetic image (สีเขียวบน background ดำ) แล้วรัน
extract_color + extract_texture แล้วรวมกับ area_ratio + variety_enc
แล้วตรวจว่าครอบคลุม FEATURE_COLS ครบทุกตัว

ถ้า test นี้พัง หมายความว่า feature order หรือ key เปลี่ยนไป
→ model จะรับ input ผิดโดยไม่มี error ใดๆ เตือน
"""
import math

import numpy as np
import pytest

from src.features.color import extract_color
from src.features.texture import extract_texture
from src.model import FEATURE_COLS


# ---------------------------------------------------------------------------
# Fixtures: synthetic image
# ---------------------------------------------------------------------------

@pytest.fixture
def green_patch():
    """512×512 RGB image ที่มีสี่เหลี่ยมเขียวขนาด 300×300 ตรงกลาง"""
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    img[106:406, 106:406] = [34, 139, 34]  # Forest Green
    return img


@pytest.fixture
def green_mask():
    """Mask ที่ตรงกับ green_patch"""
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[106:406, 106:406] = 255
    return mask


@pytest.fixture
def empty_mask():
    return np.zeros((512, 512), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Test #2a — FEATURE_COLS ครอบคลุมครบ
# ---------------------------------------------------------------------------

def test_all_feature_cols_present(green_patch, green_mask):
    """extractor ต้องผลิต key ทุกตัวใน FEATURE_COLS"""
    color_feats = extract_color(green_patch, green_mask)
    texture_feats = extract_texture(green_patch, green_mask)

    # รวม feature ทั้งหมดที่ inference.py จะสร้าง
    all_feats = {
        **color_feats,
        **texture_feats,
        "area_ratio": 0.3,
        "variety_enc": 0,
    }

    missing = [col for col in FEATURE_COLS if col not in all_feats]
    assert not missing, (
        f"Feature ต่อไปนี้อยู่ใน FEATURE_COLS แต่ไม่ได้ถูกผลิตโดย extractor: {missing}\n"
        f"ถ้าเพิ่ม/ลบ feature ต้องอัปเดต FEATURE_COLS ใน src/model.py ด้วย"
    )


def test_no_extra_unexpected_cols(green_patch, green_mask):
    """ตรวจว่า extractor ไม่ผลิต feature นอกเหนือจากที่รู้จัก (เพื่อจับ rename)"""
    color_feats = extract_color(green_patch, green_mask)
    texture_feats = extract_texture(green_patch, green_mask)

    # b_std ผลิตแต่ไม่ได้ใช้ใน model — feature importance ต่ำ ตัดออกตอน feature selection
    known_keys = set(FEATURE_COLS) | {"b_std"}
    produced_keys = set(color_feats) | set(texture_feats) | {"area_ratio", "variety_enc"}

    unexpected = produced_keys - known_keys
    assert not unexpected, (
        f"Extractor ผลิต feature ที่ไม่อยู่ใน FEATURE_COLS และไม่ได้ระบุว่าไม่ใช้: {unexpected}\n"
        f"ถ้าเพิ่ม feature ใหม่ ให้เพิ่มใน FEATURE_COLS หรือ whitelist ด้านบนนี้"
    )


def test_feature_values_are_finite(green_patch, green_mask):
    """ค่า feature ต้องไม่เป็น NaN หรือ inf เมื่อ mask ปกติ"""
    color_feats = extract_color(green_patch, green_mask)
    texture_feats = extract_texture(green_patch, green_mask)

    all_feats = {**color_feats, **texture_feats}
    for key, val in all_feats.items():
        assert math.isfinite(val), f"feature '{key}' = {val} (ไม่ใช่ตัวเลขปกติ)"


# ---------------------------------------------------------------------------
# Test #2b — empty mask คืน NaN (ไม่ crash)
# ---------------------------------------------------------------------------

def test_empty_mask_returns_nan_not_crash(green_patch, empty_mask):
    """ถ้า mask ว่างเปล่า (ไม่มี pixel ใบ) ต้องคืน NaN ไม่ใช่ raise exception"""
    color_feats = extract_color(green_patch, empty_mask)
    texture_feats = extract_texture(green_patch, empty_mask)

    for key, val in color_feats.items():
        assert math.isnan(val), f"color feature '{key}' ควรเป็น NaN เมื่อ mask ว่าง"

    for key, val in texture_feats.items():
        assert math.isnan(val), f"texture feature '{key}' ควรเป็น NaN เมื่อ mask ว่าง"


# ---------------------------------------------------------------------------
# Test #2c — FEATURE_COLS order ต้อง stable (ป้องกัน silent scramble)
# ---------------------------------------------------------------------------

def test_feature_cols_order_is_pinned():
    """
    ยึดลำดับของ FEATURE_COLS ไว้ตรงๆ
    ถ้า list นี้เปลี่ยนลำดับ model จะรับ input ผิดโดยไม่มี error
    """
    expected_order = [
        'a_mean', 'area_ratio', 'pct_green', 'pct_yellow', 'pct_brown',
        'L_mean', 'b_mean', 'a_std', 'L_std',
        'variety_enc',
        'contrast', 'correlation', 'energy', 'homogeneity',
    ]
    assert FEATURE_COLS == expected_order, (
        "FEATURE_COLS ถูกเปลี่ยนลำดับ!\n"
        f"ที่คาดหวัง: {expected_order}\n"
        f"ที่พบจริง:  {FEATURE_COLS}\n"
        "การเปลี่ยนลำดับทำให้ model รับ feature ผิดตำแหน่ง → prediction ผิดทั้งหมด\n"
        "ถ้าต้องการเปลี่ยนจริงๆ ต้อง retrain model แล้วอัปเดต expected_order นี้ด้วย"
    )
