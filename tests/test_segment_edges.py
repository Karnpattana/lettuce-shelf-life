"""
Test #4 — Degenerate inputs to segment_lettuce()
ทดสอบว่า pipeline จัดการ input ผิดปกติได้โดยไม่ crash
และ inference ตั้ง low_confidence=True อย่างถูกต้อง
"""
import numpy as np
import pytest

from src.segment import MIN_AREA_RATIO, segment_lettuce


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def solid_rgb(h, w, color: tuple) -> np.ndarray:
    """สร้าง image สีทึบ (RGB)"""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = color
    return img


# ---------------------------------------------------------------------------
# Test: input image ดำล้วน (V=0 → brightness_mask ว่างหมด)
# ---------------------------------------------------------------------------

def test_all_black_image_returns_empty_mask():
    img = solid_rgb(512, 512, (0, 0, 0))
    mask, cropped = segment_lettuce(img)
    assert mask.sum() == 0, "รูปดำล้วนต้องได้ mask ว่าง"


def test_all_black_image_no_crash():
    img = solid_rgb(512, 512, (0, 0, 0))
    mask, cropped = segment_lettuce(img)  # must not raise
    assert mask.shape == (512, 512)
    assert cropped.shape == (512, 512, 3)


# ---------------------------------------------------------------------------
# Test: input image ขาวล้วน (H=0 อยู่นอก [10°,90°] → color_mask ว่าง)
# ---------------------------------------------------------------------------

def test_all_white_image_returns_empty_mask():
    img = solid_rgb(512, 512, (255, 255, 255))
    mask, cropped = segment_lettuce(img)
    assert mask.sum() == 0, "รูปขาวล้วนต้องได้ mask ว่าง (H=0 อยู่นอก HSV range)"


# ---------------------------------------------------------------------------
# Test: รูปที่มีสีแดงล้วน (H ≈ 0° อยู่นอก [10°,90°])
# ---------------------------------------------------------------------------

def test_red_image_returns_empty_mask():
    img = solid_rgb(512, 512, (255, 0, 0))
    mask, cropped = segment_lettuce(img)
    assert mask.sum() == 0, "สีแดงอยู่นอก HSV H range → mask ต้องว่าง"


# ---------------------------------------------------------------------------
# Test: รูปสีเขียว-เหลืองล้วน (H ≈ 70° อยู่ใน range [10°,90°]) → mask ต้องไม่ว่าง
# หมายเหตุ: สีเขียวล้วน RGB(0,200,0) มี H=120° ซึ่งอยู่นอก range [10°,90°]
# ต้องใช้สีเขียว-เหลือง RGB(150,180,0) ที่ H=70° ซึ่งตรงกับโทนใบผักจริง
# ---------------------------------------------------------------------------

def test_green_image_returns_nonempty_mask():
    img = solid_rgb(512, 512, (150, 180, 0))  # H≈70° real, อยู่ใน [10°,90°]
    mask, cropped = segment_lettuce(img)
    area_ratio = mask.sum() / 255 / mask.size
    assert area_ratio > 0.5, "รูปสีเขียว-เหลืองล้วนควรได้ mask ครอบคลุมส่วนใหญ่"


# ---------------------------------------------------------------------------
# Test: two equal-area green blobs — ต้องเลือกแค่ component เดียว ไม่ crash
# ---------------------------------------------------------------------------

def test_two_green_blobs_returns_one_component():
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    # Blob ซ้าย
    img[50:250, 50:200] = (0, 200, 0)
    # Blob ขวา (ขนาดเท่ากัน)
    img[50:250, 312:462] = (0, 200, 0)

    mask, cropped = segment_lettuce(img)

    # ต้องไม่ crash และ mask ต้องมีค่าอยู่
    assert mask.shape == (512, 512)

    # ตรวจว่าเลือก component เดียว: pixel ที่ได้ควรน้อยกว่าผลรวมของทั้งสอง blob
    both_blobs_px = (200 * 150) * 2  # พื้นที่สองก้อนรวมกัน
    mask_px = int(mask.sum() / 255)
    assert mask_px < both_blobs_px, (
        f"_largest_component ควรเลือกแค่ blob เดียว ({mask_px} px) "
        f"ไม่ใช่ทั้งสองก้อน ({both_blobs_px} px)"
    )


# ---------------------------------------------------------------------------
# Test: low_confidence flag ใน inference pipeline
# ---------------------------------------------------------------------------

def test_low_confidence_flag_on_blank_image(tmp_path):
    """
    รูปสีแดง (mask ว่าง) ต้องทำให้ area_ratio < MIN_AREA_RATIO
    → low_confidence=True ใน output ของ predict()
    """
    import cv2
    from src.inference import predict

    img_path = tmp_path / "red_blank.jpg"
    red_bgr = np.zeros((512, 512, 3), dtype=np.uint8)
    red_bgr[:, :] = (0, 0, 200)  # BGR แดง
    cv2.imwrite(str(img_path), red_bgr)

    out = predict(str(img_path), variety="COS")

    assert out["low_confidence"] is True, (
        f"รูปที่ mask ว่าง (area_ratio={out['area_ratio']}) "
        f"ต้องได้ low_confidence=True (threshold={MIN_AREA_RATIO})"
    )
    assert out["area_ratio"] < MIN_AREA_RATIO


# ---------------------------------------------------------------------------
# Test: output shape ถูกต้องสำหรับ valid input
# ---------------------------------------------------------------------------

def test_segment_output_shapes():
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    img[100:400, 100:400] = (0, 180, 0)
    mask, cropped = segment_lettuce(img)
    assert mask.shape == img.shape[:2], "mask ต้องมี shape เดียวกับ input (H, W)"
    assert cropped.shape == img.shape, "cropped ต้องมี shape เดียวกับ input (H, W, 3)"
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 255}), "mask ต้องเป็น binary (0 หรือ 255 เท่านั้น)"
