import csv
import re
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.preprocess import preprocess_pipeline

# HSV range สำหรับ lettuce — ขยาย H ลงถึง 10° เพื่อรับใบน้ำตาลจัด D7–D8
# V > 25 กรองพื้นหลังดำออก (threshold ต่ำกว่า 30 เพื่อรับใบที่คล้ำมาก)
HSV_H_MIN = 10   # hue lower bound (degrees / 2 ใน OpenCV = 5)
HSV_H_MAX = 90   # hue upper bound (degrees / 2 ใน OpenCV = 45)
HSV_V_MIN = 25   # brightness lower bound

# area threshold — ถ้า largest component < 5% → flag low confidence
MIN_AREA_RATIO = 0.05

FILENAME_PATTERN = re.compile(
    r'^(COS|GOK)(\d{2})_([AB])_D(\d+)_([ME])_(top|side)\.jpg$',
    re.IGNORECASE
)


def segment_lettuce(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    ตัดพื้นหลังออกเหลือเฉพาะใบผัก

    Returns
    -------
    mask   : binary mask (uint8, 0/255)
    cropped: image ที่พื้นหลังเป็นดำ
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # OpenCV เก็บ H เป็น [0,179] (degrees/2)
    h_min = int(HSV_H_MIN / 2)
    h_max = int(HSV_H_MAX / 2)

    # brightness mask: V > 25
    brightness_mask = (hsv[:, :, 2] > HSV_V_MIN).astype(np.uint8) * 255

    # color mask: H ใน [10°, 90°]
    color_mask = cv2.inRange(hsv, (h_min, 0, 0), (h_max, 255, 255))

    # AND ทั้ง 2 mask
    combined = cv2.bitwise_and(brightness_mask, color_mask)

    # Morphological opening (5x5) — กำจัด noise เล็ก
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_open)

    # Morphological closing (7x7) — อุดรูในใบ
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)

    # หา largest connected component
    mask = _largest_component(closed)

    cropped = cv2.bitwise_and(img, img, mask=mask)
    return mask, cropped


def _largest_component(binary: np.ndarray) -> np.ndarray:
    """คืน mask เฉพาะ connected component ที่ใหญ่ที่สุด"""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    if num_labels <= 1:
        return np.zeros_like(binary)

    # label 0 = background → เริ่มจาก label 1
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = np.where(labels == largest_label, np.uint8(255), np.uint8(0))
    return mask


def segment_all(
    input_dir: str | Path,
    output_dir: str | Path,
    issues_log: str | Path = "data/segment_issues.csv",
) -> dict:
    """
    Segment ทุกภาพใน input_dir
    บันทึก mask → output_dir/masks/ และ cropped → output_dir/cropped/
    ภาพที่มีปัญหา log ลง issues_log

    Returns dict สรุป: total, success, flagged
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    masks_dir = output_dir / "masks"
    cropped_dir = output_dir / "cropped"
    masks_dir.mkdir(parents=True, exist_ok=True)
    cropped_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(input_dir.glob("*.jpg"))
    total_pixels = 512 * 512

    issues = []
    success = 0
    flagged = 0

    for img_path in tqdm(image_files, desc="Segmenting"):
        try:
            img = preprocess_pipeline(str(img_path))
            mask, cropped = segment_lettuce(img)

            # ตรวจ area
            area = int(np.sum(mask > 0))
            area_ratio = area / total_pixels
            flag = None

            if area_ratio < MIN_AREA_RATIO:
                flag = "low_confidence_area"
                flagged += 1

            # ตรวจ special cases จากชื่อไฟล์
            m = FILENAME_PATTERN.match(img_path.name)
            if m:
                variety = m.group(1).upper()
                plant_id = int(m.group(2))
                day = int(m.group(4))
                if (variety == "GOK" and plant_id == 10 and day == 1) or \
                   (variety == "GOK" and plant_id == 6 and day == 6):
                    note = "known leaf loss — check mask"
                    issues.append({
                        "filename": img_path.name,
                        "issue": note,
                        "area_ratio": round(area_ratio, 4),
                    })

            if flag:
                issues.append({
                    "filename": img_path.name,
                    "issue": flag,
                    "area_ratio": round(area_ratio, 4),
                })

            # บันทึก mask และ cropped
            stem = img_path.stem
            cv2.imwrite(str(masks_dir / f"{stem}_mask.png"), mask)
            cropped_bgr = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(cropped_dir / f"{stem}_cropped.jpg"), cropped_bgr)
            success += 1

        except Exception as e:
            flagged += 1
            issues.append({
                "filename": img_path.name,
                "issue": f"error: {e}",
                "area_ratio": 0.0,
            })

    # บันทึก issues log
    issues_log = Path(issues_log)
    issues_log.parent.mkdir(parents=True, exist_ok=True)
    with open(issues_log, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "issue", "area_ratio"])
        writer.writeheader()
        writer.writerows(issues)

    print(f"\nSegmented {success} / {len(image_files)} images, {flagged} flagged")
    print(f"Issues log: {issues_log}")
    return {"total": len(image_files), "success": success, "flagged": flagged}


if __name__ == "__main__":
    segment_all(
        input_dir="data/raw",
        output_dir="data/segmented",
        issues_log="data/segment_issues.csv",
    )
