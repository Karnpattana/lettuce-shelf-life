# Development Log — Lettuce Shelf Life Prediction

บันทึกการทำงานแต่ละขั้นตอน: สิ่งที่ทำ / ตัดสินใจ / เหตุผล  
ใช้อ้างอิงตอน thesis defense เพื่อแสดง iterative development process

---

## Phase 0 — Project Setup
**วันที่:** 2026-05-12  
**Git tag:** `phase-0-done`

### สิ่งที่ทำ
- สร้างโครงสร้างโฟลเดอร์ตาม requirement v4
- เขียน `requirements.txt` (opencv, scikit-image, scikit-learn, xgboost, pandas, streamlit ฯลฯ)
- สร้าง `.gitignore` กัน raw images / models / segmented data ไม่ให้เข้า git
- สร้าง `src/__init__.py` และ `src/features/__init__.py`  
  — `features/__init__.py` เก็บ shared Lab* thresholds (GREEN/YELLOW/BROWN_LAB_RANGE) ป้องกัน threshold drift ระหว่างไฟล์
- สร้าง `data/metadata.csv` placeholder (อุณหภูมิรอเติมทีหลัง)
- เขียน `README.md`
- ติดตั้ง virtual environment `.venv` + รัน `pip install -r requirements.txt` สำเร็จ
- Init git repo, commit แรก, tag `phase-0-done`

### Acceptance ผ่าน
- ✅ `import cv2, skimage, sklearn, xgboost, pandas, yaml` ไม่ error
- ✅ metadata.csv มีครบทุกรอบถ่าย
- ✅ โครงสร้างโฟลเดอร์ตรงแผน

---

## Phase 0 → Phase 1 — ตรวจ Dataset (ก่อนเริ่ม Phase 1)
**วันที่:** 2026-05-12

### ผลการตรวจ
- ภาพทั้งหมด **2,920 ไฟล์** (COS 1,480 | GOK 1,440)
- Day range: D0–D8 (requirement เดิมระบุ D0–D7 แต่พบ D8 เพิ่มมา)
- **D8** มีเฉพาะ COS01–10, session M เท่านั้น (40 ภาพ = 10 ต้น × 2 sides × 2 views)
- GOK ไม่มี D8

### การตัดสินใจ
| เรื่อง | การตัดสินใจ | เหตุผล |
|--------|------------|--------|
| รวม D8 หรือไม่ | **รวม** | มีภาพจริง เพิ่ม data point วันเสื่อมจัด ช่วย model เรียน degradation ระยะท้าย |
| D3-M และ D4-E กลุ่ม 01-10 | **ใช้ภาพได้** (แก้ metadata) | ภาพมีอยู่จริง 80 ภาพ/session — metadata เดิมบันทึกผิด |

### แก้ไข metadata.csv
- แถว D3-M กลุ่ม 01-10: ลบ "missing session" → บันทึกว่าอุณหภูมิยังไม่ทราบ
- แถว D4-E กลุ่ม 01-10: ลบ "missing session" → บันทึกว่าอุณหภูมิยังไม่ทราบ
- เพิ่มแถว D8-M กลุ่ม 01-10 (อุณหภูมิยังไม่ทราบ)
- ⚠️ **TODO:** เติมอุณหภูมิจริงสำหรับ D3-M, D4-E, D8-M ก่อน Phase 2

### สถิติภาพแต่ละ Day
| Day | COS | GOK | รวม |
|-----|-----|-----|-----|
| D0 | 120 | 120 | 240 |
| D1 | 160 | 160 | 320 |
| D2 | 240 | 240 | 480 |
| D3 | 200 | 200 | 400 |
| D4 | 240 | 240 | 480 |
| D5 | 240 | 240 | 480 |
| D6 | 160 | 160 | 320 |
| D7 | 80 | 80 | 160 |
| D8 | 40 | 0 | 40 |
| **รวม** | **1,480** | **1,440** | **2,920** |

---

## Phase 1 — Image Preprocessing + Segmentation
**วันที่:** 2026-05-12  
**Git tag:** `phase-1-done` (เดิม) → ต้อง re-run + tag ใหม่หลัง segment.py แก้

### สิ่งที่ทำ
- **Step 1.1** `src/preprocess.py`: `load_image` (BGR→RGB), `resize_with_padding` (512×512 pad ดำ), `denoise` (Gaussian 5×5 σ=1)
- **Step 1.2** `src/segment.py`: HSV mask H=[10°,90°] + V>25 → opening(5×5) → largest component → **binary_fill_holes**
- **Step 1.3** `notebooks/00_segment_check.ipynb`: สุ่ม 5 ภาพ/Day + ภาพใน issues log
- **Step 1.4** edge cases: `area_ratio < 5%` → flag, GOK06 D6 / GOK10 D1 (leaf loss) → log

### การแก้ไข segment.py (หลัง visual check)
- ปัญหา: เงาในใบถูกตัดออกเป็นพื้นหลัง (รูเล็กกระจายทั่วใบ)
- แก้: เปลี่ยนจาก `closing kernel` → `binary_fill_holes` (scipy)
  - `binary_fill_holes` อุดรูทุกขนาดในใบโดยไม่ต้องเดา kernel size
  - Pipeline ใหม่: HSV mask → opening(5×5) → largest component → binary_fill_holes
- ผลทดสอบ (`_test_segment.py`): mask ขาวทึบสม่ำเสมอ ไม่มีรู ✅

### ⚠️ งานค้าง — ต้องทำต่อในแชทหน้า
1. **re-run `python -m src.segment`** ทั้ง 2,920 ภาพด้วย segment.py เวอร์ชันใหม่
   - ไฟล์ใน `data/segmented/` ยังเป็นเวอร์ชันเก่า (closing 7×7, มีรู)
   - รันแล้วจะ overwrite ทับอัตโนมัติ
2. **ดูตา 40–50 ภาพ** ใน `notebooks/00_segment_check.ipynb` หลัง re-run
3. **commit** หลัง re-run เสร็จ + tag `phase-1-done` ใหม่

### วิธีเริ่มต่อ (แชทหน้า)
```
python -m src.segment
```
แล้วเปิด `notebooks/00_segment_check.ipynb` ดูภาพ ถ้าโอเค → Phase 2

### ผลตรวจตา (visual check)
| | จำนวน | ถูก | % |
|---|---|---|---|
| Section 1 (สุ่ม 5/Day) | 45 | 36 | 80.0% |
| Section 2 (issues log) | 16 | 16 | 100.0% |
| **รวม** | **61** | **52** | **85.2%** |

ต่ำกว่า acceptance 90% — ยอมรับและบันทึกเป็น known issues (ดู segment_issues.csv)

### Known issues จาก visual check
| ภาพ | ปัญหา |
|-----|-------|
| COS18_B_D0_E_side | เงาชัดมาก ตัดใบออกเยอะ (worst case) |
| COS07_B_D1_E_side | ตัดใบออกเล็กน้อย ~<7% |
| GOK11_A_D6_M_top | เงารวมเข้าใบ ~15–20% (false positive) |
| GOK D5–D8 ทั่วไป | ตัดขอบใบออกเล็กน้อย — ยอมรับได้ |

การตัดสินใจ: **ยอมรับ 85.2% และไป Phase 2** — ปัญหาหลักคือเงาใกล้ขอบใบซึ่งแก้ไม่ได้โดยไม่รวม background เข้า mask (background-in-mask แย่กว่า missing-leaf สำหรับ feature extraction)

### Acceptance
- ✅ `src/preprocess.py` และ `src/segment.py` เขียนเสร็จ
- ✅ `notebooks/00_segment_check.ipynb` เขียนเสร็จ
- ✅ re-run segment 2920/2920 ด้วย binary_fill_holes
- ✅ `segment_issues.csv` อัปเดต (16 leaf-loss + 3 shadow issues)
- ✅ ดูตา 61 ภาพ → 85.2% (ต่ำกว่า 90% แต่ยอมรับ — บันทึก known issues แล้ว)

---

## Phase 2 — Feature Extraction
*(รอ Phase 1)*

---

## Phase 3 — EDA
*(รอ Phase 2)*

---

## Phase 4 — Model Training
*(รอ Phase 3)*

---

## Phase 5 — Grade Mapping & Threshold Calibration
*(รอ Phase 4)*

---

## Phase 6 — Inference Pipeline
*(รอ Phase 5)*

---

## Phase 7 — Demo UI (Optional)
*(รอ Phase 6)*

---

## Known Issues / TODO

| # | รายการ | Phase ที่เกี่ยวข้อง | สถานะ |
|---|--------|-------------------|-------|
| 1 | เติมอุณหภูมิ D3-M, D4-E, D8-M ใน metadata.csv | ก่อน Phase 2 | ⏳ รอข้อมูล |
| 2 | ตรวจ segmentation ภาพ D6–D8 (ใบน้ำตาลจัด) เป็นพิเศษ | Phase 1 | ✅ done — known issues บันทึกใน segment_issues.csv |
| 3 | GOK ไม่มี D8 → model อาจ predict D8 สำหรับ GOK ไม่ได้ | Phase 4 | ⏳ |

---

## การตัดสินใจสำคัญ (Decision Log)

| วันที่ | เรื่อง | ตัดสินใจ | เหตุผล |
|--------|--------|---------|--------|
| 2026-05-12 | รวม D8 ใน dataset | รวม | มีภาพ COS จริง 40 ภาพ เพิ่ม data point ระยะท้าย |
| 2026-05-12 | D3-M / D4-E metadata | แก้เป็น "ไม่ทราบอุณหภูมิ" | ภาพมีจริง metadata เดิมผิด |
| 2026-05-12 | HSV H range | ขยายเป็น [10°, 90°] | รับใบน้ำตาลจัดที่ D7–D8 (hue ~10°–25°) |
| 2026-05-12 | Δ-features baseline | แยกตาม view (top/side) | top กับ side มี feature ต่างกันมาก |
