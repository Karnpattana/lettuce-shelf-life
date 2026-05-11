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
**Git tag:** `phase-1-done`

### สิ่งที่ทำ
- **Step 1.1** `src/preprocess.py`: `load_image` (BGR→RGB), `resize_with_padding` (512×512 pad ดำ), `denoise` (Gaussian 5×5 σ=1)
- **Step 1.2** `src/segment.py`: HSV mask H=[10°,90°] + V>25 → AND → opening(5×5) → closing(7×7) → largest component
- **Step 1.3** `notebooks/00_segment_check.ipynb`: สุ่ม 5 ภาพ/Day + ภาพใน issues log
- **Step 1.4** edge cases: `area_ratio < 5%` → flag, GOK06 D6 / GOK10 D1 (leaf loss) → log

### ผลการรัน
- Segmented **2,920 / 2,920 images** (100%), **0 ภาพ low_confidence_area**
- Issues log: **16 แถว** ล้วนเป็น "known leaf loss" (GOK06 D6: 8 ภาพ, GOK10 D1: 8 ภาพ) — ตัดได้แต่ flag ไว้

### Acceptance ผ่าน
- ✅ `python -m src.segment` รันเสร็จไม่ error
- ✅ masks + cropped ครบ 2,920 ไฟล์ ทุกภาพ 512×512
- ✅ `segment_issues.csv` พร้อม (16 flagged known cases)
- ✅ มีรายงาน: "Segmented 2920 / 2920 images, 0 flagged"
- ⏳ ดูตา 40–50 ภาพ ใน segment_check notebook (ทำหลัง commit)

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
| 2 | ตรวจ segmentation ภาพ D6–D8 (ใบน้ำตาลจัด) เป็นพิเศษ | Phase 1 | ⏳ |
| 3 | GOK ไม่มี D8 → model อาจ predict D8 สำหรับ GOK ไม่ได้ | Phase 4 | ⏳ |

---

## การตัดสินใจสำคัญ (Decision Log)

| วันที่ | เรื่อง | ตัดสินใจ | เหตุผล |
|--------|--------|---------|--------|
| 2026-05-12 | รวม D8 ใน dataset | รวม | มีภาพ COS จริง 40 ภาพ เพิ่ม data point ระยะท้าย |
| 2026-05-12 | D3-M / D4-E metadata | แก้เป็น "ไม่ทราบอุณหภูมิ" | ภาพมีจริง metadata เดิมผิด |
| 2026-05-12 | HSV H range | ขยายเป็น [10°, 90°] | รับใบน้ำตาลจัดที่ D7–D8 (hue ~10°–25°) |
| 2026-05-12 | Δ-features baseline | แยกตาม view (top/side) | top กับ side มี feature ต่างกันมาก |
