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
**วันที่:** 2026-05-12

### สิ่งที่ทำ
- `src/features/color.py`: color features ใน Lab* space (L/a/b mean+std, pct_green/yellow/brown)
- `src/features/texture.py`: GLCM features (contrast, correlation, energy, homogeneity) จาก L* channel
- `src/features/extract.py`: pipeline รวม → `data/features.csv`
- `notebooks/01_feature_check.ipynb`: sanity check trends + correlation heatmap

### ผล extract
- 2,920 rows × 23 columns
- NaN: temp_min/max 240 แถว (D3-M, D4-E, D8-M — รอเติม TODO #1)
- features: filename, variety, plant_id, ab_group, day, session, view, L_mean, L_std, a_mean, a_std, b_mean, b_std, pct_green, pct_yellow, pct_brown, contrast, correlation, energy, homogeneity, area_ratio, temp_min, temp_max

### ผล feature check
| Feature | Corr กับ day | หมายเหตุ |
|---------|-------------|---------|
| a_mean | +0.86 | best single predictor |
| area_ratio | -0.74 | ใบหดตัวตามวัน |
| pct_yellow | +0.72 | |
| pct_green | -0.68 | |
| L_mean | +0.27 | non-linear — keep ไว้ให้ model จัดการ |
| energy, homogeneity | ~0 | ตัดออกตอน Phase 4 |

### Acceptance
- ✅ features.csv 2920 rows ไม่มี NaN นอกจาก temp ที่ยังไม่มีข้อมูล
- ✅ 01_feature_check.ipynb: trend สมเหตุสมผลทางชีววิทยา, top features ชัดเจน

---

## Phase 3 — EDA
**วันที่:** 2026-05-13

### สิ่งที่ทำ
- `notebooks/02_eda.ipynb`: trend, boxplot, view comparison, outlier detection, scatter

### ผล EDA
| ประเด็น | สรุป |
|--------|------|
| Top features | a_mean, area_ratio, pct_green, pct_yellow, pct_brown |
| view difference | minimal — top/side แทบทับกัน รวม view ได้ ไม่ต้องแยก model |
| variety difference | ทิศทางเดียวกัน แต่ baseline ต่าง (GOK เขียวกว่า/area สูงกว่า) → ใช้ threshold แยกพันธุ์ |
| outlier | 336 แถว (3×IQR) → flag ไว้ ไม่ drop |
| L_mean | ไม่ monotonic พอ → feature เสริม หรือตัดออกใน Phase 4 |
| pct_brown | สัญญาณเริ่มหลัง D4–D5 เท่านั้น — early indicator ไม่ดี |
| core features | a_mean × area_ratio เห็น day gradient ชัดทั้ง 2 พันธุ์ |

### การตัดสินใจสำคัญ
- รวม top+side เป็น model เดียว (view difference minimal)
- ต้องแยก threshold ตาม variety (baseline ต่าง)
- outlier 336 แถว → flag ไว้ ให้ model จัดการ

### Acceptance
- ✅ 02_eda.ipynb รันครบทุก section
- ✅ สรุป EDA กรอกครบ

---

## Phase 4 — Model Training
**วันที่:** 2026-05-13

### สิ่งที่ทำ
- `src/model.py`: XGBoost regressor, GroupKFold(5) by plant_id, save/load
- `notebooks/03_model.ipynb`: train, feature importance, pred vs actual, error by day
- `models/xgb_model.json`: model ที่ refit บน data ทั้งหมด

### ผล CV (GroupKFold 5)
| Metric | Mean | Std |
|--------|------|-----|
| MAE | — | — |
| RMSE | — | — |
| R² | — | — |

| Variety | MAE | RMSE | R² |
|---------|-----|------|----|
| COS | — | — | 0.888 |
| GOK | — | — | 0.898 |

### Feature Importance
| Feature | Importance |
|---------|-----------|
| a_mean | ~46% |
| pct_yellow | ~24% |
| pct_brown | ~8% |
| รวม top 3 | ~78% |
| variety_enc | ต่ำสุด |

### การตัดสินใจสำคัญ
- **Regression + map → grade** (ไม่ train classifier ตรง)
  เหตุผล: threshold ยังไม่ calibrate (Phase 5), ordinal order ถูกต้อง, ไม่ต้อง retrain ถ้าปรับ boundary
- ไม่แยก model ตาม variety (R² ใกล้กันมาก)
- Boundary effect D0/D7/D8 ยอมรับได้ในระดับ grade

### Acceptance
- ✅ src/model.py เขียนเสร็จ
- ✅ 03_model.ipynb รันครบ
- ✅ models/xgb_model.json บันทึกแล้ว

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
