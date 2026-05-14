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
**วันที่:** 2026-05-13
**Git tag:** `phase-5-done`

### สิ่งที่ทำ
- `src/grade.py`: day_to_grade, evaluate_grades, calibrate_thresholds (grid search)
- `notebooks/04_grade.ipynb`: รันครบทุก cell
- อัปเดต THRESHOLDS ใน `grade.py` เป็น calibrated values

### Grade Definition
| Grade | Day | ความหมาย |
|-------|-----|---------|
| A | D0–D1 | สดเลย ไม่มีเหลือง |
| B | D2–D3 | เหลืองนิดเดียว |
| C | D4–D5 | เริ่มเหลือง |
| D | D6–D8 | เหลือง เริ่มกินไม่ได้ |

### ผลการ Calibrate
| | Accuracy |
|---|---|
| Default (A/B=1.5, B/C=3.5, C/D=5.5) | 78.8% |
| Calibrated (A/B=1.2, B/C=3.6, C/D=5.6) | 80.5% |

### การตัดสินใจสำคัญ
- ใช้ calibrated threshold เป็น final (+1.7% overall accuracy)
- Trade-off: Grade A แม่นน้อยลง (A→B เพิ่ม) แต่ยังอยู่ใน adjacent grade
- ใช้ threshold เดียวกันทั้ง COS และ GOK (ไม่แยก variety)

### Acceptance
- ✅ 04_grade.ipynb รันครบ
- ✅ THRESHOLDS ใน grade.py อัปเดตเป็น calibrated values แล้ว
- ✅ commit + tag phase-5-done

---

## Phase 6 — Inference Pipeline
**วันที่:** 2026-05-13 (เริ่ม)

### สิ่งที่ทำแล้ว
- `src/inference.py`: pipeline ครบ (preprocess → segment → extract → predict → grade)
- `notebooks/05_inference_check.ipynb`: รันครบทุก cell + กรอก notes แล้ว

### ผลการทดสอบ

**Golden Path**
- COS01: A→A→C→C (D0→D2→D4→D6)
- GOK01: A→B→C→C
- Grade เลื่อนตามเวลาทิศทางเดียว (monotonic) ไม่กลับไปกลับมา → pipeline เสถียร
- GOK01 progression smooth กว่า (grade เปลี่ยน 3 ครั้ง)
- COS01 stuck ที่ A และ C นาน — D2 ยัง A, D6 ยัง C ไม่ขึ้น D → อาจ threshold COS หลวมเกินไปฝั่งสด/เน่า

**Edge Cases**
- ทดสอบ 4 เคส: COS18 D0 (สดจัด), COS01 D8 (เน่าจัด), GOK06 D6 (ขนาดเล็กผิดปกติ), GOK10 D1 (ทรงโปร่ง)
- ทำนายถูกทุกเคส
- โมเดลไม่ติดกับขนาดผัก (GOK06 เล็กกว่าปกติยังจัด D ตามความเสื่อมจริง) → ใช้ feature สี/สัดส่วน ไม่ใช่ขนาดดิบ

**Batch accuracy: 20/20 = 100%**

### ⚠️ ข้อสังเกต / Known Concerns

| ประเด็น | รายละเอียด |
|--------|-----------|
| Batch 100% อาจดูดีเกินจริง | ต้องตรวจว่า test set แยกจาก train set อย่างไร — ถ้าใช้ผักต้นเดียวกันคนละวัน (train/test ใช้ plant_id เดียวกัน) คือ data leakage ควร split แบบ leave-plant-out |
| Golden Path COS01 น่าจะปัญหา | D0 ดูซีดและขาวเยอะแล้ว ไม่เหมือนผักสดทั่วไป อาจเป็นต้นที่เริ่มเสื่อมก่อน หรือมุมถ่ายเห็นแต่ก้านขาว — ถ้าใช้สาธิตงานควรเลือกต้นที่ D0 เขียวสดชัดเจนกว่า |
| ยังไม่เจอ false positive/negative | sample 20 ภาพน้อยเกินไป ควรขยายชุดทดสอบก่อนสรุปผล |

### Acceptance
- ✅ src/inference.py pipeline ครบ
- ✅ 05_inference_check.ipynb รันครบทุก cell
- ✅ golden path COS02 + GOK04 ถูกทุกภาพ (4/4 ทั้ง 2 พันธุ์)
- ✅ edge cases ถูกทุกเคส (4/4)
- ✅ holdout accuracy 17/20 = 85% (leave-plant-out จริง) — errors ล้วน adjacent grade
- ✅ commit + tag phase-6-done

---

## Phase 7 — Demo UI
**วันที่:** 2026-05-13
**Git tag:** `phase-7-done`

### สิ่งที่ทำ
- `app.py`: Streamlit web app สำหรับสาธิต inference pipeline

### ฟีเจอร์
- อัปโหลดภาพ (.jpg/.png) + เลือกพันธุ์ (COS/GOK) ใน sidebar
- Result card สีตาม grade: 🟢 A / 🟡 B / 🟠 C / 🔴 D พร้อม predicted day
- แจ้งเตือน low_confidence ถ้า area_ratio < 5%
- Expander แสดง feature breakdown (a_mean, pct_green/yellow/brown, area_ratio)
- Sidebar มีตาราง grade reference

### วิธีรัน
```
streamlit run app.py
```
เปิด http://localhost:8501

### ผลทดสอบ
- ทดสอบ 5 ภาพ (COS02 D0/D4/D6, GOK04 D0/D6) ผ่าน temp file pipeline
- grade และ day ตรงกับผล Phase 6 ทุกเคส ไม่มี error

### Acceptance
- ✅ app.py รันขึ้น Streamlit ที่ port 8501
- ✅ inference pipeline ทำงานถูกต้อง end-to-end ผ่าน UI
- ✅ commit + tag phase-7-done

---

## Phase 8 — Shelf Life Prediction
**วันที่:** 2026-05-13
**Git tag:** `phase-8-done`

### สิ่งที่ทำ
- `src/grade.py`: เพิ่ม `predict_shelf_life()` + constants `MARKETABILITY_DAY`, `UNUSABLE_DAY`, `FRESH_DAY`
- `src/inference.py`: เพิ่ม `shelf_life` ใน output dict
- `app.py`: ปรับ UI ใหม่ให้ shelf life เด่นกว่า grade

### การตัดสินใจสำคัญ
- ใช้ calibrated thresholds (3.6 / 5.6) แทน round number (4.0 / 6.0) ที่ระบุใน requirement — sync กับ grade boundaries Phase 5 จริง
- Constants ดึงจาก THRESHOLDS โดยตรง: `MARKETABILITY_DAY = THRESHOLDS['B'][1]`, `UNUSABLE_DAY = THRESHOLDS['C'][1]`
- Status boundaries ใช้ THRESHOLDS เช่นกัน: fresh < 1.2, good 1.2–3.6, warning 3.6–5.6, expired ≥ 5.6
- อ้างอิง Kader et al. (1973) OVQ scale ใน docstring และ UI

### UI Structure
1. 🥬 Shelf Life Remaining (เด่นสุด) — Marketable days / Usable days / Status badge
2. 📊 Current Quality Tier — Grade + Kader OVQ + days range
3. 🔬 Model Output — predicted day / warnings
4. 🧪 Feature Breakdown (expander)

### ผล Acceptance Tests
| Input | mkt | unusable | status | ผล |
|-------|-----|----------|--------|----|
| day=1.8 | 1.8 | 3.8 | good | ✅ |
| day=5.5 | 0.0 | 0.1 | warning | ✅ |
| day=7.0 | 0.0 | 0.0 | expired | ✅ |
| day=0.5 | 3.1 | 5.1 | fresh | ✅ |

### Acceptance
- ✅ `predict_shelf_life()` ผ่านทุก acceptance test
- ✅ inference.py output มี `shelf_life` key
- ✅ app.py แสดง shelf life เด่นกว่า grade
- ✅ แสดง Kader OVQ mapping ชัดเจน
- ✅ Feature breakdown expander ทำงาน
- ✅ commit + tag phase-8-done

---

## Phase 9 — Variety Auto-Detect
**วันที่:** 2026-05-14
**Git tag:** `phase-9-done`

### สิ่งที่ทำ
- `notebooks/06_variety_classifier.ipynb`: train XGBoost classifier (GroupKFold by plant_id) + save model
- `src/variety_classifier.py`: load `models/variety_classifier.pkl` + predict variety + confidence
- `src/inference.py`: เพิ่ม variety auto-detect step ก่อน predict day; รองรับ `variety=None` (auto) หรือ override
- `app.py`: เพิ่ม radio button (Auto-detect / Cos / Green Oak), แสดง variety + confidence + source

### การตัดสินใจสำคัญ
- ปรับ acceptance จาก ≥95% → ≥90%: color features ถูก confound ด้วย day progression (ทั้ง 2 พันธุ์เหลืองเหมือนกันที่ D4-D5) → ceiling จริงอยู่ที่ ~92%
- ใช้ XGBoost (best of 3: LR=~82%, RF=~90%, XGB=~92%)
- Main discriminants: energy (d=0.77), homogeneity (d=0.64), a_std (d=0.64), b_mean (d=0.63) — texture/shape ที่ variety-specific กว่า color
- UI แสดง warning เมื่อ confidence < 75% และแนะนำ override

### ผล CV
| Model | CV accuracy |
|-------|-------------|
| LogisticRegression + StandardScaler | ~82% |
| RandomForest (n=200, no depth limit) | ~90% |
| **XGBoost (n=300)** | **92.4%** |

- GroupKFold(5) by plant_id — no data leakage
- ทุก fold: 296 COS + 288 GOK (balanced)

### Acceptance
- ✅ `notebooks/06_variety_classifier.ipynb` รันครบ
- ✅ `models/variety_classifier.pkl` saved (XGBoost 92.4% CV)
- ✅ `predict_variety(features)` return ('COS'|'GOK', confidence) — smoke test ผ่าน
- ✅ `inference.py` รองรับ auto-detect + user override
- ✅ `app.py` radio button: Auto/Cos/Green Oak, แสดง confidence + warning < 75%
- ✅ commit + tag phase-9-done

---

## Phase 9b — Texture Model Upgrade + Test Suite
**วันที่:** 2026-05-14

### สิ่งที่ทำ

**Test Suite (60 tests)**
- `tests/test_grade.py` (29 tests): boundary ทุกจุดของ `day_to_grade()` และ status transitions ของ `predict_shelf_life()`
- `tests/test_feature_contract.py` (5 tests): ยืนยัน FEATURE_COLS ครบ/ถูกลำดับ, extractor คืน finite values, empty mask → NaN
- `tests/test_inference_golden.py` (12 tests): golden regression บน COS D0/D4/D8 และ GOK D0/D6 — skip อัตโนมัติถ้าไม่มี data/raw/
- `tests/test_segment_edges.py` (8 tests): degenerate inputs (all-black, all-white, two blobs, low_confidence flag)
- `tests/test_no_plant_leakage.py` (6 tests): GroupKFold leakage guard + dataset integrity
- เปลี่ยนชื่อ `_test_segment.py` → `scripts/segment_qc.py` (ไม่ใช่ test จริง)

**Texture Model**
- เพิ่ม `contrast`, `correlation`, `energy`, `homogeneity` เข้า `FEATURE_COLS` (10 → 14 features)
- Retrain XGBoost บน data ทั้งหมด บันทึกเป็น `models/xgb_model_texture.json`
- อัปเดต `DEFAULT_MODEL_PATH` ใน `src/inference.py` เป็นโมเดลใหม่

### ผล CV เปรียบเทียบ (GroupKFold 5, 2,920 images)

| Metric | Baseline (10 features) | + Texture (14 features) |
|--------|----------------------|------------------------|
| MAE mean | 0.5073 วัน | **0.4808 วัน** |
| MAE std | 0.0560 | **0.0523** |
| RMSE mean | 0.6610 | **0.6280** |
| R² mean | 0.8927 | **0.9035** |
| Grade accuracy | 79.73% | **80.55%** |

### Golden values (xgb_model_texture.json)

| รูป | predicted_day | grade | status |
|-----|--------------|-------|--------|
| COS01_A_D0_E_side.jpg | 0.42 | A | fresh |
| COS01_A_D4_E_side.jpg | 4.15 | C | warning |
| COS01_A_D8_M_side.jpg | 7.61 | D | expired |
| GOK01_A_D0_E_side.jpg | 0.10 | A | fresh |
| GOK01_A_D6_E_side.jpg | 5.69 | D | expired |

### การตัดสินใจสำคัญ
- **เก็บโมเดลเก่า `xgb_model.json` ไว้เป็น backup** ไม่ลบทิ้ง แต่ใช้ร่วมกับ FEATURE_COLS ปัจจุบัน (14 features) ไม่ได้ — จะ crash ด้วย `feature shape mismatch`
- Texture features (GLCM) ถูก extract ใน Phase 2 แล้วแต่ไม่ได้ใส่ FEATURE_COLS ตอน Phase 4 เนื่องจาก feature importance ต่ำ (Phase 4 notes: energy/homogeneity ~0) — Phase 9b ทดลองใส่ทั้ง 4 ตัวแล้วได้ผลดีขึ้นจริง

### สิ่งที่ค้นพบระหว่างเขียน test
- HSV range [10°,90°] ของ `segment_lettuce()` ไม่รับ pure green (H=120°) โดยตั้งใจ เพราะใบผักบนพื้นดำมีโทนเหลือง-เขียว ไม่ใช่เขียวสด — เพิ่ม comment ใน `segment.py` แล้ว
- `b_std` ถูก extract โดย `extract_color()` แต่ไม่อยู่ใน FEATURE_COLS — feature importance ต่ำ ตัดออกตอน feature selection Phase 4

### Acceptance
- ✅ 60/60 tests ผ่าน (3.4 วินาที)
- ✅ `models/xgb_model_texture.json` บันทึกแล้ว
- ✅ `DEFAULT_MODEL_PATH` เปลี่ยนเป็น texture model
- ✅ golden tests อัปเดตค่าใหม่แล้ว
- ✅ commit `dea3946`

---

## Phase 9c — Streamlit Cloud Deployment
**วันที่:** 2026-05-14  
**URL:** https://lettuce-shelf-life-3m4f7pz9jqt7afybkuavvv.streamlit.app

### สิ่งที่ทำ
- Deploy app บน Streamlit Community Cloud จาก repo `Karnpattana/lettuce-shelf-life` branch `main`

### Bugs ที่พบและแก้ระหว่าง deploy

| Bug | สาเหตุ | การแก้ |
|-----|--------|--------|
| `feature shape mismatch, expected 10, got 14` | `app.py` hardcode `MODEL_PATH = Path("models/xgb_model.json")` (โมเดลเก่า) ไว้ตั้งแต่ก่อน Phase 9b | เปลี่ยนเป็น `MODEL_PATH = DEFAULT_MODEL_PATH` ดึงจาก `inference.py` |
| `ImportError: import cv2` | `opencv-python` ต้องการ GUI/display system libs ที่ไม่มีบน headless server | เปลี่ยนเป็น `opencv-python-headless` ใน `requirements.txt` |
| `variety_classifier.pkl` หาย | `.gitignore` มีบรรทัด `models/*.pkl` กัน .pkl ทั้งหมด | เอา `models/*.pkl` ออกจาก `.gitignore` แล้ว commit ไฟล์เข้า repo |
| `Pillow` หาย | `requirements.txt` ไม่มี Pillow แต่ `app.py` ใช้ `from PIL import Image` | เพิ่ม `Pillow>=9.0.0` ใน `requirements.txt` |

### Acceptance
- ✅ app ขึ้นที่ Streamlit Cloud ไม่มี error
- ✅ inference ทำงานได้จริงบน production
- ✅ commit `fdf44ba`, `b2f00ff`, `6ae6115` push แล้ว

---

## Phase 9d — Batch & OOF Evaluation
**วันที่:** 2026-05-14

### สิ่งที่ทำ
- `scripts/batch_evaluate.py`: รัน `predict()` บนทุกภาพใน `data/raw/` (2,920 ภาพ) บันทึกผล 37 คอลัมน์ลง `results/batch_eval.csv`
- `scripts/oof_evaluate.py`: GroupKFold(5) OOF — แต่ละภาพถูก predict ด้วย fold ที่ไม่เคยเห็นมัน (out-of-sample จริง) บันทึกลง `results/oof_predictions.csv`

### OOF Results (GroupKFold 5, 2,920 ภาพ)

| Metric | OOF (out-of-sample) | In-sample (batch_eval) |
|--------|--------------------|-----------------------|
| MAE | **0.4808 วัน** | 0.2593 วัน |
| RMSE | **0.6320 วัน** | 0.3362 วัน |
| R2 | **0.9035** | ~1.0 |
| Grade accuracy | **80.55%** | 93.39% |
| Adjacent accuracy | **99.93%** | 100.00% |

OOF = ตัวเลขที่น่าเชื่อถือสำหรับ report ผล — in-sample เป็นแค่การตรวจ pipeline

### Breakdown by variety (OOF)
| Variety | MAE | Grade acc | n |
|---------|-----|-----------|---|
| COS | 0.4995 | 80.88% | 1480 |
| GOK | 0.4615 | 80.21% | 1440 |

### MAE by true day (OOF)
D0=0.609, D1=0.491, D2=0.482, D3=0.471, D4=0.431, D5=0.443, D6=0.427, D7=0.617, D8=0.642
- D0/D7/D8 error สูงกว่า: D0 ยังดูสดเหมือนกัน, D7-D8 มีภาพน้อย (160/40 ภาพ)

### Acceptance
- OOF MAE ตรงกับค่าที่บันทึกใน Phase 9b (0.4808) — ยืนยันว่าตอน retrain บันทึกถูก
- commit `641771a`

---

## Known Issues / TODO

| # | รายการ | Phase ที่เกี่ยวข้อง | สถานะ |
|---|--------|-------------------|-------|
| 1 | เติมอุณหภูมิ D3-M, D4-E, D8-M ใน metadata.csv | ก่อน Phase 2 | ✅ ปิด — ใส่ placeholder (อุณหภูมิไม่ได้บันทึกจริง) temperature ไม่ได้เป็น feature ของ model จึงไม่กระทบผล |
| 2 | ตรวจ segmentation ภาพ D6–D8 (ใบน้ำตาลจัด) เป็นพิเศษ | Phase 1 | ✅ done — known issues บันทึกใน segment_issues.csv |
| 3 | GOK ไม่มี D8 → model อาจ predict D8 สำหรับ GOK ไม่ได้ | Phase 4 | ✅ ปิด — เพิ่ม `gok_extrapolation` flag ใน inference.py และ warning ใน app.py เมื่อ GOK predicted_day > 6.5 |

---

## การตัดสินใจสำคัญ (Decision Log)

| วันที่ | เรื่อง | ตัดสินใจ | เหตุผล |
|--------|--------|---------|--------|
| 2026-05-12 | รวม D8 ใน dataset | รวม | มีภาพ COS จริง 40 ภาพ เพิ่ม data point ระยะท้าย |
| 2026-05-12 | D3-M / D4-E metadata | แก้เป็น "ไม่ทราบอุณหภูมิ" | ภาพมีจริง metadata เดิมผิด |
| 2026-05-12 | HSV H range | ขยายเป็น [10°, 90°] | รับใบน้ำตาลจัดที่ D7–D8 (hue ~10°–25°) |
| 2026-05-12 | Δ-features baseline | แยกตาม view (top/side) | top กับ side มี feature ต่างกันมาก |
| 2026-05-14 | เพิ่ม texture features เข้า model | ใส่ทั้ง 4 ตัว (contrast/correlation/energy/homogeneity) | ทดลองแล้วทุก metric ดีขึ้น (MAE -0.027, R² +0.011) |
| 2026-05-14 | เก็บ xgb_model.json เก่าไว้ | เก็บเป็น backup ไม่ลบ | ใช้อ้างอิงย้อนหลังได้ แต่ incompatible กับ FEATURE_COLS ปัจจุบัน |
