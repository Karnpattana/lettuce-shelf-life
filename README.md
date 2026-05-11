# Lettuce Quality & Shelf-Life Prediction

วิเคราะห์คุณภาพและทำนายอายุการเก็บรักษาของผัก **Cos** และ **Green Oak** จากภาพถ่าย  
ใช้ Feature Engineering + Classical ML อ้างอิงมาตรฐาน Kader et al. (1973) OVQ scale

## เป้าหมาย

ระบบรับภาพถ่ายผักแล้วส่งออก:
- **Predicted Day** (D0-D7) — ระยะการเก็บรักษา
- **OVQ Score** (1-9) — Overall Visual Quality ตาม Kader scale
- **Grade** (A/B/C) — สดเยี่ยม / เริ่มเสื่อม / ควรรีบบริโภค
- **Shelf life** — วันที่เหลือก่อนถึง marketability / usability limit

## Dataset

- ผัก **Cos** และ **Green Oak** (GOK) อย่างละ 30 ต้น
- ถ่ายภาพ D0–D7 ทุกวัน (บางกลุ่ม D0–D5)
- 4 ภาพ/ต้น/รอบ: side A/B × view top/side
- ชื่อไฟล์: `{COS|GOK}{NN}_{A|B}_D{n}_{M|E}_{top|side}.jpg`

## มาตรฐานคุณภาพ: Kader OVQ Scale

| OVQ | Grade | ความหมาย |
|-----|-------|-----------|
| 7-9 | A | Fresh / Excellent |
| 5-6 | B | Fair / Marketability limit |
| 3-4 | C | Poor / Below marketability |
| 1-2 | — | Unusable |

อ้างอิง: Kader, A. A., Lipton, W. J., & Morris, L. L. (1973). *HortScience*, 8(5), 408-409.

## วิธีรัน

```bash
# 1. ติดตั้ง dependencies
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt

# 2. วาง raw images ใน data/raw/

# 3. Segment
python -m src.segment

# 4. Extract features
python -m src.extract_all

# 5. Train model
python src/train.py --variety cos
python src/train.py --variety gok

# 6. Inference
python predict.py path/to/image.jpg --variety cos --temperature 30 --session E --view top

# 7. Demo UI (optional)
streamlit run app.py
```

## โครงสร้างโฟลเดอร์

```
lettuce_shelf_life/
├── data/raw/                 # ภาพต้นฉบับ (read-only)
├── data/segmented/           # masks + cropped
├── data/metadata.csv         # อุณหภูมิแต่ละรอบถ่าย
├── notebooks/                # EDA + modeling notebooks
├── src/                      # source code
│   ├── features/             # color, texture, shape extractors
│   ├── preprocess.py
│   ├── segment.py
│   ├── extract_all.py
│   ├── train.py
│   └── grade_mapping.py
├── models/                   # trained models (.pkl)
├── config/grade_thresholds.yaml
├── results/                  # metrics, plots
├── predict.py                # inference script
└── app.py                    # Streamlit demo (Phase 7)
```

## Tech Stack

Python 3.10+ · OpenCV · scikit-image · scikit-learn · XGBoost · pandas · Streamlit
