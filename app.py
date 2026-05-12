"""Streamlit demo — Lettuce Shelf Life Prediction"""
from pathlib import Path
import tempfile

import numpy as np
import streamlit as st
from PIL import Image

from src.inference import predict

MODEL_PATH = Path("models/xgb_model.json")

GRADE_CONFIG = {
    "A": {"label": "Grade A — สดมาก",      "color": "#2ecc71", "days": "D0–D1", "emoji": "🟢"},
    "B": {"label": "Grade B — สด",          "color": "#f1c40f", "days": "D2–D3", "emoji": "🟡"},
    "C": {"label": "Grade C — เริ่มเสื่อม", "color": "#e67e22", "days": "D4–D5", "emoji": "🟠"},
    "D": {"label": "Grade D — เสื่อมมาก",  "color": "#e74c3c", "days": "D6–D8", "emoji": "🔴"},
}

FEATURE_THAI = {
    "a_mean":       "ค่าเฉลี่ย a* (ความแดง/เขียว)",
    "pct_yellow":   "สัดส่วนใบเหลือง (%)",
    "pct_green":    "สัดส่วนใบเขียว (%)",
    "pct_brown":    "สัดส่วนใบน้ำตาล (%)",
    "area_ratio":   "สัดส่วนพื้นที่ใบ",
}

st.set_page_config(
    page_title="Lettuce Shelf Life",
    page_icon="🥬",
    layout="centered",
)

st.title("🥬 Lettuce Shelf Life Prediction")
st.caption("อัปโหลดภาพผักกาดหอม → ระบบประเมิน Grade ความสด (A–D)")

# --- Sidebar ---
with st.sidebar:
    st.header("ตั้งค่า")
    variety = st.radio("พันธุ์ผัก", ["COS", "GOK"], horizontal=True)
    st.markdown("---")
    st.markdown(
        "**Grade ความสด**\n"
        "| Grade | วัน | ความหมาย |\n"
        "|-------|-----|----------|\n"
        "| 🟢 A | D0–D1 | สดมาก |\n"
        "| 🟡 B | D2–D3 | สด |\n"
        "| 🟠 C | D4–D5 | เริ่มเสื่อม |\n"
        "| 🔴 D | D6–D8 | เสื่อมมาก |"
    )
    st.markdown("---")
    st.caption("Phase 7 Demo — Lettuce Shelf Life Prediction\nThesis 2026")

# --- Upload ---
uploaded = st.file_uploader(
    "อัปโหลดภาพ (.jpg / .png)",
    type=["jpg", "jpeg", "png"],
    help="ภาพผักกาดหอม 1 ต้น ถ่ายพื้นหลังสีเดียว",
)

if uploaded is None:
    st.info("อัปโหลดภาพเพื่อเริ่มการประเมิน")
    st.stop()

# --- Display image ---
img_pil = Image.open(uploaded)
col_img, col_meta = st.columns([2, 1])
with col_img:
    st.image(img_pil, caption=uploaded.name, use_container_width=True)
with col_meta:
    st.markdown(f"**พันธุ์:** {variety}")
    st.markdown(f"**ขนาดภาพ:** {img_pil.width} × {img_pil.height} px")

# --- Predict ---
with st.spinner("กำลังวิเคราะห์ภาพ..."):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        img_pil.convert("RGB").save(tmp.name, format="JPEG")
        tmp_path = Path(tmp.name)

    try:
        result = predict(tmp_path, variety, model_path=MODEL_PATH)
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาด: {e}")
        st.stop()
    finally:
        tmp_path.unlink(missing_ok=True)

# --- Result ---
st.markdown("---")
grade = result["grade"]
cfg = GRADE_CONFIG[grade]

st.markdown(
    f"""
    <div style="
        background:{cfg['color']}22;
        border-left: 6px solid {cfg['color']};
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 12px;
    ">
        <h2 style="margin:0; color:{cfg['color']}">
            {cfg['emoji']} {cfg['label']}
        </h2>
        <p style="margin:4px 0 0; font-size:1rem; color:#555">
            อายุโดยประมาณ <b>{result['predicted_day']:.1f} วัน</b>
            &nbsp;({cfg['days']})
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if result["low_confidence"]:
    st.warning(
        "⚠️ ความมั่นใจต่ำ — พื้นที่ใบที่ตรวจพบน้อยกว่า 5% ของภาพ "
        "ผลลัพธ์อาจไม่แม่นยำ ลองถ่ายภาพให้ผักอยู่ตรงกลางและเต็มเฟรมมากขึ้น"
    )

if result["gok_extrapolation"]:
    st.warning(
        "⚠️ Extrapolation — ชุด training ของ GOK มีข้อมูลถึง D7 เท่านั้น (ไม่มี D8) "
        f"ค่าที่ทำนายได้ ({result['predicted_day']:.1f} วัน) เกินช่วงที่โมเดลเคยเห็น "
        "ผลลัพธ์ Grade D ยังถูกต้อง แต่ตัวเลขวันอาจคลาดเคลื่อน"
    )

# --- Feature breakdown ---
with st.expander("ดู Feature ที่ใช้ในการประเมิน"):
    feats = result["features"]
    cols = st.columns(len(FEATURE_THAI))
    for col, (key, label) in zip(cols, FEATURE_THAI.items()):
        val = feats.get(key, 0.0)
        display = f"{val*100:.1f}%" if key.startswith("pct") else f"{val:.3f}"
        col.metric(label=label, value=display)

    st.caption(
        f"area_ratio = {result['area_ratio']:.4f}  |  "
        f"low_confidence = {result['low_confidence']}"
    )
