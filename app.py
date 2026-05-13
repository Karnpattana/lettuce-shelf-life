"""Streamlit demo — Lettuce Shelf Life Prediction"""
from pathlib import Path
import tempfile

import streamlit as st
from PIL import Image

from src.inference import predict
from src.grade import MARKETABILITY_DAY, UNUSABLE_DAY

MODEL_PATH = Path("models/xgb_model.json")

GRADE_CONFIG = {
    "A": {"label": "Grade A — สดมาก",      "color": "#2ecc71", "days": "D0–D1", "emoji": "🟢", "ovq": "8–9"},
    "B": {"label": "Grade B — สด",          "color": "#f1c40f", "days": "D2–D3", "emoji": "🟡", "ovq": "6–7"},
    "C": {"label": "Grade C — เริ่มเสื่อม", "color": "#e67e22", "days": "D4–D5", "emoji": "🟠", "ovq": "4–5"},
    "D": {"label": "Grade D — เสื่อมมาก",  "color": "#e74c3c", "days": "D6–D8", "emoji": "🔴", "ovq": "1–3"},
}

STATUS_CONFIG = {
    "fresh":   {"emoji": "🟢", "label": "Fresh",   "color": "#2ecc71"},
    "good":    {"emoji": "🟢", "label": "Good",    "color": "#27ae60"},
    "warning": {"emoji": "🟡", "label": "Warning", "color": "#e67e22"},
    "expired": {"emoji": "🔴", "label": "Expired", "color": "#e74c3c"},
}

FEATURE_THAI = {
    "a_mean":     "a* เฉลี่ย (แดง/เขียว)",
    "pct_yellow": "สัดส่วนเหลือง",
    "pct_green":  "สัดส่วนเขียว",
    "pct_brown":  "สัดส่วนน้ำตาล",
    "area_ratio": "สัดส่วนพื้นที่ใบ",
}

st.set_page_config(
    page_title="Lettuce Shelf Life",
    page_icon="🥬",
    layout="centered",
)

st.title("🥬 Lettuce Shelf Life Prediction")
st.caption("อัปโหลดภาพผักกาดหอม → ระบบประเมินอายุการเก็บรักษา")

# --- Sidebar ---
with st.sidebar:
    st.header("ตั้งค่า")
    variety = st.radio("พันธุ์ผัก", ["COS", "GOK"], horizontal=True)
    st.markdown("---")
    st.markdown(
        "**Grade & Kader OVQ**\n"
        "| Grade | วัน | OVQ | ความหมาย |\n"
        "|-------|-----|-----|----------|\n"
        "| 🟢 A | D0–D1 | 8–9 | สดมาก |\n"
        "| 🟡 B | D2–D3 | 6–7 | สด |\n"
        "| 🟠 C | D4–D5 | 4–5 | เริ่มเสื่อม |\n"
        "| 🔴 D | D6–D8 | 1–3 | เสื่อมมาก |"
    )
    st.markdown("---")
    st.caption(
        f"Marketability limit: day {MARKETABILITY_DAY} (OVQ ≤ 5)\n"
        f"Unusable threshold: day {UNUSABLE_DAY} (OVQ ≤ 3)\n"
        "Kader et al. (1973)"
    )
    st.markdown("---")
    st.caption("Phase 8 Demo — Lettuce Shelf Life Prediction\nThesis 2026")

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

sl   = result["shelf_life"]
grade = result["grade"]
gcfg  = GRADE_CONFIG[grade]
scfg  = STATUS_CONFIG[sl["status"]]

st.markdown("---")

# --- 1. Shelf Life (เด่นสุด) ---
st.subheader("🥬 Shelf Life Remaining")
col_m, col_u = st.columns(2)
with col_m:
    st.metric(
        label="Marketable",
        value=f"{sl['days_to_marketability_limit']} วัน",
        help=f"ขายได้ตามมาตรฐานตลาด (Kader OVQ ≥ 5) — ถึง day {MARKETABILITY_DAY}",
    )
    st.caption("ขายได้ตามมาตรฐานตลาด")
with col_u:
    st.metric(
        label="Usable",
        value=f"{sl['days_to_unusable']} วัน",
        help=f"บริโภคได้ปลอดภัย (Kader OVQ ≥ 3) — ถึง day {UNUSABLE_DAY}",
    )
    st.caption("บริโภคได้ปลอดภัย")

st.markdown(
    f"""
    <div style="
        background:{scfg['color']}22;
        border-left:5px solid {scfg['color']};
        border-radius:6px;
        padding:10px 16px;
        margin-top:8px;
    ">
        <b style="color:{scfg['color']}; font-size:1.1rem">
            {scfg['emoji']} Status: {scfg['label']}
        </b>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- 2. Quality Tier ---
st.markdown("---")
st.subheader("📊 Current Quality Tier")
st.markdown(
    f"""
    <div style="
        background:{gcfg['color']}22;
        border-left:5px solid {gcfg['color']};
        border-radius:6px;
        padding:10px 16px;
    ">
        <b style="color:{gcfg['color']}; font-size:1.05rem">
            {gcfg['emoji']} {gcfg['label']}
        </b><br>
        <span style="color:#555">Kader OVQ: {gcfg['ovq']} &nbsp;|&nbsp; {gcfg['days']}</span><br>
        <span style="color:#888; font-size:0.85rem">
            Based on Kader et al. (1973) postharvest visual quality scale
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- 3. Model Output ---
st.markdown("---")
st.subheader("🔬 Model Output")
st.markdown(
    f"Predicted day of degradation: **{result['predicted_day']:.1f}** / 8"
)

if result["low_confidence"]:
    st.warning(
        "⚠️ ความมั่นใจต่ำ — พื้นที่ใบที่ตรวจพบน้อยกว่า 5% ของภาพ "
        "ลองถ่ายภาพให้ผักอยู่ตรงกลางและเต็มเฟรมมากขึ้น"
    )
if result["gok_extrapolation"]:
    st.warning(
        "⚠️ Extrapolation — ชุด training ของ GOK มีข้อมูลถึง D7 เท่านั้น "
        f"({result['predicted_day']:.1f} วัน เกินช่วงที่โมเดลเคยเห็น) "
        "Grade D ถูกต้อง แต่ตัวเลขวันอาจคลาดเคลื่อน"
    )

# --- 4. Feature Breakdown ---
st.markdown("---")
with st.expander("🧪 Feature Breakdown — proves it's ML, not just day→grade"):
    feats = result["features"]
    cols = st.columns(len(FEATURE_THAI))
    for col, (key, label) in zip(cols, FEATURE_THAI.items()):
        val = feats.get(key, 0.0)
        display = f"{val*100:.1f}%" if key.startswith("pct") else f"{val:.3f}"
        col.metric(label=label, value=display)
    st.caption(
        f"area_ratio = {result['area_ratio']:.4f}  |  "
        f"low_confidence = {result['low_confidence']}  |  "
        f"gok_extrapolation = {result['gok_extrapolation']}"
    )
