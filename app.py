"""Streamlit demo — Lettuce Shelf Life Prediction"""
import io
from pathlib import Path
import tempfile

import streamlit as st
from PIL import Image

from src.inference import predict, DEFAULT_MODEL_PATH
from src.grade import MARKETABILITY_DAY, UNUSABLE_DAY

MODEL_PATH = DEFAULT_MODEL_PATH

GRADE_CONFIG = {
    "A": {"label": "Grade A — สด",              "color": "#2ecc71", "emoji": "🟢", "ovq": "8–9", "advice": "ผักสด เหมาะสำหรับจำหน่ายทันที"},
    "B": {"label": "Grade B — เริ่มเสื่อม",     "color": "#f1c40f", "emoji": "🟡", "ovq": "6–7", "advice": "เริ่มเสื่อมคุณภาพ ควรจำหน่ายภายใน 1–2 วัน"},
    "C": {"label": "Grade C — ไม่ควรบริโภค",   "color": "#e67e22", "emoji": "🟠", "ovq": "4–5", "advice": "ไม่ควรบริโภค ไม่เหมาะสำหรับจำหน่าย"},
    "D": {"label": "Grade D — เน่า",            "color": "#e74c3c", "emoji": "🔴", "ovq": "1–3", "advice": "เน่าเสีย ไม่สามารถบริโภคได้"},
}

STATUS_CONFIG = {
    "fresh":   {"emoji": "🟢", "label": "สดมาก",      "color": "#2ecc71"},
    "good":    {"emoji": "🟢", "label": "ยังดี",       "color": "#27ae60"},
    "warning": {"emoji": "🟡", "label": "ใกล้หมดอายุ", "color": "#e67e22"},
    "expired": {"emoji": "🔴", "label": "หมดอายุ",     "color": "#e74c3c"},
}

VARIETY_DISPLAY = {
    "COS": "Cos (Romaine)",
    "GOK": "Green Oak",
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

    variety_option = st.radio(
        "พันธุ์ผัก",
        ["Auto-detect", "Cos (Romaine)", "Green Oak"],
        index=0,
    )
    if variety_option == "Auto-detect":
        variety_input = None
    elif variety_option == "Cos (Romaine)":
        variety_input = "COS"
    else:
        variety_input = "GOK"

    st.markdown("---")
    st.markdown(
        "**Grade & Kader OVQ**\n"
        "| Grade | OVQ | ความหมาย |\n"
        "|-------|-----|----------|\n"
        "| 🟢 A | 8–9 | สด |\n"
        "| 🟡 B | 6–7 | เริ่มเสื่อม |\n"
        "| 🟠 C | 4–5 | ไม่ควรบริโภค |\n"
        "| 🔴 D | 1–3 | เน่า |"
    )
    st.markdown("---")
    st.caption(
        f"Marketability limit: day {MARKETABILITY_DAY} (OVQ ≤ 5)\n"
        f"Unusable threshold: day {UNUSABLE_DAY} (OVQ ≤ 3)\n"
        "Kader et al. (1973)"
    )

# --- Upload ---
uploaded = st.file_uploader(
    "อัปโหลดภาพ (.jpg / .png)",
    type=["jpg", "jpeg", "png"],
    help="ภาพผักกาดหอม 1 ต้น — พื้นหลังควรหลีกเลี่ยงสีเขียวหรือเหลือง",
)

if uploaded is None:
    st.info("อัปโหลดภาพเพื่อเริ่มการประเมิน")
    st.stop()

# --- Load image ---
img_pil = Image.open(io.BytesIO(uploaded.read()))

# --- Predict ---
with st.spinner("กำลังวิเคราะห์ภาพ..."):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        img_pil.convert("RGB").save(tmp.name, format="JPEG")
        tmp_path = Path(tmp.name)
    try:
        result = predict(tmp_path, variety=variety_input, model_path=MODEL_PATH)
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาด: {e}")
        st.stop()
    finally:
        tmp_path.unlink(missing_ok=True)

sl    = result["shelf_life"]
grade = result["grade"]
gcfg  = GRADE_CONFIG[grade]
scfg  = STATUS_CONFIG[sl["status"]]
variety_name = VARIETY_DISPLAY.get(result["variety"], result["variety"])

# --- ภาพ + ผลหลัก (2 คอลัมน์) ---
col_img, col_result = st.columns([2, 1])

with col_img:
    st.image(img_pil, caption=uploaded.name, width="stretch")

with col_result:
    # ชนิดผัก
    if result["variety_source"] == "auto":
        conf_pct = result["variety_confidence"] * 100
        variety_sub = f"Auto-detect ({conf_pct:.0f}% confidence)"
    else:
        variety_sub = "ระบุเอง"
    st.markdown(f"**ชนิดผัก:** {variety_name}")
    st.caption(variety_sub)

    # เกรด
    st.markdown(
        f"**เกรด:** "
        f"<span style='color:{gcfg['color']};font-weight:bold'>"
        f"{gcfg['emoji']} {gcfg['label']}</span>",
        unsafe_allow_html=True,
    )
    st.caption(f"Kader OVQ: {gcfg['ovq']}")

    # เก็บได้อีก
    mkt = sl["days_to_marketability_limit"]
    usable = sl["days_to_unusable"]
    st.markdown("**เก็บได้อีก:**")
    st.markdown(
        f"<div style='font-size:0.95rem; line-height:1.8'>"
        f"ขายได้&nbsp;&nbsp;&nbsp; <b>{mkt} วัน</b><br>"
        f"บริโภคได้ <b>{usable} วัน</b>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ข้อแนะนำ
    st.markdown(f"**ข้อแนะนำ:** {gcfg['advice']}")

# --- Warning: variety confidence ต่ำ ---
if result["variety_source"] == "auto" and result["variety_confidence"] < 0.75:
    st.warning(
        f"⚠️ ความมั่นใจในการตรวจจับพันธุ์ต่ำ ({result['variety_confidence']*100:.0f}%) — "
        "ลองเลือกพันธุ์เองใน Sidebar เพื่อผลที่แม่นยำขึ้น"
    )

# --- คุณภาพ ---
st.markdown("---")
st.subheader("📊 คุณภาพผัก")
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
        <span style="color:#555">Kader OVQ: {gcfg['ovq']}</span><br>
        <span style="color:#888; font-size:0.85rem">
            อ้างอิง Kader et al. (1973) postharvest visual quality scale
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Model Output ---
st.markdown("---")
st.subheader("🔬 ผลจากโมเดล")
st.markdown(f"วันที่ประเมิน (predicted day): **{result['predicted_day']:.1f}** / 8")

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

# --- Feature Breakdown ---
st.markdown("---")
with st.expander("🧪 Feature Breakdown — proves it's ML, not just day→grade"):
    feats = result["features"]
    cols = st.columns(len(FEATURE_THAI))
    for col, (key, label) in zip(cols, FEATURE_THAI.items()):
        val = feats.get(key, 0.0)
        display = f"{val*100:.1f}%" if key.startswith("pct") else f"{val:.3f}"
        col.metric(label=label, value=display)
    st.caption(f"predicted_day = {result['predicted_day']}  |  area_ratio = {result['area_ratio']:.4f}")
