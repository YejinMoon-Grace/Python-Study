import streamlit as st
import numpy as np
import joblib

# ========================
# Load model & threshold
# ========================
model = joblib.load("xgb_stroke_model.pkl")
thr = joblib.load("decision_threshold.pkl")  # High Risk 기준

# ========================
# Page config
# ========================
st.set_page_config(
    page_title="Stroke Risk Prediction",
    layout="centered"
)

# ========================
# Global CSS
# ========================
st.markdown("""
<style>
/* 전체 배경 */
.stApp { background-color: #0e1117; }

/* header/footer 제거 */
header, footer { visibility: hidden; }

/* 전체 여백 정리 */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* ===== 빈 block 제거 ===== */
div[data-testid="stVerticalBlock"] > div:empty {
    display: none !important;
}

/* 버튼 아래 여백 제거 */
div[data-testid="stVerticalBlock"] > div:has(button):not(:has(*:not(button))) {
    margin-bottom: 0 !important;
}

/* 카드 */
.card {
    padding: 28px;
    border-radius: 16px;
    background-color: #161a23;
    box-shadow: 0 8px 28px rgba(0,0,0,0.45);
    margin-bottom: 0px;
}

/* 텍스트 */
h1, h2, h3, h4, p, label { color: #e6e6e6; }

/* Risk 컬러 */
.risk-high { color: #ff4b4b; font-weight: 700; }
.risk-medium { color: #f1c40f; font-weight: 700; }
.risk-low { color: #2ecc71; font-weight: 700; }

/* ===== Gauge ===== */
.gauge {
    position: relative;
    height: 12px;
    background: #2a2f3a;
    border-radius: 6px;
    margin-top: 14px;
}
.gauge-fill {
    height: 100%;
    background: linear-gradient(90deg, #2ecc71, #f1c40f, #ff4b4b);
    border-radius: 6px;
}
.marker {
    position: absolute;
    top: -18px;
    transform: translateX(-50%);
    font-size: 13px;
    font-weight: 700;
}
.marker-boundary { color: #f1c40f; }
.marker-danger { color: #ff4b4b; }
</style>
""", unsafe_allow_html=True)

# ========================
# Title
# ========================
st.title("🧠 Stroke Risk Prediction")

# ========================
# 안내 배너 (빈 박스 제거됨)
# ========================
st.markdown("""
<div class="card" style="text-align:center; font-size:18px;">
    수치를 입력해주세요.
</div>
""", unsafe_allow_html=True)

# ========================
# Input Card
# ========================
#st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 120, 60)
    bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
    avg_glucose = st.number_input("Avg Glucose Level", 50.0, 400.0, 100.0)

with col2:
    hypertension_txt = st.selectbox("Hypertension", ["No", "Yes"])
    heart_disease_txt = st.selectbox("Heart Disease", ["No", "Yes"])
    smoking = st.selectbox(
        "Smoking Status",
        ["never smoked", "smokes", "formerly smoked", "Unknown"]
    )

predict_clicked = st.button("Predict Stroke Risk")
#
# ========================
# Encoding
# ========================
hypertension = 1 if hypertension_txt == "Yes" else 0
heart_disease = 1 if heart_disease_txt == "Yes" else 0

smoking_status_0 = 1 if smoking == "never smoked" else 0
smoking_status_2 = 1 if smoking == "formerly smoked" else 0
smoking_status_3 = 1 if smoking == "Unknown" else 0

X_input = np.array([[
    hypertension,
    heart_disease,
    age,
    avg_glucose,
    bmi,
    smoking_status_0,
    smoking_status_2,
    smoking_status_3
]])

# ========================
# Prediction Result Card
# ========================
if predict_clicked:
    prob = model.predict_proba(X_input)[0, 1]
    p = prob * 100
    thr_pct = thr * 100

#    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Prediction Result")

    st.metric("Stroke Probability", f"{prob:.2%}")

    # ----- Gauge (HTML only) -----
    st.markdown(f"""
    <div class="gauge">
        <div class="gauge-fill" style="width:{p}%;"></div>
        <div class="marker marker-boundary" style="left:30%;">∇ 경계</div>
        <div class="marker marker-danger" style="left:{thr_pct}%;">∇ 위험</div>
    </div>
    """, unsafe_allow_html=True)

    # ----- Risk 판단 + 텍스트 (HTML 분리) -----
    if prob < 0.30:
        st.markdown('<p class="risk-low">🟢 Low Risk</p>', unsafe_allow_html=True)
        st.info("현재 생활습관을 유지하세요!")
    elif prob < thr:
        st.markdown('<p class="risk-medium">🟡 Medium Risk</p>', unsafe_allow_html=True)
        st.warning("추천 생활 습관입니다.")
    else:
        st.markdown('<p class="risk-high">🔴 High Risk</p>', unsafe_allow_html=True)
        st.error("병원 방문을 권장합니다.")

        # 병원 안내 (접기/펼치기)
        with st.expander("가까운 병원 안내 보기"):
            st.image(
                "hospital_example.png",
                caption="가까운 병원을 방문해 전문의 상담을 권장합니다.",
                use_container_width=True
            )
            st.link_button(
                "지도에서 병원 찾기",
                "https://map.kakao.com/"
            )

    st.caption(f"High Risk 결정 기준(threshold): {thr:.3f}")
    st.markdown('</div>', unsafe_allow_html=True)
