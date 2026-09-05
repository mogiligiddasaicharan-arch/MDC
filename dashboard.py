import streamlit as st
import requests
import plotly.express as px
import pandas as pd
import base64

st.set_page_config(page_title="MDC | Defect Intelligence", layout="wide", page_icon="🛠️", initial_sidebar_state="expanded")

API_URL = "http://127.0.0.1:8000"

st.markdown("""
<style>
@import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap");

html, body, [class*="css"] { font-family: "Inter", sans-serif; }

.main { background-color: #0b0e14; }

.hero {
    background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
    padding: 32px 36px;
    border-radius: 16px;
    border: 1px solid #2d3340;
    margin-bottom: 28px;
}
.hero h1 {
    font-size: 32px;
    font-weight: 800;
    color: #ffffff;
    margin: 0;
    background: linear-gradient(90deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero p {
    color: #9ca3af;
    font-size: 15px;
    margin-top: 6px;
}

.card {
    background: #151922;
    border: 1px solid #262b36;
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 16px;
}
.card h3 {
    color: #e5e7eb;
    font-size: 16px;
    font-weight: 700;
    margin-top: 0;
    margin-bottom: 14px;
    letter-spacing: 0.3px;
}

.badge-ok {
    display: inline-block;
    background: rgba(34,197,94,0.15);
    color: #4ade80;
    border: 1px solid rgba(34,197,94,0.35);
    padding: 4px 14px;
    border-radius: 999px;
    font-weight: 600;
    font-size: 13px;
}
.badge-warn {
    display: inline-block;
    background: rgba(234,179,8,0.15);
    color: #facc15;
    border: 1px solid rgba(234,179,8,0.35);
    padding: 4px 14px;
    border-radius: 999px;
    font-weight: 600;
    font-size: 13px;
}
.badge-offline {
    display: inline-block;
    background: rgba(239,68,68,0.15);
    color: #f87171;
    border: 1px solid rgba(239,68,68,0.35);
    padding: 4px 14px;
    border-radius: 999px;
    font-weight: 600;
    font-size: 13px;
}

.stMetric {
    background-color: #151922 !important;
    border: 1px solid #262b36;
    padding: 16px !important;
    border-radius: 12px !important;
}

section[data-testid="stSidebar"] {
    background-color: #0f1218;
    border-right: 1px solid #262b36;
}

.stButton>button {
    background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    padding: 10px 0;
}
.stButton>button:hover {
    opacity: 0.9;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("### 🛠️ MDC Console")
    st.caption("Manufacturing Defect Classification")
    st.divider()

    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        domains = health.get("loaded_domains", [])
        st.markdown('<span class="badge-ok">● API Online</span>', unsafe_allow_html=True)
        st.metric("Specialists Loaded", len(domains))
        st.markdown("**Trained Domains**")
        for d in domains:
            st.markdown(f"- `{d}`")
    except Exception:
        domains = []
        st.markdown('<span class="badge-offline">● API Offline</span>', unsafe_allow_html=True)
        st.caption("Start backend: `python -m uvicorn app:app --reload`")

    st.divider()
    st.caption("Model: MSA-Net | Domain Classifier (95.08% val acc)")

# ---------- HERO ----------
st.markdown("""
<div class="hero">
    <h1>Defect Intelligence Dashboard</h1>
    <p>Upload a manufacturing image to route it through the domain classifier and specialist defect model.</p>
</div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1.5], gap="large")

with col_left:
    st.markdown('<div class="card"><h3>📤 Upload Image</h3>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png", "bmp", "webp"], label_visibility="collapsed")
    if uploaded_file:
        st.image(uploaded_file, use_container_width=True)
        predict_btn = st.button("🔍  Run Prediction", use_container_width=True)
    else:
        predict_btn = False
        st.caption("JPG, PNG, BMP, or WEBP — up to 200MB")
    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="card"><h3>📊 Prediction Results</h3>', unsafe_allow_html=True)

    if uploaded_file and predict_btn:
        with st.spinner("Running inference..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            try:
                resp = requests.post(f"{API_URL}/predict", files=files, timeout=30)
                resp.raise_for_status()
                result = resp.json()

                m1, m2, m3 = st.columns(3)
                m1.metric("Domain", result["domain"], f'{result["domain_confidence"]*100:.1f}%')
                m2.metric("Defect", result["defect"], f'{result["defect_confidence"]*100:.1f}%' if result["defect_confidence"] else "N/A")

                if result.get("low_confidence_warning"):
                    m3.markdown('<span class="badge-warn">⚠ Low Confidence</span>', unsafe_allow_html=True)
                    st.warning(result.get("message", "Low confidence prediction"))
                else:
                    m3.markdown('<span class="badge-ok">✓ Confident</span>', unsafe_allow_html=True)
                    st.success("High-confidence prediction")

                tab1, tab2, tab3 = st.tabs(["🌐 Domain Probabilities", "🔬 Defect Probabilities", "🔥 Grad-CAM"])

                with tab1:
                    df = pd.DataFrame(list(result["domain_probabilities"].items()), columns=["Domain", "Probability"])
                    df = df.sort_values("Probability", ascending=True)
                    fig = px.bar(df, x="Probability", y="Domain", orientation="h", height=700,
                                 color="Probability", color_continuous_scale="Blues")
                    fig.update_layout(paper_bgcolor="#151922", plot_bgcolor="#151922", font_color="#e5e7eb")
                    st.plotly_chart(fig, use_container_width=True)

                with tab2:
                    if result["defect_probabilities"]:
                        df2 = pd.DataFrame(list(result["defect_probabilities"].items()), columns=["Defect", "Probability"])
                        df2 = df2.sort_values("Probability", ascending=True)
                        fig2 = px.bar(df2, x="Probability", y="Defect", orientation="h", height=400,
                                      color="Probability", color_continuous_scale="Purples")
                        fig2.update_layout(paper_bgcolor="#151922", plot_bgcolor="#151922", font_color="#e5e7eb")
                        st.plotly_chart(fig2, use_container_width=True)
                    else:
                        st.info("No specialist model loaded for this domain yet.")

                with tab3:
                    if result.get("gradcam_base64"):
                        st.image(base64.b64decode(result["gradcam_base64"]), caption="Grad-CAM Heatmap", use_container_width=True)
                    else:
                        st.info("Grad-CAM not available for this prediction.")

            except requests.exceptions.ConnectionError:
                st.error("Cannot reach the API. Make sure `uvicorn app:app --reload` is running on port 8000.")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
    else:
        st.info("Upload an image on the left and click **Run Prediction** to see results here.")

    st.markdown("</div>", unsafe_allow_html=True)
