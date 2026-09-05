import streamlit as st
import requests
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import base64
from datetime import datetime

st.set_page_config(page_title="MDC | Defect Intelligence", layout="wide", page_icon="🛠️", initial_sidebar_state="expanded")

API_URL = "http://127.0.0.1:8000"

if "history" not in st.session_state:
    st.session_state.history = []

st.markdown("""
<style>
@import url("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap");
html, body, [class*="css"] { font-family: "Inter", sans-serif; }
.main { background-color: #0a0d13; }

.hero {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    padding: 30px 36px; border-radius: 18px; border: 1px solid #263042;
    margin-bottom: 24px; display: flex; justify-content: space-between; align-items: center;
}
.hero h1 { font-size: 30px; font-weight: 800; color: #fff; margin: 0;
    background: linear-gradient(90deg, #60a5fa, #c084fc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.hero p { color: #94a3b8; font-size: 14px; margin-top: 4px; }

.card { background: #12161f; border: 1px solid #202634; border-radius: 16px;
    padding: 22px 24px; margin-bottom: 18px; }
.card h3 { color: #f1f5f9; font-size: 15px; font-weight: 700; margin: 0 0 14px 0;
    text-transform: uppercase; letter-spacing: 0.6px; }

.badge-ok { background: rgba(34,197,94,0.15); color: #4ade80; border: 1px solid rgba(34,197,94,0.35);
    padding: 5px 16px; border-radius: 999px; font-weight: 600; font-size: 13px; }
.badge-warn { background: rgba(234,179,8,0.15); color: #facc15; border: 1px solid rgba(234,179,8,0.35);
    padding: 5px 16px; border-radius: 999px; font-weight: 600; font-size: 13px; }
.badge-offline { background: rgba(239,68,68,0.15); color: #f87171; border: 1px solid rgba(239,68,68,0.35);
    padding: 5px 16px; border-radius: 999px; font-weight: 600; font-size: 13px; }

.stMetric { background-color: #12161f !important; border: 1px solid #202634;
    padding: 14px !important; border-radius: 12px !important; }

section[data-testid="stSidebar"] { background-color: #0d1017; border-right: 1px solid #202634; }

.stButton>button { background: linear-gradient(90deg, #3b82f6, #a855f7); color: white; border: none;
    border-radius: 10px; font-weight: 600; padding: 11px 0; transition: 0.2s; }
.stButton>button:hover { opacity: 0.88; color: white; transform: scale(1.01); }

.history-item { background: #161b25; border: 1px solid #262d3a; border-radius: 10px;
    padding: 10px 12px; margin-bottom: 8px; font-size: 13px; color: #cbd5e1; }

img { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

def confidence_gauge(value, title):
    color = "#4ade80" if value >= 0.75 else ("#facc15" if value >= 0.5 else "#f87171")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        title={"text": title, "font": {"size": 14, "color": "#94a3b8"}},
        number={"suffix": "%", "font": {"size": 26, "color": "#f1f5f9"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#475569"},
            "bar": {"color": color},
            "bgcolor": "#12161f",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 50], "color": "#1f2937"},
                {"range": [50, 75], "color": "#27303f"},
            ],
        }
    ))
    fig.update_layout(height=180, margin=dict(l=20, r=20, t=40, b=10),
                       paper_bgcolor="#12161f", font_color="#e5e7eb")
    return fig

with st.sidebar:
    st.markdown("### 🛠️ MDC Console")
    st.caption("Manufacturing Defect Classification")
    st.divider()
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        domains = health.get("loaded_domains", [])
        st.markdown('<span class="badge-ok">● API Online</span>', unsafe_allow_html=True)
        st.metric("Specialists Loaded", len(domains))
        with st.expander("Trained Domains", expanded=False):
            for d in domains:
                st.markdown(f"- `{d}`")
    except Exception:
        domains = []
        st.markdown('<span class="badge-offline">● API Offline</span>', unsafe_allow_html=True)
        st.caption("Start backend: `python -m uvicorn app:app --reload`")

    st.divider()
    st.caption("Model: MSA-Net | Domain Classifier (95.08% val acc)")

    st.divider()
    st.markdown("**Recent Predictions**")
    if st.session_state.history:
        for h in reversed(st.session_state.history[-6:]):
            st.markdown(f'<div class="history-item">🕒 {h["time"]}<br><b>{h["domain"]}</b> → {h["defect"]} ({h["conf"]:.0f}%)</div>', unsafe_allow_html=True)
        if st.button("Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()
    else:
        st.caption("No predictions yet this session.")

st.markdown("""
<div class="hero">
    <div>
        <h1>Defect Intelligence Dashboard</h1>
        <p>Upload a manufacturing image to route it through the domain classifier and specialist defect model.</p>
    </div>
</div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1.6], gap="large")

with col_left:
    st.markdown('<div class="card"><h3>📤 Upload Image</h3>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png", "bmp", "webp"], label_visibility="collapsed")
    if uploaded_file:
        st.image(uploaded_file, use_container_width=True)
        predict_btn = st.button("🔍  Run Prediction", use_container_width=True)
    else:
        predict_btn = False
        st.caption("JPG, PNG, BMP, or WEBP")
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

                st.session_state.history.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "domain": result["domain"],
                    "defect": result["defect"],
                    "conf": result["domain_confidence"] * 100,
                })

                g1, g2 = st.columns(2)
                with g1:
                    st.plotly_chart(confidence_gauge(result["domain_confidence"], "Domain Confidence"), use_container_width=True)
                with g2:
                    defect_conf = result["defect_confidence"] if result["defect_confidence"] else 0
                    st.plotly_chart(confidence_gauge(defect_conf, "Defect Confidence"), use_container_width=True)

                r1, r2 = st.columns(2)
                r1.markdown(f"**Predicted Domain**  \n### {result['domain']}")
                r2.markdown(f"**Predicted Defect**  \n### {result['defect']}")

                if result.get("low_confidence_warning"):
                    st.warning(f'⚠ {result.get("message", "Low confidence prediction")}')
                else:
                    st.success("✓ High-confidence prediction")

                tab1, tab2, tab3 = st.tabs(["🌐 Top Domains", "🔬 Defect Breakdown", "🔥 Grad-CAM"])

                with tab1:
                    df = pd.DataFrame(list(result["domain_probabilities"].items()), columns=["Domain", "Probability"])
                    df = df.sort_values("Probability", ascending=False).head(10).sort_values("Probability", ascending=True)
                    fig = px.bar(df, x="Probability", y="Domain", orientation="h", height=420,
                                 color="Probability", color_continuous_scale="Blues", text_auto=".1%")
                    fig.update_layout(paper_bgcolor="#12161f", plot_bgcolor="#12161f", font_color="#e5e7eb",
                                       margin=dict(l=10, r=10, t=10, b=10))
                    st.plotly_chart(fig, use_container_width=True)

                with tab2:
                    if result["defect_probabilities"]:
                        df2 = pd.DataFrame(list(result["defect_probabilities"].items()), columns=["Defect", "Probability"])
                        df2 = df2.sort_values("Probability", ascending=True)
                        fig2 = px.bar(df2, x="Probability", y="Defect", orientation="h", height=350,
                                      color="Probability", color_continuous_scale="Purples", text_auto=".1%")
                        fig2.update_layout(paper_bgcolor="#12161f", plot_bgcolor="#12161f", font_color="#e5e7eb",
                                            margin=dict(l=10, r=10, t=10, b=10))
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
