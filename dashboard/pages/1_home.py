import streamlit as st
import requests

st.markdown('''<style>
    .stApp { background-color: #0A1628; color: #E8EDF5; }
    .stSidebar { background-color: #0D1F3C; border-right: 1px solid #1E3A5F; }
    .stButton > button {
        background: linear-gradient(135deg, #2E75B6, #1E3A5F);
        color: white; border: none; border-radius: 8px;
    }
    h1, h2, h3 { color: #BDD7EE !important; }
</style>''', unsafe_allow_html=True)

API_URL = "http://localhost:8000"

st.title("SimCLR Contrastive Learner")
st.markdown("### End-to-End Self-Supervised Learning System")
st.markdown("---")

# Live stats from API
try:
    response = requests.get(f"{API_URL}/health")
    if response.status_code == 200:
        st.success("API Status: Online")
    else:
        st.warning("API Status: Offline")
except Exception:
    st.warning("API Status: Offline — Start FastAPI server")

st.markdown("---")

# Metric cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Model", value="ResNet50")

with col2:
    st.metric(label="Accuracy", value="83%+")

with col3:
    st.metric(label="Dataset", value="CIFAR-10")

with col4:
    st.metric(label="Training Epochs", value="200")

st.markdown("---")

# Tech stack
st.markdown("### Tech Stack")
cols = st.columns(7)
badges = ["PyTorch", "ResNet50", "Groq AI", 
          "FastAPI", "Streamlit", "Docker", "GitHub Actions"]

for col, badge in zip(cols, badges):
    col.markdown(f"""
    <div style='background: #1E3A5F; border: 1px solid #2E75B6;
    border-radius: 8px; padding: 8px; text-align: center;
    color: #BDD7EE; font-size: 12px;'>{badge}</div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Project description
st.markdown("### About This Project")
st.info("""
Built end-to-end SimCLR Contrastive Self-Supervised Learning system
on CIFAR-10 using ResNet50 achieving 83%+ accuracy.
Integrated Groq Llama 3 for Generative AI explainability and
an 8-step Agentic AI pipeline for autonomous image analysis.
Deployed via FastAPI + Streamlit with Docker + GitHub Actions CI/CD.
""")

# Navigation button
if st.button("Start Predicting"):
    st.info("Please select 'predict' from the sidebar!")
