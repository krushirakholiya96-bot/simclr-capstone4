import streamlit as st

st.set_page_config(
    page_title='SimCLR Dashboard',
    page_icon='🧠',
    layout='wide'
)

st.markdown('''<style>
    .stApp { background-color: #0A1628; color: #E8EDF5; }
    .stSidebar { background-color: #0D1F3C; border-right: 1px solid #1E3A5F; }
    .metric-card {
        background: linear-gradient(135deg, #1E3A5F, #0D2137);
        border: 1px solid #2E75B6;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 0 20px rgba(46,117,182,0.3);
    }
    .stButton > button {
        background: linear-gradient(135deg, #2E75B6, #1E3A5F);
        color: white;
        border: none;
        border-radius: 8px;
        box-shadow: 0 4px 15px rgba(46,117,182,0.4);
    }
    .stButton > button:hover {
        box-shadow: 0 6px 20px rgba(46,117,182,0.7);
    }
    h1, h2, h3 { color: #BDD7EE !important; }
    .stMetric label { color: #7EB3D4 !important; }
</style>''', unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("SimCLR Navigation")
st.sidebar.markdown("---")

pages = {
    "Home": "pages/1_home.py",
    "Predict": "pages/2_predict.py",
    "Agent Pipeline": "pages/3_agent.py",
    "Visualize": "pages/4_visualize.py",
    "History": "pages/5_history.py"
}

# Home page content
st.title("SimCLR Contrastive Learner")
st.markdown("### End-to-End Self-Supervised Learning System")

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
    st.metric(label="Epochs", value="200")

st.markdown("---")

# Tech stack badges
st.markdown("### Tech Stack")
st.markdown("""
**PyTorch** | **ResNet50** | **Groq Llama 3** | 
**FastAPI** | **Streamlit** | **Docker** | **GitHub Actions**
""")

st.markdown("---")
st.markdown("### Project Description")
st.info("""
Built end-to-end SimCLR Contrastive Self-Supervised Learning system 
on CIFAR-10 using ResNet50 achieving 83%+ accuracy. 
Integrated Groq Llama 3 for Generative AI explainability and 
an 8-step Agentic AI pipeline for autonomous image analysis.
""")