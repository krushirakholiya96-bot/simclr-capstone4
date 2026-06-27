import streamlit as st
import requests
import time
from PIL import Image

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

STEPS = [
    'Image Received & Validated',
    'Features Extracted via ResNet50 Encoder',
    'Top-5 Predictions Generated',
    'Similar Past Cases Retrieved from Database',
    'AI Explanation Generated (Groq Llama 3)',
    'Confidence Check & Warning Analysis',
    'Final Report Compiled',
    'Results Saved to Database'
]

st.title("Agent Pipeline")
st.markdown("### 8-Step Autonomous Analysis")
st.markdown("---")

uploaded_file = st.file_uploader(
    "Upload Image for Agent Analysis",
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=200)

    if st.button("Run Agent Pipeline"):
        # Show live progress
        progress_bar = st.progress(0)
        step_container = st.container()

        with step_container:
            for i, step in enumerate(STEPS):
                with st.spinner(f"Step {i+1}/8: {step}..."):
                    time.sleep(0.5)
                st.success(f"Step {i+1} Complete: {step}")
                progress_bar.progress((i + 1) / len(STEPS))

        # Call agent API
        with st.spinner("Generating final report..."):
            response = requests.post(
                f"{API_URL}/agent/run",
                files={'file': (
                    uploaded_file.name,
                    uploaded_file.getvalue()
                )}
            )

        if response.status_code == 200:
            report = response.json()
            st.balloons()

            st.markdown("---")
            st.markdown("### Final Report")

            # Report card
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #1E3A5F, #0D2137);
            border: 1px solid #2E75B6; border-radius: 12px;
            padding: 20px; color: #E8EDF5;'>
            <h4 style='color: #BDD7EE;'>Prediction Results</h4>
            <p><b>Class:</b> {report['predicted_class'].upper()}</p>
            <p><b>Confidence:</b> {report['confidence']:.2f}%</p>
            <p><b>Similar Cases Found:</b> {report['similar_cases']}</p>
            <h4 style='color: #BDD7EE;'>AI Explanation</h4>
            <p>{report['explanation']}</p>
            <h4 style='color: #BDD7EE;'>Summary</h4>
            <p>{report['summary']}</p>
            </div>
            """, unsafe_allow_html=True)

            if report.get('warning'):
                st.warning(report['warning'])

            # Download report
            import json
            report_json = json.dumps(report, indent=2)
            st.download_button(
                label="Download Report",
                data=report_json,
                file_name="agent_report.json",
                mime="application/json"
            )
        else:
            st.error("Agent pipeline failed — make sure API is running!")