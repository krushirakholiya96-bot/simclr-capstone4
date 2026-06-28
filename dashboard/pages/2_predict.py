import streamlit as st
from PIL import Image

st.markdown('''<style>
    .stApp { background-color: #0A1628; color: #E8EDF5; }
    h1, h2, h3 { color: #BDD7EE !important; }
    .stButton > button {
        background: linear-gradient(135deg, #2E75B6, #1E3A5F);
        color: white; border: none; border-radius: 8px;
    }
</style>''', unsafe_allow_html=True)

st.title("Image Prediction")
st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=['jpg', 'jpeg', 'png']
    )
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

with col2:
    if uploaded_file:
        if st.button("Predict"):
            st.warning("API is running locally. Run FastAPI server to get predictions.")
            st.code("uvicorn api.main:app --host 0.0.0.0 --port 8000")

    st.markdown("### API Endpoints")
    st.markdown("""
    - `POST /predict` — Image prediction
    - `POST /agent/run` — Full agent pipeline
    - `POST /explain` — AI explanation
    - `GET /history` — Past predictions
    - `GET /health` — Server status
    """)
    
