import streamlit as st
import requests
import plotly.graph_objects as go
from PIL import Image
import io

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
            with st.spinner("Predicting..."):
                # Call predict API
                files = {'file': uploaded_file.getvalue()}
                response = requests.post(
                    f"{API_URL}/predict",
                    files={'file': (
                        uploaded_file.name,
                        uploaded_file.getvalue()
                    )}
                )

                if response.status_code == 200:
                    result = response.json()

                    # Store in session
                    st.session_state['prediction'] = result

        if 'prediction' in st.session_state:
            result = st.session_state['prediction']

            st.markdown(f"### Predicted: **{result['predicted_class'].upper()}**")

            # Confidence gauge
            confidence = result['confidence']
            fig = go.Figure(go.Indicator(
                mode='gauge+number',
                value=confidence,
                title={'text': 'Confidence %',
                       'font': {'color': '#BDD7EE', 'size': 16}},
                gauge={
                    'axis': {'range': [0, 100],
                             'tickcolor': '#BDD7EE'},
                    'bar': {'color': '#2E75B6'},
                    'bgcolor': '#0D1F3C',
                    'steps': [
                        {'range': [0, 50], 'color': '#3D0000'},
                        {'range': [50, 75], 'color': '#3D3000'},
                        {'range': [75, 100], 'color': '#003D00'}
                    ],
                    'threshold': {
                        'value': confidence,
                        'line': {'color': '#00D4FF', 'width': 4}
                    }
                }
            ))
            fig.update_layout(
                paper_bgcolor='#0A1628',
                font_color='#BDD7EE',
                height=250
            )
            st.plotly_chart(fig, use_container_width=True)

            # Top-5 bar chart
            st.markdown("### Top-5 Predictions")
            top5 = result['top5']
            classes = [t['class'] for t in top5]
            confidences = [t['confidence'] for t in top5]

            import plotly.express as px
            fig2 = px.bar(
                x=confidences,
                y=classes,
                orientation='h',
                template='plotly_dark',
                color=confidences,
                color_continuous_scale='Blues'
            )
            fig2.update_layout(
                paper_bgcolor='#0A1628',
                plot_bgcolor='#0D1F3C',
                height=250,
                showlegend=False
            )
            st.plotly_chart(fig2, use_container_width=True)

            # Get AI explanation
            if st.button("Get AI Explanation"):
                with st.spinner("Generating explanation..."):
                    exp_response = requests.post(
                        f"{API_URL}/explain",
                        json={
                            'predicted_class': result['predicted_class'],
                            'confidence': result['confidence'],
                            'top5': result['top5']
                        }
                    )
                    if exp_response.status_code == 200:
                        exp = exp_response.json()
                        st.markdown("### AI Explanation")
                        st.markdown(f"""
                        <div style='background: #1E3A5F;
                        border: 1px solid #2E75B6;
                        border-radius: 8px; padding: 15px;
                        color: #E8EDF5;'>
                        {exp['explanation']}
                        </div>
                        """, unsafe_allow_html=True)

                        if exp.get('warning'):
                            st.warning(exp['warning'])

            # Navigation buttons
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Run Full Agent"):
                    st.switch_page("pages/3_agent.py")
            with col_b:
                if st.button("View History"):
                    st.switch_page("pages/5_history.py")