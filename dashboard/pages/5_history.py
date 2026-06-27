import streamlit as st
import requests
import pandas as pd
import plotly.express as px

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

st.title("Prediction History")
st.markdown("### All Past Predictions")
st.markdown("---")

# Fetch history from API
try:
    response = requests.get(f"{API_URL}/history")
    if response.status_code == 200:
        history = response.json()
    else:
        history = []
except Exception:
    history = []
    st.warning("API offline — Start FastAPI server")

if history:
    df = pd.DataFrame(history)

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        classes = ['All'] + sorted(df['predicted_class'].unique().tolist())
        selected_class = st.selectbox("Filter by Class", classes)

    with col2:
        min_conf = st.slider("Minimum Confidence", 0, 100, 0)

    # Apply filters
    filtered_df = df.copy()
    if selected_class != 'All':
        filtered_df = filtered_df[
            filtered_df['predicted_class'] == selected_class
        ]
    filtered_df = filtered_df[
        filtered_df['confidence'] >= min_conf
    ]

    st.markdown(f"### Showing {len(filtered_df)} predictions")

    # Predictions table
    st.dataframe(
        filtered_df[['id', 'image_name', 'predicted_class',
                     'confidence', 'timestamp']],
        use_container_width=True
    )

    # Confidence trend chart
    st.markdown("### Confidence Trend")
    fig = px.line(
        filtered_df,
        x='timestamp',
        y='confidence',
        color='predicted_class',
        template='plotly_dark',
        title='Confidence Over Time'
    )
    fig.update_layout(
        paper_bgcolor='#0A1628',
        plot_bgcolor='#0D1F3C'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Export CSV
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="Export to CSV",
        data=csv,
        file_name="predictions_history.csv",
        mime="text/csv"
    )

    # Delete prediction
    st.markdown("---")
    st.markdown("### Delete Prediction")
    delete_id = st.number_input("Enter Prediction ID to delete",
                                min_value=1, step=1)
    if st.button("Delete"):
        del_response = requests.delete(
            f"{API_URL}/history/{int(delete_id)}"
        )
        if del_response.status_code == 200:
            st.success("Prediction deleted!")
            st.rerun()
        else:
            st.error("Delete failed!")
else:
    st.info("No predictions yet — go to Predict page and make some predictions!")