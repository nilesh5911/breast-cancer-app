import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

from utils import load_model, save_model, load_ranges, save_ranges

# Configure page
st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    body {
        background: linear-gradient(to right, #f8f9fa, #eef2f7);
        font-family: 'Segoe UI', sans-serif;
    }
    .block-container {
        padding: 2rem 3rem;
        border-radius: 12px;
    }
    h1, h2, h3 {
        color: #1f2937;
        font-weight: 600;
        text-align: center;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 8px;
        padding: 10px 16px;
        font-size: 16px;
    }
    .stDownloadButton>button {
        background-color: #2ecc71;
        color: white;
        border-radius: 6px;
        padding: 8px 14px;
    }
    </style>
""", unsafe_allow_html=True)

# App title
st.markdown("<h1>🧬 Breast Cancer Prediction Web App</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>AI-powered diagnostic tool for early detection and insights</h3>", unsafe_allow_html=True)
st.divider()

# Load saved model
model = load_model()
ranges = load_ranges()

# Sidebar
page = st.sidebar.radio("📁 Navigation", ["📊 Make Prediction", "🔁 Retrain Model"])

# ---------------- Prediction Page ---------------- #
if page == "📊 Make Prediction":
    st.subheader("Enter Tumor Feature Values")

    col1, col2, col3 = st.columns(3)
    with col1:
        radius = st.number_input("🔘 Mean Radius", value=0.0)
        texture = st.number_input("🌀 Mean Texture", value=0.0)
    with col2:
        perimeter = st.number_input("📏 Mean Perimeter", value=0.0)
        area = st.number_input("🧱 Mean Area", value=0.0)
    with col3:
        smoothness = st.number_input("✨ Mean Smoothness", value=0.0)
        compactness = st.number_input("📦 Mean Compactness", value=0.0)

    input_data = pd.DataFrame(
        [[radius, texture, perimeter, area, smoothness, compactness]],
        columns=[
            'Mean Radius', 'Mean Texture', 'Mean Perimeter',
            'Mean Area', 'Mean Smoothness', 'Mean Compactness'
        ]
    )

    st.markdown("### 📉 Live Prediction Result")
    if st.button("🚀 Run Prediction"):
        try:
            pred = model.predict(input_data)[0]
            conf = model.predict_proba(input_data).max() if hasattr(model, 'predict_proba') else None

            label_map = {0: 'Benign', 1: 'Malignant'}
            label = label_map.get(pred, 'Unknown')
            expl = "🟢 Benign: Usually non-cancerous." if pred == 0 \
                   else "🔴 Malignant: Cancerous. Immediate attention required."

            st.success(f"🧪 **Prediction:** {label}")
            if conf is not None:
                st.info(f"🎯 Confidence: **{round(conf * 100, 2)}%**")
            st.markdown(f"📝 **Explanation:** {expl}")

            # Downloadable Report
            input_data['Prediction'] = label
            if conf is not None:
                input_data['Confidence (%)'] = round(conf * 100, 2)
            input_data['Timestamp'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            csv = input_data.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Prediction Report", data=csv, file_name="prediction_report.csv", mime="text/csv")

        except Exception as e:
            st.error(f"❌ Error: {e}")

# ---------------- Retrain Page ---------------- #
elif page == "🔁 Retrain Model":
    st.subheader("Upload New Dataset to Retrain the Model")
    uploaded = st.file_uploader("📂 Upload CSV file", type=['csv'])

    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.write("🔍 **Preview of Uploaded Data:**", df.head())

            if 'target' not in df.columns:
                st.error("❌ Your dataset must include a `target` column.")
            else:
                num_samples = len(df)
                test_size = st.slider("📏 Test set size (%)", 10, 50, 20, 5) / 100.0
                rand_state = st.number_input("🔁 Random state", min_value=0, value=42, step=1)

                max_k = max(1, num_samples - 1)
                default_k = min(int(np.sqrt(num_samples)), max_k)
                k = st.number_input("👥 Neighbors (k)", 1, max_k, default_k, 1)

                X = df.drop(columns='target')
                y = df['target']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rand_state)

                new_model = KNeighborsClassifier(n_neighbors=k)
                new_model.fit(X_train, y_train)

                save_model(new_model)
                save_ranges(df)
                st.success(f"✅ Model retrained and saved (k={k}, test={int(test_size*100)}%)")

                y_pred = new_model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                rpt = classification_report(y_test, y_pred, labels=[0, 1], zero_division=0)
                cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

                st.markdown(f"### 📈 Accuracy: **{round(acc * 100, 2)}%**")
                st.text("📋 Classification Report:\n" + rpt)

                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_title('Confusion Matrix')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Error during training: {e}")
