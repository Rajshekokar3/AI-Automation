import streamlit as st
import pandas as pd
import joblib
import os
from Predictive_automation import PredictiveAutomation

# Initialize the Predictive Automation class
predictor = PredictiveAutomation()

st.title("🔍 Predictive Automation")

# Sidebar Navigation
page = st.sidebar.radio("Select Option", ["Upload & Train", "Predict"])

# Ensure session state variables exist
if "feature_names" not in st.session_state:
    st.session_state.feature_names = []
if "problem_type" not in st.session_state:
    st.session_state.problem_type = None
if "trained_models" not in st.session_state:
    st.session_state.trained_models = []

# Upload & Train Page
if page == "Upload & Train":
    st.header("📂 Upload Dataset & Train Model")
    file_type = st.selectbox("Select file type:", ["CSV", "Excel", "Text"])

    # File uploader
    uploaded_file = st.file_uploader("Upload a file", type=["csv", "xlsx", "xls", "txt"])

    if uploaded_file is not None:
        # Read file based on type
        if file_type == "CSV":
            df = pd.read_csv(uploaded_file)
        elif file_type == "Excel":
            df = pd.read_excel(uploaded_file)
        elif file_type == "Text":
            content = uploaded_file.read().decode("utf-8")
            st.write("Uploaded Text File Content:")
            st.text(content)
            st.stop()

        st.write("📊 **Dataset Preview:**")
        st.dataframe(df)

        # Target & Feature Selection
        target_col = st.selectbox("🎯 Select Target Column:", df.columns)
        feature_cols = st.multiselect("📌 Select Feature Columns:", df.columns, default=[col for col in df.columns if col != target_col])

        if target_col and feature_cols:
            st.session_state.feature_names = feature_cols  # Save feature names
            problem_type = "classification" if df[target_col].nunique() <= 10 else "regression"
            st.session_state.problem_type = problem_type

            st.write(f"**🔍 Detected Problem Type:** {problem_type.capitalize()}")

            # Available models based on problem type
            classification_models = ["Logistic Regression", "Random Forest", "SVM", "KNN", "Decision Tree", "Naive Bayes"]
            regression_models = ["Linear Regression", "Random Forest Regressor", "SVR", "KNN Regressor", "Decision Tree Regressor"]

            available_models = classification_models if problem_type == "classification" else regression_models

            selected_models = st.multiselect("📌 Select Algorithms to Train:", available_models)

            # Train models
            if st.button("🚀 Train Model"):
                results = predictor.train_model(df, target_col, feature_cols, selected_models)
                st.session_state.trained_models = selected_models  # Store trained models
                st.success(f"✅ Training Results: {results}")

# Prediction Page
elif page == "Predict":
    st.header("🎯 Make a Prediction")

    if not st.session_state.feature_names or not st.session_state.trained_models:
        st.warning("⚠️ No trained model found. Please upload and train a model first.")
    else:
        st.write(f"📌 **Model trained with {len(st.session_state.feature_names)} features:** {', '.join(st.session_state.feature_names)}")

        # Select trained model
        model_choice = st.selectbox("📌 Select a Trained Model:", st.session_state.trained_models)

        # Automatically generate input fields for each feature
        feature_inputs = [st.number_input(f"Enter value for {feature}", step=0.01) for feature in st.session_state.feature_names]

        if st.button("🔮 Predict"):
            result = predictor.predict(feature_inputs, model_choice)
            st.success(f"🎯 **Predicted Value:** {result}")
