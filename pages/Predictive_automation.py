import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error

class PredictiveAutomation:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.models_dir = "models"
        os.makedirs(self.models_dir, exist_ok=True)

    def train_model(self, df, target_col, feature_cols, selected_models):
        try:
            df = df.dropna()  # Handle missing values
            X = df[feature_cols]
            y = df[target_col]

            problem_type = "classification" if df[target_col].nunique() <= 10 else "regression"
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            results = {}
            model_dict = {
                "Logistic Regression": LogisticRegression() if problem_type == "classification" else None,
                "Random Forest": RandomForestClassifier() if problem_type == "classification" else RandomForestRegressor(),
                "SVM": SVC() if problem_type == "classification" else SVR(),
                "KNN": KNeighborsClassifier() if problem_type == "classification" else KNeighborsRegressor(),
                "Decision Tree": DecisionTreeClassifier() if problem_type == "classification" else DecisionTreeRegressor(),
                "Naive Bayes": GaussianNB() if problem_type == "classification" else None,
                "Linear Regression": None if problem_type == "classification" else LinearRegression(),
            }

            for model_name in selected_models:
                model = model_dict.get(model_name)
                if model is None:
                    continue

                if model_name in ["SVM", "KNN"]:
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)
                    self.scalers[model_name] = scaler
                    joblib.dump(scaler, f"{self.models_dir}/{model_name}_scaler.pkl")

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                if problem_type == "classification":
                    score = accuracy_score(y_test, y_pred) * 100  # Accuracy for classification
                else:
                    score = mean_squared_error(y_test, y_pred) ** 0.5  # RMSE for regression

                results[model_name] = round(score, 2)

                self.models[model_name] = model

                joblib.dump(model, f"{self.models_dir}/{model_name.replace(' ', '_')}.pkl")

            return results
        except Exception as e:
            return f"Training Error: {str(e)}"

    def predict(self, features, model_name):
        try:
            model_path = f"{self.models_dir}/{model_name.replace(' ', '_')}.pkl"
            if not os.path.exists(model_path):
                return "Model not found. Please train a model first."

            model = joblib.load(model_path)
            features = np.array([features])

            scaler_path = f"{self.models_dir}/{model_name}_scaler.pkl"
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
                features = scaler.transform(features)

            prediction = model.predict(features)[0]
            return prediction
        except Exception as e:
            return f"Prediction Error: {str(e)}"

# Initialize
predictor = PredictiveAutomation()

st.title("🔍 Predictive Automation")

page = st.sidebar.radio("Select Option", ["Upload & Train", "Predict"], key="page_selection")

if "feature_names" not in st.session_state:
    st.session_state.feature_names = []
if "problem_type" not in st.session_state:
    st.session_state.problem_type = None
if "trained_models" not in st.session_state:
    st.session_state.trained_models = []

if page == "Upload & Train":
    st.header("📂 Upload Dataset & Train Model")
    file_type = st.selectbox("Select file type:", ["CSV", "Excel", "Text"], key="file_type")
    uploaded_file = st.file_uploader("Upload a file", type=["csv", "xlsx", "xls", "txt"], key="uploaded_file")

    if uploaded_file is not None:
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

        target_col = st.selectbox("🎯 Select Target Column:", df.columns, key="target_col")
        feature_cols = st.multiselect("📌 Select Feature Columns:", df.columns, default=[col for col in df.columns if col != target_col], key="feature_cols")

        if target_col and feature_cols:
            st.session_state.feature_names = feature_cols
            problem_type = "classification" if df[target_col].nunique() <= 10 else "regression"
            st.session_state.problem_type = problem_type

            st.write(f"**🔍 Detected Problem Type:** {problem_type.capitalize()}")

            classification_models = ["Logistic Regression", "Random Forest", "SVM", "KNN", "Decision Tree", "Naive Bayes"]
            regression_models = ["Linear Regression", "Random Forest", "SVR", "KNN Regressor", "Decision Tree Regressor"]

            available_models = classification_models if problem_type == "classification" else regression_models
            selected_models = st.multiselect("📌 Select Algorithms to Train:", available_models, key="selected_models")

            if st.button("🚀 Train Model", key="train_button"):
                results = predictor.train_model(df, target_col, feature_cols, selected_models)
                st.session_state.trained_models = selected_models
                st.success(f"✅ Training Results: {results}")

elif page == "Predict":
    st.header("🎯 Make a Prediction")
    if not st.session_state.feature_names or not st.session_state.trained_models:
        st.warning("⚠️ No trained model found. Please upload and train a model first.")
    else:
        st.write(f"📌 **Model trained with {len(st.session_state.feature_names)} features:** {', '.join(st.session_state.feature_names)}")
        model_choice = st.selectbox("📌 Select a Trained Model:", st.session_state.trained_models, key="model_choice")
        feature_inputs = [st.number_input(f"Enter value for {feature}", step=0.01, key=f"input_{feature}") for feature in st.session_state.feature_names]

        if st.button("🔮 Predict", key="predict_button"):
            result = predictor.predict(feature_inputs, model_choice)
            st.success(f"🎯 **Predicted Value:** {result}")
