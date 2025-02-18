import streamlit as st
import os

# Set Page Configuration
st.set_page_config(page_title="AI Automation Suite", layout="wide")

# Initialize session state variables
if "feature_index" not in st.session_state:
    st.session_state.feature_index = 0

# Custom CSS for Styling
st.markdown("""
    <style>
        .stApp {
            background-color: rgba(42, 43, 41, 0.93);
        }
        * {
            font-family: 'Helvetica Neue', sans-serif;
        }
        .title {
            text-align: center;
            color: orange;
            font-size: 50px;
            font-weight: bold;
        }
        .subtitle {
            text-align: center;
            color: white;
            font-size: 20px;
            margin-bottom: 30px;
        }
        .feature-container {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 300px;
        }
        .feature-box {
            width: 500px;
            height: 250px;
            font-size: 28px;
            font-weight: bold;
            color: black;
            text-align: center;
            background-color: #A9CCE3;
            border-radius: 20px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: transform 0.3s ease-in-out, background 0.3s ease-in-out;
            border: 4px solid #2980B9;
            box-shadow: 5px 5px 15px rgba(0,0,0,0.3);
        }
        .feature-box:hover {
            transform: scale(1.08);
            background: linear-gradient(45deg, #A9CCE3, #85C1E9);
        }
        .nav-container {
            display: flex;
            justify-content: space-between;
            margin-top: 20px;
        }
        .nav-button {
            background-color: white;
            color: black;
            padding: 15px 25px;
            font-size: 20px;
            border-radius: 12px;
            cursor: pointer;
            transition: background 0.3s ease-in-out;
            border: none;
            font-weight: bold;
        }
        .nav-button:hover {
            background-color: #ddd;
        }
    </style>
""", unsafe_allow_html=True)

# Page Title
st.markdown('<h1 class="title">🚀 AI Automation Suite</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="subtitle">Click on a Feature to Launch</h3>', unsafe_allow_html=True)

# Features Dictionary
features = [
    ("Predictive Model", "app.py"),
    ("Clustering Model", "clustering.py"),
    ("URL Chatbot", "URl_chatbot.py"),
    ("Image Segmentation", "image_segmentation.py"),
    ("Resume Chatbot", "resume_chatbot.py"),
    ("Live Chatbot", "live_chatbot.py"),
    ("Fraud Detection", "fraud_detection.py"),
    ("Support System", "support_system.py"),
    ("Anomaly Detection", "anomaly_detection.py"),
]

# **Navigation Buttons & Feature Display**
col1, col2, col3 = st.columns([1, 3, 1])

with col1:
    if st.button("⬅ Previous", key="prev_button"):
        st.session_state.feature_index = (st.session_state.feature_index - 1) % len(features)

# Get Current Feature
feature_name, feature_script = features[st.session_state.feature_index]

# **Big Clickable Solid Box**
with col2:
    st.markdown(
        f"""
        <div class="feature-container">
            <div class="feature-box" onclick="window.location.href='{feature_script}'">
                🚀 {feature_name}
            </div>
        </div>
        """, unsafe_allow_html=True
    )

# Handle Click Event in Python
if st.session_state.get("clicked", False):
    st.write(f"🔄 Launching {feature_name}...")
    os.system(f"streamlit run {feature_script}")

with col3:
    if st.button("Next ➡", key="next_button"):
        st.session_state.feature_index = (st.session_state.feature_index + 1) % len(features)
