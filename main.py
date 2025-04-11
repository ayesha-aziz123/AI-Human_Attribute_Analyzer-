import streamlit as st
import PIL.Image
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=gemini_api_key)

# Load Gemini model
model = genai.GenerativeModel("models/gemini-1.5-flash-latest")

# Custom prompt-based function
def analyze_human_attribute(img):
    prompt = """
You are a highly accurate AI specialized in analyzing human facial attributes from uploaded images. 
Observe the image and provide the following details in well-structured, readable **Markdown format**:

Only return clear, confident results — no apologies or assumptions.

Required Attributes:
- **Gender** (Male/Female/Non-binary)
- **Age Estimate** (e.g., 24 years old)
- **Ethnicity** (e.g., Pakistani, Indian, Asian, Caucasian, African, etc.)
- **Mood** (e.g., Happy, Sad, Neutral, Angry, Excited)
- **Facial Expression** (e.g., Smiling, Frowning, Neutral, Laughing)
- **Wearing Glasses** (Yes/No)
- **Beard** (Yes/No)
- **Hair Color** (Black, Blonde, Brown, Grey, etc.)
- **Eye Color** (Brown, Blue, Green, etc.)
- **Headwear** (Yes/No — if yes, mention type like Cap, Hijab, Helmet etc.)
- **Emotions Detected** (e.g., Joyful, Calm, Focused, Nervous, Angry)
- **Confidence Score** (Overall prediction confidence in percentage)

Make sure the output is easy to read, with bold labels and clean formatting.
"""
    response = model.generate_content([prompt, img])
    return response.text.strip()

# Streamlit UI config
st.set_page_config(page_title="Human Attribute Detection", page_icon="🧠", layout="centered")

# Light mode styling
st.markdown("""
    <style>
    body {
        background-color: #FAFAFA;
        color: #333;
        font-family: 'Segoe UI', sans-serif;
    }
    .stApp {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        color: #4A90E2;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Page title
st.markdown("<h1>👤 Human Attribute Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center'>Upload a human image to detect attributes using Gemini AI.</p>", unsafe_allow_html=True)
st.divider()

# Upload section
uploaded_image = st.file_uploader("📤 Upload an Image", type=['png', 'jpg', 'jpeg'])

if uploaded_image:
    img = PIL.Image.open(uploaded_image).convert("RGB")
    
    with st.spinner("Analyzing image... 🔍"):
        try:
            person_info = analyze_human_attribute(img)
        except Exception as e:
            st.error(f"Error: {e}")
            person_info = None

    if person_info:
        col1, col2 = st.columns([1, 1.2])

        with col1:
            st.image(img, caption="Uploaded Image", use_container_width=True)

        with col2:
            st.markdown("### 📝 Attributes Detected")
            st.markdown(person_info)
else:
    st.info("Please upload a face image to begin.")
