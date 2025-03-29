import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import time

# Load ML Model & Vectorizer
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Text Preprocessing Function
def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    cleaned_tokens = [word for word in tokens if word.isalnum()]
    cleaned_tokens = [word for word in cleaned_tokens if word not in stopwords.words('english')]
    ps = PorterStemmer()
    cleaned_tokens = [ps.stem(word) for word in cleaned_tokens]
    return " ".join(cleaned_tokens)

# Set Page Configuration
st.set_page_config(page_title="Spam Classifier", page_icon="📩", layout="centered")

# Custom CSS for Styling
st.markdown("""
    <style>
        .main-title {
            text-align: center;
            font-size: 28px;
            font-weight: bold;
        }
        .result-box {
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            padding: 15px;
            border-radius: 10px;
            color: white;
            margin-top: 20px;
        }
        .spam {
            background-color: #ff4b4b;
        }
        .not-spam {
            background-color: #4CAF50;
        }
    </style>
""", unsafe_allow_html=True)

# App Title
st.markdown('<h1 class="main-title">📩 SMS & Email Spam Classifier</h1>', unsafe_allow_html=True)

# User Input
input_text = st.text_area("✍️ Enter Your Message:", height=150, help="Type or paste your SMS or email message here.")

# Buttons for Prediction & Clear
col1, col2 = st.columns([0.7, 0.3])
with col1:
    predict_btn = st.button("🔍 Predict")
with col2:
    clear_btn = st.button("❌ Clear")

# Clear Input Button
if clear_btn:
    st.experimental_rerun()

# Prediction Logic
if predict_btn:
    if input_text.strip() == "":
        st.warning("⚠️ Please enter a message to classify.")
    else:
        with st.spinner("🔄 Analyzing message..."):
            time.sleep(2)  # Simulate Processing Time
            
            # Preprocess & Predict
            transformed_sms = transform_text(input_text)
            vector_input = tfidf.transform([transformed_sms])
            result = model.predict(vector_input)[0]

            # Display Result
            if result == 1:
                st.markdown('<div class="result-box spam">🚨 This message is SPAM!</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-box not-spam">✅ This message is NOT Spam</div>', unsafe_allow_html=True)
