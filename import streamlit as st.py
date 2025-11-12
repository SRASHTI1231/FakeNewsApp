# ==============================
# 🧠 Fake News Detection App
# Built by Srashti 💜
# ==============================

import streamlit as st
import requests
import spacy
from textblob import TextBlob
import ftfy
import os

# ====================================
# PAGE CONFIG
# ====================================
st.set_page_config(
    page_title="🧠 Fake News Detection App",
    page_icon="📰",
    layout="wide"
)

# ====================================
# CUSTOM STYLING
# ====================================
st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #4a148c, #880e4f, #311b92);
            color: white;
        }
        .stApp {
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(12px);
            border-radius: 20px;
            padding: 2rem;
            margin: 2rem;
            box-shadow: 0 8px 25px rgba(0,0,0,0.2);
        }
        h1 {
            text-align: center;
            color: #f3e5f5;
            font-size: 3rem;
            text-shadow: 0px 0px 10px #ce93d8;
        }
        .sub-header {
            text-align: center;
            color: #f8bbd0;
            font-size: 1.2rem;
            margin-bottom: 2rem;
        }
        .result-box {
            background: rgba(255,255,255,0.15);
            padding: 1.5rem;
            border-radius: 15px;
            margin-top: 1rem;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# ====================================
# CACHED SPACY MODEL LOADER
# ====================================
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except:
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

# ====================================
# APP HEADER
# ====================================
st.markdown("<h1>📰 Fake News Detector</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Developed by 💜 Srashti • Powered by SpaCy + Google Fact Check + TextBlob</p>", unsafe_allow_html=True)

# ====================================
# USER INPUT AREA
# ====================================
user_input = st.text_area(
    "🖊️ Enter a news headline or short article to analyze:",
    height=150,
    placeholder="Type or paste your news text here..."
)

# ====================================
# ANALYZE BUTTON
# ====================================
if st.button("🔍 Analyze Now"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        with st.spinner("🧠 Analyzing your text... please wait..."):
            text = ftfy.fix_text(user_input)
            doc = nlp(text)

            # -------------------------------
            # Sentiment Analysis (TextBlob)
            # -------------------------------
            sentiment = TextBlob(text).sentiment.polarity
            sentiment_label = (
                "🟢 Positive" if sentiment > 0 
                else "🔴 Negative" if sentiment < 0 
                else "🟡 Neutral"
            )

            # -------------------------------
            # Google Fact Check API
            # -------------------------------
            API_KEY = os.getenv("GOOGLE_API_KEY")  # stored in Streamlit Secrets
            data = {}

            if API_KEY:
                try:
                    url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={text}&key={API_KEY}"
                    response = requests.get(url)
                    data = response.json()
                except Exception as e:
                    st.error(f"Error connecting to Google Fact Check API: {e}")
            else:
                st.info("⚙️ Google API key not set. Add it in Streamlit → Settings → Secrets → GOOGLE_API_KEY")

            st.success("✅ Analysis Completed!")

            # -------------------------------
            # DISPLAY RESULTS
            # -------------------------------
            st.markdown("<div class='result-box'>", unsafe_allow_html=True)

            st.subheader("🧩 Sentiment Analysis")
            st.write(f"Overall Sentiment: **{sentiment_label}** (score: {sentiment:.2f})")

            st.subheader("📚 Key Entities Found")
            entities = [ent.text for ent in doc.ents]
            st.write(", ".join(entities) if entities else "No named entities found.")

            st.subheader("🔎 Fact Check Results")
            if data and "claims" in data:
                for claim in data["claims"]:
                    claim_text = claim.get("text", "")
                    claim_rating = claim.get("claimReview", [{}])[0].get("textualRating", "No rating available")
                    st.info(f"📰 **Claim:** {claim_text}\n\n📊 **Rating:** {claim_rating}")
            else:
                st.write("No relevant fact checks found for this text.")

            st.markdown("</div>", unsafe_allow_html=True)

# ====================================
# FOOTER
# ====================================
st.caption("✨ Built with ❤️ by Srashti | Powered by Streamlit, SpaCy & Google Fact Check API")
