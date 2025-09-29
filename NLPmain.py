# ============================================
# 📌 Streamlit NLP Phase-wise with All Models
# ============================================

import streamlit as st
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from textblob import TextBlob

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import chardet

# ============================
# Auto Encoding Loader
# ============================
def load_csv_auto(path):
    with open(path, "rb") as f:
        result = chardet.detect(f.read(100000))  
        encoding = result["encoding"]
        confidence = result["confidence"]
        print(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")
    return pd.read_csv(path, encoding=encoding, errors="replace")

# ============================
# Load SpaCy & Globals
# ============================
nlp = spacy.load("en_core_web_sm")
stop_words = STOP_WORDS

# ============================
# Phase Feature Extractors
# ============================
def lexical_preprocess(text):
    try:
        doc = nlp(text.lower())
        tokens = [token.lemma_ for token in doc if token.text not in stop_words and token.is_alpha]
        return " ".join(tokens)
    except:
        return ""

def syntactic_features(text):
    try:
        doc = nlp(text)
        return " ".join([token.pos_ for token in doc])
    except:
        return ""

def semantic_features(text):
    try:
        blob = TextBlob(text)
        return [blob.sentiment.polarity, blob.sentiment.subjectivity]
    except:
        return [0, 0]

def discourse_features(text):
    try:
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        return f"{len(sentences)} {' '.join([s.split()[0] for s in sentences if len(s.split()) > 0])}"
    except:
        return "0"

pragmatic_words = ["must", "should", "might", "could", "will", "?", "!"]
def pragmatic_features(text):
    try:
        text = text.lower()
        return [text.count(w) for w in pragmatic_words]
    except:
        return [0] * len(pragmatic_words)

# ============================
# Train & Evaluate All Models
# ============================
def evaluate_models(X_features, y, test_size):
    results = {}
    models = {
        "Naive Bayes": MultinomialNB(),
        "Decision Tree": DecisionTreeClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=200),
        "SVM": SVC()
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=test_size, random_state=42
    )

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred) * 100
            results[name] = round(acc, 2)
        except Exception as e:
            results[name] = f"Error: {str(e)}"

    return results

# ============================
# Streamlit UI
# ============================
st.set_page_config(
    page_title="🧠 Fake vs Real NLP Analyzer",
    page_icon="📊",
    layout="wide"
)

st.title("📰 Fake vs Real News Detection – NLP Phase-wise Analysis")

# Sidebar for file upload
st.header("📁 Upload Your Dataset")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], label_visibility="visible")

st.info("ℹ️ Upload a CSV with a **text column** and a **label column** to begin analysis.")

if uploaded_file:
    st.success("File uploaded successfully!")

    # Save uploaded file temporarily
    with open("temp.csv", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Use auto-detect function
    df = load_csv_auto("temp.csv")

    # Handle large datasets
    if len(df) > 5000:
        st.warning("Dataset is large — using a sample of 5000 rows for faster processing.")
        df = df.sample(5000, random_state=42)

    st.header("⚙️ Select Columns & Settings")
    text_col = st.selectbox("Select Text Column:", df.columns)
    target_col = st.selectbox("Select Target Column:", df.columns)

    phase = st.selectbox("Select NLP Phase:", [
        "Lexical & Morphological",
        "Syntactic", 
        "Semantic",
        "Discourse",
        "Pragmatic"
    ])

    test_size = st.slider("Select Test Size (%)", 10, 50, 20) / 100

    run_analysis = st.button("🚀 Run Analysis", type="primary")

    # Main content area
    st.write("### 🔎 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)
    
    if run_analysis:
        st.write("---")
        st.write(f"### 🧠 Running Phase: {phase}")
        
        with st.spinner("Processing data and training models..."):
            X = df[text_col].astype(str)
            y = df[target_col]

            if phase == "Lexical & Morphological":
                X_processed = X.apply(lexical_preprocess)
                X_features = CountVectorizer().fit_transform(X_processed)

            elif phase == "Syntactic":
                X_processed = X.apply(syntactic_features)
                X_features = CountVectorizer().fit_transform(X_processed)

            elif phase == "Semantic":
                X_features = pd.DataFrame(X.apply(semantic_features).tolist(),
                                          columns=["polarity", "subjectivity"])

            elif phase == "Discourse":
                X_processed = X.apply(discourse_features)
                X_features = CountVectorizer().fit_transform(X_processed)

            elif phase == "Pragmatic":
                X_features = pd.DataFrame(X.apply(pragmatic_features).tolist(),
                                          columns=pragmatic_words)

            results = evaluate_models(X_features, y, test_size)

        results_df = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])
        results_df = results_df.sort_values(by="Accuracy", ascending=False).reset_index(drop=True)

        st.write("---")
        st.subheader("📊 Model Accuracy Comparison")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        bars = ax1.bar(results_df["Model"], results_df["Accuracy"], 
                       color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
        
        best_idx = results_df["Accuracy"].idxmax()
        bars[best_idx].set_color('#FFD93D')
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(2)
        
        for i, (model, acc) in enumerate(zip(results_df["Model"], results_df["Accuracy"])):
            ax1.text(i, acc + 1, f'{acc:.1f}%', ha='center', va='bottom', 
                     fontsize=12, fontweight='bold', color='darkblue')
        
        ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax1.set_title(f'Model Performance - {phase}\n', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, min(100, max(results_df["Accuracy"]) + 15))
        ax1.grid(axis='y', alpha=0.3)
        ax1.tick_params(axis='x', rotation=15)
        
        wedges, texts, autotexts = ax2.pie(results_df["Accuracy"], 
                                           labels=results_df["Model"], 
                                           autopct='%1.1f%%',
                                           startangle=90,
                                           colors=colors,
                                           explode=[0.1 if i == best_idx else 0 for i in range(len(results_df))])
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        
        ax2.set_title('⚖️ Model Performance Share\n', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.write("### 🏆 Performance Metrics")
        cols = st.columns(4)
        for idx, (model, accuracy) in enumerate(zip(results_df["Model"], results_df["Accuracy"])): 
            with cols[idx % 4]:  # wrap metrics
                if idx == best_idx:
                    st.metric(label=f"🥇 {model}", value=f"{accuracy:.1f}%", delta="Best Performance")
                else:
                    st.metric(label=model, value=f"{accuracy:.1f}%",
                              delta=f"{-round(accuracy - results_df.loc[best_idx, 'Accuracy'], 1)}%")
        
        st.write("### 📋 Accuracy Results Table")
        results_display = results_df.copy()
        results_display["Accuracy"] = results_display["Accuracy"].apply(lambda x: f"{x:.1f}%")
        results_display["Rank"] = range(1, len(results_display) + 1)
        results_display = results_display[["Rank", "Model", "Accuracy"]]
        st.dataframe(results_display, use_container_width=True)

        # Download option
        csv = results_display.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results as CSV", data=csv,
                           file_name="model_results.csv", mime="text/csv")

else:
    st.info("👈 Please upload a CSV file to begin the analysis.")

# Styling
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
    }
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)
