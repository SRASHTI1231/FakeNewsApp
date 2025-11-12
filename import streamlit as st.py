# ============================
# STREAMLIT NLP ANALYZER APP (Google Fact Check Focused)
# ============================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import time

# ============================
# GOOGLE FACT CHECK API WRAPPER
# ============================
class GoogleFactCheckAPI:
    def __init__(self):
        try:
            self.api_key = st.secrets["GOOGLE_FACTCHECK_API_KEY"]
        except:
            self.api_key = None
            st.warning("Google API Key not found! Fact check won't work.")

    def batch_fact_check(self, texts, max_checks=3):
        """Fake implementation placeholder: replace with actual API calls"""
        results = []
        for text in texts:
            # This is just a sample structure
            claims = [
                {
                    "text": text[:50] + "...",
                    "claimReview": [
                        {
                            "textualRating": np.random.choice(["True", "False", "Mixed"]),
                            "publisher": {"name": "FactCheckOrg"},
                            "url": "#"
                        }
                    ],
                    "claimDate": "2025-11-12"
                }
            ]
            results.append({
                "text": text,
                "has_claims": True,
                "claim_count": len(claims),
                "fact_check_results": claims
            })
        return results

# ============================
# FACT CHECK VISUALIZER
# ============================
class FactCheckVisualizer:
    @staticmethod
    def display_claim_analysis(claim):
        """Display individual claim analysis"""
        claim_text = claim.get('text', 'No text available')
        review = claim.get('claimReview', [{}])[0] if claim.get('claimReview') else {}
        rating = review.get('textualRating', 'Unknown').lower()
        publisher = review.get('publisher', {}).get('name', 'Unknown Publisher')
        review_url = review.get('url', '#')
        claim_date = claim.get('claimDate', 'Unknown date')

        if 'true' in rating:
            rating_badge = '<span style="color:green; font-weight:bold;">TRUE</span>'
        elif 'false' in rating:
            rating_badge = '<span style="color:red; font-weight:bold;">FALSE</span>'
        elif 'mixed' in rating:
            rating_badge = '<span style="color:orange; font-weight:bold;">MIXED</span>'
        else:
            rating_badge = '<span style="color:gray; font-weight:bold;">UNKNOWN</span>'

        st.markdown(f"""
        <div style="background-color:#222; padding:10px; border-radius:8px; margin-bottom:8px;">
            <div><strong>Claim:</strong> {claim_text}</div>
            <div style="display:flex; justify-content:space-between; margin-top:5px;">
                <div>Rating: {rating_badge} | Publisher: {publisher}</div>
                <div style="color:#aaa; font-size:0.8rem;">{claim_date}</div>
            </div>
            <div style="margin-top:5px;"><a href="{review_url}" target="_blank" style="color:#4285F4;">View Full Review</a></div>
        </div>
        """, unsafe_allow_html=True)
# ============================
# SIDEBAR CONFIGURATION
# ============================
def setup_sidebar():
    st.sidebar.markdown("<h2 style='color:#FFD700;'>NLP ANALYZER PRO</h2>", unsafe_allow_html=True)
    st.sidebar.markdown("---")
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV File",
        type=["csv"],
        help="Upload your dataset for analysis"
    )
    
    st.sidebar.markdown("<h4 style='color:#FFD700;'>ADVANCED SETTINGS</h4>", unsafe_allow_html=True)
    use_smote = st.sidebar.checkbox(
        "Enable SMOTE", True,
        help="Handle class imbalance for better model performance"
    )
    
    enable_fact_check = st.sidebar.checkbox(
        "Enable Fact Check API", True,
        help="Verify claims using Google Fact Check API"
    )
    
    max_fact_checks = st.sidebar.slider(
        "Max Fact Checks per Text", 1, 10, 3,
        help="Limit number of fact checks to save API calls"
    )
    
    df = None
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
            except Exception as e:
                st.sidebar.error(f"Failed to read file: {e}")
        
        if df is not None:
            st.session_state.df = df
            st.session_state.file_uploaded = True
            st.session_state.use_smote = use_smote
            st.session_state.enable_fact_check = enable_fact_check
            st.session_state.max_fact_checks = max_fact_checks

            st.sidebar.success(f"Loaded {df.shape[0]} rows & {df.shape[1]} columns")
            
            st.sidebar.markdown("<h4 style='color:#FFD700;'>ANALYSIS SETUP</h4>", unsafe_allow_html=True)
            text_col = st.sidebar.selectbox("Text Column", df.columns)
            target_col = st.sidebar.selectbox("Target Column", df.columns)
            feature_type = st.sidebar.selectbox(
                "Feature Type", ["Lexical", "Semantic", "Syntactic", "Pragmatic"]
            )
            
            st.session_state.config = {
                "text_col": text_col,
                "target_col": target_col,
                "feature_type": feature_type
            }
            
            if st.sidebar.button("START ANALYSIS"):
                st.session_state.analyze_clicked = True
            else:
                st.session_state.analyze_clicked = False
        else:
            st.session_state.file_uploaded = False
    else:
        st.session_state.file_uploaded = False
        st.session_state.analyze_clicked = False

# ============================
# MAIN DASHBOARD LAYOUT
# ============================
def main_content():
    st.markdown("<h1 style='color:#FFD700; text-align:center;'>NLP ANALYZER PRO</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#ccc;'>Advanced Text Intelligence with Google Fact Check</p>", unsafe_allow_html=True)
    
    if not st.session_state.get('file_uploaded', False):
        st.info("Upload a CSV file to start analyzing your text data!")
        return
    
    df = st.session_state.df
    config = st.session_state.get('config', {})
    use_smote = st.session_state.get('use_smote', True)
    enable_fact_check = st.session_state.get('enable_fact_check', True)
    max_fact_checks = st.session_state.get('max_fact_checks', 3)
    
    st.markdown("<h3 style='color:#FFD700;'>DATASET OVERVIEW</h3>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("TOTAL RECORDS", df.shape[0])
    with col2:
        st.metric("FEATURES", df.shape[1])
    with col3:
        st.metric("MISSING VALUES", df.isnull().sum().sum())
    with col4:
        n_classes = df[config['target_col']].nunique() if config['target_col'] in df.columns else 0
        st.metric("UNIQUE CLASSES", n_classes)
    
    with st.expander("DATA PREVIEW", expanded=True):
        st.dataframe(df.head(10))
    
    if st.session_state.get('analyze_clicked', False):
        perform_analysis(df, config, use_smote)
        
        if enable_fact_check:
            show_fact_check_section(df, config, max_fact_checks)

# ============================
# FEATURE EXTRACTION & ML
# ============================
def perform_analysis(df, config, use_smote=True):
    st.markdown("<h3 style='color:#FFD700;'>ANALYSIS RESULTS</h3>", unsafe_allow_html=True)
    
    X = df[config['text_col']].fillna("").astype(str)
    y = df[config['target_col']]
    
    # Dummy feature extraction: character count + word count
    X_features = pd.DataFrame({
        "char_count": X.apply(len),
        "word_count": X.apply(lambda t: len(t.split()))
    })
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)
    
    if use_smote:
        sm = SMOTE()
        X_train, y_train = sm.fit_resample(X_train, y_train)
    
    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(probability=True),
        "Naive Bayes": MultinomialNB()
    }
    
    results = {}
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results[name] = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
                "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
                "f1_score": f1_score(y_test, y_pred, average='weighted', zero_division=0),
                "smote_applied": use_smote,
                "n_classes": len(np.unique(y))
            }
        except Exception as e:
            results[name] = {"error": str(e)}
    
    # Display results graphically
    st.markdown("### Model Performance")
    fig, ax = plt.subplots(figsize=(10, 4))
    names = []
    accs = []
    for k, v in results.items():
        if "accuracy" in v:
            names.append(k)
            accs.append(v["accuracy"])
    ax.barh(names, accs, color="#FFD700")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Accuracy")
    st.pyplot(fig)
    
    # Recommended model
    best_model = max([v for v in results.values() if "accuracy" in v], key=lambda x: x['accuracy'])
    st.success(f"Recommended Model: {list(results.keys())[list(results.values()).index(best_model)]} with Accuracy {best_model['accuracy']:.2f}")
