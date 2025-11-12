import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

# ============================
# Sidebar Configuration
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
# Fact Check Placeholder
# ============================
def show_fact_check_section(df, config, max_checks=3):
    st.markdown("<h3 style='color:#FFD700;'>FACT CHECK SECTION</h3>", unsafe_allow_html=True)
    st.info("Fact Check API integration placeholder. Texts would be verified here.")

# ============================
# Main Content Layout
# ============================
def main_content():
    st.markdown("<h1 style='color:#FFD700; text-align:center;'>NLP ANALYZER PRO</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#ccc;'>Advanced Text Intelligence Platform</p>", unsafe_allow_html=True)
    
    if not st.session_state.get('file_uploaded', False):
        st.info("Upload a CSV file to start analyzing your text data!")
        return
    
    df = st.session_state.df
    config = st.session_state.get('config', {})
    use_smote = st.session_state.get('use_smote', True)
    enable_fact_check = st.session_state.get('enable_fact_check', True)
    max_fact_checks = st.session_state.get('max_fact_checks', 3)
    
    # Dataset Overview
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
# Feature Extraction & ML
# ============================
def perform_analysis(df, config, use_smote=True):
    st.markdown("<h3 style='color:#FFD700;'>ANALYSIS RESULTS</h3>", unsafe_allow_html=True)
    
    X = df[config['text_col']].fillna("").astype(str)
    y = df[config['target_col']]
    
    # Simple feature extraction: char count + word count
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

# ============================
# Main Application
# ============================
def main():
    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = False
    if 'analyze_clicked' not in st.session_state:
        st.session_state.analyze_clicked = False
    
    setup_sidebar()
    main_content()

if __name__ == "__main__":
    main()
