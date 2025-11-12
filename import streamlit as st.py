# ============================
# IMPORTS
# ============================
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

# ============================
# SESSION STATE INIT
# ============================
if 'file_uploaded' not in st.session_state:
    st.session_state.file_uploaded = False
if 'analyze_clicked' not in st.session_state:
    st.session_state.analyze_clicked = False
if 'use_smote' not in st.session_state:
    st.session_state.use_smote = True
if 'enable_fact_check' not in st.session_state:
    st.session_state.enable_fact_check = True
if 'max_fact_checks' not in st.session_state:
    st.session_state.max_fact_checks = 3

# ============================
# SIDEBAR
# ============================
def setup_sidebar():
    st.sidebar.title("NLP Analyzer Pro")
    st.sidebar.markdown("---")
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV Dataset", type=["csv"], help="Choose your CSV file for analysis"
    )
    
    use_smote = st.sidebar.checkbox("Enable SMOTE", value=True)
    enable_fact_check = st.sidebar.checkbox("Enable Fact Check", value=True)
    max_fact_checks = st.sidebar.slider("Max Fact Checks per text", 1, 10, 3)

    st.session_state.use_smote = use_smote
    st.session_state.enable_fact_check = enable_fact_check
    st.session_state.max_fact_checks = max_fact_checks

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8', errors='replace')
            st.session_state.df = df
            st.session_state.file_uploaded = True
            st.sidebar.success(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

            st.sidebar.markdown("### Analysis Setup")
            text_col = st.sidebar.selectbox("Text Column", df.columns)
            target_col = st.sidebar.selectbox("Target Column", df.columns)
            feature_type = st.sidebar.selectbox(
                "Feature Type", ["Lexical", "Semantic", "Syntactic", "Pragmatic"]
            )

            st.session_state.config = {
                'text_col': text_col,
                'target_col': target_col,
                'feature_type': feature_type
            }

            if st.sidebar.button("Start Analysis", use_container_width=True):
                st.session_state.analyze_clicked = True
            else:
                st.session_state.analyze_clicked = False
        except Exception as e:
            st.sidebar.error(f"Error reading CSV: {str(e)}")
    else:
        st.session_state.file_uploaded = False
        st.session_state.analyze_clicked = False

# ============================
# FACT CHECK (SAMPLE)
# ============================
def show_fact_check_section(df, config, max_checks=3):
    st.markdown("## Fact Check Results")
    texts = df[config['text_col']].astype(str).tolist()
    num_texts_to_check = st.slider(
        "Number of texts to check", 1, min(20, len(texts)), min(5, len(texts))
    )
    texts_to_check = texts[:num_texts_to_check]

    if st.button("Run Fact Check", key="fact_btn"):
        st.info("Fact Check API call would process here (sample shown).")
        for i, text in enumerate(texts_to_check):
            st.markdown(f"**Text {i+1}:** {text[:100]}...")
            st.success("Fact Check Result: TRUE/FALSE (sample)")

# ============================
# FEATURE EXTRACTION (SAMPLE)
# ============================
def extract_features(df, config):
    # Simple sample: text length, word count
    df['text_length'] = df[config['text_col']].astype(str).apply(len)
    df['word_count'] = df[config['text_col']].astype(str).apply(lambda x: len(x.split()))
    return df[['text_length', 'word_count']]

# ============================
# MODEL TRAINING
# ============================
def train_models(df, config, use_smote):
    X = extract_features(df, config)
    y = df[config['target_col']]
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    if use_smote:
        sm = SMOTE(random_state=42)
        X, y = sm.fit_resample(X, y)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=300),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier()
    }
    results = {}
    for name, model in models.items():
        model.fit(X, y)
        y_pred = model.predict(X)
        results[name] = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, average='macro'),
            "recall": recall_score(y, y_pred, average='macro'),
            "f1_score": f1_score(y, y_pred, average='macro'),
            "n_classes": len(np.unique(y)),
            "smote_applied": use_smote
        }
    return results

# ============================
# MAIN CONTENT
# ============================
def main_content():
    st.title("NLP Analyzer Pro")

    if not st.session_state.get('file_uploaded', False):
        st.info("Upload your CSV file from the sidebar to start analysis.")
        return

    df = st.session_state.df
    config = st.session_state.config
    use_smote = st.session_state.use_smote
    enable_fact_check = st.session_state.enable_fact_check
    max_fact_checks = st.session_state.max_fact_checks

    # Dataset Overview
    st.subheader("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", df.shape[0])
    col2.metric("Features", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())
    col4.metric(
        "Unique Classes",
        df[config['target_col']].nunique() if config['target_col'] in df.columns else 0
    )

    # Data preview
    with st.expander("Preview Dataset"):
        st.dataframe(df.head(10))
        st.write(df.describe(include='all'))

    if st.session_state.analyze_clicked:
        st.subheader("Model Training Results")
        results = train_models(df, config, use_smote)
        for model_name, res in results.items():
            st.markdown(f"**{model_name}**")
            st.write(res)
        # Show best model
        best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
        st.success(f"Recommended Model: {best_model[0]} with accuracy {best_model[1]['accuracy']:.2%}")

        if enable_fact_check:
            show_fact_check_section(df, config, max_fact_checks)

# ============================
# MAIN
# ============================
def main():
    setup_sidebar()
    main_content()

if __name__ == "__main__":
    main()
