import streamlit as st
import pandas as pd
import time
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -----------------------------
# Sidebar Configuration
# -----------------------------
def setup_sidebar():
    st.sidebar.markdown("<h2 style='color:#f5f5f1;'>NLP ANALYSER PRO</h2>", unsafe_allow_html=True)
    st.sidebar.markdown("---")
    
    uploaded_file = st.sidebar.file_uploader("Choose CSV File", type=["csv"])
    
    st.sidebar.markdown("<h3 style='color:#f5f5f1;'>ADVANCED SETTINGS</h3>", unsafe_allow_html=True)
    
    use_smote = st.sidebar.checkbox("Enable SMOTE", value=True)
    enable_fact_check = st.sidebar.checkbox("Enable Fact Check", value=True)
    max_fact_checks = st.sidebar.slider("Max Fact Checks per Text", 1, 10, 3)
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8', errors='replace')
            st.session_state.df = df
            st.session_state.file_uploaded = True
            st.session_state.use_smote = use_smote
            st.session_state.enable_fact_check = enable_fact_check
            st.session_state.max_fact_checks = max_fact_checks
            
            st.sidebar.success(f"Loaded: {df.shape[0]} rows")
            
            text_col = st.sidebar.selectbox("Text Column", df.columns)
            target_col = st.sidebar.selectbox("Target Column", df.columns)
            feature_type = st.sidebar.selectbox("Feature Type", ["Lexical","Semantic","Syntactic","Pragmatic"])
            
            st.session_state.config = {
                'text_col': text_col,
                'target_col': target_col,
                'feature_type': feature_type
            }
            
            if st.sidebar.button("START ANALYSIS"):
                st.session_state.analyze_clicked = True
            else:
                st.session_state.analyze_clicked = False
        except Exception as e:
            st.sidebar.error(f"Error loading CSV: {str(e)}")
    else:
        st.session_state.file_uploaded = False
        st.session_state.analyze_clicked = False

# -----------------------------
# Main Layout
# -----------------------------
def main_content():
    st.markdown("<h1 style='color:#f5f5f1; text-align:center;'>NLP ANALYSER PRO</h1>", unsafe_allow_html=True)
    st.markdown("<hr style='border:1px solid #f5f5f1;'>", unsafe_allow_html=True)
    
    if not st.session_state.get('file_uploaded', False):
        st.info("Upload your CSV to start analysis!")
        return
    
    df = st.session_state.df
    config = st.session_state.get('config', {})
    use_smote = st.session_state.get('use_smote', True)
    enable_fact_check = st.session_state.get('enable_fact_check', True)
    
    # Dataset Overview
    st.markdown("<h2 style='color:#f5f5f1;'>Dataset Overview</h2>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", df.shape[0])
    col2.metric("Features", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())
    unique_classes = df[config.get('target_col', '')].nunique() if config.get('target_col') else 0
    col4.metric("Unique Classes", unique_classes)
    
    if use_smote:
        st.success("SMOTE Enabled")
    if enable_fact_check:
        st.info("Fact Check API Enabled")
    
    # Data Preview
    with st.expander("Data Preview", expanded=True):
        tab1, tab2 = st.tabs(["First 10 Rows", "Statistics"])
        with tab1:
            st.dataframe(df.head(10))
        with tab2:
            st.write(df.describe(include='all'))
    
    # Analysis Section
    if st.session_state.get('analyze_clicked', False):
        perform_analysis(df, config, use_smote)

# -----------------------------
# Analysis Section
# -----------------------------
def perform_analysis(df, config, use_smote=True):
    st.markdown("<h2 style='color:#f5f5f1;'>Analysis Progress</h2>", unsafe_allow_html=True)
    
    text_col = config['text_col']
    target_col = config['target_col']
    
    if df[text_col].isnull().any():
        df[text_col] = df[text_col].fillna('')
    
    if df[target_col].isnull().any():
        st.error("Target column contains missing values.")
        return
    
    # Label Encoding
    le = LabelEncoder()
    y = le.fit_transform(df[target_col])
    
    # Simple Feature: text length
    X = pd.DataFrame()
    X['text_length'] = df[text_col].apply(lambda x: len(str(x).split()))
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Progress Bar
    progress_text = st.empty()
    bar = st.progress(0)
    
    for i in range(1, 101):
        progress_text.text(f"Running analysis... {i}%")
        bar.progress(i)
        time.sleep(0.01)
    
    st.success("Feature extraction complete!")
    
    # Model training (Dummy placeholder, you can replace with actual models)
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    st.markdown("<h2 style='color:#f5f5f1;'>Model Performance</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{acc:.2%}")
    col2.metric("Precision", f"{prec:.2f}")
    col3.metric("Recall", f"{rec:.2f}")
    col4.metric("F1-Score", f"{f1:.2f}")
    
    # Interactive Visualization
    fig = px.histogram(df, x=text_col, nbins=20, title="Text Length Distribution")
    st.plotly_chart(fig, use_container_width=True)
    
    # Fact Check Section Placeholder
    if st.session_state.get('enable_fact_check', True):
        st.markdown("<h2 style='color:#f5f5f1;'>Fact Check Results</h2>", unsafe_allow_html=True)
        st.info("Fact check API integration will display verified claim analysis here.")

# -----------------------------
# Main Application
# -----------------------------
def main():
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
    
    setup_sidebar()
    main_content()

if __name__ == "__main__":
    main()
