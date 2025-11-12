# ============================
# NLP Analyzer Pro - Interactive Dashboard
# ============================

import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# ============================
# Sidebar Configuration
# ============================
def setup_sidebar():
    st.sidebar.markdown("## NLP Analyzer Pro")
    st.sidebar.markdown("---")
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV File",
        type=["csv"],
        help="Upload your dataset for analysis"
    )
    
    st.sidebar.markdown("## Advanced Settings")
    use_smote = st.sidebar.checkbox("Enable SMOTE", value=True)
    enable_fact_check = st.sidebar.checkbox("Enable Fact Check", value=True)
    max_fact_checks = st.sidebar.slider(
        "Max Fact Checks per Text", min_value=1, max_value=10, value=3
    )
    
    st.session_state.use_smote = use_smote
    st.session_state.enable_fact_check = enable_fact_check
    st.session_state.max_fact_checks = max_fact_checks
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8', errors='replace')
            st.session_state.df = df
            st.session_state.file_uploaded = True
            
            st.sidebar.success(f"Loaded: {df.shape[0]} rows x {df.shape[1]} columns")
            
            st.sidebar.markdown("## Select Columns")
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
            
            if st.sidebar.button("Start Analysis"):
                st.session_state.analyze_clicked = True
            else:
                st.session_state.analyze_clicked = False
                
        except Exception as e:
            st.sidebar.error(f"Error loading CSV: {str(e)}")
    else:
        st.session_state.file_uploaded = False
        st.session_state.analyze_clicked = False

# ============================
# Main Dashboard Layout
# ============================
def main_content():
    st.title("NLP Analyzer Pro")
    
    if not st.session_state.get('file_uploaded', False):
        st.info("Upload a CSV file from the sidebar to start analysis.")
        return
    
    df = st.session_state.df
    config = st.session_state.config
    use_smote = st.session_state.use_smote
    enable_fact_check = st.session_state.enable_fact_check
    
    # ============================
    # Dataset Overview Cards
    # ============================
    st.subheader("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", df.shape[0])
    with col2:
        st.metric("Total Features", df.shape[1])
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    with col4:
        st.metric("Unique Classes", df[config['target_col']].nunique())
    
    # ============================
    # Data Preview Tabs
    # ============================
    with st.expander("Data Preview"):
        tab1, tab2 = st.tabs(["First 10 Rows", "Statistics"])
        with tab1:
            st.dataframe(df.head(10))
        with tab2:
            st.write(df.describe(include='all'))
    
    # ============================
    # Analysis
    # ============================
    if st.session_state.get('analyze_clicked', False):
        perform_analysis(df, config, use_smote)
        if enable_fact_check:
            st.subheader("Fact Check Section")
            st.info("Fact Check API integration would display verified claims here.")
            # Integrate Google Fact Check API as per your API key

# ============================
# Perform ML Analysis
# ============================
def perform_analysis(df, config, use_smote):
    st.subheader("Analysis Results")
    
    # Fill missing text values
    if df[config['text_col']].isnull().any():
        df[config['text_col']] = df[config['text_col']].fillna('')
    
    # Target column validation
    if df[config['target_col']].isnull().any():
        st.error("Target column contains missing values.")
        return
    if df[config['target_col']].nunique() < 2:
        st.error("Target column must have at least 2 unique classes.")
        return
    
    # Simple feature extraction (word count)
    X = df[config['text_col']].apply(lambda x: len(str(x).split())).values.reshape(-1, 1)
    y = df[config['target_col']]
    
    if use_smote:
        smote = SMOTE()
        X, y = smote.fit_resample(X, y)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(probability=True)
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
                "f1_score": f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }
        except Exception as e:
            results[name] = {"error": str(e)}
    
    # Display model results
    st.subheader("Model Performance")
    for name, metrics in results.items():
        if "error" in metrics:
            st.error(f"{name}: {metrics['error']}")
        else:
            st.markdown(f"**{name}**")
            st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
            st.metric("Precision", f"{metrics['precision']:.2%}")
            st.metric("Recall", f"{metrics['recall']:.2%}")
            st.metric("F1 Score", f"{metrics['f1_score']:.2%}")
    
    # Bar chart visualization
    metric_df = pd.DataFrame(results).T.drop(columns=[c for c in results[list(results.keys())[0]].keys() if c == "error"], errors='ignore')
    st.subheader("Performance Comparison")
    st.bar_chart(metric_df)

# ============================
# Main Function
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
