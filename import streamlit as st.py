import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time

# ============================
# Sidebar Configuration
# ============================
def setup_sidebar():
    """Setup Hotstar-style sidebar for NLP Analyzer"""
    st.sidebar.markdown("<div style='font-size:24px; font-weight:bold; color:#FF6F00;'>NLP ANALYZER PRO</div>", unsafe_allow_html=True)
    st.sidebar.markdown("---")
    
    st.sidebar.markdown("<div style='font-size:18px; font-weight:bold;'>UPLOAD DATA</div>", unsafe_allow_html=True)
    uploaded_file = st.sidebar.file_uploader(
        "Choose CSV File",
        type=["csv"],
        help="Upload your dataset for analysis"
    )
    
    # SMOTE Configuration
    st.sidebar.markdown("<div style='font-size:18px; font-weight:bold;'>ADVANCED SETTINGS</div>", unsafe_allow_html=True)
    
    use_smote = st.sidebar.checkbox(
        "Enable SMOTE", 
        value=True,
        help="Use SMOTE to handle class imbalance (recommended)"
    )
    
    enable_fact_check = st.sidebar.checkbox(
        "Enable Fact Check", 
        value=True,
        help="Use Google Fact Check API to verify claims"
    )
    
    max_fact_checks = st.sidebar.slider(
        "Max Fact Checks per Text",
        min_value=1,
        max_value=10,
        value=3,
        help="Maximum number of fact checks to perform per text"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8', on_bad_lines='skip')
            st.session_state.df = df
            st.session_state.file_uploaded = True
            st.session_state.use_smote = use_smote
            st.session_state.enable_fact_check = enable_fact_check
            st.session_state.max_fact_checks = max_fact_checks
            
            st.sidebar.success(f"Loaded: {df.shape[0]} rows")
            
            st.sidebar.markdown("<div style='font-size:18px; font-weight:bold;'>ANALYSIS SETUP</div>", unsafe_allow_html=True)
            
            text_col = st.sidebar.selectbox(
                "Text Column",
                df.columns,
                help="Select text data column"
            )
            
            target_col = st.sidebar.selectbox(
                "Target Column",
                df.columns,
                help="Select labels column"
            )
            
            feature_type = st.sidebar.selectbox(
                "Feature Type",
                ["Lexical", "Semantic", "Syntactic", "Pragmatic"],
                help="Choose analysis type"
            )
            
            st.session_state.config = {
                'text_col': text_col,
                'target_col': target_col,
                'feature_type': feature_type
            }
            
            if st.sidebar.button("START ANALYSIS", use_container_width=True):
                st.session_state.analyze_clicked = True
            else:
                st.session_state.analyze_clicked = False
                
        except Exception as e:
            st.sidebar.error(f"Error loading CSV: {str(e)}")
    else:
        st.session_state.file_uploaded = False
        st.session_state.analyze_clicked = False

# ============================
# Main Content
# ============================
def main_content():
    """Main content with Hotstar-style layout"""
    
    st.markdown("<h1 style='text-align:center; color:#FF6F00;'>NLP ANALYZER PRO</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#333;'>Advanced Text Intelligence Platform</p>", unsafe_allow_html=True)
    
    if not st.session_state.get('file_uploaded', False):
        st.info("Upload a CSV file from the sidebar to start analysis.")
        return
    
    df = st.session_state.df
    config = st.session_state.get('config', {})
    use_smote = st.session_state.get('use_smote', True)
    enable_fact_check = st.session_state.get('enable_fact_check', True)
    max_fact_checks = st.session_state.get('max_fact_checks', 3)
    
    # ============================
    # Dataset Overview with Progress Bar
    # ============================
    st.markdown("### Dataset Overview")
    progress_bar = st.progress(0)
    for percent_complete in range(0, 101, 20):
        time.sleep(0.1)
        progress_bar.progress(percent_complete)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("TOTAL RECORDS", df.shape[0])
    with col2:
        st.metric("FEATURES", df.shape[1])
    with col3:
        st.metric("MISSING VALUES", df.isnull().sum().sum())
    with col4:
        unique_classes = df[config.get('target_col', '')].nunique() if config.get('target_col') in df.columns else 0
        st.metric("UNIQUE CLASSES", unique_classes)
    
    # Data Preview
    with st.expander("Preview Data", expanded=True):
        st.dataframe(df.head(10))
    
    # Show placeholder for charts
    st.markdown("### Data Distribution")
    if config.get('target_col') in df.columns:
        fig = px.pie(
            df,
            names=config['target_col'],
            title="Target Column Distribution",
            color_discrete_sequence=px.colors.sequential.Turbo
        )
        st.plotly_chart(fig, use_container_width=True)
