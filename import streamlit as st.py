import streamlit as st
import pandas as pd
import numpy as np
from time import sleep
import plotly.express as px

# ============================
# Sidebar Configuration
# ============================
def setup_sidebar():
    st.sidebar.markdown("<h2 style='color:#FFD700;'>NLP ANALYZER PRO</h2>", unsafe_allow_html=True)
    st.sidebar.markdown("---")
    
    st.sidebar.markdown("<h3>UPLOAD DATA</h3>", unsafe_allow_html=True)
    uploaded_file = st.sidebar.file_uploader(
        "Choose CSV File",
        type=["csv"],
        help="Upload your dataset for analysis"
    )
    
    st.sidebar.markdown("<h3>ADVANCED SETTINGS</h3>", unsafe_allow_html=True)
    use_smote = st.sidebar.checkbox("Enable SMOTE", value=True, help="Handle class imbalance")
    enable_fact_check = st.sidebar.checkbox("Enable Fact Check", value=True, help="Verify claims via Google API")
    max_fact_checks = st.sidebar.slider("Max Fact Checks per Text", 1, 10, 3)
    
    if uploaded_file is not None:
        try:
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(uploaded_file, encoding='latin1')
            st.session_state.df = df
            st.session_state.file_uploaded = True
            st.session_state.use_smote = use_smote
            st.session_state.enable_fact_check = enable_fact_check
            st.session_state.max_fact_checks = max_fact_checks
            
            st.sidebar.success(f"Loaded: {df.shape[0]} rows")
            
            st.sidebar.markdown("<h3>ANALYSIS SETUP</h3>", unsafe_allow_html=True)
            text_col = st.sidebar.selectbox("Text Column", df.columns)
            target_col = st.sidebar.selectbox("Target Column", df.columns)
            feature_type = st.sidebar.selectbox("Feature Type", ["Lexical", "Semantic", "Syntactic", "Pragmatic"])
            
            st.session_state.config = {'text_col': text_col, 'target_col': target_col, 'feature_type': feature_type}
            
            if st.sidebar.button("START ANALYSIS", use_container_width=True):
                st.session_state.analyze_clicked = True
            else:
                st.session_state.analyze_clicked = False
                
        except Exception as e:
            st.sidebar.error(f"Error: {str(e)}")
    else:
        st.session_state.file_uploaded = False
        st.session_state.analyze_clicked = False

# ============================
# Fact Check Section
# ============================
class GoogleFactCheckAPI:
    def batch_fact_check(self, texts, max_checks=3):
        results = []
        for text in texts:
            sleep(0.5)  # simulate API delay
            results.append({
                'text': text,
                'has_claims': True,
                'claim_count': np.random.randint(1, max_checks+1),
                'fact_check_results': [
                    {
                        'textualRating': np.random.choice(["True", "False", "Partially True"]),
                        'publisher': {'name': "Sample Publisher"},
                        'url': '#'
                    } for _ in range(np.random.randint(1, max_checks+1))
                ]
            })
        return results

class FactCheckVisualizer:
    @staticmethod
    def display_claim_analysis(claim):
        st.markdown(f"**Rating:** {claim['claimReview'][0]['textualRating']}")
        st.markdown(f"**Publisher:** {claim['claimReview'][0]['publisher']['name']}")
        st.markdown(f"**URL:** {claim['claimReview'][0]['url']}")
    
    @staticmethod
    def create_fact_check_summary(results):
        st.markdown(f"**Total Texts Checked:** {len(results)}")
        total_claims = sum([r['claim_count'] for r in results])
        st.markdown(f"**Total Claims Found:** {total_claims}")

def show_fact_check_section(df, config, max_checks=3):
    st.markdown("<h3 style='color:#FFD700;'>FACT CHECK ANALYSIS</h3>", unsafe_allow_html=True)
    fact_checker = GoogleFactCheckAPI()
    texts = df[config['text_col']].astype(str).tolist()
    num_texts_to_check = st.slider("Number of texts to fact-check", 1, min(20, len(texts)), min(5, len(texts)))
    texts_to_check = texts[:num_texts_to_check]
    
    if st.button("START FACT CHECK", use_container_width=True, key="fact_check_btn"):
        with st.spinner("Fact-checking..."):
            progress_bar = st.progress(0)
            fact_check_results = []
            for i, text in enumerate(texts_to_check):
                fact_check_results.extend(fact_checker.batch_fact_check([text], max_checks))
                progress_bar.progress(int((i+1)/len(texts_to_check)*100))
            display_fact_check_results(fact_check_results)

def display_fact_check_results(fact_check_results):
    FactCheckVisualizer.create_fact_check_summary(fact_check_results)
    for i, result in enumerate(fact_check_results):
        with st.expander(f"Text {i+1}: {result['text'][:50]}..."):
            st.markdown(f"**Original Text:** {result['text']}")
            if result['has_claims']:
                st.markdown(f"**Found {result['claim_count']} verified claims:**")
                for claim in result['fact_check_results']:
                    FactCheckVisualizer.display_claim_analysis({'claimReview':[claim]})
            else:
                st.info("No claims found")

# ============================
# Main Content
# ============================
def main_content():
    st.markdown("<h1 style='color:#FFD700;'>NLP ANALYZER PRO</h1>", unsafe_allow_html=True)
    
    if not st.session_state.get('file_uploaded', False):
        st.markdown("Upload a CSV file from the sidebar to start analysis.")
        return
    
    df = st.session_state.df
    config = st.session_state.get('config', {})
    use_smote = st.session_state.get('use_smote', True)
    enable_fact_check = st.session_state.get('enable_fact_check', True)
    
    # Dataset Overview
    st.markdown("<h3 style='color:#FFD700;'>DATASET OVERVIEW</h3>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", df.shape[0])
    col2.metric("Features", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())
    col4.metric("Unique Classes", df[config.get('target_col', '')].nunique() if config.get('target_col') in df.columns else 0)
    
    # Data Preview
    with st.expander("DATA PREVIEW", expanded=True):
        tab1, tab2 = st.tabs(["First 10 Rows", "Statistics"])
        with tab1:
            st.dataframe(df.head(10), use_container_width=True)
        with tab2:
            st.write(df.describe(include='all'))
    
    # Fact Check Section
    if enable_fact_check:
        show_fact_check_section(df, config, st.session_state.get('max_fact_checks', 3))

# ============================
# Main Application
# ============================
def main():
    if 'file_uploaded' not in st.session_state: st.session_state.file_uploaded = False
    if 'analyze_clicked' not in st.session_state: st.session_state.analyze_clicked = False
    if 'use_smote' not in st.session_state: st.session_state.use_smote = True
    if 'enable_fact_check' not in st.session_state: st.session_state.enable_fact_check = True
    if 'max_fact_checks' not in st.session_state: st.session_state.max_fact_checks = 3
    
    setup_sidebar()
    main_content()

if __name__ == "__main__":
    main()
