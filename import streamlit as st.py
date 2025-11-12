import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from time import sleep

# ============================
# Sidebar Configuration
# ============================
def setup_sidebar():
    """Setup Hotstar-style sidebar"""
    st.sidebar.markdown("<h2 style='color:#f9c21b;'>NLP ANALYZER PRO</h2>", unsafe_allow_html=True)
    st.sidebar.markdown("---")
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV File",
        type=["csv"],
        help="Upload your dataset for analysis"
    )
    
    st.sidebar.markdown("<h4 style='color:#f9c21b;'>Advanced Settings</h4>", unsafe_allow_html=True)
    
    use_smote = st.sidebar.checkbox(
        "Enable SMOTE", 
        value=True,
        help="Use SMOTE to handle class imbalance"
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
        value=3
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8', on_bad_lines='skip')
            st.session_state.df = df
            st.session_state.file_uploaded = True
            st.session_state.use_smote = use_smote
            st.session_state.enable_fact_check = enable_fact_check
            st.session_state.max_fact_checks = max_fact_checks
            
            st.sidebar.success(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
            
            st.sidebar.markdown("<h4 style='color:#f9c21b;'>Analysis Setup</h4>", unsafe_allow_html=True)
            
            text_col = st.sidebar.selectbox("Text Column", df.columns)
            target_col = st.sidebar.selectbox("Target Column", df.columns)
            feature_type = st.sidebar.selectbox("Feature Type", ["Lexical","Semantic","Syntactic","Pragmatic"])
            
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
# Main Content
# ============================
def main_content():
    """Main content with Hotstar-like graphics"""
    st.markdown("<h1 style='color:#f9c21b; text-align:center;'>NLP ANALYZER PRO</h1>", unsafe_allow_html=True)
    
    if not st.session_state.get('file_uploaded', False):
        st.info("Upload your CSV file to start analysis!")
        return
    
    df = st.session_state.df
    config = st.session_state.get('config', {})
    
    # Dataset overview
    st.markdown("<h3 style='color:#f9c21b;'>Dataset Overview</h3>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", df.shape[0])
    with col2:
        st.metric("Features", df.shape[1])
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    with col4:
        target = config.get('target_col','')
        classes = df[target].nunique() if target in df.columns else 0
        st.metric("Unique Classes", classes)
    
    # Data Preview
    with st.expander("Preview Data"):
        st.dataframe(df.head(10))
    
    # Analysis Trigger
    if st.session_state.get('analyze_clicked', False):
        perform_analysis(df, config)

# ============================
# Analysis with Progress + Charts
# ============================
def perform_analysis(df, config):
    """Perform analysis with interactive charts and progress"""
    st.markdown("<h3 style='color:#f9c21b;'>Analysis in Progress...</h3>", unsafe_allow_html=True)
    
    progress_bar = st.progress(0)
    
    # Simulate feature extraction
    for i in range(0, 50, 10):
        sleep(0.2)
        progress_bar.progress(i+10)
    
    # Simulate model results
    models = ["Logistic Regression","Random Forest","SVM","Naive Bayes","KNN"]
    accuracy = [0.85,0.88,0.86,0.82,0.80]
    precision = [0.83,0.87,0.85,0.80,0.78]
    recall = [0.82,0.86,0.84,0.81,0.79]
    f1 = [0.82,0.87,0.85,0.80,0.78]
    
    st.markdown("<h3 style='color:#f9c21b;'>Model Accuracy Comparison</h3>", unsafe_allow_html=True)
    
    fig = px.bar(x=models, y=accuracy, text=[f"{a*100:.1f}%" for a in accuracy],
                 labels={'x':'Model','y':'Accuracy'},
                 color=accuracy, color_continuous_scale='Viridis')
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("<h3 style='color:#f9c21b;'>Feature Importance (Sample)</h3>", unsafe_allow_html=True)
    
    feature_types = ["Lexical","Semantic","Syntactic","Pragmatic"]
    feature_counts = [30,30,20,20]
    
    fig2 = px.pie(values=feature_counts, names=feature_types, color=feature_types, 
                  color_discrete_sequence=px.colors.sequential.Tealrose)
    st.plotly_chart(fig2, use_container_width=True)
# ============================
# Fact Check Section
# ============================
def show_fact_check_section(df, config, max_checks=3):
    """Display fact-checking results with interactive charts"""
    st.markdown("<h3 style='color:#f9c21b;'>Fact-Check Analysis</h3>", unsafe_allow_html=True)
    
    # Initialize fake Google Fact Check API class (replace with actual API)
    fact_checker = GoogleFactCheckAPI()
    
    texts = df[config['text_col']].astype(str).tolist()
    num_texts = min(10, len(texts))
    
    num_texts_to_check = st.slider(
        "Number of texts to fact-check",
        min_value=1, max_value=num_texts, value=min(5,num_texts)
    )
    
    texts_to_check = texts[:num_texts_to_check]
    
    if st.button("Start Fact Check"):
        with st.spinner("Checking claims..."):
            # Simulate API call
            fact_check_results = fact_checker.batch_fact_check(texts_to_check, max_checks)
        
        display_fact_check_results(fact_check_results)

# ============================
# Display Fact-Check Results
# ============================
def display_fact_check_results(fact_check_results):
    """Show fact-check results with charts"""
    
    # Summary metrics
    total_texts = len(fact_check_results)
    texts_with_claims = sum(1 for r in fact_check_results if r['has_claims'])
    texts_without_claims = total_texts - texts_with_claims
    
    st.markdown("### Summary")
    st.metric("Total Texts Checked", total_texts)
    st.metric("Texts with Verified Claims", texts_with_claims)
    st.metric("Texts without Verified Claims", texts_without_claims)
    
    # Pie chart for claim presence
    fig = px.pie(
        names=["With Claims","Without Claims"], 
        values=[texts_with_claims,texts_without_claims],
        color=["With Claims","Without Claims"],
        color_discrete_sequence=px.colors.sequential.Turbo
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed claim table
    st.markdown("### Detailed Claim Verification")
    for i, result in enumerate(fact_check_results):
        with st.expander(f"Text {i+1}: {result['text'][:80]}..."):
            st.write("**Original Text:**", result['text'])
            if result['has_claims']:
                st.write(f"**Found {result['claim_count']} verified claims:**")
                for claim in result['fact_check_results']:
                    st.write(f"- Claim Rating: {claim['textualRating']} by {claim['publisher']['name']}")
            else:
                st.info("No verified claims found.")

# ============================
# Fake GoogleFactCheckAPI Class
# ============================
class GoogleFactCheckAPI:
    """Simulated API class - replace with actual Google Fact Check API logic"""
    def batch_fact_check(self, texts, max_checks=3):
        results = []
        for text in texts:
            # Simulate some texts having claims
            has_claims = "?" in text or "$" in text
            fact_check_results = []
            if has_claims:
                for i in range(max_checks):
                    fact_check_results.append({
                        "textualRating": "True" if i%2==0 else "False",
                        "publisher": {"name": f"FactChecker {i+1}"}
                    })
            results.append({
                "text": text,
                "has_claims": has_claims,
                "claim_count": len(fact_check_results),
                "fact_check_results": fact_check_results
            })
        return results

# ============================
# Main Application
# ============================
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
    
    # Fact check section if enabled
    if st.session_state.get('enable_fact_check', False) and st.session_state.get('file_uploaded', False):
        show_fact_check_section(st.session_state.df, st.session_state.config, st.session_state.max_fact_checks)

if __name__ == "__main__":
    main()
