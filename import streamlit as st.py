# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import plotly.express as px
import plotly.graph_objects as go
import time

# ============================
# Sidebar Configuration
# ============================
def setup_sidebar():
    st.sidebar.markdown("<h2 style='color: #e50914;'>NLP ANALYSER PRO</h2>", unsafe_allow_html=True)
    st.sidebar.markdown("---")
    
    uploaded_file = st.sidebar.file_uploader(
        "Choose CSV File",
        type=["csv"],
        help="Upload your dataset for analysis"
    )
    
    st.sidebar.markdown("<h3>Advanced Settings</h3>", unsafe_allow_html=True)
    
    use_smote = st.sidebar.checkbox("Enable SMOTE", value=True, help="Handle class imbalance")
    enable_fact_check = st.sidebar.checkbox("Enable Fact Check", value=True, help="Use Google Fact Check API")
    
    max_fact_checks = st.sidebar.slider(
        "Max Fact Checks per Text",
        min_value=1,
        max_value=10,
        value=3,
        help="Limit number of fact checks per text"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8-sig', on_bad_lines='skip')
            st.session_state.df = df
            st.session_state.file_uploaded = True
            st.session_state.use_smote = use_smote
            st.session_state.enable_fact_check = enable_fact_check
            st.session_state.max_fact_checks = max_fact_checks
            
            st.sidebar.success(f"Loaded: {df.shape[0]} rows")
            
            st.sidebar.markdown("<h3>Analysis Setup</h3>", unsafe_allow_html=True)
            
            text_col = st.sidebar.selectbox("Text Column", df.columns)
            target_col = st.sidebar.selectbox("Target Column", df.columns)
            feature_type = st.sidebar.selectbox(
                "Feature Type",
                ["Lexical", "Semantic", "Syntactic", "Pragmatic"]
            )
            
            st.session_state.config = {
                'text_col': text_col,
                'target_col': target_col,
                'feature_type': feature_type
            }
            
            st.session_state.analyze_clicked = st.sidebar.button("START ANALYSIS", use_container_width=True)
                
        except Exception as e:
            st.sidebar.error(f"Error loading CSV: {str(e)}")
    else:
        st.session_state.file_uploaded = False
        st.session_state.analyze_clicked = False

# ============================
# Mock Fact Check API
# ============================
class GoogleFactCheckAPI:
    def batch_fact_check(self, texts, max_checks=3):
        results = []
        for text in texts:
            # Simulate fact check
            results.append({
                'text': text,
                'has_claims': True,
                'claim_count': 1,
                'fact_check_results': [
                    {'textualRating': np.random.choice(['True', 'False', 'Partially True']),
                     'publisher': {'name': 'FactCheckOrg'},
                     'url': '#'}
                ]
            })
        return results

# ============================
# Feature Extractor (Mock)
# ============================
class FeatureExtractor:
    def extract_features(self, X, feature_type="Lexical"):
        # For simplicity, numeric features only (length, word count)
        df = pd.DataFrame()
        df['length'] = X.apply(len)
        df['word_count'] = X.apply(lambda x: len(str(x).split()))
        return df

# ============================
# Model Trainer
# ============================
class ModelTrainer:
    def __init__(self, use_smote=True):
        self.use_smote = use_smote
    
    def train_and_evaluate(self, X, y):
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.3, random_state=42)
        
        if self.use_smote:
            sm = SMOTE(random_state=42)
            X_train, y_train = sm.fit_resample(X_train, y_train)
        
        models = {
            "Logistic Regression": LogisticRegression(max_iter=500),
            "Random Forest": RandomForestClassifier(n_estimators=100),
            "SVM": SVC(probability=True)
        }
        
        results = {}
        progress = st.progress(0)
        for idx, (name, model) in enumerate(models.items()):
            time.sleep(0.5)  # Simulate training
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1_score': f1_score(y_test, y_pred, average='weighted')
            }
            progress.progress((idx+1)/len(models))
        
        return results, le

# ============================
# Fact Check Display
# ============================
def show_fact_check(df, config, max_checks=3):
    st.markdown("### Fact Check Results")
    fact_checker = GoogleFactCheckAPI()
    texts = df[config['text_col']].astype(str).tolist()
    num_texts = min(5, len(texts))
    
    if st.button("Run Fact Check"):
        results = fact_checker.batch_fact_check(texts[:num_texts], max_checks)
        for res in results:
            st.markdown(f"**Text:** {res['text']}")
            for claim in res['fact_check_results']:
                st.markdown(f"- **Rating:** {claim['textualRating']} | **Publisher:** {claim['publisher']['name']}")

# ============================
# Main Analysis
# ============================
def perform_analysis(df, config):
    st.markdown("<h2 style='color:#e50914;'>Analysis Results</h2>", unsafe_allow_html=True)
    
    X = df[config['text_col']].astype(str)
    y = df[config['target_col']]
    
    extractor = FeatureExtractor()
    X_features = extractor.extract_features(X, config['feature_type'])
    
    trainer = ModelTrainer(use_smote=st.session_state.use_smote)
    results, le = trainer.train_and_evaluate(X_features, y)
    
    # Show results table
    results_df = pd.DataFrame(results).T
    results_df.index.name = 'Model'
    st.dataframe(results_df)
    
    # Bar chart
    fig_bar = px.bar(results_df.reset_index(), x='Model', y='accuracy',
                     title="Model Accuracy Comparison", text='accuracy',
                     labels={'accuracy':'Accuracy'})
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Pie chart (accuracy distribution)
    fig_pie = px.pie(results_df.reset_index(), names='Model', values='accuracy',
                     title="Accuracy Distribution")
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # Fact check section
    if st.session_state.enable_fact_check:
        show_fact_check(df, config, st.session_state.max_fact_checks)

# ============================
# Dataset Overview
# ============================
def dataset_overview(df, config):
    st.markdown("<h2 style='color:#e50914;'>Dataset Overview</h2>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Total Records", df.shape[0])
    col2.metric("Features", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())
    col4.metric("Unique Classes", df[config['target_col']].nunique())
    
    st.dataframe(df.head(10))
    
    # Target column distribution
    fig = px.pie(df, names=config['target_col'], title="Target Column Distribution",
                 color_discrete_sequence=px.colors.sequential.Tealrose)
    st.plotly_chart(fig, use_container_width=True)

# ============================
# Main App
# ============================
def main():
    st.set_page_config(page_title="NLP Analyser", layout="wide")
    st.markdown("<h1 style='text-align:center;color:#e50914;'>NLP ANALYSER PRO</h1>", unsafe_allow_html=True)
    
    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = False
        st.session_state.analyze_clicked = False
        st.session_state.use_smote = True
        st.session_state.enable_fact_check = True
        st.session_state.max_fact_checks = 3
    
    setup_sidebar()
    
    if st.session_state.file_uploaded:
        df = st.session_state.df
        config = st.session_state.config
        dataset_overview(df, config)
        if st.session_state.analyze_clicked:
            perform_analysis(df, config)

if __name__ == "__main__":
    main()
