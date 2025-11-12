# ========================================
# Streamlit NLP Analyzer - Neon Dark Theme
# ========================================

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

# ------------------------------
# FEATURE EXTRACTOR (SAMPLE)
# ------------------------------
class FeatureExtractor:
    def extract_lexical_features(self, texts):
        # Dummy vectorization (replace with real features)
        return pd.DataFrame([list(map(len, t.split())) for t in texts])

    def extract_semantic_features(self, texts):
        # Dummy semantic (replace with real sentiment or embeddings)
        return pd.DataFrame([np.random.rand(5) for _ in texts])

    def extract_syntactic_features(self, texts):
        # Dummy syntactic (replace with POS/tag features)
        return pd.DataFrame([np.random.randint(0,5,5) for _ in texts])

    def extract_pragmatic_features(self, texts):
        # Dummy pragmatic (intent/modality)
        return pd.DataFrame([np.random.rand(3) for _ in texts])

# ------------------------------
# ML MODEL TRAINER
# ------------------------------
class ModelTrainer:
    def __init__(self, use_smote=True):
        self.use_smote = use_smote
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
            "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced'),
            "Support Vector": SVC(random_state=42, probability=True, class_weight='balanced'),
            "Naive Bayes": MultinomialNB()
        }

    def train_and_evaluate(self, X, y):
        results = {}
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        n_classes = len(le.classes_)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        for name, model in self.models.items():
            X_train_final, y_train_final = X_train, y_train
            if self.use_smote and n_classes > 1:
                smote = SMOTE(random_state=42)
                X_train_final, y_train_final = smote.fit_resample(X_train, y_train)

            model.fit(X_train_final, y_train_final)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                'model': model,
                'predictions': y_pred,
                'true_labels': y_test,
                'probabilities': y_proba,
                'n_classes': n_classes
            }
        return results, le

# ------------------------------
# FACT CHECK VISUALIZER
# ------------------------------
class FactCheckVisualizer:
    @staticmethod
    def display_claim(claim):
        st.markdown(f"""
        <div style='background:#1a1a2e; padding:10px; border-left:5px solid #e50914; margin-bottom:10px; border-radius:5px;'>
            <p style='color:#fff; margin:0'><strong>Claim:</strong> {claim['text']}</p>
            <p style='color:#f5f5f1; margin:0'>Rating: <strong style='color:#e50914'>{claim.get('rating', 'Unknown')}</strong> | Publisher: {claim.get('publisher', 'Unknown')} | Date: {claim.get('date', '')}</p>
        </div>
        """, unsafe_allow_html=True)

# ------------------------------
# SIDEBAR CONFIGURATION
# ------------------------------
def setup_sidebar():
    st.sidebar.markdown("<h2 style='color:#e50914'>NLP ANALYZER PRO</h2>", unsafe_allow_html=True)
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    
    use_smote = st.sidebar.checkbox("Enable SMOTE", True)
    enable_fact_check = st.sidebar.checkbox("Enable Fact Check", True)
    max_fact_checks = st.sidebar.slider("Max Fact Checks per Text", 1, 10, 3)
    
    config = {}
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.session_state.file_uploaded = True
        st.session_state.use_smote = use_smote
        st.session_state.enable_fact_check = enable_fact_check
        st.session_state.max_fact_checks = max_fact_checks

        config['text_col'] = st.sidebar.selectbox("Text Column", df.columns)
        config['target_col'] = st.sidebar.selectbox("Target Column", df.columns)
        config['feature_type'] = st.sidebar.selectbox("Feature Type", ["Lexical","Semantic","Syntactic","Pragmatic"])
        st.session_state.config = config

        if st.sidebar.button("Start Analysis"):
            st.session_state.analyze_clicked = True
        else:
            st.session_state.analyze_clicked = False
    else:
        st.session_state.file_uploaded = False
        st.session_state.analyze_clicked = False

# ------------------------------
# MAIN CONTENT
# ------------------------------
def main_content():
    st.markdown("<h1 style='color:#e50914;text-align:center;'>NLP Analyzer Pro</h1>", unsafe_allow_html=True)
    if not st.session_state.get('file_uploaded', False):
        st.info("Upload CSV from sidebar to start analysis")
        return

    df = st.session_state.df
    config = st.session_state.config
    use_smote = st.session_state.use_smote

    # Dataset overview
    st.markdown("### Dataset Overview")
    st.dataframe(df.head(10))
    st.markdown(f"**Total Records:** {df.shape[0]} | **Features:** {df.shape[1]} | **Unique Classes:** {df[config['target_col']].nunique()}")

    # Feature Extraction
    st.markdown("### Feature Extraction")
    extractor = FeatureExtractor()
    texts = df[config['text_col']].astype(str)
    if config['feature_type']=="Lexical":
        X = extractor.extract_lexical_features(texts)
    elif config['feature_type']=="Semantic":
        X = extractor.extract_semantic_features(texts)
    elif config['feature_type']=="Syntactic":
        X = extractor.extract_syntactic_features(texts)
    else:
        X = extractor.extract_pragmatic_features(texts)
    y = df[config['target_col']]
    st.success(f"Features extracted: {X.shape[1]}")

    # Train Models
    if st.session_state.get('analyze_clicked', False):
        st.markdown("### Model Training")
        trainer = ModelTrainer(use_smote)
        results, le = trainer.train_and_evaluate(X, y)

        for model_name, res in results.items():
            st.markdown(f"<div style='background:#1a1a2e; padding:10px; border-radius:5px; margin-bottom:10px; color:#fff;'>"
                        f"<strong>{model_name}</strong> | Accuracy: {res['accuracy']:.2%} | Precision: {res['precision']:.3f} | Recall: {res['recall']:.3f} | F1: {res['f1_score']:.3f}</div>", 
                        unsafe_allow_html=True)
        
        best_model = max(results.items(), key=lambda x:x[1]['accuracy'])
        st.markdown(f"<h3 style='color:#e50914;'>Recommended Model: {best_model[0]} | Accuracy: {best_model[1]['accuracy']:.2%}</h3>", unsafe_allow_html=True)

    # Fact Check Section (Sample)
    if st.session_state.get('enable_fact_check', False):
        st.markdown("### Fact Check")
        sample_claims = [
            {"text":"Climate change is caused by humans","rating":"True","publisher":"Climate Facts","date":"2024-01-15"},
            {"text":"Vaccines cause autism","rating":"False","publisher":"Medical Org","date":"2024-01-10"}
        ]
        for claim in sample_claims:
            FactCheckVisualizer.display_claim(claim)

# ------------------------------
# MAIN FUNCTION
# ------------------------------
def main():
    # Init session state
    for key in ['file_uploaded','analyze_clicked','use_smote','enable_fact_check','max_fact_checks']:
        if key not in st.session_state:
            st.session_state[key] = False if 'clicked' in key or 'uploaded' in key else True
    setup_sidebar()
    main_content()

if __name__=="__main__":
    main()
