# ============================
# IMPORTS
# ============================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import time

# ============================
# CUSTOM CSS FOR NEON-DARK THEME
# ============================
st.markdown("""
<style>
/* General background */
body, .stApp {
    background-color: #0d0d0d;
    color: #f5f5f5;
}

/* Header */
.neon-header {
    text-align: center;
    color: #00ffff;
    font-size: 3rem;
    font-weight: 900;
    text-shadow: 0 0 10px #00ffff, 0 0 20px #00ffff, 0 0 30px #00ffff;
    margin-bottom: 1rem;
}

/* Sidebar */
.stSidebar {
    background-color: #111;
    color: #f5f5f5;
}

/* Cards */
.metric-card, .model-card, .fact-card {
    background-color: #1a1a1a;
    padding: 1rem;
    border-radius: 10px;
    box-shadow: 0 0 10px #00ffff;
    margin-bottom: 1rem;
}

/* Badges */
.true-badge { color: #00ff00; font-weight: bold; }
.false-badge { color: #ff2e2e; font-weight: bold; }
.mixed-badge { color: #ffcc00; font-weight: bold; }

/* Feature buttons */
.feature-btn {
    background-color: #222;
    color: #00ffff;
    padding: 0.5rem 1rem;
    margin: 0.2rem;
    border-radius: 5px;
    border: none;
    font-weight: bold;
}
.feature-btn:hover {
    background-color: #00ffff;
    color: #0d0d0d;
    cursor: pointer;
}
</style>
""", unsafe_allow_html=True)

# ============================
# SIDEBAR CONFIGURATION
# ============================
def setup_sidebar():
    st.sidebar.markdown("<h2 style='color:#00ffff;'>NLP DASHBOARD PRO</h2>", unsafe_allow_html=True)
    st.sidebar.markdown("---")
    
    uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
    
    st.sidebar.markdown("### Settings")
    use_smote = st.sidebar.checkbox("Enable SMOTE", value=True)
    enable_fact_check = st.sidebar.checkbox("Enable Fact Check API", value=True)
    max_fact_checks = st.sidebar.slider("Max Fact Checks per Text", 1, 10, 3)
    
    config = {}
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.session_state.file_uploaded = True
        st.session_state.use_smote = use_smote
        st.session_state.enable_fact_check = enable_fact_check
        st.session_state.max_fact_checks = max_fact_checks
        
        st.sidebar.success(f"Loaded: {df.shape[0]} rows")
        
        # Column selection
        config['text_col'] = st.sidebar.selectbox("Text Column", df.columns)
        config['target_col'] = st.sidebar.selectbox("Target Column", df.columns)
        config['feature_type'] = st.sidebar.radio(
            "Feature Type", ["Lexical", "Semantic", "Syntactic", "Pragmatic"], index=0
        )
        st.session_state.config = config
        
        if st.sidebar.button("Start Analysis"):
            st.session_state.analyze_clicked = True
        else:
            st.session_state.analyze_clicked = False
    else:
        st.session_state.file_uploaded = False
        st.session_state.analyze_clicked = False

# ============================
# FEATURE EXTRACTION PLACEHOLDER
# ============================
class FeatureExtractor:
    """Dummy feature extractor (replace with your real logic)"""
    def extract_lexical_features(self, texts):
        return pd.DataFrame({"length": texts.str.len()})
    def extract_semantic_features(self, texts):
        return pd.DataFrame({"words": texts.str.split().str.len()})
    def extract_syntactic_features(self, texts):
        return pd.DataFrame({"punctuation": texts.str.count(r'[^\w\s]')})
    def extract_pragmatic_features(self, texts):
        return pd.DataFrame({"exclamations": texts.str.count(r'!')})

# ============================
# MODEL TRAINER
# ============================
class ModelTrainer:
    def __init__(self, use_smote=True):
        self.use_smote = use_smote
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
            "SVM": SVC(probability=True, random_state=42, class_weight='balanced'),
            "Naive Bayes": MultinomialNB()
        }
        
    def train_and_evaluate(self, X, y):
        results = {}
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        n_classes = len(le.classes_)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
        
        # Check imbalance
        class_counts = pd.Series(y_encoded).value_counts()
        imbalance_ratio = class_counts.max() / class_counts.min() if class_counts.min() > 0 else 1
        
        for name, model in self.models.items():
            X_train_final, y_train_final = X_train, y_train
            if self.use_smote and imbalance_ratio > 1.5:
                smote = SMOTE(random_state=42, k_neighbors=min(5, class_counts.min()-1))
                X_train_final, y_train_final = smote.fit_resample(X_train, y_train)
            
            model.fit(X_train_final, y_train_final)
            y_pred = model.predict(X_test)
            results[name] = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
                "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
                "f1_score": f1_score(y_test, y_pred, average='weighted', zero_division=0),
                "smote_applied": self.use_smote and imbalance_ratio > 1.5
            }
        return results, le

# ============================
# FACT CHECK VISUALIZER (SAMPLE)
# ============================
class FactCheckVisualizer:
    @staticmethod
    def display_claim(claim_text, rating, publisher, date):
        badge_class = "true-badge" if rating.lower() == "true" else ("false-badge" if rating.lower() == "false" else "mixed-badge")
        st.markdown(f"""
        <div class="fact-card">
            <p><strong>Claim:</strong> {claim_text}</p>
            <p>Rating: <span class="{badge_class}">{rating}</span> | Publisher: {publisher} | Date: {date}</p>
        </div>
        """, unsafe_allow_html=True)

# ============================
# MAIN CONTENT
# ============================
def main_content():
    st.markdown("<div class='neon-header'>NLP DASHBOARD PRO</div>", unsafe_allow_html=True)
    
    if not st.session_state.get('file_uploaded', False):
        st.info("Upload your CSV file in the sidebar to start analysis.")
        return
    
    df = st.session_state.df
    config = st.session_state.config
    use_smote = st.session_state.use_smote
    
    # Dataset Overview
    st.markdown("### Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.markdown(f"<div class='metric-card'>Rows<br>{df.shape[0]}</div>", unsafe_allow_html=True)
    with col2: st.markdown(f"<div class='metric-card'>Columns<br>{df.shape[1]}</div>", unsafe_allow_html=True)
    with col3: st.markdown(f"<div class='metric-card'>Missing<br>{df.isnull().sum().sum()}</div>", unsafe_allow_html=True)
    with col4: st.markdown(f"<div class='metric-card'>Classes<br>{df[config['target_col']].nunique()}</div>", unsafe_allow_html=True)
    
    # Feature Extraction
    extractor = FeatureExtractor()
    texts = df[config['text_col']].astype(str)
    if config['feature_type'] == "Lexical":
        X_features = extractor.extract_lexical_features(texts)
    elif config['feature_type'] == "Semantic":
        X_features = extractor.extract_semantic_features(texts)
    elif config['feature_type'] == "Syntactic":
        X_features = extractor.extract_syntactic_features(texts)
    else:
        X_features = extractor.extract_pragmatic_features(texts)
    y = df[config['target_col']]
    
    st.success(f"Features extracted: {config['feature_type']}")
    
    # Model Training
    trainer = ModelTrainer(use_smote)
    results, le = trainer.train_and_evaluate(X_features, y)
    
    st.markdown("### Model Performance")
    for name, res in results.items():
        st.markdown(f"""
        <div class='model-card'>
            <h4>{name} {"(SMOTE)" if res['smote_applied'] else ""}</h4>
            Accuracy: {res['accuracy']:.2%} | Precision: {res['precision']:.3f} | Recall: {res['recall']:.3f} | F1: {res['f1_score']:.3f}
        </div>
        """, unsafe_allow_html=True)
    
    # Recommended model
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    st.markdown(f"<div class='metric-card'>Recommended Model: {best_model[0]} ({best_model[1]['accuracy']:.2%})</div>", unsafe_allow_html=True)
    
    # Sample Fact Check
    if st.session_state.enable_fact_check:
        st.markdown("### Sample Fact Check")
        sample_claims = [
            ("The Earth is round", "True", "Science Org", "2024-01-01"),
            ("Vaccines cause autism", "False", "Medical Org", "2024-01-05")
        ]
        for claim_text, rating, publisher, date in sample_claims:
            FactCheckVisualizer.display_claim(claim_text, rating, publisher, date)

# ============================
# MAIN APP
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
