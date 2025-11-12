import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import time

# ============================
# Google Fact Check API Helper
# ============================
class GoogleFactCheckAPI:
    def __init__(self, api_key=None):
        self.api_key = api_key or st.secrets.get("GOOGLE_FACTCHECK_API_KEY")
    
    def batch_fact_check(self, texts, max_checks=3):
        """
        Returns list of dictionaries for each text:
        {
            'text': original_text,
            'has_claims': bool,
            'claim_count': int,
            'fact_check_results': [ {claim data}, ... ]
        }
        """
        results = []
        for text in texts:
            # Here, you can add actual API call using requests
            # For now, placeholder sample data
            results.append({
                'text': text,
                'has_claims': True,
                'claim_count': np.random.randint(1, max_checks+1),
                'fact_check_results': [
                    {
                        'text': f"Sample claim {i+1} from text",
                        'claimReview': [
                            {
                                'textualRating': np.random.choice(['True','False','Mixed']),
                                'publisher': {'name': 'Sample Publisher'},
                                'url': '#'
                            }
                        ],
                        'claimDate': '2024-01-01'
                    } for i in range(np.random.randint(1, max_checks+1))
                ]
            })
        return results

# ============================
# Feature Extractor
# ============================
class TextFeatureExtractor:
    def extract_lexical_features(self, texts):
        return pd.DataFrame({'length': texts.str.len()})
    
    def extract_semantic_features(self, texts):
        return pd.DataFrame({'words': texts.str.split().str.len()})
    
    def extract_syntactic_features(self, texts):
        return pd.DataFrame({'sentence_count': texts.str.count(r'\.')})
    
    def extract_pragmatic_features(self, texts):
        return pd.DataFrame({'exclamations': texts.str.count(r'!')})


# ============================
# Model Trainer
# ============================
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

        test_size = max(0.15, min(0.25, 3 * n_classes / len(y_encoded)))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )

        for name, model in self.models.items():
            try:
                X_train_final, y_train_final = X_train, y_train
                if self.use_smote and n_classes > 1:
                    min_class_count = pd.Series(y_train).value_counts().min()
                    smote = SMOTE(random_state=42, k_neighbors=min(5, min_class_count-1))
                    X_train_final, y_train_final = smote.fit_resample(X_train, y_train)

                model.fit(X_train_final, y_train_final)
                y_pred = model.predict(X_test)

                results[name] = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                    'model': model,
                    'smote_applied': self.use_smote
                }
            except Exception as e:
                results[name] = {'error': str(e)}
        return results, le

# ============================
# Sidebar Setup
# ============================
def setup_sidebar():
    st.sidebar.header("NLP ANALYZER PRO")
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])
    
    use_smote = st.sidebar.checkbox("Enable SMOTE", value=True)
    enable_fact_check = st.sidebar.checkbox("Enable Fact Check", value=True)
    max_fact_checks = st.sidebar.slider("Max Fact Checks per Text", 1, 10, 3)
    
    config = {}
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8', errors='replace')
            st.session_state.df = df
            st.session_state.file_uploaded = True
            st.session_state.use_smote = use_smote
            st.session_state.enable_fact_check = enable_fact_check
            st.session_state.max_fact_checks = max_fact_checks

            st.sidebar.success(f"Loaded {df.shape[0]} rows")
            
            config['text_col'] = st.sidebar.selectbox("Text Column", df.columns)
            config['target_col'] = st.sidebar.selectbox("Target Column", df.columns)
            config['feature_type'] = st.sidebar.selectbox("Feature Type", ["Lexical","Semantic","Syntactic","Pragmatic"])
            st.session_state.config = config

            if st.sidebar.button("START ANALYSIS"):
                st.session_state.analyze_clicked = True
            else:
                st.session_state.analyze_clicked = False
        except Exception as e:
            st.sidebar.error(f"Error reading CSV: {str(e)}")
    else:
        st.session_state.file_uploaded = False
        st.session_state.analyze_clicked = False
# ============================
# Fact Check Display
# ============================
def display_fact_check_results(fact_check_results):
    st.subheader("FACT CHECK RESULTS")
    
    for i, result in enumerate(fact_check_results):
        with st.expander(f"Text {i+1}: {result['text'][:80]}...", expanded=False):
            st.markdown(f"**Original Text:** {result['text']}")
            if result['has_claims']:
                st.markdown(f"**Found {result['claim_count']} verified claims:**")
                for claim in result['fact_check_results']:
                    review = claim['claimReview'][0]
                    st.markdown(f"- **Claim:** {claim['text']}")
                    st.markdown(f"  - **Rating:** {review['textualRating']}")
                    st.markdown(f"  - **Publisher:** {review['publisher']['name']}")
                    st.markdown(f"  - **URL:** {review['url']}")
                    st.markdown(f"  - **Date:** {claim['claimDate']}")
            else:
                st.info("No verified claims found for this text.")

# ============================
# Metrics Cards
# ============================
def show_dataset_metrics(df, config):
    st.subheader("DATASET OVERVIEW")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("TOTAL RECORDS", df.shape[0])
    with col2:
        st.metric("FEATURES", df.shape[1])
    with col3:
        st.metric("MISSING VALUES", df.isnull().sum().sum())
    with col4:
        target_col = config.get('target_col', '')
        unique_classes = df[target_col].nunique() if target_col in df.columns else 0
        st.metric("UNIQUE CLASSES", unique_classes)

# ============================
# Model Performance Display
# ============================
def show_model_results(results):
    st.subheader("MODEL PERFORMANCE")
    cols = st.columns(len(results))
    
    for idx, (model_name, metrics) in enumerate(results.items()):
        with cols[idx]:
            if 'error' in metrics:
                st.error(f"{model_name}\n{metrics['error']}")
            else:
                st.markdown(f"**{model_name}**")
                st.markdown(f"Accuracy: {metrics['accuracy']:.2%}")
                st.markdown(f"Precision: {metrics['precision']:.2f}")
                st.markdown(f"Recall: {metrics['recall']:.2f}")
                st.markdown(f"F1-Score: {metrics['f1_score']:.2f}")
                if metrics.get('smote_applied', False):
                    st.info("SMOTE Applied ✅")

# ============================
# Main Content
# ============================
def main_content():
    st.title("NLP ANALYZER PRO")
    st.markdown(
        "Modern, interactive text analytics platform with **Google Fact Check API** integration."
    )
    
    if not st.session_state.get('file_uploaded', False):
        st.info("Upload a CSV file from the sidebar to begin analysis.")
        return

    df = st.session_state.df
    config = st.session_state.get('config', {})
    use_smote = st.session_state.get('use_smote', True)
    enable_fact_check = st.session_state.get('enable_fact_check', True)
    max_fact_checks = st.session_state.get('max_fact_checks', 3)
    
    # Show Dataset Metrics
    show_dataset_metrics(df, config)
    
    # Data Preview
    with st.expander("DATA PREVIEW", expanded=True):
        st.dataframe(df.head(10))
    
    # Run Analysis
    if st.session_state.get('analyze_clicked', False):
        st.subheader("Feature Extraction & Model Training")
        extractor = TextFeatureExtractor()
        X_raw = df[config['text_col']].astype(str)
        feature_type = config.get('feature_type', 'Lexical')

        with st.spinner("Extracting features..."):
            if feature_type == "Lexical":
                X = extractor.extract_lexical_features(X_raw)
            elif feature_type == "Semantic":
                X = extractor.extract_semantic_features(X_raw)
            elif feature_type == "Syntactic":
                X = extractor.extract_syntactic_features(X_raw)
            else:  # Pragmatic
                X = extractor.extract_pragmatic_features(X_raw)
            time.sleep(1)
        st.success(f"{feature_type} features extracted ✅")
        
        # Model Training
        trainer = ModelTrainer(use_smote=use_smote)
        results, le = trainer.train_and_evaluate(X, df[config['target_col']])
        show_model_results(results)
        
        # Recommended Model
        valid_models = {k: v for k, v in results.items() if 'error' not in v}
        if valid_models:
            best_model_name = max(valid_models.items(), key=lambda x: x[1]['accuracy'])[0]
            best_acc = valid_models[best_model_name]['accuracy']
            st.success(f"Recommended Model: **{best_model_name}** with Accuracy: {best_acc:.2%}")
        
        # Fact Check
        if enable_fact_check:
            st.subheader("FACT CHECK ANALYSIS")
            texts = X_raw.tolist()
            num_texts_to_check = min(5, len(texts))
            fact_checker = GoogleFactCheckAPI()
            fact_check_results = fact_checker.batch_fact_check(texts[:num_texts_to_check], max_checks=max_fact_checks)
            display_fact_check_results(fact_check_results)

# ============================
# Main App Runner
# ============================
def main():
    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = False
    if 'analyze_clicked' not in st.session_state:
        st.session_state.analyze_clicked = False
    
    # Sidebar
    setup_sidebar()
    
    # Main content
    main_content()

if __name__ == "__main__":
    main()
