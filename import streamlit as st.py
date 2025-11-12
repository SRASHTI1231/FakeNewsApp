# ============================
# IMPORTS
# ============================
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

# ============================
# FEATURE EXTRACTOR
# ============================
class FeatureExtractor:
    def extract_lexical_features(self, texts):
        return pd.DataFrame({"length": texts.str.len()})

    def extract_semantic_features(self, texts):
        return pd.DataFrame({"word_count": texts.str.split().str.len()})

    def extract_syntactic_features(self, texts):
        return pd.DataFrame({"punct_count": texts.str.count(r"[.,!?]")})

    def extract_pragmatic_features(self, texts):
        return pd.DataFrame({"uppercase_ratio": texts.str.isupper().astype(int)})

# ============================
# MODEL TRAINER
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

    def analyze_class_distribution(self, y):
        class_counts = pd.Series(y).value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(class_counts.index, class_counts.values, color="#e50914")
        ax.set_title("Class Distribution")
        ax.set_xlabel("Classes")
        ax.set_ylabel("Count")
        for i, val in enumerate(class_counts.values):
            ax.text(i, val + 0.01, str(val), ha='center')
        plt.tight_layout()
        return fig, class_counts

    def train_and_evaluate(self, X, y):
        results = {}
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        n_classes = len(le.classes_)

        distribution_fig, class_counts = self.analyze_class_distribution(y)
        st.pyplot(distribution_fig)

        min_class_count = class_counts.min()
        max_class_count = class_counts.max()
        imbalance_ratio = max_class_count / min_class_count if min_class_count > 0 else float('inf')

        if self.use_smote and imbalance_ratio > 2:
            st.success(f"SMOTE will be applied (imbalance ratio: {imbalance_ratio:.1f}:1)")
        elif self.use_smote:
            st.info("Class distribution is fairly balanced; SMOTE may have minor effect.")
        else:
            st.warning("SMOTE disabled.")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        for name, model in self.models.items():
            with st.spinner(f"Training {name}..."):
                if self.use_smote and imbalance_ratio > 1.5 and n_classes > 1:
                    smote = SMOTE(random_state=42, k_neighbors=min(5, min_class_count-1))
                    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
                else:
                    X_train_resampled, y_train_resampled = X_train, y_train

                model.fit(X_train_resampled, y_train_resampled)
                y_pred = model.predict(X_test)

                results[name] = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                    'smote_applied': self.use_smote and imbalance_ratio > 1.5 and n_classes > 1
                }

        return results, le

# ============================
# SIDEBAR
# ============================
def setup_sidebar():
    st.sidebar.title("NLP Analyzer Pro")

    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    use_smote = st.sidebar.checkbox("Enable SMOTE", value=True)
    enable_fact_check = st.sidebar.checkbox("Enable Fact Check API", value=False)
    max_fact_checks = st.sidebar.slider("Max Fact Checks per Text", 1, 10, 3)

    analyze_clicked = st.sidebar.button("Start Analysis")

    st.session_state.use_smote = use_smote
    st.session_state.enable_fact_check = enable_fact_check
    st.session_state.max_fact_checks = max_fact_checks
    st.session_state.analyze_clicked = analyze_clicked

    if uploaded_file:
        try:
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
            except UnicodeDecodeError:
                df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
            st.session_state.df = df
            st.session_state.file_uploaded = True

            text_col = st.sidebar.selectbox("Text Column", df.columns)
            target_col = st.sidebar.selectbox("Target Column", df.columns)
            feature_type = st.sidebar.selectbox("Feature Type", ["Lexical", "Semantic", "Syntactic", "Pragmatic"])

            st.session_state.config = {
                'text_col': text_col,
                'target_col': target_col,
                'feature_type': feature_type
            }

        except Exception as e:
            st.sidebar.error(f"Error reading file: {e}")
    else:
        st.session_state.file_uploaded = False

# ============================
# MAIN CONTENT
# ============================
def main_content():
    if not st.session_state.get('file_uploaded', False):
        st.info("Upload a CSV file to start analysis.")
        return

    df = st.session_state.df
    config = st.session_state.config
    use_smote = st.session_state.use_smote
    enable_fact_check = st.session_state.enable_fact_check
    max_fact_checks = st.session_state.max_fact_checks

    # Dataset overview
    st.subheader("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", df.shape[0])
    with col2:
        st.metric("Features", df.shape[1])
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    with col4:
        target_unique = df[config['target_col']].nunique()
        st.metric("Unique Classes", target_unique)

    # Data preview
    with st.expander("Data Preview", expanded=True):
        st.dataframe(df.head(10))
        st.write(df.describe(include='all'))

    # Analysis
    if st.session_state.analyze_clicked:
        extractor = FeatureExtractor()
        X = df[config['text_col']].astype(str)
        y = df[config['target_col']]

        if config['feature_type'] == "Lexical":
            X_features = extractor.extract_lexical_features(X)
        elif config['feature_type'] == "Semantic":
            X_features = extractor.extract_semantic_features(X)
        elif config['feature_type'] == "Syntactic":
            X_features = extractor.extract_syntactic_features(X)
        else:
            X_features = extractor.extract_pragmatic_features(X)

        trainer = ModelTrainer(use_smote=use_smote)
        results, _ = trainer.train_and_evaluate(X_features, y)

        st.subheader("Model Performance")
        for model_name, metrics in results.items():
            with st.expander(model_name):
                st.write(f"Accuracy: {metrics['accuracy']:.2%}")
                st.write(f"Precision: {metrics['precision']:.3f}")
                st.write(f"Recall: {metrics['recall']:.3f}")
                st.write(f"F1 Score: {metrics['f1_score']:.3f}")
                st.write(f"SMOTE Applied: {metrics['smote_applied']}")

        # Fact-checking placeholder
        if enable_fact_check:
            st.subheader("Fact Check Section")
            num_texts = st.slider("Number of texts to fact-check", 1, min(20, len(df)), 5)
            st.info(f"Fact-checking for {num_texts} texts will be implemented here. Max checks per text: {max_fact_checks}")

# ============================
# MAIN APP
# ============================
def main():
    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = False
    if 'analyze_clicked' not in st.session_state:
        st.session_state.analyze_clicked = False
    if 'use_smote' not in st.session_state:
        st.session_state.use_smote = True

    setup_sidebar()
    main_content()

if __name__ == "__main__":
    main()
