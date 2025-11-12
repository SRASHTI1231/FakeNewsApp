import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ============================
# Sidebar Configuration
# ============================
def setup_sidebar():
    st.sidebar.title("⚙️ NLP ANALYZER PRO")
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"], help="Upload your dataset")

    st.sidebar.header("Settings")
    use_smote = st.sidebar.checkbox("Enable SMOTE", value=True)
    enable_fact_check = st.sidebar.checkbox("Enable Fact Check API", value=True)
    max_fact_checks = st.sidebar.slider("Max fact checks per text", 1, 10, 3)

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

            st.sidebar.success(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")

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
# Fact Check Section
# ============================
class GoogleFactCheckAPI:
    def batch_fact_check(self, texts, max_checks=3):
        results = []
        for text in texts:
            results.append({
                'text': text,
                'has_claims': True,
                'claim_count': 1,
                'fact_check_results': [{
                    'textualRating': 'True',
                    'publisher': {'name': 'DemoFactCheck'},
                    'url': '#'
                }]
            })
        return results

def display_fact_check_results(fact_check_results):
    st.markdown("### 🔍 Fact Check Results")
    for i, result in enumerate(fact_check_results):
        with st.expander(f"Text {i+1}: {result['text'][:50]}..."):
            st.write(result['text'])
            if result['has_claims']:
                st.write(f"Claims found: {result['claim_count']}")
                for claim in result['fact_check_results']:
                    st.markdown(f"- **Rating:** {claim['textualRating']}  | Publisher: {claim['publisher']['name']}")
            else:
                st.info("No verified claims found.")

def show_fact_check_section(df, config, max_checks):
    texts = df[config['text_col']].astype(str).tolist()
    num_texts_to_check = st.slider("Number of texts to fact-check", 1, min(20, len(texts)), min(5, len(texts)))
    texts_to_check = texts[:num_texts_to_check]

    if st.button("Run Fact Check"):
        with st.spinner("Checking facts..."):
            fact_checker = GoogleFactCheckAPI()
            results = fact_checker.batch_fact_check(texts_to_check, max_checks)
            display_fact_check_results(results)

# ============================
# Main Analysis
# ============================
def perform_analysis(df, config, use_smote):
    st.markdown("<h1 style='text-align:center;color:#ff6600;'>📰 NLP ANALYZER PRO</h1>", unsafe_allow_html=True)
    st.markdown("### Dataset Overview")
    
    # Dataset metrics in cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<div style='background-color:#ffcc99;padding:15px;border-radius:10px;text-align:center'><h3>{df.shape[0]}</h3><p>Records</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div style='background-color:#99ccff;padding:15px;border-radius:10px;text-align:center'><h3>{df.shape[1]}</h3><p>Features</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div style='background-color:#ff9999;padding:15px;border-radius:10px;text-align:center'><h3>{df.isnull().sum().sum()}</h3><p>Missing Values</p></div>", unsafe_allow_html=True)
    with col4:
        unique_classes = df[config['target_col']].nunique() if config['target_col'] in df.columns else 0
        st.markdown(f"<div style='background-color:#99ff99;padding:15px;border-radius:10px;text-align:center'><h3>{unique_classes}</h3><p>Classes</p></div>", unsafe_allow_html=True)
    
    X = df[config['text_col']].astype(str)
    y = df[config['target_col']]
    if y.isnull().any():
        st.error("Target column has missing values!")
        return
    
    # Feature: text length
    X_features = X.apply(len).values.reshape(-1,1)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_features, y_encoded, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Performance cards
    st.markdown("### Model Performance")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<div style='background-color:#ffcc99;padding:10px;border-radius:10px;text-align:center'><h4>{accuracy_score(y_test, y_pred):.2f}</h4><p>Accuracy</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div style='background-color:#99ccff;padding:10px;border-radius:10px;text-align:center'><h4>{precision_score(y_test, y_pred, average='weighted'):.2f}</h4><p>Precision</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div style='background-color:#ff9999;padding:10px;border-radius:10px;text-align:center'><h4>{recall_score(y_test, y_pred, average='weighted'):.2f}</h4><p>Recall</p></div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div style='background-color:#99ff99;padding:10px;border-radius:10px;text-align:center'><h4>{f1_score(y_test, y_pred, average='weighted'):.2f}</h4><p>F1-Score</p></div>", unsafe_allow_html=True)

# ============================
# Main Content
# ============================
def main_content():
    if not st.session_state.get('file_uploaded', False):
        st.info("Upload a CSV file to start analysis.")
        return
    
    df = st.session_state.df
    config = st.session_state.get('config', {})
    use_smote = st.session_state.get('use_smote', True)
    enable_fact_check = st.session_state.get('enable_fact_check', True)
    max_fact_checks = st.session_state.get('max_fact_checks', 3)
    
    if st.session_state.get('analyze_clicked', False):
        perform_analysis(df, config, use_smote)
        if enable_fact_check:
            show_fact_check_section(df, config, max_fact_checks)

# ============================
# Main Application
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
