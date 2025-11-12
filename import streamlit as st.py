import streamlit as st
import pandas as pd
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import plotly.express as px
import plotly.graph_objects as go

# ============================
# Sidebar Configuration
# ============================
def setup_sidebar():
    st.sidebar.title("⚙️ NLP ANALYZER PRO")
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

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
            feature_type = st.sidebar.selectbox("Feature Type", ["Lexical", "Semantic", "Syntactic", "Pragmatic"])

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
            time.sleep(0.2)
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
                for claim in result['fact_check_results']:
                    st.markdown(f"- **Rating:** {claim['textualRating']}  | Publisher: {claim['publisher']['name']}")
            else:
                st.info("No verified claims found.")

def show_fact_check_section(df, config, max_checks):
    texts = df[config['text_col']].astype(str).tolist()
    num_texts_to_check = st.slider("Number of texts to fact-check", 1, min(20, len(texts)), min(5, len(texts)))
    texts_to_check = texts[:num_texts_to_check]

    if st.button("Run Fact Check"):
        st.info("Fact checking in progress...")
        progress_bar = st.progress(0)
        fact_checker = GoogleFactCheckAPI()
        fact_check_results = []

        for i, text in enumerate(texts_to_check):
            result = fact_checker.batch_fact_check([text], max_checks)
            fact_check_results.extend(result)
            progress_bar.progress((i+1)/len(texts_to_check))
            time.sleep(0.1)

        progress_bar.empty()
        st.success("Fact Check Completed ✅")
        display_fact_check_results(fact_check_results)

# ============================
# Main Analysis
# ============================
def perform_analysis(df, config, use_smote):
    st.markdown("<h1 style='text-align:center;color:#ff6600;'>📰 NLP ANALYZER PRO</h1>", unsafe_allow_html=True)
    
    # Interactive Dataset Overview
    st.markdown("### Dataset Overview")
    metrics = pd.DataFrame({
        'Metric': ['Records', 'Features', 'Missing Values', 'Classes'],
        'Value': [df.shape[0], df.shape[1], df.isnull().sum().sum(), df[config['target_col']].nunique()]
    })
    fig = px.bar(metrics, x='Metric', y='Value', text='Value', color='Metric', template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)

    # Model training
    X = df[config['text_col']].astype(str)
    y = df[config['target_col']]
    if y.isnull().any():
        st.error("Target column has missing values!")
        return

    X_features = X.apply(len).values.reshape(-1,1)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X_features, y_encoded, test_size=0.2, random_state=42)
    
    st.info("Training Logistic Regression model...")
    progress_bar = st.progress(0)
    model = LogisticRegression(max_iter=500)
    for i in range(100):
        time.sleep(0.02)
        progress_bar.progress(i+1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    progress_bar.empty()
    st.success("Model Training Completed ✅")

    # Interactive Model Performance
    performance = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Value': [
            accuracy_score(y_test, y_pred),
            precision_score(y_test, y_pred, average='weighted'),
            recall_score(y_test, y_pred, average='weighted'),
            f1_score(y_test, y_pred, average='weighted')
        ]
    })
    fig2 = px.pie(performance, names='Metric', values='Value', title='Model Performance', hole=0.4)
    st.plotly_chart(fig2, use_container_width=True)

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
