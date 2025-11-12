import streamlit as st
import pandas as pd
import plotly.express as px
from time import sleep

# ----------------------------
# Sidebar Setup
# ----------------------------
def setup_sidebar():
    st.sidebar.markdown("<h2 style='color: gold;'>NLP ANALYZER 🧠</h2>", unsafe_allow_html=True)
    uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=['csv'])

    use_smote = st.sidebar.checkbox("Enable SMOTE (Class Balancing)", value=True)
    enable_fact_check = st.sidebar.checkbox("Enable Google Fact Check API", value=True)
    max_fact_checks = st.sidebar.slider("Max Fact Checks per Text", 1, 10, 3)

    text_col = None
    target_col = None
    feature_type = None

    if uploaded_file is not None:
        try:
            # Handling encoding issues
            try:
                df = pd.read_csv(uploaded_file)
            except UnicodeDecodeError:
                df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')

            st.session_state.df = df
            st.session_state.file_uploaded = True
            st.session_state.use_smote = use_smote
            st.session_state.enable_fact_check = enable_fact_check
            st.session_state.max_fact_checks = max_fact_checks

            st.sidebar.success(f"Loaded {df.shape[0]} rows & {df.shape[1]} columns")

            # Select columns
            text_col = st.sidebar.selectbox("Text Column", df.columns)
            target_col = st.sidebar.selectbox("Target Column", df.columns)
            feature_type = st.sidebar.selectbox("Feature Type", ["Lexical", "Semantic", "Syntactic", "Pragmatic"])

            st.session_state.config = {
                'text_col': text_col,
                'target_col': target_col,
                'feature_type': feature_type
            }

            if st.sidebar.button("START ANALYSIS"):
                st.session_state.analyze_clicked = True
            else:
                st.session_state.analyze_clicked = False

        except Exception as e:
            st.sidebar.error(f"Error loading CSV: {str(e)}")
    else:
        st.session_state.file_uploaded = False
        st.session_state.analyze_clicked = False

# ----------------------------
# Dataset Overview
# ----------------------------
def dataset_overview(df, config):
    st.markdown("<h2 style='color: gold;'>📊 Dataset Overview</h2>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Records", df.shape[0])
    with col2:
        st.metric("Features", df.shape[1])
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    with col4:
        target = config.get('target_col')
        unique_classes = df[target].nunique() if target in df.columns else 0
        st.metric("Unique Classes", unique_classes)

# ----------------------------
# Model Performance Visualization
# ----------------------------
def show_model_performance(results):
    st.markdown("<h2 style='color: gold;'>📈 Model Performance Dashboard</h2>", unsafe_allow_html=True)

    if not results:
        st.info("No models trained yet.")
        return

    # Prepare data for Plotly
    model_names, accuracies, precisions, recalls, f1_scores = [], [], [], [], []
    for model, res in results.items():
        if 'error' not in res:
            model_names.append(model)
            accuracies.append(res['accuracy'])
            precisions.append(res['precision'])
            recalls.append(res['recall'])
            f1_scores.append(res['f1_score'])

    df_perf = pd.DataFrame({
        'Model': model_names,
        'Accuracy': accuracies,
        'Precision': precisions,
        'Recall': recalls,
        'F1-Score': f1_scores
    })

    # Accuracy Bar Chart
    fig = px.bar(
        df_perf,
        x='Model',
        y='Accuracy',
        text=df_perf['Accuracy'].apply(lambda x: f"⭐ {x:.1%}"),
        color='Accuracy',
        color_continuous_scale=px.colors.sequential.Viridis,
        title="Model Accuracy Comparison",
        labels={'Accuracy':'Accuracy (%)'}
    )
    fig.update_traces(textposition='outside', marker_line_width=1.5, marker_line_color="black")
    fig.update_layout(yaxis=dict(tickformat=".0%"))
    st.plotly_chart(fig, use_container_width=True)

    # Hover Data Chart
    fig_hover = px.bar(
        df_perf,
        x='Model',
        y='Accuracy',
        text='Accuracy',
        hover_data={
            'Precision': df_perf['Precision'],
            'Recall': df_perf['Recall'],
            'F1-Score': df_perf['F1-Score']
        },
        color='F1-Score',
        color_continuous_scale=px.colors.sequential.Tealrose,
        title="Model Performance with Hover Info"
    )
    st.plotly_chart(fig_hover, use_container_width=True)

# ----------------------------
# Fact Check Visualization
# ----------------------------
def show_fact_check_chart(claim_counts):
    st.markdown("<h2 style='color: gold;'>🧐 Fact Check Summary</h2>", unsafe_allow_html=True)
    if not claim_counts:
        st.info("No fact-check data yet.")
        return
    df_fact = pd.DataFrame(claim_counts.items(), columns=['Claim Status', 'Count'])
    fig = px.pie(df_fact, names='Claim Status', values='Count',
                 title='Fact Check Result Distribution', color='Claim Status',
                 color_discrete_map={'True':'green', 'False':'red', 'Unverified':'gray'})
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Main Content
# ----------------------------
def main_content():
    st.markdown("<h1 style='color: gold;'>🧠 NLP ANALYZER PRO</h1>", unsafe_allow_html=True)

    if not st.session_state.get('file_uploaded', False):
        st.info("Upload your CSV file to start analysis.")
        return

    df = st.session_state.df
    config = st.session_state.get('config', {})
    dataset_overview(df, config)

    if st.session_state.get('analyze_clicked', False):
        with st.spinner("Performing feature extraction and model training..."):
            sleep(2)  # simulate processing

        # Simulate results for demonstration
        results = {
            "Logistic Regression": {"accuracy": 0.82, "precision":0.79, "recall":0.81, "f1_score":0.80},
            "Random Forest": {"accuracy":0.87, "precision":0.85, "recall":0.86, "f1_score":0.86},
            "SVM": {"accuracy":0.83, "precision":0.80, "recall":0.82, "f1_score":0.81},
        }

        show_model_performance(results)

        # Fact check simulation
        claim_counts = {"True": 5, "False": 3, "Unverified": 2}
        show_fact_check_chart(claim_counts)

# ----------------------------
# Main App
# ----------------------------
def main():
    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = False
    if 'analyze_clicked' not in st.session_state:
        st.session_state.analyze_clicked = False

    setup_sidebar()
    main_content()

if __name__ == "__main__":
    main()
