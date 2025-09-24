
"""
LSSDP: NLP-Based Fake vs Real Detection
Phase-wise NLP pipeline for Fake vs Real statement classification using Naive Bayes
"""

import pandas as pd
import numpy as np
import nltk, re, string, spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import hstack
import matplotlib.pyplot as plt

# ============================
# Download resources (only first time)
# ============================
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Download spaCy model (only first time)
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# ============================
# Step 1: Load Dataset
# ============================
DATA_PATH = r"C:\Users\Srashthi\Downloads\data.csv"
  # <-- update your path
df = pd.read_csv(DATA_PATH, encoding="latin1")   # or encoding="cp1252"
try:
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv(DATA_PATH, encoding="latin1")


print("Columns:", df.columns.tolist())

print("Dataset Shape:", df.shape)
print("Columns:", df.columns)
print(df.head())

X = df['Statement']
y = df['BinaryTarget']


# ============================
# Helper: Train NB Model
# ============================
def train_nb(X_features, y, name):
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, random_state=42
    )
    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n🔹 {name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    return acc

# ============================
# Phase 1: Lexical & Morphological Analysis
# ============================
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def lexical_preprocess(text):
    tokens = nltk.word_tokenize(str(text).lower())
    tokens = [
        lemmatizer.lemmatize(w) for w in tokens
        if w not in stop_words and w not in string.punctuation
    ]
    return " ".join(tokens)

X_lexical = X.apply(lexical_preprocess)
vec_lexical = CountVectorizer().fit_transform(X_lexical)
acc1 = train_nb(vec_lexical, y, "Lexical & Morphological Analysis")

# ============================
# Phase 2: Syntactic Analysis
# ============================
def syntactic_features(text):
    doc = nlp(str(text))
    pos_tags = " ".join([token.pos_ for token in doc])
    return pos_tags

X_syntax = X.apply(syntactic_features)
vec_syntax = CountVectorizer().fit_transform(X_syntax)
acc2 = train_nb(vec_syntax, y, "Syntactic Analysis")

# ============================
# Phase 3: Semantic Analysis
# ============================
def semantic_features(text):
    blob = TextBlob(str(text))
    return [blob.sentiment.polarity, blob.sentiment.subjectivity]

X_semantic = np.array(X.apply(semantic_features).tolist())
# NB expects non-negative integer counts; we'll scale to 0-100 and convert to int
X_semantic_scaled = (X_semantic * 100 + 100).astype(int)  # shift negative to positive
acc3 = train_nb(X_semantic_scaled, y, "Semantic Analysis")

# ============================
# Phase 4: Discourse Integration
# ============================
def discourse_features(text):
    sentences = nltk.sent_tokenize(str(text))
    return f"{len(sentences)} {' '.join([s.split()[0] for s in sentences if len(s.split())>0])}"

X_discourse = X.apply(discourse_features)
vec_discourse = CountVectorizer().fit_transform(X_discourse)
acc4 = train_nb(vec_discourse, y, "Discourse Integration")

# ============================
# Phase 5: Pragmatic Analysis
# ============================
pragmatic_words = ["must", "should", "might", "could", "will", "?", "!"]

def pragmatic_features(text):
    features = []
    text_lower = str(text).lower()
    for w in pragmatic_words:
        features.append(f"{w}_{text_lower.count(w)}")
    return " ".join(features)

X_pragmatic = X.apply(pragmatic_features)
vec_pragmatic = CountVectorizer().fit_transform(X_pragmatic)
acc5 = train_nb(vec_pragmatic, y, "Pragmatic Analysis")

# ============================
# Phase 6: Combined Features
# ============================
X_combined = hstack([vec_lexical, vec_syntax, vec_discourse, vec_pragmatic])
acc_combined = train_nb(X_combined, y, "Combined Features (All Phases)")

# ============================
# Final Results
# ============================
print("\n📊 Phase-wise Naive Bayes Accuracies:")
print(f"1. Lexical & Morphological: {acc1:.4f}")
print(f"2. Syntactic: {acc2:.4f}")
print(f"3. Semantic: {acc3:.4f}")
print(f"4. Discourse: {acc4:.4f}")
print(f"5. Pragmatic: {acc5:.4f}")
print(f"6. Combined Features: {acc_combined:.4f}")

# Optional: visualize
plt.bar(
    ["Lexical","Syntactic","Semantic","Discourse","Pragmatic","Combined"],
    [acc1, acc2, acc3, acc4, acc5, acc_combined],
    color="skyblue"
)
plt.ylabel("Accuracy")
plt.title("Phase-wise Naive Bayes Performance")
plt.show()

import streamlit as st

st.title("📰 Fake News Detection App")
st.write("Enter a news statement below to predict if it is Real or Fake:")

user_input = st.text_area("News statement:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        # Preprocess input same as in your pipeline
        cleaned_input = lexical_preprocess(user_input)  # use your existing function
        vec_input = vec_lexical.transform([cleaned_input])
        prediction = model.predict(vec_input)[0]  # use your trained NB model
        label = "🟢 Real" if prediction == 1 else "🔴 Fake"
        st.subheader(f"Prediction: {label}")

"""
LSSDP: NLP-Based Fake vs Real Detection
Phase-wise NLP pipeline for Fake vs Real statement classification using Naive Bayes
"""

import pandas as pd
import numpy as np
import nltk, re, string, spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import hstack
import matplotlib.pyplot as plt

# ============================
# Download resources (only first time)
# ============================
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Download spaCy model (only first time)
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# ============================
# Step 1: Load Dataset
# ============================
DATA_PATH = r"C:\Users\Srashthi\Downloads\data.csv"
  # <-- update your path
df = pd.read_csv(DATA_PATH, encoding="latin1")   # or encoding="cp1252"
try:
    df = pd.read_csv(DATA_PATH, encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv(DATA_PATH, encoding="latin1")


print("Columns:", df.columns.tolist())

print("Dataset Shape:", df.shape)
print("Columns:", df.columns)
print(df.head())

X = df['Statement']
y = df['BinaryTarget']


# ============================
# Helper: Train NB Model
# ============================
def train_nb(X_features, y, name):
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, random_state=42
    )
    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n🔹 {name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    return acc

# ============================
# Phase 1: Lexical & Morphological Analysis
# ============================
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def lexical_preprocess(text):
    tokens = nltk.word_tokenize(str(text).lower())
    tokens = [
        lemmatizer.lemmatize(w) for w in tokens
        if w not in stop_words and w not in string.punctuation
    ]
    return " ".join(tokens)

X_lexical = X.apply(lexical_preprocess)
vec_lexical = CountVectorizer().fit_transform(X_lexical)
acc1 = train_nb(vec_lexical, y, "Lexical & Morphological Analysis")

# ============================
# Phase 2: Syntactic Analysis
# ============================
def syntactic_features(text):
    doc = nlp(str(text))
    pos_tags = " ".join([token.pos_ for token in doc])
    return pos_tags

X_syntax = X.apply(syntactic_features)
vec_syntax = CountVectorizer().fit_transform(X_syntax)
acc2 = train_nb(vec_syntax, y, "Syntactic Analysis")

# ============================
# Phase 3: Semantic Analysis
# ============================
def semantic_features(text):
    blob = TextBlob(str(text))
    return [blob.sentiment.polarity, blob.sentiment.subjectivity]

X_semantic = np.array(X.apply(semantic_features).tolist())
# NB expects non-negative integer counts; we'll scale to 0-100 and convert to int
X_semantic_scaled = (X_semantic * 100 + 100).astype(int)  # shift negative to positive
acc3 = train_nb(X_semantic_scaled, y, "Semantic Analysis")

# ============================
# Phase 4: Discourse Integration
# ============================
def discourse_features(text):
    sentences = nltk.sent_tokenize(str(text))
    return f"{len(sentences)} {' '.join([s.split()[0] for s in sentences if len(s.split())>0])}"

X_discourse = X.apply(discourse_features)
vec_discourse = CountVectorizer().fit_transform(X_discourse)
acc4 = train_nb(vec_discourse, y, "Discourse Integration")

# ============================
# Phase 5: Pragmatic Analysis
# ============================
pragmatic_words = ["must", "should", "might", "could", "will", "?", "!"]

def pragmatic_features(text):
    features = []
    text_lower = str(text).lower()
    for w in pragmatic_words:
        features.append(f"{w}_{text_lower.count(w)}")
    return " ".join(features)

X_pragmatic = X.apply(pragmatic_features)
vec_pragmatic = CountVectorizer().fit_transform(X_pragmatic)
acc5 = train_nb(vec_pragmatic, y, "Pragmatic Analysis")

# ============================
# Phase 6: Combined Features
# ============================
X_combined = hstack([vec_lexical, vec_syntax, vec_discourse, vec_pragmatic])
acc_combined = train_nb(X_combined, y, "Combined Features (All Phases)")

# ============================
# Final Results
# ============================
print("\n📊 Phase-wise Naive Bayes Accuracies:")
print(f"1. Lexical & Morphological: {acc1:.4f}")
print(f"2. Syntactic: {acc2:.4f}")
print(f"3. Semantic: {acc3:.4f}")
print(f"4. Discourse: {acc4:.4f}")
print(f"5. Pragmatic: {acc5:.4f}")
print(f"6. Combined Features: {acc_combined:.4f}")

# Optional: visualize
plt.bar(
    ["Lexical","Syntactic","Semantic","Discourse","Pragmatic","Combined"],
    [acc1, acc2, acc3, acc4, acc5, acc_combined],
    color="skyblue"
)
plt.ylabel("Accuracy")
plt.title("Phase-wise Naive Bayes Performance")
plt.show()

import streamlit as st

st.title("📰 Fake News Detection App")
st.write("Enter a news statement below to predict if it is Real or Fake:")

user_input = st.text_area("News statement:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        # Preprocess input same as in your pipeline
        cleaned_input = lexical_preprocess(user_input)  # use your existing function
        vec_input = vec_lexical.transform([cleaned_input])
        prediction = model.predict(vec_input)[0]  # use your trained NB model
        label = "🟢 Real" if prediction == 1 else "🔴 Fake"
        st.subheader(f"Prediction: {label}")

import spacy

try:
    nlp = spacy.load("en_core_web_sm")
except:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")




