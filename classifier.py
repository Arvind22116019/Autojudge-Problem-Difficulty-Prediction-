import re
import numpy as np
import joblib
from scipy.sparse import hstack

tfidf = joblib.load("tfidf_vectorizer.pkl")
scaler = joblib.load("numeric_scaler.pkl")
clf = joblib.load("difficulty_classifier.pkl")
reg = joblib.load("difficulty_regressor.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_numeric_features(text):
    return np.array([[
        len(text),
        len(text.split()),
        len(re.findall(r"[=<>+\-*/^%]", text)),
        int(bool(re.search(r"10\^\d+", text)))
    ]])

def predict_difficulty(title, description, input_desc, output_desc):
    combined = clean_text(title + " " + description + " " + input_desc + " " + output_desc)

    X_text = tfidf.transform([combined])
    X_num = scaler.transform(extract_numeric_features(combined))
    X_final = hstack([X_text, X_num])

    difficulty_class = clf.predict(X_final)[0]
    difficulty_score = reg.predict(X_final.toarray())[0] * 10

    difficulty_score = max(0.0, min(100.0, difficulty_score))
    return difficulty_class, round(difficulty_score, 2)