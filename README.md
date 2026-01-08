# 🤖 AutoJudge: Problem Difficulty Predictor
AutoJudge is a machine learning–based system that automatically predicts the difficulty class (Easy / Medium / Hard) and a numerical difficulty score for programming problems using only their textual descriptions.
The system combines Natural Language Processing (NLP) with machine learning models and provides a simple Streamlit web interface for real-time predictions.

📌 Project Objectives

Automatically classify programming problems into:

Easy

Medium

Hard

Predict a difficulty score (0–100) based on problem complexity

Use only problem text (no solutions or user statistics)

Provide an interactive web interface for testing new problems

📂 Dataset Description

Each problem in the dataset contains:

Title

Problem Description

Input Description

Output Description

Problem Class (Easy, Medium, Hard)

Problem Score (numerical difficulty value)

The dataset is stored in JSON Lines (.jsonl) format, where each line corresponds to one problem.

⚙️ Methodology
1. Text Preprocessing

Convert text to lowercase

Remove extra whitespace and artifacts

Combine all text fields into a single input

2. Feature Engineering

TF-IDF vectors for capturing important keywords (e.g., graph, dp, bfs)

Numeric complexity features:

Character count

Word count

Mathematical symbols

Presence of large constraints (e.g., 10^5)

3. Models Used

Classification Model: Logistic Regression / Random Forest

Regression Model: Gradient Boosting Regressor

4. Prediction Pipeline

Predict difficulty class

Predict or derive difficulty score

Display results in the web interface

🌐 Web Interface

The Streamlit application allows users to:

Enter:

Problem Title

Problem Description

Input Description

Output Description

Click Predict Difficulty

View:

Predicted difficulty class

Difficulty score (0–100)

Visual progress bar

No authentication or database is required.

AutoJudge/
├── app.py                     # Streamlit web app
├── classify.py                # Backend logic (ML + scoring)
├── tfidf_vectorizer.pkl       # Trained TF-IDF model
├── numeric_scaler.pkl         # Scaler for numeric features
├── difficulty_classifier.pkl # Trained classification model
├── problems_data.jsonl        # Dataset
└── README.md                  # Project documentation

▶️ How to Run the Project
1. Install Dependencies
pip install streamlit scikit-learn numpy scipy joblib

2. Run the Web App
   
streamlit run app.py

Open the browser at:

http://localhost:8501
