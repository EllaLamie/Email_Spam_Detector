import os
import joblib
import pandas as pd
from sklearn.svm import LinearSVC
# -------------------------------------------------------
# CHANGE:
# Added imports for additional baseline models
# These models were added as part of the project task:
# Naive Bayes + Logistic Regression Implementation
# -------------------------------------------------------
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from preprocessing import preprocess_dataframe
from feature_engineering import build_tfidf_features
from evaluation import evaluate, print_evaluation

def main():
    data_path = "spam.csv"
    # Make sure the file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError("File not found")
    # Read file
    df = pd.read_csv(data_path)
    # Set text column to email and label column to label
    text_column = "message"
    label_column = "label"
    # Error check if either of message or label do not work
    if text_column not in df.columns or label_column not in df.columns:
        print("CSV must contain columns message and label")
        print("Found columns: ", list(df.columns))
        return
    # Clean emails
    df = preprocess_dataframe(df, text_column=text_column, label_column=label_column)
    # Get texts and labels from the corresponding columns
    texts = df[text_column]
    labels = df[label_column]

    # NOTE:
    # The variable name returned by build_tfidf_features is X_test.
    # The previous code used X_text which would cause a runtime error.
    # Corrected here to use X_test.

    # Convert text into numbers (TF-IDF)
    X_train, X_test, y_train, y_test, vectorizer = build_tfidf_features(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        max_features=20000,
        ngram_range=(1,2),
    )

    # -------------------------------------------------------
    # CHANGE:
    # Added Naive Bayes and Logistic Regression models
    # -------------------------------------------------------
    # Model is set
    models = {
        "SVM": LinearSVC(),
        "NaiveBayes": MultinomialNB(),
        "LogisticRegression": LogisticRegression(max_iter=1000)
    }

    trained_models = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        # Model can learn patterns
        model.fit(X_train, y_train)
        # Print the results
        results = evaluate(model, X_test, y_test)
        print(f"\n{name} Results:")
        print_evaluation(results)
        trained_models[name] = model

    # NOTE:
    # Same here; changing X_text to X_test
    # But if it was intentional, feel free to revert, all I did was 
    # Change X_text to X_test.
    results = evaluate(model, X_test, y_test)
    print_evaluation(results)
    os.makedirs("results", exist_ok=True)

    # -------------------------------------------------------
    # CHANGE:
    # Saving trained Naive Bayes and Logistic Regression models here
    # -------------------------------------------------------
    # Saves trained model and vectorizer for TF-IDF
    joblib.dump(trained_models["SVM"], "models/spam_model.pkl")
    joblib.dump(trained_models["NaiveBayes"], "models/naive_bayes_model.pkl")
    joblib.dump(trained_models["LogisticRegression"], "models/logistic_regression_model.pkl")
    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

    print("Saved model and vectorizer:")
    print(" - models/spam_model.pkl (SVM used by app)")
    print(" - models/naive_bayes_model.pkl")
    print(" - models/logistic_regression_model.pkl")
    print(" - models/tfidf_vectorizer.pkl")

if __name__ == "__main__": # this needs to be aligned with the main()
    main()
