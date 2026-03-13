import os
import joblib
import pandas as pd
from sklearn.svm import LinearSVC
from preprocessing import preprocess_dataframe
from feature_engineering import build_tfidf_features
from evaluation import evaluate, print_evaluation

def main():
    data_path = "spam.csv"

    if not os.path.exists(data_path):
        raise FileNotFoundError("File not found")
    df = pd.read_csv(data_path)
    text_column = "message"
    label_column = "label"
    if text_column not in df.columns or label_column not in df.columns:
        print("CSV must contain columns message and label")
        print("Found columns: ", list(df.columns))
        return
    df = preprocess_dataframe(df, text_column=text_column, label_column=label_column)
    texts = df[text_column]
    labels = df[label_column]

    X_train, X_text, y_train, y_test, vectorizer = build_tfidf_features(
        texts,
        labels,
        test_size=0.2,
        random_state=42,
        max_features=20000,
        ngram_range=(1,2),
    )

    model = LinearSVC()
    model.fit(X_train, y_train)

    results = evaluate(model, X_text, y_test)
    print_evaluation(results)
    os.makedirs("results", exist_ok=True)
    joblib.dump(model, "models/spam_model.pkl")
    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
    print("Saved model and vectorizer: models/spam_model.pkl and models/tfidf_vectorizer.pkl")

    if __name__ == "__main__":
        main()
