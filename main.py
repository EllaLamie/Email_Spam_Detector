from feature_engineering import build_tfidf_features
from models.svm import train_svm, predict_svm

texts = [
    "WIN a free iPhone now!!!",
    "Hey are we still meeting at 3?",
    "Congratulations you won cash prize",
    "Can you send me the homework?"
]
labels = [1, 0, 1, 0]  # 1=spam, 0=ham

X_train, X_test, y_train, y_test, vec = build_tfidf_features(
    texts, labels, test_size=0.5
)

model = train_svm(X_train, y_train, C=1)
preds = predict_svm(model, X_test)

print("preds:", preds)
print("y_test:", y_test)