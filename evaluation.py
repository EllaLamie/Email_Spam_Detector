from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label="spam"),
        "recall": recall_score(y_test, y_pred, pos_label="spam"),
        "f1": f1_score(y_test, y_pred, pos_label="spam"),
        "classification_report": classification_report(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }
    return results

def print_evaluation(results):
    print("Model Evaluation")
    print("Accuracy: {}".format(results["accuracy"]))
    print("Precision: {}".format(results["precision"]))
    print("Recall: {}".format(results["recall"]))
    print("F1: {}".format(results["f1"]))
    print("Classification Report: {}".format(results["classification_report"]))
    print("Confusion Matrix: {}".format(results["confusion_matrix"]))