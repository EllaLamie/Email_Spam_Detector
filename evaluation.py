from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Evaluates how good our model is
def evaluate(model, X_test, y_test):
    # Predicts if the email is spam or ham
    y_pred = model.predict(X_test)
    results = {
        # Percent of correct predictions
        "accuracy": accuracy_score(y_test, y_pred),
        # Shows how many emails were correctly identified as spam
        "precision": precision_score(y_test, y_pred, pos_label="spam"),
        # Shows how many spam emails were caught
        "recall": recall_score(y_test, y_pred, pos_label="spam"),
        # Combining both precision and recall
        "f1": f1_score(y_test, y_pred, pos_label="spam"),
        # Report for the results
        "classification_report": classification_report(y_test, y_pred),
        # Matrix with number of ham, spam, mistake, etc
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }
    return results

# Print everything out
def print_evaluation(results):
    print("Model Evaluation")
    print("Accuracy: {}".format(results["accuracy"]))
    print("Precision: {}".format(results["precision"]))
    print("Recall: {}".format(results["recall"]))
    print("F1: {}".format(results["f1"]))
    print("Classification Report: {}".format(results["classification_report"]))
    print("Confusion Matrix: {}".format(results["confusion_matrix"]))