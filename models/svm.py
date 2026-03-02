from sklearn.svm import SVC

def train_svm(X_train, y_train, C=1):
    """
    Train a linear Support Vector Machine classifier.
    """
    model = SVC(kernel='linear', C=C)
    model.fit(X_train, y_train)
    return model


def predict_svm(model, X_test):
    """
    Predict labels using trained SVM model.
    """
    return model.predict(X_test)

if __name__ == "__main__":    
    import numpy as np

    # tiny fake dataset (just to prove it runs)
    X_train = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
    y_train = np.array([0, 1, 1, 0])

    model = train_svm(X_train, y_train, C=1)
    preds = predict_svm(model, X_train)

    print("SVM ran ")
    print("preds:", preds)