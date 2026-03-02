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