"""
naive_bayes.py

Train and use a Multinomial Naive Bayes classifier for email spam detection.
Includes simple hyperparameter tuning using GridSearchCV.
"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV


def train_naive_bayes(X_train, y_train, alpha_values=None):
    """
    Train a Multinomial Naive Bayes classifier.

    Parameters
    ----------
    X_train : sparse matrix
        TF-IDF feature matrix for training emails.
    y_train : array-like
        Corresponding labels for training data ("spam" or "ham").
    alpha_values : list, optional
        List of smoothing values to test during hyperparameter tuning.

    Returns
    -------
    best_model : MultinomialNB
        The trained Naive Bayes model with the best hyperparameters.
    best_params : dict
        The best alpha value found during tuning.
    """

    # Default smoothing parameters if none are provided
    if alpha_values is None:
        alpha_values = [0.1, 0.5, 1.0]

    # Hyperparameter search space
    param_grid = {
        "alpha": alpha_values
    }

    # GridSearchCV performs cross-validation to find the best alpha value
    grid = GridSearchCV(
        estimator=MultinomialNB(),
        param_grid=param_grid,
        cv=3,              # 3-fold cross validation
        scoring="f1",      # optimize for F1 score (important for spam detection)
        n_jobs=-1          # use all CPU cores
    )

    # Train models with different alpha values
    grid.fit(X_train, y_train)

    # Return the best performing model and its parameters
    return grid.best_estimator_, grid.best_params_


def predict_naive_bayes(model, X_test):
    """
    Predict labels for new data using a trained Naive Bayes model.

    Parameters
    ----------
    model : MultinomialNB
        Trained Naive Bayes model.
    X_test : sparse matrix
        TF-IDF feature matrix for test emails.

    Returns
    -------
    predictions : array
        Predicted labels ("spam" or "ham").
    """

    # Generate predictions
    return model.predict(X_test)