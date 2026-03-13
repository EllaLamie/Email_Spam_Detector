"""
logistic_regression.py

Train and use a Logistic Regression classifier for email spam detection.
Includes simple hyperparameter tuning using GridSearchCV.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


def train_logistic_regression(X_train, y_train, c_values=None):
    """
    Train a Logistic Regression classifier.

    Parameters
    ----------
    X_train : sparse matrix
        TF-IDF feature matrix for training emails.
    y_train : array-like
        Corresponding labels for training data ("spam" or "ham").
    c_values : list, optional
        Regularization strengths to test during hyperparameter tuning.

    Returns
    -------
    best_model : LogisticRegression
        The trained Logistic Regression model with optimal hyperparameters.
    best_params : dict
        Best parameters discovered during GridSearchCV.
    """

    # Default regularization values if none are provided
    if c_values is None:
        c_values = [0.1, 1.0, 10.0]

    # Hyperparameter search space
    param_grid = {
        "C": c_values,
        "solver": ["liblinear"]  # stable solver for smaller datasets
    }

    # GridSearchCV tests multiple parameter combinations
    grid = GridSearchCV(
        estimator=LogisticRegression(max_iter=1000),
        param_grid=param_grid,
        cv=3,            # 3-fold cross validation
        scoring="f1",    # optimize for F1 score
        n_jobs=-1
    )

    # Train the model with hyperparameter tuning
    grid.fit(X_train, y_train)

    # Return the best performing model and parameters
    return grid.best_estimator_, grid.best_params_


def predict_logistic_regression(model, X_test):
    """
    Predict labels for new emails using the trained Logistic Regression model.

    Parameters
    ----------
    model : LogisticRegression
        Trained logistic regression classifier.
    X_test : sparse matrix
        TF-IDF features for the test data.

    Returns
    -------
    predictions : array
        Predicted labels ("spam" or "ham").
    """

    # Generate predictions
    return model.predict(X_test)