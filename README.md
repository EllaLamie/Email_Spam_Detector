# Email Spam Detector
CS 450 – Artificial Intelligence Final Project

## Team Members
- Esteban Naranjo
- Ella Lamie
- Steven
- Trinh Ho
- Kristine Alba

---

# Project Overview

This project implements a machine learning system for detecting spam emails using Natural Language Processing (NLP) techniques and supervised learning models.

The system processes raw email text, converts it into numerical features using TF-IDF vectorization, and trains multiple machine learning models to classify messages as **spam** or **not spam**.

The best-performing model is then deployed using a **Streamlit web application** that allows users to test predictions interactively.

---

# Machine Learning Pipeline

The project follows a standard machine learning workflow:

Raw Email Text  
↓  
Text Preprocessing  
↓  
TF-IDF Feature Engineering  
↓  
Machine Learning Model Training  
↓  
Model Evaluation  
↓  
Model Saving  
↓  
Streamlit Web Application

---

# Dataset

The project uses a spam email dataset stored as:

spam.csv

The dataset contains two primary fields:

| Column | Description |
|------|-------------|
| label | Indicates spam or ham |
| message | Email message text |

---

# Project Structure

Email_Spam_Detector

preprocessing.py  
Text cleaning and preparation  

feature_engineering.py  
TF-IDF vectorization and train/test split  

models/  
Saved trained models  

evaluation.py  
Model evaluation metrics  

main.py  
Training pipeline  

app.py  
Streamlit web interface  

spam.csv  
Dataset  

README.md  
Project documentation  

---

# Implemented Models

The system trains and compares multiple machine learning classifiers.

### Support Vector Machine (SVM)
A linear SVM classifier used for high-dimensional text classification.

### Naive Bayes
Multinomial Naive Bayes, commonly used for spam filtering.

### Logistic Regression
A probabilistic classifier used for binary classification.

These models are trained and evaluated within `main.py`.

---

# Feature Engineering

Text features are generated using **TF-IDF (Term Frequency–Inverse Document Frequency)**.

Configuration:

- Stop words removed
- Maximum features: 20,000
- N-grams: (1,2)

This converts email text into numerical feature vectors suitable for machine learning algorithms.

---

# Model Evaluation

Each model is evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

These metrics are implemented in `evaluation.py`.

---

# Model Saving

After training, models are saved in the `models/` directory:

models/  
spam_model.pkl  
naive_bayes_model.pkl  
logistic_regression_model.pkl  
tfidf_vectorizer.pkl  

These files allow the trained model and vectorizer to be reused without retraining.

---

# Streamlit Web Application

The project includes a simple interactive web interface built with **Streamlit**.

Users can:

1. Enter an email message  
2. Click **Predict**  
3. Receive a classification result (spam or not spam)

Run the application using:

streamlit run app.py

Then open:

http://localhost:8501

---

# Completed Project Tasks

✔ Dataset Selection & Preprocessing  
✔ TF-IDF Feature Engineering  
✔ Naive Bayes Implementation  
✔ Logistic Regression Implementation  
✔ SVM Implementation  
✔ Model Evaluation Metrics  
✔ Streamlit Deployment  

---

# Future Improvements

Possible enhancements for the system include:

- Hyperparameter tuning for model optimization
- Using larger or more diverse spam datasets
- Improving Streamlit UI and visualization
- Adding prediction probability scores
- Deploying the model to a cloud service

---

# Technologies Used

- Python
- Scikit-Learn
- Pandas
- Streamlit
- TF-IDF Vectorization
- Machine Learning Classification Algorithms

---

# Conclusion

This project demonstrates how machine learning and natural language processing can be used to automatically detect spam emails. By comparing multiple models and deploying an interactive interface, the system provides both analytical insights and a functional application.