import re
import pandas as pd

# Takes an email and cleans it up to be read
def clean_text(text: str) -> str:
    # Make sure the input is a string
    if not isinstance(text, str):
        text = str(text)
    # Converts everything to lowercase
    text = text.lower()
    # Remove links
    text = re.sub(r"http\S+|www\S+", " ", text)
    # Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)
    # Remove everything but letters and spaces
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    # Removes extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text
# Processes and cleans the dataset
def preprocess_dataframe(df: pd.DataFrame, text_column: str, label_column: str) -> pd.DataFrame:
    # Keep only the email and the label
    df = df[[text_column, label_column]].copy()
    # Call clean_text for every email
    df[text_column] = df[text_column].fillna("").apply(clean_text)
    # Remove empty emails
    df = df[df[text_column].str.strip() != ""]
    return df
