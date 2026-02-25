import pandas as pd
import re
import nltk
import os
#from google.colab import drive
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# 1. Setup Environment
#drive.mount('/content/drive')

output_folder = '../data'
os.makedirs(output_folder, exist_ok=True)

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

file_path = '../data/fake reviews dataset.csv'

# 2. Load and Initial Cleaning
try:
    # Standard CSV load
    df = pd.read_csv(file_path)

    # Standardize column names
    if 'text_' in df.columns:
        df.rename(columns={'text_': 'text'}, inplace=True)
    df.columns = df.columns.str.lower()

    # Map labels to binary (OR=Real=0, CG=Fake=1)
    if df['label'].dtype == 'object':
        df['label'] = df['label'].map({'OR': 0, 'CG': 1})

    # Initial null check
    initial_shape = df.shape
    df = df.dropna(subset=['text', 'label'])

    # Ensure text column is string type
    df['text'] = df['text'].astype(str)

    print(f"Initial Data Loaded. Shape: {df.shape}")
    print(f"Rows dropped due to nulls: {initial_shape[0] - df.shape[0]}")

except Exception as e:
    print(f"Error loading file: {e}")
    raise

# 3. Define Cleaning Functions
def clean_for_regression(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    cleaned_words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(cleaned_words)

def clean_for_lstm(text):
    text = str(text).lower()
    # Keeping only alphanumeric is fine, but consider keeping punctuation
    # if you want the model to detect "!!!" vs "." patterns.
    # For now, your logic is valid for a word-based LSTM:
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def clean_for_transformer(text):
    text = str(text)
    # TWEAK: Replace internal newlines/tabs with a single space
    # Transformers hate random \n characters in the middle of sentences
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def perform_sanity_check(df, name):
    print(f"\n--- Sanity Check: {name} ---")

    # Check for nulls
    null_count = df.isnull().sum().sum()
    if null_count > 0:
        print(f"WARNING: Found {null_count} null values. Dropping...")
        df = df.dropna()

    # Check for empty strings
    empty_strings = df[df['text'].str.strip() == ''].shape[0]
    if empty_strings > 0:
        print(f"WARNING: Found {empty_strings} empty strings after cleaning. Dropping...")
        df = df[df['text'].str.strip().str.len() > 0]

    print(f"Final Valid Shape: {df.shape}")
    print(f"Label Distribution:\n{df['label'].value_counts()}")
    return df

# 4. Process and Save Datasets

# --- Regression Data ---
print("Processing Regression Data...")
df_reg = df.copy()
df_reg['text'] = df_reg['text'].apply(clean_for_regression)
df_reg = perform_sanity_check(df_reg, "Regression Data")
df_reg.to_csv(os.path.join(output_folder, 'data_for_regression.csv'), index=False)

# --- LSTM Data ---
print("Processing LSTM Data...")
df_lstm = df.copy()
df_lstm['text'] = df_lstm['text'].apply(clean_for_lstm)
df_lstm = perform_sanity_check(df_lstm, "LSTM Data")
df_lstm.to_csv(os.path.join(output_folder, 'data_for_lstm.csv'), index=False)

# --- Transformer Data ---
print("Processing Transformer Data...")
df_bert = df.copy()
df_bert['text'] = df_bert['text'].apply(clean_for_transformer)
df_bert = perform_sanity_check(df_bert, "Transformer Data")
df_bert.to_csv(os.path.join(output_folder, 'data_for_transformer.csv'), index=False)

print("\nAll files saved successfully.")