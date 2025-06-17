# preprocess.py
import pandas as pd
import re
import nltk
import joblib
import logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocess.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logging.info("🚀 Starting preprocessing...")

nltk.download('punkt')
nltk.download('stopwords')

# Text cleaning
stop_words = set(stopwords.words('english'))

# Load dataset
try:
    df = pd.read_csv("movies.csv")
    logging.info("✅ Dataset loaded successfully. Total rows: %d", len(df))
except Exception as e:
    logging.error("❌ Failed to load dataset: %s", str(e))
    raise e

def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", str(text))
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Keep required columns
required_columns = ["genres", "keywords", "overview", "title"]
df = df[required_columns].dropna().reset_index(drop=True)
df['combined'] = df['genres'] + ' ' + df['keywords'] + ' ' + df['overview']

logging.info("🧹 Cleaning text...")
df['cleaned_text'] = df['combined'].apply(preprocess_text)
logging.info("✅ Text cleaned.")

# Save cleaned data only
joblib.dump(df, 'df_cleaned.pkl')
logging.info("💾 df_cleaned.pkl saved.")
logging.info("✅ Preprocessing complete.")
