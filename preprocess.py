# preprocess.py
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")

INPUT = "Amazon Sales Dataset.csv"
OUTPUT = "cleaned_data.csv"

df = pd.read_csv(INPUT, low_memory=False)

# auto-detect text column (priority keywords)
priority = ["review", "text", "comment", "feedback", "message", "description"]
text_col = None
cols = [c.lower() for c in df.columns]

for key in priority:
    for orig in df.columns:
        if key in orig.lower():
            text_col = orig
            break
    if text_col:
        break

# fallback: first object dtype column with many non-null strings
if not text_col:
    for orig in df.columns:
        if df[orig].dtype == object and df[orig].dropna().shape[0] > 10:
            text_col = orig
            break

if not text_col:
    raise SystemExit("No text column detected. Open the CSV and set TEXT_COLUMN manually in this script.")

print("Using text column:", text_col)
sample = df[text_col].dropna().astype(str).head(5).tolist()
print("Sample values:\n", sample)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_text(t):
    if not isinstance(t, str):
        return ""
    t = t.lower()
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"[^a-z\s]", " ", t)
    tokens = [w for w in t.split() if w not in stop_words and len(w) > 1]
    tokens = [lemmatizer.lemmatize(tok) for tok in tokens]
    return " ".join(tokens)

df["cleaned_text"] = df[text_col].apply(clean_text)
df.to_csv(OUTPUT, index=False)
print("Saved cleaned data to", OUTPUT)
