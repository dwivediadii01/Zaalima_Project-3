
import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

INPUT = "cleaned_data.csv"
OUTPUT = "sentiment_results.csv"
TEXT_COLUMN = "cleaned_text"  # always created by preprocess.py

df = pd.read_csv(INPUT, low_memory=False)
analyzer = SentimentIntensityAnalyzer()


# Step 1: TextBlob polarity

def textblob_score(text):
    try:
        return TextBlob(str(text)).sentiment.polarity
    except:
        return 0.0


# Step 2: VADER compound score

def vader_score(text):
    try:
        return analyzer.polarity_scores(str(text))["compound"]
    except:
        return 0.0

df["polarity_textblob"] = df[TEXT_COLUMN].apply(textblob_score)
df["vader_compound"] = df[TEXT_COLUMN].apply(vader_score)


# Step 3: Convert scores into labels

def classify(score, pos=0.05, neg=-0.05):
    if score >= pos:
        return "positive"
    if score <= neg:
        return "negative"
    return "neutral"

df["label_textblob"] = df["polarity_textblob"].apply(classify)
df["label_vader"] = df["vader_compound"].apply(classify)

# Save final results
df.to_csv(OUTPUT, index=False)
print("Saved:", OUTPUT)
