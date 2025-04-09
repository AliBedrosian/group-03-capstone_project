# sentiment_model.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import joblib
import pandas as pd
import numpy as np

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

# Load your trained regression model (trained to predict % change from sentiment score)
regression_model = joblib.load("model.pkl")


def analyze_sentiment(text):
    """Returns RoBERTa sentiment + placeholder polarity and engagement"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        scores = outputs.logits[0].softmax(dim=0).numpy()

    pos_score = float(scores[2])
    neg_score = float(scores[0])
    neu_score = float(scores[1])

    # Simulate polarity and engagement behind the scenes
    from textblob import TextBlob
    polarity = float(TextBlob(text).sentiment.polarity)  # range: [-1, 1]
    engagement_score = 50000  # you can randomize this later if you want

    return pos_score, neg_score, neu_score, polarity, engagement_score



def predict_stock_change(pos, neg, neu, polarity, engagement):
    """Predict using 5 features"""
    X = pd.DataFrame([[pos, neg, neu, polarity, engagement]],
                     columns=["roberta_pos_score", "roberta_neg_score", "roberta_neu_score",
                              "sentiment_polarity", "engagement_score"])
    prediction = regression_model.predict(X)[0]
    return round(float(prediction), 2)

