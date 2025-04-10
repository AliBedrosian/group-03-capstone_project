

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import joblib
import pandas as pd
import numpy as np


tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")


xgboost_model = joblib.load("model_pipeline.pkl")


def analyze_sentiment(text):
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        scores = outputs.logits[0].softmax(dim=0).numpy()

    pos_score = float(scores[2])
    neg_score = float(scores[0])
    neu_score = float(scores[1])

    
    from textblob import TextBlob
    polarity = float(TextBlob(text).sentiment.polarity)  
    engagement_score = 50000  

    return pos_score, neg_score, neu_score, polarity, engagement_score



def predict_stock_change(pos, neg, neu, polarity, engagement):
    
    X = pd.DataFrame([[pos, neg, neu, polarity, engagement]],
                     columns=["roberta_pos_score", "roberta_neg_score", "roberta_neu_score",
                              "sentiment_polarity", "engagement_score"])
    prediction = xgboost_model.predict(X)[0]
    return round(float(prediction), 2)

