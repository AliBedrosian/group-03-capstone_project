from flask import Flask, render_template, request
from sentiment_model import analyze_sentiment, predict_stock_change

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predictor", methods=["GET", "POST"])
def predictor():
    prediction = None

    if request.method == "POST":
        try:
            tweet_text = request.form["tweet_text"]
            print("Tweet received:", tweet_text)

            # Updated to return all 5 features
            pos, neg, neu, polarity, engagement = analyze_sentiment(tweet_text)
            print("Scores → Pos:", pos, "Neg:", neg, "Neu:", neu)
            print("Polarity:", polarity, "Engagement:", engagement)

            prediction_raw = predict_stock_change(pos, neg, neu, polarity, engagement)
            print("Prediction:", prediction_raw)

            prediction = f"{prediction_raw:+.2f}%"
        except Exception as e:
            print("🔥 ERROR during prediction:", e)
            prediction = "Something went wrong."

    return render_template("predictor.html", prediction=prediction)







if __name__ == "__main__":
    app.run(debug=True)
