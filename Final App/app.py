from flask import Flask, render_template, request
from model_helper import analyze_sentiment, predict_stock_change

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

            
            pos, neg, neu, polarity, engagement = analyze_sentiment(tweet_text)
            print("Scores → Pos:", pos, "Neg:", neg, "Neu:", neu)
            print("Polarity:", polarity, "Engagement:", engagement)

            prediction_raw = predict_stock_change(pos, neg, neu, polarity, engagement)
            print("Prediction:", prediction_raw)

            prediction = f"{prediction_raw:+.2f}%"
        except Exception as e:
            print("ERROR:", e)
            prediction = "Something went wrong."

    return render_template("predictor.html", prediction=prediction)


@app.route("/dashboard1")
def show_dashboard1():
    return render_template("dashboard1.html")

@app.route("/dashboard2")
def show_dashboard2():
    return render_template("dashboard2.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/works-cited")
def works_cited():
    return render_template("works_cited.html")

@app.route("/writeup")
def writeup():
    return render_template("writeup.html")


if __name__ == "__main__":
    app.run(debug=True)
