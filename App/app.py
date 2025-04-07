from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predictor", methods=["GET", "POST"])
def predictor():
    prediction = None
    if request.method == "POST":
        user_text = request.form["tweet_text"]
        # 👉 Replace the line below with your actual model prediction
        prediction = "+2.5%"  # Dummy output

    return render_template("predictor.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
