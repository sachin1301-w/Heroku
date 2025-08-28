from flask import Flask, render_template, request
import pandas as pd
import joblib

# Load trained model
model = joblib.load("spot_the_fake_model.pkl")

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    reason = None

    if request.method == "POST":
        url = request.form["url"]
        title = request.form["title"]
        html = request.form["html"]

        # Convert to dataframe for prediction
        sample = pd.DataFrame([{"url": url, "title": title, "html": html}])
        pred = model.predict(sample)[0]

        prediction = "⚠️ Suspicious / Fake" if pred == 1 else "✅ Legitimate"
        
        # Optional "explainability" (very simple here)
        reason = "Suspicious patterns detected in URL/title/html." if pred == 1 else "Content looks safe."

    return render_template("index.html", prediction=prediction, reason=reason)

if __name__ == "__main__":
    app.run(debug=True)
