from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("customer_churn_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    age = int(request.form["age"])
    gender = int(request.form["gender"])
    tenure = int(request.form["tenure"])
    usage_frequency = int(request.form["usage_frequency"])
    support_calls = int(request.form["support_calls"])
    payment_delay = int(request.form["payment_delay"])
    subscription_type = int(request.form["subscription_type"])
    contract_length = int(request.form["contract_length"])
    total_spend = int(request.form["total_spend"])
    last_interaction = int(request.form["last_interaction"])

    features = np.array([[
        age,
        gender,
        tenure,
        usage_frequency,
        support_calls,
        payment_delay,
        subscription_type,
        contract_length,
        total_spend,
        last_interaction
    ]])

    prediction = model.predict(features)[0]

    result = "Customer is likely to Churn" if prediction == 1 else "Customer is likely to Stay"

    return render_template(
        "index.html",
        prediction_text=result
    )

if __name__ == "__main__":
    app.run(debug=True)