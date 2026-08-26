from flask import Flask, render_template, request
import pandas as pd
import joblib


app = Flask(__name__)


model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:

        age = request.form.get("age")
        income = request.form.get("income")
        loan = request.form.get("loan")
        debt = request.form.get("debt")
        employment = request.form.get("employment")
        history = request.form.get("history")

        if not all([
            age,
            income,
            loan,
            debt,
            employment,
            history
        ]):
            return render_template(
                "index.html",
                error="Please enter all six values."
            )


       
        age = float(age)
        income = float(income)
        loan = float(loan)
        debt = float(debt)
        employment = float(employment)
        history = float(history)


        customer = pd.DataFrame({
            "Age": [age],
            "Income": [income],
            "LoanAmount": [loan],
            "Debt": [debt],
            "EmploymentYears": [employment],
            "CreditHistory": [history]
        })
        
        prediction = model.predict(customer)[0]
        probabilities = model.predict_proba(customer)[0]
        probability = round(max(probabilities) * 100, 2)


       
        if prediction == 1:
            result = "✅ Good Credit"
        else:
            result = "❌ Bad Credit"


        
        return render_template(
            "index.html",

            prediction=result,
            probability=probability,

            age=age,
            income=income,
            loan=loan,
            debt=debt,
            employment=employment,
            history=history
        )


    except ValueError:

        return render_template(
            "index.html",
            error="Please enter valid numbers in all fields."
        )


    except Exception as e:

        print("Error:", e)

        return render_template(
            "index.html",
            error="Something went wrong. Check the Flask terminal."
        )


if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )