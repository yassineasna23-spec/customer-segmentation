"""Flask web app for mall customer segmentation with a trained K-Means model."""

import logging
import math
import os
from pathlib import Path

import joblib
from flask import Flask, flash, redirect, render_template, request, url_for

BASE_DIR = Path(__file__).resolve().parent
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Set FLASK_SECRET_KEY in your environment for a stable, private secret.
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or "local-development-key-change-me"

try:
    model = joblib.load(BASE_DIR / "kmeans_model.pkl")
    scaler = joblib.load(BASE_DIR / "scaler.pkl")
except (FileNotFoundError, OSError, ValueError) as exc:
    logger.exception("Could not load the trained model artifacts: %s", exc)
    model = None
    scaler = None

# Names are based on the cluster averages in Mall_Customers.csv.
CLUSTER_DETAILS = {
    0: {
        "name": "Moderate-Income, Moderate-Spending Customers",
        "advice": "Test seasonal campaigns, useful bundles, and loyalty offers; compare campaign results before scaling.",
    },
    1: {
        "name": "High-Income, High-Spending Customers",
        "advice": "Consider VIP loyalty benefits, early access, and personalized premium offers.",
    },
    2: {
        "name": "Lower-Income, High-Spending Customers",
        "advice": "Test trend-focused products and accessible promotions while monitoring affordability and response.",
    },
    3: {
        "name": "High-Income, Low-Spending Customers",
        "advice": "Test product discovery, service benefits, and targeted recommendations to understand barriers to spending.",
    },
    4: {
        "name": "Lower-Income, Low-Spending Customers",
        "advice": "Consider value-focused offers and discounts, and evaluate whether they improve engagement.",
    },
}


@app.get("/")
def home():
    return render_template("index.html", result=None)


@app.post("/predict")
def predict():
    if model is None or scaler is None:
        logger.error("Prediction requested while model artifacts are unavailable.")
        flash("The model is unavailable. Check that model files are present and readable.", "danger")
        return redirect(url_for("home"))

    raw_income = request.form.get("income", "").strip()
    raw_score = request.form.get("score", "").strip()
    if not raw_income or not raw_score:
        flash("Please enter both annual income and spending score.", "warning")
        return redirect(url_for("home"))

    try:
        income = float(raw_income)
        score = float(raw_score)
    except ValueError:
        flash("Please enter valid numeric values.", "danger")
        return redirect(url_for("home"))

    if not math.isfinite(income) or not math.isfinite(score):
        flash("Values must be finite numbers.", "warning")
        return redirect(url_for("home"))
    if income < 0:
        flash("Annual income cannot be negative.", "warning")
        return redirect(url_for("home"))
    if not 1 <= score <= 100:
        flash("Spending score must be between 1 and 100.", "warning")
        return redirect(url_for("home"))

    try:
        scaled_values = scaler.transform([[income, score]])
        cluster_id = int(model.predict(scaled_values)[0])
        details = CLUSTER_DETAILS.get(cluster_id)
        if details is None:
            raise ValueError(f"No display information configured for cluster {cluster_id}.")

        result = {
            "income": income,
            "score": score,
            "cluster": cluster_id,
            "segment": details["name"],
            "advice": details["advice"],
        }
        return render_template("index.html", result=result)
    except Exception:
        logger.exception("Prediction failed.")
        flash("The prediction could not be completed. Check the model files and dependencies.", "danger")
        return redirect(url_for("home"))


if __name__ == "__main__":
    # Debug mode is opt-in and intended only for local development.
    debug_enabled = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=debug_enabled)
