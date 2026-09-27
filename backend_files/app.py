# SuperKart Sales Forecast - Flask backend API
import logging

import joblib
import pandas as pd
from flask import Flask, jsonify, request

# ---------- configuration ----------
MODEL_PATH = "superkart_model.joblib"
REFERENCE_YEAR = 2025   # must match the value used during training
PERISHABLES = ["Dairy", "Meat", "Fruits and Vegetables", "Breakfast", "Breads", "Seafood"]
FEATURES = [
    "Product_Weight", "Product_Sugar_Content", "Product_Allocated_Area", "Product_MRP",
    "Store_Size", "Store_Location_City_Type", "Store_Type",
    "Store_Age_Years", "Product_Type_Category", "Product_Id_char",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("superkart_api")

# ---------- app + model ----------
superkart_api = Flask("SuperKart Sales Forecast API")
superkart_api.json.sort_keys = False   # keep batch results in row order
model = joblib.load(MODEL_PATH)   # full pipeline: preprocessing + regressor
logger.info("Model loaded from %s", MODEL_PATH)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive engineered features from raw columns when needed and return the model columns."""
    df = df.copy()
    if "Product_Sugar_Content" in df:
        df["Product_Sugar_Content"] = df["Product_Sugar_Content"].replace({"reg": "Regular"})
    if "Store_Age_Years" not in df and "Store_Establishment_Year" in df:
        df["Store_Age_Years"] = REFERENCE_YEAR - df["Store_Establishment_Year"].astype(int)
    if "Product_Type_Category" not in df and "Product_Type" in df:
        df["Product_Type_Category"] = df["Product_Type"].apply(
            lambda t: "Perishables" if t in PERISHABLES else "Non Perishables")
    if "Product_Id_char" not in df and "Product_Id" in df:
        df["Product_Id_char"] = df["Product_Id"].astype(str).str[:2]

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required fields: {missing}")
    return df[FEATURES]


@superkart_api.get("/")
def home():
    """Health check / welcome route."""
    return "Welcome to the SuperKart Sales Forecast API! Use POST /v1/predict or /v1/predictbatch."


@superkart_api.post("/v1/predict")
def predict_sales():
    """Online inference: JSON payload for one product-store combination -> predicted sales."""
    try:
        payload = request.get_json(force=True)
        features = prepare_features(pd.DataFrame([payload]))
        prediction = float(model.predict(features)[0])
        logger.info("Single prediction: %.2f", prediction)
        return jsonify({"Predicted_Sales": round(prediction, 2)})
    except KeyError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:  # any other failure
        logger.exception("Prediction failed")
        return jsonify({"error": f"Prediction failed: {e}"}), 500


@superkart_api.post("/v1/predictbatch")
def predict_sales_batch():
    """Batch inference: CSV file (form field 'file') -> {row_index: predicted_sales}."""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded under the key 'file'"}), 400
        batch = pd.read_csv(request.files["file"])
        features = prepare_features(batch)
        predictions = [round(float(p), 2) for p in model.predict(features)]
        logger.info("Batch prediction for %d rows", len(predictions))
        return jsonify({str(i): p for i, p in enumerate(predictions)})
    except KeyError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception("Batch prediction failed")
        return jsonify({"error": f"Batch prediction failed: {e}"}), 500


if __name__ == "__main__":
    # Local run (inside Docker, gunicorn is used instead - see Dockerfile)
    superkart_api.run(host="0.0.0.0", port=7860, debug=False)
