# ============================================================
# SuperKart Sales Forecast - Streamlit frontend
# ============================================================
import os

import pandas as pd
import requests
import streamlit as st

# Backend address: container name on the shared Docker network (overridable via env var)
BACKEND_URL = os.getenv("BACKEND_URL", "http://superkart-backend:7860")
REFERENCE_YEAR = 2025
PERISHABLES = ["Dairy", "Meat", "Fruits and Vegetables", "Breakfast", "Breads", "Seafood"]
PRODUCT_TYPES = ["Fruits and Vegetables", "Snack Foods", "Frozen Foods", "Dairy", "Household",
                 "Baking Goods", "Canned", "Health and Hygiene", "Meat", "Soft Drinks", "Breads",
                 "Hard Drinks", "Others", "Starchy Foods", "Breakfast", "Seafood"]

st.set_page_config(page_title="SuperKart Sales Forecast", page_icon="🛒", layout="centered")
st.title("🛒 SuperKart Sales Forecast")
st.caption("Predict the total sales of a product in a SuperKart store.")

tab_single, tab_batch = st.tabs(["Single prediction", "Batch prediction"])

# ---------------- Online (single) inference ----------------
with tab_single:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Product")
        product_id_char = st.selectbox("Product ID prefix", ["FD", "DR", "NC"],
                                       help="FD = food, DR = drinks, NC = non-consumables")
        product_type = st.selectbox("Product type", PRODUCT_TYPES)
        sugar = st.selectbox("Sugar content", ["Low Sugar", "Regular", "No Sugar"])
        weight = st.number_input("Product weight", min_value=0.0, max_value=50.0, value=12.66, step=0.1)
        mrp = st.number_input("Product MRP", min_value=0.0, max_value=500.0, value=147.0, step=1.0)
        area = st.number_input("Allocated display area (ratio)", min_value=0.0, max_value=1.0,
                               value=0.056, step=0.001, format="%.3f")
    with col2:
        st.subheader("Store")
        store_type = st.selectbox("Store type", ["Supermarket Type2", "Supermarket Type1",
                                                 "Departmental Store", "Food Mart"])
        store_size = st.selectbox("Store size", ["Medium", "High", "Small"])
        city_type = st.selectbox("City tier", ["Tier 2", "Tier 1", "Tier 3"])
        est_year = st.number_input("Store establishment year", min_value=1950,
                                   max_value=REFERENCE_YEAR, value=2009, step=1)

    if st.button("Predict sales", type="primary"):
        payload = {
            "Product_Weight": weight,
            "Product_Sugar_Content": sugar,
            "Product_Allocated_Area": area,
            "Product_MRP": mrp,
            "Store_Size": store_size,
            "Store_Location_City_Type": city_type,
            "Store_Type": store_type,
            "Product_Id_char": product_id_char,
            "Store_Age_Years": int(REFERENCE_YEAR - est_year),
            "Product_Type_Category": "Perishables" if product_type in PERISHABLES else "Non Perishables",
        }
        try:
            resp = requests.post(f"{BACKEND_URL}/v1/predict", json=payload, timeout=30)
            if resp.status_code == 200:
                st.success(f"Predicted sales: **{resp.json()['Predicted_Sales']:,.2f}**")
            else:
                st.error(f"API error {resp.status_code}: {resp.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach the backend at {BACKEND_URL}: {e}")

# ---------------- Batch inference ----------------
with tab_batch:
    st.write("Upload a CSV with either the engineered model features or the raw SuperKart columns.")
    uploaded = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
        st.dataframe(batch_df.head())
        if st.button("Predict for all rows", type="primary"):
            try:
                resp = requests.post(f"{BACKEND_URL}/v1/predictbatch",
                                     files={"file": uploaded.getvalue()}, timeout=120)
                if resp.status_code == 200:
                    preds = resp.json()
                    batch_df["Predicted_Sales"] = [preds[str(i)] for i in range(len(batch_df))]
                    st.success(f"Predicted {len(batch_df)} rows. Total forecast: {batch_df['Predicted_Sales'].sum():,.2f}")
                    st.dataframe(batch_df)
                    st.download_button("Download predictions", batch_df.to_csv(index=False),
                                       file_name="superkart_predictions.csv", mime="text/csv")
                else:
                    st.error(f"API error {resp.status_code}: {resp.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Could not reach the backend at {BACKEND_URL}: {e}")
