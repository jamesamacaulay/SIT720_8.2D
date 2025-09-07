# app.py
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="House Price Demo", page_icon="🏠", layout="wide")

# --- load model artifact ---
ART = joblib.load("house_price_rf_pipeline.joblib")
pipe = ART["pipeline"]
FEATURES = ART["feature_order"]  # list of raw columns expected by the pipeline

st.title("🏠 Housing Price Predictor (demo)")
st.write("Enter property details and get a predicted price. This demo uses the tuned Random Forest pipeline.")

# ---- Build inputs (keep it short but useful) ----
# Choose a handful of important features for the UI. We'll fill the rest as NaN and let the pipeline imputer handle them.
col1, col2, col3 = st.columns(3)

with col1:
    beds   = st.number_input("Bedrooms", 0, 12, 3, step=1)
    baths  = st.number_input("Bathrooms", 0, 8, 2, step=1)
    parking= st.number_input("Parking spaces", 0, 8, 1, step=1)

with col2:
    land_size       = st.number_input("Land size (m²)", 0.0, 200000.0, 400.0, step=10.0)
    distance_to_cbd = st.number_input("Distance to CBD (km)", 0.0, 500.0, 10.0, step=0.5)
    # rooms_total is AUTO (see below)

with col3:
    nearest_supermarket = st.number_input("Nearest supermarket (km)", 0.0, 200.0, 0.8, step=0.1)
    nearest_train       = st.number_input("Nearest train (km)", 0.0, 200.0, 1.2, step=0.1)
    nearest_bus         = st.number_input("Nearest bus (km)", 0.0, 200.0, 0.3, step=0.1)
    nearest_park        = st.number_input("Nearest park (km)", 0.0, 200.0, 0.5, step=0.1)

# Dropdowns that will be mapped to one-hot flags
property_type = st.selectbox(
    "Property type",
    ["House", "Apartment / Unit / Flat / Studio", "Townhouse", "Villa", "Land"],
    index=0
)

sale_method = st.selectbox(
    "Sale method",
    ["Private treaty", "Auction", "Prior auction", "Active (listed)"],
    index=0
)
rooms_total = int(beds + baths + parking)
amenity_access_index = (
    1/(1e-6 + nearest_supermarket) +
    1/(1e-6 + nearest_train) +
    1/(1e-6 + nearest_bus) +
    1/(1e-6 + nearest_park)
)
st.caption(f"Computed rooms_total = **{rooms_total}**  •  Amenity Access Index ≈ **{amenity_access_index:.2f}**")
# Map dropdown choices to the one-hot column names used in training
PT_COLS = {
    "House": "property_type_house",
    "Apartment / Unit / Flat / Studio": "property_type_apartment_unit_flat_studio",
    "Townhouse": "property_type_townhouse",
    "Villa": "property_type_villa",
    "Land": "property_type_land",
}
SM_COLS = {
    "Private treaty": "sale_method_private_treaty",
    "Auction": "sale_method_auction",
    "Prior auction": "sale_method_prior_auction",
    "Active (listed)": "sale_method_active",
}

def put_if_present(d, name, value):
    """Set d[name] = value only if the model expects that column."""
    if name in d:
        d[name] = value

def build_input_row() -> pd.DataFrame:
    # Start with all features present as NaN so your pipeline's imputer can fill the rest.
    data = {c: np.nan for c in FEATURES}

    # Basic numeric fields
    put_if_present(data, "beds", beds)
    put_if_present(data, "baths", baths)
    put_if_present(data, "parking", parking)
    put_if_present(data, "land_size", land_size)
    put_if_present(data, "distance_to_cbd", distance_to_cbd)
    put_if_present(data, "rooms_total", rooms_total)

    # Distances & engineered index
    put_if_present(data, "nearest_supermarket", nearest_supermarket)
    put_if_present(data, "nearest_train", nearest_train)
    put_if_present(data, "nearest_bus", nearest_bus)
    put_if_present(data, "nearest_park", nearest_park)
    put_if_present(data, "amenity_access_index", amenity_access_index)

    # One-hot: set all related columns to 0, then the selected one to 1
    for col in PT_COLS.values():
        put_if_present(data, col, 0)
    sel_pt_col = PT_COLS[property_type]
    put_if_present(data, sel_pt_col, 1)

    for col in SM_COLS.values():
        put_if_present(data, col, 0)
    sel_sm_col = SM_COLS[sale_method]
    put_if_present(data, sel_sm_col, 1)

    # Build DF in the exact training column order
    return pd.DataFrame([data], columns=FEATURES)

# ---------------- Predict ----------------
st.markdown("---")
if st.button("Predict price"):
    X_row = build_input_row()
    yhat = pipe.predict(X_row)[0]
    st.success(f"Estimated price: **A${yhat:,.0f}**")

    with st.expander("Show model input (non-NaN fields)"):
        show = {k: v for k, v in X_row.iloc[0].items() if pd.notna(v)}
        st.write(pd.DataFrame([show]).T.rename(columns={0: "value"}))