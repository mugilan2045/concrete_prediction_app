import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

# ---------------- Load Models ----------------
with open("best_strength_model.pkl", "rb") as f:
    strength_model = pickle.load(f)

with open("best_cost_model.pkl", "rb") as f:
    cost_model = pickle.load(f)

# ---------------- Constants ----------------
cost_cement = 6.0
cost_slag = 2.0
cost_ash = 1.5
cost_water = 0.01
cost_super = 15.0
cost_coarse = 0.8
cost_fine = 0.6

# ---------------- Helper Functions ----------------
def classify_strength(val):
    if val >= 50:
        return "Good"
    elif val >= 30:
        return "Moderate"
    else:
        return "Bad"

def preprocess_input(inputs):
    """Feature engineering and transformations for user input"""
    inputs["w/c_ratio"] = inputs["water"] / inputs["cement"]
    inputs["binder"] = inputs["cement"] + inputs["slag"] + inputs["ash"]
    inputs["cement_ratio"] = inputs["cement"] / inputs["binder"]
    inputs["slag_ratio"] = inputs["slag"] / inputs["binder"]
    inputs["binder_agg_ratio"] = inputs["binder"] / (inputs["coarseagg"] + inputs["fineagg"])
    
    # Apply same transforms
    inputs["slag"] = np.sqrt(inputs["slag"])
    inputs["superplastic"] = np.sqrt(inputs["superplastic"])
    
    # cost (reverting sqrt for real values)
    inputs["cost"] = (
        inputs["cement"] * cost_cement +
        (inputs["slag"]**2) * cost_slag +
        inputs["ash"] * cost_ash +
        inputs["water"] * cost_water +
        (inputs["superplastic"]**2) * cost_super +
        inputs["coarseagg"] * cost_coarse +
        inputs["fineagg"] * cost_fine
    )
    
    df = pd.DataFrame([inputs])
    df.drop(["water", "cement", "binder"], axis=1, inplace=True)
    return df

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Concrete Strength & Cost Predictor", layout="wide")
st.title("🧱 Concrete Strength & Cost Prediction App")

col1, col2 = st.columns(2)

with col1:
    slag = st.number_input("Slag (kg/m³)", 0.0, 300.0, 50.0)
    ash = st.number_input("Fly Ash (kg/m³)", 0.0, 300.0, 30.0)
    superplastic = st.number_input("Superplasticizer (kg/m³)", 0.0, 30.0, 5.0)
    coarseagg = st.number_input("Coarse Aggregate (kg/m³)", 500.0, 1200.0, 800.0)

with col2:
    fineagg = st.number_input("Fine Aggregate (kg/m³)", 200.0, 1200.0, 700.0)
    water = st.number_input("Water (kg/m³)", 100.0, 300.0, 180.0)
    cement = st.number_input("Cement (kg/m³)", 100.0, 600.0, 350.0)
    age = st.text_input("Age (days)", "28")

if st.button("🔮 Predict"):
    user_input = {
        "slag": slag,
        "ash": ash,
        "superplastic": superplastic,
        "coarseagg": coarseagg,
        "fineagg": fineagg,
        "water": water,
        "cement": cement,
        "age": age
    }
    
    processed = preprocess_input(user_input)
    pred_strength = strength_model.predict(processed.drop("cost", axis=1))[0]
    pred_cost = cost_model.predict(processed.drop("cost", axis=1))[0]
    pred_class = classify_strength(pred_strength)
    
    st.subheader("✅ Predictions")
    st.write(f"**Predicted Strength:** {pred_strength:.2f} MPa ({pred_class})")
    st.write(f"**Predicted Cost:** {pred_cost:.2f} currency units/m³")

    # ----------- SHAP Explainability -----------
    st.subheader("📊 Explainable AI (SHAP Feature Impact)")
    explainer = shap.TreeExplainer(strength_model["regressor"])  # get model from pipeline
    shap_values = explainer.shap_values(processed.drop("cost", axis=1))
    
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, processed.drop("cost", axis=1), plot_type="bar", show=False)
    st.pyplot(fig)
