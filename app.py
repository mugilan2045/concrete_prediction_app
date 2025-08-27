%%writefile app.py
import streamlit as st
import pickle
import numpy as np

# Load trained model (save your ML model as pickle first)
model = pickle.load(open("cement_model.pkl", "rb"))

st.title("Cement Strength & Cost Prediction App")

cement = st.number_input("Cement (kg/m³)", 0.0, 500.0, 200.0)
slag = st.number_input("Slag (kg/m³)", 0.0, 500.0, 50.0)
ash = st.number_input("Fly Ash (kg/m³)", 0.0, 200.0, 20.0)
water = st.number_input("Water (kg/m³)", 0.0, 300.0, 180.0)
superplastic = st.number_input("Superplasticizer (kg/m³)", 0.0, 50.0, 5.0)
coarseagg = st.number_input("Coarse Aggregate (kg/m³)", 0.0, 1200.0, 900.0)
fineagg = st.number_input("Fine Aggregate (kg/m³)", 0.0, 1000.0, 800.0)
age = st.number_input("Age of Concrete (days)", 1, 365, 28)

if st.button("Predict"):
    X = np.array([[cement, slag, ash, water, superplastic, coarseagg, fineagg, age]])
    prediction = model.predict(X)[0]
    cost = 0.25*cement + 0.15*slag + 0.10*ash + 0.05*superplastic + 0.07*water + 0.08*coarseagg + 0.09*fineagg
    
    st.subheader(f"Predicted Strength: {prediction:.2f} MPa")
    st.subheader(f"Estimated Cost: ₹{cost:.2f}")

    # Strength Classification
    if prediction >= 40:
        st.success("Strength Quality: Good ✅")
    elif 25 <= prediction < 40:
        st.warning("Strength Quality: Moderate ⚠️")
    else:
        st.error("Strength Quality: Bad ❌")

    # Explainability
    st.write("📌 Factors influencing prediction: More cement & lower water improves strength, "
             "while material costs are mainly influenced by cement & aggregates.")
