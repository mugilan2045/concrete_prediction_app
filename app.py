# app.py
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.optimize import minimize
import joblib

# Load and preprocess
df = pd.read_csv("concrete.csv")
df.drop_duplicates(inplace=True)
df['w/c_ratio'] = df['water'] / df['cement']
df.drop(['water', 'cement'], axis=1, inplace=True)
df['age'] = df['age'].astype(str)
df['slag'] = np.sqrt(df['slag'])
df['superplastic'] = np.sqrt(df['superplastic'])

X = df.drop('strength', axis=1)
y = df['strength']

numeric_features = ['w/c_ratio', 'slag', 'ash', 'superplastic', 'coarseagg', 'fineagg']
categorical_features = ['age']

feature_stats = df[numeric_features].agg(['mean', 'std']).T
feature_stats['min_suggest'] = (feature_stats['mean'] - feature_stats['std']).round(2)
feature_stats['max_suggest'] = (feature_stats['mean'] + feature_stats['std']).round(2)

# Preprocessing
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

model = XGBRegressor(objective='reg:squarederror', random_state=42)
pipeline = Pipeline([('preprocess', preprocessor), ('model', model)])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
pipeline.fit(X_train, y_train)
explainer = shap.Explainer(pipeline.named_steps['model'].predict, preprocessor.transform(X_test))
X_test_df = pd.DataFrame(preprocessor.transform(X_test), columns=preprocessor.get_feature_names_out())

# Streamlit App
st.set_page_config(layout="centered", page_title="🧱 Concrete Strength Predictor")

st.title("🧠 Concrete Strength AI App")
st.subheader("Choose an Option")

option = st.selectbox("What do you want to do?", ["Manual Prediction", "AI Mix Optimizer"])

if option == "Manual Prediction":
    st.markdown("### ✍️ Enter Mix Parameters")
    user_input = {
        "slag": st.number_input("Slag (kg/m³)", min_value=0.0),
        "ash": st.number_input("Fly Ash (kg/m³)", min_value=0.0),
        "superplastic": st.number_input("Superplasticizer (kg/m³)", min_value=0.0),
        "coarseagg": st.number_input("Coarse Aggregate (kg/m³)", min_value=800.0, max_value=1200.0),
        "fineagg": st.number_input("Fine Aggregate (kg/m³)", min_value=600.0, max_value=1000.0),
        "water": st.number_input("Water (kg/m³)", min_value=100.0),
        "cement": st.number_input("Cement (kg/m³)", min_value=100.0),
        "age": st.selectbox("Age (days)", ['1', '3', '7', '14', '28', '90', '180'])
    }

    if st.button("🔍 Predict"):
        user_input["w/c_ratio"] = user_input["water"] / user_input["cement"]
        user_input["slag"] = np.sqrt(user_input["slag"])
        user_input["superplastic"] = np.sqrt(user_input["superplastic"])
        df_input = pd.DataFrame([user_input])
        df_input.drop(['water', 'cement'], axis=1, inplace=True)

        pred = pipeline.predict(df_input)[0]
        st.success(f"🔮 Predicted Strength: {pred:.2f} MPa")

        if pred >= 50:
            st.markdown("✅ **Quality: Excellent**")
        elif pred >= 20:
            st.markdown("⚠️ **Quality: Moderate**")
        else:
            st.markdown("❌ **Quality: Poor**")

        # SHAP Plot
        shap_input = pd.DataFrame(preprocessor.transform(df_input), columns=preprocessor.get_feature_names_out())
        shap_values = explainer(shap_input)

        st.subheader("📊 Feature Impact (Top 5)")
        for feat, val in sorted(zip(shap_input.columns, shap_values[0].values), key=lambda x: abs(x[1]), reverse=True)[:5]:
            impact = "↑" if val > 0 else "↓"
            base_feat = feat.split("__")[-1]
            st.write(f"{impact} **{base_feat}** impacted strength by **{abs(val):.2f} MPa**")

        # Suggestions
        st.subheader("🛠 Suggestions")
        for feat, val in sorted(zip(shap_input.columns, shap_values[0].values), key=lambda x: abs(x[1]), reverse=True)[:5]:
            base_feat = feat.split("__")[-1]
            if base_feat in feature_stats.index:
                direction = "increase" if val < 0 else "adjust"
                min_v = feature_stats.loc[base_feat, 'min_suggest']
                max_v = feature_stats.loc[base_feat, 'max_suggest']
                st.write(f"→ Consider to {direction} **{base_feat}** in range [{min_v}, {max_v}]")

        st.subheader("🌊 SHAP Waterfall Plot")
        fig, ax = plt.subplots(figsize=(8, 6))
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        st.pyplot(fig)

elif option == "AI Mix Optimizer":
    st.markdown("### 🎯 Enter Target Strength")
    target_strength = st.number_input("Target Concrete Strength (MPa)", min_value=5.0, max_value=100.0)
    age = st.selectbox("Age (days)", ['1', '3', '7', '14', '28', '90', '180'])

    if st.button("🚀 Optimize Mix"):
        def objective(x):
            sample = {
                'w/c_ratio': x[0],
                'slag': np.sqrt(x[1]),
                'ash': x[2],
                'superplastic': np.sqrt(x[3]),
                'coarseagg': x[4],
                'fineagg': x[5],
                'age': age
            }
            df_sample = pd.DataFrame([sample])
            pred = pipeline.predict(df_sample)[0]
            return abs(pred - target_strength)

        bounds = [
            (0.3, 0.7), (0.1, 100), (0.0, 200),
            (0.1, 10), (800, 1200), (600, 1000)
        ]
        x0 = [0.5, 30, 50, 5, 1000, 800]

        result = minimize(objective, x0=x0, bounds=bounds, method='L-BFGS-B')
        if result.success:
            st.success("🎉 Optimization Successful!")
            x_opt = result.x
            pred = pipeline.predict(pd.DataFrame([{
                'w/c_ratio': x_opt[0],
                'slag': np.sqrt(x_opt[1]),
                'ash': x_opt[2],
                'superplastic': np.sqrt(x_opt[3]),
                'coarseagg': x_opt[4],
                'fineagg': x_opt[5],
                'age': age
            }]))[0]

            st.markdown(f"### 🧪 Achieved Strength: **{pred:.2f} MPa**")
            st.markdown("### 📋 Optimized Mix Design:")
            labels = ['w/c_ratio', 'slag', 'ash', 'superplastic', 'coarseagg', 'fineagg']
            for i, val in zip(labels, x_opt):
                st.write(f"• {i}: {val:.2f}")
        else:
            st.error("❌ Optimization Failed. Try again.")
