# ✅ Required Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# ✅ Load and preprocess data
df = pd.read_csv('/content/concrete.csv')
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

# Save stats for suggestions
feature_stats = df[numeric_features].agg(['mean', 'std']).T
feature_stats['min_suggest'] = (feature_stats['mean'] - feature_stats['std']).round(2)
feature_stats['max_suggest'] = (feature_stats['mean'] + feature_stats['std']).round(2)

# ✅ Preprocessing pipeline
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# ✅ Model & Tuning
model = XGBRegressor(objective='reg:squarederror', random_state=42)
params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.7, 1],
    'colsample_bytree': [0.7, 1]
}
search = RandomizedSearchCV(model, params, n_iter=5, scoring='r2', cv=3, verbose=1, n_jobs=-1)
xgb_pipeline = Pipeline([('preprocess', preprocessor), ('model', search)])

# ✅ Train-Test Split and Fit
X_train_pipe, X_test_pipe, y_train_pipe, y_test_pipe = train_test_split(X, y, test_size=0.3, random_state=1)
xgb_pipeline.fit(X_train_pipe, y_train_pipe)
best_xgb = search.best_estimator_

# ✅ Predict & Score
y_preds = xgb_pipeline.predict(X_test_pipe)
pipe_score = r2_score(y_test_pipe, y_preds)
print(f"\n✅ Tuned XGBoost Pipeline Test R² Score: {pipe_score:.4f}")

# ✅ Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test_pipe, y_preds, alpha=0.7, edgecolors='k')
plt.xlabel('Actual Strength')
plt.ylabel('Predicted Strength')
plt.title('Actual vs Predicted Concrete Strength')
plt.grid(True)
plt.show()

# ✅ Final model
final_model = Pipeline([
    ('preprocess', preprocessor),
    ('Regressor', best_xgb)
])
final_model.fit(X_train_pipe, y_train_pipe)

# ✅ SHAP Setup
print("\n⚙️ Generating SHAP explanations...")
X_test_transformed = preprocessor.transform(X_test_pipe)
feature_names = preprocessor.get_feature_names_out()
X_test_df_transformed = pd.DataFrame(X_test_transformed, columns=feature_names)

explainer = shap.Explainer(best_xgb.predict, X_test_df_transformed)
shap_values = explainer(X_test_df_transformed)
shap.summary_plot(shap_values, features=X_test_df_transformed)

# ✅ Manual Prediction
def predict_with_explanation():
    print("\n🧾 Enter the following values:")
    user_input = {
        "slag": float(input("Slag (kg/m^3): ")),
        "ash": float(input("Fly Ash (kg/m^3): ")),
        "superplastic": float(input("Superplasticizer (kg/m^3): ")),
        "coarseagg": float(input("Coarse Aggregate (kg/m^3): ")),
        "fineagg": float(input("Fine Aggregate (kg/m^3): ")),
        "water": float(input("Water (kg/m^3): ")),
        "cement": float(input("Cement (kg/m^3): ")),
        "age": input("Age (in days): ")
    }

    user_input["w/c_ratio"] = user_input["water"] / user_input["cement"]
    user_input["slag"] = np.sqrt(user_input["slag"])
    user_input["superplastic"] = np.sqrt(user_input["superplastic"])
    user_df = pd.DataFrame([user_input])
    user_df.drop(['water', 'cement'], axis=1, inplace=True)

    strength = final_model.predict(user_df)[0]
    print(f"\n🔮 Predicted Concrete Strength: {strength:.2f} MPa")

    if strength >= 50:
        quality = "Excellent"
    elif strength >= 20:
        quality = "Moderate"
    else:
        quality = "Poor"
    print(f"📊 Strength Category: {quality}")

    # Transform and align SHAP input
    transformed = preprocessor.transform(user_df)
    shap_input = pd.DataFrame(transformed, columns=preprocessor.get_feature_names_out())
    shap_val = explainer(shap_input)

    print("\n📌 Feature Influence Summary:")
    vals = list(zip(shap_input.columns, shap_val[0].values))
    top_features = sorted(vals, key=lambda x: abs(x[1]), reverse=True)[:5]

    for feat, val in top_features:
        impact = "increased" if val > 0 else "decreased"
        base_feat = feat.split("__")[-1]
        print(f"• '{base_feat}' {impact} the strength by {abs(val):.2f} MPa")

    print("\n🛠 Suggestions to Improve Strength:")
    for feat, val in top_features:
        base_feat = feat.split("__")[-1]
        if base_feat in feature_stats.index:
            direction = "increase" if val < 0 else "adjust"
            min_v = feature_stats.loc[base_feat, 'min_suggest']
            max_v = feature_stats.loc[base_feat, 'max_suggest']
            print(f"→ Consider to {direction} '{base_feat}' in range [{min_v}, {max_v}]")

    shap.plots.waterfall(shap_val[0], show=True)

# ✅ AI-Based Mix Optimizer
def optimize_mix(target_strength, age='28'):
    print(f"\n🧠 Optimizing mix for target strength: {target_strength} MPa")

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
        pred = final_model.predict(df_sample)[0]
        return abs(pred - target_strength)

    bounds = [
        (0.3, 0.7),       # w/c_ratio
        (0.1, 100),       # slag
        (0.0, 200),       # ash
        (0.1, 10),        # superplastic
        (800, 1200),      # coarseagg
        (600, 1000)       # fineagg
    ]

    x0 = [0.5, 30, 50, 5, 1000, 800]

    result = minimize(objective, x0=x0, bounds=bounds, method='L-BFGS-B')
    if result.success:
        x_opt = result.x
        optimized_values = {
            'w/c_ratio': x_opt[0],
            'slag': x_opt[1],
            'ash': x_opt[2],
            'superplastic': x_opt[3],
            'coarseagg': x_opt[4],
            'fineagg': x_opt[5],
            'age': age
        }
        pred = final_model.predict(pd.DataFrame([{
            'w/c_ratio': x_opt[0],
            'slag': np.sqrt(x_opt[1]),
            'ash': x_opt[2],
            'superplastic': np.sqrt(x_opt[3]),
            'coarseagg': x_opt[4],
            'fineagg': x_opt[5],
            'age': age
        }]))[0]
        print(f"\n🎯 Achieved Strength: {pred:.2f} MPa")
        print(f"🔧 Suggested Mix Design:")
        for k, v in optimized_values.items():
            if isinstance(v, (int, float)):
                print(f"• {k}: {v:.2f}")
            else:
                print(f"• {k}: {v}")
    else:
        print("❌ Optimization failed. Try another target value.")

# ✅ Run Predict or Optimize
predict_with_explanation()
optimize_mix(target_strength=float(input("\n🎯 Enter target strength for mix optimization (MPa): "))) 
