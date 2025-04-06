# ✅ Install SHAP and XGBoost if not installed
# !pip install shap xgboost

# ✅ Imports
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

# Save feature stats for suggestions
feature_stats = df[numeric_features].agg(['mean', 'std']).T
feature_stats['min_suggest'] = (feature_stats['mean'] - feature_stats['std']).round(2)
feature_stats['max_suggest'] = (feature_stats['mean'] + feature_stats['std']).round(2)

# ✅ Preprocessing pipeline
numeric_transformer = Pipeline([('scaler', StandardScaler())])
categorical_transformer = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# ✅ Model & Hyperparameter Tuning
model = XGBRegressor(objective='reg:squarederror', random_state=42)
params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.7, 1],
    'colsample_bytree': [0.7, 1]
}
search = RandomizedSearchCV(model, params, n_iter=5, scoring='r2', cv=3, n_jobs=-1)
pipeline = Pipeline([('preprocess', preprocessor), ('model', search)])

# ✅ Train-Test Split and Fit
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
pipeline.fit(X_train, y_train)
best_model = search.best_estimator_

# ✅ SHAP Explainer Setup
X_transformed = pd.DataFrame(preprocessor.transform(X_test),
                             columns=preprocessor.get_feature_names_out())
explainer = shap.Explainer(best_model.predict, X_transformed)

# ✅ Function: Manual Prediction with SHAP Explanation
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

    # Predict
    strength = pipeline.predict(user_df)[0]
    print(f"\n🔮 Predicted Concrete Strength: {strength:.2f} MPa")

    quality = "Excellent" if strength >= 50 else "Moderate" if strength >= 20 else "Poor"
    print(f"📊 Strength Category: {quality}")

    # SHAP Explanation
    transformed = preprocessor.transform(user_df)
    shap_input = pd.DataFrame(transformed, columns=preprocessor.get_feature_names_out())
    shap_val = explainer(shap_input)

    print("\n📌 Feature Influence Summary:")
    top_feats = sorted(zip(shap_input.columns, shap_val[0].values), key=lambda x: abs(x[1]), reverse=True)[:5]
    for feat, val in top_feats:
        impact = "increased" if val > 0 else "decreased"
        print(f"• '{feat.split('__')[-1]}' {impact} strength by {abs(val):.2f} MPa")

    print("\n🛠 Suggestions to Improve Strength:")
    for feat, val in top_feats:
        base_feat = feat.split("__")[-1]
        if base_feat in feature_stats.index:
            action = "increase" if val < 0 else "adjust"
            min_v = feature_stats.loc[base_feat, 'min_suggest']
            max_v = feature_stats.loc[base_feat, 'max_suggest']
            print(f"→ Consider to {action} '{base_feat}' in range [{min_v}, {max_v}]")

    shap.plots.waterfall(shap_val[0], show=True)

# ✅ Function: AI-Based Mix Optimizer
def inverse_prediction(target_strength=50, age="28"):
    print(f"\n🧠 Optimizing mix for target strength: {target_strength} MPa, Age: {age} days")
    init = feature_stats['mean'].values
    bounds = list(zip(feature_stats['min_suggest'], feature_stats['max_suggest']))

    def objective(x):
        features = {
            'w/c_ratio': x[0],
            'slag': x[1],
            'ash': x[2],
            'superplastic': x[3],
            'coarseagg': x[4],
            'fineagg': x[5],
            'age': age
        }
        input_df = pd.DataFrame([features])
        input_df['slag'] = np.sqrt(input_df['slag'])
        input_df['superplastic'] = np.sqrt(input_df['superplastic'])
        pred = pipeline.predict(input_df)[0]
        return abs(pred - target_strength)

    result = minimize(objective, init, bounds=bounds, method='L-BFGS-B')
    best_mix = result.x

    print("\n🧪 Suggested Mix Design (AI Optimized):")
    for name, val in zip(numeric_features, best_mix):
        print(f"• {name}: {val:.2f}")

    opt_df = pd.DataFrame([{
        'w/c_ratio': best_mix[0],
        'slag': np.sqrt(best_mix[1]),
        'ash': best_mix[2],
        'superplastic': np.sqrt(best_mix[3]),
        'coarseagg': best_mix[4],
        'fineagg': best_mix[5],
        'age': age
    }])
    pred_strength = pipeline.predict(opt_df)[0]
    print(f"\n🎯 Predicted Strength for Optimized Mix: {pred_strength:.2f} MPa")

# ✅ Run Options
predict_with_explanation()
inverse_prediction(target_strength=45, age="14")
