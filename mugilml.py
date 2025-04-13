 !pip install shap

  # Import Dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

# Load and preprocess dataset
df = pd.read_csv('/content/concrete.csv')
df.drop_duplicates(inplace=True)
df['w/c_ratio'] = df['water'] / df['cement']
df.drop(['water', 'cement'], axis=1, inplace=True)
df['age'] = df['age'].astype(str)
df['slag'] = df['slag'].apply(np.sqrt)
df['superplastic'] = df['superplastic'].apply(np.sqrt)

X = df.drop('strength', axis=1)
y = df['strength']

# Preprocessing pipelines
one_hot_cols = ['age']
numeric_features = ['w/c_ratio', 'slag', 'ash', 'superplastic', 'coarseagg', 'fineagg']

one_hot_transform = Pipeline(steps=[('oneHot', OneHotEncoder(handle_unknown='ignore'))])
numeric_transform = Pipeline(steps=[('scaler', StandardScaler())])

preprocessor = ColumnTransformer(transformers=[
    ('one_hot', one_hot_transform, one_hot_cols),
    ('num', numeric_transform, numeric_features)
])

# XGBoost model + Hyperparameter tuning
xgb = XGBRegressor(objective='reg:squarederror', random_state=42)
param_dist = {
    'n_estimators': [100, 300, 500],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 1],
    'colsample_bytree': [0.7, 1],
    'gamma': [0, 0.1],
    'reg_alpha': [0, 0.1],
    'reg_lambda': [1, 2]
}
random_search_xgb = RandomizedSearchCV(xgb, param_distributions=param_dist, n_iter=10, cv=3,
                                       scoring='r2', n_jobs=-1, verbose=1, random_state=42)

# Pipeline
xgb_pipeline = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('Regressor', random_search_xgb)
])

# Train-test split
X_train_pipe, X_test_pipe, y_train_pipe, y_test_pipe = train_test_split(X, y, random_state=1, test_size=0.3)
xgb_pipeline.fit(X_train_pipe, y_train_pipe)
best_xgb = random_search_xgb.best_estimator_

# Predict and score
y_preds = xgb_pipeline.predict(X_test_pipe)
pipe_score = r2_score(y_test_pipe, y_preds)
print(f"\n✅ Tuned XGBoost Pipeline Test R² Score: {pipe_score:.4f}")

# Visual: Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test_pipe, y_preds, alpha=0.7, edgecolors='k')
plt.xlabel('Actual Strength')
plt.ylabel('Predicted Strength')
plt.title('Actual vs Predicted Concrete Strength')
plt.grid(True)
plt.show()

# EXPLAINABLE AI (XAI) using SHAP
print("\n⚙️ Generating SHAP explanations...")

# Extract the model used after preprocessing
final_model = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('Regressor', best_xgb)
])
final_model.fit(X_train_pipe, y_train_pipe)

# Use SHAP explainer
X_transformed = preprocessor.transform(X_test_pipe)
explainer = shap.Explainer(best_xgb, X_transformed)
shap_values = explainer(X_transformed)

# SHAP Summary Plot
shap.summary_plot(shap_values, X_transformed, feature_names=preprocessor.get_feature_names_out())

# --- Manual Input Section ---
def predict_manual_input():
    print("\n🧾 Enter values for manual prediction:")

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

    # Feature Engineering
    user_input["w/c_ratio"] = user_input["water"] / user_input["cement"]
    user_input["slag"] = np.sqrt(user_input["slag"])
    user_input["superplastic"] = np.sqrt(user_input["superplastic"])
    user_input_df = pd.DataFrame([user_input])
    user_input_df.drop(['water', 'cement'], axis=1, inplace=True)

    # Prediction
    pred = final_model.predict(user_input_df)[0]
    print(f"\n🔮 Predicted Concrete Strength: {pred:.2f} MPa")

    # SHAP for single prediction
    user_input_transformed = preprocessor.transform(user_input_df)
    shap_val = explainer(user_input_transformed)
    shap.plots.waterfall(shap_val[0], show=True)

# Call the manual prediction function
predict_manual_input()
