# Install dependencies
!pip install shap catboost lightgbm xgboost

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Load dataset
df = pd.read_csv('/content/concrete.csv')
df.drop_duplicates(inplace=True)

# ✅ Feature Engineering
df['w/c_ratio'] = df['water'] / df['cement']
df['binder'] = df['cement'] + df['slag'] + df['ash']
df['cement_ratio'] = df['cement'] / df['binder']
df['slag_ratio'] = df['slag'] / df['binder']
df['binder_agg_ratio'] = df['binder'] / (df['coarseagg'] + df['fineagg'])

# ✅ Cost Calculation (synthetic industry cost per m³)
cost_cement = 6.0   # per kg
cost_slag = 2.0     # per kg
cost_ash = 1.5      # per kg
cost_water = 0.01   # per kg
cost_super = 15.0   # per kg
cost_coarse = 0.8   # per kg
cost_fine = 0.6     # per kg

df['cost'] = (
    df['cement'] * cost_cement +
    df['slag'] * cost_slag +
    df['ash'] * cost_ash +
    df['water'] * cost_water +
    df['superplastic'] * cost_super +
    df['coarseagg'] * cost_coarse +
    df['fineagg'] * cost_fine
)

# Transformations
df['age'] = df['age'].astype(str)
df['slag'] = df['slag'].apply(np.sqrt)
df['superplastic'] = df['superplastic'].apply(np.sqrt)

# Features and targets
X = df.drop(['strength', 'cost'], axis=1)
y_strength = df['strength']
y_cost = df['cost']

# Preprocessing
one_hot_cols = ['age']
numeric_features = ['w/c_ratio', 'slag', 'ash', 'superplastic',
                    'coarseagg', 'fineagg',
                    'cement_ratio', 'slag_ratio', 'binder_agg_ratio']

one_hot_transform = Pipeline(steps=[('oneHot', OneHotEncoder(handle_unknown='ignore'))])
numeric_transform = Pipeline(steps=[('scaler', StandardScaler())])

preprocessor = ColumnTransformer([
    ('one_hot', one_hot_transform, one_hot_cols),
    ('num', numeric_transform, numeric_features)
])

# Train-test split
X_train, X_test, y_train_s, y_test_s = train_test_split(X, y_strength, test_size=0.3, random_state=1)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_cost, test_size=0.3, random_state=1)

# Models and hyperparameters
models = {
    'Random Forest': (RandomForestRegressor(), {
        'n_estimators': [100, 300],
        'max_depth': [None, 10, 20]
    }),
    'Extra Trees': (ExtraTreesRegressor(), {
        'n_estimators': [100, 300],
        'max_depth': [None, 10, 20]
    }),
    'XGBoost': (XGBRegressor(objective='reg:squarederror'), {
        'n_estimators': [100, 300],
        'max_depth': [4, 6],
        'learning_rate': [0.05, 0.1]
    }),
    'LightGBM': (LGBMRegressor(), {
        'n_estimators': [100, 300],
        'max_depth': [4, 6],
        'learning_rate': [0.05, 0.1]
    }),
    'CatBoost': (CatBoostRegressor(verbose=0), {
        'iterations': [100, 300],
        'depth': [4, 6],
        'learning_rate': [0.05, 0.1]
    })
}

# ---------- Strength Model Selection ----------
best_model_s = None
best_score_s = -np.inf
best_name_s = ""
final_pipeline_s = None

for name, (model, params) in models.items():
    search = RandomizedSearchCV(model, param_distributions=params, n_iter=5, cv=3,
                                scoring='r2', random_state=42, n_jobs=-1)
    pipe = Pipeline([
        ('preprocess', preprocessor),
        ('regressor', search)
    ])
    pipe.fit(X_train, y_train_s)
    score = pipe.score(X_test, y_test_s)
    print(f"{name} (Strength) R2 Score: {score:.4f}")

    if score > best_score_s:
        best_score_s = score
        best_model_s = search.best_estimator_
        best_name_s = name
        final_pipeline_s = Pipeline([
            ('preprocess', preprocessor),
            ('regressor', best_model_s)
        ])

# Predictions for Strength
y_preds_s = final_pipeline_s.predict(X_test)
rmse_s = np.sqrt(mean_squared_error(y_test_s, y_preds_s))
mae_s = mean_absolute_error(y_test_s, y_preds_s)
mape_s = np.mean(np.abs((y_test_s - y_preds_s) / y_test_s)) * 100

print(f"\n✅ Best Strength Model: {best_name_s}")
print(f"R² Score: {best_score_s:.4f}")
print(f"RMSE: {rmse_s:.4f}")
print(f"MAE: {mae_s:.4f}")
print(f"MAPE: {mape_s:.2f}%")

# Plot Strength
plt.figure(figsize=(8, 6))
plt.scatter(y_test_s, y_preds_s, alpha=0.7, edgecolors='k')
plt.xlabel('Actual Strength')
plt.ylabel('Predicted Strength')
plt.title(f'Actual vs Predicted Strength ({best_name_s})')
plt.grid(True)
plt.show()

# ---------- Cost Model Selection ----------
best_model_c = None
best_score_c = -np.inf
best_name_c = ""
final_pipeline_c = None

for name, (model, params) in models.items():
    search = RandomizedSearchCV(model, param_distributions=params, n_iter=5, cv=3,
                                scoring='r2', random_state=42, n_jobs=-1)
    pipe = Pipeline([
        ('preprocess', preprocessor),
        ('regressor', search)
    ])
    pipe.fit(X_train_c, y_train_c)
    score = pipe.score(X_test_c, y_test_c)
    print(f"{name} (Cost) R2 Score: {score:.4f}")

    if score > best_score_c:
        best_score_c = score
        best_model_c = search.best_estimator_
        best_name_c = name
        final_pipeline_c = Pipeline([
            ('preprocess', preprocessor),
            ('regressor', best_model_c)
        ])

# Predictions for Cost
y_preds_c = final_pipeline_c.predict(X_test_c)
rmse_c = np.sqrt(mean_squared_error(y_test_c, y_preds_c))
mae_c = mean_absolute_error(y_test_c, y_preds_c)
mape_c = np.mean(np.abs((y_test_c - y_preds_c) / y_test_c)) * 100

print(f"\n✅ Best Cost Model: {best_name_c}")
print(f"R² Score: {best_score_c:.4f}")
print(f"RMSE: {rmse_c:.4f}")
print(f"MAE: {mae_c:.4f}")
print(f"MAPE: {mape_c:.2f}%")

# Plot Cost
plt.figure(figsize=(8, 6))
plt.scatter(y_test_c, y_preds_c, alpha=0.7, edgecolors='k')
plt.xlabel('Actual Cost')
plt.ylabel('Predicted Cost')
plt.title(f'Actual vs Predicted Cost ({best_name_c})')
plt.grid(True)
plt.show()

# ---------- NEW: Classification & Strength vs Cost Graph ----------
# Classification function
def classify_strength(val):
    if val >= 50:
        return "Good"
    elif val >= 30:
        return "Moderate"
    else:
        return "Bad"

# Build results DataFrame aligned on test set index
# Use the intersection of indices just in case (should be same)
idx = X_test.index
results_df = pd.DataFrame({
    "Actual Strength": y_test_s.loc[idx].values,
    "Predicted Strength": y_preds_s,
    "Predicted Cost": y_preds_c
}, index=idx)

results_df["Strength Class"] = results_df["Predicted Strength"].apply(classify_strength)

print("\n--- Strength Classification Sample ---")
print(results_df.head(10))

# Strength vs Cost scatter plot colored by class
colors = {"Good": "green", "Moderate": "orange", "Bad": "red"}
plt.figure(figsize=(9, 6))
for cls in results_df["Strength Class"].unique():
    subset = results_df[results_df["Strength Class"] == cls]
    plt.scatter(subset["Predicted Strength"], subset["Predicted Cost"],
                c=colors[cls], label=cls, alpha=0.7, edgecolors='k')

plt.xlabel("Predicted Strength (MPa)")
plt.ylabel("Predicted Cost (currency units/m³)")
plt.title("Strength vs Cost Trade-off (Predicted)")
plt.legend(title="Strength Class")
plt.grid(True)
plt.show()

# ---------- Manual Prediction Function (updated to show class and plot) ----------
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

    # ✅ Feature Engineering
    user_input["w/c_ratio"] = user_input["water"] / user_input["cement"]
    user_input["binder"] = user_input["cement"] + user_input["slag"] + user_input["ash"]
    user_input["cement_ratio"] = user_input["cement"] / user_input["binder"]
    user_input["slag_ratio"] = user_input["slag"] / user_input["binder"]
    user_input["binder_agg_ratio"] = user_input["binder"] / (user_input["coarseagg"] + user_input["fineagg"])

    # apply same transforms as training
    user_input["slag"] = np.sqrt(user_input["slag"])
    user_input["superplastic"] = np.sqrt(user_input["superplastic"])

    # Cost calculation (use original quantities for cost)
    user_input["cost"] = (
        user_input["cement"] * cost_cement +
        (user_input["slag"]**2) * cost_slag +  # revert sqrt for real cost
        user_input["ash"] * cost_ash +
        user_input["water"] * cost_water +
        (user_input["superplastic"]**2) * cost_super +
        user_input["coarseagg"] * cost_coarse +
        user_input["fineagg"] * cost_fine
    )

    user_df = pd.DataFrame([user_input])
    user_df.drop(["water", "cement", "binder"], axis=1, inplace=True)

    # Predictions
    pred_strength = final_pipeline_s.predict(user_df.drop("cost", axis=1))[0]
    pred_cost = final_pipeline_c.predict(user_df.drop("cost", axis=1))[0]
    pred_class = classify_strength(pred_strength)

    print(f"\n🔮 Predicted Strength: {pred_strength:.2f} MPa ({pred_class})")
    print(f"💰 Predicted Cost: {pred_cost:.2f} currency units/m³")

    # Plot the single predicted point on Strength vs Cost
    plt.figure(figsize=(6, 5))
    plt.scatter(pred_strength, pred_cost, c=colors[pred_class], s=150, edgecolors='k')
    plt.xlabel("Predicted Strength (MPa)")
    plt.ylabel("Predicted Cost (currency units/m³)")
    plt.title("Manual Prediction: Strength vs Cost")
    plt.grid(True)
    plt.show()

# Uncomment to run manual prediction interactively
predict_manual_input()
