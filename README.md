# AMPS – ADVANCED MODELING FOR PREDICTIVE STRENGTH OF CONCRETE

### Project Overview

AMPS (Advanced Modeling for Predictive Strength) is a machine learning-based application designed to predict concrete compressive strength and estimate material cost using concrete mix proportions.

The project addresses the limitations of traditional concrete strength testing methods, which involve time-consuming laboratory experiments, destructive testing, and repeated trial-and-error procedures.

The proposed system uses advanced ensemble machine learning algorithms, including XGBoost, CatBoost, and LightGBM, to deliver accurate predictions. It also incorporates Explainable AI (XAI) techniques such as SHAP to identify the contribution of individual concrete mix components to the predicted results.

The system is integrated with a user-friendly Streamlit web application, allowing civil engineers, contractors, and researchers to obtain predictions without requiring advanced machine learning knowledge.

## Project Objectives

- Develop an intelligent machine learning framework for predicting concrete compressive strength.
- Estimate material costs simultaneously to support economical concrete mix design.
- Implement advanced ensemble learning algorithms to improve prediction accuracy.
- Apply feature engineering and hyperparameter optimization to improve model performance.
- Integrate Explainable AI (SHAP) to understand the influence of individual mix components.
- Develop an interactive web application for real-time prediction and analysis.
- Reduce dependency on traditional laboratory testing, material wastage, and trial-and-error methods.

## Key Features

- **Concrete Strength Prediction:** Predicts concrete compressive strength in MPa using concrete mix proportions and curing age.
- **Material Cost Estimation:** Estimates the material cost per cubic metre of concrete.
- **Advanced Machine Learning:** Implements XGBoost, CatBoost, LightGBM, and other regression algorithms.
- **Explainable AI:** Uses SHAP analysis to identify important features and understand their impact on predictions.
- **Hyperparameter Optimization:** Applies Randomized Search with Cross-Validation to improve model performance.
- **Interactive Web Application:** Provides a simple Streamlit interface for entering mix proportions and viewing predictions.
- **Mix Optimization Feedback:** Provides quality-level assessments and suggestions to support improved concrete mix design.
- **Actual vs. Predicted Analysis:** Evaluates model performance by comparing predicted values with experimental results.

## Input Variables

The system uses the following eight input parameters:

| Variable | Description |
|---|---|
| X1 | Cement Content |
| X2 | Water Content |
| X3 | Coarse Aggregate Content |
| X4 | Fine Aggregate Content |
| X5 | Slag Content |
| X6 | Fly Ash Content |
| X7 | Superplasticizer Content |
| X8 | Curing Age (Days) |

## Output Variables

| Variable | Description |
|---|---|
| Y1 | Concrete Compressive Strength (MPa) |
| Y2 | Material Cost (Currency Units/m³) |

## Machine Learning Algorithms

The project implements and evaluates multiple machine learning algorithms to identify the most suitable prediction model.

- Linear Regression (LR)
- Support Vector Machine (SVM)
- Artificial Neural Network (ANN)
- Bayesian Neural Network (BNN)
- Bagging-based models
- Random Forest
- CatBoost
- XGBoost
- LightGBM

**XGBoost** achieved the best compressive strength prediction performance among the evaluated models.

## Technical Approach

1. **Data Collection:** Uses a publicly available concrete compressive strength dataset containing concrete mix proportions and experimentally measured strength values.
2. **Data Preprocessing:** Performs duplicate removal, anomaly detection, standard scaling, and encoding techniques.
3. **Feature Engineering:** Creates additional features such as Water-Cement Ratio, Binder Ratio, Cement Ratio, and Aggregate-Binder Ratio.
4. **Dataset Splitting:** Divides the dataset into 70% training data and 30% testing data.
5. **Model Training:** Trains multiple machine learning and ensemble regression models.
6. **Hyperparameter Optimization:** Uses Randomized Search and K-Fold Cross-Validation to identify optimal model configurations.
7. **Model Evaluation:** Evaluates prediction performance using R² Score, RMSE, MAE, and MAPE.
8. **Explainability Analysis:** Applies SHAP to interpret feature importance and their contributions to predictions.
9. **Web Application Deployment:** Integrates the trained model into a Streamlit application for interactive predictions.

## Model Performance

The proposed framework was evaluated using standard regression metrics.

### Compressive Strength Prediction

| Model | R² Score |
|---|---:|
| Linear Regression | 0.6081 |
| SVM | 0.6428 |
| ANN | 0.8653 |
| BNN | 0.8324 |
| Bagging Model | 0.9069 |
| CatBoost | 0.9356 |
| XGBoost | **0.9380** |
| LightGBM | 0.9328 |

### Best Model Results – XGBoost

- R² Score: 0.9380
- RMSE: 4.2112
- MAE: 2.6162

### Material Cost Prediction

- R² Score: 0.9986

The original software project presentation also reports an XGBoost R² score of approximately 0.95 and an example prediction of 28.83 MPa compared with a laboratory-tested value of 29.00 MPa.

## Technologies Used

**Programming Language**
- Python

**Machine Learning & Data Processing**
- Scikit-learn
- XGBoost
- CatBoost
- LightGBM
- Pandas
- NumPy

**Explainable AI**
- SHAP

**Model Optimization**
- RandomizedSearchCV
- K-Fold Cross-Validation

**Web Application**
- Streamlit

**Development & Deployment**
- Google Colab
- Joblib
- GitHub
- Streamlit Community Cloud

## Application Workflow

1. User enters concrete mix proportions through the Streamlit web application.
2. The application processes the input values and performs feature engineering.
3. The trained machine learning models analyze the input parameters.
4. The system predicts concrete compressive strength and estimates material cost.
5. SHAP analysis explains the influence of different concrete ingredients.
6. The application provides prediction results and mix-quality feedback.

## Advantages

- Reduces dependency on destructive laboratory testing.
- Minimizes time and material wastage associated with traditional testing.
- Enables early decision-making in concrete mix design.
- Provides simultaneous strength and material cost predictions.
- Improves transparency through Explainable AI.
- Offers a user-friendly interface for non-technical users.
- Supports economical and sustainable construction planning.

## Research Publication

**Title:** Concrete Compressive Strength and Material Cost Prediction Using an Explainable Machine Learning Framework.

**Conference:** 8th International Conference on Inventive Material Science and Applications (ICIMA 2026).

## Acknowledgement

This project was developed as an application of machine learning and Explainable AI techniques to improve concrete strength prediction, material cost estimation, and sustainable construction planning.
