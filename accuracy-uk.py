# ============================================
# Cash Flow Forecasting for SMEs using Invoice Data
# Machine Learning Model Evaluation Framework
# ============================================
# Author: Ali Nabizadeh Lamiry
# Date: July 2026
# Description: This script evaluates 5 ML models for predicting DaysToSettle
#              using invoice data from SMEs. 
# ============================================

# ============================================
# 1. IMPORTS
# ============================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import warnings
import os
import joblib

warnings.filterwarnings('ignore')

print("=" * 60)
print("UK PAYMENT PRACTICES - LIQUIDITY SCORE PREDICTION")
print("=" * 60)

# ============================================
# 2. DATA LOADING
# ============================================
print("\n[1] LOADING DATASET...")

INPUT_FILE = 'uk_payment_practices.csv'
OUTPUT_DIR = './results_uk'

os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    df = pd.read_csv(INPUT_FILE)
    print(f"   ✅ Dataset loaded successfully!")
    print(f"   📊 Total records: {len(df):,}")
except FileNotFoundError:
    print(f"   ❌ Error: File '{INPUT_FILE}' not found!")
    print(f"   📌 Please check the filename and try again.")
    exit()

# ============================================
# 3. DATA CLEANING & PREPROCESSING
# ============================================
print("\n[2] DATA CLEANING & PREPROCESSING...")

# Clean column names
df.columns = df.columns.str.strip()

essential_cols = [
    'Average time to pay',
    'E-Invoicing offered',
    'Supply-chain financing offered',
    'Participates in payment codes'
]

extra_cols = [
    'Shortest (or only) standard payment period',
    'Longest standard payment period',
    'Payment terms have changed'
]

df_clean = df.dropna(subset=essential_cols)

for col in extra_cols[:2]:
    if col in df_clean.columns:
        median_val = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(median_val)

if 'Payment terms have changed' in df_clean.columns:
    df_clean['Payment terms have changed'] = df_clean['Payment terms have changed'].fillna(False)

print(f"   ✅ Records after cleaning: {len(df_clean):,}")

# ============================================
# 4. FEATURE ENGINEERING & TARGET DEFINITION
# ============================================
print("\n[3] FEATURE ENGINEERING...")

# Target: Percentage of 60-day threshold utilised
THRESHOLD = 60
df_clean['Target'] = (df_clean['Average time to pay'] / THRESHOLD) * 100

bool_columns = [
    'E-Invoicing offered',
    'Supply-chain financing offered',
    'Participates in payment codes',
    'Payment terms have changed'
]

for col in bool_columns:
    if df_clean[col].dtype == 'bool':
        df_clean[col] = df_clean[col].astype(int)
    else:
        df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()
        df_clean[col] = df_clean[col].map({'true': 1, 'false': 0, '1': 1, '0': 0}).fillna(0).astype(int)

feature_columns = [
    'E-Invoicing offered',
    'Supply-chain financing offered',
    'Participates in payment codes',
    'Shortest (or only) standard payment period',
    'Longest standard payment period',
    'Payment terms have changed'
]

target_column = 'Target'

X = df_clean[feature_columns]
y = df_clean[target_column]

print(f"   📊 Feature set: {feature_columns}")
print(f"   🎯 Target variable: {target_column}")
print(f"   📈 Target range: {y.min():.2f}% - {y.max():.2f}%")

# ============================================
# 5. TRAIN-TEST SPLIT & DATA SCALING
# ============================================
print("\n[4] TRAIN-TEST SPLIT & SCALING...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"   📊 Training data: {X_train_scaled.shape[0]:,} records")
print(f"   📊 Test data: {X_test_scaled.shape[0]:,} records")

# ============================================
# 6. MODEL TRAINING & EVALUATION
# ============================================
print("\n[5] TRAINING MACHINE LEARNING MODELS...")

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
    'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
    'Neural Network': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
}

results = []
predictions = {}
trained_models = {}

for name, model in models.items():
    print(f"   ▶️ Training {name}...")
    
    if name in ['Random Forest', 'XGBoost', 'LightGBM']:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    else:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    
    trained_models[name] = model
    predictions[name] = y_pred
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results.append({
        'Model': name,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    })
    print(f"      ✅ {name}: RMSE = {rmse:.4f}, MAE = {mae:.4f}, R2 = {r2:.4f}")

# ============================================
# 7. RESULTS TABLE & COEFFICIENTS
# ============================================
print("\n[6] RESULTS - MODEL COMPARISON")

results_df = pd.DataFrame(results).sort_values('RMSE')

print("\n   📊 Table 1: Model Performance Comparison:")
print("   " + "-" * 70)
print(results_df.to_string(index=False))
print("   " + "-" * 70)

results_df.to_csv(f'{OUTPUT_DIR}/uk_results.csv', index=False)
print(f"\n   💾 Saved to: {OUTPUT_DIR}/uk_results.csv")

lin_reg = trained_models['Linear Regression']
coef_df = pd.DataFrame({
    'Feature': feature_columns,
    'Coefficient (Target %)': lin_reg.coef_,
    'Impact (Days)': (lin_reg.coef_ / 100) * THRESHOLD
})

print("\n   📊 Table 2: Linear Regression Coefficients (Interpretability):")
print("   " + "-" * 70)
print(coef_df.to_string(index=False))
print("   " + "-" * 70)
coef_df.to_csv(f'{OUTPUT_DIR}/uk_linear_coefficients.csv', index=False)

# ============================================
# 8. FEATURE IMPORTANCE
# ============================================
print("\n[7] FEATURE IMPORTANCE ANALYSIS...")

feature_importance_data = []

for name in ['Random Forest', 'XGBoost', 'LightGBM']:
    model = trained_models.get(name)
    if model and hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        for i, feat in enumerate(feature_columns):
            feature_importance_data.append({
                'Model': name,
                'Feature': feat,
                'Importance': importance[i]
            })

if feature_importance_data:
    fi_df = pd.DataFrame(feature_importance_data)
    fi_df = fi_df.sort_values(['Model', 'Importance'], ascending=[True, False])
    
    print("\n   📊 Feature Importance (Top features per model):")
    for model_name in ['Random Forest', 'XGBoost', 'LightGBM']:
        print(f"\n   🔹 {model_name}:")
        top_features = fi_df[fi_df['Model'] == model_name].head(3)
        for _, row in top_features.iterrows():
            print(f"      - {row['Feature']}: {row['Importance']:.4f}")
    
    fi_df.to_csv(f'{OUTPUT_DIR}/uk_feature_importance.csv', index=False)
    print(f"\n   💾 Saved to: {OUTPUT_DIR}/uk_feature_importance.csv")

# ============================================
# 9. SAVE BEST MODEL
# ============================================
print("\n[9] SAVING BEST MODEL...")

best_model_name = results_df.iloc[0]['Model']
best_model = trained_models[best_model_name]
best_rmse = results_df.iloc[0]['RMSE']
best_r2 = results_df.iloc[0]['R2']

model_path = f'{OUTPUT_DIR}/best_model_{best_model_name.replace(" ", "_")}.pkl'
joblib.dump(best_model, model_path)
print(f"   ✅ Best model saved to: {model_path}")

scaler_path = f'{OUTPUT_DIR}/scaler.pkl'
joblib.dump(scaler, scaler_path)
print(f"   ✅ Scaler saved to: {scaler_path}")

# ============================================
# 10. PREDICTIONS ON TEST SET
# ============================================
print("\n[10] GENERATING PREDICTIONS ON TEST SET...")

if best_model_name in ['Random Forest', 'XGBoost', 'LightGBM']:
    y_pred_best = best_model.predict(X_test)
else:
    y_pred_best = best_model.predict(X_test_scaled)

predictions_df = X_test.copy()
predictions_df['Actual_Target'] = y_test.values
predictions_df['Predicted_Target'] = y_pred_best
predictions_df['Predicted_Days'] = (predictions_df['Predicted_Target'] / 100) * THRESHOLD
predictions_df['Error'] = predictions_df['Predicted_Target'] - predictions_df['Actual_Target']

predictions_df.to_csv(f'{OUTPUT_DIR}/uk_test_predictions.csv', index=False)
print(f"   ✅ Test predictions saved to: {OUTPUT_DIR}/uk_test_predictions.csv")

# ============================================
# 11. FINAL SUMMARY
# ============================================
print("\n" + "=" * 60)
print("🎯 UK ANALYSIS COMPLETE!")
print("=" * 60)

print(f"\n📁 All results saved to: {OUTPUT_DIR}/")
print("   📄 uk_results.csv - Model comparison (RMSE, MAE, R²)")
print("   📄 uk_linear_coefficients.csv - Linear regression coefficients (interpretable)")
print("   📄 uk_feature_importance.csv - Feature importance from tree models")
print("   📄 uk_test_predictions.csv - Predictions on test set")
print("   📄 best_model_*.pkl - Trained best model for production")
print("   📄 scaler.pkl - StandardScaler for production")

print(f"\n🏆 Best model: {best_model_name} (RMSE = {best_rmse:.4f}, R² = {best_r2:.4f})")

print("\n📌 Business Impact (Linear Regression coefficients):")
for _, row in coef_df.iterrows():
    if abs(row['Impact (Days)']) > 0.01:
        direction = "reduces" if row['Impact (Days)'] < 0 else "increases"
        print(f"   • {row['Feature']}: {direction} payment time by {abs(row['Impact (Days)']):.2f} days on average")

print("\n🚀 Model ready for deployment and research paper.")
print("=" * 60)
