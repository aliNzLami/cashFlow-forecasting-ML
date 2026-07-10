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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings('ignore')

print("=" * 60)
print("CASH FLOW FORECASTING FOR SMES - IBM DATASET")
print("=" * 60)

# ============================================
# 2. DATA LOADING
# ============================================
print("\n[1] LOADING DATASET...")

INPUT_FILE = 'WA_Fn-UseC_-Accounts-Receivable.csv'  # file name
OUTPUT_DIR = './results'

os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    df = pd.read_csv(INPUT_FILE)
    print(f"   ✅ Dataset loaded successfully!")
    print(f"   📊 Total records: {len(df)}")
except FileNotFoundError:
    print(f"   ❌ Error: File '{INPUT_FILE}' not found!")
    print(f"   📌 Please check the filename and try again.")
    exit()

# ============================================
# 3. DATA CLEANING & PREPROCESSING
# ============================================
print("\n[2] DATA CLEANING & PREPROCESSING...")

df.columns = df.columns.str.strip()

duplicate_cols = df.columns[df.columns.duplicated()].tolist()
if duplicate_cols:
    print(f"   ⚠️ Duplicate columns found: {duplicate_cols}")
    df = df.loc[:, ~df.columns.duplicated()]
    print(f"   ✅ Duplicate columns removed.")
else:
    print(f"   ✅ No duplicate columns found.")

print(f"\n   📋 Available columns:")
for i, col in enumerate(df.columns):
    print(f"      {i}: '{col}'")

# ============================================
# 4. FEATURE ENGINEERING
# ============================================
print("\n[3] FEATURE ENGINEERING...")

# تبدیل ستون‌های تاریخ به datetime
date_cols = ['InvoiceDate', 'DueDate', 'SettledDate']
for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        print(f"   ✅ {col} converted to datetime.")

# Credit Period 
df['CreditPeriod'] = (df['DueDate'] - df['InvoiceDate']).dt.days

# (Disputed, PaperlessBill) to 1 and 0
if 'Disputed' in df.columns:
    df['Disputed'] = df['Disputed'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
    print(f"   ✅ Disputed converted to binary.")
if 'PaperlessBill' in df.columns:
    df['PaperlessBill'] = df['PaperlessBill'].map({'Paper': 1, 'Electronic': 0}).fillna(0).astype(int)
    print(f"   ✅ PaperlessBill converted to binary.")

# defien features
feature_columns = [
    'InvoiceAmount',
    'DaysLate',
    'CreditPeriod',
    'Disputed',
    'PaperlessBill'
]
target_column = 'DaysToSettle'

# remove lost records
df_clean = df[feature_columns + [target_column]].dropna()
print(f"   ✅ Records after cleaning: {len(df_clean)}")

X = df_clean[feature_columns]
y = df_clean[target_column]

print(f"   📊 Feature set: {feature_columns}")
print(f"   🎯 Target variable: {target_column}")

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

print(f"   📊 Training data: {X_train_scaled.shape[0]} records")
print(f"   📊 Test data: {X_test_scaled.shape[0]} records")

# ============================================
# 6. MODEL TRAINING & EVALUATION (RMSE, MAE, R²)
# ============================================
print("\n[5] TRAINING MACHINE LEARNING MODELS...")

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
    'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
    'Neural Network': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
}

results = []
predictions = {}

for name, model in models.items():
    print(f"   ▶️ Training {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
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
# 7. RESULTS TABLE
# ============================================
print("\n[6] RESULTS - MODEL COMPARISON (RMSE, MAE, R²)")

results_df = pd.DataFrame(results).sort_values('RMSE')

print("\n   📊 Table 1: Model Performance Comparison:")
print("   " + "-" * 65)
print(results_df.to_string(index=False))
print("   " + "-" * 65)

results_df.to_csv(f'{OUTPUT_DIR}/ibm_results.csv', index=False)
print(f"\n   💾 Saved to: {OUTPUT_DIR}/ibm_results.csv")

