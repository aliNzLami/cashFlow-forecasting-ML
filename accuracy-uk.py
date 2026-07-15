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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import lightgbm as lgb
import warnings
import os
import joblib

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

warnings.filterwarnings('ignore')

print("=" * 60)
print("UK PAYMENT PRACTICES - LIQUIDITY SCORE PREDICTION (TIME-VALIDATED)")
print("=" * 60)

# ============================================
# 2. DATA LOADING
# ============================================
print("\n[1] LOADING DATASET...")

INPUT_FILE = 'uk_payment_practices.csv'
OUTPUT_DIR = './results_uk_fixed'

os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    df = pd.read_csv(INPUT_FILE)
    print(f"   ✅ Dataset loaded successfully!")
    print(f"   📊 Total records: {len(df):,}")
except FileNotFoundError:
    print(f"   ❌ Error: File '{INPUT_FILE}' not found!")
    exit()

# ============================================
# 3. DATA PREPROCESSING (GLOBAL CLEANING)
# ============================================
print("\n[2] GLOBAL DATA CLEANING...")

df.columns = df.columns.str.strip()


df['Start date'] = pd.to_datetime(df['Start date'], errors='coerce')
df['End date'] = pd.to_datetime(df['End date'], errors='coerce')


df = df.dropna(subset=['Start date', 'Company number'])
print(f"   ✅ Records with valid dates and company IDs: {len(df):,}")

# ============================================
# 4. FEATURE ENGINEERING
# ============================================
print("\n[3] FEATURE ENGINEERING...")


THRESHOLD = 60
df['Target'] = (df['Average time to pay'] / THRESHOLD) * 100


feature_columns = [
    'E-Invoicing offered',
    'Supply-chain financing offered',
    'Participates in payment codes',
    'Shortest (or only) standard payment period',
    'Longest standard payment period',
    'Payment terms have changed'
]

target_column = 'Target'


essential_features = ['E-Invoicing offered', 'Supply-chain financing offered', 'Participates in payment codes']
df_clean = df.dropna(subset=essential_features + ['Company number', 'Start date', target_column])

print(f"   📊 Records after essential cleaning: {len(df_clean):,}")

# ============================================
# 5. GROUPED TIME-SERIES SPLIT (CRITICAL FIX)
# ============================================
print("\n[4] GROUPED TIME-SERIES TRAIN-TEST SPLIT...")


company_first_date = df_clean.groupby('Company number')['Start date'].min().reset_index()
company_first_date = company_first_date.sort_values('Start date')


split_idx = int(len(company_first_date) * 0.7)
train_companies = company_first_date.iloc[:split_idx]['Company number'].tolist()
test_companies = company_first_date.iloc[split_idx:]['Company number'].tolist()

print(f"   📊 Number of companies: {len(company_first_date)}")
print(f"   📊 Train companies: {len(train_companies)} (First report up to {company_first_date.iloc[split_idx-1]['Start date'].date()})")
print(f"   📊 Test companies: {len(test_companies)} (From {company_first_date.iloc[split_idx]['Start date'].date()} onwards)")


train_df = df_clean[df_clean['Company number'].isin(train_companies)]
test_df = df_clean[df_clean['Company number'].isin(test_companies)]

print(f"   📊 Train records: {len(train_df)}")
print(f"   📊 Test records: {len(test_df)}")

# ============================================
# 6. SEPARATE IMPUTATION (TRAIN STATISTICS ONLY)
# ============================================
print("\n[5] IMPUTING MISSING VALUES (USING TRAIN STATISTICS)...")


X_train_raw = train_df[feature_columns].copy()
y_train = train_df[target_column].copy()
X_test_raw = test_df[feature_columns].copy()
y_test = test_df[target_column].copy()


numeric_cols = ['Shortest (or only) standard payment period', 'Longest standard payment period']
for col in numeric_cols:
    if col in X_train_raw.columns:
        median_val = X_train_raw[col].median()
        X_train_raw[col] = X_train_raw[col].fillna(median_val)
        X_test_raw[col] = X_test_raw[col].fillna(median_val)
        print(f"   ✅ Imputed '{col}' with median = {median_val:.2f} (from Train)")


bool_cols = ['Payment terms have changed']
for col in bool_cols:
    if col in X_train_raw.columns:
        mode_val = X_train_raw[col].mode().iloc[0] if not X_train_raw[col].mode().empty else 0
        X_train_raw[col] = X_train_raw[col].fillna(mode_val)
        X_test_raw[col] = X_test_raw[col].fillna(mode_val)
        print(f"   ✅ Imputed '{col}' with mode = {mode_val} (from Train)")


bool_columns = [
    'E-Invoicing offered',
    'Supply-chain financing offered',
    'Participates in payment codes',
    'Payment terms have changed'
]

for col in bool_columns:
    
    X_train_raw[col] = X_train_raw[col].astype(str).str.strip().str.lower()
    X_train_raw[col] = X_train_raw[col].map({'true': 1, 'false': 0, '1': 1, '0': 0}).fillna(0).astype(int)
    
    X_test_raw[col] = X_test_raw[col].astype(str).str.strip().str.lower()
    X_test_raw[col] = X_test_raw[col].map({'true': 1, 'false': 0, '1': 1, '0': 0}).fillna(0).astype(int)

print(f"   ✅ All boolean features converted to binary (0/1).")

# ============================================
# 7. DATA SCALING (FIT ONLY ON TRAIN)
# ============================================
print("\n[6] DATA SCALING...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)


X_train_tree = X_train_raw.values
X_test_tree = X_test_raw.values

# ============================================
# 8. MODEL TRAINING & EVALUATION
# ============================================
print("\n[7] TRAINING MACHINE LEARNING MODELS (TIME-VALIDATED)...")

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
        model.fit(X_train_tree, y_train)
        y_pred = model.predict(X_test_tree)
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
# 9. RESULTS TABLE
# ============================================
print("\n[8] RESULTS - MODEL COMPARISON (TIME-VALIDATED, GROUPED)")

results_df = pd.DataFrame(results).sort_values('RMSE')

print("\n   📊 Table: Model Performance Comparison (UK - Fixed):")
print("   " + "-" * 70)
print(results_df.to_string(index=False))
print("   " + "-" * 70)

results_df.to_csv(f'{OUTPUT_DIR}/uk_results_fixed.csv', index=False)
print(f"\n   💾 Saved to: {OUTPUT_DIR}/uk_results_fixed.csv")

# ============================================
# 10. LINEAR REGRESSION COEFFICIENTS
# ============================================
lin_reg = trained_models['Linear Regression']
coef_df = pd.DataFrame({
    'Feature': feature_columns,
    'Coefficient (Target %)': lin_reg.coef_,
    'Impact (Days)': (lin_reg.coef_ / 100) * THRESHOLD
})

print("\n   📊 Table: Linear Regression Coefficients (Interpretable):")
print("   " + "-" * 70)
print(coef_df.to_string(index=False))
print("   " + "-" * 70)
coef_df.to_csv(f'{OUTPUT_DIR}/uk_linear_coefficients_fixed.csv', index=False)

# ============================================
# 11. SAVE BEST MODEL & SCALER
# ============================================
best_model_name = results_df.iloc[0]['Model']
best_model = trained_models[best_model_name]

joblib.dump(best_model, f'{OUTPUT_DIR}/best_model_{best_model_name.replace(" ", "_")}_fixed.pkl')
joblib.dump(scaler, f'{OUTPUT_DIR}/scaler_fixed.pkl')

print(f"\n🏆 Best model: {best_model_name} (RMSE = {results_df.iloc[0]['RMSE']:.4f}, R² = {results_df.iloc[0]['R2']:.4f})")
print(f"\n📁 All results saved to: {OUTPUT_DIR}/")

print("\n" + "=" * 60)
print("✅ ANALYSIS COMPLETE. Results are now realistic and publication-ready.")
print("=" * 60)
