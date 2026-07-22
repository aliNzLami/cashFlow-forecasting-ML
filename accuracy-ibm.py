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
import warnings
import os

warnings.filterwarnings('ignore')

# ============================================
# [NEW] 1.5. BOOTSTRAP FUNCTION FOR UNCERTAINTY
# ============================================
def bootstrap_metrics(y_true, y_pred, n_bootstrap=1000, random_state=42):
    """
    Calculates standard deviation and 95% CI for RMSE, MAE, R2 
    using bootstrap resampling on the test set indices.
    """
    np.random.seed(random_state)
    n = len(y_true)
    # Convert to numpy arrays for faster indexing
    y_true_np = y_true.values if hasattr(y_true, 'values') else np.array(y_true)
    y_pred_np = y_pred if isinstance(y_pred, np.ndarray) else np.array(y_pred)
    
    rmse_list = []
    mae_list = []
    r2_list = []
    
    for _ in range(n_bootstrap):
        # Sample with replacement from test indices
        idx = np.random.choice(n, n, replace=True)
        y_true_bs = y_true_np[idx]
        y_pred_bs = y_pred_np[idx]
        
        rmse_list.append(np.sqrt(mean_squared_error(y_true_bs, y_pred_bs)))
        mae_list.append(mean_absolute_error(y_true_bs, y_pred_bs))
        r2_list.append(r2_score(y_true_bs, y_pred_bs))
    
    return {
        'RMSE': {
            'mean': np.mean(rmse_list),
            'std': np.std(rmse_list),
            'ci_lower': np.percentile(rmse_list, 2.5),
            'ci_upper': np.percentile(rmse_list, 97.5)
        },
        'MAE': {
            'mean': np.mean(mae_list),
            'std': np.std(mae_list),
            'ci_lower': np.percentile(mae_list, 2.5),
            'ci_upper': np.percentile(mae_list, 97.5)
        },
        'R2': {
            'mean': np.mean(r2_list),
            'std': np.std(r2_list),
            'ci_lower': np.percentile(r2_list, 2.5),
            'ci_upper': np.percentile(r2_list, 97.5)
        }
    }

print("=" * 60)
print("CASH FLOW FORECASTING FOR SMES - IBM DATASET (FIXED v2)")
print("=" * 60)

# ============================================
# 2. DATA LOADING
# ============================================
print("\n[1] LOADING DATASET...")

INPUT_FILE = 'WA_Fn-UseC_-Accounts-Receivable.csv'  
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

# ============================================
# 4. FEATURE ENGINEERING (✅ CORRECTED)
# ============================================
print("\n[3] FEATURE ENGINEERING...")

# datetime
date_cols = ['InvoiceDate', 'DueDate', 'SettledDate']
for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        print(f"   ✅ {col} converted to datetime.")

# Credit Period 
df['CreditPeriod'] = (df['DueDate'] - df['InvoiceDate']).dt.days

# Disputed & PaperlessBill 
if 'Disputed' in df.columns:
    df['Disputed'] = df['Disputed'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
    print(f"   ✅ Disputed converted to binary.")
if 'PaperlessBill' in df.columns:
    # Paper=1, Electronic=0
    df['PaperlessBill'] = df['PaperlessBill'].map({'Paper': 1, 'Electronic': 0}).fillna(0).astype(int)
    print(f"   ✅ PaperlessBill converted to binary.")

# ============================================
# 👇👇👇 CRITICAL CHANGE: TARGET & FEATURES DEFINITION
# ============================================
# ===== CHANGE 1: REMOVING 'DaysLate' and 'Disputed' from features =====
# Because 'DaysLate' and 'Disputed' are not known at the time the invoice is issued.
# Objective: To predict the number of days until full settlement (DaysToSettle) at the time of invoice issuance.
feature_columns = [
    'InvoiceAmount',      
    'CreditPeriod',       
    'PaperlessBill'       
]
target_column = 'DaysToSettle'

df_clean = df[feature_columns + [target_column, 'InvoiceDate']].dropna()
print(f"   ✅ Records after cleaning: {len(df_clean)}")

# ============================================
# 👇👇👇 CRITICAL CHANGE: TIME-SERIES SPLIT
# ============================================
# ===== CHANGE 2: SORTING BY DATE AND CHRONOLOGICAL SPLIT =====
print("\n[4] TIME-SERIES TRAIN-TEST SPLIT...")

df_clean = df_clean.sort_values('InvoiceDate').reset_index(drop=True)

split_idx = int(len(df_clean) * 0.7)
train_data = df_clean.iloc[:split_idx]
test_data = df_clean.iloc[split_idx:]

X_train = train_data[feature_columns]
y_train = train_data[target_column]
X_test = test_data[feature_columns]
y_test = test_data[target_column]

print(f"   📊 Train period: {train_data['InvoiceDate'].min()} to {train_data['InvoiceDate'].max()}")
print(f"   📊 Test period:  {test_data['InvoiceDate'].min()} to {test_data['InvoiceDate'].max()}")
print(f"   📊 Training records: {len(X_train)}")
print(f"   📊 Test records: {len(X_test)}")

# ============================================
# 5. DATA SCALING (FIT ON TRAIN ONLY)
# ============================================
print("\n[5] DATA SCALING...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================
# 6. MODEL TRAINING & EVALUATION (WITH BOOTSTRAP)
# ============================================
print("\n[6] TRAINING MACHINE LEARNING MODELS...")

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
    'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
    'Neural Network': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
}

results = []
predictions = {}
trained_models = {}

for name, model in models.items():
    print(f"   ▶️ Training {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    predictions[name] = y_pred
    trained_models[name] = model  # [NEW] برای ذخیره اگر لازم شد
    
    # --- Original metrics ---
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # ============================================
    # [NEW] BOOTSTRAP UNCERTAINTY CALCULATION
    # ============================================
    bs = bootstrap_metrics(y_test, y_pred, n_bootstrap=500)  # 500 for speed
    
    results.append({
        'Model': name,
        'RMSE': rmse,
        'RMSE_std': bs['RMSE']['std'],
        'MAE': mae,
        'MAE_std': bs['MAE']['std'],
        'R2': r2,
        'R2_std': bs['R2']['std']
    })
    
    # Print with ± format for quick review
    print(f"      ✅ {name}: RMSE = {rmse:.4f} (±{bs['RMSE']['std']:.4f}), "
          f"MAE = {mae:.4f} (±{bs['MAE']['std']:.4f}), "
          f"R2 = {r2:.4f} (±{bs['R2']['std']:.4f})")

# ============================================
# 7. RESULTS TABLE (WITH UNCERTAINTY)
# ============================================
print("\n[7] RESULTS - MODEL COMPARISON (WITH STD DEVIATION)")

results_df = pd.DataFrame(results).sort_values('RMSE')

# Create a formatted string for LaTeX/Table
print("\n   📊 Table: Model Performance Comparison (IBM - with Bootstrap Std):")
print("   " + "-" * 90)
print(f"   {'Model':<20} {'RMSE (±std)':<20} {'MAE (±std)':<20} {'R² (±std)'}")
print("   " + "-" * 90)
for _, row in results_df.iterrows():
    print(f"   {row['Model']:<20} {row['RMSE']:.4f} ± {row['RMSE_std']:.4f}   "
          f"{row['MAE']:.4f} ± {row['MAE_std']:.4f}   "
          f"{row['R2']:.4f} ± {row['R2_std']:.4f}")
print("   " + "-" * 90)

# Save to CSV with std columns
results_df.to_csv(f'{OUTPUT_DIR}/ibm_results_fixed_with_std.csv', index=False)
print(f"\n   💾 Saved to: {OUTPUT_DIR}/ibm_results_fixed_with_std.csv")

print("\n" + "=" * 60)
print("✅ ANALYSIS COMPLETE. Results now include Bootstrap Standard Deviation.")
print("=" * 60)
