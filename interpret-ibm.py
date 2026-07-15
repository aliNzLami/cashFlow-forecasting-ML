# ============================================
# IBM LATE PAYMENT HISTORIES - INTERPRETABILITY ANALYSIS (FIXED v2)
# Time-Series Validated + Removed Leaky Features
# ============================================

# ============================================
# 1. IMPORTS
# ============================================
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from captum.attr import IntegratedGradients, DeepLift
import shap
import lime
import lime.lime_tabular
import warnings
import os
import random

try:
    from treeinterpreter import treeinterpreter as ti
except ImportError:
    ti = None
    print("⚠️ treeinterpreter not installed. Install with: pip install treeinterpreter")

warnings.filterwarnings('ignore')

print("=" * 60)
print("IBM LATE PAYMENT HISTORIES - INTERPRETABILITY (TIME-VALIDATED)")
print("=" * 60)

# ============================================
# 2. DATA LOADING
# ============================================
print("\n[1] LOADING DATASET...")

INPUT_FILE = 'WA_Fn-UseC_-Accounts-Receivable.csv'
OUTPUT_DIR = './results_ibm_interpretability_fixed'

os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    df = pd.read_csv(INPUT_FILE)
    print(f"   ✅ Dataset loaded successfully!")
    print(f"   📊 Total records: {len(df)}")
except FileNotFoundError:
    print(f"   ❌ Error: File '{INPUT_FILE}' not found!")
    exit()

# ============================================
# 3. DATA CLEANING & PREPROCESSING
# ============================================
print("\n[2] DATA CLEANING & PREPROCESSING...")

df.columns = df.columns.str.strip()

# ===== CHANGE 1: REMOVE LEAKY FEATURES =====
# DaysLate and Disputed are NOT available at invoice issuance time.
# We keep only: InvoiceAmount, CreditPeriod, PaperlessBill
print("\n   ⚠️ REMOVING LEAKY FEATURES: 'DaysLate' and 'Disputed'")
print("   📌 These are not available at the time of invoice issuance.")

feature_columns = [
    'InvoiceAmount',
    'CreditPeriod',
    'PaperlessBill'
]
target_column = 'DaysToSettle'

# ============================================
# 4. FEATURE ENGINEERING
# ============================================
print("\n[3] FEATURE ENGINEERING...")

# Convert dates
date_cols = ['InvoiceDate', 'DueDate', 'SettledDate']
for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

# Credit Period
df['CreditPeriod'] = (df['DueDate'] - df['InvoiceDate']).dt.days

# PaperlessBill: Paper=1, Electronic=0
if 'PaperlessBill' in df.columns:
    df['PaperlessBill'] = df['PaperlessBill'].map({'Paper': 1, 'Electronic': 0}).fillna(0).astype(int)

# Keep only records with valid target and features
df_clean = df[feature_columns + [target_column, 'InvoiceDate']].dropna()
print(f"   ✅ Records after cleaning: {len(df_clean)}")

# ============================================
# ===== CHANGE 2: TIME-SERIES SPLIT =====
# ============================================
print("\n[4] TIME-SERIES TRAIN-TEST SPLIT...")

# Sort by InvoiceDate (oldest to newest)
df_clean = df_clean.sort_values('InvoiceDate').reset_index(drop=True)

# Chronological split: first 70% train, last 30% test
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
# 5. SCALING (FIT ON TRAIN ONLY)
# ============================================
print("\n[5] DATA SCALING...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# For PyTorch
X_train_torch = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_torch = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_torch = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_torch = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# ============================================
# 6. TRAIN SCIKIT-LEARN MODELS
# ============================================
print("\n[6] TRAINING SCIKIT-LEARN MODELS (on time-validated split)...")

# Linear Regression (uses scaled data)
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)

# Tree-based models (use unscaled data)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
xgb_model.fit(X_train, y_train)

lgb_model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
lgb_model.fit(X_train, y_train)

print("   ✅ All Scikit-Learn models trained.")

# ============================================
# 7. TRAIN PYTORCH NEURAL NETWORK
# ============================================
print("\n[7] TRAINING PYTORCH NEURAL NETWORK...")

class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

input_dim = X_train_scaled.shape[1]
nn_model = SimpleNN(input_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(nn_model.parameters(), lr=0.001)

batch_size = 64
train_dataset = TensorDataset(X_train_torch, y_train_torch)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

epochs = 100
nn_model.train()
for epoch in range(epochs):
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = nn_model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if (epoch+1) % 20 == 0:
        print(f"   Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")

print("   ✅ PyTorch Neural Network trained.")

# ============================================
# 8. INTERPRETABILITY: LINEAR REGRESSION (Table 8)
# ============================================
print("\n[8] LINEAR REGRESSION COEFFICIENTS (Table 8)")

coef_df = pd.DataFrame({
    'Feature': feature_columns,
    'Coefficient': lin_reg.coef_,
    'Impact Direction': ['increases' if c > 0 else 'decreases' for c in lin_reg.coef_],
    'Magnitude': np.abs(lin_reg.coef_)
})
coef_df = coef_df.sort_values('Magnitude', ascending=False)

print("\n   " + "-" * 70)
print(coef_df.to_string(index=False))
print("   " + "-" * 70)
coef_df.to_csv(f'{OUTPUT_DIR}/ibm_linear_coefficients_fixed.csv', index=False)
print(f"\n   💾 Saved to: {OUTPUT_DIR}/ibm_linear_coefficients_fixed.csv")

# ============================================
# 9. INTERPRETABILITY: TREE-BASED (Tables 4 & 5)
# ============================================
print("\n[9] TREE-BASED INTERPRETABILITY (Tables 4 & 5)")
print("   Methods: TreeSHAP, TreeInterpreter (native), LIME")

tree_models = {
    'Random Forest': rf,
    'XGBoost': xgb_model,
    'LightGBM': lgb_model
}

tree_results = []

# Sample subset for LIME and TreeInterpreter
random.seed(42)
sample_indices = random.sample(range(len(X_test)), min(30, len(X_test)))

for name, model in tree_models.items():
    print(f"\n   🔹 {name}:")
    
    # ---- TreeSHAP ----
    try:
        explainer_shap = shap.TreeExplainer(model)
        shap_values = explainer_shap.shap_values(X_test.values)
        mean_shap = np.abs(shap_values).mean(axis=0)
        for i, feat in enumerate(feature_columns):
            tree_results.append({
                'Model': name,
                'Feature': feat,
                'Method': 'TreeSHAP',
                'Value': mean_shap[i]
            })
        print(f"      ✅ TreeSHAP computed.")
    except Exception as e:
        print(f"      ⚠️ TreeSHAP failed: {e}")
    
    # ---- TreeInterpreter ----
    try:
        print(f"      📊 Computing TreeInterpreter contributions...")
        all_contribs = []
        
        for idx in sample_indices:
            sample_x = X_test.iloc[idx].values.reshape(1, -1)
            contribs = None
            
            if isinstance(model, RandomForestRegressor):
                if ti is not None:
                    _, _, contributions = ti.predict(model, sample_x)
                    contribs = contributions[0]
                else:
                    print(f"         ⚠️ treeinterpreter not installed; skipping RF")
                    break
            elif isinstance(model, xgb.XGBRegressor):
                dmat = xgb.DMatrix(sample_x, feature_names=feature_columns)
                contribs_array = model.get_booster().predict(dmat, pred_contribs=True)
                contribs = contribs_array[0][:-1]
            elif isinstance(model, lgb.LGBMRegressor):
                contribs_array = model.predict(sample_x, pred_contrib=True)
                contribs = contribs_array[0][:-1]
            else:
                continue
            
            if contribs is not None:
                all_contribs.append(contribs)
        
        if all_contribs:
            mean_contribs = np.mean(np.abs(all_contribs), axis=0)
            for i, feat in enumerate(feature_columns):
                tree_results.append({
                    'Model': name,
                    'Feature': feat,
                    'Method': 'TreeInterpreter',
                    'Value': mean_contribs[i]
                })
            print(f"      ✅ TreeInterpreter computed on {len(all_contribs)} samples.")
        else:
            for i, feat in enumerate(feature_columns):
                tree_results.append({
                    'Model': name,
                    'Feature': feat,
                    'Method': 'TreeInterpreter',
                    'Value': np.nan
                })
            print(f"      ⚠️ No samples processed.")
            
    except Exception as e:
        print(f"      ⚠️ TreeInterpreter failed: {e}")
        for i, feat in enumerate(feature_columns):
            tree_results.append({
                'Model': name,
                'Feature': feat,
                'Method': 'TreeInterpreter',
                'Value': np.nan
            })
    
    # ---- LIME ----
    try:
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train.values,
            feature_names=feature_columns,
            mode='regression',
            training_data_stats=None  # Ensure no leakage
        )
        lime_values = []
        for idx in sample_indices:
            exp = lime_explainer.explain_instance(
                X_test.iloc[idx].values,
                model.predict,
                num_features=len(feature_columns)
            )
            weights_dict = {}
            for key, val in exp.as_list():
                matched_feature = None
                for feat in feature_columns:
                    if feat in key:
                        matched_feature = feat
                        break
                if matched_feature:
                    weights_dict[matched_feature] = val
            lime_values.append(weights_dict)
        
        lime_agg = {}
        for feat in feature_columns:
            vals = [abs(d.get(feat, 0)) for d in lime_values]
            lime_agg[feat] = np.mean(vals) if vals else 0
        
        for i, feat in enumerate(feature_columns):
            tree_results.append({
                'Model': name,
                'Feature': feat,
                'Method': 'LIME',
                'Value': lime_agg.get(feat, 0)
            })
        print(f"      ✅ LIME computed on {len(sample_indices)} samples.")
    except Exception as e:
        print(f"      ⚠️ LIME failed: {e}")

# Save tree interpretability results
tree_df = pd.DataFrame(tree_results)
pivot_tree = tree_df.pivot_table(
    index=['Model', 'Feature'],
    columns='Method',
    values='Value'
).reset_index()

print("\n   " + "-" * 70)
print(pivot_tree.to_string(index=False))
print("   " + "-" * 70)
pivot_tree.to_csv(f'{OUTPUT_DIR}/ibm_tree_interpretability_fixed.csv', index=False)
print(f"\n   💾 Saved to: {OUTPUT_DIR}/ibm_tree_interpretability_fixed.csv")

# ============================================
# 10. INTERPRETABILITY: NEURAL NETWORK (Tables 6 & 7)
# ============================================
print("\n[10] NEURAL NETWORK INTERPRETABILITY (Tables 6 & 7)")
print("   Methods: Integrated Gradients, DeepLIFT, LIME")

nn_results = []

# ---- Integrated Gradients ----
try:
    ig = IntegratedGradients(nn_model)
    X_test_batch = X_test_torch[:min(100, len(X_test_torch))]
    attributions_ig = ig.attribute(X_test_batch, n_steps=50)
    mean_ig = attributions_ig.abs().mean(dim=0).detach().numpy()
    for i, feat in enumerate(feature_columns):
        nn_results.append({
            'Model': 'Neural Network',
            'Feature': feat,
            'Method': 'Integrated Gradients',
            'Value': mean_ig[i]
        })
    print(f"   ✅ Integrated Gradients computed.")
except Exception as e:
    print(f"   ⚠️ Integrated Gradients failed: {e}")

# ---- DeepLIFT ----
try:
    dl = DeepLift(nn_model)
    X_test_batch = X_test_torch[:min(100, len(X_test_torch))]
    attributions_dl = dl.attribute(X_test_batch)
    mean_dl = attributions_dl.abs().mean(dim=0).detach().numpy()
    for i, feat in enumerate(feature_columns):
        nn_results.append({
            'Model': 'Neural Network',
            'Feature': feat,
            'Method': 'DeepLIFT',
            'Value': mean_dl[i]
        })
    print(f"   ✅ DeepLIFT computed.")
except Exception as e:
    print(f"   ⚠️ DeepLIFT failed: {e}")

# ---- LIME for Neural Network ----
try:
    def nn_predict_proba(X):
        nn_model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            preds = nn_model(X_tensor).numpy().flatten()
        return preds
    
    lime_explainer_nn = lime.lime_tabular.LimeTabularExplainer(
        X_train_scaled,
        feature_names=feature_columns,
        mode='regression'
    )
    lime_values_nn = []
    for idx in sample_indices:
        exp = lime_explainer_nn.explain_instance(
            X_test_scaled[idx],
            nn_predict_proba,
            num_features=len(feature_columns)
        )
        weights_dict = {}
        for key, val in exp.as_list():
            matched_feature = None
            for feat in feature_columns:
                if feat in key:
                    matched_feature = feat
                    break
            if matched_feature:
                weights_dict[matched_feature] = val
        lime_values_nn.append(weights_dict)
    
    lime_agg_nn = {}
    for feat in feature_columns:
        vals = [abs(d.get(feat, 0)) for d in lime_values_nn]
        lime_agg_nn[feat] = np.mean(vals) if vals else 0
    
    for i, feat in enumerate(feature_columns):
        nn_results.append({
            'Model': 'Neural Network',
            'Feature': feat,
            'Method': 'LIME',
            'Value': lime_agg_nn.get(feat, 0)
        })
    print(f"   ✅ LIME for NN computed on {len(sample_indices)} samples.")
except Exception as e:
    print(f"   ⚠️ LIME for NN failed: {e}")

# Save NN interpretability results
nn_df = pd.DataFrame(nn_results)
pivot_nn = nn_df.pivot_table(
    index=['Model', 'Feature'],
    columns='Method',
    values='Value'
).reset_index()

print("\n   " + "-" * 70)
print(pivot_nn.to_string(index=False))
print("   " + "-" * 70)
pivot_nn.to_csv(f'{OUTPUT_DIR}/ibm_nn_interpretability_fixed.csv', index=False)
print(f"\n   💾 Saved to: {OUTPUT_DIR}/ibm_nn_interpretability_fixed.csv")

# ============================================
# 11. FINAL SUMMARY
# ============================================
print("\n" + "=" * 60)
print("🎯 IBM INTERPRETABILITY ANALYSIS COMPLETE (FIXED VERSION)!")
print("=" * 60)

print(f"\n📁 All results saved to: {OUTPUT_DIR}/")
print("   📄 ibm_linear_coefficients_fixed.csv - Table 8 (Linear Regression)")
print("   📄 ibm_tree_interpretability_fixed.csv - Tables 4 & 5 (TreeSHAP, TreeInterpreter, LIME)")
print("   📄 ibm_nn_interpretability_fixed.csv - Tables 6 & 7 (Integrated Gradients, DeepLIFT, LIME)")

print("\n🚀 Interpretability results ready for research paper.")
print("=" * 60)
