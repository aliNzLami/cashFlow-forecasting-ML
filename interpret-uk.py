# ============================================
# UK PAYMENT PRACTICES - INTERPRETABILITY ANALYSIS (FIXED v2)
# Time-Series Validated + Grouped Split by Company Number
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
print("UK PAYMENT PRACTICES - INTERPRETABILITY (TIME-VALIDATED + GROUPED)")
print("=" * 60)

# ============================================
# 2. DATA LOADING
# ============================================
print("\n[1] LOADING DATASET...")

INPUT_FILE = 'uk_payment_practices.csv'
OUTPUT_DIR = './results_uk_interpretability_fixed'

os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    df = pd.read_csv(INPUT_FILE)
    print(f"   ✅ Dataset loaded successfully!")
    print(f"   📊 Total records: {len(df):,}")
except FileNotFoundError:
    print(f"   ❌ Error: File '{INPUT_FILE}' not found!")
    exit()

# ============================================
# 3. DATA CLEANING & PREPROCESSING
# ============================================
print("\n[2] DATA CLEANING & PREPROCESSING...")

df.columns = df.columns.str.strip()

# Convert date columns
df['Start date'] = pd.to_datetime(df['Start date'], errors='coerce')
df['End date'] = pd.to_datetime(df['End date'], errors='coerce')

# Essential columns for feature engineering
essential_cols = [
    'Average time to pay',
    'E-Invoicing offered',
    'Supply-chain financing offered',
    'Participates in payment codes',
    'Company number',
    'Start date'
]

# Drop rows missing essential data
df_clean = df.dropna(subset=essential_cols)
print(f"   ✅ Records after essential cleaning: {len(df_clean):,}")

# ============================================
# 4. FEATURE ENGINEERING
# ============================================
print("\n[3] FEATURE ENGINEERING...")

# Target: Percentage of 60-day threshold utilised
THRESHOLD = 60
df_clean['Target'] = (df_clean['Average time to pay'] / THRESHOLD) * 100

# Binary feature conversion
bool_columns = [
    'E-Invoicing offered',
    'Supply-chain financing offered',
    'Participates in payment codes',
    'Payment terms have changed'
]

for col in bool_columns:
    if col in df_clean.columns:
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

print(f"   📊 Feature set: {feature_columns}")

# ============================================
# ===== CHANGE 1: GROUPED TIME-SERIES SPLIT =====
# ============================================
print("\n[4] GROUPED TIME-SERIES TRAIN-TEST SPLIT...")

# For each company, find the earliest reporting date
company_first_date = df_clean.groupby('Company number')['Start date'].min().reset_index()
company_first_date = company_first_date.sort_values('Start date')

# 70% oldest companies -> Train, 30% newest -> Test
split_idx = int(len(company_first_date) * 0.7)
train_companies = company_first_date.iloc[:split_idx]['Company number'].tolist()
test_companies = company_first_date.iloc[split_idx:]['Company number'].tolist()

print(f"   📊 Number of companies: {len(company_first_date)}")
print(f"   📊 Train companies: {len(train_companies)} (up to {company_first_date.iloc[split_idx-1]['Start date'].date()})")
print(f"   📊 Test companies: {len(test_companies)} (from {company_first_date.iloc[split_idx]['Start date'].date()})")

train_df = df_clean[df_clean['Company number'].isin(train_companies)]
test_df = df_clean[df_clean['Company number'].isin(test_companies)]

print(f"   📊 Train records: {len(train_df)}")
print(f"   📊 Test records: {len(test_df)}")

# ============================================
# ===== CHANGE 2: IMPUTATION USING TRAIN STATISTICS =====
# ============================================
print("\n[5] IMPUTING MISSING VALUES (USING TRAIN STATISTICS)...")

X_train_raw = train_df[feature_columns].copy()
y_train = train_df[target_column].copy()
X_test_raw = test_df[feature_columns].copy()
y_test = test_df[target_column].copy()

# Impute numeric columns with median from train
numeric_cols = ['Shortest (or only) standard payment period', 'Longest standard payment period']
for col in numeric_cols:
    if col in X_train_raw.columns:
        median_val = X_train_raw[col].median()
        X_train_raw[col] = X_train_raw[col].fillna(median_val)
        X_test_raw[col] = X_test_raw[col].fillna(median_val)
        print(f"   ✅ Imputed '{col}' with median = {median_val:.2f} (from Train)")

# Impute boolean columns with mode from train
bool_cols = ['Payment terms have changed']
for col in bool_cols:
    if col in X_train_raw.columns:
        mode_val = X_train_raw[col].mode().iloc[0] if not X_train_raw[col].mode().empty else 0
        X_train_raw[col] = X_train_raw[col].fillna(mode_val)
        X_test_raw[col] = X_test_raw[col].fillna(mode_val)
        print(f"   ✅ Imputed '{col}' with mode = {mode_val} (from Train)")

# Ensure all boolean columns are ints
for col in bool_columns:
    if col in X_train_raw.columns:
        X_train_raw[col] = X_train_raw[col].astype(int)
        X_test_raw[col] = X_test_raw[col].astype(int)

# ============================================
# 6. DATA SCALING (FIT ON TRAIN ONLY)
# ============================================
print("\n[6] DATA SCALING...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

X_train_torch = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_torch = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_torch = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_torch = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# For tree models (unscaled)
X_train_tree = X_train_raw.values
X_test_tree = X_test_raw.values

# ============================================
# 7. TRAIN SCIKIT-LEARN MODELS
# ============================================
print("\n[7] TRAINING SCIKIT-LEARN MODELS (on time-validated split)...")

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)

# Tree-based models (unscaled)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_tree, y_train)

xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
xgb_model.fit(X_train_tree, y_train)

lgb_model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
lgb_model.fit(X_train_tree, y_train)

print("   ✅ All Scikit-Learn models trained.")

# ============================================
# 8. TRAIN PYTORCH NEURAL NETWORK
# ============================================
print("\n[8] TRAINING PYTORCH NEURAL NETWORK...")

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
# 9. INTERPRETABILITY: LINEAR REGRESSION
# ============================================
print("\n[9] LINEAR REGRESSION COEFFICIENTS")

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
coef_df.to_csv(f'{OUTPUT_DIR}/uk_linear_coefficients_fixed.csv', index=False)
print(f"\n   💾 Saved to: {OUTPUT_DIR}/uk_linear_coefficients_fixed.csv")

# ============================================
# 10. INTERPRETABILITY: TREE-BASED
# ============================================
print("\n[10] TREE-BASED INTERPRETABILITY")
print("   Methods: TreeSHAP, TreeInterpreter (native), LIME")

tree_models = {
    'Random Forest': rf,
    'XGBoost': xgb_model,
    'LightGBM': lgb_model
}

tree_results = []

random.seed(42)
sample_indices = random.sample(range(len(X_test_raw)), min(30, len(X_test_raw)))

for name, model in tree_models.items():
    print(f"\n   🔹 {name}:")
    
    # TreeSHAP
    try:
        explainer_shap = shap.TreeExplainer(model)
        shap_values = explainer_shap.shap_values(X_test_tree)
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
    
    # TreeInterpreter
    try:
        print(f"      📊 Computing TreeInterpreter contributions...")
        all_contribs = []
        
        for idx in sample_indices:
            sample_x = X_test_tree[idx].reshape(1, -1)
            contribs = None
            
            if isinstance(model, RandomForestRegressor):
                if ti is not None:
                    _, _, contributions = ti.predict(model, sample_x)
                    contribs = contributions[0]
                else:
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
    
    # LIME
    try:
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train_tree,
            feature_names=feature_columns,
            mode='regression'
        )
        lime_values = []
        for idx in sample_indices:
            exp = lime_explainer.explain_instance(
                X_test_tree[idx],
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

tree_df = pd.DataFrame(tree_results)
pivot_tree = tree_df.pivot_table(
    index=['Model', 'Feature'],
    columns='Method',
    values='Value'
).reset_index()

print("\n   " + "-" * 70)
print(pivot_tree.to_string(index=False))
print("   " + "-" * 70)
pivot_tree.to_csv(f'{OUTPUT_DIR}/uk_tree_interpretability_fixed.csv', index=False)
print(f"\n   💾 Saved to: {OUTPUT_DIR}/uk_tree_interpretability_fixed.csv")

# ============================================
# 11. INTERPRETABILITY: NEURAL NETWORK
# ============================================
print("\n[11] NEURAL NETWORK INTERPRETABILITY")
print("   Methods: Integrated Gradients, DeepLIFT, LIME")

nn_results = []

# Integrated Gradients
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

# DeepLIFT
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

# LIME for NN
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

nn_df = pd.DataFrame(nn_results)
pivot_nn = nn_df.pivot_table(
    index=['Model', 'Feature'],
    columns='Method',
    values='Value'
).reset_index()

print("\n   " + "-" * 70)
print(pivot_nn.to_string(index=False))
print("   " + "-" * 70)
pivot_nn.to_csv(f'{OUTPUT_DIR}/uk_nn_interpretability_fixed.csv', index=False)
print(f"\n   💾 Saved to: {OUTPUT_DIR}/uk_nn_interpretability_fixed.csv")

# ============================================
# 12. FINAL SUMMARY
# ============================================
print("\n" + "=" * 60)
print("🎯 UK INTERPRETABILITY ANALYSIS COMPLETE (FIXED VERSION)!")
print("=" * 60)

print(f"\n📁 All results saved to: {OUTPUT_DIR}/")
print("   📄 uk_linear_coefficients_fixed.csv - Linear Regression")
print("   📄 uk_tree_interpretability_fixed.csv - Tree-based (SHAP, TI, LIME)")
print("   📄 uk_nn_interpretability_fixed.csv - Neural Network (IG, DL, LIME)")

print("\n🚀 Interpretability results ready for research paper.")
print("=" * 60)
