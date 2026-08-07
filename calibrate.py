#!/usr/bin/env python3
"""
Model-Free Calibration V6.0 - FINAL
با دو دیتاست محلی (IBM و UK) | بدون دانلود | بدون خطا
Constraint: 95% of contexts must have D > 0.45
"""

import os
import sys
import json
import time
import warnings
from itertools import product

# ============================================================
# نصب خودکار کتابخانه‌ها (فقط numpy و pandas)
# ============================================================
def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet"])

try:
    import numpy as np
except ImportError:
    import subprocess
    print("Installing numpy...")
    install("numpy")
    import numpy as np

try:
    import pandas as pd
except ImportError:
    import subprocess
    print("Installing pandas...")
    install("pandas")
    import pandas as pd

warnings.filterwarnings('ignore')

# ============================================================
# توابع اصلی
# ============================================================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def compute_requirements(ctx, a1, a2, a3, a4):
    V, N, G, rho, E = ctx
    b1, b2, b3, b4 = 1 - a1, 1 - a2, 1 - a3, 1 - a4
    r_interp = a1 * (1 - sigmoid(10 * (E - 0.5))) + b1 * rho
    r_robust = a2 * sigmoid(12 * (N - 0.35)) + b2 * tanh(2 * rho)
    r_scal = a3 * tanh(3 * V) + b3 * G
    r_rep = a4 * G + b4 * E
    return np.array([r_interp, r_robust, r_scal, r_rep])

def compute_all_sensitivities(ctx, a1, a2, a3, a4, delta=1e-5):
    sens = np.zeros((4, 5))
    for req_idx in range(4):
        for var_idx in range(5):
            def r_func(c):
                return compute_requirements(c, a1, a2, a3, a4)[req_idx]
            c_plus = ctx.copy()
            c_plus[var_idx] = min(1.0, c_plus[var_idx] + delta)
            c_minus = ctx.copy()
            c_minus[var_idx] = max(0.0, c_minus[var_idx] - delta)
            sens[req_idx, var_idx] = (r_func(c_plus) - r_func(c_minus)) / (2 * delta)
    return sens

def compute_dominance_ratio(ctx, a1, a2, a3, a4):
    pairs = [(0, 4, 3), (1, 1, 3), (2, 0, 2), (3, 2, 4)]
    D = np.zeros(4)
    sm = compute_all_sensitivities(ctx, a1, a2, a3, a4)
    for idx, (req, prim, sec) in enumerate(pairs):
        sp = abs(sm[req, prim])
        ss = abs(sm[req, sec])
        D[idx] = sp / (sp + ss + 1e-12)
    return D

# ============================================================
# بارگذاری دیتاست‌های محلی
# ============================================================
def load_local_datasets():
    """
    بارگذاری دیتاست‌های IBM و UK از فایل‌های محلی.
    نام فایل‌ها:
        - IBM: WA_Fn-UseC_-Accounts-Receivable.csv
        - UK:  payment-practices.csv
    """
    datasets = []
    
    # مسیرهای احتمالی
    ibm_paths = [
        "WA_Fn-UseC_-Accounts-Receivable.csv",
        "dataset/WA_Fn-UseC_-Accounts-Receivable.csv",
        "data/WA_Fn-UseC_-Accounts-Receivable.csv"
    ]
    uk_paths = [
        "payment-practices.csv",
        "dataset/payment-practices.csv",
        "data/payment-practices.csv"
    ]
    
    # پیدا کردن IBM
    ibm_file = None
    for p in ibm_paths:
        if os.path.exists(p):
            ibm_file = p
            break
    if ibm_file:
        print(f"✅ Found IBM dataset: {ibm_file}")
        df = pd.read_csv(ibm_file)
        datasets.append(("IBM", df, "target"))
    else:
        print("⚠️ IBM dataset not found locally.")
    
    # پیدا کردن UK
    uk_file = None
    for p in uk_paths:
        if os.path.exists(p):
            uk_file = p
            break
    if uk_file:
        print(f"✅ Found UK dataset: {uk_file}")
        df = pd.read_csv(uk_file)
        datasets.append(("UK", df, "target"))
    else:
        print("⚠️ UK dataset not found locally.")
    
    if not datasets:
        print("❌ No local datasets found! Using synthetic contexts only.")
    
    return datasets

# ============================================================
# استخراج بافت از DataFrame
# ============================================================
def compute_context_from_df(df, target_col='target', E=0.5):
    X = df.drop(columns=[target_col], errors='ignore')
    n, p = X.shape
    V = np.clip(np.log10(max(n, 1)) / 6, 0, 1)
    rho = np.clip(p / max(n, 1), 0, 1)
    missing = X.isnull().sum().sum() / (n * p) if n * p > 0 else 0
    outlier_ratio = 0
    num_cols = X.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        std = X[col].std()
        if std > 0:
            outliers = ((X[col] - X[col].mean()).abs() > 3 * std).sum()
            outlier_ratio += outliers / max(n, 1)
    outlier_ratio = outlier_ratio / max(1, len(num_cols))
    N = np.clip(0.5 * missing + 0.5 * outlier_ratio, 0, 1)
    date_cols = X.select_dtypes(include=['datetime64']).columns
    if len(date_cols) > 0:
        try:
            dates = X[date_cols[0]].dropna().sort_values()
            if len(dates) > 1:
                deltas = dates.diff().dropna()
                median_delta = deltas.median().total_seconds()
                G = np.clip(86400 / max(median_delta, 86400), 0, 1)
            else:
                G = 0.5
        except:
            G = 0.5
    else:
        G = np.clip(np.log10(max(n, 1)) / 6, 0, 1)
    return np.array([V, N, G, rho, E])

def extract_contexts_from_datasets(datasets):
    """استخراج بافت از دیتاست‌های محلی با ۵ سطح مختلف E"""
    all_ctxs = []
    E_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    for name, df, target_col in datasets:
        print(f"\n📊 Processing {name}: {df.shape[0]:,} rows, {df.shape[1]} columns")
        for E in E_levels:
            try:
                ctx = compute_context_from_df(df, target_col, E)
                all_ctxs.append(ctx)
            except Exception as e:
                print(f"  ⚠️ Error at E={E}: {e}")
        print(f"  ✅ Extracted {len(E_levels)} contexts from {name}")
    
    return all_ctxs

# ============================================================
# فضای بافت مصنوعی
# ============================================================
def generate_latin_hypercube(n=3000, seed=42, low=0.05, high=0.95):
    np.random.seed(seed)
    samples = np.zeros((n, 5))
    for j in range(5):
        perm = np.random.permutation(n)
        raw = (perm + np.random.uniform(0, 1, n)) / n
        samples[:, j] = low + (high - low) * raw
    return samples

def generate_grid(steps=7, low=0.05, high=0.95):
    grid = np.linspace(low, high, steps)
    return np.array(list(product(grid, repeat=5)))

# ============================================================
# قیدها
# ============================================================
def check_dominance_quantile(ctxs, a1, a2, a3, a4, thresh=0.45, quantile=0.95):
    all_D = []
    for ctx in ctxs:
        D = compute_dominance_ratio(ctx, a1, a2, a3, a4)
        all_D.append(D)
    all_D = np.array(all_D)
    for req_idx in range(4):
        D_req = all_D[:, req_idx]
        q = np.quantile(D_req, quantile)
        if q <= thresh:
            return False
    return True

def check_bounded(ctxs, a1, a2, a3, a4):
    for ctx in ctxs:
        r = compute_requirements(ctx, a1, a2, a3, a4)
        if np.any(r < 0) or np.any(r > 1):
            return False
    return True

# ============================================================
# جستجوی شبکه‌ای
# ============================================================
def grid_search(ctxs, step=0.05):
    vals = np.arange(0.50, 0.96, step)
    feasible = []
    total = len(vals) ** 4
    count = 0
    print(f"\n🔍 Grid search over {total} combos...")
    start_time = time.time()
    
    for a1 in vals:
        for a2 in vals:
            for a3 in vals:
                for a4 in vals:
                    count += 1
                    if count % 5000 == 0:
                        elapsed = time.time() - start_time
                        print(f"  ⏱️ {count}/{total} | {elapsed:.1f}s")
                    
                    if not check_dominance_quantile(ctxs, a1, a2, a3, a4):
                        continue
                    if not check_bounded(ctxs, a1, a2, a3, a4):
                        continue
                    
                    feasible.append({
                        'a1': round(a1, 2), 'a2': round(a2, 2),
                        'a3': round(a3, 2), 'a4': round(a4, 2),
                        'sum_a': round(a1 + a2 + a3 + a4, 4)
                    })
    
    feasible.sort(key=lambda x: x['sum_a'])
    return feasible

# ============================================================
# تحلیل
# ============================================================
def analyze(feasible):
    if not feasible:
        return {'status': 'NO FEASIBLE'}
    
    best = feasible[0]
    best_sum = best['sum_a']
    near = [c for c in feasible if c['sum_a'] <= best_sum * 1.05]
    
    ranges = {}
    for k in ['a1', 'a2', 'a3', 'a4']:
        vals = [c[k] for c in near]
        ranges[k] = {'min': min(vals), 'max': max(vals), 'std': round(np.std(vals), 4)}
    
    if len(near) == 1:
        status = 'STRONG'
    elif any(ranges[k]['std'] > 0.03 for k in ranges):
        status = 'WEAK'
    else:
        status = 'STABLE'
    
    return {
        'status': status,
        'best': {'a1': best['a1'], 'a2': best['a2'], 'a3': best['a3'], 'a4': best['a4']},
        'best_sum': best_sum,
        'near_count': len(near),
        'ranges': ranges,
        'near_optimal': near[:10]
    }

# ============================================================
# اجرا
# ============================================================
def main():
    print("=" * 80)
    print("🚀 MODEL-FREE CALIBRATION V6.0")
    print("Local datasets (IBM + UK) + Synthetic contexts")
    print("Quantile: 95% | Threshold: 0.45")
    print("=" * 80)

    # --- بارگذاری دیتاست‌های محلی ---
    print("\n📂 Loading local datasets...")
    datasets = load_local_datasets()
    
    # --- استخراج بافت از دیتاست‌های محلی ---
    real_ctxs = []
    if datasets:
        real_ctxs = extract_contexts_from_datasets(datasets)
        print(f"\n✅ {len(real_ctxs)} real contexts extracted.")
    else:
        print("\n⚠️ No datasets found. Using only synthetic contexts.")
    
    # --- فضای مصنوعی ---
    print("\n🧪 Generating synthetic contexts...")
    lhs = generate_latin_hypercube(3000, 42, low=0.05, high=0.95)
    grid = generate_grid(7, low=0.05, high=0.95)
    synth_ctxs = np.vstack([lhs, grid])
    print(f"  ✅ {len(synth_ctxs)} synthetic contexts")
    
    # --- ترکیب ---
    all_ctxs = list(synth_ctxs) + list(real_ctxs)
    print(f"  📊 Total contexts: {len(all_ctxs)}")
    
    # --- جستجو ---
    feasible = grid_search(all_ctxs, step=0.05)
    
    if not feasible:
        print("\n❌ No feasible coefficients found.")
        sys.exit(1)
    
    result = analyze(feasible)
    best = result['best']
    
    # --- خروجی ---
    print("\n" + "=" * 80)
    print("✅ FINAL RESULT")
    print("=" * 80)
    print(f"a1 = {best['a1']:.2f}  (Interpretability)")
    print(f"a2 = {best['a2']:.2f}  (Robustness)")
    print(f"a3 = {best['a3']:.2f}  (Scalability)")
    print(f"a4 = {best['a4']:.2f}  (Rep. Capacity)")
    print(f"Sum = {result['best_sum']:.2f}")
    print(f"Identifiability: {result['status']}")
    
    if result['status'] == 'WEAK':
        print("\n  Near-optimal ranges:")
        for k, v in result['ranges'].items():
            print(f"    {k}: [{v['min']:.2f}, {v['max']:.2f}] (std={v['std']:.3f})")
    
    # --- ذخیره JSON ---
    os.makedirs('output', exist_ok=True)
    report = {
        'methodology': 'Quantile-Based (95% contexts D > 0.45)',
        'datasets_used': [name for name, _, _ in datasets] if datasets else ['NONE'],
        'context_range': '[0.05, 0.95]',
        'synthetic_contexts': len(synth_ctxs),
        'real_contexts': len(real_ctxs),
        'total_contexts': len(all_ctxs),
        'selected': best,
        'identifiability': result['status'],
        'near_optimal_count': result['near_count'],
        'coefficient_ranges': result['ranges'],
        'near_optimal_sets': result['near_optimal']
    }
    with open('output/best_coefficients.json', 'w') as f:
        json.dump(report, f, indent=4)
    
    print("\n📁 Report saved to output/best_coefficients.json")
    print("=" * 80)

if __name__ == "__main__":
    main()
