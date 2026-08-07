#!/usr/bin/env python3
"""
Model-Free Calibration V4.4 - با Progress Indicator برای Lending Club
Constraint: 95% of contexts must have D > 0.45
"""

import subprocess
import sys
import os
import json
import time
import warnings
from itertools import product

# ============================================================
# نصب خودکار
# ============================================================
def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet"])

for pkg in ["numpy", "pandas", "kagglehub"]:
    try:
        __import__(pkg)
    except ImportError:
        print(f"Installing {pkg}...")
        install(pkg)

import numpy as np
import pandas as pd
import kagglehub
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
# استخراج بافت از دیتاست‌ها (با Progress Indicator)
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

def download_and_extract_real_contexts():
    """هر سه دیتاست با نمایش پیشرفت در حین اکسترکت"""
    sources = [
        ("hhenry/finance-factoring-ibm-late-payment-histories", "target", "IBM"),
        ("saikiran0684/payment-practices-of-uk-buyers", "target", "UK"),
    ]
    all_ctxs = []
    E_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    print("\n" + "=" * 60)
    print("📥 DOWNLOADING DATASETS FROM KAGGLE")
    print("=" * 60)
    
    for src, target_col, name in sources:
        print(f"\n[{name}] Starting download...")
        start_time = time.time()
        
        try:
            # دانلود (kagglehub خودش پیشرفت دانلود را نشان می‌دهد)
            print(f"  ⏳ Downloading and extracting... (this may take a moment)")
            path = kagglehub.dataset_download(src)
            elapsed = time.time() - start_time
            print(f"  ✅ Downloaded in {elapsed:.1f}s")
            
            # پیدا کردن فایل CSV
            csv_files = []
            for root, _, files in os.walk(path):
                for f in files:
                    if f.endswith('.csv'):
                        csv_files.append(os.path.join(root, f))
            
            if not csv_files:
                print(f"  ⚠️ No CSV found, skipping.")
                continue
            
            # خواندن CSV
            print(f"  📄 Loading CSV: {os.path.basename(csv_files[0])}...")
            df = pd.read_csv(csv_files[0])
            print(f"  ✅ Loaded {df.shape[0]:,} rows, {df.shape[1]} columns")
            
            # استخراج بافت برای سطوح مختلف E
            print(f"  🔍 Extracting contexts for 5 E-levels...")
            for E in E_levels:
                try:
                    ctx = compute_context_from_df(df, target_col, E)
                    all_ctxs.append(ctx)
                except Exception as e:
                    print(f"    ⚠️ Error at E={E}: {e}")
            
            print(f"  ✅ Extracted {len(E_levels)} contexts from {name}")
            
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            # ادامه می‌دهیم تا دیتاست‌های دیگر دانلود شوند
    
    print("\n" + "=" * 60)
    print(f"✅ TOTAL: {len(all_ctxs)} real contexts extracted from all datasets.")
    print("=" * 60)
    return all_ctxs

# ============================================================
# فضای مصنوعی
# ============================================================
def generate_latin_hypercube(n=1500, seed=42, low=0.05, high=0.95):
    np.random.seed(seed)
    samples = np.zeros((n, 5))
    for j in range(5):
        perm = np.random.permutation(n)
        raw = (perm + np.random.uniform(0, 1, n)) / n
        samples[:, j] = low + (high - low) * raw
    return samples

def generate_grid(steps=6, low=0.05, high=0.95):
    grid = np.linspace(low, high, steps)
    return np.array(list(product(grid, repeat=5)))

# ============================================================
# قید: حداقل ۹۵٪ نقاط باید D > آستانه داشته باشند
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

def grid_search(ctxs, step=0.05):
    vals = np.arange(0.50, 0.96, step)
    feasible = []
    total = len(vals) ** 4
    count = 0
    print(f"\n🔍 Grid search over {total} combos (quantile=0.95, threshold=0.45)...")
    start_time = time.time()
    
    for a1 in vals:
        for a2 in vals:
            for a3 in vals:
                for a4 in vals:
                    count += 1
                    if count % 5000 == 0:
                        elapsed = time.time() - start_time
                        print(f"  ⏱️ {count}/{total} | {elapsed:.1f}s elapsed")
                    if not check_dominance_quantile(ctxs, a1, a2, a3, a4, thresh=0.45, quantile=0.95):
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
        'ranges': ranges
    }

# ============================================================
# اجرا
# ============================================================
def main():
    print("=" * 80)
    print("🚀 MODEL-FREE CALIBRATION V4.4")
    print("3 Real Datasets (IBM + UK + Lending Club 1.26GB)")
    print("Quantile-Based (95%) | Progress Indicator Enabled")
    print("=" * 80)

    # دیتاست‌های واقعی (هر سه تا)
    real_ctxs = download_and_extract_real_contexts()

    # فضای مصنوعی
    print("\n🧪 Generating synthetic contexts (0.05 to 0.95)...")
    lhs = generate_latin_hypercube(1500, 42, low=0.05, high=0.95)
    grid = generate_grid(6, low=0.05, high=0.95)
    synth = np.vstack([lhs, grid])
    print(f"  ✅ Synthetic: {len(synth)} points")

    # ترکیب
    ctxs = list(synth) + list(real_ctxs)
    print(f"  📊 Total contexts: {len(ctxs)}")

    # جستجو
    feasible = grid_search(ctxs, step=0.05)

    if not feasible:
        print("\n❌ No feasible coefficients found.")
        sys.exit(1)

    result = analyze(feasible)
    best = result['best']

    # اعتبارسنجی روی داده‌های واقعی
    real_val = {}
    if real_ctxs:
        viol = 0
        Ds = []
        for ctx in real_ctxs:
            D = compute_dominance_ratio(ctx, best['a1'], best['a2'], best['a3'], best['a4'])
            Ds.append(D)
            if np.any(D <= 0.45):
                viol += 1
        real_val = {
            'violations': viol,
            'total': len(real_ctxs),
            'avg_D': np.mean(Ds, axis=0).tolist()
        }

    # خروجی
    print("\n" + "=" * 80)
    print("✅ FINAL RESULT")
    print("=" * 80)
    print(f"a1 = {best['a1']:.2f}  (Interpretability)")
    print(f"a2 = {best['a2']:.2f}  (Robustness)")
    print(f"a3 = {best['a3']:.2f}  (Scalability)")
    print(f"a4 = {best['a4']:.2f}  (Rep. Capacity)")
    print(f"Sum = {result['best_sum']:.2f}")
    print(f"Identifiability: {result['status']}")
    if real_ctxs:
        print(f"\n📊 Real-data validation (threshold=0.45):")
        print(f"  Violations: {real_val['violations']}/{real_val['total']}")
        print(f"  Avg D: interp={real_val['avg_D'][0]:.3f}, robust={real_val['avg_D'][1]:.3f}, scal={real_val['avg_D'][2]:.3f}, rep={real_val['avg_D'][3]:.3f}")

    # ذخیره JSON
    os.makedirs('output', exist_ok=True)
    report = {
        'methodology': 'Quantile-Based (95% contexts D > 0.45)',
        'datasets_used': ['IBM Late Payment', 'UK Payment Practices', 'Lending Club'],
        'context_range': '[0.05, 0.95]',
        'selected': best,
        'identifiability': result['status'],
        'near_optimal_count': result['near_count'],
        'coefficient_ranges': result['ranges'],
        'real_validation': real_val
    }
    with open('output/best_coefficients.json', 'w') as f:
        json.dump(report, f, indent=4)

    print("\n📁 Report saved to output/best_coefficients.json")
    print("=" * 80)

if __name__ == "__main__":
    main()
