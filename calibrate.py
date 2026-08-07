#!/usr/bin/env python3
"""
Model-Free Coefficient Calibration - V4
خودکفا: همه‌ی کتابخانه‌ها را خودش نصب می‌کند.
هیچ requirements.txt یا فایل دیگری نیاز نیست.
"""

import subprocess
import sys
import os
import json
import warnings
from itertools import product

# ============================================================
# 0. نصب خودکار کتابخانه‌های مورد نیاز
# ============================================================
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])

required = ["numpy", "pandas", "kagglehub"]
for pkg in required:
    try:
        __import__(pkg)
    except ImportError:
        print(f"📦 Installing {pkg}...")
        install(pkg)

# حالا ایمپورت می‌کنیم
import numpy as np
import pandas as pd
import kagglehub
warnings.filterwarnings('ignore')

# ============================================================
# 1. توابع اصلی
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
# 2. استخراج بافت از دیتاست‌ها
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
    sources = [
        ("hhenry/finance-factoring-ibm-late-payment-histories", "target"),
        ("saikiran0684/payment-practices-of-uk-buyers", "target"),
        ("wordsforthewise/lending-club", "target")
    ]
    all_ctxs = []
    E_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    print("\n📥 Downloading datasets from Kaggle...")
    for src, target_col in sources:
        try:
            path = kagglehub.dataset_download(src)
            print(f"  ✅ {src}")
            csv_files = []
            for root, _, files in os.walk(path):
                for f in files:
                    if f.endswith('.csv'):
                        csv_files.append(os.path.join(root, f))
            if not csv_files:
                continue
            df = pd.read_csv(csv_files[0])
            for E in E_levels:
                try:
                    all_ctxs.append(compute_context_from_df(df, target_col, E))
                except:
                    pass
        except Exception as e:
            print(f"  ❌ {src}: {e}")
    print(f"  ✅ {len(all_ctxs)} real context points extracted.")
    return all_ctxs

# ============================================================
# 3. فضای بافت مصنوعی
# ============================================================
def generate_latin_hypercube(n=3000, seed=42):
    np.random.seed(seed)
    samples = np.zeros((n, 5))
    for j in range(5):
        perm = np.random.permutation(n)
        samples[:, j] = (perm + np.random.uniform(0, 1, n)) / n
    return samples

def generate_grid(steps=9):
    grid = np.linspace(0, 1, steps)
    return np.array(list(product(grid, repeat=5)))

# ============================================================
# 4. قیدها و جستجو
# ============================================================
def check_dominance(ctxs, a1, a2, a3, a4, thresh=0.5):
    thresh += 1e-6
    for ctx in ctxs:
        D = compute_dominance_ratio(ctx, a1, a2, a3, a4)
        if np.any(D <= thresh):
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
    print(f"\n🔍 Grid search over {total} combos...")
    for a1 in vals:
        for a2 in vals:
            for a3 in vals:
                for a4 in vals:
                    count += 1
                    if count % 5000 == 0:
                        print(f"   {count}/{total}")
                    if not check_dominance(ctxs, a1, a2, a3, a4):
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
# 5. تحلیل شناسایی‌پذیری
# ============================================================
def analyze(feasible):
    if not feasible:
        return {'status': 'NO FEASIBLE SOLUTION'}
    best = feasible[0]
    best_sum = best['sum_a']
    near = [c for c in feasible if c['sum_a'] <= best_sum * 1.05]
    ranges = {}
    for k in ['a1', 'a2', 'a3', 'a4']:
        vals = [c[k] for c in near]
        ranges[k] = {'min': min(vals), 'max': max(vals), 'std': round(np.std(vals), 4)}
    if len(near) == 1:
        status = 'STRONGLY IDENTIFIED'
    elif any(ranges[k]['std'] > 0.03 for k in ranges):
        status = 'WEAKLY IDENTIFIED'
    else:
        status = 'STABLE'
    return {
        'status': status,
        'best': {'a1': best['a1'], 'a2': best['a2'], 'a3': best['a3'], 'a4': best['a4']},
        'best_sum': best_sum,
        'near_count': len(near),
        'ranges': ranges,
        'near_top': near[:10]
    }

# ============================================================
# 6. اجرای اصلی
# ============================================================
def main():
    print("=" * 80)
    print("MODEL-FREE CALIBRATION V4")
    print("Minimal Sufficient Dominance | Self-contained")
    print("=" * 80)

    # دیتاست‌های واقعی
    real_ctxs = download_and_extract_real_contexts()

    # فضای مصنوعی
    print("\n🧪 Generating synthetic contexts...")
    lhs = generate_latin_hypercube(3000, 42)
    grid = generate_grid(9)
    synth = np.vstack([lhs, grid])
    print(f"   {len(synth)} synthetic points")

    # ترکیب
    ctxs = list(synth)
    if real_ctxs:
        ctxs.extend(real_ctxs)
    print(f"   Total: {len(ctxs)} contexts")

    # جستجو
    feasible = grid_search(ctxs, step=0.05)

    if not feasible:
        print("\n❌ No feasible coefficients found.")
        sys.exit(1)

    # تحلیل
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
            if np.any(D <= 0.5):
                viol += 1
        real_val = {'violations': viol, 'total': len(real_ctxs), 'avg_D': np.mean(Ds, axis=0).tolist()}

    # چاپ نتیجه
    print("\n" + "=" * 80)
    print("✅ RESULT")
    print("=" * 80)
    print(f"a1 = {best['a1']:.2f}  (Interpretability)")
    print(f"a2 = {best['a2']:.2f}  (Robustness)")
    print(f"a3 = {best['a3']:.2f}  (Scalability)")
    print(f"a4 = {best['a4']:.2f}  (Rep. Capacity)")
    print(f"Sum = {result['best_sum']:.2f}")
    print(f"\nIdentifiability: {result['status']}")
    if result['status'] == 'WEAKLY IDENTIFIED':
        for k, v in result['ranges'].items():
            print(f"   {k}: [{v['min']:.2f}, {v['max']:.2f}]")
    if real_ctxs:
        print(f"\nReal-data validation: {real_val['violations']}/{real_val['total']} violations")
        print(f"   Avg D: interp={real_val['avg_D'][0]:.3f}, robust={real_val['avg_D'][1]:.3f}, scal={real_val['avg_D'][2]:.3f}, rep={real_val['avg_D'][3]:.3f}")

    # ذخیره JSON
    os.makedirs('output', exist_ok=True)
    report = {
        'methodology': 'Minimal Sufficient Dominance (D > 0.5)',
        'selected': best,
        'identifiability': result['status'],
        'near_optimal_count': result['near_count'],
        'coefficient_ranges': result['ranges'],
        'real_validation': real_val,
        'context_count': len(ctxs)
    }
    with open('output/best_coefficients.json', 'w') as f:
        json.dump(report, f, indent=4)

    print("\n📁 Report saved to output/best_coefficients.json")
    print("=" * 80)

if __name__ == "__main__":
    main()
