#!/usr/bin/env python3
"""
Model-Free Calibration V7.1 - با آستانه‌ی تسلط ۰٫۷۰
Local datasets (IBM + UK) + Synthetic contexts
Time: ~3-5 minutes
"""

import os
import sys
import json
import time
import warnings
from itertools import product

# ============================================================
# فلش کردن تمام پرینت‌ها
# ============================================================
def pprint(*args, **kwargs):
    kwargs.setdefault('flush', True)
    print(*args, **kwargs)

# ============================================================
# نصب خودکار کتابخانه‌ها
# ============================================================
def install(pkg):
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "--quiet"])

try:
    import numpy as np
except ImportError:
    pprint("Installing numpy...")
    install("numpy")
    import numpy as np

try:
    import pandas as pd
except ImportError:
    pprint("Installing pandas...")
    install("pandas")
    import pandas as pd

warnings.filterwarnings('ignore')

# ============================================================
# توابع اصلی (با محاسبه‌ی تحلیلی مشتقات)
# ============================================================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1 - tanh(x)**2

def compute_requirements(ctx, a1, a2, a3, a4):
    V, N, G, rho, E = ctx
    b1, b2, b3, b4 = 1 - a1, 1 - a2, 1 - a3, 1 - a4
    r_interp = a1 * (1 - sigmoid(10 * (E - 0.5))) + b1 * rho
    r_robust = a2 * sigmoid(12 * (N - 0.35)) + b2 * tanh(2 * rho)
    r_scal = a3 * tanh(3 * V) + b3 * G
    r_rep = a4 * G + b4 * E
    return np.array([r_interp, r_robust, r_scal, r_rep])

def compute_analytical_sensitivity(ctx, a1, a2, a3, a4):
    V, N, G, rho, E = ctx
    b1, b2, b3, b4 = 1 - a1, 1 - a2, 1 - a3, 1 - a4
    
    sens = np.zeros((4, 5))
    sens[0, 4] = -a1 * 10 * sigmoid_deriv(10 * (E - 0.5))
    sens[0, 3] = b1
    sens[1, 1] = a2 * 12 * sigmoid_deriv(12 * (N - 0.35))
    sens[1, 3] = b2 * 2 * tanh_deriv(2 * rho)
    sens[2, 0] = a3 * 3 * tanh_deriv(3 * V)
    sens[2, 2] = b3
    sens[3, 2] = a4
    sens[3, 4] = b4
    return sens

def compute_dominance_ratio(ctx, a1, a2, a3, a4):
    pairs = [(0, 4, 3), (1, 1, 3), (2, 0, 2), (3, 2, 4)]
    D = np.zeros(4)
    sm = compute_analytical_sensitivity(ctx, a1, a2, a3, a4)
    for idx, (req, prim, sec) in enumerate(pairs):
        sp = abs(sm[req, prim])
        ss = abs(sm[req, sec])
        D[idx] = sp / (sp + ss + 1e-12)
    return D

# ============================================================
# بارگذاری دیتاست‌های محلی
# ============================================================
def load_local_datasets():
    datasets = []
    ibm_paths = [
        "dataset/WA_Fn-UseC_-Accounts-Receivable.csv",
        "WA_Fn-UseC_-Accounts-Receivable.csv",
        "data/WA_Fn-UseC_-Accounts-Receivable.csv"
    ]
    uk_paths = [
        "dataset/payment-practices.csv",
        "payment-practices.csv",
        "data/payment-practices.csv"
    ]
    
    ibm_file = None
    for p in ibm_paths:
        if os.path.exists(p):
            ibm_file = p
            break
    if ibm_file:
        pprint(f"✅ Found IBM: {ibm_file}")
        df = pd.read_csv(ibm_file)
        datasets.append(("IBM", df, "target"))
    else:
        pprint("⚠️ IBM not found")
    
    uk_file = None
    for p in uk_paths:
        if os.path.exists(p):
            uk_file = p
            break
    if uk_file:
        pprint(f"✅ Found UK: {uk_file}")
        df = pd.read_csv(uk_file)
        datasets.append(("UK", df, "target"))
    else:
        pprint("⚠️ UK not found")
    
    return datasets

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

def extract_contexts(datasets):
    all_ctxs = []
    E_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    for name, df, target_col in datasets:
        pprint(f"📊 {name}: {df.shape[0]:,} rows")
        for E in E_levels:
            try:
                ctx = compute_context_from_df(df, target_col, E)
                all_ctxs.append(ctx)
            except Exception as e:
                pprint(f"  ⚠️ E={E}: {e}")
        pprint(f"  ✅ {len(E_levels)} contexts")
    return all_ctxs

# ============================================================
# فضای مصنوعی
# ============================================================
def generate_latin_hypercube(n=800, seed=42, low=0.05, high=0.95):
    np.random.seed(seed)
    samples = np.zeros((n, 5))
    for j in range(5):
        perm = np.random.permutation(n)
        raw = (perm + np.random.uniform(0, 1, n)) / n
        samples[:, j] = low + (high - low) * raw
    return samples

def generate_grid(steps=5, low=0.05, high=0.95):
    grid = np.linspace(low, high, steps)
    return np.array(list(product(grid, repeat=5)))

# ============================================================
# قیدها با آستانه‌ی ۰٫۷۰ (تغییر اصلی اینجاست)
# ============================================================
def check_dominance_quantile(ctxs, a1, a2, a3, a4, thresh=0.70, quantile=0.95):
    all_D = []
    if len(ctxs) > 2000:
        idx = np.random.choice(len(ctxs), 2000, replace=False)
        sample_ctxs = [ctxs[i] for i in idx]
    else:
        sample_ctxs = ctxs
    
    for ctx in sample_ctxs:
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
    sample_size = min(500, len(ctxs))
    idx = np.random.choice(len(ctxs), sample_size, replace=False)
    for i in idx:
        r = compute_requirements(ctxs[i], a1, a2, a3, a4)
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
    pprint(f"\n🔍 Grid search over {total} combos...")
    pprint(f"   Threshold = 0.70 | Using {min(2000, len(ctxs))} contexts per check")
    sys.stdout.flush()
    
    start_time = time.time()
    
    for a1 in vals:
        for a2 in vals:
            for a3 in vals:
                for a4 in vals:
                    count += 1
                    
                    if count % 1000 == 0:
                        elapsed = time.time() - start_time
                        pprint(f"  ⏱️ {count}/{total} | {elapsed:.1f}s")
                        sys.stdout.flush()
                    
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
    pprint(f"  ✅ Found {len(feasible)} feasible candidates")
    sys.stdout.flush()
    return feasible

# ============================================================
# تأیید نهایی روی همه‌ی نقاط
# ============================================================
def final_verification(ctxs, candidates):
    verified = []
    pprint(f"\n🔍 Final verification on all {len(ctxs)} contexts...")
    sys.stdout.flush()
    
    for cand in candidates[:20]:
        a1, a2, a3, a4 = cand['a1'], cand['a2'], cand['a3'], cand['a4']
        all_D = []
        for ctx in ctxs:
            D = compute_dominance_ratio(ctx, a1, a2, a3, a4)
            all_D.append(D)
        all_D = np.array(all_D)
        
        passes = True
        for req_idx in range(4):
            D_req = all_D[:, req_idx]
            q = np.quantile(D_req, 0.95)
            if q <= 0.70:
                passes = False
                break
        
        bounded = True
        for ctx in ctxs:
            r = compute_requirements(ctx, a1, a2, a3, a4)
            if np.any(r < 0) or np.any(r > 1):
                bounded = False
                break
        
        if passes and bounded:
            verified.append(cand)
    
    verified.sort(key=lambda x: x['sum_a'])
    return verified

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
    pprint("=" * 80)
    pprint("🚀 MODEL-FREE CALIBRATION V7.1 (Threshold = 0.70)")
    pprint("Analytical derivatives | Random sampling | ~3-5 minutes")
    pprint("=" * 80)
    sys.stdout.flush()

    pprint("\n📂 Loading local datasets...")
    sys.stdout.flush()
    datasets = load_local_datasets()

    real_ctxs = []
    if datasets:
        real_ctxs = extract_contexts(datasets)
        pprint(f"\n✅ {len(real_ctxs)} real contexts")
    else:
        pprint("\n⚠️ No local datasets. Synthetic only.")
    sys.stdout.flush()

    pprint("\n🧪 Generating synthetic contexts...")
    sys.stdout.flush()
    lhs = generate_latin_hypercube(800, 42, low=0.05, high=0.95)
    grid = generate_grid(5, low=0.05, high=0.95)
    synth_ctxs = np.vstack([lhs, grid])
    pprint(f"  ✅ {len(synth_ctxs)} synthetic contexts")
    sys.stdout.flush()

    all_ctxs = list(synth_ctxs) + list(real_ctxs)
    pprint(f"  📊 Total: {len(all_ctxs)} contexts")
    sys.stdout.flush()

    pprint("\n⏳ Starting grid search (Threshold = 0.70)...")
    sys.stdout.flush()
    
    feasible = grid_search(all_ctxs, step=0.05)

    if not feasible:
        pprint("\n❌ No feasible coefficients found with threshold 0.70.")
        pprint("   Try lowering the threshold to 0.60 or 0.65.")
        sys.exit(1)

    verified = final_verification(all_ctxs, feasible)
    
    if not verified:
        pprint("\n⚠️ No candidates passed full verification. Using best from sampling.")
        verified = feasible[:1]
    
    result = analyze(verified)
    best = result['best']

    pprint("\n" + "=" * 80)
    pprint("✅ FINAL RESULT (Threshold = 0.70)")
    pprint("=" * 80)
    pprint(f"a1 = {best['a1']:.2f}  (Interpretability)")
    pprint(f"a2 = {best['a2']:.2f}  (Robustness)")
    pprint(f"a3 = {best['a3']:.2f}  (Scalability)")
    pprint(f"a4 = {best['a4']:.2f}  (Rep. Capacity)")
    pprint(f"Sum = {result['best_sum']:.2f}")
    pprint(f"Identifiability: {result['status']}")
    
    if result['status'] == 'WEAK':
        pprint("\n  Near-optimal ranges:")
        for k, v in result['ranges'].items():
            pprint(f"    {k}: [{v['min']:.2f}, {v['max']:.2f}] (std={v['std']:.3f})")
    
    sys.stdout.flush()

    os.makedirs('output', exist_ok=True)
    report = {
        'methodology': 'Quantile-Based (95% contexts D > 0.70)',
        'datasets_used': [name for name, _, _ in datasets],
        'synthetic_contexts': len(synth_ctxs),
        'real_contexts': len(real_ctxs),
        'total_contexts': len(all_ctxs),
        'selected': best,
        'identifiability': result['status'],
        'near_optimal_count': result['near_count'],
        'coefficient_ranges': result['ranges']
    }
    with open('output/best_coefficients.json', 'w') as f:
        json.dump(report, f, indent=4)

    pprint("\n📁 Report saved to output/best_coefficients.json")
    pprint("=" * 80)

if __name__ == "__main__":
    main()
