#!/usr/bin/env python3
"""
Model-Free Calibration V8.0 - FINAL
Three datasets (IBM, UK, Lending Club) | Analytical derivatives
Threshold: 0.70 | Quantile: 95% | Time: ~5-8 minutes
"""

import os
import sys
import json
import time
import warnings
from itertools import product

# ============================================================
# Force flush for all prints (GitHub Actions compatibility)
# ============================================================
def pprint(*args, **kwargs):
    kwargs.setdefault('flush', True)
    print(*args, **kwargs)

# ============================================================
# Auto-install dependencies
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

try:
    import kagglehub
except ImportError:
    pprint("Installing kagglehub...")
    install("kagglehub")
    import kagglehub

warnings.filterwarnings('ignore')

# ============================================================
# Core mathematical functions (analytical derivatives)
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
# Context extraction from DataFrames
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

# ============================================================
# Dataset loaders (local + kagglehub for Lending Club)
# ============================================================
def find_csv_file(download_path, keyword=None):
    """Find the first CSV file in a directory, optionally matching a keyword."""
    for root, _, files in os.walk(download_path):
        for f in files:
            if f.endswith('.csv') or f.endswith('.csv.gz'):
                if keyword is None or keyword.lower() in f.lower():
                    return os.path.join(root, f)
    return None

def load_ibm_data():
    """Load IBM dataset from local path."""
    paths = [
        "dataset/WA_Fn-UseC_-Accounts-Receivable.csv",
        "WA_Fn-UseC_-Accounts-Receivable.csv",
        "data/WA_Fn-UseC_-Accounts-Receivable.csv"
    ]
    for p in paths:
        if os.path.exists(p):
            pprint(f"  ✅ Found IBM locally: {p}")
            return pd.read_csv(p)
    pprint("  ⚠️ IBM not found locally.")
    return None

def load_uk_data():
    """Load UK dataset from local path."""
    paths = [
        "dataset/payment-practices.csv",
        "payment-practices.csv",
        "data/payment-practices.csv"
    ]
    for p in paths:
        if os.path.exists(p):
            pprint(f"  ✅ Found UK locally: {p}")
            return pd.read_csv(p)
    pprint("  ⚠️ UK not found locally.")
    return None

def load_lending_club_data():
    """Load Lending Club dataset via kagglehub."""
    pprint("  [Lending Club] Downloading from Kaggle (1.26GB)...")
    try:
        path = kagglehub.dataset_download("wordsforthewise/lending-club")
        csv_file = find_csv_file(path, "accepted")
        if csv_file is None:
            csv_file = find_csv_file(path)
        if csv_file:
            pprint(f"  ✅ Found: {os.path.basename(csv_file)}")
            return pd.read_csv(csv_file)
        else:
            pprint("  ❌ No accepted loans CSV found.")
            return None
    except Exception as e:
        pprint(f"  ❌ Lending Club download failed: {e}")
        return None

def load_all_datasets():
    """Load all three datasets (IBM, UK, Lending Club)."""
    datasets = []
    
    pprint("\n📂 Loading datasets...")
    
    # IBM
    df = load_ibm_data()
    if df is not None:
        datasets.append(("IBM", df, "target"))
    
    # UK
    df = load_uk_data()
    if df is not None:
        datasets.append(("UK", df, "target"))
    
    # Lending Club
    df = load_lending_club_data()
    if df is not None:
        datasets.append(("Lending Club", df, "target"))
    
    return datasets

def extract_contexts(datasets):
    """Extract contexts from all datasets with 5 E-levels."""
    all_ctxs = []
    E_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    for name, df, target_col in datasets:
        pprint(f"\n📊 {name}: {df.shape[0]:,} rows, {df.shape[1]} columns")
        for E in E_levels:
            try:
                ctx = compute_context_from_df(df, target_col, E)
                all_ctxs.append(ctx)
            except Exception as e:
                pprint(f"  ⚠️ E={E} error: {e}")
        pprint(f"  ✅ Extracted {len(E_levels)} contexts")
    return all_ctxs

# ============================================================
# Synthetic context space
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
# Constraints with threshold = 0.70
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
# Grid search
# ============================================================
def grid_search(ctxs, step=0.05):
    vals = np.arange(0.50, 0.96, step)
    feasible = []
    total = len(vals) ** 4
    count = 0
    pprint(f"\n🔍 Grid search over {total} combinations...")
    pprint(f"   Threshold = 0.70 | Sampling {min(2000, len(ctxs))} contexts per check")
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
# Final verification on all contexts
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
# Identifiability analysis
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
# Main execution
# ============================================================
def main():
    pprint("=" * 80)
    pprint("🚀 MODEL-FREE CALIBRATION V8.0 (FINAL)")
    pprint("Three datasets: IBM + UK (local) + Lending Club (Kaggle)")
    pprint("Analytical derivatives | Threshold: 0.70 | ~5-8 minutes")
    pprint("=" * 80)
    sys.stdout.flush()

    # Load all datasets
    datasets = load_all_datasets()
    
    if not datasets:
        pprint("\n❌ No datasets found. Exiting.")
        sys.exit(1)

    # Extract real contexts
    real_ctxs = extract_contexts(datasets)
    pprint(f"\n✅ {len(real_ctxs)} real contexts extracted.")
    sys.stdout.flush()

    # Generate synthetic contexts
    pprint("\n🧪 Generating synthetic contexts...")
    sys.stdout.flush()
    lhs = generate_latin_hypercube(800, 42, low=0.05, high=0.95)
    grid = generate_grid(5, low=0.05, high=0.95)
    synth_ctxs = np.vstack([lhs, grid])
    pprint(f"  ✅ {len(synth_ctxs)} synthetic contexts")
    sys.stdout.flush()

    # Combine
    all_ctxs = list(synth_ctxs) + list(real_ctxs)
    pprint(f"  📊 Total contexts: {len(all_ctxs)}")
    sys.stdout.flush()

    # Grid search
    pprint("\n⏳ Starting grid search (threshold = 0.70)...")
    sys.stdout.flush()
    
    feasible = grid_search(all_ctxs, step=0.05)

    if not feasible:
        pprint("\n❌ No feasible coefficients found with threshold 0.70.")
        pprint("   Try lowering the threshold to 0.60 or 0.65.")
        sys.exit(1)

    # Final verification
    verified = final_verification(all_ctxs, feasible)
    
    if not verified:
        pprint("\n⚠️ No candidates passed full verification. Using best from sampling.")
        verified = feasible[:1]
    
    result = analyze(verified)
    best = result['best']

    # Output results
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

    # Save report
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
