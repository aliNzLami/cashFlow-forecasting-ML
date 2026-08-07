#!/usr/bin/env python3
"""
Model-Free Coefficient Calibration - V4 (Production Version)
===============================================================================
This module calibrates the coefficients a1, a2, a3, a4 for the context-to-
requirement mapping functions WITHOUT training any ML models.

Methodology: Minimal Sufficient Dominance
    - Constraint: D > 0.5 (primary driver dominates secondary) across all contexts
    - Objective: Minimize sum(a_i) to ensure parsimony

Dependencies (installed automatically in GitHub Actions, but checked here):
    - numpy, pandas, kagglehub

Usage:
    python scripts/calibrate.py

Output:
    JSON report saved to output/best_coefficients.json
===============================================================================
"""

import os
import sys
import json
import warnings
from itertools import product

# ============================================================================
# 0. Dependency Check (no external requirements.txt needed)
# ============================================================================

def check_dependencies():
    """Check if required libraries are installed; if not, print guidance and exit."""
    missing = []
    try:
        import numpy as np
    except ImportError:
        missing.append('numpy')
    try:
        import pandas as pd
    except ImportError:
        missing.append('pandas')
    try:
        import kagglehub
    except ImportError:
        missing.append('kagglehub')
    
    if missing:
        print("=" * 80)
        print("ERROR: Missing required Python libraries:")
        for lib in missing:
            print(f"  - {lib}")
        print("\nPlease install them using:")
        print(f"  pip install {' '.join(missing)}")
        print("=" * 80)
        sys.exit(1)
    
    # Import them now (safe)
    global np, pd, kagglehub
    import numpy as np
    import pandas as pd
    import kagglehub
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')

check_dependencies()

# ============================================================================
# 1. Core Mathematical Functions
# ============================================================================

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def compute_requirements(ctx, a1, a2, a3, a4):
    """Return (r_interp, r_robust, r_scal, r_rep)."""
    V, N, G, rho, E = ctx
    b1, b2, b3, b4 = 1 - a1, 1 - a2, 1 - a3, 1 - a4

    r_interp = a1 * (1 - sigmoid(10 * (E - 0.5))) + b1 * rho
    r_robust = a2 * sigmoid(12 * (N - 0.35)) + b2 * tanh(2 * rho)
    r_scal = a3 * tanh(3 * V) + b3 * G
    r_rep = a4 * G + b4 * E

    return np.array([r_interp, r_robust, r_scal, r_rep])

def compute_all_sensitivities(ctx, a1, a2, a3, a4, delta=1e-5):
    """Numerical derivatives of 4 requirements w.r.t 5 context vars. Returns (4x5)."""
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
    """
    Primary Influence Proportion for each requirement.
    Pairs: (req_idx, primary_var_idx, secondary_var_idx)
    """
    pairs = [(0, 4, 3), (1, 1, 3), (2, 0, 2), (3, 2, 4)]
    D = np.zeros(4)
    sens_matrix = compute_all_sensitivities(ctx, a1, a2, a3, a4)
    
    for idx, (req, prim, sec) in enumerate(pairs):
        s_prim = abs(sens_matrix[req, prim])
        s_sec = abs(sens_matrix[req, sec])
        D[idx] = s_prim / (s_prim + s_sec + 1e-12)
    return D

# ============================================================================
# 2. Context Extraction from Real Datasets
# ============================================================================

def compute_context_from_df(df, target_col='target', E=0.5):
    """Compute (V, N, G, rho, E) from a DataFrame."""
    X = df.drop(columns=[target_col], errors='ignore')
    n, p = X.shape

    # Volume
    V = np.clip(np.log10(max(n, 1)) / 6, 0, 1)
    
    # Feature-to-instance ratio
    rho = np.clip(p / max(n, 1), 0, 1)
    
    # Noise: missingness + outliers
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
    
    # Granularity: temporal first, then fallback
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

# ============================================================================
# 3. Dataset Download & Context Extraction (Real Datasets)
# ============================================================================

def download_and_extract_real_contexts():
    """
    Download the three Kaggle datasets and extract contexts.
    Returns list of context vectors (each is [V, N, G, rho, E]).
    """
    dataset_sources = [
        ("hhenry/finance-factoring-ibm-late-payment-histories", "target"),
        ("saikiran0684/payment-practices-of-uk-buyers", "target"),
        ("wordsforthewise/lending-club", "target")
    ]
    
    all_ctxs = []
    E_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    print("\nDownloading real datasets from Kaggle...")
    for src, target_col in dataset_sources:
        try:
            path = kagglehub.dataset_download(src)
            print(f"  Downloaded: {src}")
            
            # Find the first CSV file in the downloaded folder
            csv_files = []
            for root, _, files in os.walk(path):
                for f in files:
                    if f.endswith('.csv'):
                        csv_files.append(os.path.join(root, f))
            
            if not csv_files:
                print(f"    No CSV found in {src}, skipping.")
                continue
            
            # Use the first CSV (most datasets have only one main file)
            df = pd.read_csv(csv_files[0])
            print(f"    Loaded {os.path.basename(csv_files[0])} with shape {df.shape}")
            
            # Extract contexts for different E levels
            for E in E_levels:
                try:
                    ctx = compute_context_from_df(df, target_col=target_col, E=E)
                    all_ctxs.append(ctx)
                except Exception as e:
                    print(f"    Error extracting context for E={E}: {e}")
                    
        except Exception as e:
            print(f"  Failed to download {src}: {e}")
    
    print(f"\nExtracted {len(all_ctxs)} real context points.")
    return all_ctxs

# ============================================================================
# 4. Synthetic Context Space Generation
# ============================================================================

def generate_latin_hypercube(n_samples=3000, seed=42):
    np.random.seed(seed)
    samples = np.zeros((n_samples, 5))
    for j in range(5):
        perm = np.random.permutation(n_samples)
        samples[:, j] = (perm + np.random.uniform(0, 1, n_samples)) / n_samples
    return samples

def generate_grid_contexts(steps=9):
    grid = np.linspace(0, 1, steps)
    mesh = np.array(list(product(grid, repeat=5)))
    return mesh

# ============================================================================
# 5. Constraint Checkers
# ============================================================================

def check_dominance_constraint(ctxs, a1, a2, a3, a4, threshold=0.5):
    """Returns True if D > threshold for ALL contexts (strict)."""
    effective_threshold = threshold + 1e-6
    for ctx in ctxs:
        D = compute_dominance_ratio(ctx, a1, a2, a3, a4)
        if np.any(D <= effective_threshold):
            return False
    return True

def check_boundedness(ctxs, a1, a2, a3, a4):
    """Check all requirements stay within [0, 1]."""
    for ctx in ctxs:
        r = compute_requirements(ctx, a1, a2, a3, a4)
        if np.any(r < 0) or np.any(r > 1):
            return False
    return True

# ============================================================================
# 6. Grid Search (Minimal Sufficient Dominance)
# ============================================================================

def grid_search_minimal(ctxs, step=0.05):
    vals = np.arange(0.50, 0.96, step)
    feasible = []
    total = len(vals) ** 4
    count = 0
    
    print(f"\nGrid search over {total} combinations (step={step})...")
    print("Constraint: D > 0.5 for ALL contexts. Objective: minimize sum(a_i).")
    
    for a1 in vals:
        for a2 in vals:
            for a3 in vals:
                for a4 in vals:
                    count += 1
                    if count % 5000 == 0:
                        print(f"  Progress: {count}/{total}")
                    
                    if not check_dominance_constraint(ctxs, a1, a2, a3, a4):
                        continue
                    if not check_boundedness(ctxs, a1, a2, a3, a4):
                        continue
                    
                    feasible.append({
                        'a1': round(a1, 2), 'a2': round(a2, 2),
                        'a3': round(a3, 2), 'a4': round(a4, 2),
                        'sum_a': round(a1 + a2 + a3 + a4, 4)
                    })
    
    feasible.sort(key=lambda x: x['sum_a'])
    return feasible

# ============================================================================
# 7. Identifiability Analysis
# ============================================================================

def analyze_landscape(feasible):
    if not feasible:
        return {'status': 'NO FEASIBLE SOLUTION'}
    
    best = feasible[0]
    best_sum = best['sum_a']
    
    # Near-optimal within 5% of best sum
    near_opt = [c for c in feasible if c['sum_a'] <= best_sum * 1.05]
    
    ranges = {}
    for key in ['a1', 'a2', 'a3', 'a4']:
        vals = [c[key] for c in near_opt]
        ranges[key] = {
            'min': round(min(vals), 2),
            'max': round(max(vals), 2),
            'std': round(np.std(vals), 4)
        }
    
    if len(near_opt) == 1:
        status = 'STRONGLY IDENTIFIED'
    elif any(ranges[k]['std'] > 0.03 for k in ranges):
        status = 'WEAKLY IDENTIFIED'
    else:
        status = 'STABLE'
    
    return {
        'status': status,
        'best_coefficients': {'a1': best['a1'], 'a2': best['a2'], 'a3': best['a3'], 'a4': best['a4']},
        'best_sum': best['sum_a'],
        'near_optimal_count': len(near_opt),
        'coefficient_ranges': ranges,
        'near_optimal_sets': near_opt[:10]
    }

# ============================================================================
# 8. Final Validation on Real Contexts
# ============================================================================

def validate_on_real_contexts(ctxs, coeffs):
    """Check D > 0.5 for all real contexts."""
    a1, a2, a3, a4 = coeffs['a1'], coeffs['a2'], coeffs['a3'], coeffs['a4']
    violations = 0
    D_all = []
    for ctx in ctxs:
        D = compute_dominance_ratio(ctx, a1, a2, a3, a4)
        D_all.append(D)
        if np.any(D <= 0.5):
            violations += 1
    return {
        'violations': violations,
        'total': len(ctxs),
        'avg_D': np.mean(D_all, axis=0).tolist()
    }

# ============================================================================
# 9. Main Execution
# ============================================================================

def main():
    print("=" * 80)
    print("MODEL-FREE COEFFICIENT CALIBRATION V4 (Production)")
    print("Real datasets + synthetic space | Minimal Sufficient Dominance")
    print("=" * 80)
    
    # --- Step 1: Get real dataset contexts ---
    real_ctxs = download_and_extract_real_contexts()
    
    # --- Step 2: Generate synthetic contexts ---
    print("\nGenerating synthetic context spaces...")
    lhs_ctxs = generate_latin_hypercube(n_samples=3000, seed=42)
    grid_ctxs = generate_grid_contexts(steps=9)
    synth_ctxs = np.vstack([lhs_ctxs, grid_ctxs])
    print(f"  Synthetic contexts: {len(synth_ctxs)}")
    
    # --- Step 3: Combine contexts ---
    all_ctxs = list(synth_ctxs)
    if real_ctxs:
        all_ctxs.extend(real_ctxs)
    print(f"  Total contexts: {len(all_ctxs)} (incl. {len(real_ctxs)} real)")
    
    # --- Step 4: Grid Search ---
    feasible = grid_search_minimal(all_ctxs, step=0.05)
    
    if not feasible:
        print("\n❌ ERROR: No feasible coefficients found.")
        print("   Try: increasing step size or relaxing the dominance threshold.")
        sys.exit(1)
    
    # --- Step 5: Identifiability ---
    ident = analyze_landscape(feasible)
    best = ident['best_coefficients']
    
    # --- Step 6: Validation on real contexts (if available) ---
    real_val = {}
    if real_ctxs:
        real_val = validate_on_real_contexts(real_ctxs, best)
    
    # --- Step 7: Report ---
    print("\n" + "=" * 80)
    print("✅ CALIBRATION COMPLETE")
    print("=" * 80)
    print(f"Selected Coefficients (Minimal Sum):")
    print(f"  a1 (Interpretability) = {best['a1']:.2f}")
    print(f"  a2 (Robustness)       = {best['a2']:.2f}")
    print(f"  a3 (Scalability)      = {best['a3']:.2f}")
    print(f"  a4 (Rep. Capacity)    = {best['a4']:.2f}")
    print(f"  Sum of coefficients   = {ident['best_sum']:.2f}")
    print(f"\nIdentifiability Status: {ident['status']}")
    
    if ident['status'] == 'WEAKLY IDENTIFIED':
        print("  Near-optimal ranges:")
        for k, v in ident['coefficient_ranges'].items():
            print(f"    {k}: [{v['min']:.2f}, {v['max']:.2f}] (std={v['std']:.3f})")
    
    if real_ctxs:
        print(f"\nReal-Dataset Validation:")
        print(f"  Dominance violations: {real_val['violations']}/{real_val['total']}")
        print(f"  Avg D: Interp={real_val['avg_D'][0]:.3f}, Robust={real_val['avg_D'][1]:.3f}, Scal={real_val['avg_D'][2]:.3f}, Rep={real_val['avg_D'][3]:.3f}")
    
    # --- Step 8: Save JSON ---
    os.makedirs('output', exist_ok=True)
    report = {
        'methodology': 'Minimal Sufficient Dominance (D > 0.5)',
        'selected_coefficients': best,
        'identifiability': ident,
        'feasible_candidates_found': len(feasible),
        'context_space': {
            'synthetic_lhs': 3000,
            'synthetic_grid_steps': 9,
            'real_contexts': len(real_ctxs),
            'total': len(all_ctxs)
        },
        'real_dataset_validation': real_val if real_ctxs else None,
        'limitations': [
            'Granularity G uses temporal fallback; if no temporal data, row count proxy is used.',
            'Noise N is based on missingness and outliers; other noise sources not captured.',
            'Expertise E is scenario-based, not data-driven.'
        ]
    }
    
    with open('output/best_coefficients.json', 'w') as f:
        json.dump(report, f, indent=4)
    
    print(f"\nDetailed report saved to: output/best_coefficients.json")
    print("=" * 80)

if __name__ == "__main__":
    main()
