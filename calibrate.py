"""
Model-Free Coefficient Calibration - V4 (Test Version)
Runs entirely on synthetic context space. No ML models, no data downloads.

Principle: Minimal Sufficient Dominance
- Constraint: D > 0.5 (primary driver dominates secondary) across all contexts.
- Objective: Minimize sum(a_i) to ensure parsimony.
"""

import os
import json
import numpy as np
import pandas as pd
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. Core Mathematical Functions
# ============================================================

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
    """
    Compute numerical derivatives of all 4 requirements w.r.t all 5 context vars.
    Returns matrix (4x5).
    """
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
    Calculate the Primary Influence Proportion for each requirement.
    Pairs: (req_idx, primary_var_idx, secondary_var_idx)
    Interp: primary=E(4), secondary=rho(3)
    Robust: primary=N(1), secondary=rho(3)
    Scal: primary=V(0), secondary=G(2)
    Rep: primary=G(2), secondary=E(4)
    """
    pairs = [(0, 4, 3), (1, 1, 3), (2, 0, 2), (3, 2, 4)]
    D = np.zeros(4)
    sens_matrix = compute_all_sensitivities(ctx, a1, a2, a3, a4)
    
    for idx, (req, prim, sec) in enumerate(pairs):
        s_prim = abs(sens_matrix[req, prim])
        s_sec = abs(sens_matrix[req, sec])
        # Avoid division by zero
        if s_prim + s_sec == 0:
            D[idx] = 0.5
        else:
            D[idx] = s_prim / (s_prim + s_sec + 1e-12)
    return D

# ============================================================
# 2. Context Space Generation (Synthetic only for this test)
# ============================================================

def generate_latin_hypercube(n_samples=3000, seed=42):
    """Latin Hypercube Sampling for better space coverage."""
    np.random.seed(seed)
    samples = np.zeros((n_samples, 5))
    for j in range(5):
        perm = np.random.permutation(n_samples)
        samples[:, j] = (perm + np.random.uniform(0, 1, n_samples)) / n_samples
    return samples

def generate_grid_contexts(steps=9):
    """Deterministic grid for exact boundary checking."""
    grid = np.linspace(0, 1, steps)
    mesh = np.array(list(product(grid, repeat=5)))
    return mesh

# ============================================================
# 3. Constraint Checkers
# ============================================================

def check_dominance_constraint(ctxs, a1, a2, a3, a4, threshold=0.5):
    """
    Returns True if dominance ratio > threshold for ALL contexts.
    We use strict inequality to avoid trivial solutions (e.g., a4=0.5).
    """
    # Adding a tiny epsilon to enforce strict dominance
    effective_threshold = threshold + 1e-6
    for ctx in ctxs:
        D = compute_dominance_ratio(ctx, a1, a2, a3, a4)
        if np.any(D <= effective_threshold):
            return False
    return True

def check_boundedness(ctxs, a1, a2, a3, a4):
    """Check that all requirements stay within [0, 1]."""
    for ctx in ctxs:
        r = compute_requirements(ctx, a1, a2, a3, a4)
        if np.any(r < 0) or np.any(r > 1):
            return False
    return True

# ============================================================
# 4. Grid Search (Minimal Sufficient Dominance)
# ============================================================

def grid_search_minimal(ctxs, step=0.05):
    """
    Search a_i in [0.50, 0.95].
    Feasibility: D > 0.5 for all contexts.
    Objective: Minimize sum(a_i).
    """
    vals = np.arange(0.50, 0.96, step)
    feasible = []
    total = len(vals) ** 4
    count = 0
    
    print(f"Starting grid search over {total} combinations (step={step})...")
    print("Constraint: D > 0.5 for ALL contexts. Objective: min(sum of coefficients).")
    
    for a1 in vals:
        for a2 in vals:
            for a3 in vals:
                for a4 in vals:
                    count += 1
                    if count % 1000 == 0:
                        print(f"  Progress: {count}/{total}")
                    
                    # Check hard constraints
                    if not check_dominance_constraint(ctxs, a1, a2, a3, a4):
                        continue
                    if not check_boundedness(ctxs, a1, a2, a3, a4):
                        continue
                    
                    # If feasible, store
                    sum_a = a1 + a2 + a3 + a4
                    feasible.append({
                        'a1': round(a1, 2), 'a2': round(a2, 2),
                        'a3': round(a3, 2), 'a4': round(a4, 2),
                        'sum_a': round(sum_a, 4)
                    })
    
    if not feasible:
        return []
    
    # Sort by sum_a ascending (parsimony)
    feasible.sort(key=lambda x: x['sum_a'])
    return feasible

# ============================================================
# 5. Identifiability Analysis
# ============================================================

def analyze_landscape(feasible):
    if not feasible:
        return {'status': 'NO FEASIBLE SOLUTION'}
    
    best = feasible[0]
    best_sum = best['sum_a']
    
    # Find near-optimal (within 5% of best sum)
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

# ============================================================
# 6. Main Execution
# ============================================================

def main():
    print("=" * 80)
    print("TEST: MODEL-FREE CALIBRATION V4 (Minimal Sufficient Dominance)")
    print("Running purely on synthetic context space.")
    print("=" * 80)
    
    # --- Build context space ---
    print("\nGenerating synthetic context spaces...")
    lhs_ctxs = generate_latin_hypercube(n_samples=3000, seed=42)
    grid_ctxs = generate_grid_contexts(steps=9)  # 9^5 = 59049 points
    all_ctxs = np.vstack([lhs_ctxs, grid_ctxs])
    print(f"Total contexts generated: {len(all_ctxs)}")
    
    # --- Run Grid Search ---
    print("\n--- Calibration Phase ---")
    feasible = grid_search_minimal(all_ctxs, step=0.05)
    
    if not feasible:
        print("ERROR: No feasible coefficients found with the given constraints.")
        print("Try relaxing the dominance threshold or checking the context space.")
        return
    
    # --- Analyze Results ---
    ident = analyze_landscape(feasible)
    best = ident['best_coefficients']
    
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
    
    # --- Save Report ---
    os.makedirs('output', exist_ok=True)
    
    report = {
        'test_mode': True,
        'methodology': 'Minimal Sufficient Dominance (D > 0.5)',
        'context_space': {
            'latin_hypercube_samples': 3000,
            'grid_steps': 9,
            'total_contexts': len(all_ctxs)
        },
        'grid_search_step': 0.05,
        'selected_coefficients': best,
        'identifiability': ident,
        'feasible_candidates_found': len(feasible)
    }
    
    with open('output/best_coefficients.json', 'w') as f:
        json.dump(report, f, indent=4)
    
    print(f"\nDetailed report saved to: output/best_coefficients.json")
    print("=" * 80)
    
    # Optional: Quick validation of the best coefficients
    print("\nQuick Validation (Average Dominance Ratios over 100 random contexts):")
    test_ctxs = np.random.rand(100, 5)
    D_avg = np.zeros(4)
    for ctx in test_ctxs:
        D = compute_dominance_ratio(ctx, best['a1'], best['a2'], best['a3'], best['a4'])
        D_avg += D
    D_avg /= 100
    print(f"  Interpretability D: {D_avg[0]:.3f}")
    print(f"  Robustness D:       {D_avg[1]:.3f}")
    print(f"  Scalability D:      {D_avg[2]:.3f}")
    print(f"  Representation D:   {D_avg[3]:.3f}")

if __name__ == "__main__":
    main()
