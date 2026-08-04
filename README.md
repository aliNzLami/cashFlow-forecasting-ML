# Decision-Making Framework on ML Model Selection for SME's Cash Flow Forecasting

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-Submitted_to_Applied_Intelligence-brightgreen.svg)]()

This repository contains the official implementation and replication code for the research paper:

> **"A Context-Aware Compatibility Framework for SME Cash Flow Forecasting: A Leakage-Free Evaluation of Accuracy and Interpretability"**  
> *Submitted to Applied Intelligence (Springer)*

This study rigorously evaluates five machine learning models—**Linear Regression, Random Forest, XGBoost, LightGBM, and Neural Network (MLP)**—for predicting SME cash flow. Unlike prior studies, we strictly prevent **look-ahead bias** using time-series validation, and we introduce a **novel context-aware decision framework** based on **continuous mapping functions** to recommend the optimal model **before any training**, without relying on expensive AutoML or heuristic weighting schemes.

---

## 📄 Overview

Small and Medium Enterprises (SMEs) face significant liquidity risks due to unpredictable payment delays. While advanced ML models promise high accuracy, they often fail in practice due to **data leakage** and **lack of interpretability**.

**Our key contributions are:**

1. **Leakage-Free Validation:** Removal of future-dependent features (e.g., `DaysLate`) and strict chronological train/test splits for realistic performance estimates.
2. **Comprehensive Interpretability:** Comparison of TreeSHAP, TreeInterpreter, LIME, and Captum (Integrated Gradients/DeepLIFT) across five models, revealing internal inconsistencies in Neural Network attributions.
3. **The Context-Aware Compatibility Framework (Training-Free):** 
   - Translates **5 contextual attributes** (data volume, noise level, granularity, feature-to-instance ratio, and user expertise) into **4 operational requirements** (interpretability, robustness, scalability, and representation capacity) via **continuous mathematical functions** (sigmoid and hyperbolic tangent).
   - Uses a **coefficient-based weighting scheme** (derived from the same functions, not data-dependent heuristics like CRITIC) with normalised weights: **Interpretability (0.293), Robustness (0.241), Scalability (0.276), and Representation Capacity (0.190)**.
   - Recommends models via a **compatibility score** (Manhattan distance) with **O(m·k)** computational complexity—deployable on standard office hardware.
4. **Empirical Validation:** Tested on two real-world SME datasets (IBM invoice-level, UK Government firm-level) and externally validated on the Lending Club dataset under three managerial expertise scenarios (non-expert, intermediate, expert).

---

## ✨ Key Features

- **Time-Series Splitting:** Chronological ordering prevents the use of future information.
- **Grouped Splits:** Ensures all records from a single UK firm are kept within the same set (train/test) to prevent data mingling.
- **Dual Dataset Analysis:** 
  - *IBM Late Payment Histories* (Invoice-level, ~2.5k records)
  - *UK Government Payment Practices* (Firm-level, ~8k firms)
- **External Validation:** Lending Club dataset (2.26M records, 151 features) for framework generalisability.
- **Interpretability Suite:** TreeSHAP, TreeInterpreter, LIME, Integrated Gradients, and DeepLIFT.
- **Reproducible:** All random seeds fixed (42), and the code is fully documented.

---

## 📁 Repository Structure

```text
.
├── data/
│   ├── WA_Fn-UseC_-Accounts-Receivable.csv      # IBM Late Payment Histories Dataset
│   └── payment-practices.csv                    # Payment Practices of UK Buyers Dataset
|── accuracy-ibm.py                              # Accuracy Measurement for IBM Late Payment Histories
|── accuracy-uk.py                               # Accuracy Measurement for Payment Practices of UK Buyers
|── interpret-ibm.py                             # Interpretation Measurement for IBM Late Payment Histories
|── interpret-uk.py                              # Interpretation Measurement for Payment Practices of UK Buyers
|── framework-decision.py                        # To assess which model is the best fit to a particular dataset
└── README.md                                    # This file
```

---

## ⚙️ Installation & Setup

To replicate this environment, ensure you have Python 3.10 installed.

1. Clone the repository:

```text
git clone https://github.com/aliNzLami/cashFlow-forecasting-ML.git
cd cashFlow-forecasting-ML
```
2. Install dependencies:

```text
pip install numpy>=1.24.0 pandas>=2.0.0 scikit-learn>=1.3.0 xgboost>=2.0.0 lightgbm>=4.1.0 tensorflow>=2.15.0 shap>=0.44.0 lime>=0.2.0.1 captum>=0.7.0 matplotlib>=3.8.0 seaborn>=0.13.0 notebook>=7.0.0
```
---

## 📊 Datasets

This study uses two independent datasets to validate generalizability.

| **Dataset** | **Source** | **Task** | **Key Features / Preprocessing** | **Notes / Purpose** |
| :--- | :--- | :--- | :--- | :--- |
| **IBM Late Payment Histories** (Invoice-Level) | [Kaggle Link](https://www.kaggle.com/datasets/hhenry/finance-factoring-ibm-late-payment-histories/data) | Regression (Predict `DaysToSettle`) | Features restricted to those available at invoice issuance (`InvoiceAmount`, `CreditPeriod`, `PaperlessBill`). `DaysLate` and `Disputed` are strictly excluded to prevent look-ahead bias. | Used for core invoice-level accuracy and interpretability analysis. |
| **UK Government Payment Practices** (Firm-Level) | [Kaggle Link](https://www.kaggle.com/datasets/saikiran0684/payment-practices-of-uk-buyers) | Regression (Predict Average Time to Pay normalized to a 60-day threshold) | Grouped time-series split (70% oldest companies train, 30% newest test). Place raw `.csv` files in `/data`; scripts handle cleaning and scaling automatically. | Used for core firm-level accuracy and interpretability analysis. |
| **Lending Club Loan Data** (External Validation) | [Kaggle Link](https://www.kaggle.com/datasets/wordsforthewise/lending-club) | Framework Validation (Regression) | Not used in model training or framework calibration. Contains 2.26M rows and 151 features. | Used exclusively for framework validation under three managerial expertise scenarios (non-expert, intermediate, expert). |

---

## 📝 License

This project is licensed under the MIT License. See the LICENSE file for details.

The underlying datasets retain their respective licenses (IBM under CC BY 4.0; UK Government under Open Government Licence v3.0; Lending Club under their own terms).

---

## 🤝 Acknowledgments

- The authors would like to thank the open-source communities behind Scikit-learn, XGBoost, LightGBM, SHAP, and Captum.
- The UK Government for providing the Payment Practices dataset under OGL v3.0.
- Smart Data Foundry (SDF) for informing our understanding of SME data structures, though access to their microdata was beyond the scope of this study.

---

## 📧 Contact

For questions, issues, or requests regarding the code, please open an issue on this GitHub repository or contact:

Ali Nabizadeh Lamiry

ali.nabizadeh79@yahoo.com


