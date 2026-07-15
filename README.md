# SME Cash Flow Forecasting: Accuracy vs. Interpretability

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-Submitted-brightgreen.svg)]()

This repository contains the official implementation and replication code for the research paper:

> **"Context-Aware Model Selection for SME Cash Flow Forecasting: A Leakage-Free Evaluation of Accuracy and Interpretability"**  
> *Submitted to Applied Intelligence (Springer)*

This study rigorously evaluates five machine learning models—**Linear Regression, Random Forest, XGBoost, LightGBM, and Neural Network (MLP)**—for predicting SME cash flow. Unlike prior studies, we strictly prevent **look-ahead bias** using time-series validation, and we introduce a **novel dynamic decision framework (Ω)** to help non-technical SME managers select the optimal model based on data context.

---

## 📄 Overview

Small and Medium Enterprises (SMEs) face significant liquidity risks due to unpredictable payment delays. While advanced ML models promise high accuracy, they often fail in practice due to **data leakage** and **lack of interpretability**.

**Our key contributions are:**
1.  **Leakage-Free Validation:** Removal of future-dependent features (e.g., `DaysLate`) and strict chronological train/test splits for realistic performance estimates.
2.  **Comprehensive Interpretability:** Comparison of TreeSHAP, TreeInterpreter, LIME, and Captum (Integrated Gradients/DeepLIFT) across models.
3.  **The Ω Framework:** A weighted decision rule (\( \Omega = 0.464L + 0.233\rho + 0.176E + 0.085V + 0.042\Phi \)) derived via **Analytic Hierarchy Process (AHP)** to recommend Linear Regression (for invoice-level data) or Random Forest (for firm-level data).

---

## ✨ Key Features

- **Time-Series Splitting:** Chronological ordering prevents the use of future information.
- **Grouped Splits:** Ensures all records from a single UK firm are kept within the same set (train/test) to prevent data mingling.
- **Dual Dataset Analysis:** 
  - *IBM Late Payment Histories* (Invoice-level, ~2.5k records)
  - *UK Government Payment Practices* (Firm-level, ~8k firms)
- **Economic Impact:** Converts RMSE improvements into tangible monetary savings (£).
- **Reproducible:** All random seeds are fixed (42) and the code is fully documented.

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


1. IBM Late Payment Histories (Invoice-Level)

Source:  [Finance Factoring - IBM Late Payment Histories](https://www.kaggle.com/datasets/hhenry/finance-factoring-ibm-late-payment-histories/data)

Task: Regression (Predict DaysToSettle).

Process: Features are restricted to those available at invoice issuance (InvoiceAmount, CreditPeriod, PaperlessBill). DaysLate and Disputed are strictly excluded to prevent look-ahead bias.

2. UK Government Payment Practices (Firm-Level)
   
Source: [UK Government Payment Practices](https://www.kaggle.com/datasets/saikiran0684/payment-practices-of-uk-buyers)

Task: Regression (Predict Average Time to Pay normalized to a 60-day threshold).

Splitting: Grouped time-series split (70% oldest companies train, 30% newest test).

Note: To use the exact preprocessed data, place the raw .csv files in the /data directory. The preprocessing scripts will automatically handle cleaning and scaling.

---

## 📝 License
This project is licensed under the MIT License. See the LICENSE file for details.

The underlying datasets retain their respective licenses (IBM under CC BY 4.0; UK Government under Open Government Licence v3.0).


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


