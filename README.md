# About Project

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-Submitted_to_Digital_Finance_Journal-brightgreen.svg)]()

This repository contains the official implementation and replication code for the research paper:

> **"A Mathematical Pre‑Screening Framework for Training‑Free Model Selection in Time‑Series Forecasting"**  
> *Digital Finance Journal*
> 
> **ORCID: 0009-0005-0811-8091**

We evaluate five machine learning models, Linear Regression, Random Forest, XGBoost, LightGBM, and a Neural Network (MLP), for predicting cash flow in small and medium enterprises. Our work has two main pillars. First, we apply strict time-validated splits and remove future-dependent features to prevent look-ahead bias. This gives a realistic picture of model performance. Second, we introduce a context-aware decision framework that recommends the optimal model before any training begins. The framework uses continuous mapping functions derived from theory and calibrated on real SME data. It does not rely on expensive AutoML or heuristic weighting schemes. The framework translates five contextual attributes, (data volume, noise level, granularity, feature-to-instance ratio, and user expertise) into four operational requirements: interpretability, robustness, scalability, and representation capacity. A compatibility score then identifies the best-aligned model.

---

## Preprint

https://www.researchsquare.com/article/rs-11051613/v1

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
|── calibrate.py                                 # Coefficient calibration procedure
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

The author thank the open-source communities behind Scikit-learn, XGBoost, LightGBM, SHAP, and Captum for their contributions. 

We also thank the providers of the three datasets used in this study: IBM for the Late Payment Histories dataset, the UK Government for the Payment Practices dataset under the Open Government Licence v3.0, and Lending Club for the Loan Data dataset. 

We further acknowledge Kaggle as the platform that made these datasets publicly accessible.

---

## 📧 Contact

For questions, issues, or requests regarding the code, please open an issue on this GitHub repository or contact:

lamiry@financetech.dev

Ali Nabizadeh Lamiry

ali.nabizadeh79@yahoo.com


