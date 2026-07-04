# Cash Flow Forecasting for SMEs Using Invoice Data

## 📌 Project Overview

This repository contains the complete implementation and analysis code for a research project evaluating machine learning models for cash flow forecasting in Small and Medium Enterprises (SMEs) using invoice data. The study investigates the accuracy-interpretability trade-off across five different ML models.

**📝 Paper Status:** *Manuscript in Preparation* – Under review for publication

**🎯 Research Questions:**
1. Which machine learning model achieves the highest predictive accuracy?
2. Which model provides the best interpretability for non-expert SME managers?
3. Which model offers the optimal balance between accuracy and interpretability?

---

## 📊 Dataset Structure

The dataset combines real-world invoice data from IBM with feature engineering methodology from the ICCMS 2026 framework. The final dataset includes the following columns:

| Column | Description |
| :--- | :--- |
| `countryCode` | Customer country identifier |
| `customerID` | Unique customer identifier |
| `invoiceNumber` | Invoice reference number |
| `InvoiceDate` | Date of invoice issuance |
| `InvoiceAmount` | Total invoice amount |
| `DaysToSettle` | **Target variable** - Days between invoice date and full payment |
| `DaysLate` | Days past due date |
| `gross_receivables` | Gross receivable amount |
| `amount_discounted` | Discounted amount |
| `adjustments` | Financial adjustments |
| `credit_sale_amount` | Final credit sale amount |
| `due_date` | Invoice due date |
| `date_full_paid` | Date of full payment |

**📌 Important:** `DaysToSettle` is the **target variable** and is NOT used as a feature to prevent data leakage.

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/sme-cashflow-forecasting.git
cd sme-cashflow-forecasting
