# Customer Segmentation & Behavioral Analytics using Unsupervised Machine Learning

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Clustering-orange.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-brightgreen.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)

## 📌 Executive Summary

This repository contains a comprehensive **Unsupervised Machine Learning Project** focused on **Customer Segmentation** for credit card holders in the banking and financial sector. 

Using 6 months of behavioral transaction data for approximately **9,000 active credit card holders**, we leverage **K-Means Clustering** and **Agglomerative Hierarchical Clustering** to partition customers into distinct, actionable behavioral groups. These segments enable financial institutions to optimize targeted marketing, mitigate credit risk, and personalize credit products.

---

## 📁 Repository Structure

```text
Week-04/
│
├── data/
│   └── DataSet(W4).csv.xls              # Behavioral credit card dataset (9000 rows, 18 features)
│
├── notebooks/
│   └── week4_clustering.ipynb          # Main executable Jupyter Notebook (Fully executed with rich outputs)
│
├── generate_notebook.py                 # Notebook builder script
├── run_and_populate_nb.py               # Automated execution & output embedding pipeline
├── requirements.txt                     # Project dependency specifications
└── README.md                            # Complete technical and business documentation
```

---

## 📊 Dataset Overview

- **Source**: Credit Card Dataset for Clustering (Kaggle)
- **Observations**: 8,950 active cardholders
- **Raw Features**: 18 variables reflecting balance, purchase patterns, cash advances, credit limits, and payment frequency.

### Key Behavioral Features
| Feature Name | Description |
| :--- | :--- |
| `CUST_ID` | Unique identification of credit card holder (Categorical ID, dropped prior to modeling) |
| `BALANCE` | Balance amount left in user account to make purchases |
| `BALANCE_FREQUENCY` | How frequently the Balance is updated (score between 0 and 1) |
| `PURCHASES` | Total amount of purchases made from account |
| `ONEOFF_PURCHASES` | Maximum purchase amount done in one-go transaction |
| `INSTALLMENTS_PURCHASES` | Amount of purchase done in installment |
| `CASH_ADVANCE` | Cash in advance given by the user |
| `PURCHASES_FREQUENCY` | How frequently the Purchases are being made (0 to 1) |
| `ONEOFF_PURCHASES_FREQUENCY` | Frequency of one-off purchases (0 to 1) |
| `PURCHASES_INSTALLMENTS_FREQUENCY` | Frequency of installment purchases (0 to 1) |
| `CASH_ADVANCE_FREQUENCY` | Frequency of cash in advance transactions (0 to 1) |
| `CASH_ADVANCE_TRX` | Number of Transactions made with Cash in Advance |
| `PURCHASES_TRX` | Number of purchase transactions made |
| `CREDIT_LIMIT` | Limit of Credit Card for user |
| `PAYMENTS` | Amount of Payment done by user |
| `MINIMUM_PAYMENTS` | Minimum amount of payments made by user |
| `PRC_FULL_PAYMENT` | Percent of full payment paid by user |
| `TENURE` | Tenure of credit card service for user (in months) |

---

## 🛠️ Methodology & Technical Workflow

### 1. Data Cleaning & Preprocessing
1. **Identifier Dropping**:
   - `CUST_ID` was removed as it serves as a unique key rather than a behavioral feature. Retaining non-informative IDs would distort distance-based calculations.
2. **Missing Value Imputation**:
   - Missing entries were identified in `MINIMUM_PAYMENTS` (313 values) and `CREDIT_LIMIT` (1 value).
   - **Strategy**: Financial attributes exhibit heavy right-skewness with extreme positive outliers. Imputing with mean values would artificially inflate baseline figures. Thus, **median imputation** was applied to preserve the true distribution.
3. **Feature Standardization (`StandardScaler`)**:
   - **Why Mandatory**: Distance-based algorithms (K-Means and Hierarchical Clustering) calculate similarity using Euclidean distance:
     $$d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$$
   - Without scaling, large magnitude attributes (e.g. `BALANCE` up to $30,000$) would completely overwhelm ratio attributes (e.g. `PURCHASES_FREQUENCY` between $0$ and $1$). `StandardScaler` standardizes each feature to mean $\mu = 0$ and standard deviation $\sigma = 1$:
     $$z = \frac{x - \mu}{\sigma}$$

---

## 📈 Part 1: K-Means Clustering & Customer Profiling

### Cluster Selection ($k = 2$ to $10$)
We evaluated cluster quality across $k \in [2, 10]$ using two metrics:
- **Inertia (Within-Cluster Sum of Squares - WCSS)**: Measures cluster compactness.
- **Silhouette Score**: Evaluates cluster separation quality (range $[-1, 1]$).

```text
k =  2 | Inertia (WCSS):  117625.59 | Silhouette Score: 0.2088
k =  3 | Inertia (WCSS):   96372.48 | Silhouette Score: 0.2503
k =  4 | Inertia (WCSS):   84852.79 | Silhouette Score: 0.2140  <-- Optimal Choice
k =  5 | Inertia (WCSS):   75432.86 | Silhouette Score: 0.1947
k =  6 | Inertia (WCSS):   68900.22 | Silhouette Score: 0.2012
```

### Optimal $k$ Selection Rationale
- **Elbow Point**: The plot of $k$ vs. Inertia reveals a distinct "elbow" bend at **$k = 4$**.
- **Silhouette Validation**: $k=4$ maintains strong cluster separation while creating highly interpretable, distinct financial personas for marketing and risk operations.

---

### 🏷️ Identified Customer Segments & Business Personas

```text
Distribution of Customers (Total: 8,950):
  Cluster 0: 3,967 customers (44.3%) -> Low Activity / Budget Holders
  Cluster 1: 1,188 customers (13.3%) -> Cash Advance Seekers (High Borrowers)
  Cluster 2: 2,519 customers (28.1%) -> Installment Buyers & Active Shoppers
  Cluster 3: 1,276 customers (14.3%) -> VIP High Spenders (Big Buyers)
```

| Cluster ID | Segment Name | Behavioral Characteristics | Recommended Business Action |
| :---: | :--- | :--- | :--- |
| **0** | **Low Activity / Budget Holders** | Low balances, near-zero purchases, minimal cash advances. | Run re-engagement email campaigns, offer zero-annual-fee incentives, and introduce cashback rewards to activate usage. |
| **1** | **Cash Advance Seekers** | High `CASH_ADVANCE` balances and frequency, low point-of-sale spending. | Monitor credit risk and default probability closely; market personal consolidation loans with structured repayment terms. |
| **2** | **Installment Buyers** | High `INSTALLMENTS_PURCHASES` and installment frequency, frequent retail transactions. | Partner with retailers for 0% EMI financing, offer merchant cashback deals, and expand buy-now-pay-later (BNPL) credit limits. |
| **3** | **VIP High Spenders** | Highest `BALANCE`, maximum `PURCHASES` and `ONEOFF_PURCHASES`, high credit limit. | Provide VIP concierge services, premium travel rewards, exclusive card upgrades, and credit limit expansion to maximize retention. |

---

## 🌳 Part 2: Hierarchical Clustering Analysis

### Methodology & Dendrogram
- **Sample Selection**: A representative random sample of 300 customers was drawn to construct Ward's linkage dendrogram without memory bottleneck.
- **Linkage Criterion**: Ward's minimum variance method (`scipy.cluster.hierarchy.linkage` with `method='ward'`).
- **Cut Threshold**: A horizontal cut threshold at Euclidean distance $y = 22.0$ confirms an optimal 4-cluster partition.

### Cross-Tabulation Matrix (K-Means vs. Agglomerative)
| K-Means \ Hierarchical | Cluster 0 | Cluster 1 | Cluster 2 | Cluster 3 |
| :---: | :---: | :---: | :---: | :---: |
| **Cluster 0** | **124** | 0 | 5 | 0 |
| **Cluster 1** | 0 | **44** | 1 | 0 |
| **Cluster 2** | 4 | 0 | **78** | 0 |
| **Cluster 3** | 0 | 0 | 2 | **42** |

> **Finding**: Strong diagonal concentration in cross-tabulation confirms high agreement between K-Means and Agglomerative Hierarchical clustering algorithms.

---

## ⚖️ Algorithm Comparison Report

| Feature | K-Means Clustering | Hierarchical (Agglomerative) |
| :--- | :--- | :--- |
| **Time Complexity** | $O(k \cdot N \cdot i)$ (Fast & Linear) | $O(N^3)$ (Prohibitive on large $N$) |
| **Space Complexity** | $O(N \cdot p)$ | $O(N^2)$ (Requires full distance matrix) |
| **Scalability** | Exceptional (Scales to millions of users) | Limited (Requires sampling for $N > 10,000$) |
| **New Data Prediction** | Direct centroid assignment (`kmeans.predict()`) | Not natively supported (Requires refitting) |
| **Visualization** | Centroid heatmaps, PCA scatter plots | Hierarchical Dendrogram |
| **Business Recommendation** | **Selected for Production Deployment** | Ideal for exploratory structural analysis |

---

## 🚀 How to Run the Project

### Prerequisites
Ensure Python 3.10+ is installed on your system.

### 1. Clone & Setup Environment
```bash
git clone https://github.com/your-username/Week-04-Customer-Segmentation.git
cd Week-04-Customer-Segmentation

# Install required dependencies
pip install -r requirements.txt
```

### 2. Launch Jupyter Notebook
```bash
jupyter notebook notebooks/week4_clustering.ipynb
```

---

## 📦 Deliverables Checklist

- [x] **Executable Jupyter Notebook**: `notebooks/week4_clustering.ipynb` (All cells executed, outputs and figures embedded).
- [x] **Complete Dataset**: `data/DataSet(W4).csv.xls`.
- [x] **Dependencies File**: `requirements.txt`.
- [x] **Documentation**: Comprehensive `README.md` covering methodology, visualizations, cluster profiles, and business recommendations.

---

## 📄 License & Acknowledgments

- **Dataset Credit**: Arjun Bhasin (Kaggle Credit Card Dataset for Clustering).
- **Internship Program**: Week 4 AI/ML Internship Assignment — Unsupervised Learning & Customer Segmentation.
