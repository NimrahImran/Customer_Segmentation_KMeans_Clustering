# Mall Customer Segmentation — KMeans Clustering

An end-to-end unsupervised machine learning pipeline that segments mall customers into distinct groups based on income and spending behavior, using **KMeans Clustering**.

## 📌 Problem Statement
- **Objective:** Group mall customers into meaningful segments based on their annual income and spending patterns, without any predefined labels
- **Dataset:** Mall Customer Segmentation dataset (`Mall_Customers.csv`) — 200 records, 5 columns
- **Problem Type:** Unsupervised Learning (Clustering) — no target variable
- **Business Relevance:** Enables targeted marketing strategies by identifying distinct customer groups (e.g. high-income low-spenders, budget-conscious shoppers, big spenders), helping businesses tailor promotions and improve customer retention
- **Success Metric:** Silhouette Score (higher is better, range -1 to 1) → **Achieved: 0.5584**

## 📊 Dataset
- **Rows:** 200 (196 after removing 4 duplicate rows)
- **Features used for clustering:** `Annual Income (k$)`, `Spending Score (1-100)`
- **Columns excluded:** `CustomerID` (identifier), `Gender`, `Age` — clustering was performed on income and spending behavior only
- No missing values or constant columns detected

## 🛠 Tech Stack
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn

## 🔍 Pipeline / Methodology
1. **Data Loading & Validation** — shape, info, summary statistics
2. **Data Cleaning** — removed 4 duplicate rows, checked for missing/constant columns
3. **Exploratory Data Analysis** — distribution plots, boxplots for income and spending score
4. **Outlier Treatment** — boxplot-based capping applied to numeric features
5. **Feature Engineering** — checked for highly correlated features; none found (only 2 features used)
6. **Preprocessing** — scaling applied to numeric features before clustering
7. **Algorithm Selection** — **KMeans** clustering algorithm
8. **Optimal Cluster Search** — tested multiple values of *k* and selected the best using silhouette score
9. **Model Fitting** — final KMeans model fit with the optimal number of clusters
10. **Cluster Evaluation** — Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Score, Inertia
11. **Visualization** — 2D cluster scatter plots (3D visualization skipped, as only 2 features were used)

## 📈 Results

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| **Number of Clusters Found** | **5** | Determined to be optimal via silhouette analysis |
| **Silhouette Score** | **0.5584** | Reasonably well-separated, moderately dense clusters (range: -1 to 1, higher is better) |
| Davies-Bouldin Index | 0.5674 | Good — indicates low overlap between clusters (lower is better) |
| Calinski-Harabasz Score | 250.75 | Higher score supports well-defined, dense clusters |
| Inertia (KMeans) | 62.71 | Sum of squared distances of points to their closest cluster center |
| Noise Points | 0 | KMeans assigns every point to a cluster (no outlier/noise detection, unlike DBSCAN) |

> Note: This is a **clustering** problem, so there is no accuracy, precision, or recall — those metrics require known labels, which don't exist in unsupervised learning. Cluster quality is instead judged using Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Score.

## 🔑 Key Insights
- The optimal number of customer segments was found to be **5**, confirmed by testing a range of cluster counts and selecting the one with the highest silhouette score
- A silhouette score of 0.5584 indicates **moderately well-separated clusters** — not perfectly distinct, but with clear structure (scores above 0.5 are generally considered reasonable for real-world customer data)
- Since only `Annual Income` and `Spending Score` were used, the resulting 5 segments likely correspond to intuitive customer types such as: high income/high spending, high income/low spending, low income/high spending, low income/low spending, and an average-income/average-spending middle segment

## 📂 Folder Structure
```
Mall-Customer-Segmentation/
│
├── data/
│   └── Mall_Customers.csv
├── Customer_Segmentation_KMeans_Clustering.ipynb
└── README.md
```

## 👤 Author
**Nimra Muhammad Imran**
*(Your GitHub / LinkedIn link)*
