# Task-8-Mall-Customer-Segmentation-using-K-Means-Clustering

Overview
This project applies K-Means clustering to segment mall customers based on their features like age, income, and spending score. PCA is used for 2D visualization, and the Silhouette Score is used for evaluation.

Dataset
- Source: `Mall_Customers.csv`
- Features used:
  - Gender (encoded)
  - Age
  - Annual Income (k$)
  - Spending Score (1–100)

 Steps Performed
1. Loaded the dataset using Pandas.
2. Encoded the categorical `Gender` feature.
3. Standardized all numerical features.
4. Used the Elbow Method to find the optimal number of clusters (K).
5. Applied K-Means ith the selected value of K.
6. Performed  PCA  to reduce data to 2D.
7. Visualized clusters using Matplotlib and Seaborn.
8. Evaluated cluster quality using the Silhouette Score.

Key Findings
- Optimal number of clusters: **5**
- Silhouette Score: **0.317**

 Requirements
- Python 3.x
- pandas, matplotlib, seaborn, sklearn

Dataset : https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python
