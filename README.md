🚀 Predictive Maintenance using Machine Learning (AI4I 2020 Dataset)
Advanced Supervised ML Project | Real-World Manufacturing Failure Prediction
📌 Project Overview

Unexpected machine breakdowns can cause massive production losses in manufacturing industries.
This project builds an end-to-end Predictive Maintenance System that predicts machine failure before it happens using real industrial sensor telemetry.

The system helps industries:

Reduce unplanned downtime

Prevent sudden breakdowns

Optimize maintenance schedules

Minimize repair costs

Improve reliability & productivity

This project uses:

Supervised ML

Advanced feature engineering

Imbalanced data handling

Explainability with SHAP

Deployment using Streamlit

It simulates a real industrial setup with failures caused by:

Mechanical load

Torque spikes

Tool wear

Overheating

High rotational speed

🎯 Business Problem Statement

Manufacturing machines operate under varying loads, temperatures, and wear conditions. Unexpected failures lead to:

Production downtime

High maintenance cost

Delay in delivery

Increased safety risks

👉 Goal:
Build a machine learning model that predicts Machine Failure (0/1) ahead of time so teams can schedule preventive maintenance instead of reacting after breakdowns.

📊 Dataset Information

Dataset: AI4I 2020 Predictive Maintenance Dataset
Source: UCI ML Repository / Kaggle

🔧 Key Input Features
Feature Description
Air Temperature (K) Environmental temperature
Process Temperature (K) Machine internal temperature
Rotational Speed (rpm) Mechanical rotation rate
Torque (Nm) Applied mechanical load
Tool Wear (min) Wear level of machine tool
🎯 Target Variable

Machine Failure

0 → No Failure

1 → Failure

🔍 Additional Failure Types

(Treated as leakage → removed)

TWF — Tool Wear Failure

HDF — Heat Dissipation Failure

PWF — Power Failure

OSF — Overstrain Failure

RNF — Random Failure

✨ Key Features of This Project
🔍 1. Exploratory Data Analysis (EDA)

Includes:

Distribution plots

Outlier detection (IQR)

Correlation heatmap

Failure vs sensor patterns

Class imbalance visualization

🛠 2. Real Domain-Based Feature Engineering

Created high-impact features used in real manufacturing:

Feature Meaning
Temp_diff Thermal stress indicator
Power Torque × Rotational Speed
Overheat_Risk 1 if Temp_diff > 20 K
High_Load High torque conditions
Tool_Wear_Ratio % tool wear
Tool_Expiry_Risk 1 if wear > 80%
Mechanical_Stress_Index Combined stress score

✔ These features significantly increased model performance.

⚖️ 3. Imbalanced Data Handling

Applied SMOTE oversampling

Ensured no train-test leakage

🤖 4. Supervised ML Models Trained

Logistic Regression

Random Forest

XGBoost

Gradient Boosting

📈 Evaluation Metrics Used

ROC-AUC

Recall (critical in predictive maintenance)

Precision, F1-Score

Confusion Matrix

🏆 5. Best Model Selection

Best Model: Random Forest / XGBoost
Why?

High Recall → fewer missed failures

Robust against noise

Strong feature importance interpretability

🔝 Top Contributing Features

Power

Mechanical Stress Index

Tool Wear

Torque

Temp_diff

📁 Project Workflow
1️⃣ Data Cleaning

Checked missing values

Converted datatypes

Removed leakage columns

2️⃣ Exploratory Data Analysis

Histograms

Boxplots

Pairplots

Correlation matrix

Outlier visualization

3️⃣ Feature Engineering

Created domain-driven synthetic features (listed above).

4️⃣ Handling Imbalance

Applied SMOTE on training data only

5️⃣ Model Training & Optimization

Compared models

Tuned hyperparameters

Selected best performer

6️⃣ Deployment

Built a live inference pipeline using:

sklearn.pipeline

joblib for model saving

Streamlit UI for real-time failure prediction

📊 Model Performance
Metric Score
ROC-AUC ~0.96
Recall High (critical metric)
F1 Score Strong
🖼️ Visuals Included

✔ Correlation Heatmap
✔ Outlier Plots
✔ ROC Curve
✔ Confusion Matrix
✔ Feature Importance
✔ Failure Probability Plot

⚙️ Technologies & Tools Used

Languages:

Python

Libraries:

pandas, numpy

matplotlib, seaborn

scikit-learn

imbalanced-learn (SMOTE)

xgboost

shap

streamlit

Others:

Git & GitHub for version control

📂 Project Structure
predictive_maintenance_project/
│
├── data/
│ └── ai4i2020.csv
│
├── notebooks/
│ └── EDA_and_Model_Training.ipynb
│
├── models/
│ └── predictive_maintenance_model.pkl
│
├── app.py
├── requirements.txt
└── README.md

💡 Business Impact

Reduced unplanned machine downtime

Enabled predictive & proactive maintenance

Improved operational efficiency

Reduced maintenance costs

Provided explainable ML-driven insights

⭐ Conclusion

This project demonstrates a complete end-to-end real-world Predictive Maintenance System including:

✔ Data analysis
✔ Feature engineering
✔ ML modeling
✔ Deployment
✔ Explainability

It simulates exactly how predictive maintenance works in industries using IoT sensor data.
