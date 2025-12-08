Predictive Maintenance using Machine Learning (AI4I 2020 Dataset)

Advanced Supervised ML Project | Real-World Manufacturing Failure Prediction
..............................................................................................
Project Overview

Unexpected machine failures cause major financial losses across manufacturing industries. This project builds an end-to-end Predictive Maintenance Machine Learning system to predict machine failure in advance using real industrial sensor telemetry data.

The system helps organizations:

Reduce unplanned downtime

Optimize maintenance schedules

Minimize repair costs

Improve machine reliability

This solution uses supervised learning, advanced feature engineering, class imbalance handling, model explainability (SHAP), and real-time deployment using Streamlit.

It simulates a real manufacturing setup where machines experience failures due to:

Mechanical load

Torque fluctuations

Tool wear

Overheating

High rotational speeds
..............................................................................................
The objective is to build an end-to-end pipeline that includes:

✔️ Exploratory Data Analysis (EDA)
✔️ Feature Engineering (Domain-driven)
✔️ Handling class imbalance
✔️ ML model training & optimization
✔️ Evaluation using real-world metrics
✔️ Deployment-ready pipeline

This project is designed to reflect real-world predictive maintenance use cases commonly used in manufacturing analytics
....................................................................................................
🎯 Business Problem Statement

Manufacturing machines operate under varying loads, temperatures, and wear conditions. Sudden failures lead to production loss and high maintenance costs.
The goal is to predict whether a machine will fail before it actually breaks down, allowing preventive maintenance instead of reactive repairs.
...................................................................................................
Dataset Information

Dataset Name: AI4I 2020 Predictive Maintenance Dataset
Source: UCI Machine Learning Repository / Kaggle

Key Features Used:

Air Temperature (K)

Process Temperature (K)

Rotational Speed (rpm)

Torque (Nm)

Tool Wear (minutes)

Target Variable:

Machine Failure

0 → No Failure

1 → Failure

Additional Failure Labels:

Tool Wear Failure (TWF)

Heat Dissipation Failure (HDF)

Power Failure (PWF)

Overstrain Failure (OSF)

Random Failure (RNF)
....................................................................................
✨ Key Features of This Project
🔍 1. Exploratory Data Analysis (EDA)

Distribution of sensor data

Outlier detection using IQR

Correlation heatmaps

Failure vs sensor relationships

Class imbalance analysis

🛠 2. Real Domain-Based Feature Engineering

Created high-signal synthetic features used in industry:

Temp_diff → Thermal stress indicator

Power = Torque × Rotational Speed

High_Load flag

Tool_Wear_Ratio

Tool_Expiry_Risk (>80% wear)

Mechanical_Stress_Index

Removal of leakage columns (TWF, HDF, PWF, OSF, RNF)

⚖️ 3. Imbalanced Data Handling

Applied SMOTE oversampling

Ensured no data leakage in train-test split

🤖 4. Supervised ML Modeling

Models trained & compared:

Logistic Regression

Random Forest Classifier

XGBoost Classifier

Gradient Boosting

Metrics used:

AUC-ROC

F1-score (important due to imbalance)

Precision, Recall

Confusion Matrix

🚀 5. Best Model Selection

Random Forest / XGBoost emerges as the best performer with high recall — essential in predictive maintenance.
...............................................................................................
Project Workflow
1️⃣ Import & Clean Data

Checked missing values

Converted datatypes

Removed leakage features

2️⃣ Exploratory Data Analysis

Pairplots

Histograms

Correlation heatmaps

Boxplots comparing failure vs non-failure sensors

Outlier analysis

3️⃣ Feature Engineering

Created multiple high-impact features:

Feature Meaning
Temp_diff Difference between process & air temperature
Overheat_Risk 1 if Temp_diff > 20 K
Power Torque × Rotational Speed
High_Load Flag for high torque
Tool_Wear_Ratio Percentage wear of tool
Tool_Expiry_Risk 1 if Tool_Wear_Ratio > 0.8
Mechanical_Stress_Index Combined stress score

These significantly boost model performance.

4️⃣ Imbalance Handling

Applied SMOTE

5️⃣ Model Training

Compared multiple models.
The final model achieved:

ROC-AUC: ~0.96

Recall: High → Fewer missed failures

F1 Score: Balanced

6️⃣ Deployment-Ready Pipeline

Built using:

sklearn.pipeline

ColumnTransformer

joblib for model saving

🏆 Results

Best Model: Random Forest / XGBoost
Why?

High recall

Low false negatives

Robust to noise

Good interpretability via feature importance

Top Features:

Power

Mechanical Stress Index

Tool Wear

Torque

Temp_diff

📈 Visuals Included in the Notebook

This project contains:

✔️ Correlation heatmap
✔️ Outlier plots
✔️ ROC curve
✔️ Confusion matrix
✔️ Feature importance plot
✔️ Failure probability visualization
.....................................................................................
⚙️ Technologies & Tools Used

Programming: Python

Libraries:

pandas, numpy

matplotlib, seaborn

scikit-learn

imbalanced-learn (SMOTE)

xgboost

shap

streamlit

Model Deployment: Streamlit

Version Control: Git & GitHub
....................................................................................
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

..............................................................................
Key Results & Business Impact

Reduced unplanned downtime through early failure detection

Improved maintenance efficiency using predictive alerts

Enabled proactive decision-making using real-time risk scoring

Produced an explainable, production-ready ML system
