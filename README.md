# 🛍️ Retail Customer Intelligence: Future High-Value Customer Prediction

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-green)
![SHAP](https://img.shields.io/badge/Explainable%20AI-SHAP-red)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B)
![Azure](https://img.shields.io/badge/Hosted_on-Azure-blue)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Project Overview

This project demonstrates an **end-to-end machine learning pipeline** for predicting **future high-value customers** using historical purchasing behaviour.

Instead of relying solely on descriptive analytics, this project applies predictive analytics to identify customers who are likely to become valuable in the future. The solution integrates **feature engineering, machine learning, explainable AI (SHAP), and Streamlit deployment** to provide transparent and actionable predictions.

This project reuses the analytical dataset generated from my previous project:

➡️ **Retail Data Engineering Pipeline**

---

## 🌐 Live Demo

🚀 **Try the deployed application here:**

**https://retail-ai-aqilah-bze9g2anaxhjfage.eastasia-01.azurewebsites.net**

The application is deployed on **Microsoft Azure App Service** with **GitHub Actions** for automated CI/CD deployment.

---

## 🎯 Business Problem

Retail companies often have limited marketing budgets and cannot provide premium promotions to every customer.

The objective of this project is to identify customers who are likely to become **future high-value customers**, enabling businesses to:

- Target personalized marketing campaigns
- Improve customer retention
- Prioritize loyalty rewards
- Optimize promotional spending

---

## 🏗️ System Architecture

```mermaid
flowchart TD

A[Retail Analytics Dataset] --> B[Feature Engineering]

B --> C[Customer Behaviour Features]

C --> D[Machine Learning Models]

D --> E[Logistic Regression]

D --> F[Random Forest]

D --> G[XGBoost]

F --> H[Best Model]

H --> I[SHAP Explainability]

H --> J[Streamlit Web Application]
```

---

## 🚀 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- SHAP
- Joblib
- Streamlit
- Matplotlib

---

## 📂 Project Structure

```text
retail-customer-prediction/

│
├── app/
│   └── app.py
│
├── data/
│   ├── analytics_sales.csv
│   ├── customer_future_features.csv
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   └── y_test.csv
│
├── models/
│   └── best_model.pkl
│
├── notebook/
│   ├── 01_prepare_future_prediction_dataset.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_explainability.ipynb
│
├── screenshots/
│
├── requirements.txt
│
└── README.md
```

---

## 📊 Feature Engineering

Customer-level features were generated from transactional data, including:

- Historical Revenue
- Total Orders
- Purchase Frequency
- Historical Average Order Value
- Total Quantity Purchased
- Average Items per Order
- Customer Tenure
- Days Since Last Purchase
- Membership Tier
- Favourite Product Category
- Preferred Payment Method
- Customer Demographics

---

## 🎯 Target Variable

Because this project uses synthetic retail data, a **synthetic future customer value score** was generated using business-inspired behavioural assumptions.

The score incorporates:

- Historical customer spending
- Purchase frequency
- Customer engagement
- Membership tier
- Purchase recency
- Controlled random variation

Customers within the **top 25%** of the generated future value score were labelled as **Future High-Value Customers**.

This approach creates a realistic supervised learning scenario for demonstrating end-to-end machine learning while remaining transparent about the synthetic nature of the dataset.

---

## 🤖 Machine Learning Models

Three classification models were evaluated.

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|--------|---------:|----------:|--------:|---------:|--------:|
| Logistic Regression | 91.3% | 0.800 | 0.870 | 0.833 | 0.969 |
| Random Forest ⭐ | **96.7%** | **1.000** | **0.870** | **0.929** | **0.989** |
| XGBoost | 94.6% | 0.909 | 0.870 | 0.889 | 0.982 |

Random Forest achieved the best overall performance and was selected for deployment.

---

## 🔍 Explainable AI

To improve model transparency, **SHAP (SHapley Additive exPlanations)** was applied to interpret prediction results.

SHAP provides:

- Global feature importance
- Local explanations for individual customers
- Transparent interpretation of model predictions

### Global Feature Importance

<img width="2355" height="2217" alt="shap_beeswarm" src="https://github.com/user-attachments/assets/b159312d-2d03-4468-84cf-1b608796fe2e" />

---

### SHAP Summary Plot

<img width="2402" height="2669" alt="shap_global_bar" src="https://github.com/user-attachments/assets/23711be5-ea5d-4d2a-b8be-d5ff463ecdb3" />

---

### Local Customer Explanation

<img width="2447" height="2644" alt="shap_waterfall_customer_0" src="https://github.com/user-attachments/assets/0c9ef243-9e88-43b5-a065-5c0a33c0b979" />

---

## 💻 Streamlit Application

An interactive Streamlit application was developed to demonstrate real-time prediction.

Users can input customer information including:

- Age
- Membership Tier
- Purchase Frequency
- Historical Spending
- Total Orders
- Customer Tenure

The application predicts:

- Future High-Value Customer
- Prediction Probability

<img width="1470" height="788" alt="image" src="https://github.com/user-attachments/assets/ca43f354-afa8-4c8a-9255-6bbb3dd99672" />

<img width="2822" height="524" alt="image" src="https://github.com/user-attachments/assets/21702d52-8e35-442e-a2d0-7f5189841d37" />

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/retail-customer-prediction.git

cd retail-customer-prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Launch Streamlit

```bash
streamlit run app/app.py
```

---

## 📈 Future Improvements

Potential future enhancements include:

- Containerize the application using Docker
- Deploy using Azure Container Apps or Kubernetes
- Build an automated MLOps pipeline
- Real-time prediction using Kafka
  
---

## 🔗 Related Project

This project uses the analytical dataset generated by my previous repository:

➡️ **Retail Data Engineering Pipeline**

The previous project demonstrates:

- Synthetic retail data generation
- PostgreSQL data warehouse
- Apache Spark ETL
- SQL analytics
- Tableau dashboard development

Together, both projects demonstrate a complete **data engineering → analytics → machine learning → explainable AI workflow**.

---

## 👤 Author

**Aqilah Syahirah*

Master of Science (Computer Science)

Interested in Machine Learning, Explainable AI, Data Engineering, Computer Vision and Applied AI.
