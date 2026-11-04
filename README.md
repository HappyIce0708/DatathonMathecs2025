# DatathonMathecs2025
House Price Forecasting by State

Power BI Dashboard: https://app.powerbi.com/view?r=eyJrIjoiZjZhZDlhM2MtZmNkNC00MDk4LWFmNjgtM2UwYTUxYTYwNmRjIiwidCI6IjI4YzZkZjUyLWIzMTItNDQ0Mi1hZjU5LTIwNzI1OGJmNjRhYiIsImMiOjEwfQ%3D%3D 

Streamlit website: https://datathonmathecs2025.streamlit.app/


# 🏡 Housing Affordability and Living Standards: An Index-Based and Predictive Analysis

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20App-red?logo=streamlit)](https://datathonmathecs2025.streamlit.app/)
[![Power BI](https://img.shields.io/badge/Power%20BI-Interactive%20Dashboard-yellow?logo=powerbi)](https://app.powerbi.com/)

An end-to-end data science and business intelligence solution developed by **Team Mathecs** for the **DOSM Datathon 2025**, investigating Malaysia’s housing affordability crisis, regional living quality disparities, and predictive modeling for future property prices[cite: 18, 19].

---

## 🔍 Project Overview
Malaysia faces a structural economic mismatch: median house prices have outpaced household income growth, pushing national median multiples well above affordable benchmarks[cite: 22]. To address this, our team developed a dual-objective framework:
1. **Objective 1 (Index Construction):** Conducted a state-level **Housing Affordability Index (HAI)** (2010–2024) using the Median Multiple method and a multidimensional **Living Quality Index (LQI)** (2016–2022) integrating poverty rates, literacy, and water access metrics[cite: 23, 27, 28].
2. **Objective 2 (Predictive Modeling):** Applied machine learning and time-series techniques in Python to forecast future median house prices across Malaysian states based on macroeconomic drivers (GDP, CPI, population, lending rates, and household income)[cite: 9, 30].

---

## 📊 Live Deliverables & Preview
* **Interactive Web App:** [Streamlit Live Application](https://datathonmathecs2025.streamlit.app/)[cite: 16]
* **Business Intelligence Dashboard:** [Power BI Interactive Report](https://app.powerbi.com/view?r=eyJrIjoiZjZhZDIhM2MtZmNzMnNC00MDk4LWFlZmNlMjUwUyTYXTYWNmRjliwidCl6Ij4yZk4zKjU5LWIwMi00MjU5LTU1LTIwMjFjOGJmN0JjYiIsImMiOjEwMi5-ZU5LTQ0Mi0zUjUtLTZnNS1lbG0zNjFjOGJnNmRyYjlsbWMiOjEwMi5jOjEwMiJ9)[cite: 45]

<p align="center">
  <img src="output/Datathon 2025 Streamlit Layout.png" alt="Streamlit App Preview" width="80%"/>
</p>

---

## 🗂️ Repository Structure
```text
DatathonMathecs2025/
├── data/
│   └── Datathon_ML_Hybrid_Ori.csv      # Processed socioeconomic and property dataset
├── output/
│   ├── Datathon 2025 Power BI Layout.png  # Power BI dashboard layout preview
│   └── Datathon 2025 Streamlit Layout.png # Streamlit web application interface preview
├── num2.py                              # Core Streamlit app & ML forecasting pipeline script
├── requirements.txt                     # Python package dependencies
├── runtime.txt                          # Python version specification for deployment
└── README.md                            # Project documentation and portfolio showcase
