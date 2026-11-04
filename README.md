# 🏡 Housing Affordability and Living Standards: An Index-Based and Predictive Analysis

An end-to-end data science and business intelligence solution developed by **Team Mathecs** for the **DOSM Datathon 2025**, investigating Malaysia’s housing affordability crisis, regional living quality disparities, and predictive modeling for future property prices.

## 👥 Authors & Acknowledgments

### **Team Mathecs**
* **Thisant A/L Ragu**<sup>1</sup>
* **Jonathan Lau Lee Bin**<sup>2</sup>
* **Madzna**<sup>3</sup>

<sup>1, 2, 3</sup> *Programme of Mathematics with Economics, Faculty of Science and Technology, Universiti Malaysia Sabah*

### **Academic Guidance**
* **Supervisor:** Associate Professor Dr. Siti Rahayu Mohd. Hashim
  * *Programme of Mathematics with Economics, Faculty of Science and Technology, Universiti Malaysia Sabah*

## 🔍 Project Overview
Malaysia faces a structural economic mismatch: median house prices have outpaced household income growth, pushing national median multiples well above affordable benchmarks. To address this, our team developed a dual-objective framework:
1. **Objective 1 (Index Construction):** Conducted a state-level **Housing Affordability Index (HAI)** (2010–2024) using the Median Multiple method and a multidimensional **Living Quality Index (LQI)** (2016–2022) integrating poverty rates, literacy, and water access metrics.
2. **Objective 2 (Predictive Modeling):** Applied machine learning and time-series techniques in Python to forecast future median house prices across Malaysian states based on macroeconomic drivers (GDP, CPI, population, lending rates, and household income).

---

## 📊 Live Deliverables & Preview
* **Interactive Web App:** [Streamlit Live Application](https://datathonmathecs2025.streamlit.app/)
* **Business Intelligence Dashboard:** [Power BI Interactive Report](https://app.powerbi.com/view?r=eyJrIjoiZjZhZDlhM2MtZmNkNC00MDk4LWFmNjgtM2UwYTUxYTYwNmRjIiwidCI6IjI4YzZkZjUyLWIzMTItNDQ0Mi1hZjU5LTIwNzI1OGJmNjRhYiIsImMiOjEwfQ%3D%3D)

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
