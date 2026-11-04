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

## 📂 Data Sources & Collection Methodology

The dataset utilized in this project was compiled from trusted official institutional and governmental portals spanning from 2010 to 2024 to ensure high accuracy, consistency, and reliability. 

### **Primary Data Sources:**
1. **Department of Statistics Malaysia (DOSM):** 
   * Sourced for median household income, poverty rates (`poverty_absolute`, `poverty_hardcore`, `poverty_relative`), literacy levels, and water access rates.
   * Sourced annual population breakdowns by state.
2. **National Property Information Centre (NAPIC):** 
   * Sourced for residential property transaction data, annual/quarterly median house prices, housing stock distributions, and property types.
3. **Malaysian Government Open Data Portal:** 
   * Utilized for macroeconomic time-series data, including monthly Consumer Price Index (CPI) by state, annual Gross Domestic Product (GDP), and monthly lending rates.

### **Data Processing & Handling:**
* **Interpolation (CAGR):** Addressed missing values for survey years in median income and poverty indices by applying the Compound Annual Growth Rate (CAGR) formula to maintain time-series consistency.
* **Index Normalization:** Applied **Min-Max scaling** normalization (0–1 scale) to process positive indicators (e.g., water access, literacy) and negative indicators (e.g., poverty rate) for the Living Quality Index (LQI).
* **Aggregation:** Restructured quarterly primary housing and economic transactions into consolidated annual metrics to align with state-level identifiers.

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
