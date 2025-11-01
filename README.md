# 🧠 Sales Forecasting and Business Insights Project

**Tech Stack:** Python, Prophet, Streamlit, Plotly, Ollama (LLaMA)

---

###📘 Overview
This project builds an **end-to-end Sales Forecasting and Insights system** that predicts future sales trends using **time-series analysis** and provides **AI-generated recommendations** to improve forecasted margins.

The solution combines **data science** (Prophet-based forecasting) with **AI insights** (LLaMA via Ollama) in a single interactive **Streamlit dashboard**, allowing users to upload their own datasets and obtain both forecasts and business recommendations — all running locally, with full data privacy.

---

###🎯 Objectives
- Forecast future sales using historical data and time-series modeling.  
- Identify seasonal trends, growth patterns, and demand fluctuations.  
- Provide actionable, AI-generated recommendations to improve business performance.  
- Deliver a simple, interactive, and local web interface for end-users.

---

## ⚙️ Features
✅ Upload any time-series CSV file (Date + Sales columns).  
✅ Perform **Exploratory Data Analysis (EDA)** with visual trends.  
✅ Train and forecast using **Prophet** for accurate future predictions.  
✅ Generate interactive plots with **Plotly**.  
✅ Use **LLaMA (via Ollama)** to analyze forecasts and suggest ways to:
   - Improve profit margins  
   - Optimize demand planning  
   - Enhance forecast accuracy  
✅ Fully local – No external API calls or cloud dependencies.

---

## 🧩 Architecture
```plaintext
┌──────────────────────────┐
│        User Uploads      │
│     (CSV: Date, Sales)   │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│   Data Cleaning & EDA     │
│ (Pandas, Plotly, Seaborn) │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│   Prophet Time-Series     │
│    Forecast Generation    │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│   LLaMA (via Ollama)      │
│ AI-generated Insights     │
└────────────┬─────────────┘
             ↓
┌──────────────────────────┐
│ Streamlit UI Dashboard   │
└──────────────────────────┘
