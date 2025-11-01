import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly
import requests

st.set_page_config(page_title="Sales Forecast + LLM Insights", layout="wide")

st.title("📈 Sales Forecasting & AI Insights")

st.markdown("""
Upload your sales data (CSV with **Date** and **Sales** columns).  
The app will forecast future sales using **Prophet**, then ask a **locally hosted LLaMA (via Ollama)**  
to provide business insights on how to improve the forecasted margins.
""")

# ---------------- STEP 1: Upload CSV ----------------
uploaded = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.subheader("📊 Uploaded Data Preview")
    st.dataframe(df.head())

    # Column selection
    cols = df.columns.tolist()
    date_col = st.selectbox("Select Date column", cols)
    target_col = st.selectbox("Select Sales/Target column", cols)

    # Convert and clean data
    df[date_col] = pd.to_datetime(df[date_col], infer_datetime_format=True, errors='coerce')
    df = df.dropna(subset=[date_col, target_col])
    df = df.rename(columns={date_col: 'ds', target_col: 'y'})

    # Show quick stats
    st.subheader("📈 Data Overview")
    st.write(df.describe())
    fig = px.line(df, x='ds', y='y', title="Historical Sales Trend")
    st.plotly_chart(fig, use_container_width=True)

    # ---------------- STEP 2: Forecast ----------------
    st.subheader("🔮 Forecast Future Sales")

    periods = st.number_input("Enter number of future periods to forecast (weeks)", min_value=1, max_value=52, value=12)

    if st.button("Run Forecast"):
        with st.spinner("Training Prophet model..."):
            model = Prophet()
            model.fit(df)

            future = model.make_future_dataframe(periods=periods, freq='W')
            forecast = model.predict(future)

            st.success("✅ Forecast completed!")
            st.subheader("Predicted Sales")
            st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods))

            # Plot
            fig2 = plot_plotly(model, forecast)
            st.plotly_chart(fig2, use_container_width=True)

            # ---------------- STEP 3: Prepare LLM Prompt ----------------
            # ---------------- STEP 3: Prepare LLM Prompt ----------------
st.subheader("🤖 Generate AI Insights")

# Save forecast in session_state to persist after reruns
if 'forecast' not in st.session_state:
    st.session_state.forecast = None
if 'prompt' not in st.session_state:
    st.session_state.prompt = None

# Save forecast only after running Prophet
if 'last_forecast' not in st.session_state:
    st.session_state.last_forecast = None

if 'forecast' not in st.session_state or st.session_state.forecast is None:
    st.session_state.forecast = forecast
    st.session_state.last_forecast = forecast

recent_data = df.tail(8).to_dict(orient='records')
forecast_tail = st.session_state.forecast[['ds', 'yhat']].tail(8).to_dict(orient='records')

prompt = f"""
You are a business analyst AI.
Based on the past and forecasted sales data below, provide:
1. A short summary of the forecast (2-3 sentences)
2. Top 3 actionable steps to improve forecasted margins or revenue
3. Key suggestions for better data collection or feature improvements

Recent historical data: {recent_data}
Forecasted data: {forecast_tail}
Be concise and use bullet points.
"""

st.session_state.prompt = prompt
st.code(prompt, language='text')

# ---------------- STEP 4: Call Local LLaMA via Ollama ----------------
if st.button("Ask LLM for Insights"):
    if 'prompt' not in st.session_state or st.session_state.prompt is None:
        st.warning("Please run the forecast first.")
    else:
        with st.spinner("Generating insights using local LLaMA (Ollama)..."):
            try:
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={"model": "llama3.2", "prompt": st.session_state.prompt, "stream": False},
                    timeout=180
                )
                if response.status_code == 200:
                    try:
                        data = response.json()
                        # Handle different response structures
                        output = (
                            data.get("output") or
                            data.get("response") or
                            data.get("message") or
                            data.get("text") or
                            (data["choices"][0]["message"]["content"] if "choices" in data else None)
                        )
                        if not output:
                            # Some Ollama versions return plain text
                            output = str(data)
                        st.subheader("💡 LLM Insights")
                        st.markdown(output)
                    except Exception as e:
                        st.error(f"Error parsing Ollama response: {e}")
                        st.text(response.text)
                else:
                    st.error(f"❌ Ollama returned status {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"⚠️ Error calling Ollama API: {e}")
