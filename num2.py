import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from pmdarima import auto_arima
import warnings

warnings.filterwarnings("ignore")

# =========================================================
# App Title
# =========================================================
st.set_page_config(page_title="House Price Forecasting", layout="wide")
st.title("🏡 House Price Forecasting by State")
with st.sidebar:
    st.header("Data")
    st.caption("Upload your CSV or use a path in project.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    default_path = st.text_input("Or CSV path", value="Datathon_ML_Hybrid_Ori.csv")
    load_btn = st.button("Load Data")

# =========================================================
# Load dataset
# =========================================================
@st.cache_data
def load_data(file):
    if isinstance(file, str):
        df = pd.read_csv(file)
    else:
        df = pd.read_csv(file)
    df['State'] = df['State'].str.strip().str.upper()
    df.drop(columns=[col for col in df.columns if col.startswith("Unnamed")], inplace=True, errors='ignore')
    return df

if "df" not in st.session_state:
    st.session_state.df = None

if load_btn:
    if uploaded is not None:
        st.session_state.df = load_data(uploaded)
    else:
        st.session_state.df = load_data(default_path)

if st.session_state.df is None:
    st.info("Please load your dataset to proceed.")
    st.stop()

df = st.session_state.df


# =========================================================
# Sidebar Inputs
# =========================================================
features = ["Household_Income_RM", "Lending_Rate", "CPI", "GDP_RM", "Population_000"]
target_col = "Median_House_Price"
unique_states = sorted(df['State'].unique())
state_input = st.sidebar.selectbox("Select a State", unique_states)

# Temporarily define state_df to get last_year
temp_df = df[df["State"] == state_input].sort_values("Year").reset_index(drop=True)
if temp_df.empty or len(temp_df) < 5:
    st.sidebar.warning("Not enough data for this state.")
    st.stop()

last_year = temp_df['Year'].max()
forecast_year = st.sidebar.number_input(f"Forecast Year (>{last_year})", min_value=last_year + 1, step=1)

confirm_selection = st.sidebar.button("✅ Confirm Selection")

if not confirm_selection:
    st.info("Please select a state and forecast year, then click 'Confirm Selection'.")
    st.stop()

# Now define state_df after confirmation
state_df = temp_df

if st.sidebar.button("🔄 Reset"):
    st.experimental_rerun()

# =========================================================
# Prepare data
# =========================================================
state_df = df[df["State"] == state_input].sort_values("Year").reset_index(drop=True)

if len(state_df) < 5:
    st.error(f"Not enough data for {state_input}. Minimum 5 years required.")
    st.stop()

state_df[features] = state_df[features].fillna(state_df[features].mean())
y = state_df[target_col].fillna(state_df[target_col].mean())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(state_df[features])
X = pd.DataFrame(X_scaled, columns=features)

split = int(0.8 * len(X))
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# =========================================================
# Train Models
# =========================================================
results = {}

try:
    arima_model = auto_arima(y_train, seasonal=False, suppress_warnings=True)
    y_pred_arima = arima_model.predict(n_periods=len(y_test))
    results["ARIMA"] = mean_squared_error(y_test, y_pred_arima)
except Exception as e:
    st.warning(f"ARIMA failed: {e}")
    results["ARIMA"] = np.inf

rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
results["RandomForest"] = mean_squared_error(y_test, y_pred_rf)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
results["LinearRegression"] = mean_squared_error(y_test, y_pred_lr)

# =========================================================
# Select Best Model
# =========================================================
best_model_name = min(results, key=results.get)
best_mse = results[best_model_name]
rmse = np.sqrt(best_mse)

# =========================================================
# Forecast
# =========================================================
if best_model_name == "ARIMA":
    model = auto_arima(y, seasonal=False, suppress_warnings=True)
    n_periods = int(forecast_year - state_df["Year"].max())
    if n_periods <= 0:
        st.error("Forecast year must be greater than the last available year.")
        st.stop()
    arima_future = model.predict(n_periods=n_periods)
    forecast_value = float(arima_future.iloc[-1])
else:
    model = rf if best_model_name == "RandomForest" else lr
    model.fit(X, y)
    latest_features = X.iloc[-1].values.reshape(1, -1)
    forecast_value = float(model.predict(latest_features)[0])

lower_bound = max(0, forecast_value - rmse)
upper_bound = forecast_value + rmse

# =========================================================
# Feature Importance
# =========================================================
rf_explain = RandomForestRegressor(random_state=42)
rf_explain.fit(X, y)
importances = permutation_importance(rf_explain, X, y)
importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances.importances_mean
}).sort_values(by="Importance", ascending=False)

# =========================================================
# Display Results
# =========================================================
st.subheader("📊 Forecast Results")
st.markdown(f"**State:** {state_input}")
st.markdown(f"**Forecast Year:** {forecast_year}")
st.markdown(f"**🏆 Best Model:** {best_model_name}")
st.metric("💰 Predicted Median House Price", f"RM {forecast_value:,.2f}")
st.metric("🔻 Lower Bound", f"RM {lower_bound:,.2f}")
st.metric("🔺 Upper Bound", f"RM {upper_bound:,.2f}")
st.metric("📉 RMSE", f"{rmse:,.2f}")

st.subheader("📌 Feature Importance")
st.dataframe(importance_df, use_container_width=True)