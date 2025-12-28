import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.stattools import adfuller
import pmdarima as pm
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, IsolationForest
import matplotlib.pyplot as plt

st.title("📊 Energy Forecasting & Analysis")

# -----------------------------------------------------
# CHECK DATA AVAILABLE
# -----------------------------------------------------
if "clean_data" not in st.session_state:
    st.error("⚠ Please upload and clean your dataset on Page 1 before accessing this page.")
    st.stop()

df = st.session_state["clean_data"]

# -----------------------------------------------------
# Prepare Data
# -----------------------------------------------------
df_unit = df[["date","E-Today(KWH)","event","Season","months","days","week","IsWeekend","Quarter"]].drop_duplicates(subset="date", ignore_index=True)
df_unit["fflags"] = df_unit["event"].apply(lambda x: 0 if x=="no fault" else 1)
df_unit = df_unit[df_unit["date"] <= pd.Timestamp("2025-08-26 00:00:00")]

# -----------------------------------------------------
# Plot kWh Production
# -----------------------------------------------------
st.subheader("📈 Daily kWh Production")
fig = px.line(df_unit, x='date', y='E-Today(KWH)', title='Daily kWh Production', markers=True)
st.plotly_chart(fig)

# -----------------------------------------------------
# Stationarity Test (ADF)
# -----------------------------------------------------
st.subheader("🔍 ADF Stationarity Test")
def adf_test(series):
    result = adfuller(series)
    st.write(f"ADF Statistic: {result[0]:.4f}")
    st.write(f"p-value: {result[1]:.4f}")
    if result[1] <= 0.05:
        st.success("Reject null hypothesis: Data is stationary ✅")
    else:
        st.warning("Fail to reject null hypothesis: Data is not stationary ⚠️")

adf_test(df_unit['E-Today(KWH)'])

# -----------------------------------------------------
# SARIMAX Forecast
# -----------------------------------------------------
st.subheader("📊 SARIMAX Forecast (Next 4 Days)")
df_unit['Season'] = df_unit['Season'].astype('category')
season_dummies = pd.get_dummies(df_unit['Season'], prefix='Season', drop_first=True)
exog = pd.concat([df_unit[['fflags']], season_dummies], axis=1).astype(float)
y = df_unit['E-Today(KWH)'].astype(float)

model = pm.auto_arima(y, exog=exog, start_p=2, start_q=1, max_p=4, max_q=2,
                      m=7, start_P=0, seasonal=True, d=1, D=1,
                      trace=True, error_action='ignore', suppress_warnings=True)

st.write("SARIMAX model fitted successfully!")

# Forecast next 4 days (assume no faults)
future_exog = pd.DataFrame({
    'fflags': [1,1,1,1],
    'Season_Spring': [0,0,0,0],
    'Season_Summer': [0,0,0,0],
    'Season_Winter': [0,0,0,0]
}, dtype=float)

forecast = model.predict(n_periods=4, exog=future_exog)

st.write("Predicted kWh (next 4 days):")
st.dataframe(pd.DataFrame(forecast, columns=['Forecast']))

# Plot forecast
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df_unit['date'], y=y, mode='lines+markers', name='Actual'))
future_dates = pd.date_range(df_unit['date'].max() + pd.Timedelta(days=1), periods=4)
fig2.add_trace(go.Scatter(x=future_dates, y=forecast, mode='lines+markers', name='Forecast'))
fig2.update_layout(title="SARIMAX Forecast vs Actual", xaxis_title="Date", yaxis_title="kWh")
st.plotly_chart(fig2)

# -----------------------------------------------------
# RANDOM FOREST FEATURE IMPORTANCE
# -----------------------------------------------------
st.subheader("🌳 Random Forest Feature Importance")

df_rf = df_unit.drop(columns=["Quarter","fflags",'days',"week"]).copy()
categorical_cols = ['Season',"event","months"]
encoder = OrdinalEncoder()
df_rf[categorical_cols] = encoder.fit_transform(df_rf[categorical_cols])

X = df_rf.drop(columns=['E-Today(KWH)', 'date'])
y_rf = df_rf['E-Today(KWH)']

X_train, X_test, y_train, y_test = train_test_split(X, y_rf, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=200,max_depth=7, random_state=42)
rf_model.fit(X_train, y_train)

importance = rf_model.feature_importances_
features = X.columns

# Display feature importance
importance_df = pd.DataFrame({'Feature': features, 'Importance': importance}).sort_values(by='Importance', ascending=False)
st.dataframe(importance_df)

# Plot feature importance
fig3, ax = plt.subplots()
ax.bar(importance_df['Feature'], importance_df['Importance'])
ax.set_xlabel("Importance")
ax.set_title("Random Forest Feature Importance")
st.pyplot(fig3)

# -----------------------------------------------------
# ISOLATION FOREST ANOMALY DETECTION
# -----------------------------------------------------
st.subheader("🚨 Anomaly Detection with Isolation Forest")

df = df_unit.drop(columns=["Quarter","fflags",'days',"week"]).copy()

# Encode categorical variables
categorical_cols = ['Season',"event","months"]


encoder = OrdinalEncoder()
df[categorical_cols] = encoder.fit_transform(df[categorical_cols])



df_iso = df.copy()
df_iso['IsWeekend'] = df_iso['IsWeekend'].astype(int)
features_iso = ['E-Today(KWH)', 'event', 'Season', 'months', 'IsWeekend']
X_iso = df_iso[features_iso]

iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
iso_forest.fit(X_iso)
df_iso['anomaly'] = iso_forest.predict(X_iso)
anomalies = df_iso[df_iso['anomaly'] == -1]

st.write("Detected anomalous days:")
st.dataframe(anomalies[['date', 'E-Today(KWH)', 'anomaly']])

# Plot anomalies
fig4, ax = plt.subplots(figsize=(12,5))
ax.plot(df_iso['date'], df_iso['E-Today(KWH)'], label='Energy Production')
ax.scatter(anomalies['date'], anomalies['E-Today(KWH)'], color='red', label='Anomaly')
ax.set_xlabel('Date')
ax.set_ylabel('E-Today(KWH)')
ax.set_title('Energy Production Anomalies')
ax.legend()
st.pyplot(fig4)
