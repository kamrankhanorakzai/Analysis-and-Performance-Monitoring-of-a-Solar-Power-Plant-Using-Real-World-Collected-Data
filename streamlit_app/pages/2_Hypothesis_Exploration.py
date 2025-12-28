import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import f_oneway
import numpy as np

st.title("📊 Hypothesis & Question-Driven Analysis")
st.write("Explore deep insights and patterns in your solar plant data.")

# -----------------------------------------------------
# CHECK DATA AVAILABLE
# -----------------------------------------------------
if "clean_data" not in st.session_state:
    st.error("⚠ Please upload and clean your dataset on Page 1 before accessing this page.")
    st.stop()

df = st.session_state["clean_data"]
df.reset_index(inplace=True)

st.success("Cleaned dataset loaded successfully! 🎉")

# -----------------------------------------------------
# UI STYLE
# -----------------------------------------------------
st.markdown("""
<style>
.section {
    padding: 15px;
    border-radius: 12px;
    background-color: #f8f9fa;
    margin-bottom: 25px;
    border-left: 6px solid #4a90e2;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# Q1: Does grid fault lower kWh production?
# =====================================================
with st.expander("🔌 Q1: Does the grid fault lower daily kWh production?", expanded=False):

    faults_per_day = df.groupby("date")["event"].apply(lambda x: (x != "no fault").sum())
    kwh_per_day = df.groupby("date")["E-Today(KWH)"].mean()

    daily_summary = pd.DataFrame({"fault_count": faults_per_day, "kwh": kwh_per_day})
    corr = daily_summary["fault_count"].corr(daily_summary["kwh"])

    st.write(f"**Correlation:** `{corr:.4f}` (Negative means more faults → lower production)")

    fig, ax = plt.subplots(figsize=(6,4))
    sns.regplot(x="fault_count", y="kwh", data=daily_summary, scatter_kws={"alpha":0.6}, ax=ax)
    ax.set_title("Impact of Fault Count on Daily kWh")
    st.pyplot(fig)

# =====================================================
# Q2: Highest monthly production?
# =====================================================
with st.expander("📅 Q2: Which months show the highest average energy production?"):
    result = df.groupby("months")["E-Today(KWH)"].mean().sort_values(ascending=False)
    st.dataframe(result)

# =====================================================
# Q3: Yearly trend?
# =====================================================
with st.expander("📈 Q3: Is energy production increasing or decreasing over 2024–2025?"):

    df['date_ordinal'] = df['date'].map(pd.Timestamp.toordinal)
    X = df['date_ordinal'].values.reshape(-1,1)
    y = df['E-Today(KWH)'].values

    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)

    fig, ax = plt.subplots(figsize=(10,4))
    ax.scatter(df['date'], df['E-Today(KWH)'], alpha=0.5, label='Actual')
    ax.plot(df['date'], y_pred, color='red', label='Trend Line')
    ax.set_title("Energy Production Trend (2024–2025)")
    ax.legend()
    st.pyplot(fig)

    st.write(f"📌 **Trend Slope:** `{model.coef_[0]:.4f}`")

# =====================================================
# Q4: Days with highest kWh
# =====================================================
with st.expander("📆 Q4: Which days of week show highest production?"):
    st.dataframe(df.groupby("days")["E-Today(KWH)"].mean().sort_values(ascending=False))

# =====================================================
# Q5: Weekends vs weekdays
# =====================================================
with st.expander("🌞 Q5: Do weekends produce more or less energy?"):

    weekend_avg = df.groupby("IsWeekend")["E-Today(KWH)"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(5,3))
    sns.barplot(x='IsWeekend', y='E-Today(KWH)', data=weekend_avg, ax=ax)
    ax.set_title("Weekend vs Weekday Production")
    st.pyplot(fig)

    st.dataframe(weekend_avg)

# =====================================================
# Q6: Quarter analysis
# =====================================================
with st.expander("🟦 Q6: Production across quarters"):
    st.dataframe(df.groupby("Quarter")["E-Today(KWH)"].agg(['mean','std','sum']))

# =====================================================
# Q7: Seasons
# =====================================================

with st.expander("❄️☀️ Q7: Seasonal trends (Summer, Winter, etc.)"):
   
    season_avg = df.groupby("Season")["E-Today(KWH)"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(6,4))
    sns.barplot(x='Season', y='E-Today(KWH)', data=season_avg, order=['Winter','Spring','Summer','Autumn'], ax=ax)
    ax.set_title("Seasonal Energy Production")
    st.pyplot(fig)

    st.dataframe(season_avg)

# =====================================================
# Q8: Outlier energy days
# =====================================================
with st.expander("🚨 Q8: Unusually high or low production days"):

    mean_kwh = df['E-Today(KWH)'].mean()
    std_kwh = df['E-Today(KWH)'].std()
    high_threshold = mean_kwh + 2*std_kwh
    low_threshold = mean_kwh - 2*std_kwh

    high = df[df['E-Today(KWH)'] > high_threshold][['date',"E-Today(KWH)"]].drop_duplicates(subset=["date"])
    low = df[df['E-Today(KWH)'] < low_threshold][['date',"E-Today(KWH)"]].drop_duplicates(subset=["date"])

    st.write("🔺 **High energy days:**")
    st.write(high)

    st.write("🔻 **Low energy days:**")
    st.write(low)

# =====================================================
# Q9: Most common fault types
# =====================================================
with st.expander("⚡ Q9: Which fault types are most common?"):

    fault_counts = df[df['event'] != 'no fault']['event'].value_counts()

    fig, ax = plt.subplots(figsize=(14,4))
    sns.barplot(x=fault_counts.index, y=fault_counts.values, ax=ax)
    ax.set_title("Most Common Fault Types")
    st.pyplot(fig)

    st.dataframe(fault_counts)

# =====================================================
# Q10: Faults causing biggest energy drop
# =====================================================
with st.expander("📉 Q10: Which faults reduce energy the most?"):

    fault_energy = df[df['event'] != 'no fault'].groupby("event")["E-Today(KWH)"].mean().sort_values()

    fig, ax = plt.subplots(figsize=(14,4))
    sns.barplot(x=fault_energy.index, y=fault_energy.values, ax=ax)
    ax.set_title("Faults vs Energy Production")
    st.pyplot(fig)

    st.dataframe(fault_energy)

# =====================================================
# Q11: Fault rate on low-energy days
# =====================================================
with st.expander("⚠️ Q11: How many faults occur on low energy (<20 kWh) days?"):

    low_energy = df[df["E-Today(KWH)"] < 20]
    low_faults = low_energy[low_energy["event"] != "no fault"]

    percentage = (low_faults.shape[0] / low_energy.shape[0]) * 100

    st.write(f"🔧 Faults on low-energy days: `{low_faults.shape[0]}`")
    st.write(f"📌 Percentage: `{percentage:.2f}%`")

# =====================================================
# Q12: Months with most grid faults
# =====================================================
with st.expander("📛 Q12: Which months have most grid faults?"):

    grid_faults = df[df["event"] != "no event"]
    monthly_faults = grid_faults.groupby("months")["event"].count().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(14,4))
    sns.barplot(x=monthly_faults.index, y=monthly_faults.values, ax=ax)
    ax.set_title("Monthly Grid-Related Faults")
    st.pyplot(fig)

    st.dataframe(monthly_faults)

# =====================================================
# Q13: Lag effect of faults
# =====================================================
with st.expander("⏳ Q13: Does energy stay low for days after a fault?"):

    grid_fault_list = ['A0-Grid over voltage', 'A1-Grid under voltage', 'A2-Grid absent']
    df['GridFaultFlag'] = df['event'].apply(lambda x: 1 if x in grid_fault_list else 0)

    df["FaultYesterday"] = df["GridFaultFlag"].shift(1)
    df["Fault2DaysAgo"] = df["GridFaultFlag"].shift(2)
    df["Fault3DaysAgo"] = df["GridFaultFlag"].shift(3)

    st.write("**Avg Energy 1 day after fault:**", df[df["FaultYesterday"]==1]["E-Today(KWH)"].mean())
    st.write("**Avg Energy 2 days after fault:**", df[df["Fault2DaysAgo"]==1]["E-Today(KWH)"].mean())
    st.write("**Avg Energy 3 days after fault:**", df[df["Fault3DaysAgo"]==1]["E-Today(KWH)"].mean())

# =====================================================
# Q14: Highest-energy weeks
# =====================================================
with st.expander("📅 Q14: Which week has highest average energy?"):
    st.dataframe(
        df.groupby(["years","week"])
          .agg({"E-Today(KWH)":"mean"})
          .reset_index()
          .sort_values(by=["years","E-Today(KWH)"],ascending=[True,False])
          .drop_duplicates(subset="years")
    )

# =====================================================
# Q15: Monthly variability
# =====================================================
with st.expander("📉 Q15: Months with highest standard deviation?"):
    st.dataframe(df.groupby(["years","months"])["E-Today(KWH)"].std().sort_values(ascending=False))

# =====================================================
# Q16: Seasonal ANOVA
# =====================================================
with st.expander("📊 Q16: Does production differ between seasons (ANOVA test)?"):

    winter = df[df['Season']=='Winter']['E-Today(KWH)']
    spring = df[df['Season']=='Spring']['E-Today(KWH)']
    summer = df[df['Season']=='Summer']['E-Today(KWH)']
    autumn = df[df['Season']=='Autumn']['E-Today(KWH)']

    F, p = f_oneway(winter, spring, summer, autumn)

    st.write(f"F-statistic: `{F:.4f}`")
    st.write(f"p-value: `{p:.10f}`")

    if p < 0.05:
        st.success("✔ Significant difference between seasons!")
    else:
        st.info("No significant difference found.")

# =====================================================
# Q17: Fault frequency on high vs low production days
# =====================================================
with st.expander("📉 Q17: Do faults occur more on low-production days?"):

    df["fault_flag"] = df["event"].apply(lambda x:1 if x!="no fault" else 0)
    daily_stats = df.groupby("days").agg({"E-Today(KWH)":"sum","fault_flag":"sum"})
    median_val = daily_stats["E-Today(KWH)"].median()

    high_faults = daily_stats[daily_stats["E-Today(KWH)"] >= median_val]["fault_flag"].sum()
    low_faults = daily_stats[daily_stats["E-Today(KWH)"] < median_val]["fault_flag"].sum()

    st.write("📌 Faults on **high-production days:**", high_faults)
    st.write("📌 Faults on **low-production days:**", low_faults)
    st.info("⚠ Note: Fault category is imbalanced (many faults, fewer no-fault days).")

# =====================================================
# Q18: Fault probability per month
# =====================================================
with st.expander("🔥 Q18: Does probability of fault increase in hot months? (Summer Heat)"):

    total_monthly = df.groupby("months")["fault_flag"].count()
    fault_monthly = df.groupby("months")["fault_flag"].sum()
    season_name=df.groupby("months")["Season"].first()
    monthly_stats = pd.DataFrame({
        "total_events": total_monthly,
        "fault_events": fault_monthly,
        "Season":season_name
    })

    monthly_stats["fault_probability"] = monthly_stats["fault_events"] / monthly_stats["total_events"]

    st.dataframe(monthly_stats.sort_values(by="fault_probability",ascending=False))
