import streamlit as st

# ------------ PAGE CONFIG ------------
st.set_page_config(
    page_title="Solar Analytics App",
    layout="wide",
)


# ------------ TITLE ------------
st.title("⚡ Solar Energy Analytics Dashboard")

# ------------ PROJECT SUMMARY ------------
st.subheader("🏠 Analysis and Performance Monitoring of a Residential Solar Power Plant")

st.markdown("""
**📌 Problem Statement**  
Residential solar systems generate daily energy, but most users cannot analyze long-term
performance, identify energy production patterns, or detect system faults. Without proper data
analysis, homeowners may face reduced efficiency, unnoticed errors, and financial losses.
This project analyzes real-world solar plant data to understand production trends, detect anomalies,
and provide actionable insights for better solar system performance.
""")

st.markdown("""
**🎯 Objectives**
- Analyze daily solar energy generation and identify trends across days, months, and seasons.
- Detect anomalies or unusual patterns using event/error logs.
- Build predictive or descriptive models to understand factors affecting energy generation.
- Visualize solar performance for easier interpretation and decision-making.
- Evaluate system errors/events and correlate them with drops in energy production.
""")
st.markdown("### 📂 Dataset Description")

st.markdown("**🌐 Data Source Description")

st.write("""
The dataset was web-scraped from the solar monitoring website **Cloud Inverter**.  
It contains **real-world daily energy production records** and **system event/error logs** from a residential solar plant.  

This data provides insights into:
- Energy generation patterns
- System performance trends
- Fault occurrences over time  

Enabling detailed analysis and predictive modeling for smarter solar management.
""")

st.markdown("""

**Dataset 1: Daily Energy Production Data (284 rows)**
- Features: `date` (datetime), `E-Today(KWH)` (integer)
- Purpose: Analyze daily, monthly, and yearly generation performance

**Dataset 2: Solar Plant Event/Error Log (1280 rows)**
- Features: `time`, `event` (error), `date`
- Purpose: Understand system faults, identify error patterns, and correlate with energy drops
""")


st.markdown("""
**✅ Expected Outcomes**
- Identify best and worst performing months/days
- Detect performance drops caused by system errors
- Trend insights related to weather and seasonal effects
- Create a data-driven monitoring dashboard
- Provide recommendations to improve solar power efficiency
""")
