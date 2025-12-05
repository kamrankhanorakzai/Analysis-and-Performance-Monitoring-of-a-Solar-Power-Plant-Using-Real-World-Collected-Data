
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
# -------------------------------
# PAGE CONFIG & TITLE
# -------------------------------
st.set_page_config(layout="wide")

st.markdown("""
    <style>
        .main {
            background-color: #F5F7FA;
        }
        .block-title {
            font-size: 26px;
            font-weight: 600;
            padding: 10px 0;
            color: #1F4E79;
        }
        .sub-title {
            font-size: 20px;
            font-weight: 500;
            padding-top: 20px;
            color: #38598B;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='block-title'>📊 General Overview of Solar Dataset</div>", unsafe_allow_html=True)

# -------------------------------
# LOAD DATA
# -------------------------------
uploaded = st.file_uploader("Upload pkl File", type=["pkl"])

if uploaded:
        df = pd.read_pickle(uploaded)
        df = df.sort_values("date")


        # -------------------------------
        # DATASET PREVIEW
        # -------------------------------
        st.markdown("<div class='sub-title'>📄 Dataset Preview</div>", unsafe_allow_html=True)
        st.write("### `df.head()`")
        st.write(df.head())

        # -------------------------------
        # DATASET INFO
        # -------------------------------
        st.markdown("<div class='sub-title'>ℹ Dataset Info</div>", unsafe_allow_html=True)
        # create a string buffer
        buffer = io.StringIO()

        # write info into buffer
        df.info(buf=buffer)

        # get string from buffer
        info_str = buffer.getvalue()

        # display in Streamlit
        st.code(info_str)


        # -------------------------------
        # DESCRIPTIVE STATISTICS
        # -------------------------------
        st.markdown("<div class='sub-title'>📉 Descriptive Statistics</div>", unsafe_allow_html=True)
        st.write(df.describe())


        # -------------------------------
        # FEATURE ENGINEERING
        # -------------------------------
        st.markdown("<div class='sub-title'>🛠 Feature Engineering</div>", unsafe_allow_html=True)
        def get_season(month):
            if month in ["December", "January", "February"]:
                     return "Winter"
            elif month in ["March", "April", "May"]:
                     return "Spring"
            elif month in ["June", "July", "August"]:
                     return "Summer"
            else:
                     return "Autumn"

        df["months"] = df["date"].dt.month_name()
        df["days"] = df["date"].dt.day_name()
        df["years"] = df["date"].dt.year
        df["week"] = df["date"].dt.isocalendar().week
        df["DayOfWeek"] = df["date"].dt.weekday
        df["IsWeekend"] = df["date"].dt.weekday >= 5
        df["Quarter"] = df["date"].dt.quarter
        df["event"].fillna("no fault",inplace=True)
        df['Season'] = df['months'].apply(get_season)

        st.success("Date-based columns (months, week, quarter, weekdays, weekend flag) successfully added.")

        # -------------------------------
        # MISSING VALUES CHECK
        # -------------------------------
        st.markdown("<div class='sub-title'>🕳 Missing Values Overview</div>", unsafe_allow_html=True)

        st.write("### Missing Values before Cleaning:")
        st.write(df.isnull().sum())

        st.info("""
        📌 **Issue Identified**  
        There are missing values in `E-Today(KWH)`, especially between **2025-08-17 to 2025-08-27**  
        (~10% missing).  
        Since spline interpolation preserves smoothness, we use:
        ➡️ `interpolate(method="spline", order=2).ffill().bfill()`
        """)

        # -------------------------------
        # BEFORE/AFTER INTERPOLATION PLOT
        # -------------------------------
        st.markdown("<div class='sub-title'>📈 Missing Value Treatment Visualization</div>", 
                    unsafe_allow_html=True)

        # store original series
        df.set_index("date",inplace=True)
        # apply spline interpolation for plotting only (not yet modifying df)
        df_spline = df["E-Today(KWH)"].interpolate(method="spline", order=3).ffill().bfill()

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(df["E-Today(KWH)"], label="Original with NaN", alpha=0.5)
        ax.plot(df_spline, label="After Spline Interpolation", linewidth=2)
        ax.set_title("Solar kWh Time Series Before vs After Interpolation")
        ax.set_xlabel("Date")
        ax.set_ylabel("Energy (kWh)")
        ax.grid()
        ax.legend()
        st.pyplot(fig)

        # -------------------------------
        # APPLY INTERPOLATION TO DATAFRAME
        # -------------------------------
        series=df["E-Today(KWH)"].copy()
        df["E-Today(KWH)"] = df["E-Today(KWH)"].interpolate(method="spline", order=2).ffill().bfill()

        st.success("Spline interpolation applied. Missing values filled successfully.")

        # -------------------------------
        # HISTOGRAM COMPARISON
        # -------------------------------
        st.markdown("<div class='sub-title'>📊 Distribution Before vs After</div>", 
                    unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            ax1.hist(df_spline, bins=30, edgecolor="black", alpha=0.7)
            ax1.set_title("Histogram (Spline Filled)")
            ax1.set_xlabel("Daily Energy (kWh)")
            ax1.set_ylabel("Frequency")
            st.pyplot(fig1)

        with col2:
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.hist(series.dropna(), bins=30, edgecolor="black", alpha=0.7)
            ax2.set_title("Histogram (NaNs Dropped)")
            ax2.set_xlabel("Daily Energy (kWh)")
            st.pyplot(fig2)

        # -------------------------------
        # EVENT COLUMN DESCRIPTION
        # -------------------------------
        st.markdown("<div class='sub-title'>⚠ Event Column Explanation</div>", unsafe_allow_html=True)

        st.info("""
        ### Event Counts & Meanings

        #### 1️⃣ A0 – Grid Over Voltage *(411 times)*  
        - Grid voltage too high → inverter disconnects.

        #### 2️⃣ A2 – Grid Absent *(265 times)*  
        - No grid → inverter shuts down.

        #### 3️⃣ A1 – Grid Under Voltage *(242 times)*  
        - Grid voltage too low → reduced output or shutdown.

        #### 4️⃣ A4 – Grid Under Frequency *(226 times)*  
        - Frequency < 50Hz → inverter waits to stabilize.

        #### 5️⃣ **no fault** *(162 times)*  
        - System working normally.

        #### 6️⃣ A3 – Grid Over Frequency *(124 times)*  
        - Frequency too high → inverter disconnects.

        #### 7️⃣ A6 – Grid Abnormal *(13 times)*  
        - Combination of multiple issues (voltage + frequency).
        """)
        st.session_state["clean_data"] = df
        