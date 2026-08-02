# BEML Solar Plant Diagnostics Streamlit Dashboard
# Written in a simple student style with advanced plotting, sidebar uploader, and rule sliders

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
import io
import os

# Set page configuration
st.set_page_config(page_title="BEML Solar Plant Diagnostics", layout="wide")

# Dashboard Title
st.title("Solar PV Performance Diagnostics Dashboard")
st.markdown("Interactive operations and maintenance platform for BEML Kolar (400 kW) site.")

# --- STEP 1: DEFINE PREPROCESSING & RULE ENGINE HELPERS ---

def process_raw_excel(df_raw):
    """Reconstructs timestamps, scales registers, and resamples raw Modbus Excel data."""
    # Reconstruct true date-time from individual Modbus registers
    years = (df_raw["param_45"].fillna(0).astype(int) + 2000).astype(str)
    months = df_raw["param_44"].fillna(0).astype(int).astype(str).str.zfill(2)
    days = df_raw["param_43"].fillna(0).astype(int).astype(str).str.zfill(2)
    hours = df_raw["param_41"].fillna(0).astype(int).astype(str).str.zfill(2)
    minutes = df_raw["param_42"].fillna(0).astype(int).astype(str).str.zfill(2)

    date_strings = years + "-" + months + "-" + days + " " + hours + ":" + minutes + ":00"
    df_raw["device_time"] = pd.to_datetime(date_strings, errors="coerce")
    df_raw["device_time"] = df_raw["device_time"].fillna(pd.to_datetime(df_raw["recieved_at"]))

    # Process weather sensor data
    df_sens = df_raw[(df_raw["pkt_type"] == 5) | (df_raw["sub_deviceid"] == 5)].copy()
    df_sens["module_temperature"] = pd.to_numeric(df_sens["param_01"], errors="coerce") * 0.0265596
    df_sens["irradiance_wm2"] = pd.to_numeric(df_sens["param_02"], errors="coerce") * 0.5319149

    df_sens.loc[(df_sens["irradiance_wm2"] < 0.0) | (df_sens["irradiance_wm2"] > 1400.0), "irradiance_wm2"] = np.nan
    df_sens.loc[(df_sens["module_temperature"] < -20.0) | (df_sens["module_temperature"] > 100.0), "module_temperature"] = np.nan

    df_sens.drop_duplicates(subset=["device_time"], inplace=True)
    df_sens.set_index("device_time", inplace=True)
    df_sens.sort_index(inplace=True)

    df_sens_10m = df_sens.resample("10min").agg({
        "irradiance_wm2": "mean",
        "module_temperature": "mean"
    }).interpolate(method="time", limit=3)

    # Modbus registers gain multipliers map
    inverter_gain_map = {
        "param_01": 0.1, "param_02": 0.1, "param_03": 0.1,
        "param_04": 0.1, "param_05": 0.1, "param_06": 0.1,
        "param_07": 0.1, "param_08": 0.1, "param_09": 0.1,
        "param_10": 0.0001, "param_11": 0.1, "param_12": 0.1, "param_13": 0.1,
        "param_14": 0.01, "param_15": 0.001,
        "param_16": 0.1, "param_17": 0.1,
        "param_18": 0.1, "param_19": 0.1,
        "param_20": 0.1, "param_21": 0.1,
        "param_22": 0.1, "param_23": 0.1,
        "param_24": 0.1, "param_25": 0.1,
        "param_26": 0.1, "param_27": 0.1,
        "param_28": 0.1, "param_29": 0.1,
        "param_37": 0.1, "param_38": 0.1,
        "param_39": 0.1, "param_40": 0.1
    }

    inverter_name_map = {
        "param_01": "grid_voltage_ab", "param_02": "grid_voltage_bc", "param_03": "grid_voltage_ac",
        "param_04": "grid_voltage_a", "param_05": "grid_voltage_b", "param_06": "grid_voltage_c",
        "param_07": "grid_current_a", "param_08": "grid_current_b", "param_09": "grid_current_c",
        "param_10": "ac_power_kw",
        "param_11": "heat_sink_temp", "param_12": "igbt_temp", "param_13": "inductance_temp",
        "param_14": "grid_frequency_hz", "param_15": "power_factor",
        "param_16": "dc_voltage_1", "param_17": "dc_current_1",
        "param_18": "dc_voltage_2", "param_19": "dc_current_2",
        "param_20": "dc_voltage_3", "param_21": "dc_current_3",
        "param_22": "dc_voltage_4", "param_23": "dc_current_4",
        "param_24": "dc_voltage_5", "param_25": "dc_current_5",
        "param_26": "dc_voltage_6", "param_27": "dc_current_6",
        "param_28": "today_generation_kwh", "param_29": "total_generation_kwh",
        "param_37": "dc_voltage_7", "param_38": "dc_current_7",
        "param_39": "dc_voltage_8", "param_40": "dc_current_8"
    }

    resample_config = {
        "ac_power_kw": "mean",
        "grid_voltage_a": "mean", "grid_voltage_b": "mean", "grid_voltage_c": "mean",
        "grid_voltage_ab": "mean", "grid_voltage_bc": "mean", "grid_voltage_ac": "mean",
        "grid_current_a": "mean", "grid_current_b": "mean", "grid_current_c": "mean",
        "heat_sink_temp": "mean", "igbt_temp": "mean", "inductance_temp": "mean",
        "grid_frequency_hz": "mean", "power_factor": "mean",
        "total_generation_kwh": "last"
    }
    for k in range(1, 9):
        resample_config[f"dc_voltage_{k}"] = "mean"
        resample_config[f"dc_current_{k}"] = "mean"

    df_inv_list = []
    for inv_id in [1, 2, 3, 4]:
        df_inv = df_raw[(df_raw["sub_deviceid"] == inv_id) & (df_raw["pkt_type"] != 5)].copy()
        
        for col, gain in inverter_gain_map.items():
            if col in df_inv.columns:
                df_inv[col] = pd.to_numeric(df_inv[col], errors="coerce") * gain
        df_inv.rename(columns=inverter_name_map, inplace=True)
        
        df_inv.loc[(df_inv["ac_power_kw"] < -1.0) | (df_inv["ac_power_kw"] > 120.0), "ac_power_kw"] = np.nan
        df_inv.loc[(df_inv["grid_frequency_hz"] < 45.0) | (df_inv["grid_frequency_hz"] > 55.0), "grid_frequency_hz"] = np.nan
        df_inv.loc[(df_inv["power_factor"] < -1.0) | (df_inv["power_factor"] > 1.0), "power_factor"] = np.nan
        
        df_inv.drop_duplicates(subset=["device_time"], inplace=True)
        df_inv.set_index("device_time", inplace=True)
        df_inv.sort_index(inplace=True)
        
        active_resample_map = {col: agg for col, agg in resample_config.items() if col in df_inv.columns}
        df_inv_10m = df_inv.resample("10min").agg(active_resample_map)
        df_inv_10m = df_inv_10m.interpolate(method="time", limit=2)
        
        df_joined = df_inv_10m.join(df_sens_10m, how="left")
        df_joined.fillna(0.0, inplace=True)
        df_joined = df_joined.reset_index()
        df_joined["inverter_id"] = inv_id
        
        df_inv_list.append(df_joined)

    df_all = pd.concat(df_inv_list, ignore_index=True)
    return df_all.sort_values(["inverter_id", "device_time"]).reset_index(drop=True)

def run_feature_engineering_and_labeling(data, rules_config=None):
    """Calculates all physics-based features and generates rule-based labels."""
    df = data.copy()
    
    if rules_config is None:
        rules_config = {
            "offline_irr": 50.0,
            "offline_ac": 0.5,
            "string_imbalance": 0.25,
            "string_dev": 0.30,
            "string_irr": 300.0,
            "string_ratio": 0.005,
            "inv_temp": 163.0,
            "inv_residual": -0.05,
            "grid_vuf": 1.5,
            "grid_vmin": 200.0,
            "grid_vmax": 265.0,
            "grid_freq": 0.3,
            "standby_irr": 50.0
        }
        
    # Filter daytime
    df = df[(df["irradiance_wm2"] > 20) | (df["ac_power_kw"] > 0.5)].reset_index(drop=True)
    df["device_time"] = pd.to_datetime(df["device_time"])
    
    # Time parameters
    df["hour"] = df["device_time"].dt.hour
    df["month"] = df["device_time"].dt.month
    df["month_str"] = df["device_time"].dt.strftime("%Y-%m")
    
    # Cyclic time mapping
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Thermal parameters
    df["thermal_stress"] = df["igbt_temp"] / 178.4
    df["temp_diff"] = df["igbt_temp"] - df["module_temperature"]
    
    # DC Current imbalance metrics
    dc_curr_cols = ["dc_current_1", "dc_current_2", "dc_current_3", "dc_current_4", "dc_current_5", "dc_current_6"]
    df["dc_curr_mean"] = df[dc_curr_cols].mean(axis=1)
    df["dc_curr_std"] = df[dc_curr_cols].std(axis=1)
    df["dc_imbalance"] = df["dc_curr_std"] / (df["dc_curr_mean"] + 0.01)
    
    for i in range(1, 7):
        df[f"dc_dev_{i}"] = (df[f"dc_current_{i}"] - df["dc_curr_mean"]) / (df["dc_curr_mean"] + 0.01)
    dev_cols = [f"dc_dev_{i}" for i in range(1, 7)]
    df["max_string_dev"] = df[dev_cols].abs().max(axis=1)
    
    # DC Voltage features
    dc_volt_cols = ["dc_voltage_1", "dc_voltage_2", "dc_voltage_3", "dc_voltage_4", "dc_voltage_5", "dc_voltage_6"]
    df["dc_volt_mean"] = df[dc_volt_cols].mean(axis=1)
    df["dc_volt_std"] = df[dc_volt_cols].std(axis=1)
    df["dc_volt_imbalance_ratio"] = df["dc_volt_std"] / (df["dc_volt_mean"] + 0.01)

    # AC grid voltage metrics
    grid_volt_cols = ["grid_voltage_a", "grid_voltage_b", "grid_voltage_c"]
    df["grid_v_mean"] = df[grid_volt_cols].mean(axis=1)
    df["VUF"] = (df[grid_volt_cols].sub(df["grid_v_mean"], axis=0).abs().max(axis=1) / df["grid_v_mean"]) * 100
    
    df["grid_v_diff_ab"] = (df["grid_voltage_a"] - df["grid_voltage_b"]).abs()
    df["grid_v_diff_bc"] = (df["grid_voltage_b"] - df["grid_voltage_c"]).abs()
    df["grid_v_diff_ca"] = (df["grid_voltage_c"] - df["grid_voltage_a"]).abs()

    # Expected power regression baseline
    split_date = pd.Timestamp("2026-03-01")
    before_split = df["device_time"] < split_date
    
    is_offline = (df["ac_power_kw"] <= rules_config["offline_ac"])
    is_string_fault = (df["dc_imbalance"] > rules_config["string_imbalance"]) & (df["max_string_dev"] > rules_config["string_dev"])
    is_grid_fault = (df["VUF"] > rules_config["grid_vuf"]) | (df["grid_v_mean"] < rules_config["grid_vmin"])
    
    healthy_mask = (
        (df["irradiance_wm2"] > rules_config["string_irr"])
        & (df["module_temperature"] < 50)
        & before_split
        & (~is_offline)
        & (~is_string_fault)
        & (~is_grid_fault)
        & (df["ac_power_kw"] > 10.0)
    )
    
    train_lr = df[healthy_mask].copy()
    if not train_lr.empty:
        X_lr = train_lr[["irradiance_wm2", "module_temperature"]].copy()
        X_lr["irradiance_sq"] = X_lr["irradiance_wm2"] ** 2
        y_lr = train_lr["ac_power_kw"]
        
        lr_model = LinearRegression()
        lr_model.fit(X_lr, y_lr)
        
        all_X_lr = df[["irradiance_wm2", "module_temperature"]].copy()
        all_X_lr["irradiance_sq"] = all_X_lr["irradiance_wm2"] ** 2
        df["expected_power"] = lr_model.predict(all_X_lr).clip(min=0)
    else:
        df["expected_power"] = df["ac_power_kw"]
        
    df["power_residual"] = (df["ac_power_kw"] - df["expected_power"]) / 100.0
    
    # Calculate energy terms in kWh
    df["actual_energy_kwh"] = df["ac_power_kw"] * (10 / 60)
    df["expected_energy_kwh"] = df["expected_power"] * (10 / 60)
    df["lost_energy_kwh"] = np.maximum(0, df["expected_energy_kwh"] - df["actual_energy_kwh"])

    # Physical fault labeling
    df["label"] = 0
    
    # F4: Offline
    offline_condition = (df["ac_power_kw"] <= rules_config["offline_ac"]) & (df["irradiance_wm2"] > rules_config["offline_irr"])
    df.loc[offline_condition, "label"] = 4
    
    # F3: Grid Fault
    grid_condition = (
        (df["ac_power_kw"] > rules_config["offline_ac"])
        & (df["grid_v_mean"] > 180)
        & (
            (df["VUF"] > rules_config["grid_vuf"])
            | ((df["grid_frequency_hz"] - 50).abs() > rules_config["grid_freq"])
            | (df[grid_volt_cols] > rules_config["grid_vmax"]).any(axis=1)
            | (df[grid_volt_cols] < rules_config["grid_vmin"]).any(axis=1)
        )
    )
    df.loc[grid_condition & (df["label"] == 0), "label"] = 3
    
    # F1: DC String Fault
    string_condition = (
        (df["ac_power_kw"] > rules_config["offline_ac"])
        & (df["irradiance_wm2"] > rules_config["string_irr"])
        & (
            ((df["dc_imbalance"] > rules_config["string_imbalance"]) & (df["max_string_dev"] > rules_config["string_dev"]))
            | (df["dc_curr_mean"] < rules_config["string_ratio"] * df["irradiance_wm2"])
        )
    )
    df.loc[string_condition & (df["label"] == 0), "label"] = 1
    
    # F2: Inverter Underperformance
    inverter_condition = (
        (df["ac_power_kw"] > rules_config["offline_ac"])
        & (df["igbt_temp"] > rules_config["inv_temp"])
        & (df["power_residual"] < rules_config["inv_residual"])
    )
    df.loc[inverter_condition & (df["label"] == 0), "label"] = 2
    
    # Enforce standby filter
    df.loc[df["irradiance_wm2"] <= rules_config["standby_irr"], "label"] = 0
    
    # Map label names
    label_map = {0: "Normal", 1: "DC String Fault", 2: "Inverter Fault", 3: "Grid Fault", 4: "Inverter Offline"}
    df["fault_type"] = df["label"].map(label_map)
    df["date"] = df["device_time"].dt.date
    
    return df

# --- STEP 2: EVENT BUILDER AND AGGREGATION CODES ---

def build_fault_events(df):
    """Groups consecutive row-level fault alerts into discrete fault events."""
    data = df[df["label"] != 0].copy()
    if data.empty:
        return pd.DataFrame()
        
    data = data.sort_values(["inverter_id", "device_time"]).reset_index(drop=True)
    data["time_gap_min"] = data.groupby("inverter_id")["device_time"].diff().dt.total_seconds() / 60
    data["label_changed"] = data["label"].ne(data["label"].shift())
    
    # 20-minute gap threshold to split events
    data["new_event"] = (data["time_gap_min"].isna()) | (data["time_gap_min"] > 20) | (data["label_changed"])
    data["event_id"] = data["new_event"].cumsum()
    
    label_to_name = {
        1: "DC String Fault (F1)",
        2: "Inverter Underperformance (F2)",
        3: "Grid Fault (F3)",
        4: "Inverter Offline (F4)"
    }
    
    severity_map = {
        1: "🟠 Warning",
        2: "🟠 Warning",
        3: "🔵 Notice",
        4: "🔴 Critical"
    }
    
    events = []
    for ev_id, ev in data.groupby("event_id"):
        start = ev["device_time"].min()
        end = ev["device_time"].max() + pd.Timedelta(minutes=10)
        duration_hr = len(ev) * (10 / 60)
        actual_p = ev["ac_power_kw"].mean()
        expected_p = ev["expected_power"].mean()
        
        row_losses = (ev["expected_power"] - ev["ac_power_kw"]).clip(lower=0)
        loss_kwh = row_losses.sum() * (10 / 60)
        
        events.append({
            "Event ID": ev_id,
            "Month": start.strftime("%Y-%m"),
            "Inverter ID": ev["inverter_id"].iloc[0],
            "Fault Type": label_to_name.get(ev["label"].iloc[0], "Unknown"),
            "Severity": severity_map.get(ev["label"].iloc[0], "🔵 Notice"),
            "Start Time": start,
            "End Time": end,
            "Duration (hrs)": round(duration_hr, 2),
            "Energy Lost (kWh)": round(loss_kwh, 2),
            "Avg Power Deficit (kW)": round(expected_p - actual_p, 2),
            "label_code": ev["label"].iloc[0]
        })
        
    return pd.DataFrame(events)

def compute_daily_aggregation_table(df):
    """Computes daily fault aggregation summary table."""
    events_df = build_fault_events(df)
    if events_df.empty:
        return pd.DataFrame()
    
    events_df = events_df.copy()
    events_df["Date"] = events_df["Start Time"].dt.date
    
    daily = events_df.groupby("Date").agg(
        total_events=("Event ID", "count"),
        total_duration=("Duration (hrs)", "sum"),
        total_loss=("Energy Lost (kWh)", "sum")
    ).reset_index()
    
    worst_invs = []
    faults_logged = []
    for d in daily["Date"]:
        day_evs = events_df[events_df["Date"] == d]
        worst_inv = day_evs.groupby("Inverter ID")["Duration (hrs)"].sum().idxmax()
        worst_invs.append(f"Inverter {worst_inv}")
        
        unique_faults = sorted(day_evs["Fault Type"].unique())
        faults_logged.append(", ".join(unique_faults))
        
    daily["Worst Inverter"] = worst_invs
    daily["Logged Fault Types"] = faults_logged
    
    daily["Total Duration (hrs)"] = daily["total_duration"].round(2)
    daily["Total Energy Lost (kWh)"] = daily["total_loss"].round(2)
    
    daily.drop(columns=["total_duration", "total_loss"], inplace=True)
    daily.rename(columns={
        "total_events": "Total Events Count",
    }, inplace=True)
    
    return daily.sort_values("Date", ascending=False).reset_index(drop=True)

# --- STEP 3: STREAMLIT CACHED PIPELINE ---

@st.cache_data
def get_processed_dataframe(file_bytes, filename, rules_config):
    """Loads and processes the uploaded dataset (Excel or CSV). Results are cached for speed."""
    if filename.endswith(".xlsx"):
        df_raw = pd.read_excel(io.BytesIO(file_bytes), sheet_name="MicroSystem_1125_0426", header=22)
        df_pre = process_raw_excel(df_raw)
        df_labeled = run_feature_engineering_and_labeling(df_pre, rules_config=rules_config)
    else: # .csv
        df_pre = pd.read_csv(io.BytesIO(file_bytes))
        df_labeled = run_feature_engineering_and_labeling(df_pre, rules_config=rules_config)
    return df_labeled

# --- STEP 4: STREAMLIT SIDEBAR CONTROLS & FILE UPLOADER ---

st.sidebar.header("📁 Data Loader")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload Telemetry File (Raw Excel or CSV)", type=["xlsx", "csv"])

# Sliders inside expander
with st.sidebar.expander("🛠️ Customize Fault Thresholds", expanded=False):
    offline_irr = st.slider("Offline Irradiance Limit (W/m²)", 20.0, 150.0, 50.0, 5.0)
    offline_ac = st.slider("Offline AC Power Limit (kW)", 0.1, 5.0, 0.5, 0.1)
    string_imbalance = st.slider("DC String Imbalance (std/mean)", 0.10, 0.50, 0.25, 0.01)
    string_dev = st.slider("Max String Current Deviation", 0.10, 0.60, 0.30, 0.01)
    string_irr = st.slider("String Fault Active Irradiance (W/m²)", 100.0, 500.0, 300.0, 10.0)
    string_ratio = st.slider("Current-to-Irradiance Ratio", 0.001, 0.010, 0.005, 0.001)
    inv_temp = st.slider("IGBT Temp Limit (°C)", 100.0, 180.0, 163.0, 1.0)
    inv_residual = st.slider("Expected Power Residual Limit", -0.20, -0.01, -0.05, 0.01)
    grid_vuf = st.slider("Grid Voltage Unbalance (VUF %)", 0.5, 5.0, 1.5, 0.1)
    grid_vmin = st.slider("Min Grid Voltage (V)", 150.0, 220.0, 200.0, 5.0)
    grid_vmax = st.slider("Max Grid Voltage (V)", 240.0, 280.0, 265.0, 5.0)
    grid_freq = st.slider("Grid Frequency Delta (Hz)", 0.1, 1.0, 0.3, 0.05)
    standby_irr = st.slider("Standby Suppression Irradiance (W/m²)", 10.0, 100.0, 50.0, 5.0)

rules_config = {
    "offline_irr": offline_irr,
    "offline_ac": offline_ac,
    "string_imbalance": string_imbalance,
    "string_dev": string_dev,
    "string_irr": string_irr,
    "string_ratio": string_ratio,
    "inv_temp": inv_temp,
    "inv_residual": inv_residual,
    "grid_vuf": grid_vuf,
    "grid_vmin": grid_vmin,
    "grid_vmax": grid_vmax,
    "grid_freq": grid_freq,
    "standby_irr": standby_irr
}

# --- STEP 5: CONDITIONAL DATASET INGESTION (NO preloading by default) ---

df_loaded = None

if uploaded_file is not None:
    try:
        # Read uploaded file bytes for caching
        file_bytes = uploaded_file.getvalue()
        filename = uploaded_file.name
        
        # Load and process data (using the streamlit cache wrapper)
        df_loaded = get_processed_dataframe(file_bytes, filename, rules_config)
        st.sidebar.success("File loaded and processed successfully!")
    except Exception as e:
        st.error(f"Error processing uploaded file: {e}")
        df_loaded = None

# --- STEP 6: RENDER TABS AND CHARTS ---

if df_loaded is None:
    st.warning("⚠️ Please upload your solar plant SCADA telemetry file (raw Excel `.xlsx` or preprocessed CSV `.csv`) in the sidebar to start the diagnostics platform.")
else:
    # Sidebar Filters
    st.sidebar.header("Global Filters")
    min_date = df_loaded["device_time"].min().date()
    max_date = df_loaded["device_time"].max().date()
    start_date, end_date = st.sidebar.date_input("Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

    # 1. Month filter multiselect
    all_months = sorted(df_loaded["month_str"].unique())
    selected_months = st.sidebar.multiselect("Select Months", options=all_months, default=all_months)

    # 2. Inverter filter multiselect
    selected_inverters = st.sidebar.multiselect("Select Inverters", options=[1, 2, 3, 4], default=[1, 2, 3, 4])

    # Filter dataframe by all sidebar filters
    df_filtered = df_loaded[
        (df_loaded["device_time"].dt.date >= start_date)
        & (df_loaded["device_time"].dt.date <= end_date)
        & (df_loaded["month_str"].isin(selected_months))
        & (df_loaded["inverter_id"].isin(selected_inverters))
    ].copy()

    # Create Tabs
    tab1, tab2 = st.tabs(["📊 Visual Analytics", "🔍 Diagnostics Dashboard"])

    # --- TAB 1: VISUAL ANALYTICS ---
    with tab1:
        st.header("Overall Plant KPIs & Performance Analysis")

        # Calculate KPIs
        total_actual_gen = df_filtered["actual_energy_kwh"].sum()
        total_expected_gen = df_filtered["expected_energy_kwh"].sum()
        
        # Lost energy is calculated for rows that are not Normal (label != 0)
        total_lost_energy = df_filtered[df_filtered["label"] != 0]["lost_energy_kwh"].sum()
        loss_percentage = (total_lost_energy / total_expected_gen * 100) if total_expected_gen > 0 else 0.0

        # KPI Metrics Cards
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        kpi_col1.metric("Actual Generation", f"{total_actual_gen:,.2f} kWh")
        kpi_col2.metric("Expected Generation (Healthy)", f"{total_expected_gen:,.2f} kWh")
        kpi_col3.metric("Energy Lost to Faults", f"{total_lost_energy:,.2f} kWh")
        kpi_col4.metric("Loss Percentage", f"{loss_percentage:.2f} %")

        st.markdown("---")

        # Split charts into two columns
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.subheader("Monthly Generation vs. Energy Loss")
            # Resample by month
            monthly_data = df_filtered.groupby("month_str").agg({
                "actual_energy_kwh": "sum",
                "lost_energy_kwh": "sum"
            }).reset_index()

            # Create bar plot
            fig_monthly = go.Figure()
            fig_monthly.add_trace(go.Bar(
                x=monthly_data["month_str"], y=monthly_data["actual_energy_kwh"],
                name="Actual Generation (kWh)", marker_color="royalblue"
            ))
            fig_monthly.add_trace(go.Bar(
                x=monthly_data["month_str"], y=monthly_data["lost_energy_kwh"],
                name="Lost Energy (kWh)", marker_color="tomato"
            ))
            fig_monthly.update_layout(barmode="stack", xaxis_title="Month", yaxis_title="Energy (kWh)", legend_title="Legend")
            st.plotly_chart(fig_monthly, use_container_width=True)

        with chart_col2:
            st.subheader("Fault Type Burden Duration (Hours)")
            # Calculate total hours spent in each fault state
            fault_hours = df_filtered[df_filtered["label"] != 0].groupby("fault_type").size() * (10 / 60)
            fault_hours = fault_hours.reset_index(name="hours")

            if not fault_hours.empty:
                fig_pie = px.pie(fault_hours, values="hours", names="fault_type", color_discrete_sequence=px.colors.qualitative.Pastel)
                fig_pie.update_traces(textposition="inside", textinfo="percent+label")
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No faults recorded in the selected date range.")

        st.markdown("---")

        # Inverter-Wise Fault Burden Breakdown
        st.subheader("Inverter-Wise Fault Duration Breakdown (Hours)")
        inverter_fault = df_filtered[df_filtered["label"] != 0].groupby(["inverter_id", "fault_type"]).size() * (10 / 60)
        inverter_fault = inverter_fault.reset_index(name="hours")
        inverter_fault["inverter_id"] = "Inverter " + inverter_fault["inverter_id"].astype(str)

        if not inverter_fault.empty:
            fig_inv_bar = px.bar(
                inverter_fault, x="inverter_id", y="hours", color="fault_type",
                barmode="group", labels={"hours": "Fault Duration (Hours)", "inverter_id": "Inverter"},
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig_inv_bar, use_container_width=True)
        else:
            st.info("No inverter faults recorded in the selected date range.")


    # --- TAB 2: DIAGNOSTICS DASHBOARD ---
    with tab2:
        st.subheader("🔍 Operations Diagnostics & Root-Cause Logs")
        
        # Build event dataframe
        events_df = build_fault_events(df_filtered)
        
        # Compute daily aggregated summary table
        daily_df = compute_daily_aggregation_table(df_filtered)
        
        if daily_df.empty:
            st.success("No fault events found matching current filter configuration!")
        else:
            st.markdown("### 📅 Daily Aggregated Fault Summary")
            st.markdown("Click on column headers to sort and identify worst-performing days.")
            st.dataframe(daily_df, use_container_width=True, hide_index=True)
            
            # Date selector dropdown populated only with dates containing active faults
            active_dates = sorted(daily_df["Date"].unique(), reverse=True)
            date_options = []
            for d in active_dates:
                day_row = daily_df[daily_df["Date"] == d].iloc[0]
                faults = day_row["Logged Fault Types"]
                date_options.append(f"{d} | Faults: {faults}")
                
            selected_date_str = st.selectbox("Select Date for Detailed Diagnostics", date_options)
            selected_date = pd.to_datetime(selected_date_str.split(" | ")[0]).date()
            
            # Filter day events and telemetry
            day_events = events_df[events_df["Start Time"].dt.date == selected_date]
            day_telemetry = df_filtered[df_filtered["device_time"].dt.date == selected_date].sort_values("device_time")
            active_inverters = sorted(day_telemetry["inverter_id"].unique())
            
            # Daily KPI Cards
            st.write("")
            col_k1, col_k2, col_k3 = st.columns(3)
            day_hours = day_events["Duration (hrs)"].sum()
            day_loss = day_events["Energy Lost (kWh)"].sum()
            worst_inv = day_events.groupby("Inverter ID")["Duration (hrs)"].sum().idxmax()
            
            col_k1.metric("Daily Fault Duration", f"{day_hours:.2f} hrs")
            col_k2.metric("Daily Energy Lost", f"{day_loss:.2f} kWh")
            col_k3.metric("Worst-Performing Inverter", f"Inverter {worst_inv}")
            
            # Display Simple Chronological Logs of Detected Events
            st.subheader(f"📝 Flagged Fault Events for {selected_date}")
            for idx, ev in day_events.sort_values("Start Time").iterrows():
                inv_id = ev["Inverter ID"]
                lbl_code = ev["label_code"]
                start_str = ev["Start Time"].strftime("%H:%M")
                end_str = ev["End Time"].strftime("%H:%M")
                dur_hr = ev["Duration (hrs)"]
                loss_kwh = ev["Energy Lost (kWh)"]
                fault_name = ev["Fault Type"]
                severity = ev["Severity"]
                
                # Simple structured summary without verbose reasoning
                item_text = f"**{severity}** | **{start_str} - {end_str}** | **Inverter {inv_id}** | **{fault_name}** | Duration: **{dur_hr:.2f} hrs** | Energy Lost: **{loss_kwh:.2f} kWh**"
                st.info(item_text)

            # Helper to merge overlapping intervals for a given fault code on a single day
            def merge_fault_intervals(day_events, label_code):
                ev_subset = day_events[day_events["label_code"] == label_code].sort_values("Start Time")
                if ev_subset.empty:
                    return []
                merged = []
                for _, row in ev_subset.iterrows():
                    s = row["Start Time"]
                    e = row["End Time"]
                    inv = row["Inverter ID"]
                    if not merged:
                        merged.append([s, e, {inv}])
                    else:
                        last_s, last_e, last_invs = merged[-1]
                        if s <= last_e: # overlap
                            merged[-1][1] = max(last_e, e)
                            merged[-1][2].add(inv)
                        else:
                            merged.append([s, e, {inv}])
                return merged

            st.write("---")
            st.subheader("📈 Telemetry Visualizations")
            plot_mode = st.selectbox("Select Plotting Mode", ["All Faults (Full Day Timelines)", "Individual Event Zoom-In"])
            
            if plot_mode == "All Faults (Full Day Timelines)":
                # Plot 1: Inverter AC Power & Solar Irradiance (Offline F4 Timeline)
                st.markdown("#### 1. Inverter AC Power & Solar Irradiance (Offline F4)")
                fig_f4_day = make_subplots(specs=[[{"secondary_y": True}]])
                for inv in active_inverters:
                    inv_day_tele = day_telemetry[day_telemetry["inverter_id"] == inv]
                    if not inv_day_tele.empty:
                        fig_f4_day.add_trace(
                            go.Scatter(x=inv_day_tele["device_time"], y=inv_day_tele["ac_power_kw"], mode="lines", name=f"Inv {inv} Power"),
                            secondary_y=False
                        )
                # Add irradiance
                fig_f4_day.add_trace(
                    go.Scatter(x=day_telemetry["device_time"].unique(), y=day_telemetry.groupby("device_time")["irradiance_wm2"].mean(), mode="lines", name="Irradiance", line=dict(color="#ffd700", dash="dash")),
                    secondary_y=True
                )
                
                # Highlight Offline F4 windows
                merged_f4 = merge_fault_intervals(day_events, 4)
                for start_t, end_t, invs in merged_f4:
                    invs_label = ", ".join(f"Inv {i}" for i in sorted(invs))
                    fig_f4_day.add_vrect(
                        x0=start_t, x1=end_t,
                        fillcolor="rgba(128, 128, 128, 0.15)", layer="below", line_width=0,
                        annotation=dict(text=f"{invs_label} Offline", textangle=0, y=1.02, yanchor="bottom", font=dict(size=10, color="gray"), showarrow=False)
                    )
                    
                fig_f4_day.update_layout(xaxis_title="Time of Day", hovermode="x unified", height=320)
                fig_f4_day.update_yaxes(title_text="AC Power (kW)", secondary_y=False)
                fig_f4_day.update_yaxes(title_text="Irradiance (W/m²)", secondary_y=True)
                st.plotly_chart(fig_f4_day, use_container_width=True)
                
                # Plot 2: DC String Currents (String Fault F1 Timeline)
                st.write("")
                st.markdown("#### 2. DC String Currents (String Fault F1)")
                sel_inv_str = st.selectbox("Select Inverter to View String Currents Timeline", active_inverters)
                inv_day_str = day_telemetry[day_telemetry["inverter_id"] == sel_inv_str]
                if not inv_day_str.empty:
                    fig_f1_day = go.Figure()
                    for i in range(1, 7):
                        fig_f1_day.add_trace(
                            go.Scatter(x=inv_day_str["device_time"], y=inv_day_str[f"dc_current_{i}"], mode="lines", name=f"String {i}")
                        )
                        
                    # Highlight String Faults F1
                    merged_f1 = merge_fault_intervals(day_events[day_events["Inverter ID"] == sel_inv_str], 1)
                    for start_t, end_t, _ in merged_f1:
                        fig_f1_day.add_vrect(
                            x0=start_t, x1=end_t,
                            fillcolor="rgba(255, 140, 0, 0.12)", layer="below", line_width=0,
                            annotation=dict(text="String Fault", textangle=0, y=1.02, yanchor="bottom", font=dict(size=10, color="darkorange"), showarrow=False)
                        )
                    fig_f1_day.update_layout(title=f"Inverter {sel_inv_str} String Currents", xaxis_title="Time of Day", yaxis_title="Current (A)", hovermode="x unified", height=320)
                    st.plotly_chart(fig_f1_day, use_container_width=True)

                # Plot 3: Grid Phase Voltages (Grid Fault F3 Timeline)
                st.write("")
                st.markdown("#### 3. Grid Phase Voltages (Grid Fault F3)")
                fig_f3_day = go.Figure()
                for inv in active_inverters:
                    inv_day_tele = day_telemetry[day_telemetry["inverter_id"] == inv]
                    if not inv_day_tele.empty:
                        for phase in ["a", "b", "c"]:
                            fig_f3_day.add_trace(
                                go.Scatter(x=inv_day_tele["device_time"], y=inv_day_tele[f"grid_voltage_{phase}"], mode="lines", name=f"Inv {inv} Phase {phase.upper()}")
                            )
                
                # Highlight Grid Faults F3
                merged_f3 = merge_fault_intervals(day_events, 3)
                for start_t, end_t, invs in merged_f3:
                    invs_label = ", ".join(f"Inv {i}" for i in sorted(invs))
                    fig_f3_day.add_vrect(
                        x0=start_t, x1=end_t,
                        fillcolor="rgba(128, 128, 128, 0.15)", layer="below", line_width=0,
                        annotation=dict(text=f"{invs_label} Grid Fault", textangle=0, y=1.02, yanchor="bottom", font=dict(size=10, color="gray"), showarrow=False)
                    )
                fig_f3_day.update_layout(xaxis_title="Time of Day", yaxis_title="Voltage (V)", hovermode="x unified", height=320)
                st.plotly_chart(fig_f3_day, use_container_width=True)

                # Plot 4: Inverter Performance (Underperformance F2 Timeline)
                st.write("")
                st.markdown("#### 4. Inverter Actual vs Expected AC Power (Underperformance F2)")
                for inv in active_inverters:
                    inv_day_perf = day_telemetry[day_telemetry["inverter_id"] == inv]
                    if not inv_day_perf.empty:
                        st.markdown(f"##### Inverter {inv} Performance")
                        fig_inv_perf = make_subplots(specs=[[{"secondary_y": True}]])
                        
                        # Actual AC
                        fig_inv_perf.add_trace(
                            go.Scatter(x=inv_day_perf["device_time"], y=inv_day_perf["ac_power_kw"], mode="lines", name=f"Inv {inv} Actual AC", line=dict(color="#1f77b4")),
                            secondary_y=False
                        )
                        # Expected AC
                        fig_inv_perf.add_trace(
                            go.Scatter(x=inv_day_perf["device_time"], y=inv_day_perf["expected_power"], mode="lines", name=f"Inv {inv} Expected AC", line=dict(color="#2ca02c", dash="dash")),
                            secondary_y=False
                        )
                        # Irradiance
                        fig_inv_perf.add_trace(
                            go.Scatter(x=inv_day_perf["device_time"], y=inv_day_perf["irradiance_wm2"], mode="lines", name="Solar Irradiance", line=dict(color="#ffd700", dash="dot")),
                            secondary_y=True
                        )
                        
                        # Highlight Inverter Underperformance F2
                        merged_f2 = merge_fault_intervals(day_events[day_events["Inverter ID"] == inv], 2)
                        for start_t, end_t, _ in merged_f2:
                            fig_inv_perf.add_vrect(
                                x0=start_t, x1=end_t,
                                fillcolor="rgba(128, 128, 128, 0.15)", layer="below", line_width=0,
                                annotation=dict(text="Underperformance", textangle=0, y=1.02, yanchor="bottom", font=dict(size=10, color="gray"), showarrow=False)
                            )
                            
                        fig_inv_perf.update_layout(xaxis_title="Time of Day", hovermode="x unified", height=240, margin=dict(t=30, b=30))
                        fig_inv_perf.update_yaxes(title_text="AC Power (kW)", secondary_y=False)
                        fig_inv_perf.update_yaxes(title_text="Irradiance (W/m²)", secondary_y=True)
                        st.plotly_chart(fig_inv_perf, use_container_width=True)

            else:
                # Individual Event Zoom-In
                event_options = []
                for idx, ev in day_events.sort_values("Start Time").iterrows():
                    event_options.append(f"Event {ev['Event ID']} | Inverter {ev['Inverter ID']} | {ev['Fault Type']} | {ev['Start Time'].strftime('%H:%M')} - {ev['End Time'].strftime('%H:%M')}")
                
                selected_event_str = st.selectbox("Select Event to Zoom-In", event_options)
                selected_ev_id = int(selected_event_str.split(" | ")[0].split(" ")[1])
                
                event_row = events_df[events_df["Event ID"] == selected_ev_id].iloc[0]
                label_code = event_row["label_code"]
                inv_id = event_row["Inverter ID"]
                
                # Padding around event times
                pad_start = event_row["Start Time"] - pd.Timedelta(minutes=30)
                pad_end = event_row["End Time"] + pd.Timedelta(minutes=30)
                
                event_rows = day_telemetry[
                    (day_telemetry["device_time"] >= pad_start) &
                    (day_telemetry["device_time"] <= pad_end) &
                    (day_telemetry["inverter_id"] == inv_id)
                ]
                
                if not event_rows.empty:
                    if label_code == 1: # DC String Fault (F1)
                        fig_ev_str = make_subplots(specs=[[{"secondary_y": True}]])
                        for i in range(1, 7):
                            fig_ev_str.add_trace(
                                go.Scatter(x=event_rows["device_time"], y=event_rows[f"dc_current_{i}"], mode="lines+markers", name=f"String {i}"),
                                secondary_y=False
                            )
                        fig_ev_str.add_trace(
                            go.Scatter(x=event_rows["device_time"], y=event_rows["irradiance_wm2"], mode="lines+markers", name="Irradiance (W/m²)", line=dict(color="#ffd700", dash="dash")),
                            secondary_y=True
                        )
                        fig_ev_str.update_layout(title="Raw String Currents & Solar Irradiance during Event Window", xaxis_title="Time", hovermode="x unified")
                        fig_ev_str.update_yaxes(title_text="Current (A)", secondary_y=False)
                        fig_ev_str.update_yaxes(title_text="Solar Irradiance (W/m²)", secondary_y=True)
                        
                    elif label_code == 2: # Inverter Underperformance (F2)
                        fig_ev_str = go.Figure()
                        fig_ev_str.add_trace(go.Scatter(x=event_rows["device_time"], y=event_rows["ac_power_kw"], mode="lines+markers", name="Actual AC Power (kW)", line=dict(color="#1f77b4")))
                        fig_ev_str.add_trace(go.Scatter(x=event_rows["device_time"], y=event_rows["expected_power"], mode="lines+markers", name="Expected AC Power (kW)", line=dict(color="#2ca02c", dash="dash")))
                        fig_ev_str.update_layout(title="Raw Actual vs Expected AC Power during Event Window", xaxis_title="Time", yaxis_title="Power (kW)", hovermode="x unified")
                        
                    elif label_code == 3: # Grid Fault (F3)
                        fig_ev_str = go.Figure()
                        fig_ev_str.add_trace(go.Scatter(x=event_rows["device_time"], y=event_rows["grid_voltage_a"], mode="lines+markers", name="Phase A Voltage (V)"))
                        fig_ev_str.add_trace(go.Scatter(x=event_rows["device_time"], y=event_rows["grid_voltage_b"], mode="lines+markers", name="Phase B Voltage (V)"))
                        fig_ev_str.add_trace(go.Scatter(x=event_rows["device_time"], y=event_rows["grid_voltage_c"], mode="lines+markers", name="Phase C Voltage (V)"))
                        fig_ev_str.update_layout(title="Raw Phase Voltages during Event Window", xaxis_title="Time", yaxis_title="Voltage (V)", hovermode="x unified")
                        
                    else: # Inverter Offline (F4)
                        fig_ev_str = go.Figure()
                        fig_ev_str.add_trace(go.Scatter(x=event_rows["device_time"], y=event_rows["irradiance_wm2"], mode="lines+markers", name="Irradiance (W/m²)", line=dict(color="#ff7f0e")))
                        fig_ev_str.add_trace(go.Scatter(x=event_rows["device_time"], y=event_rows["ac_power_kw"], mode="lines+markers", name="AC Power (kW)", line=dict(color="#1f77b4")))
                        fig_ev_str.update_layout(title="Raw Solar Irradiance vs AC Power during Event Window", xaxis_title="Time", hovermode="x unified")
                        
                    # Highlight the actual event duration
                    fig_ev_str.add_vrect(
                        x0=event_row["Start Time"], x1=event_row["End Time"],
                        fillcolor="rgba(128, 128, 128, 0.15)", layer="below", line_width=0
                    )
                    st.plotly_chart(fig_ev_str, use_container_width=True)
