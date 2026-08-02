import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="BEML Solar Diagnostics Dashboard", layout="wide")

st.title("☀️ BEML 400 kW Solar PV Plant Diagnostics Dashboard")
st.markdown("Interactive performance analysis, physical fault labeling, and raw-data-driven root-cause diagnostics.")

# --- SIDEBAR FOR CONTROLS ---
st.sidebar.header("📁 Data Source & Filters")

# Ingestion mode
data_source = st.sidebar.radio("Select Data Input Mode", ["Upload Preprocessed CSV (Fast)", "Upload Raw Excel (Full Pipeline)"])

uploaded_file = st.sidebar.file_uploader("Choose a file", type=["csv", "xlsx"])

# --- FAULT RULES & THRESHOLDS BENDING PANEL ---
with st.sidebar.expander("⚙️ Customize Fault Thresholds"):
    st.markdown("Fine-tune physical parameters to dynamically re-label faults in real-time.")
    
    rules_config = {}
    st.markdown("**Inverter Offline (F4)**")
    rules_config["offline_irr"] = st.slider("F4: Min Irradiance (W/m²)", 10.0, 200.0, 50.0, 10.0)
    rules_config["offline_ac"] = st.slider("F4: Max AC Power (kW)", 0.1, 5.0, 0.5, 0.1)
    
    st.markdown("**DC String Fault (F1)**")
    rules_config["string_imbalance"] = st.slider("F1: Min imbalance ratio (σ/μ)", 0.10, 0.80, 0.25, 0.05)
    rules_config["string_dev"] = st.slider("F1: Min string deviation", 0.10, 0.80, 0.30, 0.05)
    rules_config["string_irr"] = st.slider("F1: Min irradiance (W/m²)", 100.0, 600.0, 300.0, 50.0)
    rules_config["string_ratio"] = st.slider("F1: All-string multiplier", 0.002, 0.015, 0.005, 0.001, format="%.4f")
    
    st.markdown("**Inverter Underperformance (F2)**")
    rules_config["inv_temp"] = st.slider("F2: Min IGBT Temp (°C)", 100.0, 180.0, 163.0, 5.0)
    rules_config["inv_residual"] = st.slider("F2: Power residual limit", -0.20, -0.01, -0.05, 0.01)
    
    st.markdown("**Grid Fault (F3)**")
    rules_config["grid_vuf"] = st.slider("F3: Max Voltage Unbalance (%)", 0.5, 5.0, 1.5, 0.1)
    rules_config["grid_vmin"] = st.slider("F3: Min Phase Voltage (V)", 100.0, 220.0, 200.0, 5.0)
    rules_config["grid_vmax"] = st.slider("F3: Max Phase Voltage (V)", 240.0, 290.0, 265.0, 5.0)
    rules_config["grid_freq"] = st.slider("F3: Frequency tolerance (Hz)", 0.1, 1.0, 0.3, 0.05)
    
    rules_config["standby_irr"] = st.slider("Standby: Evaluation cut-off (W/m²)", 10.0, 100.0, 50.0, 5.0)

# --- CORE DATA PROCESSING PIPELINE ---

@st.cache_data
def process_raw_excel(file):
    """Processes the raw Excel sheets to reconstruct timestamps and resample inverter telemetry."""
    st.info("Parsing raw Excel sheet... This may take a minute.")
    
    # Read the raw sheet (skipping metadata headers)
    df_raw = pd.read_excel(file, sheet_name="MicroSystem_1125_0426", header=22)
    
    # Reconstruct true timestamps from Modbus registers
    years = (df_raw["param_45"].fillna(0).astype(int) + 2000).astype(str)
    months = df_raw["param_44"].fillna(0).astype(int).astype(str).str.zfill(2)
    days = df_raw["param_43"].fillna(0).astype(int).astype(str).str.zfill(2)
    hours = df_raw["param_41"].fillna(0).astype(int).astype(str).str.zfill(2)
    minutes = df_raw["param_42"].fillna(0).astype(int).astype(str).str.zfill(2)
    
    date_strings = years + "-" + months + "-" + days + " " + hours + ":" + minutes + ":00"
    df_raw["device_time"] = pd.to_datetime(date_strings, errors="coerce")
    df_raw["device_time"] = df_raw["device_time"].fillna(pd.to_datetime(df_raw["recieved_at"]))
    
    # Resample weather sensor (sub_deviceid = 5)
    df_sens = df_raw[(df_raw["pkt_type"] == 5) | (df_raw["sub_deviceid"] == 5)].copy()
    df_sens["module_temperature"] = pd.to_numeric(df_sens["param_01"], errors="coerce") * 0.0265596
    df_sens["irradiance_wm2"] = pd.to_numeric(df_sens["param_02"], errors="coerce") * 0.5319149
    
    # Outlier filter
    df_sens.loc[(df_sens["irradiance_wm2"] < 0) | (df_sens["irradiance_wm2"] > 1400), "irradiance_wm2"] = np.nan
    df_sens.loc[(df_sens["module_temperature"] < -20) | (df_sens["module_temperature"] > 100), "module_temperature"] = np.nan
    
    df_sens.drop_duplicates(subset=["device_time"], inplace=True)
    df_sens.set_index("device_time", inplace=True)
    df_sens.sort_index(inplace=True)
    
    df_sens_10m = df_sens.resample("10min").agg({
        "irradiance_wm2": "mean",
        "module_temperature": "mean"
    }).interpolate(method="time", limit=3)
    
    # Inverter columns config
    inverter_gain_map = {
        "param_01": 0.1, "param_02": 0.1, "param_03": 0.1,
        "param_04": 0.1, "param_05": 0.1, "param_06": 0.1,
        "param_07": 0.1, "param_08": 0.1, "param_09": 0.1,
        "param_10": 0.0001, "param_11": 0.1, "param_12": 0.1, "param_13": 0.1,
        "param_14": 0.01, "param_15": 0.001,
        "param_16": 0.1, "param_17": 0.1, "param_18": 0.1, "param_19": 0.1,
        "param_20": 0.1, "param_21": 0.1, "param_22": 0.1, "param_23": 0.1,
        "param_24": 0.1, "param_25": 0.1, "param_26": 0.1, "param_27": 0.1,
        "param_37": 0.1, "param_38": 0.1, "param_39": 0.1, "param_40": 0.1
    }
    
    inverter_name_map = {
        "param_01": "grid_voltage_ab", "param_02": "grid_voltage_bc", "param_03": "grid_voltage_ac",
        "param_04": "grid_voltage_a", "param_05": "grid_voltage_b", "param_06": "grid_voltage_c",
        "param_07": "grid_current_a", "param_08": "grid_current_b", "param_09": "grid_current_c",
        "param_10": "ac_power_kw", "param_11": "heat_sink_temp", "param_12": "igbt_temp", "param_13": "inductance_temp",
        "param_14": "grid_frequency_hz", "param_15": "power_factor",
        "param_16": "dc_voltage_1", "param_17": "dc_current_1",
        "param_18": "dc_voltage_2", "param_19": "dc_current_2",
        "param_20": "dc_voltage_3", "param_21": "dc_current_3",
        "param_22": "dc_voltage_4", "param_23": "dc_current_4",
        "param_24": "dc_voltage_5", "param_25": "dc_current_5",
        "param_26": "dc_voltage_6", "param_27": "dc_current_6",
        "param_37": "dc_voltage_7", "param_38": "dc_current_7",
        "param_39": "dc_voltage_8", "param_40": "dc_current_8"
    }
    
    resample_config = {
        "ac_power_kw": "mean", "grid_voltage_a": "mean", "grid_voltage_b": "mean", "grid_voltage_c": "mean",
        "grid_voltage_ab": "mean", "grid_voltage_bc": "mean", "grid_voltage_ac": "mean",
        "grid_current_a": "mean", "grid_current_b": "mean", "grid_current_c": "mean",
        "heat_sink_temp": "mean", "igbt_temp": "mean", "inductance_temp": "mean",
        "grid_frequency_hz": "mean", "power_factor": "mean"
    }
    for i in range(1, 7):
        resample_config[f"dc_voltage_{i}"] = "mean"
        resample_config[f"dc_current_{i}"] = "mean"
        
    inverter_dfs = []
    
    # Process 4 inverters
    for inv_id in [1, 2, 3, 4]:
        df_inv = df_raw[(df_raw["sub_deviceid"] == inv_id) & (df_raw["pkt_type"] != 5)].copy()
        
        for col, gain in inverter_gain_map.items():
            if col in df_inv.columns:
                df_inv[col] = pd.to_numeric(df_inv[col], errors="coerce") * gain
        df_inv.rename(columns=inverter_name_map, inplace=True)
        
        df_inv.loc[(df_inv["ac_power_kw"] < -1) | (df_inv["ac_power_kw"] > 120), "ac_power_kw"] = np.nan
        df_inv.loc[(df_inv["grid_frequency_hz"] < 45) | (df_inv["grid_frequency_hz"] > 55), "grid_frequency_hz"] = np.nan
        df_inv.loc[(df_inv["power_factor"] < -1) | (df_inv["power_factor"] > 1), "power_factor"] = np.nan
        
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
        
        inverter_dfs.append(df_joined)
        
    combined = pd.concat(inverter_dfs, ignore_index=True)
    return combined

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
    df["month_str"] = df["device_time"].dt.strftime("%Y-%m")
    
    # DC Current imbalance metrics
    dc_curr_cols = ["dc_current_1", "dc_current_2", "dc_current_3", "dc_current_4", "dc_current_5", "dc_current_6"]
    df["dc_curr_mean"] = df[dc_curr_cols].mean(axis=1)
    df["dc_curr_std"] = df[dc_curr_cols].std(axis=1)
    df["dc_imbalance"] = df["dc_curr_std"] / (df["dc_curr_mean"] + 0.01)
    
    for i in range(1, 7):
        df[f"dc_dev_{i}"] = (df[f"dc_current_{i}"] - df["dc_curr_mean"]) / (df["dc_curr_mean"] + 0.01)
    dev_cols = [f"dc_dev_{i}" for i in range(1, 7)]
    df["max_string_dev"] = df[dev_cols].abs().max(axis=1)
    
    # AC grid voltage metrics
    grid_volt_cols = ["grid_voltage_a", "grid_voltage_b", "grid_voltage_c"]
    df["grid_v_mean"] = df[grid_volt_cols].mean(axis=1)
    df["VUF"] = (df[grid_volt_cols].sub(df["grid_v_mean"], axis=0).abs().max(axis=1) / df["grid_v_mean"]) * 100
    
    # Expected power regression baseline (trained on healthy historical data)
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
    
    # Physical fault labeling
    df["label"] = 0
    
    # F4: Offline
    offline_condition = (df["ac_power_kw"] <= rules_config["offline_ac"]) & (df["irradiance_wm2"] > rules_config["offline_irr"])
    df.loc[offline_condition, "label"] = 4
    
    # F3: Grid Fault
    grid_condition = (
        (df["ac_power_kw"] > rules_config["offline_ac"])
        & (
            (df["VUF"] > rules_config["grid_vuf"])
            | ((df["grid_frequency_hz"] - 50).abs() > rules_config["grid_freq"])
            | (df[grid_volt_cols] > rules_config["grid_vmax"]).any(axis=1)
            | (df[grid_volt_cols] < rules_config["grid_vmin"]).any(axis=1)
        )
    )
    df.loc[grid_condition & (df["label"] == 0), "label"] = 3
    
    # F1: DC String Fault (either spatial imbalance OR all strings underproducing together under high irradiance)
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
    
    # Enforce standby filter: faults are only evaluated when irradiance > standby_irr
    df.loc[df["irradiance_wm2"] <= rules_config["standby_irr"], "label"] = 0
    
    return df

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
        
        # Calculate clipped row-by-row loss sum
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

def compute_fault_distribution(df):
    """Computes row counts and percentages for daytime faults."""
    counts = df["label"].value_counts()
    total = len(df)
    
    data = []
    labels_info = [
        (0, "Normal (F0)"),
        (4, "Inverter Offline (F4)"),
        (2, "Inverter Underperformance (F2)"),
        (1, "DC String Fault (F1)"),
        (3, "Grid Fault (F3)")
    ]
    
    for code, name in labels_info:
        cnt = counts.get(code, 0)
        pct = (cnt / total * 100) if total > 0 else 0.0
        data.append({
            "Fault Code & Name": name,
            "Count of Rows": cnt,
            "Percentage of Daytime Data": f"{pct:.2f}%"
        })
    return pd.DataFrame(data)

def compute_inverter_burden(df):
    """Computes inverter-wise cumulative hours for each fault code."""
    inv_labels = df.groupby(["inverter_id", "label"]).size().unstack(fill_value=0)
    for l in [0, 1, 2, 3, 4]:
        if l not in inv_labels.columns:
            inv_labels[l] = 0
            
    # Convert counts to hours (1 row = 10 min = 1/6 hrs)
    inv_labels = inv_labels / 6.0
    
    inv_labels.rename(columns={
        0: "Normal (F0) hrs",
        1: "String Fault (F1) hrs",
        2: "Inverter Underperformance (F2) hrs",
        3: "Grid Fault (F3) hrs",
        4: "Offline (F4) hrs"
    }, inplace=True)
    
    # Sort columns
    ordered_cols = ["Normal (F0) hrs", "Offline (F4) hrs", "Inverter Underperformance (F2) hrs", "String Fault (F1) hrs", "Grid Fault (F3) hrs"]
    inv_labels = inv_labels[ordered_cols]
    inv_labels.index.name = "Inverter"
    return inv_labels.reset_index()

def compute_fault_durations_table(df):
    """Computes total duration, energy loss, and average hourly loss for each fault type."""
    events_df = build_fault_events(df)
    if events_df.empty:
        return pd.DataFrame()
        
    summary = events_df.groupby("Fault Type").agg(
        total_dur=("Duration (hrs)", "sum"),
        total_loss=("Energy Lost (kWh)", "sum")
    ).reset_index()
    
    summary["Average Loss per Hour (kW)"] = (summary["total_loss"] / summary["total_dur"]).round(2).fillna(0.0)
    
    summary.rename(columns={
        "Fault Type": "Fault Type",
        "total_dur": "Total Duration (hrs)",
        "total_loss": "Total Energy Lost (kWh)"
    }, inplace=True)
    
    # Order rows by F4, F2, F1, F3
    order = {"Inverter Offline (F4)": 0, "Inverter Underperformance (F2)": 1, "DC String Fault (F1)": 2, "Grid Fault (F3)": 3}
    summary["rank"] = summary["Fault Type"].map(order).fillna(4)
    summary = summary.sort_values("rank").drop(columns=["rank"]).reset_index(drop=True)
    return summary

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
    
    # Sort by Date descending by default
    return daily.sort_values("Date", ascending=False).reset_index(drop=True)

# --- LOAD DATA IN APP ---

df_loaded = None

if uploaded_file is not None:
    if data_source == "Upload Raw Excel (Full Pipeline)":
        try:
            raw_data = process_raw_excel(uploaded_file)
            df_loaded = run_feature_engineering_and_labeling(raw_data, rules_config=rules_config)
            st.success("Successfully processed and labeled raw Excel sheet!")
        except Exception as e:
            st.error(f"Error parsing raw Excel: {e}")
    else:
        try:
            raw_data = pd.read_csv(uploaded_file, parse_dates=["device_time"])
            df_loaded = run_feature_engineering_and_labeling(raw_data, rules_config=rules_config)
            st.success("Loaded preprocessed dataset and applied custom rules!")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
else:
    st.warning("Please upload your BEML Excel or CSV telemetry file in the sidebar to begin analysis.")

# --- RENDER DASHBOARD CONTROLS AND CHARTS ---

if df_loaded is not None:
    # Ensure dates and month columns exist
    df_loaded["device_time"] = pd.to_datetime(df_loaded["device_time"])
    df_loaded["month_str"] = df_loaded["device_time"].dt.strftime("%Y-%m")
    
    # Filter selections
    months_avail = sorted(df_loaded["month_str"].unique())
    selected_months = st.sidebar.multiselect("Select Months", months_avail, default=months_avail)
    
    inverters_avail = sorted(df_loaded["inverter_id"].unique())
    selected_inverters = st.sidebar.multiselect("Select Inverters", inverters_avail, default=inverters_avail)
    
    min_irr, max_irr = int(df_loaded["irradiance_wm2"].min()), int(df_loaded["irradiance_wm2"].max())
    selected_irr = st.sidebar.slider("Minimum Irradiance (W/m²)", min_irr, max_irr, 20)
    
    # Fallback to all values if nothing is selected (prevents empty screen errors)
    active_months = selected_months if selected_months else months_avail
    active_inverters = selected_inverters if selected_inverters else inverters_avail
    
    # Apply filters to dataframe
    df_filtered = df_loaded[
        (df_loaded["month_str"].isin(active_months)) &
        (df_loaded["inverter_id"].isin(active_inverters)) &
        (df_loaded["irradiance_wm2"] >= selected_irr)
    ].copy()
    
    # --- 1. KPI SUMMARY CARDS ---
    st.subheader("📊 Plant Performance Metrics")
    
    actual_gen = df_filtered["ac_power_kw"].sum() * (10 / 60)
    expected_gen = df_filtered["expected_power"].sum() * (10 / 60)
    
    # Clipped row-by-row loss summation
    row_losses = (df_filtered["expected_power"] - df_filtered["ac_power_kw"]).clip(lower=0)
    total_loss = row_losses.sum() * (10 / 60)
    loss_rate = (total_loss / expected_gen * 100) if expected_gen > 0 else 0.0
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Actual Generation", f"{actual_gen:,.1f} kWh")
    col2.metric("Expected Generation (Baseline)", f"{expected_gen:,.1f} kWh")
    col3.metric("Total Energy Lost", f"{total_loss:,.1f} kWh", delta=f"{loss_rate:.2f}% loss rate", delta_color="inverse")
    
    # Overall uptime hours
    uptime_hrs = len(df_filtered[df_filtered["label"] == 0]) * (10 / 60)
    total_hrs = len(df_filtered) * (10 / 60)
    availability = (uptime_hrs / total_hrs * 100) if total_hrs > 0 else 100.0
    col4.metric("Healthy Availability", f"{availability:.2f}%")
    
    # --- 2. MULTI-TAB DISPLAY ---
    tab1, tab2 = st.tabs(["📉 Visual Analytics", "📋 Diagnostics Dashboard"])
    
    with tab1:
        st.subheader("Temporal Fault Durations & Performance Trends")
        
        # Scale aggregator selection
        time_scale = st.selectbox("Select Temporal Aggregation Scale", ["Daily", "Weekly", "Monthly"])
        
        # Add aggregation column
        if time_scale == "Daily":
            df_filtered["agg_period"] = df_filtered["device_time"].dt.date
        elif time_scale == "Weekly":
            df_filtered["agg_period"] = df_filtered["device_time"].dt.to_period("W").dt.start_time.dt.strftime("%Y-W%U")
        else:
            df_filtered["agg_period"] = df_filtered["month_str"]
            
        # Plot A: Monthly Generation vs. Loss
        monthly_summary = df_filtered.groupby("month_str").agg(
            actual_gen=("ac_power_kw", lambda x: x.sum() * (10/60)),
            expected_gen=("expected_power", lambda x: x.sum() * (10/60))
        ).reset_index()
        monthly_summary["loss_gen"] = (monthly_summary["expected_gen"] - monthly_summary["actual_gen"]).clip(lower=0)
        
        fig_monthly = go.Figure()
        fig_monthly.add_trace(go.Bar(name="Actual AC Generation", x=monthly_summary["month_str"], y=monthly_summary["actual_gen"], marker_color="#1f77b4"))
        fig_monthly.add_trace(go.Scatter(name="Energy Lost", x=monthly_summary["month_str"], y=monthly_summary["loss_gen"], mode="lines+markers", line=dict(color="#d62728", width=3)))
        fig_monthly.update_layout(xaxis_title="Month", yaxis_title="Energy (kWh)", barmode="group", hovermode="x unified", height=380)
        st.plotly_chart(fig_monthly, use_container_width=True)
        
        # Plot B: Cumulative Fault Hours grouped by time scale
        st.subheader(f"Downtime Trend ({time_scale} Aggregation)")
        fault_summary = df_filtered[df_filtered["label"] != 0].groupby(["agg_period", "label"]).size().reset_index(name="counts")
        fault_summary["hours"] = fault_summary["counts"] * (10 / 60)
        
        label_names = {1: "DC String Fault (F1)", 2: "Inverter Underperformance (F2)", 3: "Grid Fault (F3)", 4: "Inverter Offline (F4)"}
        fault_summary["Fault Type"] = fault_summary["label"].map(label_names)
        
        fig_dur = px.bar(fault_summary, x="agg_period", y="hours", color="Fault Type", 
                         color_discrete_map={"DC String Fault (F1)": "#ff7f0e", "Inverter Underperformance (F2)": "#d62728", "Grid Fault (F3)": "#2ca02c", "Inverter Offline (F4)": "#7f7f7f"},
                         labels={"agg_period": "Period", "hours": "Downtime Duration (hrs)"}, height=400)
        fig_dur.update_layout(barmode="stack")
        st.plotly_chart(fig_dur, use_container_width=True)
        
        # Automated Text Analysis Card based on selection
        st.subheader("💡 Aggregated Performance Analysis")
        if not fault_summary.empty:
            # Find the period with maximum fault hours
            worst_period_df = fault_summary.groupby("agg_period")["hours"].sum().reset_index()
            worst_row = worst_period_df.loc[worst_period_df["hours"].idxmax()]
            worst_period = worst_row["agg_period"]
            worst_hours = worst_row["hours"]
            
            # Find dominant fault in worst period
            period_faults = fault_summary[fault_summary["agg_period"] == worst_period]
            worst_fault_row = period_faults.loc[period_faults["hours"].idxmax()]
            worst_fault = worst_fault_row["Fault Type"]
            worst_fault_pct = (worst_fault_row["hours"] / worst_hours) * 100
            
            # Find most affected inverter in worst period
            worst_inv_df = df_filtered[
                (df_filtered["agg_period"] == worst_period) & (df_filtered["label"] != 0)
            ].groupby("inverter_id").size().reset_index(name="counts")
            worst_inv_id = worst_inv_df.loc[worst_inv_df["counts"].idxmax()]["inverter_id"]
            
            st.info(
                f"**Worst Performance Period**: Under your selected **{time_scale}** view, the worst period was **{worst_period}** logging **{worst_hours:.2f} hours** of total downtime. "
                f"The primary loss driver was **{worst_fault}**, accounting for **{worst_fault_pct:.1f}%** of the total fault hours. "
                f"Inverter **{worst_inv_id}** was the most affected unit during this window."
            )
        else:
            st.success("No fault records logged for the selected configuration.")
            
        st.write("---")
        st.subheader("📋 Industrial Performance & Fault Burden Analysis")
        
        st.markdown("### Daytime Fault Distribution")
        fault_dist = compute_fault_distribution(df_filtered)
        st.dataframe(fault_dist, width="stretch", hide_index=True)
        
        st.write("")
        st.markdown("### Fault Durations & Losses Summary")
        durations_table = compute_fault_durations_table(df_filtered)
        if not durations_table.empty:
            st.dataframe(durations_table, width="stretch", hide_index=True)
        else:
            st.info("No fault events found matching current filters.")
            
        st.write("")
        st.markdown("### Overall Plant Performance & Monthly Energy Loss")
        monthly_perf = df_filtered.groupby("month_str").agg(
            actual_kwh=("ac_power_kw", lambda x: x.sum() * (10/60)),
            expected_kwh=("expected_power", lambda x: x.sum() * (10/60))
        ).reset_index()
        monthly_perf["Energy Lost (kWh)"] = (monthly_perf["expected_kwh"] - monthly_perf["actual_kwh"]).clip(lower=0)
        monthly_perf["Loss Rate (%)"] = (monthly_perf["Energy Lost (kWh)"] / monthly_perf["expected_kwh"] * 100).fillna(0.0)
        
        monthly_perf_disp = pd.DataFrame({
            "Month": monthly_perf["month_str"],
            "Actual Generation (kWh)": monthly_perf["actual_kwh"].map(lambda x: f"{x:,.1f}"),
            "Energy Lost (kWh)": monthly_perf["Energy Lost (kWh)"].map(lambda x: f"{x:,.1f}"),
            "Overall Loss Rate (%)": monthly_perf["Loss Rate (%)"].map(lambda x: f"{x:.2f}%")
        })
        st.dataframe(monthly_perf_disp, width="stretch", hide_index=True)
        
        st.write("")
        st.markdown("### Inverter-Wise Fault Burden (Hours)")
        inv_burden_table = compute_inverter_burden(df_filtered)
        st.dataframe(inv_burden_table, width="stretch", hide_index=True)
            
    with tab2:
        st.subheader("📋 Advanced Diagnostics & Root-Cause Logs")
        
        # Build event dataframe
        events_df = build_fault_events(df_filtered)
        
        # Compute daily aggregated summary table
        daily_df = compute_daily_aggregation_table(df_filtered)
        
        if daily_df.empty:
            st.success("No fault events found matching current filter configuration!")
        else:
            st.markdown("### 📅 Daily Aggregated Fault Summary")
            st.markdown("Click on column headers (e.g. **Total Duration (hrs)** or **Total Energy Lost (kWh)**) to sort and identify worst-performing days.")
            st.dataframe(daily_df, width="stretch", hide_index=True)
            
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
            
            # Daily KPI Cards
            st.write("")
            col_k1, col_k2, col_k3 = st.columns(3)
            day_hours = day_events["Duration (hrs)"].sum()
            day_loss = day_events["Energy Lost (kWh)"].sum()
            worst_inv = day_events.groupby("Inverter ID")["Duration (hrs)"].sum().idxmax()
            
            col_k1.metric("Daily Fault Duration", f"{day_hours:.2f} hrs")
            col_k2.metric("Daily Energy Lost", f"{day_loss:.2f} kWh")
            col_k3.metric("Worst-Performing Inverter", f"Inverter {worst_inv}")
            
            # Dynamic Anomaly Summary Narrative
            st.subheader(f"📝 Raw Sensor Diagnostics Summary for {selected_date}")
            for idx, ev in day_events.sort_values("Start Time").iterrows():
                inv_id = ev["Inverter ID"]
                lbl_code = ev["label_code"]
                start_str = ev["Start Time"].strftime("%H:%M")
                end_str = ev["End Time"].strftime("%H:%M")
                
                # Extract event telemetry rows
                ev_tele = day_telemetry[
                    (day_telemetry["device_time"] >= ev["Start Time"]) &
                    (day_telemetry["device_time"] < ev["End Time"]) &
                    (day_telemetry["inverter_id"] == inv_id)
                ]
                
                if ev_tele.empty:
                    continue
                    
                if lbl_code == 1: # DC String Fault
                    dc_curr_cols = ["dc_current_1", "dc_current_2", "dc_current_3", "dc_current_4", "dc_current_5", "dc_current_6"]
                    mean_currents = [ev_tele[col].mean() for col in dc_curr_cols]
                    dropped_idx = np.argmin(mean_currents)
                    dropped_num = dropped_idx + 1
                    dropped_val = mean_currents[dropped_idx]
                    avg_others = np.mean([mean_currents[i] for i in range(6) if i != dropped_idx])
                    avg_irr = ev_tele["irradiance_wm2"].mean()
                    
                    if avg_others < 0.006 * avg_irr:
                        item_text = f"🔴 **{start_str} - {end_str}** | **Inverter {inv_id}**: Plant-wide DC string collapse. Average string current fell to **{np.mean(mean_currents):.2f} A** under **{avg_irr:.1f} W/m²** irradiance (probable main combiner box fuse trip)."
                    else:
                        item_text = f"🟠 **{start_str} - {end_str}** | **Inverter {inv_id}**: DC String Fault isolated on **String {dropped_num}**. String current fell to **{dropped_val:.2f} A** while the other 5 healthy strings averaged **{avg_others:.2f} A** under **{avg_irr:.1f} W/m²** irradiance."
                        
                elif lbl_code == 2: # Inverter Underperformance
                    mean_ac = ev_tele["ac_power_kw"].mean()
                    mean_expected = ev_tele["expected_power"].mean()
                    mean_igbt = ev_tele["igbt_temp"].mean()
                    item_text = f"🟠 **{start_str} - {end_str}** | **Inverter {inv_id}**: Inverter Underperformance. Raw AC power was **{mean_ac:.2f} kW** (Expected: **{mean_expected:.2f} kW** with IGBT temperature at **{mean_igbt:.1f}°C**)."
                    
                elif lbl_code == 3: # Grid Fault
                    mean_va = ev_tele["grid_voltage_a"].mean()
                    mean_vb = ev_tele["grid_voltage_b"].mean()
                    mean_vc = ev_tele["grid_voltage_c"].mean()
                    mean_freq = ev_tele["grid_frequency_hz"].mean()
                    item_text = f"🔵 **{start_str} - {end_str}** | **Inverter {inv_id}**: Grid parameter sag. Phase Voltages averaged **[A: {mean_va:.1f}V, B: {mean_vb:.1f}V, C: {mean_vc:.1f}V]** and Grid Frequency averaged **{mean_freq:.3f} Hz**."
                    
                elif lbl_code == 4: # Inverter Offline
                    mean_irr = ev_tele["irradiance_wm2"].mean()
                    mean_ac = ev_tele["ac_power_kw"].mean()
                    item_text = f"🔴 **{start_str} - {end_str}** | **Inverter {inv_id}**: Inverter Offline. AC power remained at **{mean_ac:.2f} kW** despite raw solar irradiance averaging **{mean_irr:.1f} W/m²**."
                    
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

            # Row 3: Visualization plotting mode selection
            st.write("---")
            st.subheader("📈 Telemetry Visualizations")
            plot_mode = st.selectbox("Select Plotting Mode", ["All Faults (Full Day Timelines)", "Individual Event Zoom-In"])
            
            if plot_mode == "All Faults (Full Day Timelines)":
                # Plot 1: Inverter AC Power & Irradiance (Offline F4 Timeline)
                st.markdown("#### 1. Inverter AC Power & Solar Irradiance (Offline F4)")
                from plotly.subplots import make_subplots
                fig_f4_day = make_subplots(specs=[[{"secondary_y": True}]])
                for inv in active_inverters:
                    inv_day_tele = day_telemetry[day_telemetry["inverter_id"] == inv]
                    if not inv_day_tele.empty:
                        fig_f4_day.add_trace(
                            go.Scatter(x=inv_day_tele["device_time"], y=inv_day_tele["ac_power_kw"], mode="lines", name=f"Inv {inv} Power"),
                            secondary_y=False
                        )
                # Add irradiance (constant across all inverters)
                fig_f4_day.add_trace(
                    go.Scatter(x=day_telemetry["device_time"].unique(), y=day_telemetry.groupby("device_time")["irradiance_wm2"].mean(), mode="lines", name="Irradiance", line=dict(color="#ffd700", dash="dash")),
                    secondary_y=True
                )
                
                # Highlight Offline F4 windows with unified light neutral color (merged intervals)
                merged_f4 = merge_fault_intervals(day_events, 4)
                for start_t, end_t, invs in merged_f4:
                    invs_label = ", ".join(f"Inv {i}" for i in sorted(invs))
                    fig_f4_day.add_vrect(
                        x0=start_t,
                        x1=end_t,
                        fillcolor="rgba(128, 128, 128, 0.15)",
                        layer="below",
                        line_width=0,
                        annotation=dict(
                            text=f"{invs_label} Offline",
                            textangle=0,
                            y=1.02,
                            yanchor="bottom",
                            font=dict(size=10, color="gray"),
                            showarrow=False
                        )
                    )
                    
                fig_f4_day.update_layout(xaxis_title="Time of Day", hovermode="x unified", height=320)
                fig_f4_day.update_yaxes(title_text="AC Power (kW)", secondary_y=False)
                fig_f4_day.update_yaxes(title_text="Irradiance (W/m²)", secondary_y=True)
                st.plotly_chart(fig_f4_day, width="stretch")
                
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
                        
                    # Highlight String Faults F1 for the selected inverter (merged intervals)
                    merged_f1 = merge_fault_intervals(day_events[day_events["Inverter ID"] == sel_inv_str], 1)
                    for start_t, end_t, _ in merged_f1:
                        fig_f1_day.add_vrect(
                            x0=start_t,
                            x1=end_t,
                            fillcolor="rgba(128, 128, 128, 0.15)",
                            layer="below",
                            line_width=0,
                            annotation=dict(
                                text="String Fault (F1)",
                                textangle=0,
                                y=1.02,
                                yanchor="bottom",
                                font=dict(size=10, color="gray"),
                                showarrow=False
                            )
                        )
                        
                    fig_f1_day.update_layout(xaxis_title="Time of Day", yaxis_title="DC Current (A)", hovermode="x unified", height=320)
                    st.plotly_chart(fig_f1_day, width="stretch")
                    
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
                            
                # Highlight Grid Faults F3 (merged intervals)
                merged_f3 = merge_fault_intervals(day_events, 3)
                for start_t, end_t, invs in merged_f3:
                    invs_label = ", ".join(f"Inv {i}" for i in sorted(invs))
                    fig_f3_day.add_vrect(
                        x0=start_t,
                        x1=end_t,
                        fillcolor="rgba(128, 128, 128, 0.15)",
                        layer="below",
                        line_width=0,
                        annotation=dict(
                            text=f"{invs_label} Grid Fault",
                            textangle=0,
                            y=1.02,
                            yanchor="bottom",
                            font=dict(size=10, color="gray"),
                            showarrow=False
                        )
                    )
                    
                fig_f3_day.update_layout(xaxis_title="Time of Day", yaxis_title="Voltage (V)", hovermode="x unified", height=320)
                st.plotly_chart(fig_f3_day, width="stretch")
                
                # Plot 4: Inverter Underperformance (Actual vs Expected AC Power)
                st.write("")
                st.markdown("#### 4. Inverter Actual vs Expected AC Power (Underperformance F2)")
                from plotly.subplots import make_subplots
                for inv in active_inverters:
                    inv_day_perf = day_telemetry[day_telemetry["inverter_id"] == inv]
                    if not inv_day_perf.empty:
                        st.markdown(f"##### Inverter {inv} Performance")
                        
                        fig_inv_perf = make_subplots(specs=[[{"secondary_y": True}]])
                        
                        # Actual AC (Primary Y-axis)
                        fig_inv_perf.add_trace(
                            go.Scatter(x=inv_day_perf["device_time"], y=inv_day_perf["ac_power_kw"], mode="lines", name=f"Inv {inv} Actual AC", line=dict(color="#1f77b4")),
                            secondary_y=False
                        )
                        # Expected AC (Primary Y-axis)
                        fig_inv_perf.add_trace(
                            go.Scatter(x=inv_day_perf["device_time"], y=inv_day_perf["expected_power"], mode="lines", name=f"Inv {inv} Expected AC", line=dict(color="#2ca02c", dash="dash")),
                            secondary_y=False
                        )
                        # Irradiance (Secondary Y-axis)
                        fig_inv_perf.add_trace(
                            go.Scatter(x=inv_day_perf["device_time"], y=inv_day_perf["irradiance_wm2"], mode="lines", name="Solar Irradiance", line=dict(color="#ffd700", dash="dot")),
                            secondary_y=True
                        )
                        
                        # Highlight Inverter Underperformance F2 for this specific inverter (merged intervals)
                        merged_f2 = merge_fault_intervals(day_events[day_events["Inverter ID"] == inv], 2)
                        for start_t, end_t, _ in merged_f2:
                            fig_inv_perf.add_vrect(
                                x0=start_t,
                                x1=end_t,
                                fillcolor="rgba(128, 128, 128, 0.15)",
                                layer="below",
                                line_width=0,
                                annotation=dict(
                                    text="Underperformance",
                                    textangle=0,
                                    y=1.02,
                                    yanchor="bottom",
                                    font=dict(size=10, color="gray"),
                                    showarrow=False
                                )
                            )
                            
                        fig_inv_perf.update_layout(xaxis_title="Time of Day", hovermode="x unified", height=240, margin=dict(t=30, b=30))
                        fig_inv_perf.update_yaxes(title_text="AC Power (kW)", secondary_y=False)
                        fig_inv_perf.update_yaxes(title_text="Irradiance (W/m²)", secondary_y=True)
                        st.plotly_chart(fig_inv_perf, width="stretch")
                
            else: # Individual Event Zoom-In
                # Dropdown containing Event IDs for that date
                day_ev_options = [f"Event {ev['Event ID']} | Inverter {ev['Inverter ID']} | {ev['Fault Type']}" for idx, ev in day_events.sort_values("Start Time").iterrows()]
                if not day_ev_options:
                    st.info("No individual event details available for this selection.")
                else:
                    selected_ev_str = st.selectbox("Select Specific Event to Zoom-In", day_ev_options)
                    selected_event_id = int(selected_ev_str.split(" | ")[0].replace("Event ", ""))
                    
                    event_details = events_df[events_df["Event ID"] == selected_event_id].iloc[0]
                    
                    # Extract raw event rows
                    event_rows = df_filtered[
                        (df_filtered["device_time"] >= event_details["Start Time"]) &
                        (df_filtered["device_time"] < event_details["End Time"]) &
                        (df_filtered["inverter_id"] == event_details["Inverter ID"])
                    ].copy()
                    
                    label_code = event_details["label_code"]
                    
                    if label_code == 1: # DC String Fault
                        dc_curr_cols = ["dc_current_1", "dc_current_2", "dc_current_3", "dc_current_4", "dc_current_5", "dc_current_6"]
                        mean_currents = [event_rows[col].mean() for col in dc_curr_cols]
                        dropped_string_idx = np.argmin(mean_currents)
                        dropped_string_num = dropped_string_idx + 1
                        dropped_current = mean_currents[dropped_string_idx]
                        
                        other_currents = [mean_currents[i] for i in range(6) if i != dropped_string_idx]
                        avg_other_currents = np.mean(other_currents)
                        avg_irr = event_rows["irradiance_wm2"].mean()
                        
                        if avg_other_currents < 0.006 * avg_irr:
                            diagnosis_text = (
                                f"**Conclusion**: Plant-wide DC String underperformance / Combiner Box Fault.\n\n"
                                f"**Raw Telemetry Evidence**:\n"
                                f"*   Average current across all 6 strings dropped severely to **{np.mean(mean_currents):.2f} A**.\n"
                                f"*   Expected healthy string output at **{avg_irr:.1f} W/m²** irradiance is **> {0.008 * avg_irr:.1f} A** per string.\n"
                                f"*   Because all strings are depressed together, this points to a common-mode DC failure rather than an isolated string.\n\n"
                                f"**Verdict**: High probability of a tripped main DC isolator switch, a blown main combiner fuse, or severe plant-wide solar array soiling."
                            )
                        else:
                            diagnosis_text = (
                                f"**Conclusion**: DC String Fault isolated on **String {dropped_string_num}**.\n\n"
                                f"**Raw Telemetry Evidence**:\n"
                                f"*   String {dropped_string_num} current fell to an average of **{dropped_current:.2f} A**.\n"
                                f"*   The other 5 healthy strings operated normally, averaging **{avg_other_currents:.2f} A**.\n"
                                f"*   This drop occurred under raw solar irradiance averaging **{avg_irr:.1f} W/m²**.\n\n"
                                f"**Verdict**: Indicates a physically blown string fuse, failed bypass diode, or localized obstruction on String {dropped_string_num} arrays."
                            )
                        
                        from plotly.subplots import make_subplots
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
                        
                    elif label_code == 2: # Inverter Underperformance
                        mean_igbt = event_rows["igbt_temp"].mean()
                        mean_ac = event_rows["ac_power_kw"].mean()
                        mean_expected = event_rows["expected_power"].mean()
                        
                        diagnosis_text = (
                            f"**Conclusion**: Inverter underperformance detected.\n\n"
                            f"**Raw Telemetry Evidence**:\n"
                            f"*   Raw AC power output averaged **{mean_ac:.2f} kW**.\n"
                            f"*   Expected healthy baseline generation is **{mean_expected:.2f} kW**.\n"
                            f"*   Underperformance deficit: **{(mean_expected - mean_ac):.2f} kW**.\n"
                            f"*   Raw IGBT junction temperature averaged **{mean_igbt:.1f}°C**.\n\n"
                            f"**Verdict**: Inverter is generating below its expected power baseline. This can be caused by physical temperature throttling, component degradation, or localized DC input limits."
                        )
                        
                        fig_ev_str = go.Figure()
                        fig_ev_str.add_trace(go.Scatter(x=event_rows["device_time"], y=event_rows["ac_power_kw"], mode="lines+markers", name="Actual AC Power (kW)", line=dict(color="#1f77b4")))
                        fig_ev_str.add_trace(go.Scatter(x=event_rows["device_time"], y=event_rows["expected_power"], mode="lines+markers", name="Expected AC Power (kW)", line=dict(color="#ffd700", dash="dash")))
                        fig_ev_str.update_layout(title="Raw Actual vs Expected AC Power during Event Window", xaxis_title="Time", yaxis_title="Power (kW)")
                        
                    elif label_code == 3: # Grid Fault
                        mean_va = event_rows["grid_voltage_a"].mean()
                        mean_vb = event_rows["grid_voltage_b"].mean()
                        mean_vc = event_rows["grid_voltage_c"].mean()
                        mean_freq = event_rows["grid_frequency_hz"].mean()
                        
                        diagnosis_text = (
                            f"**Conclusion**: Utility Grid parameter out-of-bounds anomaly.\n\n"
                            f"**Raw Telemetry Evidence**:\n"
                            f"*   Raw Phase-Neutral Voltages averaged: Phase A: **{mean_va:.1f} V**, Phase B: **{mean_vb:.1f} V**, Phase C: **{mean_vc:.1f} V**.\n"
                            f"*   Raw Grid Frequency averaged **{mean_freq:.3f} Hz**.\n\n"
                            f"**Verdict**: An external grid utility sag, phase imbalance, or grid frequency fluctuation forced the inverter's safety relays to restrict power injection."
                        )
                        
                        fig_ev_str = go.Figure()
                        fig_ev_str.add_trace(go.Scatter(x=event_rows["device_time"], y=event_rows["grid_voltage_a"], mode="lines+markers", name="Phase A Voltage (V)"))
                        fig_ev_str.add_trace(go.Scatter(x=event_rows["device_time"], y=event_rows["grid_voltage_b"], mode="lines+markers", name="Phase B Voltage (V)"))
                        fig_ev_str.add_trace(go.Scatter(x=event_rows["device_time"], y=event_rows["grid_voltage_c"], mode="lines+markers", name="Phase C Voltage (V)"))
                        fig_ev_str.update_layout(title="Raw Phase Voltages during Event Window", xaxis_title="Time", yaxis_title="Voltage (V)")
                        
                    else: # Inverter Offline
                        mean_irr = event_rows["irradiance_wm2"].mean()
                        mean_ac = event_rows["ac_power_kw"].mean()
                        mean_vdc = np.mean([event_rows[f"dc_voltage_{i}"].mean() for i in range(1, 7)])
                        
                        diagnosis_text = (
                            f"**Conclusion**: Inverter completely offline.\n\n"
                            f"**Raw Telemetry Evidence**:\n"
                            f"*   Inverter raw active AC power was **{mean_ac:.2f} kW**.\n"
                            f"*   Raw solar irradiance averaged **{mean_irr:.1f} W/m²**.\n"
                            f"*   Raw DC input open-circuit voltage remained high at **{mean_vdc:.1f} V**.\n\n"
                            f"**Verdict**: The presence of high DC voltage alongside zero AC power indicates a localized AC breaker trip, contactor open sag, or communication loss, rather than a solar resource outage."
                        )
                        
                        fig_ev_str = go.Figure()
                        fig_ev_str.add_trace(go.Scatter(x=event_rows["device_time"], y=event_rows["irradiance_wm2"], mode="lines+markers", name="Irradiance (W/m²)", line=dict(color="#ff7f0e")))
                        fig_ev_str.add_trace(go.Scatter(x=event_rows["device_time"], y=event_rows["ac_power_kw"], mode="lines+markers", name="AC Power (kW)", line=dict(color="#1f77b4")))
                        fig_ev_str.update_layout(title="Raw Solar Irradiance vs AC Power during Event Window", xaxis_title="Time")
                        
                    # Display Diagnostic Evidence
                    st.info(diagnosis_text)
                    st.plotly_chart(fig_ev_str, use_container_width=True)
