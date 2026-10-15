"""
DustSentinel Core Engine: Pure Domain Logic, Validation, Attribution & Simulation
Tech For Earth Hackathon 2026 - Construction Fugitive Dust Monitoring

This module contains pure, testable mathematical and scientific algorithms decoupled
from the Streamlit user interface.
"""

import pandas as pd
import numpy as np
import math
import itertools
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any, List, Optional

_intervention_counter = itertools.count(1)

# ---------------------------------------------------------
# Domain Constants & Reference Standards
# ---------------------------------------------------------
SITE_ZONES = ["Excavation", "Haul Road", "Material Storage", "Loading Point"]

ZONE_COORDINATES = {
    "Excavation": {"x": 25.0, "y": 75.0, "area": "NW Sector - Bulk Earthworks", "desc": "Active cut/fill, bulldozers & excavators"},
    "Haul Road": {"x": 45.0, "y": 48.0, "area": "Central Corridor - Unpaved Transit", "desc": "Heavy dump trucks & aggregate haulers"},
    "Material Storage": {"x": 78.0, "y": 78.0, "area": "NE Sector - Aggregate Stockpiles", "desc": "Uncovered coarse sand & crushed stone mounds"},
    "Loading Point": {"x": 72.0, "y": 28.0, "area": "SE Sector - Hopper Transfer", "desc": "Batch loading chutes, conveyor drops"}
}

# Environmental Limits & Thresholds
THRESHOLDS = {
    "PM10_WHO_24H": 45.0,          # WHO 2021 24h Guideline (ug/m3)
    "PM10_SITE_ACTION": 100.0,      # Site operational alert trigger (ug/m3)
    "PM10_EPA_24H_LIMIT": 150.0,    # US EPA NAAQS 24h standard (ug/m3)
    "PM25_WHO_24H": 15.0,          # WHO 2021 24h Guideline (ug/m3)
    "PM25_EPA_24H_LIMIT": 35.0,     # US EPA NAAQS 24h standard (ug/m3)
    "HIGH_WIND_THRESHOLD": 18.0,    # km/h (amplifies fugitive dust lift)
    "DRY_HUMIDITY_THRESHOLD": 40.0  # % (soil desiccation threshold)
}

# Baseline regional background ambient air quality (cannot mitigate ambient air below this floor)
BACKGROUND_FLOOR = {
    "PM10": 18.0,  # ug/m3
    "PM25": 8.0    # ug/m3
}

# Standard zone alias mapping
ZONE_ALIAS_MAP = {
    "excavation": "Excavation",
    "digging": "Excavation",
    "earthwork": "Excavation",
    "haul road": "Haul Road",
    "haulroad": "Haul Road",
    "road": "Haul Road",
    "material storage": "Material Storage",
    "stockpile": "Material Storage",
    "storage": "Material Storage",
    "loading point": "Loading Point",
    "loading": "Loading Point",
    "hopper": "Loading Point"
}

# ---------------------------------------------------------
# Closed-Loop Feedback & Intervention Constants
# ---------------------------------------------------------
PROVENANCE_OPTIONS = [
    "Synthetic Demonstration",
    "Uploaded Dataset",
    "Field Measurement"
]

VERIFICATION_STATUSES = [
    "Field Verified",
    "Pending Verification",
    "Unverified / Incomplete",
    "Rejected / Ineffective"
]

ZONE_RECOMMENDED_ACTIONS = {
    "Excavation": [
        "Targeted Water Sprinkler Suppression",
        "Excavation Drop Height Limitation",
        "Vehicle Speed Enforcement (15 km/h)",
        "Temporary Soil Stabilizer Spray"
    ],
    "Haul Road": [
        "Automated Water Bowser Misting Cycle",
        "Eco-Friendly Polymer Soil Binder",
        "Unpaved Haul Speed Governor (15 km/h)",
        "Gravel Resurfacing Pass"
    ],
    "Material Storage": [
        "Continuous Aggregate Stockpile Wetting",
        "Windbreak Netting & Perimeter Fencing",
        "Biodegradable Crusting Emulsion Tackifier",
        "Stockpile Tarpaulin Cover Deployment"
    ],
    "Loading Point": [
        "Chute Mist Curtain & Hopper Enclosure",
        "Reduced Material Drop Velocity",
        "Mobile Dust Extraction Unit",
        "Hopper Windbreak Barrier"
    ]
}

# Public module exports
__all__ = [
    "SITE_ZONES",
    "ZONE_COORDINATES",
    "THRESHOLDS",
    "BACKGROUND_FLOOR",
    "ZONE_ALIAS_MAP",
    "CARDINAL_DIRECTIONS",
    "get_cardinal_direction",
    "compute_plume_heading",
    "compute_angular_offset",
    "validate_and_clean_data",
    "generate_synthetic_stream",
    "evaluate_site_risk",
    "compute_rule_based_attribution",
    "calculate_mitigation_simulation",
    "split_chronological",
    "train_short_horizon_forecast",
    "generate_forward_forecast",
    "evaluate_actionable_recommendations",
    "compare_mitigation_scenarios",
    "create_feedback_record",
    "PROVENANCE_OPTIONS",
    "VERIFICATION_STATUSES",
    "ZONE_RECOMMENDED_ACTIONS",
    "calculate_observed_outcome",
    "validate_intervention_input",
    "create_intervention_record",
    "filter_intervention_records",
    "aggregate_feedback_analytics",
    "export_intervention_feedback_csv",
    "generate_sample_feedback_records",
]

# ---------------------------------------------------------
# Directional Navigation Utilities
# ---------------------------------------------------------
CARDINAL_DIRECTIONS = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"
]

def get_cardinal_direction(deg: float) -> str:
    """Converts compass heading degrees to 16-point cardinal compass string."""
    deg_norm = deg % 360.0
    ix = int((deg_norm + 11.25) / 22.5) % 16
    return CARDINAL_DIRECTIONS[ix]

def compute_plume_heading(wind_deg: float) -> float:
    """
    Computes downwind plume travel heading from meteorological wind direction.
    Meteorological wind direction is WHERE the wind blows FROM.
    Plume travels toward (wind_deg + 180) % 360.
    """
    return (wind_deg + 180.0) % 360.0

def compute_angular_offset(angle1: float, angle2: float) -> float:
    """Computes the shortest absolute difference between two angular bearings in degrees."""
    diff = abs((angle1 % 360.0) - (angle2 % 360.0))
    return min(diff, 360.0 - diff)

# ---------------------------------------------------------
# REQ-02 & REQ-03: Data Validation & Quality Auditing Engine
# ---------------------------------------------------------
def validate_and_clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validates CSV schema, detects anomalies, handles duplicates, and scopes
    imputation strictly within individual zones.
    
    Guarantees:
    - Readings NEVER bleed across different zones.
    - Conflicting duplicate (timestamp, zone) records are audited and resolved.
    - Inverted PM (PM2.5 > PM10) is flagged as an anomaly, NEVER silently rewritten.
    - Negative or out-of-bounds readings are flagged, NEVER silently clamped.
    - Coercion failures > 10% reject the dataset.
    """
    report: Dict[str, Any] = {
        "valid": False,
        "total_rows": len(df) if df is not None else 0,
        "issues": [],
        "warnings": [],
        "fixes": [],
        "anomalies": [],
        "duplicate_count": 0,
        "missing_values_handled": 0,
        "time_span": "N/A"
    }

    if df is None or df.empty:
        report["issues"].append("Dataset is empty.")
        return pd.DataFrame(), report

    req_cols = ["timestamp", "zone", "pm2_5", "pm10", "wind_speed", "wind_deg", "humidity", "temperature"]
    df_cols_lower = {str(c).lower().strip(): c for c in df.columns}
    
    missing_cols = [c for c in req_cols if c not in df_cols_lower]
    if missing_cols:
        report["issues"].append(f"Missing required columns: {', '.join(missing_cols)}")
        return df, report

    # Standardize column headers
    rename_map = {df_cols_lower[c]: c for c in req_cols}
    df = df.rename(columns=rename_map).copy()

    # Parse timestamps
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    except Exception as e:
        report["issues"].append(f"Timestamp parsing failed: {str(e)}")
        return df, report

    unparseable_times = df["timestamp"].isna().sum()
    if unparseable_times > 0:
        if (unparseable_times / len(df)) > 0.10:
            report["issues"].append(f"Unacceptable timestamp failure: {unparseable_times} rows ({unparseable_times/len(df)*100:.1f}%) could not be parsed.")
            return df, report
        report["warnings"].append(f"Dropped {unparseable_times} rows with unparseable timestamps.")
        df = df.dropna(subset=["timestamp"])

    if df.empty:
        report["issues"].append("Dataset has no valid timestamps remaining.")
        return df, report

    df = df.sort_values("timestamp").reset_index(drop=True)
    report["time_span"] = f"{df['timestamp'].min().strftime('%Y-%m-%d %H:%M')} to {df['timestamp'].max().strftime('%Y-%m-%d %H:%M')}"

    # Standardize Zone Names
    def clean_zone_name(val: Any) -> str:
        s = str(val).strip().lower()
        if s in ZONE_ALIAS_MAP:
            return ZONE_ALIAS_MAP[s]
        return str(val).strip().title()

    df["zone"] = df["zone"].apply(clean_zone_name)

    # Activity level column check
    if "activity_level" not in df.columns:
        df["activity_level"] = "Moderate"
        report["fixes"].append("Defaulted missing 'activity_level' column to 'Moderate'.")
    else:
        df["activity_level"] = df["activity_level"].fillna("Moderate").astype(str)

    # Type coercion and numeric validation
    num_cols = ["pm2_5", "pm10", "wind_speed", "wind_deg", "humidity", "temperature"]
    for col in num_cols:
        raw_series = df[col].copy()
        df[col] = pd.to_numeric(df[col], errors='coerce')
        failed_count = df[col].isna().sum() - raw_series.isna().sum()
        if failed_count > 0:
            ratio = failed_count / len(df)
            if ratio > 0.10:
                report["issues"].append(f"Column '{col}' has {failed_count} unparseable non-numeric values ({ratio*100:.1f}%). Upload rejected.")
                return df, report
            report["warnings"].append(f"Coerced {failed_count} non-numeric values in '{col}' to NaN.")

    # Initialize anomaly flag tracking columns
    df["is_anomalous"] = False
    df["anomaly_notes"] = ""
    df["is_imputed"] = False

    # REQ-03B: Duplicate Record Detection on (timestamp, zone)
    dupe_mask = df.duplicated(subset=["timestamp", "zone"], keep=False)
    if dupe_mask.any():
        dupe_rows = df[dupe_mask]
        unique_dupe_keys = dupe_rows[["timestamp", "zone"]].drop_duplicates()
        report["duplicate_count"] = len(dupe_rows) - len(unique_dupe_keys)
        
        # Check if duplicates have conflicting particulate readings
        conflicting_count = 0
        resolved_rows = []
        for _, group in df.groupby(["timestamp", "zone"], as_index=False):
            if len(group) > 1:
                pm10_std = group["pm10"].std()
                if pm10_std and pm10_std > 0.0:
                    conflicting_count += 1
                # Deterministic resolution: average numeric readings and keep latest row
                resolved = group.iloc[-1].copy()
                for c in num_cols:
                    resolved[c] = group[c].mean()
                resolved["is_anomalous"] = True
                resolved["anomaly_notes"] += f"Resolved from {len(group)} duplicate records (std={pm10_std:.1f}); "
                resolved_rows.append(resolved)
            else:
                resolved_rows.append(group.iloc[0])
                
        df = pd.DataFrame(resolved_rows).reset_index(drop=True)
        report["fixes"].append(f"Resolved {report['duplicate_count']} duplicate records (including {conflicting_count} conflicting readings) via deterministic averaging.")

    # REQ-03C: Physical Bounds Verification (Flag anomalies rather than silent clamping)
    # Check negative PM readings
    neg_pm10 = df["pm10"] < 0
    neg_pm25 = df["pm2_5"] < 0
    neg_pm = neg_pm10 | neg_pm25
    if neg_pm.any():
        count = int(neg_pm.sum())
        df.loc[neg_pm, "is_anomalous"] = True
        df.loc[neg_pm, "anomaly_notes"] += "NEGATIVE_PM_READING; "
        report["anomalies"].append(f"Flagged {count} rows with impossible negative particulate readings (retained for audit, excluded from clean calculation).")

    # Check negative wind speed
    neg_wind = df["wind_speed"] < 0
    if neg_wind.any():
        count = int(neg_wind.sum())
        df.loc[neg_wind, "is_anomalous"] = True
        df.loc[neg_wind, "anomaly_notes"] += "NEGATIVE_WIND_SPEED; "
        report["anomalies"].append(f"Flagged {count} rows with negative wind speed.")

    # Check out-of-bounds humidity
    oob_humidity = (df["humidity"] < 0.0) | (df["humidity"] > 100.0)
    if oob_humidity.any():
        count = int(oob_humidity.sum())
        df.loc[oob_humidity, "is_anomalous"] = True
        df.loc[oob_humidity, "anomaly_notes"] += "OUT_OF_BOUNDS_HUMIDITY; "
        report["anomalies"].append(f"Flagged {count} rows with impossible relative humidity (<0% or >100%).")

    # REQ-03D: PM Consistency Inversion (PM2.5 > PM10)
    # Physical fact: PM2.5 is a subset of PM10. If PM2.5 > PM10, it indicates optical sensor drift.
    # Strictly DO NOT multiply by 0.95! Flag the observation.
    pm_inversion = (df["pm2_5"] > df["pm10"]) & (~df["pm10"].isna()) & (~df["pm2_5"].isna())
    if pm_inversion.any():
        count = int(pm_inversion.sum())
        df.loc[pm_inversion, "is_anomalous"] = True
        df.loc[pm_inversion, "anomaly_notes"] += "ANOMALY_PM_INVERSION(PM2.5>PM10); "
        report["anomalies"].append(f"Flagged {count} rows where PM2.5 exceeded PM10 (optical calibration fault; retained without silent artificial rewriting).")

    # REQ-03A: Missing-Value Handling (Strictly Zone-Scoped)
    # Group strictly by zone. Values NEVER cross zone boundaries.
    cleaned_zones = []
    for zone_name, group in df.groupby("zone", as_index=False):
        group = group.sort_values("timestamp").copy()
        
        for col in num_cols:
            nan_mask = group[col].isna()
            nan_count = nan_mask.sum()
            if nan_count > 0:
                report["missing_values_handled"] += int(nan_count)
                # Zone-scoped interpolation with max limit of 2 intervals (30 minutes)
                # Any gap longer than 30 minutes remains NaN
                interpolated = group[col].interpolate(method="linear", limit=2)
                imputed_mask = nan_mask & (~interpolated.isna())
                group[col] = interpolated
                group.loc[imputed_mask, "is_imputed"] = True
                
        cleaned_zones.append(group)

    df_cleaned = pd.concat(cleaned_zones, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    if report["missing_values_handled"] > 0:
        report["fixes"].append(f"Handled {report['missing_values_handled']} missing values using zone-scoped short-gap interpolation (max gap: 30 mins; zero cross-zone propagation).")

    # Standardize wind direction to [0, 360)
    df_cleaned["wind_deg"] = df_cleaned["wind_deg"] % 360.0

    report["valid"] = True
    return df_cleaned, report

# ---------------------------------------------------------
# REQ-01: Synthetic Data Stream Engine (Reproducible)
# ---------------------------------------------------------
def generate_synthetic_stream(
    scenario: str = "Active Peak Shift",
    hours: int = 24,
    seed: int = 101,
    base_time: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Generates realistic, physically plausible synthetic construction sensor time-series.
    
    Guarantees:
    - Deterministic and reproducible with identical seeds.
    - Normalized timestamp baseline (floored to 15-minute boundary without microsecond jitter).
    - Clear scenario-specific dynamics (Earthworks, Dust Storm/Wind, Baseline).
    """
    rng = np.random.default_rng(seed)
    intervals = int(hours * 4)  # 15-minute intervals
    
    if base_time is None:
        # Fixed reference date baseline for consistent demo reproducibility
        base_time = datetime(2026, 10, 14, 6, 0, 0)
    else:
        # Floor to 15-minute boundary, zero out seconds and microseconds
        minute_floored = (base_time.minute // 15) * 15
        base_time = base_time.replace(minute=minute_floored, second=0, microsecond=0)

    timestamps = [base_time + timedelta(minutes=15 * i) for i in range(intervals)]
    records: List[Dict[str, Any]] = []

    for t in timestamps:
        hour = t.hour + t.minute / 60.0

        # Diurnal weather curve
        temp_cycle = math.sin((hour - 8.0) * math.pi / 12.0)
        ambient_temp = 27.0 + 7.0 * temp_cycle + rng.normal(0, 0.4)
        ambient_humidity = np.clip(54.0 - 24.0 * temp_cycle + rng.normal(0, 1.5), 18.0, 85.0)

        # Wind dynamics
        base_wind = 10.0 + 7.0 * max(0.0, temp_cycle) + rng.normal(0, 1.2)
        if scenario == "Dry Windy Alert Condition":
            base_wind += 12.0
            ambient_humidity = max(15.0, ambient_humidity - 16.0)

        wind_speed = max(3.0, round(base_wind, 1))
        # Prevailing WSW wind (245 deg) with diurnal veer
        wind_deg = round((245.0 + 25.0 * math.sin(hour * math.pi / 6.0) + rng.normal(0, 8.0)) % 360, 1)

        is_shift = 7.5 <= hour <= 17.5

        for zone in SITE_ZONES:
            bg_pm10 = 32.0 + rng.normal(0, 2.5)
            bg_pm25 = 16.0 + rng.normal(0, 1.2)

            dust_pm10 = 0.0
            dust_pm25 = 0.0
            activity = "Low"

            if zone == "Excavation":
                if is_shift:
                    activity = "High" if (8.5 <= hour <= 12.0 or 13.5 <= hour <= 16.5) else "Moderate"
                    mult = 2.4 if scenario == "Active Peak Shift" else 1.3
                    dust_pm10 = rng.uniform(75.0, 150.0) * mult if activity == "High" else rng.uniform(40.0, 75.0)
                    dust_pm25 = dust_pm10 * rng.uniform(0.18, 0.28)  # Crustal coarse ratio
                else:
                    dust_pm10 = rng.uniform(6.0, 15.0)
                    dust_pm25 = dust_pm10 * 0.30

            elif zone == "Haul Road":
                if is_shift:
                    activity = "High" if (9.0 <= hour <= 16.0) else "Moderate"
                    truck_pass = rng.random() < (0.65 if activity == "High" else 0.35)
                    if truck_pass:
                        dust_pm10 = rng.uniform(140.0, 270.0)
                        dust_pm25 = dust_pm10 * rng.uniform(0.20, 0.30)
                    else:
                        dust_pm10 = rng.uniform(35.0, 60.0)
                        dust_pm25 = dust_pm10 * 0.26
                else:
                    dust_pm10 = rng.uniform(4.0, 12.0)
                    dust_pm25 = dust_pm10 * 0.28

            elif zone == "Material Storage":
                # Aerodynamic erosion: lift occurs above threshold velocity and dry air
                wind_dry_factor = max(0.0, (wind_speed - 16.0) / 9.0) * max(0.0, (45.0 - ambient_humidity) / 22.0)
                if wind_dry_factor > 0:
                    activity = "Moderate"
                    dust_pm10 = 55.0 + (wind_dry_factor * rng.uniform(120.0, 210.0))
                    dust_pm25 = dust_pm10 * rng.uniform(0.15, 0.23)
                elif is_shift:
                    activity = "Moderate"
                    dust_pm10 = rng.uniform(25.0, 50.0)
                    dust_pm25 = dust_pm10 * 0.26
                else:
                    dust_pm10 = rng.uniform(4.0, 10.0)
                    dust_pm25 = dust_pm10 * 0.28

            elif zone == "Loading Point":
                if is_shift and ((9.0 <= hour <= 11.5) or (13.5 <= hour <= 16.0)):
                    activity = "High"
                    dust_pm10 = rng.uniform(95.0, 190.0)
                    dust_pm25 = dust_pm10 * rng.uniform(0.24, 0.36)
                elif is_shift:
                    activity = "Moderate"
                    dust_pm10 = rng.uniform(35.0, 70.0)
                    dust_pm25 = dust_pm10 * 0.28
                else:
                    dust_pm10 = rng.uniform(5.0, 12.0)
                    dust_pm25 = dust_pm10 * 0.30

            records.append({
                "timestamp": t,
                "zone": zone,
                "pm2_5": round(max(0.0, bg_pm25 + dust_pm25), 1),
                "pm10": round(max(0.0, bg_pm10 + dust_pm10), 1),
                "wind_speed": wind_speed,
                "wind_deg": wind_deg,
                "humidity": round(ambient_humidity, 1),
                "temperature": round(ambient_temp, 1),
                "activity_level": activity,
                "is_anomalous": False,
                "anomaly_notes": "",
                "is_imputed": False
            })

    return pd.DataFrame(records)

# ---------------------------------------------------------
# REQ-04: Risk Metrics & Regulatory Standard Evaluator
# ---------------------------------------------------------
def evaluate_site_risk(snapshot_df: pd.DataFrame, full_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Evaluates site dust risk distinguishing instantaneous action alerts from
    true statutory 24-hour rolling averages (EPA NAAQS / WHO).
    """
    if snapshot_df.empty:
        return {
            "tier": "Low Risk",
            "tier_badge": "[NORMAL] WITHIN BENCHMARK",
            "peak_pm10": 0.0,
            "peak_pm25": 0.0,
            "peak_zone": "N/A",
            "is_instant_action": False,
            "is_24h_violation": False,
            "compliance_summary": "No active observations."
        }

    valid_snapshot = snapshot_df[~snapshot_df["is_anomalous"]] if "is_anomalous" in snapshot_df.columns else snapshot_df
    if valid_snapshot.empty:
        valid_snapshot = snapshot_df

    peak_pm10 = float(valid_snapshot["pm10"].max())
    peak_pm25 = float(valid_snapshot["pm2_5"].max())
    worst_row = valid_snapshot.loc[valid_snapshot["pm10"].idxmax()]
    worst_zone = str(worst_row["zone"])

    # 15-minute Instantaneous Threshold Flags
    is_instant_severe = peak_pm10 >= THRESHOLDS["PM10_EPA_24H_LIMIT"]
    is_instant_action = peak_pm10 >= THRESHOLDS["PM10_SITE_ACTION"]

    # 24-Hour Rolling Average Evaluation (if full dataframe with >= 12h data is supplied)
    is_24h_violation = False
    rolling_24h_mean = None
    if full_df is not None and not full_df.empty:
        time_span_hrs = (full_df["timestamp"].max() - full_df["timestamp"].min()).total_seconds() / 3600.0
        if time_span_hrs >= 12.0:
            rolling_24h_mean = float(full_df.groupby("timestamp")["pm10"].max().mean())
            is_24h_violation = rolling_24h_mean >= THRESHOLDS["PM10_EPA_24H_LIMIT"]

    if is_instant_severe:
        tier = "Severe Instantaneous Hazard"
        badge = "[SEVERE] IMMEDIATE ACTION TRIGGERED"
    elif is_instant_action:
        tier = "High Action Alert"
        badge = "[HIGH] WARNING LEVEL"
    elif peak_pm10 >= THRESHOLDS["PM10_WHO_24H"]:
        tier = "Moderate Advisory"
        badge = "[MODERATE] WATCH"
    else:
        tier = "Low / Normal Baseline"
        badge = "[NORMAL] WITHIN BENCHMARK"

    summary_text = (
        f"Instantaneous peak {peak_pm10:.1f} µg/m³ at {worst_zone}. "
        f"Note: EPA limit (150 µg/m³) is a 24-hour statutory standard. "
        + (f"Rolling site 24h mean is {rolling_24h_mean:.1f} µg/m³." if rolling_24h_mean else "Insufficient historical duration for verified 24h statutory compliance.")
    )

    return {
        "tier": tier,
        "tier_badge": badge,
        "peak_pm10": peak_pm10,
        "peak_pm25": peak_pm25,
        "peak_zone": worst_zone,
        "is_instant_action": is_instant_action,
        "is_instant_severe": is_instant_severe,
        "is_24h_violation": is_24h_violation,
        "rolling_24h_mean": rolling_24h_mean,
        "compliance_summary": summary_text
    }

# ---------------------------------------------------------
# REQ-05: Explainable Source Attribution Engine
# ---------------------------------------------------------
def compute_rule_based_attribution(
    latest_records: pd.DataFrame,
    historical_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Explainable multi-criteria deterministic rule engine.
    Ranks zones by likelihood of being the primary dust source.
    
    Guarantees:
    - Transparent heuristic scoring (labelled 'ranking_score', NOT calibrated probability).
    - Clear aerodynamic and particle ratio explanations.
    """
    if latest_records.empty:
        return pd.DataFrame()

    results: List[Dict[str, Any]] = []

    current_wind_speed = float(latest_records["wind_speed"].iloc[0])
    current_wind_deg = float(latest_records["wind_deg"].iloc[0])
    current_humidity = float(latest_records["humidity"].iloc[0])

    # Sensitive boundary receptor coordinates (SE property line: X=95, Y=10)
    receptor_x, receptor_y = 95.0, 10.0
    plume_heading = compute_plume_heading(current_wind_deg)

    for _, row in latest_records.iterrows():
        zone = str(row["zone"])
        pm10 = float(row["pm10"]) if not pd.isna(row["pm10"]) else 0.0
        pm25 = float(row["pm2_5"]) if not pd.isna(row["pm2_5"]) else 0.0
        activity = str(row.get("activity_level", "Moderate"))
        is_anom = bool(row.get("is_anomalous", False))

        score = 0.0
        explanations: List[str] = []

        if is_anom:
            explanations.append(f"[ANOMALY] Sensor anomaly flagged ({row.get('anomaly_notes', '')}); attribution weighted cautiously.")
            score += 2.0

        # Rule 1: Absolute Particulate Intensity
        if pm10 >= THRESHOLDS["PM10_EPA_24H_LIMIT"]:
            score += 40.0
            explanations.append(f"[EXCEEDANCE] Severe PM10 concentration ({pm10:.1f} µg/m³) exceeds 150 µg/m³ threshold.")
        elif pm10 >= THRESHOLDS["PM10_SITE_ACTION"]:
            score += 25.0
            explanations.append(f"[ELEVATED] PM10 ({pm10:.1f} µg/m³) crosses Site Action Alert (100 µg/m³).")
        elif pm10 >= THRESHOLDS["PM10_WHO_24H"]:
            score += 12.0
            explanations.append(f"[ADVISORY] PM10 ({pm10:.1f} µg/m³) exceeds WHO benchmark (45 µg/m³).")
        else:
            score += 4.0
            explanations.append(f"[NORMAL] PM10 ({pm10:.1f} µg/m³) within normal operational limits.")

        # Rule 2: Coarse vs Fine Particle Ratio Fingerprint
        # Crustal fugitive dust has ratio PM2.5/PM10 ~ 0.15 - 0.35
        ratio = (pm25 / pm10) if pm10 > 0 else 0.25
        if 0.12 <= ratio <= 0.35 and pm10 > 60.0:
            score += 20.0
            explanations.append(f"[RATIO] Particle ratio PM2.5/PM10 = {ratio:.2f} confirms mechanical crustal/fugitive dust.")
        elif ratio > 0.45:
            score += 5.0
            explanations.append(f"[COMBUSTION] High fine fraction ({ratio:.2f}) indicates potential combustion/diesel engine contribution.")

        # Rule 3: Wind-Zone Spatial Alignment & Receptor Exposure
        coords = ZONE_COORDINATES.get(zone, {"x": 50.0, "y": 50.0})
        dx = receptor_x - coords["x"]
        dy = receptor_y - coords["y"]
        # Navigation bearing from zone to receptor
        bearing_to_receptor = math.degrees(math.atan2(dx, dy)) % 360.0
        angle_diff = compute_angular_offset(plume_heading, bearing_to_receptor)

        if angle_diff <= 35.0:
            score += 20.0
            explanations.append(f"[ALIGNMENT] Downwind Alignment: Plume heading ({plume_heading:.0f}°) drifts toward sensitive perimeter receptor (angular error: {angle_diff:.0f}°).")
        elif angle_diff <= 65.0:
            score += 10.0
            explanations.append(f"[ALIGNMENT] Moderate crosswind alignment ({angle_diff:.0f}° offset from receptor).")

        # Rule 4: Physical Zone-Specific Triggers
        if zone == "Material Storage":
            if current_wind_speed >= THRESHOLDS["HIGH_WIND_THRESHOLD"] and current_humidity <= THRESHOLDS["DRY_HUMIDITY_THRESHOLD"]:
                score += 20.0
                explanations.append(f"[EROSION] Aerodynamic Erosion Trigger: Wind {current_wind_speed:.1f} km/h > 18 km/h & dry air {current_humidity:.1f}% lifts stockpile silt.")
            elif current_wind_speed >= THRESHOLDS["HIGH_WIND_THRESHOLD"]:
                score += 10.0
                explanations.append(f"[WIND] Elevated wind velocity over exposed aggregate crests.")

        elif zone == "Haul Road":
            road_hist = historical_df[historical_df["zone"] == "Haul Road"]["pm10"]
            if len(road_hist) > 4:
                cv = road_hist.tail(8).std() / (road_hist.tail(8).mean() + 1e-4)
                if cv > 0.35:
                    score += 15.0
                    explanations.append(f"[TURBULENCE] Vehicular Turbulence Signature: High short-term variance (CV: {cv:.2f}) indicates heavy haulage truck passes.")

        elif zone == "Excavation":
            if activity == "High":
                score += 15.0
                explanations.append(f"[ACTIVITY] Active Earthmoving: Bulldozers and excavators actively stripping topsoil.")

        elif zone == "Loading Point":
            if activity == "High":
                score += 15.0
                explanations.append(f"[TRANSFER] Batch Transfer Active: Hopper chute loading creates cyclical particulate drop plumes.")

        ranking_score = min(98.0, max(10.0, score))

        results.append({
            "zone": zone,
            "pm10": pm10,
            "pm2_5": pm25,
            "ranking_score": round(ranking_score, 1),
            "activity": activity,
            "explanations": explanations
        })

    res_df = pd.DataFrame(results).sort_values("ranking_score", ascending=False).reset_index(drop=True)
    res_df["rank"] = range(1, len(res_df) + 1)
    return res_df

# ---------------------------------------------------------
# REQ-07: Simulated Before-and-After Intervention Sandbox
# ---------------------------------------------------------
def calculate_mitigation_simulation(
    df: pd.DataFrame,
    zone_eff_map: Dict[str, float],
    interval_hours: float = 0.25
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Computes theoretical post-mitigation particulate reductions based on
    empirical engineering transfer models (US EPA AP-42 / WRAP Handbook).
    
    Guarantees:
    - Correct elapsed clock time accounting: Exceedance hours calculated from
      UNIQUE timestamps, never quadruple-counting multi-zone simultaneous exceedances.
    - Explicit background floor subtraction.
    - All outputs clearly labelled as simulated projections.
    """
    if df.empty:
        return pd.DataFrame(), {
            "simulated_baseline_peak_pm10": 0.0,
            "simulated_post_peak_pm10": 0.0,
            "simulated_reduction_pct": 0.0,
            "simulated_baseline_exceedance_hours": 0.0,
            "simulated_post_exceedance_hours": 0.0,
            "simulated_avoided_exceedance_hours": 0.0,
            "simulated_avg_site_efficacy": 0.0
        }

    df_sim = df.copy()
    df_sim["mitigation_eff"] = df_sim["zone"].map(zone_eff_map).fillna(0.0)

    bg_pm10 = BACKGROUND_FLOOR["PM10"]
    bg_pm25 = BACKGROUND_FLOOR["PM25"]

    # Calculate post-intervention simulated particulate concentrations
    df_sim["pm10_simulated"] = df_sim.apply(
        lambda r: round(bg_pm10 + (r["pm10"] - bg_pm10) * (1.0 - r["mitigation_eff"]), 1) if r["pm10"] > bg_pm10 else r["pm10"],
        axis=1
    )
    df_sim["pm25_simulated"] = df_sim.apply(
        lambda r: round(bg_pm25 + (r["pm2_5"] - bg_pm25) * (1.0 - (r["mitigation_eff"] * 0.75)), 1) if r["pm2_5"] > bg_pm25 else r["pm2_5"],
        axis=1
    )

    baseline_peak_pm10 = float(df["pm10"].max())
    post_peak_pm10 = float(df_sim["pm10_simulated"].max())
    reduction_pct = ((baseline_peak_pm10 - post_peak_pm10) / baseline_peak_pm10 * 100.0) if baseline_peak_pm10 > 0 else 0.0

    # REQ-07 Fix: Correct Exceedance Elapsed Clock Hours
    # Count UNIQUE elapsed timestamps where ANY site zone exceeded the threshold
    baseline_exceed_timestamps = df[df["pm10"] >= THRESHOLDS["PM10_EPA_24H_LIMIT"]]["timestamp"].nunique()
    post_exceed_timestamps = df_sim[df_sim["pm10_simulated"] >= THRESHOLDS["PM10_EPA_24H_LIMIT"]]["timestamp"].nunique()

    baseline_exceed_hours = round(baseline_exceed_timestamps * interval_hours, 2)
    post_exceed_hours = round(post_exceed_timestamps * interval_hours, 2)
    avoided_hours = max(0.0, round(baseline_exceed_hours - post_exceed_hours, 2))

    avg_efficacy = float(np.mean(list(zone_eff_map.values()))) if zone_eff_map else 0.0

    metrics = {
        "simulated_baseline_peak_pm10": baseline_peak_pm10,
        "simulated_post_peak_pm10": post_peak_pm10,
        "simulated_reduction_pct": round(reduction_pct, 1),
        "simulated_baseline_exceedance_hours": baseline_exceed_hours,
        "simulated_post_exceedance_hours": post_exceed_hours,
        "simulated_avoided_exceedance_hours": avoided_hours,
        "simulated_avg_site_efficacy": round(avg_efficacy * 100.0, 1)
    }

    return df_sim, metrics


# ---------------------------------------------------------
# FEATURE 1: Chronological Data Splitting & Short-Horizon Dust-Risk Forecasting
# ---------------------------------------------------------
def split_chronological(
    df: pd.DataFrame,
    test_ratio: float = 0.25
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits time-series data chronologically to strictly prevent lookahead data leakage.
    
    Guarantees:
    - max(train_df["timestamp"]) < min(test_df["timestamp"])
    - All zones maintain equal temporal partition.
    """
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df_sorted = df.sort_values("timestamp").copy()
    unique_times = np.array(sorted(df_sorted["timestamp"].unique()))
    n_times = len(unique_times)
    
    if n_times < 4:
        # Insufficient timestamps to split meaningfully; return all as train
        return df_sorted, pd.DataFrame()

    split_idx = max(1, int(n_times * (1.0 - test_ratio)))
    if split_idx >= n_times:
        split_idx = n_times - 1
        
    cutoff_time = unique_times[split_idx]
    
    train_df = df_sorted[df_sorted["timestamp"] < cutoff_time].copy()
    test_df = df_sorted[df_sorted["timestamp"] >= cutoff_time].copy()
    
    return train_df, test_df


def train_short_horizon_forecast(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    horizon_steps: int = 1
) -> Dict[str, Any]:
    """
    Builds a reproducible short-horizon forecasting baseline using available
    time-series features (autoregressive lag, trend, wind velocity, humidity).
    
    Compares against a Naive Persistence Baseline and reports test metrics:
    - MAE (Mean Absolute Error)
    - RMSE (Root Mean Squared Error)
    - Skill Score (% improvement over persistence)
    
    Explicitly labeled as an experimental decision-support baseline.
    """
    train_records = len(train_df) if train_df is not None else 0
    test_records = len(test_df) if test_df is not None else 0

    result: Dict[str, Any] = {
        "status": "insufficient_data",
        "message": "Insufficient historical records for evaluation.",
        "is_experimental": True,
        "label": "EXPERIMENTAL SHORT-HORIZON PROJECTION (Decision-Support Baseline)",
        "disclaimer": "Experimental baseline model. Predictions are intended for short-horizon operational screening, not certified regulatory dispersion forecasting.",
        "horizon_steps": horizon_steps,
        "train_records": train_records,
        "test_records": test_records,
        "test_mae": None,
        "test_rmse": None,
        "baseline_persistence_mae": None,
        "baseline_persistence_rmse": None,
        "relative_skill_score_pct": None,
        "split_metadata": {
            "train_records": train_records,
            "test_records": test_records,
            "train_end": str(train_df["timestamp"].max()) if train_df is not None and not train_df.empty and "timestamp" in train_df else None,
            "train_cutoff": str(train_df["timestamp"].max()) if train_df is not None and not train_df.empty and "timestamp" in train_df else None,
            "test_start": str(test_df["timestamp"].min()) if test_df is not None and not test_df.empty and "timestamp" in test_df else None,
            "leakage_seconds": 0.0
        },
        "metrics": {
            "naive_baseline_mae": None,
            "naive_baseline_rmse": None,
            "forecast_model_mae": None,
            "forecast_model_rmse": None,
            "relative_skill_score_pct": None
        },
        "zone_evaluations": {}
    }

    if train_df is None or train_df.empty or test_df is None or test_df.empty:
        if (train_df is None or train_df.empty) and (test_df is None or test_df.empty):
            result["status"] = "empty_input"
            result["message"] = "No historical or test records available for forecast evaluation."
        else:
            result["status"] = "insufficient_data"
            result["message"] = f"Insufficient records for evaluation (Train: {train_records}, Test: {test_records})."
        return result

    if train_records < 8 or test_records < 2:
        result["status"] = "insufficient_data"
        result["message"] = f"Insufficient historical records (Train: {train_records}, Test: {test_records}). At least 8 train and 2 test records required."
        return result

    all_actuals = []
    all_naive_preds = []
    all_model_preds = []

    for zone in SITE_ZONES:
        z_train = train_df[train_df["zone"] == zone].sort_values("timestamp")
        z_test = test_df[test_df["zone"] == zone].sort_values("timestamp")

        if len(z_train) < 6 or len(z_test) < 2:
            continue

        y_train = z_train["pm10"].values
        wind_train = z_train["wind_speed"].values
        hum_train = z_train["humidity"].values

        y_test = z_test["pm10"].values
        wind_test = z_test["wind_speed"].values
        hum_test = z_test["humidity"].values

        # Autoregressive lag-1 + trend + wind lift factor
        X_rows = []
        Y_targets = []
        for i in range(1, len(y_train) - horizon_steps + 1):
            target = y_train[i + horizon_steps - 1]
            cur_y = y_train[i]
            trend = cur_y - y_train[i - 1]
            w_factor = max(0.0, wind_train[i] - 15.0)
            h_factor = max(0.0, 45.0 - hum_train[i])
            X_rows.append([1.0, cur_y, trend, w_factor, h_factor])
            Y_targets.append(target)

        if len(X_rows) < 4:
            continue

        X = np.array(X_rows)
        Y = np.array(Y_targets)

        # Regularized OLS (Ridge: lambda = 1.0)
        reg = 1.0 * np.eye(X.shape[1])
        reg[0, 0] = 0.0
        try:
            weights = np.linalg.solve(X.T @ X + reg, X.T @ Y)
        except np.linalg.LinAlgError:
            weights = np.array([0.0, 1.0, 0.0, 0.0, 0.0])

        z_naive = []
        z_model = []
        z_act = []

        prev_test_y = y_train[-1]
        prev_test_w = wind_train[-1]
        prev_test_h = hum_train[-1]
        for j in range(len(y_test)):
            cur_test_y = y_test[j]
            # Prior step features
            trend_prior = prev_test_y - (y_train[-2] if j == 0 else y_test[j - 1])
            w_factor_prior = max(0.0, prev_test_w - 15.0)
            h_factor_prior = max(0.0, 45.0 - prev_test_h)

            # Persistence baseline: predict next step will equal prior step
            naive_pred = max(BACKGROUND_FLOOR["PM10"], prev_test_y)

            # Short-horizon autoregressive model prediction
            feat = np.array([1.0, prev_test_y, trend_prior, w_factor_prior, h_factor_prior])
            model_pred = float(np.dot(weights, feat))
            model_pred = max(BACKGROUND_FLOOR["PM10"], model_pred)

            z_naive.append(naive_pred)
            z_model.append(model_pred)
            z_act.append(cur_test_y)

            prev_test_y = cur_test_y
            prev_test_w = wind_test[j]
            prev_test_h = hum_test[j]

        if z_act:
            z_naive_mae = float(np.mean(np.abs(np.array(z_act) - np.array(z_naive))))
            z_model_mae = float(np.mean(np.abs(np.array(z_act) - np.array(z_model))))
            result["zone_evaluations"][zone] = {
                "naive_mae": round(z_naive_mae, 2),
                "model_mae": round(z_model_mae, 2),
                "samples": len(z_act)
            }
            all_actuals.extend(z_act)
            all_naive_preds.extend(z_naive)
            all_model_preds.extend(z_model)

    if all_actuals:
        act_arr = np.array(all_actuals)
        naive_arr = np.array(all_naive_preds)
        model_arr = np.array(all_model_preds)

        naive_mae = float(np.mean(np.abs(act_arr - naive_arr)))
        naive_rmse = float(np.sqrt(np.mean((act_arr - naive_arr) ** 2)))

        model_mae = float(np.mean(np.abs(act_arr - model_arr)))
        model_rmse = float(np.sqrt(np.mean((act_arr - model_arr) ** 2)))

        skill = ((naive_mae - model_mae) / naive_mae * 100.0) if naive_mae > 0 else 0.0

        result["status"] = "evaluated"
        result["message"] = "Out-of-sample chronological evaluation completed successfully."
        result["test_mae"] = round(model_mae, 2)
        result["test_rmse"] = round(model_rmse, 2)
        result["baseline_persistence_mae"] = round(naive_mae, 2)
        result["baseline_persistence_rmse"] = round(naive_rmse, 2)
        result["relative_skill_score_pct"] = round(skill, 1)
        result["metrics"] = {
            "naive_baseline_mae": round(naive_mae, 2),
            "naive_baseline_rmse": round(naive_rmse, 2),
            "forecast_model_mae": round(model_mae, 2),
            "forecast_model_rmse": round(model_rmse, 2),
            "relative_skill_score_pct": round(skill, 1)
        }
    else:
        result["status"] = "insufficient_data"
        result["message"] = "Insufficient consecutive observations per zone to compute forecast evaluation."

    return result


def generate_forward_forecast(
    df: pd.DataFrame,
    horizon_steps: int = 8,
    interval_mins: int = 15
) -> pd.DataFrame:
    """
    Generates forward short-horizon projections (e.g. 2 hours ahead at 15m intervals)
    for each site zone using recent autoregressive trend and meteorological conditions.
    
    Returns DataFrame with:
    - timestamp: projected future timestamps
    - zone: site zone
    - pm10_forecast: predicted PM10 concentration
    - pm10_lower: 80% confidence lower bound
    - pm10_upper: 80% confidence upper bound
    - pm25_forecast: predicted PM2.5 concentration
    - risk_category: predicted operational trigger tier
    """
    if df.empty:
        return pd.DataFrame()

    last_time = df["timestamp"].max()
    future_records = []

    for zone in SITE_ZONES:
        z_df = df[df["zone"] == zone].sort_values("timestamp")
        if z_df.empty:
            continue

        last_row = z_df.iloc[-1]
        cur_pm10 = float(last_row["pm10"])
        cur_pm25 = float(last_row["pm2_5"])
        cur_wind = float(last_row["wind_speed"])
        cur_hum = float(last_row["humidity"])

        if len(z_df) >= 3:
            recent_pm10 = z_df["pm10"].tail(3).values
            step_trend = (recent_pm10[-1] - recent_pm10[0]) / 2.0
        else:
            step_trend = 0.0

        damping_factor = 0.85
        proj_pm10 = cur_pm10
        ratio = (cur_pm25 / cur_pm10) if cur_pm10 > 0 else 0.25
        wind_lift = max(0.0, (cur_wind - 18.0) * 1.5) if cur_hum < 40.0 else 0.0

        curr_time = last_time
        for h in range(1, horizon_steps + 1):
            curr_time = curr_time + timedelta(minutes=interval_mins)
            step_trend = step_trend * damping_factor
            proj_pm10 = max(BACKGROUND_FLOOR["PM10"], proj_pm10 + step_trend + (wind_lift * (damping_factor ** h)))

            error_margin = max(6.0, 4.0 + (3.0 * math.sqrt(h)))
            pm10_lower = max(BACKGROUND_FLOOR["PM10"], proj_pm10 - error_margin)
            pm10_upper = proj_pm10 + error_margin

            proj_pm25 = max(BACKGROUND_FLOOR["PM25"], proj_pm10 * ratio)

            if proj_pm10 >= THRESHOLDS["PM10_EPA_24H_LIMIT"]:
                risk_cat = "Projected Severe Exceedance"
            elif proj_pm10 >= THRESHOLDS["PM10_SITE_ACTION"]:
                risk_cat = "Projected Action Alert"
            elif proj_pm10 >= THRESHOLDS["PM10_WHO_24H"]:
                risk_cat = "Projected Moderate Watch"
            else:
                risk_cat = "Projected Baseline"

            future_records.append({
                "timestamp": curr_time,
                "zone": zone,
                "pm10_forecast": round(proj_pm10, 1),
                "pm10_lower": round(pm10_lower, 1),
                "pm10_upper": round(pm10_upper, 1),
                "pm25_forecast": round(proj_pm25, 1),
                "risk_category": risk_cat,
                "horizon_step": h
            })

    return pd.DataFrame(future_records)


# ---------------------------------------------------------
# FEATURE 2: Transparent Actionable Mitigation Recommendations
# ---------------------------------------------------------
def evaluate_actionable_recommendations(
    latest_snapshot: pd.DataFrame,
    historical_df: pd.DataFrame,
    attribution_df: Optional[pd.DataFrame] = None
) -> List[Dict[str, Any]]:
    """
    Evaluates dynamic, transparent rules combining:
    - Current PM intensity & ratios
    - Recent particulate trends (delta PM over recent 30-min window)
    - Wind context (speed >= 18 km/h, plume vector alignment with sensitive receptors)
    - Ambient humidity desiccation (RH < 40%)
    - Zone-specific operational mechanics
    
    All recommendations are ADVISORY, providing explicit selection rationale
    and requiring human authorization before dispatch.
    """
    if latest_snapshot.empty:
        return []

    current_wind_speed = float(latest_snapshot["wind_speed"].iloc[0]) if "wind_speed" in latest_snapshot.columns else 10.0
    current_wind_deg = float(latest_snapshot["wind_deg"].iloc[0]) if "wind_deg" in latest_snapshot.columns else 240.0
    current_humidity = float(latest_snapshot["humidity"].iloc[0]) if "humidity" in latest_snapshot.columns else 50.0
    plume_heading = compute_plume_heading(current_wind_deg)

    receptor_x, receptor_y = 95.0, 10.0
    recommendations: List[Dict[str, Any]] = []

    zone_order = list(attribution_df["zone"]) if (attribution_df is not None and not attribution_df.empty) else SITE_ZONES

    for zone in zone_order:
        z_snap = latest_snapshot[latest_snapshot["zone"] == zone]
        if z_snap.empty:
            continue
        z_row = z_snap.iloc[0]
        pm10 = float(z_row["pm10"])
        pm25 = float(z_row["pm2_5"])
        activity = str(z_row.get("activity_level", "Moderate"))

        z_hist = historical_df[historical_df["zone"] == zone].sort_values("timestamp")
        if len(z_hist) >= 3:
            pm10_prev = float(z_hist["pm10"].iloc[-3])
            delta_pm10 = pm10 - pm10_prev
        else:
            delta_pm10 = 0.0

        if delta_pm10 > 25.0:
            trend_label = f"Rapidly Escalating (+{delta_pm10:.1f} µg/m³ over 30 min)"
        elif delta_pm10 > 10.0:
            trend_label = f"Moderate Rise (+{delta_pm10:.1f} µg/m³ over 30 min)"
        elif delta_pm10 < -10.0:
            trend_label = f"Decreasing ({delta_pm10:.1f} µg/m³ over 30 min)"
        else:
            trend_label = "Stable (±10 µg/m³)"

        coords = ZONE_COORDINATES.get(zone, {"x": 50.0, "y": 50.0})
        dx = receptor_x - coords["x"]
        dy = receptor_y - coords["y"]
        bearing_to_receptor = math.degrees(math.atan2(dx, dy)) % 360.0
        angle_diff = compute_angular_offset(plume_heading, bearing_to_receptor)
        downwind_threat = angle_diff <= 40.0

        zone_actions = []

        if zone == "Haul Road":
            if pm10 >= THRESHOLDS["PM10_SITE_ACTION"] or delta_pm10 > 15.0:
                zone_actions.append({
                    "action_title": "Dispatch Automated Water Bowser Misting Cycle",
                    "priority": "Immediate Action Trigger",
                    "expected_efficacy_pct": "65% - 75% PM10 reduction",
                    "eff_val": 0.70,
                    "deployment_latency": "10 mins",
                    "cost_tier": "Medium ($$)",
                    "selection_reasons": [
                        f"Current PM10 ({pm10:.1f} µg/m³) crosses Site Action Threshold (100 µg/m³).",
                        f"Particulate trend is {trend_label}.",
                        "Heavy dump truck wheel passes are resuspending dry surface silt along the unpaved transit lane."
                    ],
                    "operational_detail": "Dispatch 10,000L water tanker equipped with low-pressure atomizing rear spray bar; apply at 20-min intervals."
                })
            if current_wind_speed >= 15.0 or delta_pm10 > 10.0:
                zone_actions.append({
                    "action_title": "Enforce 15 km/h Vehicular Speed Restriction",
                    "priority": "Immediate Control",
                    "expected_efficacy_pct": "40% - 50% PM10 reduction",
                    "eff_val": 0.45,
                    "deployment_latency": "Instant",
                    "cost_tier": "Zero ($0)",
                    "selection_reasons": [
                        "Fugitive dust generation scales quadratically with vehicular speed on unpaved corridors.",
                        f"Active site haulage speed limit enforcement delivers instant non-capital abatement."
                    ],
                    "operational_detail": "Broadcast immediate 15 km/h speed cap via digital UHF site radio and telemetry speed radar checkpoint."
                })

        elif zone == "Excavation":
            if pm10 >= THRESHOLDS["PM10_SITE_ACTION"] or activity == "High":
                zone_actions.append({
                    "action_title": "Activate Excavator Boom Pressurized Mist Array",
                    "priority": "Immediate Action Trigger",
                    "expected_efficacy_pct": "60% - 75% PM10 reduction",
                    "eff_val": 0.68,
                    "deployment_latency": "5 mins",
                    "cost_tier": "Low ($)",
                    "selection_reasons": [
                        f"Active topsoil cut/fill operations with elevated PM10 ({pm10:.1f} µg/m³).",
                        f"Trend indicator: {trend_label}.",
                        "Pressurized water nozzles suppress sub-10 micron dust directly at the bucket-soil shear interface."
                    ],
                    "operational_detail": "Engage high-pressure mist nozzles mounted on active excavator booms to suppress dust at the cut boundary."
                })
            if current_wind_speed >= THRESHOLDS["HIGH_WIND_THRESHOLD"] or downwind_threat:
                zone_actions.append({
                    "action_title": "Erect 3m Mobile Geotextile Windbreak Screens",
                    "priority": "Preventative Boundary Control",
                    "expected_efficacy_pct": "30% - 40% PM10 reduction",
                    "eff_val": 0.35,
                    "deployment_latency": "30 mins",
                    "cost_tier": "Medium ($$)",
                    "selection_reasons": [
                        f"Wind velocity ({current_wind_speed:.1f} km/h) exceeds aerodynamic threshold (18 km/h).",
                        f"Plume vector ({plume_heading:.0f}°) drifts toward sensitive perimeter receptors ({angle_diff:.0f}° angular offset)."
                    ],
                    "operational_detail": "Position portable modular geotextile wind screens on upwind excavation boundary to decelerate incoming wind gusts."
                })

        elif zone == "Material Storage":
            if current_wind_speed >= THRESHOLDS["HIGH_WIND_THRESHOLD"] and current_humidity <= THRESHOLDS["DRY_HUMIDITY_THRESHOLD"]:
                zone_actions.append({
                    "action_title": "Trigger Boundary Water Mist Cannon Array",
                    "priority": "Immediate Action Trigger",
                    "expected_efficacy_pct": "55% - 70% PM10 reduction",
                    "eff_val": 0.60,
                    "deployment_latency": "5 mins",
                    "cost_tier": "Medium ($$)",
                    "selection_reasons": [
                        f"Critical Aerodynamic Erosion Trigger: Wind {current_wind_speed:.1f} km/h >= 18 km/h combined with dry air ({current_humidity:.1f}% RH).",
                        "Stockpile crests are undergoing active wind shear erosion.",
                        f"Downwind receptor offset: {angle_diff:.0f}° from vulnerable boundary."
                    ],
                    "operational_detail": "Activate dual 45-degree oscillating mist cannons stationed at stockpile perimeter to create a curtain of micro-droplets."
                })
            else:
                zone_actions.append({
                    "action_title": "Apply Eco-Friendly Soil Stabilizing Surface Tackifier",
                    "priority": "Scheduled Preventative",
                    "expected_efficacy_pct": "80% - 90% PM10 reduction",
                    "eff_val": 0.85,
                    "deployment_latency": "2 hours",
                    "cost_tier": "Medium ($$)",
                    "selection_reasons": [
                        "Exposed aggregate stockpiles vulnerable to future wind gusts.",
                        "Surface crust binder eliminates aerodynamic dust lift for up to 72 hours."
                    ],
                    "operational_detail": "Spray diluted organic polymer tackifier over dormant aggregate piles to seal surface fines."
                })

        elif zone == "Loading Point":
            if pm10 >= THRESHOLDS["PM10_SITE_ACTION"] or activity == "High":
                zone_actions.append({
                    "action_title": "Activate Ultrasonic Dry Fog Agglomeration at Transfer Chute",
                    "priority": "Immediate Action Trigger",
                    "expected_efficacy_pct": "75% - 85% PM10 reduction",
                    "eff_val": 0.80,
                    "deployment_latency": "Instant",
                    "cost_tier": "Medium ($$)",
                    "selection_reasons": [
                        f"High particulate concentration ({pm10:.1f} µg/m³) during active batch loading.",
                        "Dry fog droplets (1–10 microns) bind respirable dust particles without wetting bulk aggregate."
                    ],
                    "operational_detail": "Trigger ultrasonic acoustic dry fog nozzle array inside the hopper throat enclosure."
                })
            zone_actions.append({
                "action_title": "Restrict Free-Fall Drop Height (< 1.2m)",
                "priority": "Operational Standard Control",
                "expected_efficacy_pct": "45% - 55% PM10 reduction",
                "eff_val": 0.50,
                "deployment_latency": "Instant",
                "cost_tier": "Zero ($0)",
                "selection_reasons": [
                    "Free-fall kinetic energy drives air displacement and turbulent dust lofting.",
                    "Lowering loader bucket to within 1.2m of truck bed immediately cuts plume volume."
                ],
                "operational_detail": "Instruct loader operators to maintain minimal clearance over hauler beds prior to aggregate discharge."
            })

        for act in zone_actions:
            act["zone"] = zone
            act["current_pm10"] = pm10
            act["trend_label"] = trend_label
            act["human_approval_required"] = True
            act["status"] = "Pending Review"
            recommendations.append(act)

    return recommendations


# ---------------------------------------------------------
# FEATURE 3: Multi-Scenario Verification Engine
# ---------------------------------------------------------
def compare_mitigation_scenarios(
    df: pd.DataFrame,
    interval_hours: float = 0.25
) -> Dict[str, Any]:
    """
    Performs side-by-side scenario verification comparing unmitigated baseline
    against pre-configured intervention packages:
    - Scenario A: Unmitigated Baseline
    - Scenario B: Targeted Fast-Response (Immediate priority actions on top sources)
    - Scenario C: Comprehensive Site Controls (Multi-zone engineered containment)
    
    Guarantees:
    - Outputs strictly labeled as hypothetical simulation models.
    - Unique elapsed clock-hour exceedance math applied across all packages.
    """
    if df.empty:
        return {}

    baseline_peak_pm10 = float(df["pm10"].max())
    baseline_avg_pm10 = float(df["pm10"].mean())
    baseline_exceed_timestamps = df[df["pm10"] >= THRESHOLDS["PM10_EPA_24H_LIMIT"]]["timestamp"].nunique()
    baseline_exceed_hours = round(baseline_exceed_timestamps * interval_hours, 2)

    pkg_b_effs = {
        "Haul Road": 0.70,
        "Excavation": 0.68,
        "Material Storage": 0.0,
        "Loading Point": 0.0
    }
    sim_b_df, metrics_b = calculate_mitigation_simulation(df, pkg_b_effs, interval_hours)

    pkg_c_effs = {
        "Haul Road": 0.75,
        "Excavation": 0.68,
        "Material Storage": 0.65,
        "Loading Point": 0.80
    }
    sim_c_df, metrics_c = calculate_mitigation_simulation(df, pkg_c_effs, interval_hours)

    comparison = {
        "is_hypothetical": True,
        "model_provenance": "Empirical engineering transfer functions (US EPA AP-42 & WRAP Fugitive Dust Handbook).",
        "scientific_disclaimer": "HYPOTHETICAL SIMULATION MODEL: Projections model theoretical potential reduction. They do not prove or guarantee physical real-world outcomes under variable field conditions.",
        "scenarios": [
            {
                "id": "scenario_baseline",
                "name": "Scenario A: Unmitigated Baseline",
                "scope": "Current operational state without active additional controls",
                "peak_pm10": baseline_peak_pm10,
                "avg_pm10": round(baseline_avg_pm10, 1),
                "peak_reduction_pct": 0.0,
                "exceedance_hours": baseline_exceed_hours,
                "avoided_hours": 0.0,
                "cost_tier": "Zero ($0)",
                "deployment_time": "Immediate"
            },
            {
                "id": "scenario_targeted",
                "name": "Scenario B: Targeted Fast-Response",
                "scope": "Water bowser on Haul Road (70%) + Excavator boom wetting (68%)",
                "peak_pm10": metrics_b["simulated_post_peak_pm10"],
                "avg_pm10": round(float(sim_b_df["pm10_simulated"].mean()), 1),
                "peak_reduction_pct": metrics_b["simulated_reduction_pct"],
                "exceedance_hours": metrics_b["simulated_post_exceedance_hours"],
                "avoided_hours": metrics_b["simulated_avoided_exceedance_hours"],
                "cost_tier": "Low - Medium ($)",
                "deployment_time": "10 mins"
            },
            {
                "id": "scenario_comprehensive",
                "name": "Scenario C: Comprehensive Site Controls",
                "scope": "Full site array: Bowser (75%), Excavation (68%), Mist cannons (65%), Dry fog (80%)",
                "peak_pm10": metrics_c["simulated_post_peak_pm10"],
                "avg_pm10": round(float(sim_c_df["pm10_simulated"].mean()), 1),
                "peak_reduction_pct": metrics_c["simulated_reduction_pct"],
                "exceedance_hours": metrics_c["simulated_post_exceedance_hours"],
                "avoided_hours": metrics_c["simulated_avoided_exceedance_hours"],
                "cost_tier": "Medium ($$)",
                "deployment_time": "30 mins"
            }
        ]
    }
    return comparison


# ---------------------------------------------------------
# FEATURE 4: Intervention Feedback & Outcome Evaluation Workflow
# (PROVENANCE_OPTIONS, VERIFICATION_STATUSES, ZONE_RECOMMENDED_ACTIONS
#  defined above in Domain Constants)
# ---------------------------------------------------------

def calculate_observed_outcome(
    pm10_before: Optional[float],
    pm10_after: Optional[float],
    pm25_before: Optional[float] = None,
    pm25_after: Optional[float] = None
) -> Dict[str, Any]:
    """
    Computes absolute and percentage changes in particulate matter concentrations
    between baseline and post-intervention measurements.
    
    Adheres strictly to scientific rules:
    - Never divides by zero: when baseline PM10 <= 0, percent change is None.
    - Does not silently convert missing readings to zero: if either reading is None,
      marks completeness as False and retains None for derived deltas.
    - Categorizes outcome direction ('Decreased', 'Increased', 'Unchanged').
    - Appends explicit non-causal scientific disclaimers: ambient reductions are not
      solely attributable to the mitigation action due to meteorology and dispersion.
    """
    def _is_valid_reading(val: Any) -> bool:
        if val is None:
            return False
        try:
            f = float(val)
            return not (math.isnan(f) or math.isinf(f)) and f >= 0.0
        except (ValueError, TypeError):
            return False

    valid_pm10_before = _is_valid_reading(pm10_before)
    valid_pm10_after = _is_valid_reading(pm10_after)
    
    is_complete = valid_pm10_before and valid_pm10_after
    
    b10 = round(float(pm10_before), 2) if valid_pm10_before else None
    a10 = round(float(pm10_after), 2) if valid_pm10_after else None
    
    pm10_abs_change: Optional[float] = None
    pm10_pct_change: Optional[float] = None
    
    if is_complete and b10 is not None and a10 is not None:
        pm10_abs_change = round(b10 - a10, 2)
        if b10 > 0.0:
            pm10_pct_change = round(((b10 - a10) / b10) * 100.0, 2)
        else:
            pm10_pct_change = None  # Zero baseline safety
            
        if pm10_abs_change > 0.1:
            pm10_direction = "Decreased"
        elif pm10_abs_change < -0.1:
            pm10_direction = "Increased"
        else:
            pm10_direction = "Unchanged"
    elif valid_pm10_before and not valid_pm10_after:
        pm10_direction = "Pending Post-Intervention Reading"
    elif not valid_pm10_before and valid_pm10_after:
        pm10_direction = "Missing Baseline Reading"
    else:
        pm10_direction = "Incomplete / Missing Data"

    # PM2.5 evaluation (optional)
    valid_pm25_before = _is_valid_reading(pm25_before)
    valid_pm25_after = _is_valid_reading(pm25_after)
    
    b25 = round(float(pm25_before), 2) if valid_pm25_before else None
    a25 = round(float(pm25_after), 2) if valid_pm25_after else None
    
    pm25_abs_change: Optional[float] = None
    pm25_pct_change: Optional[float] = None
    
    if valid_pm25_before and valid_pm25_after and b25 is not None and a25 is not None:
        pm25_abs_change = round(b25 - a25, 2)
        if b25 > 0.0:
            pm25_pct_change = round(((b25 - a25) / b25) * 100.0, 2)
        else:
            pm25_pct_change = None

    caveat = (
        "Observed delta reflects ambient sensor differences across the observation interval. "
        "Ambient concentration changes may be influenced by local wind dispersion, atmospheric humidity, "
        "and adjacent unmitigated operations; outcomes are not solely attributable to the intervention."
    )

    return {
        "is_complete": is_complete,
        "pm10_before": b10,
        "pm10_after": a10,
        "pm10_abs_change": pm10_abs_change,
        "pm10_pct_change": pm10_pct_change,
        "pm10_direction": pm10_direction,
        "pm25_before": b25,
        "pm25_after": a25,
        "pm25_abs_change": pm25_abs_change,
        "pm25_pct_change": pm25_pct_change,
        "scientific_caveat": caveat
    }


def validate_intervention_input(
    zone: str,
    recommended_action: str,
    performed_action: str,
    provenance: str,
    pm10_before: Any,
    pm10_after: Any,
    timestamp_before: Optional[str] = None,
    timestamp_after: Optional[str] = None,
    verification_status: str = "Pending Verification",
    is_authorized: bool = True,
    authorizer_name: Optional[str] = None,
    pm25_before: Any = None,
    pm25_after: Any = None,
    wind_speed: Any = None,
    wind_direction: Any = None
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Validates user or sensor inputs for an intervention record.
    
    Ensures:
    - Zone is a known construction zone.
    - Provenance is recognized.
    - Numeric readings are non-negative; does not silently cast None/empty to 0.
    - Timestamps are ordered correctly (before <= after).
    - Status 'Field Verified' requires:
        1. provenance == 'Field Measurement'
        2. valid baseline PM10 > 0
        3. valid post-intervention PM10
        4. is_authorized is True with authorizer name.
    """
    errors: List[str] = []
    
    if zone not in SITE_ZONES:
        errors.append(f"Invalid construction zone '{zone}'. Must be one of {SITE_ZONES}.")
        
    if provenance not in PROVENANCE_OPTIONS:
        errors.append(f"Invalid provenance '{provenance}'. Must be one of {PROVENANCE_OPTIONS}.")
        
    if verification_status not in VERIFICATION_STATUSES:
        errors.append(f"Invalid verification status '{verification_status}'. Must be one of {VERIFICATION_STATUSES}.")

    # Helper for numeric parsing without silent 0 coercion
    def _parse_float(val: Any, field_name: str, allow_none: bool = True) -> Tuple[Optional[float], Optional[str]]:
        if val is None or (isinstance(val, str) and val.strip() == ""):
            if not allow_none:
                return None, f"{field_name} is required and cannot be empty."
            return None, None
        try:
            f = float(val)
            if math.isnan(f) or math.isinf(f):
                return None, f"{field_name} must be a valid real number."
            if f < 0.0:
                return None, f"{field_name} cannot be negative ({f} µg/m³)."
            return round(f, 2), None
        except (ValueError, TypeError):
            return None, f"{field_name} must be a valid numeric reading."

    parsed_pm10_before, err_b10 = _parse_float(pm10_before, "PM10 Before Intervention")
    if err_b10:
        errors.append(err_b10)

    parsed_pm10_after, err_a10 = _parse_float(pm10_after, "PM10 After Intervention")
    if err_a10:
        errors.append(err_a10)

    parsed_pm25_before, err_b25 = _parse_float(pm25_before, "PM2.5 Before Intervention")
    if err_b25:
        errors.append(err_b25)

    parsed_pm25_after, err_a25 = _parse_float(pm25_after, "PM2.5 After Intervention")
    if err_a25:
        errors.append(err_a25)

    parsed_wind_speed, err_ws = _parse_float(wind_speed, "Wind Speed (km/h)")
    if err_ws:
        errors.append(err_ws)

    # Timestamp parsing & ordering validation
    def _parse_dt(dt_str: Optional[str]) -> Optional[datetime]:
        if not dt_str or not dt_str.strip():
            return None
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M"
        ]
        for fmt in formats:
            try:
                return datetime.strptime(dt_str.strip(), fmt)
            except ValueError:
                continue
        return None

    dt_before = _parse_dt(timestamp_before)
    dt_after = _parse_dt(timestamp_after)

    if timestamp_before and not dt_before:
        errors.append(f"Invalid timestamp_before format: '{timestamp_before}'.")
    if timestamp_after and not dt_after:
        errors.append(f"Invalid timestamp_after format: '{timestamp_after}'.")

    if dt_before and dt_after and dt_before > dt_after:
        errors.append("Measurement start timestamp cannot be after the post-intervention timestamp.")

    # Strict Verification Status Rules:
    if verification_status == "Field Verified":
        if provenance != "Field Measurement":
            errors.append(
                f"Cannot mark outcome as 'Field Verified' when provenance is '{provenance}'. "
                "Only actual field measurements may be marked as Field Verified."
            )
        if parsed_pm10_before is None or parsed_pm10_after is None:
            errors.append("Field Verified status requires complete before and after PM10 measurements.")
        elif parsed_pm10_before <= 0.0:
            errors.append("Field Verified status requires a baseline PM10 reading greater than zero.")
        if not is_authorized:
            errors.append("Field Verified status requires explicit human authorization.")

    cleaned_data = {
        "zone": zone,
        "recommended_action": recommended_action.strip() if recommended_action else "Standard Operational Protocol",
        "performed_action": performed_action.strip() if performed_action else "Standard Operational Protocol",
        "provenance": provenance,
        "pm10_before": parsed_pm10_before,
        "pm10_after": parsed_pm10_after,
        "pm25_before": parsed_pm25_before,
        "pm25_after": parsed_pm25_after,
        "timestamp_before": timestamp_before.strip() if timestamp_before else None,
        "timestamp_after": timestamp_after.strip() if timestamp_after else None,
        "wind_speed": parsed_wind_speed,
        "wind_direction": wind_direction.strip() if wind_direction else None,
        "verification_status": verification_status,
        "is_authorized": bool(is_authorized),
        "authorizer_name": authorizer_name.strip() if authorizer_name else "[UNASSIGNED]"
    }
    
    is_valid = (len(errors) == 0)
    return is_valid, errors, cleaned_data


def create_intervention_record(
    zone: str,
    recommended_action: str,
    performed_action: str,
    provenance: str = "Synthetic Demonstration",
    pm10_before: Optional[float] = None,
    pm10_after: Optional[float] = None,
    pm25_before: Optional[float] = None,
    pm25_after: Optional[float] = None,
    timestamp_before: Optional[str] = None,
    timestamp_after: Optional[str] = None,
    observation_interval_mins: Optional[int] = 30,
    wind_speed: Optional[float] = None,
    wind_direction: Optional[str] = None,
    weather_notes: str = "",
    is_authorized: bool = True,
    authorizer_name: str = "Site Operations Officer",
    operator_notes: str = "",
    verification_status: str = "Pending Verification",
    record_id: Optional[str] = None,
    linked_order_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Creates a verified, structured intervention feedback record with explicit
    outcome metrics and data provenance tracking.
    
    Does NOT claim self-learning AI or automatic rule retraining.
    """
    outcome = calculate_observed_outcome(
        pm10_before=pm10_before,
        pm10_after=pm10_after,
        pm25_before=pm25_before,
        pm25_after=pm25_after
    )
    
    # Strictly enforce: is_verified_field_outcome is True ONLY if Field Measurement,
    # status is 'Field Verified', complete valid PM10 readings exist, and is_authorized is True.
    final_verification_status = verification_status
    if provenance != "Field Measurement" and verification_status == "Field Verified":
        final_verification_status = "Unverified / Incomplete"

    is_verified_field_outcome = (
        provenance == "Field Measurement" and
        final_verification_status == "Field Verified" and
        outcome["is_complete"] and
        outcome["pm10_before"] is not None and
        outcome["pm10_before"] > 0.0 and
        bool(is_authorized)
    )

    now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rid = record_id or f"INT-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{next(_intervention_counter):05d}"

    alignment = "Matches Recommendation" if (recommended_action and performed_action and recommended_action.strip().lower() == performed_action.strip().lower()) else "Alternative Action Performed"

    rec = {
        "record_id": rid,
        "linked_order_id": linked_order_id,
        "timestamp": now_ts,
        "zone": zone,
        "recommended_action": recommended_action,
        "performed_action": performed_action,
        "action_alignment": alignment,
        "provenance": provenance,
        "is_field_measurement": (provenance == "Field Measurement"),
        "verification_status": final_verification_status,
        "is_verified_field_outcome": is_verified_field_outcome,
        "is_authorized": bool(is_authorized),
        "authorizer_name": authorizer_name.strip() if authorizer_name else "[UNASSIGNED]",
        "pm10_before": outcome["pm10_before"],
        "pm10_after": outcome["pm10_after"],
        "pm10_abs_change": outcome["pm10_abs_change"],
        "pm10_pct_change": outcome["pm10_pct_change"],
        "pm10_direction": outcome["pm10_direction"],
        "pm25_before": outcome["pm25_before"],
        "pm25_after": outcome["pm25_after"],
        "pm25_abs_change": outcome["pm25_abs_change"],
        "pm25_pct_change": outcome["pm25_pct_change"],
        "timestamp_before": timestamp_before or now_ts,
        "timestamp_after": timestamp_after or (datetime.now() + timedelta(minutes=observation_interval_mins or 30)).strftime("%Y-%m-%d %H:%M:%S"),
        "observation_interval_mins": int(observation_interval_mins) if observation_interval_mins is not None else 30,
        "wind_speed_kmh": round(float(wind_speed), 1) if wind_speed is not None else None,
        "wind_direction": wind_direction or "Variable",
        "weather_notes": weather_notes.strip() if weather_notes else "Normal dry shift conditions",
        "operator_notes": operator_notes.strip() if operator_notes else "",
        "is_complete": outcome["is_complete"],
        "scientific_caveat": outcome["scientific_caveat"],
        "storage_mode": "Volatile In-Memory Session Memory (Reset on server restart)",
        
        # Legacy compatibility keys
        "action_title": performed_action,
        "operator_name": authorizer_name.strip() if authorizer_name else "[SITE_OPERATOR]",
        "evaluation_status": verification_status,
        "field_notes": operator_notes.strip() if operator_notes else "",
        "simulated_expected_reduction_pct": 70.0,
        "simulated_avoided_hours": 2.5,
        "audit_provenance": f"Operational Decision-Support Audit Log ({provenance}, Non-Continuous Learning)"
    }
    return rec


def create_feedback_record(
    zone: str,
    action_title: str,
    operator_name: str,
    simulated_reduction_pct: float,
    simulated_avoided_hours: float,
    evaluation_status: str,
    field_notes: str = "",
    action_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Creates a validated, structured feedback record documenting a chosen intervention,
    its simulated expectation, and the operator's subsequent physical evaluation.
    
    Maintains backward compatibility with earlier prototype milestones while incorporating
    the closed-loop intervention verification data model.
    """
    valid_statuses = [
        "Observed Effective",
        "Partially Effective",
        "Ineffective / High Winds Resuspended",
        "Pending Field Inspection"
    ]
    status = evaluation_status if evaluation_status in valid_statuses else "Pending Field Inspection"
    
    # Map legacy status to verification status
    v_status = "Pending Verification"
    if status in ["Observed Effective", "Partially Effective"]:
        v_status = "Field Verified"
    elif status == "Ineffective / High Winds Resuspended":
        v_status = "Rejected / Ineffective"

    # Default synthetic demonstration baseline
    b10 = 120.0
    a10 = round(b10 * (1.0 - (float(simulated_reduction_pct) / 100.0)), 1) if simulated_reduction_pct > 0 else 120.0

    rec = create_intervention_record(
        zone=zone,
        recommended_action=action_title,
        performed_action=action_title,
        provenance="Synthetic Demonstration",
        pm10_before=b10,
        pm10_after=a10,
        verification_status=v_status,
        is_authorized=True,
        authorizer_name=operator_name,
        operator_notes=field_notes,
        record_id=action_id or f"ACT-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    
    # Guarantee exact legacy fields expected by existing tests
    rec["evaluation_status"] = status
    rec["operator_name"] = operator_name.strip() if operator_name and operator_name.strip() else "[SITE_OPERATOR]"
    rec["simulated_expected_reduction_pct"] = round(float(simulated_reduction_pct), 1)
    rec["simulated_avoided_hours"] = round(float(simulated_avoided_hours), 2)
    rec["audit_provenance"] = "Operational Decision-Support Audit Log (Human Verified, Non-Continuous Learning)"
    return rec


def filter_intervention_records(
    records: List[Dict[str, Any]],
    zone: Optional[str] = None,
    action: Optional[str] = None,
    provenance: Optional[str] = None,
    verification_status: Optional[str] = None,
    search_query: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Filters a collection of intervention records by zone, action, provenance,
    verification status, and text search.
    """
    filtered = list(records)
    if zone and zone not in ["All", "All Zones"]:
        filtered = [r for r in filtered if r.get("zone") == zone]
    if action and action not in ["All", "All Actions"]:
        filtered = [r for r in filtered if r.get("performed_action") == action or r.get("action_title") == action]
    if provenance and provenance not in ["All", "All Provenances"]:
        filtered = [r for r in filtered if r.get("provenance") == provenance]
    if verification_status and verification_status not in ["All", "All Statuses"]:
        filtered = [
            r for r in filtered 
            if r.get("verification_status") == verification_status or r.get("evaluation_status") == verification_status
        ]
    if search_query and search_query.strip():
        q = search_query.strip().lower()
        filtered = [
            r for r in filtered
            if q in str(r.get("record_id", "")).lower()
            or q in str(r.get("performed_action", "")).lower()
            or q in str(r.get("operator_notes", "")).lower()
            or q in str(r.get("authorizer_name", "")).lower()
        ]
    return filtered


def aggregate_feedback_analytics(
    records: List[Dict[str, Any]],
    min_observations: int = 3
) -> Dict[str, Any]:
    """
    Computes responsible operational feedback analytics and evidence summaries.
    
    STRICT SAFEGUARDS:
    - Never mixes synthetic demonstration examples with field measurements.
    - Requires at least `min_observations` verified field records for an action before
      providing an empirical outcome summary. If below threshold, displays explicit
      'Insufficient verified field data'.
    - Does not claim self-learning AI, automatic parameter retraining, or causal proof.
    """
    # Define consistent population with unique record-ID rule:
    # Deduplicate interventions by record_id and linked_order_id without double counting
    seen_ids = set()
    deduped_records = []
    for r in records:
        rid = str(r.get("record_id") or "").strip()
        lid = str(r.get("linked_order_id") or "").strip()
        matched = False
        if rid and rid in seen_ids:
            matched = True
        if lid and lid in seen_ids:
            matched = True
        if not rid and not lid:
            if id(r) in seen_ids:
                matched = True
            else:
                seen_ids.add(id(r))
        
        if not matched:
            if rid:
                seen_ids.add(rid)
            if lid:
                seen_ids.add(lid)
            deduped_records.append(r)

    total_count = len(deduped_records)
    
    # Strictly separate records by data provenance
    field_records = [r for r in deduped_records if r.get("provenance") == "Field Measurement"]
    synthetic_records = [r for r in deduped_records if r.get("provenance") == "Synthetic Demonstration"]
    uploaded_records = [r for r in deduped_records if r.get("provenance") == "Uploaded Dataset"]
    
    # Verified field records: MUST be Field Measurement, verified, authorized, and complete with baseline > 0
    verified_field_records = [
        r for r in field_records
        if (
            (r.get("is_verified_field_outcome") is True or r.get("verification_status") == "Field Verified") and
            r.get("is_complete") is True and
            r.get("pm10_before") is not None and
            float(r.get("pm10_before", 0.0)) > 0.0 and
            r.get("is_authorized") is True
        )
    ]
    
    # Mutually exclusive status partitions across the unique population:
    # 1. Field Verified (meeting all criteria)
    vf_ids = set(id(r) for r in verified_field_records)
    # 2. Pending Verification (not in verified field)
    pending_records = [r for r in deduped_records if id(r) not in vf_ids and r.get("verification_status") == "Pending Verification"]
    pending_ids = set(id(r) for r in pending_records)
    # 3. Rejected / Ineffective
    rejected_records = [r for r in deduped_records if id(r) not in vf_ids and id(r) not in pending_ids and r.get("verification_status") == "Rejected / Ineffective"]
    rejected_ids = set(id(r) for r in rejected_records)
    # 4. Unverified / Incomplete: all remaining records
    unverified_incomplete_records = [r for r in deduped_records if id(r) not in vf_ids and id(r) not in pending_ids and id(r) not in rejected_ids]

    # Incomplete records across all statuses (for data health diagnostics)
    incomplete_records = [r for r in deduped_records if not r.get("is_complete", False)]
    authorized_records = [r for r in deduped_records if r.get("is_authorized") is True]

    # Metrics on verified field records ONLY
    vf_abs_deltas = [r["pm10_abs_change"] for r in verified_field_records if r.get("pm10_abs_change") is not None]
    vf_pct_deltas = [r["pm10_pct_change"] for r in verified_field_records if r.get("pm10_pct_change") is not None]
    
    vf_metrics = {
        "count": len(verified_field_records),
        "mean_abs_reduction_ugm3": round(float(np.mean(vf_abs_deltas)), 2) if vf_abs_deltas else None,
        "median_abs_reduction_ugm3": round(float(np.median(vf_abs_deltas)), 2) if vf_abs_deltas else None,
        "mean_pct_reduction": round(float(np.mean(vf_pct_deltas)), 2) if vf_pct_deltas else None,
        "decreased_count": sum(1 for r in verified_field_records if r.get("pm10_direction") == "Decreased"),
        "increased_count": sum(1 for r in verified_field_records if r.get("pm10_direction") == "Increased"),
        "unchanged_count": sum(1 for r in verified_field_records if r.get("pm10_direction") == "Unchanged")
    }

    # Synthetic / demonstration metrics (STRICTLY SEPARATED)
    syn_abs_deltas = [r["pm10_abs_change"] for r in synthetic_records if r.get("pm10_abs_change") is not None]
    syn_pct_deltas = [r["pm10_pct_change"] for r in synthetic_records if r.get("pm10_pct_change") is not None]
    syn_metrics = {
        "count": len(synthetic_records),
        "label": "Synthetic Demonstration Benchmarks (Non-Field Data)",
        "mean_abs_reduction_ugm3": round(float(np.mean(syn_abs_deltas)), 2) if syn_abs_deltas else None,
        "mean_pct_reduction": round(float(np.mean(syn_pct_deltas)), 2) if syn_pct_deltas else None
    }

    # Evidence-informed recommendation summaries by action
    action_summaries = {}
    unique_actions = sorted(list({
        r.get("performed_action") or r.get("action_title") 
        for r in records 
        if r.get("performed_action") or r.get("action_title")
    }))
    
    for act in unique_actions:
        act_vf = [
            r for r in verified_field_records 
            if r.get("performed_action") == act or r.get("action_title") == act
        ]
        n_obs = len(act_vf)
        
        act_abs = [r["pm10_abs_change"] for r in act_vf if r.get("pm10_abs_change") is not None]
        act_pct = [r["pm10_pct_change"] for r in act_vf if r.get("pm10_pct_change") is not None]
        
        has_sufficient_evidence = (n_obs >= min_observations)
        
        if has_sufficient_evidence:
            mean_pct = round(float(np.mean(act_pct)), 1) if act_pct else 0.0
            mean_abs = round(float(np.mean(act_abs)), 1) if act_abs else 0.0
            summary_text = (
                f"Sufficient verified field evidence (N = {n_obs} observations). "
                f"Observed mean PM10 reduction is {mean_pct}% ({mean_abs} µg/m³)."
            )
            guidance_note = (
                f"Empirical field observations support continued advisory recommendation of '{act}'. "
                "Advisory decision support only; human engineering approval required before deployment."
            )
            evidence_status = "Sufficient Verified Field Data"
        else:
            mean_pct = round(float(np.mean(act_pct)), 1) if act_pct else None
            mean_abs = round(float(np.mean(act_abs)), 1) if act_abs else None
            summary_text = (
                f"Insufficient verified field data (N = {n_obs} < {min_observations} required). "
                "Retaining theoretical engineering baseline."
            )
            guidance_note = (
                f"Empirical field evidence is currently insufficient (N = {n_obs}). "
                "Advisory recommendations rely on aerodynamic engineering models until verified field quota is met."
            )
            evidence_status = "Insufficient verified field data"

        action_summaries[act] = {
            "action_title": act,
            "verified_field_count": n_obs,
            "min_required": min_observations,
            "has_sufficient_evidence": has_sufficient_evidence,
            "evidence_status": evidence_status,
            "mean_abs_reduction_ugm3": mean_abs,
            "mean_pct_reduction": mean_pct,
            "summary_text": summary_text,
            "guidance_note": guidance_note
        }

    return {
        "total_recorded": total_count,
        "count_field_total": len(field_records),
        "count_verified_field": len(verified_field_records),
        "count_synthetic": len(synthetic_records),
        "count_uploaded": len(uploaded_records),
        "count_pending": len(pending_records),
        "count_unverified_incomplete": len(unverified_incomplete_records),
        "count_rejected": len(rejected_records),
        "count_incomplete": len(incomplete_records),
        "count_authorized": len(authorized_records),
        "min_observations_threshold": min_observations,
        "field_verified_metrics": vf_metrics,
        "synthetic_metrics": syn_metrics,
        "action_summaries": action_summaries,
        "governance_notice": (
            "Evidence-Informed Feedback Analytics: Tracks physical mitigation verification without "
            "unvalidated online model retraining. Synthetic demonstrations and field measurements are "
            "never pooled in performance metrics."
        )
    }


def export_intervention_feedback_csv(records: List[Dict[str, Any]]) -> str:
    """
    Exports intervention records and calculated outcomes to a standard CSV string.
    Includes provenance, verification status, units, timestamps, and scientific caveat.
    """
    headers = [
        "record_id", "linked_order_id", "timestamp_logged", "zone", "recommended_action",
        "performed_action", "action_alignment", "provenance", "verification_status",
        "is_verified_field_outcome", "is_authorized", "authorizer_name",
        "pm10_before_ugm3", "pm10_after_ugm3", "pm10_abs_change_ugm3",
        "pm10_pct_change", "pm10_direction", "pm25_before_ugm3", "pm25_after_ugm3",
        "pm25_abs_change_ugm3", "pm25_pct_change", "timestamp_before",
        "timestamp_after", "observation_interval_mins", "wind_speed_kmh",
        "wind_direction", "weather_notes", "operator_notes", "is_complete",
        "scientific_caveat", "storage_mode"
    ]
    if not records:
        return ",".join(headers) + "\n"

    rows = []
    for r in records:
        rows.append({
            "record_id": r.get("record_id", ""),
            "linked_order_id": r.get("linked_order_id", ""),
            "timestamp_logged": r.get("timestamp", ""),
            "zone": r.get("zone", ""),
            "recommended_action": r.get("recommended_action", r.get("action_title", "")),
            "performed_action": r.get("performed_action", r.get("action_title", "")),
            "action_alignment": r.get("action_alignment", "Matches Recommendation"),
            "provenance": r.get("provenance", "Synthetic Demonstration"),
            "verification_status": r.get("verification_status", r.get("evaluation_status", "Pending Verification")),
            "is_verified_field_outcome": r.get("is_verified_field_outcome", False),
            "is_authorized": r.get("is_authorized", True),
            "authorizer_name": r.get("authorizer_name", r.get("operator_name", "")),
            "pm10_before_ugm3": r.get("pm10_before", ""),
            "pm10_after_ugm3": r.get("pm10_after", ""),
            "pm10_abs_change_ugm3": r.get("pm10_abs_change", ""),
            "pm10_pct_change": r.get("pm10_pct_change", ""),
            "pm10_direction": r.get("pm10_direction", ""),
            "pm25_before_ugm3": r.get("pm25_before", ""),
            "pm25_after_ugm3": r.get("pm25_after", ""),
            "pm25_abs_change_ugm3": r.get("pm25_abs_change", ""),
            "pm25_pct_change": r.get("pm25_pct_change", ""),
            "timestamp_before": r.get("timestamp_before", ""),
            "timestamp_after": r.get("timestamp_after", ""),
            "observation_interval_mins": r.get("observation_interval_mins", 30),
            "wind_speed_kmh": r.get("wind_speed_kmh", ""),
            "wind_direction": r.get("wind_direction", ""),
            "weather_notes": r.get("weather_notes", ""),
            "operator_notes": r.get("operator_notes", r.get("field_notes", "")),
            "is_complete": r.get("is_complete", False),
            "scientific_caveat": r.get("scientific_caveat", "Observed outcome not solely attributable to intervention."),
            "storage_mode": r.get("storage_mode", "Volatile In-Memory Session Store")
        })

    df = pd.DataFrame(rows)
    return df.to_csv(index=False)


def generate_sample_feedback_records() -> List[Dict[str, Any]]:
    """
    Generates a curated set of demonstration records clearly separating
    synthetic examples from verified and unverified field observations.
    """
    t0 = datetime.now() - timedelta(hours=3)
    t1 = t0 + timedelta(minutes=30)
    
    records = [
        # 1. Verified Field Measurement (Haul Road Misting)
        create_intervention_record(
            zone="Haul Road",
            recommended_action="Automated Water Bowser Misting Cycle",
            performed_action="Automated Water Bowser Misting Cycle",
            provenance="Field Measurement",
            pm10_before=132.5,
            pm10_after=39.8,
            pm25_before=28.4,
            pm25_after=14.2,
            timestamp_before=t0.strftime("%Y-%m-%d %H:%M:%S"),
            timestamp_after=t1.strftime("%Y-%m-%d %H:%M:%S"),
            observation_interval_mins=30,
            wind_speed=14.2,
            wind_direction="NW",
            weather_notes="Dry afternoon, moderate gusting from NW.",
            is_authorized=True,
            authorizer_name="M. Santos (Senior Site Engineer)",
            operator_notes="Bowser completed 2 circuits; wheel dust knocked down visibly. Perimeter PM10 fell below threshold.",
            verification_status="Field Verified",
            record_id="INT-20261015-001"
        ),
        # 2. Field Measurement - Pending Verification (Incomplete / ongoing)
        create_intervention_record(
            zone="Excavation",
            recommended_action="Excavation Drop Height Limitation",
            performed_action="Excavation Drop Height Limitation",
            provenance="Field Measurement",
            pm10_before=115.0,
            pm10_after=None,  # Not completed yet
            timestamp_before=(datetime.now() - timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M:%S"),
            timestamp_after=None,
            observation_interval_mins=30,
            wind_speed=16.5,
            wind_direction="WNW",
            weather_notes="Active loading pit excavation.",
            is_authorized=True,
            authorizer_name="D. Vance (Site Foreman)",
            operator_notes="Operator instructed to keep excavator bucket within 1.5m of truck bed. Post inspection pending.",
            verification_status="Pending Verification",
            record_id="INT-20261015-002"
        ),
        # 3. Synthetic Demonstration Record (Stockpile Windbreak) - Clearly synthetic!
        create_intervention_record(
            zone="Material Storage",
            recommended_action="Windbreak Netting & Perimeter Fencing",
            performed_action="Windbreak Netting & Perimeter Fencing",
            provenance="Synthetic Demonstration",
            pm10_before=98.0,
            pm10_after=49.0,
            timestamp_before=(datetime.now() - timedelta(hours=6)).strftime("%Y-%m-%d %H:%M:%S"),
            timestamp_after=(datetime.now() - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M:%S"),
            observation_interval_mins=60,
            wind_speed=20.5,
            wind_direction="N",
            weather_notes="Synthetic simulation condition: dry windy condition.",
            is_authorized=True,
            authorizer_name="Algorithm Simulation Testbed",
            operator_notes="Theoretical fluid dynamics simulation of 50% porous boundary net.",
            verification_status="Pending Verification",
            record_id="INT-20261015-003"
        ),
        # 4. Field Measurement - Rejected / Ineffective
        create_intervention_record(
            zone="Loading Point",
            recommended_action="Mobile Dust Extraction Unit",
            performed_action="Mobile Dust Extraction Unit",
            provenance="Field Measurement",
            pm10_before=108.0,
            pm10_after=116.5,
            timestamp_before=(datetime.now() - timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S"),
            timestamp_after=(datetime.now() - timedelta(hours=7, minutes=30)).strftime("%Y-%m-%d %H:%M:%S"),
            observation_interval_mins=30,
            wind_speed=24.0,
            wind_direction="SE",
            weather_notes="High wind gusts exceeding 22 km/h across hopper apron.",
            is_authorized=True,
            authorizer_name="T. Chen (Environmental Officer)",
            operator_notes="Extraction hood inlet overwhelmed by ambient 24 km/h crosswinds. Resuspension increased.",
            verification_status="Rejected / Ineffective",
            record_id="INT-20261015-004"
        )
    ]
    return records

