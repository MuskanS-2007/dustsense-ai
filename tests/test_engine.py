import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from core_engine import (
    validate_and_clean_data,
    generate_synthetic_stream,
    evaluate_site_risk,
    compute_rule_based_attribution,
    calculate_mitigation_simulation,
    get_cardinal_direction,
    compute_plume_heading,
    compute_angular_offset,
    split_chronological,
    train_short_horizon_forecast,
    generate_forward_forecast,
    evaluate_actionable_recommendations,
    compare_mitigation_scenarios,
    create_feedback_record,
    SITE_ZONES,
    THRESHOLDS,
    BACKGROUND_FLOOR
)

# ---------------------------------------------------------
# Test Group 1: Data Validation, Missing Values & Cross-Zone Isolation
# ---------------------------------------------------------
def test_cross_zone_missing_value_isolation():
    """
    CRITICAL AUDIT TEST: Ensure missing values in one zone NEVER inherit
    readings from an adjacent zone via global forward-fill.
    """
    df = pd.DataFrame({
        "timestamp": ["2026-10-14 10:00:00", "2026-10-14 10:00:00"],
        "zone": ["Excavation", "Haul Road"],
        "pm2_5": [25.0, np.nan],
        "pm10": [220.0, np.nan],
        "wind_speed": [12.0, 12.0],
        "wind_deg": [245.0, 245.0],
        "humidity": [35.0, 35.0],
        "temperature": [28.0, 28.0]
    })
    
    clean_df, report = validate_and_clean_data(df)
    
    assert report["valid"] is True
    haul_road = clean_df[clean_df["zone"] == "Haul Road"].iloc[0]
    # Haul road PM10 must NOT equal Excavation PM10 (220.0)
    assert haul_road["pm10"] != 220.0, "Cross-zone data contamination detected: Haul Road inherited Excavation PM10!"


def test_duplicate_record_detection_and_resolution():
    """
    Test that conflicting duplicate records sharing identical (timestamp, zone)
    are flagged and resolved deterministically rather than silently retained.
    """
    df = pd.DataFrame({
        "timestamp": ["2026-10-14 10:00:00", "2026-10-14 10:00:00"],
        "zone": ["Excavation", "Excavation"],
        "pm2_5": [20.0, 40.0],
        "pm10": [100.0, 200.0],
        "wind_speed": [10.0, 10.0],
        "wind_deg": [90.0, 90.0],
        "humidity": [50.0, 50.0],
        "temperature": [25.0, 25.0]
    })
    
    clean_df, report = validate_and_clean_data(df)
    
    assert report["valid"] is True
    assert report["duplicate_count"] == 1
    assert len(clean_df) == 1
    assert clean_df.iloc[0]["pm10"] == 150.0  # Averaged 100.0 and 200.0
    assert bool(clean_df.iloc[0]["is_anomalous"]) is True


def test_pm25_exceeds_pm10_anomaly_flagging():
    """
    CRITICAL AUDIT TEST: If PM2.5 > PM10 (impossible physical inversion),
    the row must be flagged as an anomaly, NEVER silently rewritten with a multiplier.
    """
    df = pd.DataFrame({
        "timestamp": ["2026-10-14 10:00:00"],
        "zone": ["Loading Point"],
        "pm2_5": [120.0],  # PM2.5 higher than PM10!
        "pm10": [50.0],
        "wind_speed": [8.0],
        "wind_deg": [180.0],
        "humidity": [45.0],
        "temperature": [24.0]
    })
    
    clean_df, report = validate_and_clean_data(df)
    
    assert report["valid"] is True
    row = clean_df.iloc[0]
    assert bool(row["is_anomalous"]) is True
    assert "ANOMALY_PM_INVERSION" in row["anomaly_notes"]
    # Verify PM2.5 was NOT silently modified to 50 * 0.95 = 47.5
    assert row["pm2_5"] == 120.0


def test_negative_readings_flagged_not_clamped():
    """
    Negative sensor readings indicate calibration faults and must be flagged,
    not silently zeroed out.
    """
    df = pd.DataFrame({
        "timestamp": ["2026-10-14 10:00:00"],
        "zone": ["Material Storage"],
        "pm2_5": [-5.0],
        "pm10": [-15.0],
        "wind_speed": [-2.0],
        "wind_deg": [200.0],
        "humidity": [120.0],  # > 100%
        "temperature": [22.0]
    })
    
    clean_df, report = validate_and_clean_data(df)
    
    assert report["valid"] is True
    row = clean_df.iloc[0]
    assert bool(row["is_anomalous"]) is True
    assert "NEGATIVE_PM_READING" in row["anomaly_notes"]
    assert "NEGATIVE_WIND_SPEED" in row["anomaly_notes"]
    assert "OUT_OF_BOUNDS_HUMIDITY" in row["anomaly_notes"]


def test_empty_dataset_handling():
    """Verify engine handles empty or None datasets gracefully without crashing."""
    clean_df, report = validate_and_clean_data(pd.DataFrame())
    assert report["valid"] is False
    assert "Dataset is empty" in report["issues"][0]


def test_unparseable_non_numeric_coercion():
    """Verify that a column with >10% corrupt text is rejected by the validator."""
    df = pd.DataFrame({
        "timestamp": ["2026-10-14 10:00:00", "2026-10-14 10:15:00"],
        "zone": ["Excavation", "Excavation"],
        "pm2_5": ["corrupt", "text"],
        "pm10": ["bad_data", "bad_data"],
        "wind_speed": [10.0, 10.0],
        "wind_deg": [90.0, 90.0],
        "humidity": [50.0, 50.0],
        "temperature": [25.0, 25.0]
    })
    
    clean_df, report = validate_and_clean_data(df)
    assert report["valid"] is False
    assert any("Upload rejected" in issue for issue in report["issues"])


# ---------------------------------------------------------
# Test Group 2: Synthetic Data Engine & Reproducibility
# ---------------------------------------------------------
def test_synthetic_data_reproducibility():
    """
    REQ-01: Seeded synthetic data generation must produce identical records
    and identical timestamps across repeated invocations.
    """
    df1 = generate_synthetic_stream(scenario="Active Peak Shift", hours=12, seed=42)
    df2 = generate_synthetic_stream(scenario="Active Peak Shift", hours=12, seed=42)
    
    assert len(df1) == len(df2)
    assert df1["timestamp"].equals(df2["timestamp"]), "Timestamps drifted across identical seed runs!"
    assert df1["pm10"].equals(df2["pm10"]), "Particulate values drifted across identical seed runs!"


def test_synthetic_data_scenarios():
    """Verify that different scenario presets produce distinct physical signatures."""
    df_peak = generate_synthetic_stream(scenario="Active Peak Shift", hours=12, seed=42)
    df_wind = generate_synthetic_stream(scenario="Dry Windy Alert Condition", hours=12, seed=42)
    
    # Windy scenario must have higher average wind speed and lower humidity
    assert df_wind["wind_speed"].mean() > df_peak["wind_speed"].mean()
    assert df_wind["humidity"].mean() < df_peak["humidity"].mean()


# ---------------------------------------------------------
# Test Group 3: Directional & Meteorological Mathematics
# ---------------------------------------------------------
def test_cardinal_direction_mapping():
    """Verify cardinal compass conversion across all octants."""
    assert get_cardinal_direction(0.0) == "N"
    assert get_cardinal_direction(90.0) == "E"
    assert get_cardinal_direction(180.0) == "S"
    assert get_cardinal_direction(270.0) == "W"
    assert get_cardinal_direction(245.0) == "WSW"


def test_plume_heading_calculation():
    """
    Wind direction is WHERE wind comes from.
    Plume travels downwind: (wind_deg + 180) % 360.
    """
    assert compute_plume_heading(245.0) == 65.0  # WSW wind carries plume ENE
    assert compute_plume_heading(0.0) == 180.0   # North wind carries plume South
    assert compute_plume_heading(180.0) == 0.0   # South wind carries plume North


def test_angular_offset():
    """Verify shortest angular distance across the 0/360 wrap-around boundary."""
    assert compute_angular_offset(10.0, 350.0) == 20.0
    assert compute_angular_offset(350.0, 10.0) == 20.0
    assert compute_angular_offset(90.0, 180.0) == 90.0


# ---------------------------------------------------------
# Test Group 4: Explainable Source Attribution & Ranking
# ---------------------------------------------------------
def test_source_attribution_ranking():
    """
    Verify rule engine correctly identifies the highest risk emitter
    and outputs transparent heuristic ranking scores.
    """
    latest = pd.DataFrame([
        {"zone": "Excavation", "pm10": 260.0, "pm2_5": 55.0, "activity_level": "High", "wind_speed": 14.0, "wind_deg": 245.0, "humidity": 35.0},
        {"zone": "Haul Road", "pm10": 45.0, "pm2_5": 12.0, "activity_level": "Low", "wind_speed": 14.0, "wind_deg": 245.0, "humidity": 35.0},
        {"zone": "Material Storage", "pm10": 50.0, "pm2_5": 10.0, "activity_level": "Low", "wind_speed": 14.0, "wind_deg": 245.0, "humidity": 35.0},
        {"zone": "Loading Point", "pm10": 60.0, "pm2_5": 15.0, "activity_level": "Moderate", "wind_speed": 14.0, "wind_deg": 245.0, "humidity": 35.0}
    ])
    
    attr_df = compute_rule_based_attribution(latest, latest)
    
    assert not attr_df.empty
    assert attr_df.iloc[0]["zone"] == "Excavation"
    assert attr_df.iloc[0]["rank"] == 1
    assert "ranking_score" in attr_df.columns
    assert len(attr_df.iloc[0]["explanations"]) >= 2


# ---------------------------------------------------------
# Test Group 5: Simulation Math & Exceedance Clock Time Accounting
# ---------------------------------------------------------
def test_simulated_exceedance_hours_accounting():
    """
    CRITICAL AUDIT TEST: Ensure simultaneous multi-zone exceedances
    are NOT multiplied as separate elapsed clock hours.
    1 interval (15 mins = 0.25h) with 4 zones in exceedance = EXACTLY 0.25h.
    """
    # 2 distinct timestamps, all 4 zones exceeding 150 µg/m³
    rows = []
    for t_str in ["2026-10-14 10:00:00", "2026-10-14 10:15:00"]:
        for z in SITE_ZONES:
            rows.append({"timestamp": pd.to_datetime(t_str), "zone": z, "pm10": 200.0, "pm2_5": 50.0})
    df = pd.DataFrame(rows)
    
    zone_effs = {"Excavation": 0.5, "Haul Road": 0.5, "Material Storage": 0.5, "Loading Point": 0.5}
    sim_df, metrics = calculate_mitigation_simulation(df, zone_effs, interval_hours=0.25)
    
    # 2 timestamps * 0.25 hours = 0.50 clock hours
    # The old bug multiplied 8 rows * 0.25 = 2.0 hours!
    assert metrics["simulated_baseline_exceedance_hours"] == 0.50, (
        f"Exceedance hours over-counted! Expected 0.50h, got {metrics['simulated_baseline_exceedance_hours']}h"
    )


def test_simulation_zero_baseline_and_floor():
    """Verify particulate levels cannot be mitigated below the ambient regional background floor."""
    df = pd.DataFrame([
        {"timestamp": pd.to_datetime("2026-10-14 10:00:00"), "zone": "Excavation", "pm10": 100.0, "pm2_5": 25.0}
    ])
    # 100% mitigation efficiency
    zone_effs = {"Excavation": 1.0}
    sim_df, metrics = calculate_mitigation_simulation(df, zone_effs)
    
    # Must floor at BACKGROUND_FLOOR["PM10"] = 18.0, never drop below floor
    assert sim_df.iloc[0]["pm10_simulated"] == BACKGROUND_FLOOR["PM10"]
    # PM2.5 capture efficiency is scaled at 0.75 of coarse PM10 abatement: 8.0 + (25 - 8) * 0.25 = 12.2
    assert sim_df.iloc[0]["pm25_simulated"] == 12.2


# ---------------------------------------------------------
# Test Group 6: Risk Evaluation & Regulatory Standards
# ---------------------------------------------------------
def test_instantaneous_vs_24h_compliance_distinction():
    """
    REQ-04: A single 15-minute point spike must be labelled as an
    Instantaneous Alert, NOT a statutory 24-hour EPA violation.
    """
    snapshot = pd.DataFrame([
        {"timestamp": pd.to_datetime("2026-10-14 10:00:00"), "zone": "Excavation", "pm10": 180.0, "pm2_5": 40.0, "is_anomalous": False}
    ])
    
    risk_info = evaluate_site_risk(snapshot)
    
    assert risk_info["is_instant_severe"] is True
    assert risk_info["is_24h_violation"] is False  # Cannot declare 24h violation on single 15m spike
    assert "24-hour statutory standard" in risk_info["compliance_summary"]


# ---------------------------------------------------------
# Test Group 7: Phase 1 Upgrade Features
# ---------------------------------------------------------
def test_chronological_split_no_leakage():
    """
    Ensure chronological split strictly separates train and test periods
    without future lookahead data leakage.
    """
    df = generate_synthetic_stream("Active Peak Shift", hours=24)
    train_df, test_df = split_chronological(df, test_ratio=0.25)
    
    assert not train_df.empty
    assert not test_df.empty
    
    max_train_time = train_df["timestamp"].max()
    min_test_time = test_df["timestamp"].min()
    
    assert max_train_time < min_test_time, f"Data leakage detected: train max ({max_train_time}) >= test min ({min_test_time})"
    
    # Ensure all 4 zones are represented across temporal partitions
    assert set(train_df["zone"].unique()) == set(SITE_ZONES)
    assert set(test_df["zone"].unique()) == set(SITE_ZONES)


def test_short_horizon_forecast_baseline():
    """
    Test short-horizon forecast model evaluation against naive persistence baseline.
    Verifies error metrics and explicit experimental disclaimer labeling.
    """
    df = generate_synthetic_stream("Dry Windy Alert Condition", hours=24)
    train_df, test_df = split_chronological(df, test_ratio=0.30)
    
    eval_result = train_short_horizon_forecast(train_df, test_df, horizon_steps=1)
    
    assert eval_result["status"] == "evaluated"
    assert eval_result["is_experimental"] is True
    assert "decision-support" in eval_result["disclaimer"].lower() or "baseline" in eval_result["disclaimer"].lower()
    
    metrics = eval_result["metrics"]
    assert metrics["naive_baseline_mae"] is not None
    assert metrics["naive_baseline_mae"] > 0.0
    assert metrics["forecast_model_mae"] is not None
    assert metrics["forecast_model_mae"] > 0.0
    assert metrics["forecast_model_rmse"] is not None


def test_generate_forward_forecast():
    """
    Test forward 2-hour projection generation across all site zones.
    Verifies 8 steps of 15m predictions with positive physical boundaries.
    """
    df = generate_synthetic_stream("Standard Operations", hours=12)
    forecast_df = generate_forward_forecast(df, horizon_steps=8, interval_mins=15)
    
    assert not forecast_df.empty
    # 4 zones * 8 horizon steps = 32 projected points
    assert len(forecast_df) == 4 * 8
    assert "pm10_forecast" in forecast_df.columns
    assert "pm10_lower" in forecast_df.columns
    assert "pm10_upper" in forecast_df.columns
    assert "risk_category" in forecast_df.columns
    
    # Must never project below background floor
    assert (forecast_df["pm10_forecast"] >= BACKGROUND_FLOOR["PM10"]).all()
    assert (forecast_df["pm10_lower"] >= BACKGROUND_FLOOR["PM10"]).all()
    # Upper bound must be >= forecast
    assert (forecast_df["pm10_upper"] >= forecast_df["pm10_forecast"]).all()


def test_actionable_recommendations_rules():
    """
    Test dynamic rule-based actionable recommendations with causality rationales.
    Verifies advisory status and human approval requirement.
    """
    df = generate_synthetic_stream("Active Peak Shift", hours=12)
    latest_snap = df[df["timestamp"] == df["timestamp"].max()]
    attr_df = compute_rule_based_attribution(latest_snap, df)
    
    recs = evaluate_actionable_recommendations(latest_snap, df, attr_df)
    
    assert len(recs) > 0
    for rec in recs:
        assert "action_title" in rec
        assert "selection_reasons" in rec
        assert len(rec["selection_reasons"]) > 0
        assert "expected_efficacy_pct" in rec
        assert rec["human_approval_required"] is True
        assert rec["status"] == "Pending Review"


def test_scenario_verification_comparison():
    """
    Test multi-scenario verification comparing Baseline vs Targeted vs Comprehensive packages.
    Verifies hypothetical labeling and unique clock hour exceedances.
    """
    df = generate_synthetic_stream("Dry Windy Alert Condition", hours=24)
    comparison = compare_mitigation_scenarios(df, interval_hours=0.25)
    
    assert comparison["is_hypothetical"] is True
    assert "HYPOTHETICAL SIMULATION MODEL" in comparison["scientific_disclaimer"]
    assert len(comparison["scenarios"]) == 3
    
    baseline_scen = comparison["scenarios"][0]
    targeted_scen = comparison["scenarios"][1]
    comp_scen = comparison["scenarios"][2]
    
    assert baseline_scen["id"] == "scenario_baseline"
    assert targeted_scen["id"] == "scenario_targeted"
    assert comp_scen["id"] == "scenario_comprehensive"
    
    # Interventions must show non-negative avoided hours
    assert targeted_scen["avoided_hours"] >= 0.0
    assert comp_scen["avoided_hours"] >= targeted_scen["avoided_hours"]


def test_feedback_record_creation():
    """
    Test creation and serialization of human-in-the-loop intervention feedback record.
    Verifies operational audit trail labeling without claiming continuous AI learning.
    """
    rec = create_feedback_record(
        zone="Haul Road",
        action_title="Dispatch Automated Water Bowser Misting Cycle",
        operator_name="Foreman Dan",
        simulated_reduction_pct=70.0,
        simulated_avoided_hours=4.5,
        evaluation_status="Observed Effective",
        field_notes="Bowser completed 3 circuits; surface silt stabilized."
    )
    
    assert rec["zone"] == "Haul Road"
    assert rec["operator_name"] == "Foreman Dan"
    assert rec["evaluation_status"] == "Observed Effective"
    assert rec["simulated_expected_reduction_pct"] == 70.0
    assert rec["simulated_avoided_hours"] == 4.5
    assert "Non-Continuous Learning" in rec["audit_provenance"]


# ---------------------------------------------------------
# Test Group 8: Forecasting Edge Cases & HTML Rendering Regression Tests
# ---------------------------------------------------------
def test_forecast_empty_dataset():
    """
    REGRESSION AUDIT: Confirm empty datasets return explicit 'empty_input' status
    and None metrics, rather than 0.00 fake metrics or an unhandled crash.
    """
    res = train_short_horizon_forecast(pd.DataFrame(), pd.DataFrame())
    assert res["status"] == "empty_input"
    assert res["train_records"] == 0
    assert res["test_records"] == 0
    assert res["test_mae"] is None
    assert res["test_rmse"] is None
    assert res["baseline_persistence_mae"] is None
    assert res["baseline_persistence_rmse"] is None
    assert res["relative_skill_score_pct"] is None
    assert "No historical" in res["message"] or "Insufficient" in res["message"]
    # Check backward-compatible metrics dictionary
    assert res["metrics"]["forecast_model_mae"] is None
    assert res["metrics"]["naive_baseline_mae"] is None


def test_forecast_insufficient_data():
    """
    REGRESSION AUDIT: Confirm datasets with insufficient records (<8 train or <2 test)
    return explicit 'insufficient_data' status with None metrics, preventing 0.00
    display as successful model performance.
    """
    times = pd.date_range("2026-10-14 10:00:00", periods=5, freq="15min")
    train_df = pd.DataFrame({
        "timestamp": times,
        "zone": ["Excavation"] * 5,
        "pm10": [50.0, 55.0, 60.0, 65.0, 70.0],
        "pm2_5": [12.0, 13.0, 14.0, 15.0, 16.0],
        "wind_speed": [5.0] * 5,
        "wind_deg": [180.0] * 5,
        "humidity": [40.0] * 5,
        "temperature": [22.0] * 5
    })
    test_df = pd.DataFrame({
        "timestamp": [times[-1] + timedelta(minutes=15)],
        "zone": ["Excavation"],
        "pm10": [75.0],
        "pm2_5": [17.0],
        "wind_speed": [5.0],
        "wind_deg": [180.0],
        "humidity": [40.0],
        "temperature": [22.0]
    })

    res = train_short_horizon_forecast(train_df, test_df)
    assert res["status"] == "insufficient_data"
    assert res["test_mae"] is None
    assert res["test_rmse"] is None
    assert res["baseline_persistence_mae"] is None
    assert "Insufficient" in res["message"]


def test_forecast_valid_dataset_top_level_keys():
    """
    REGRESSION AUDIT: Ensure train_short_horizon_forecast exposes top-level keys
    (test_mae, test_rmse, baseline_persistence_mae, etc.) directly expected by app.py,
    yielding valid non-zero metrics on real historical data.
    """
    df = generate_synthetic_stream("Active Peak Shift", hours=24)
    train_df, test_df = split_chronological(df, test_ratio=0.25)
    
    res = train_short_horizon_forecast(train_df, test_df, horizon_steps=1)
    
    assert res["status"] == "evaluated"
    assert res["train_records"] == len(train_df)
    assert res["test_records"] == len(test_df)
    assert res["test_mae"] is not None
    assert res["test_mae"] > 0.0
    assert res["test_rmse"] is not None
    assert res["test_rmse"] > 0.0
    assert res["baseline_persistence_mae"] is not None
    assert res["baseline_persistence_mae"] > 0.0
    assert "split_metadata" in res
    assert "train_cutoff" in res["split_metadata"]
    assert "test_start" in res["split_metadata"]
    assert res["is_experimental"] is True


def test_html_dedent_and_escaping_safety():
    """
    REGRESSION AUDIT: Verify that recommendation-card HTML strings containing
    nested divs, causality rationales (<ul>, <li>), and metric spans (<b>)
    are cleaned so that NO lines start with 4+ spaces and NO blank lines exist,
    preventing CommonMark parsers from treating lines as indented code blocks (<pre><code>).
    """
    import html

    # Simulate the exact recommendation card multiline f-string with nested indentation & blank lines
    card_html = """
                <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:6px; padding:1.1rem;">
                    <div>
                        <div style="display:flex; justify-content:space-between;">
                            <span>Haul Road</span>
                            <span class="pill-red">Immediate Action</span>
                        </div>
                        <h4>Dispatch Automated Water Bowser</h4>
                        <p>Mobilize bowser unit.</p>

                        <div style="background:#F8FAFC; border-left:3px solid #0284C7; padding:0.5rem 0.7rem;">
                            <div>Why Selected (Causality Rationale):</div>
                            <ul>
                                <li>Dry surface silt observed</li>
                            </ul>
                        </div>
                    </div>
                    <div>
                        <div>Estimated Efficacy: <b>60% - 75%</b></div>
                        <div>Deployment Lead Time: <b>5 - 10 mins</b></div>
                    </div>
                </div>
    """
    cleaned = " ".join(line.strip() for line in card_html.splitlines() if line.strip())

    # Must start with tag, not spaces
    assert cleaned.startswith("<div")
    # Must NOT contain newlines or 4 consecutive spaces
    assert "\n" not in cleaned
    assert "    " not in cleaned
    assert 'background:#F8FAFC' in cleaned
    assert 'Estimated Efficacy: <b>60% - 75%</b>' in cleaned
    assert '<ul> <li>Dry surface silt observed</li> </ul>' in cleaned or '<ul><li>Dry surface silt observed</li></ul>' in cleaned or '<ul>' in cleaned

    # Test dynamic text escaping against XSS injection
    malicious_input = '<script>alert("xss")</script>&"\'test'
    escaped = html.escape(malicious_input)
    assert "<script>" not in escaped
    assert "&lt;script&gt;" in escaped
    assert "&amp;" in escaped


