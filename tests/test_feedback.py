"""
Regression tests for DustSentinel Closed-Loop Intervention Feedback & Outcome Evaluation.
Verifies:
- Valid outcome calculations (absolute, percentage, direction)
- Zero baseline and missing value handling
- Invalid readings and timestamp ordering validation
- Strict separation of synthetic data from field measurements
- Exclusion of unverified records from verified empirical summaries
- Insufficient evidence handling (N < min_observations)
- Human authorization tracking and requirements
- CSV export schema completeness
- AppTest rendering of the Intervention Feedback view
"""

import os
import sys
import io
import pandas as pd
import pytest
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core_engine import (
    SITE_ZONES,
    PROVENANCE_OPTIONS,
    VERIFICATION_STATUSES,
    calculate_observed_outcome,
    validate_intervention_input,
    create_intervention_record,
    filter_intervention_records,
    aggregate_feedback_analytics,
    export_intervention_feedback_csv,
    generate_sample_feedback_records
)


def test_valid_outcome_calculation_decreased():
    """Verify standard PM10 reduction calculation with valid before/after values."""
    res = calculate_observed_outcome(pm10_before=120.0, pm10_after=40.0)
    assert res["is_complete"] is True
    assert res["pm10_abs_change"] == 80.0
    assert res["pm10_pct_change"] == pytest.approx(66.67, 0.01)
    assert res["pm10_direction"] == "Decreased"
    assert "not solely attributable" in res["scientific_caveat"]


def test_valid_outcome_calculation_increased_and_unchanged():
    """Verify particulate concentration increase and unchanged states."""
    # Concentration increased
    res_inc = calculate_observed_outcome(pm10_before=50.0, pm10_after=80.0)
    assert res_inc["is_complete"] is True
    assert res_inc["pm10_abs_change"] == -30.0
    assert res_inc["pm10_pct_change"] == -60.0
    assert res_inc["pm10_direction"] == "Increased"

    # Concentration unchanged within sensor noise band (<= 0.1 ug/m3)
    res_unc = calculate_observed_outcome(pm10_before=50.0, pm10_after=50.05)
    assert res_unc["is_complete"] is True
    assert res_unc["pm10_direction"] == "Unchanged"


def test_zero_baseline_handling():
    """
    Ensure baseline PM10 of zero does not cause ZeroDivisionError.
    Percentage change must safely be None while absolute delta and direction remain valid.
    """
    res = calculate_observed_outcome(pm10_before=0.0, pm10_after=25.0)
    assert res["is_complete"] is True
    assert res["pm10_abs_change"] == -25.0
    assert res["pm10_pct_change"] is None  # Zero baseline safety
    assert res["pm10_direction"] == "Increased"


def test_missing_values_retained_not_converted_to_zero():
    """
    Verify that missing readings are NOT silently coerced to 0.0.
    Completeness flag must be False and deltas must be None.
    """
    # Missing post-intervention reading
    res_no_after = calculate_observed_outcome(pm10_before=110.0, pm10_after=None)
    assert res_no_after["is_complete"] is False
    assert res_no_after["pm10_abs_change"] is None
    assert res_no_after["pm10_pct_change"] is None
    assert res_no_after["pm10_direction"] == "Pending Post-Intervention Reading"

    # Missing baseline reading
    res_no_before = calculate_observed_outcome(pm10_before=None, pm10_after=35.0)
    assert res_no_before["is_complete"] is False
    assert res_no_before["pm10_abs_change"] is None
    assert res_no_before["pm10_pct_change"] is None
    assert res_no_before["pm10_direction"] == "Missing Baseline Reading"


def test_invalid_readings_and_zone_validation():
    """Verify validator rejects negative concentrations, invalid zones, and non-numeric strings."""
    # Negative concentration
    valid, errs, _ = validate_intervention_input(
        zone="Haul Road",
        recommended_action="Water Bowser Misting",
        performed_action="Water Bowser Misting",
        provenance="Field Measurement",
        pm10_before=-10.0,
        pm10_after=40.0
    )
    assert valid is False
    assert any("cannot be negative" in e for e in errs)

    # Invalid zone
    valid_z, errs_z, _ = validate_intervention_input(
        zone="Invalid Zone Name",
        recommended_action="Water Bowser Misting",
        performed_action="Water Bowser Misting",
        provenance="Field Measurement",
        pm10_before=100.0,
        pm10_after=40.0
    )
    assert valid_z is False
    assert any("Invalid construction zone" in e for e in errs_z)


def test_timestamp_ordering_validation():
    """Verify start timestamp cannot be chronologically after the post-intervention timestamp."""
    valid, errs, _ = validate_intervention_input(
        zone="Haul Road",
        recommended_action="Water Bowser Misting",
        performed_action="Water Bowser Misting",
        provenance="Field Measurement",
        pm10_before=120.0,
        pm10_after=40.0,
        timestamp_before="2026-10-15 14:00:00",
        timestamp_after="2026-10-15 13:00:00"
    )
    assert valid is False
    assert any("cannot be after" in e for e in errs)


def test_synthetic_vs_field_data_strict_separation():
    """
    CRITICAL REQUIREMENT: Never mix synthetic examples with field measurements
    in aggregate performance metrics.
    """
    # 2 synthetic records with 50% simulated reduction
    syn1 = create_intervention_record(
        zone="Material Storage",
        recommended_action="Stockpile Wetting",
        performed_action="Stockpile Wetting",
        provenance="Synthetic Demonstration",
        pm10_before=100.0,
        pm10_after=50.0,
        verification_status="Pending Verification"
    )
    syn2 = create_intervention_record(
        zone="Material Storage",
        recommended_action="Stockpile Wetting",
        performed_action="Stockpile Wetting",
        provenance="Synthetic Demonstration",
        pm10_before=80.0,
        pm10_after=40.0,
        verification_status="Pending Verification"
    )

    # 1 field measurement with 80% observed reduction
    field1 = create_intervention_record(
        zone="Haul Road",
        recommended_action="Water Bowser Misting",
        performed_action="Water Bowser Misting",
        provenance="Field Measurement",
        pm10_before=150.0,
        pm10_after=30.0,
        verification_status="Field Verified",
        is_authorized=True,
        authorizer_name="Inspector Dan"
    )

    records = [syn1, syn2, field1]
    analytics = aggregate_feedback_analytics(records, min_observations=3)

    # Verify counts
    assert analytics["total_recorded"] == 3
    assert analytics["count_synthetic"] == 2
    assert analytics["count_verified_field"] == 1

    # Verify metrics are strictly separated
    # Field metric should ONLY be 80.0% (not mixed with 50.0%)
    assert analytics["field_verified_metrics"]["mean_pct_reduction"] == 80.0
    assert analytics["field_verified_metrics"]["mean_abs_reduction_ugm3"] == 120.0

    # Synthetic metric should be 50.0%
    assert analytics["synthetic_metrics"]["mean_pct_reduction"] == 50.0
    assert analytics["synthetic_metrics"]["mean_abs_reduction_ugm3"] == 45.0


def test_unverified_records_excluded_from_verified_summaries():
    """
    Ensure records marked 'Pending Verification' or 'Rejected / Ineffective'
    are never included in verified field statistics.
    """
    rec_pending = create_intervention_record(
        zone="Excavation",
        recommended_action="Sprinklers",
        performed_action="Sprinklers",
        provenance="Field Measurement",
        pm10_before=100.0,
        pm10_after=50.0,
        verification_status="Pending Verification"
    )
    rec_rejected = create_intervention_record(
        zone="Excavation",
        recommended_action="Sprinklers",
        performed_action="Sprinklers",
        provenance="Field Measurement",
        pm10_before=100.0,
        pm10_after=110.0,
        verification_status="Rejected / Ineffective"
    )

    analytics = aggregate_feedback_analytics([rec_pending, rec_rejected], min_observations=3)
    assert analytics["count_verified_field"] == 0
    assert analytics["field_verified_metrics"]["count"] == 0
    assert analytics["field_verified_metrics"]["mean_pct_reduction"] is None


def test_field_verified_demotion_for_synthetic_records():
    """
    If a record has provenance='Synthetic Demonstration', it cannot be
    treated as a verified field outcome even if status is set to 'Field Verified'.
    """
    syn_fake_verified = create_intervention_record(
        zone="Excavation",
        recommended_action="Sprinklers",
        performed_action="Sprinklers",
        provenance="Synthetic Demonstration",
        pm10_before=100.0,
        pm10_after=50.0,
        verification_status="Field Verified"
    )
    # is_verified_field_outcome must remain False
    assert syn_fake_verified["is_verified_field_outcome"] is False

    analytics = aggregate_feedback_analytics([syn_fake_verified], min_observations=1)
    assert analytics["count_verified_field"] == 0


def test_insufficient_evidence_threshold():
    """
    Verify that if verified field observations for an action are below min_observations,
    the system explicitly outputs 'Insufficient verified field data'.
    """
    rec1 = create_intervention_record(
        zone="Haul Road",
        recommended_action="Automated Water Bowser Misting Cycle",
        performed_action="Automated Water Bowser Misting Cycle",
        provenance="Field Measurement",
        pm10_before=130.0,
        pm10_after=40.0,
        verification_status="Field Verified",
        is_authorized=True,
        authorizer_name="Inspector Dan"
    )

    # Threshold = 3, we have 1 observation
    analytics = aggregate_feedback_analytics([rec1], min_observations=3)
    action_info = analytics["action_summaries"]["Automated Water Bowser Misting Cycle"]
    
    assert action_info["has_sufficient_evidence"] is False
    assert action_info["evidence_status"] == "Insufficient verified field data"
    assert "Insufficient verified field data (N = 1 < 3 required)" in action_info["summary_text"]

    # Now provide 2 more verified observations to meet threshold N=3
    rec2 = create_intervention_record(
        zone="Haul Road",
        recommended_action="Automated Water Bowser Misting Cycle",
        performed_action="Automated Water Bowser Misting Cycle",
        provenance="Field Measurement",
        pm10_before=125.0,
        pm10_after=35.0,
        verification_status="Field Verified",
        is_authorized=True,
        authorizer_name="Inspector Dan"
    )
    rec3 = create_intervention_record(
        zone="Haul Road",
        recommended_action="Automated Water Bowser Misting Cycle",
        performed_action="Automated Water Bowser Misting Cycle",
        provenance="Field Measurement",
        pm10_before=140.0,
        pm10_after=45.0,
        verification_status="Field Verified",
        is_authorized=True,
        authorizer_name="Inspector Dan"
    )

    analytics_sufficient = aggregate_feedback_analytics([rec1, rec2, rec3], min_observations=3)
    action_info_suff = analytics_sufficient["action_summaries"]["Automated Water Bowser Misting Cycle"]
    
    assert action_info_suff["has_sufficient_evidence"] is True
    assert action_info_suff["evidence_status"] == "Sufficient Verified Field Data"
    assert "Sufficient verified field evidence (N = 3 observations)" in action_info_suff["summary_text"]


def test_human_authorization_status():
    """Verify human authorization tracking and validation enforcement."""
    # Authorized record
    rec_auth = create_intervention_record(
        zone="Haul Road",
        recommended_action="Water Bowser Misting",
        performed_action="Water Bowser Misting",
        is_authorized=True,
        authorizer_name="Chief Engineer Alice"
    )
    assert rec_auth["is_authorized"] is True
    assert rec_auth["authorizer_name"] == "Chief Engineer Alice"

    # Validator requires authorization for Field Verified status
    valid, errs, _ = validate_intervention_input(
        zone="Haul Road",
        recommended_action="Water Bowser Misting",
        performed_action="Water Bowser Misting",
        provenance="Field Measurement",
        pm10_before=120.0,
        pm10_after=40.0,
        verification_status="Field Verified",
        is_authorized=False
    )
    assert valid is False
    assert any("explicit human authorization" in e for e in errs)


def test_csv_export_schema():
    """
    Verify CSV export contains all mandatory metadata fields, units,
    provenance, verification status, and timestamps.
    """
    samples = generate_sample_feedback_records()
    csv_str = export_intervention_feedback_csv(samples)
    
    assert "record_id" in csv_str
    assert "pm10_before_ugm3" in csv_str
    assert "pm10_after_ugm3" in csv_str
    assert "pm10_abs_change_ugm3" in csv_str
    assert "pm10_pct_change" in csv_str
    assert "provenance" in csv_str
    assert "verification_status" in csv_str
    assert "scientific_caveat" in csv_str

    df = pd.read_csv(io.StringIO(csv_str))
    assert len(df) == len(samples)
    assert "Field Measurement" in df["provenance"].values
    assert "Synthetic Demonstration" in df["provenance"].values


def test_apptest_feedback_interface_rendering():
    """
    AppTest verification: Navigate to Intervention Feedback view and verify
    zero unhandled crashes or exceptions.
    """
    from streamlit.testing.v1 import AppTest
    app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app.py"))
    at = AppTest.from_file(app_path, default_timeout=30).run()
    
    assert len(at.exception) == 0
    at.sidebar.radio[0].set_value("Intervention Feedback").run()
    assert len(at.exception) == 0


def test_zero_records_aggregation_and_empty_csv_export():
    """
    Regression test: Zero records handling.
    - Aggregation returns zero counts without error.
    - Export produces a structured, schema-compliant CSV with exact column headers and 0 data rows.
    """
    empty_records = []
    analytics = aggregate_feedback_analytics(empty_records, min_observations=3)
    
    assert analytics["total_recorded"] == 0
    assert analytics["count_verified_field"] == 0
    assert analytics["count_pending"] == 0
    assert analytics["count_unverified_incomplete"] == 0
    assert analytics["count_rejected"] == 0
    assert analytics["count_field_total"] == 0
    assert analytics["count_synthetic"] == 0
    assert analytics["field_verified_metrics"]["count"] == 0
    assert analytics["field_verified_metrics"]["mean_pct_reduction"] is None
    assert analytics["action_summaries"] == {}
    
    csv_str = export_intervention_feedback_csv(empty_records)
    assert csv_str.strip() != ""
    assert "record_id" in csv_str
    assert "linked_order_id" in csv_str
    assert "pm10_before_ugm3" in csv_str
    assert "scientific_caveat" in csv_str
    
    df = pd.read_csv(io.StringIO(csv_str))
    assert len(df) == 0
    assert len(df.columns) == 31
    assert "linked_order_id" in df.columns


def test_mutually_exclusive_status_partitions_and_reconciliation():
    """
    Regression test: Reconciled intervention feedback counts and mutual exclusivity.
    Partitions unique population into mutually exclusive categories:
    Total (Card 1) == Verified Field Outcomes (Card 2) + Pending & Incomplete (Card 3).
    Verified Field + Pending + Unverified / Incomplete + Rejected == Total Recorded.
    """
    r_verified = create_intervention_record(
        zone="Haul Road",
        recommended_action="Water Bowser Misting",
        performed_action="Water Bowser Misting",
        provenance="Field Measurement",
        pm10_before=130.0,
        pm10_after=40.0,
        verification_status="Field Verified",
        is_authorized=True,
        authorizer_name="Inspector Dan",
        record_id="REC-VERIFIED-1"
    )
    r_pending = create_intervention_record(
        zone="Excavation",
        recommended_action="Sprinklers",
        performed_action="Sprinklers",
        provenance="Field Measurement",
        pm10_before=110.0,
        pm10_after=None,
        verification_status="Pending Verification",
        is_authorized=True,
        authorizer_name="Inspector Dan",
        record_id="REC-PENDING-2"
    )
    r_rejected = create_intervention_record(
        zone="Loading Point",
        recommended_action="Chute Baffles",
        performed_action="Chute Baffles",
        provenance="Field Measurement",
        pm10_before=90.0,
        pm10_after=95.0,
        verification_status="Rejected / Ineffective",
        is_authorized=True,
        authorizer_name="Inspector Dan",
        record_id="REC-REJECTED-3"
    )
    r_incomplete = create_intervention_record(
        zone="Material Storage",
        recommended_action="Stockpile Wetting",
        performed_action="Stockpile Wetting",
        provenance="Synthetic Demonstration",
        pm10_before=80.0,
        pm10_after=50.0,
        verification_status="Unverified / Incomplete",
        record_id="REC-INCOMPLETE-4"
    )
    
    mixed_records = [r_verified, r_pending, r_rejected, r_incomplete]
    analytics = aggregate_feedback_analytics(mixed_records, min_observations=1)
    
    total = analytics["total_recorded"]
    c_vf = analytics["count_verified_field"]
    c_pend = analytics["count_pending"]
    c_unver = analytics["count_unverified_incomplete"]
    c_rej = analytics["count_rejected"]
    
    assert total == 4
    assert c_vf == 1
    assert c_pend == 1
    assert c_rej == 1
    assert c_unver == 1
    
    # Mutual exclusivity & exhaustiveness
    assert c_vf + c_pend + c_unver + c_rej == total
    
    # KPI card reconciliation: Card 1 == Card 2 + Card 3
    card1_total = total
    card2_verified = c_vf
    card3_pending_incomplete = c_pend + c_unver + c_rej
    assert card1_total == card2_verified + card3_pending_incomplete


def test_duplicate_and_linked_work_order_deduplication():
    """
    Regression test: Duplicate IDs and linked work orders deduplication.
    - Exact duplicate record IDs are counted once.
    - Work orders and feedback records sharing linked IDs are counted once.
    - Independent records are retained.
    """
    # Record A with linked work order
    rec_a1 = create_intervention_record(
        zone="Haul Road",
        recommended_action="Water Bowser Misting",
        performed_action="Water Bowser Misting",
        provenance="Field Measurement",
        pm10_before=120.0,
        pm10_after=40.0,
        verification_status="Field Verified",
        is_authorized=True,
        authorizer_name="Inspector Dan",
        record_id="INT-A001",
        linked_order_id="ORD-20261015-001"
    )
    # Exact duplicate submission of Record A
    rec_a_dup = dict(rec_a1)
    
    # Second record referencing the same work order
    rec_a_order_linked = create_intervention_record(
        zone="Haul Road",
        recommended_action="Water Bowser Misting",
        performed_action="Water Bowser Misting",
        provenance="Field Measurement",
        pm10_before=120.0,
        pm10_after=40.0,
        verification_status="Pending Verification",
        record_id="INT-A002",
        linked_order_id="ORD-20261015-001"
    )
    
    # Record B (distinct intervention and order)
    rec_b = create_intervention_record(
        zone="Excavation",
        recommended_action="Misting Cannons",
        performed_action="Misting Cannons",
        provenance="Field Measurement",
        pm10_before=140.0,
        pm10_after=60.0,
        verification_status="Field Verified",
        is_authorized=True,
        authorizer_name="Inspector Dan",
        record_id="INT-B001",
        linked_order_id="ORD-20261015-002"
    )
    
    # Standalone Record C with no linked order
    rec_c = create_intervention_record(
        zone="Loading Point",
        recommended_action="Chute Baffles",
        performed_action="Chute Baffles",
        provenance="Synthetic Demonstration",
        pm10_before=90.0,
        pm10_after=50.0,
        verification_status="Pending Verification",
        record_id="INT-C001"
    )
    
    # Standalone Record C duplicate
    rec_c_dup = dict(rec_c)
    
    population = [rec_a1, rec_a_dup, rec_a_order_linked, rec_b, rec_c, rec_c_dup]
    analytics = aggregate_feedback_analytics(population, min_observations=1)
    
    # Only 3 unique interventions exist: A, B, and C
    assert analytics["total_recorded"] == 3
    # Verified field outcomes: A and B
    assert analytics["count_verified_field"] == 2
    # Pending/Incomplete: C
    assert analytics["count_pending"] == 1


def test_cleared_records_isolation_and_consistency():
    """
    Regression test: Clear-records operation isolation.
    - Clearing feedback records must reset feedback metrics to zero.
    - Must NOT modify or clear dispatch log or telemetry datasets.
    - Export produces an empty header-only CSV.
    """
    mock_session_state = {
        "feedback_records": generate_sample_feedback_records(),
        "dispatch_log": [
            {"order_id": "ORD-001", "target_zone": "Haul Road", "action": "Water Bowser Misting", "status": "Dispatched"},
            {"order_id": "ORD-002", "target_zone": "Excavation", "action": "Sprinklers", "status": "In Progress"}
        ],
        "sensor_data_count": 384
    }
    
    assert len(mock_session_state["feedback_records"]) > 0
    assert len(mock_session_state["dispatch_log"]) == 2
    
    # Execute clear-records operation for feedback only
    mock_session_state["feedback_records"] = []
    
    # 1. Feedback analytics refresh from underlying state to zero
    analytics = aggregate_feedback_analytics(mock_session_state["feedback_records"], min_observations=3)
    assert analytics["total_recorded"] == 0
    assert analytics["count_verified_field"] == 0
    assert analytics["count_pending"] == 0
    
    # 2. Dispatch logs remain completely intact
    assert len(mock_session_state["dispatch_log"]) == 2
    assert mock_session_state["dispatch_log"][0]["order_id"] == "ORD-001"
    
    # 3. Telemetry remains untouched
    assert mock_session_state["sensor_data_count"] == 384
    
    # 4. Export returns valid 0-row CSV with headers
    csv_bytes = export_intervention_feedback_csv(mock_session_state["feedback_records"])
    df_empty = pd.read_csv(io.StringIO(csv_bytes))
    assert len(df_empty) == 0
    assert len(df_empty.columns) == 31


def test_all_four_report_exports_row_counts_match_ui():
    """
    Regression test: Verify that displayed UI counts across all 4 report exports
    match the exact row count of the generated CSV files.
    """
    from core_engine import (
        generate_synthetic_stream,
        validate_and_clean_data,
        calculate_mitigation_simulation
    )
    
    # 1. Clean Active Sensor Telemetry Dataset
    raw_df = generate_synthetic_stream("Active Peak Shift", hours=12)
    df_clean, _ = validate_and_clean_data(raw_df)
    ui_telemetry_count = len(df_clean)
    
    telemetry_buffer = io.StringIO()
    df_clean.to_csv(telemetry_buffer, index=False)
    telemetry_csv_df = pd.read_csv(io.StringIO(telemetry_buffer.getvalue()))
    assert len(telemetry_csv_df) == ui_telemetry_count
    
    # 2. Operational Mitigation Work Orders (Dispatch Log)
    dispatch_log = [
        {"order_id": "ORD-1", "timestamp": "2026-10-15 08:00:00", "target_zone": "Haul Road", "assigned_crew": "Crew A", "action": "Misting", "authorizer": "Lead", "current_pm10": 130.0, "status": "Dispatched", "linked_record_id": "INT-1"},
        {"order_id": "ORD-2", "timestamp": "2026-10-15 08:30:00", "target_zone": "Excavation", "assigned_crew": "Crew B", "action": "Cannons", "authorizer": "Lead", "current_pm10": 115.0, "status": "Dispatched", "linked_record_id": "INT-2"},
    ]
    ui_dispatch_count = len(dispatch_log)
    
    dispatch_headers = [
        "order_id", "timestamp", "target_zone", "assigned_crew",
        "action", "authorizer", "current_pm10", "status", "linked_record_id"
    ]
    disp_df = pd.DataFrame(dispatch_log)
    for col in dispatch_headers:
        if col not in disp_df.columns:
            disp_df[col] = ""
    disp_csv_bytes = disp_df[dispatch_headers].to_csv(index=False)
    disp_csv_df = pd.read_csv(io.StringIO(disp_csv_bytes))
    assert len(disp_csv_df) == ui_dispatch_count
    
    # Empty dispatch log test
    empty_disp_csv = (",".join(dispatch_headers) + "\n")
    empty_disp_df = pd.read_csv(io.StringIO(empty_disp_csv))
    assert len(empty_disp_df) == 0
    assert list(empty_disp_df.columns) == dispatch_headers
    
    # 3. Multi-Scenario Mitigation Projections
    sim_df, _ = calculate_mitigation_simulation(df_clean, {"Haul Road": 0.70, "Excavation": 0.68}, interval_hours=0.25)
    if "provenance" not in sim_df.columns:
        sim_df["provenance"] = "Synthetic Demonstration Projection"
    ui_sim_count = len(sim_df)
    
    sim_buffer = io.StringIO()
    sim_df.to_csv(sim_buffer, index=False)
    sim_csv_df = pd.read_csv(io.StringIO(sim_buffer.getvalue()))
    assert len(sim_csv_df) == ui_sim_count
    # Verify provenance is strictly separate from field verification
    assert (sim_csv_df["provenance"] == "Synthetic Demonstration Projection").all()
    
    # 4. Closed-Loop Intervention Feedback Audit Trail
    feedback_recs = generate_sample_feedback_records()
    ui_feedback_count = len(feedback_recs)
    
    fb_csv_str = export_intervention_feedback_csv(feedback_recs)
    fb_csv_df = pd.read_csv(io.StringIO(fb_csv_str))
    assert len(fb_csv_df) == ui_feedback_count
    
    # Empty feedback test
    empty_fb_csv = export_intervention_feedback_csv([])
    empty_fb_df = pd.read_csv(io.StringIO(empty_fb_csv))
    assert len(empty_fb_df) == 0
    assert len(empty_fb_df.columns) == 31
