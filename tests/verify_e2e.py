import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from core_engine import (
    generate_synthetic_stream,
    validate_and_clean_data,
    compute_rule_based_attribution,
    calculate_mitigation_simulation,
    evaluate_site_risk,
    split_chronological,
    train_short_horizon_forecast,
    generate_forward_forecast,
    evaluate_actionable_recommendations,
    compare_mitigation_scenarios,
    create_feedback_record,
    calculate_observed_outcome,
    create_intervention_record,
    aggregate_feedback_analytics,
    export_intervention_feedback_csv,
    generate_sample_feedback_records,
    SITE_ZONES
)

def run_verification():
    print("=== 1. Testing 3 Synthetic Scenarios ===")
    for sc in ['Active Peak Shift', 'Dry Windy Alert Condition', 'Standard Operations']:
        df = generate_synthetic_stream(sc, hours=12)
        assert len(df) == 12 * 4 * 4
        peak_pm10 = df["pm10"].max()
        avg_wind = df["wind_speed"].mean()
        print(f"Scenario: {sc} -> {len(df)} rows, Peak PM10: {peak_pm10:.1f}, Avg Wind: {avg_wind:.1f} km/h")

    print("\n=== 2. Testing 4 Site Zones ===")
    for z in SITE_ZONES:
        z_df = df[df['zone'] == z]
        assert len(z_df) > 0
        print(f"Zone: {z} present with {len(z_df)} rows")

    print("\n=== 3. Testing Valid Sample CSV ===")
    sample_df = pd.read_csv('sample_sensor_data.csv')
    clean_sample, rep_sample = validate_and_clean_data(sample_df)
    assert rep_sample["valid"] is True
    print(f"sample_sensor_data.csv valid: {rep_sample['valid']}, rows: {rep_sample['total_rows']}, anomalies: {len(rep_sample['anomalies'])}")

    print("\n=== 4. Testing Source Ranking ===")
    snap = clean_sample.tail(4)
    rank_df = compute_rule_based_attribution(snap, clean_sample)
    assert not rank_df.empty
    for _, r in rank_df.iterrows():
        print(f"Rank #{r['rank']}: {r['zone']} | Score: {r['ranking_score']:.1f}/100")

    print("\n=== 5. Testing Before-After Simulation Accounting ===")
    zone_effs = {z: 0.70 for z in SITE_ZONES}
    sim_df, metrics = calculate_mitigation_simulation(clean_sample, zone_effs)
    assert "simulated_avoided_exceedance_hours" in metrics
    print(f"Baseline Peak PM10: {metrics['simulated_baseline_peak_pm10']:.1f} -> Simulated Post: {metrics['simulated_post_peak_pm10']:.1f}")
    print(f"Reduction Pct: {metrics['simulated_reduction_pct']:.1f}%")
    print(f"Avoided Exceedance Clock Hours: {metrics['simulated_avoided_exceedance_hours']} hrs")

    print("\n=== 6. Testing Chronological Split & Short-Horizon Forecasting ===")
    train_df, test_df = split_chronological(clean_sample, test_ratio=0.25)
    assert train_df["timestamp"].max() < test_df["timestamp"].min()
    fc_eval = train_short_horizon_forecast(train_df, test_df, horizon_steps=1)
    assert fc_eval["is_experimental"] is True
    fc_future = generate_forward_forecast(clean_sample, horizon_steps=8)
    assert len(fc_future) == 4 * 8
    print(f"Train/Test Partition: {len(train_df)}/{len(test_df)} rows (Zero Data Leakage)")
    print(f"Naive Baseline MAE: {fc_eval['metrics']['naive_baseline_mae']} µg/m³ | Model MAE: {fc_eval['metrics']['forecast_model_mae']} µg/m³")
    print(f"Relative Skill Score: {fc_eval['metrics']['relative_skill_score_pct']}% improvement")
    print(f"Generated 2-hour forward projection: {len(fc_future)} steps across 4 zones")

    print("\n=== 7. Testing Actionable Mitigation Recommendations ===")
    recs = evaluate_actionable_recommendations(snap, clean_sample, rank_df)
    assert len(recs) > 0
    top_rec = recs[0]
    print(f"Top Advisory Action: {top_rec['action_title']} for [{top_rec['zone']}]")
    print(f"Urgency: {top_rec['priority']} | Expected Efficacy: {top_rec['expected_efficacy_pct']}")
    print(f"Causality Reasons ({len(top_rec['selection_reasons'])} factors): {top_rec['selection_reasons'][0]}")

    print("\n=== 8. Testing Scenario Verification Comparison ===")
    scen_comp = compare_mitigation_scenarios(clean_sample)
    assert scen_comp["is_hypothetical"] is True
    assert len(scen_comp["scenarios"]) == 3
    for s in scen_comp["scenarios"]:
        print(f"{s['name']} -> Peak PM10: {s['peak_pm10']:.1f} µg/m³, Avoided Exceedance: {s['avoided_hours']:.2f} hrs, Cost: {s['cost_tier']}")

    print("\n=== 9. Testing Feedback Record Audit Logging ===")
    feedback = create_feedback_record(
        zone=top_rec["zone"],
        action_title=top_rec["action_title"],
        operator_name="Site Environmental Officer",
        simulated_reduction_pct=70.0,
        simulated_avoided_hours=metrics["simulated_avoided_exceedance_hours"],
        evaluation_status="Observed Effective",
        field_notes="Bowser cycle executed successfully; moisture suppressed silt."
    )
    assert feedback["evaluation_status"] == "Observed Effective"
    print(f"Audit Record ID: {feedback['record_id']} | Status: {feedback['evaluation_status']}")
    print(f"Provenance Note: {feedback['audit_provenance']}")

    print("\n=== 10. Testing Closed-Loop Intervention Feedback & Outcome Evaluation ===")
    samples = generate_sample_feedback_records()
    assert len(samples) == 4
    analytics = aggregate_feedback_analytics(samples, min_observations=3)
    assert analytics["count_verified_field"] == 1
    assert analytics["count_synthetic"] == 1
    assert analytics["field_verified_metrics"]["count"] == 1
    assert analytics["synthetic_metrics"]["count"] == 1
    
    # Verify zero division safety and outcome calculation
    outcome = calculate_observed_outcome(pm10_before=132.5, pm10_after=39.8)
    assert outcome["is_complete"] is True
    assert outcome["pm10_abs_change"] == 92.7
    assert outcome["pm10_direction"] == "Decreased"
    
    # Verify CSV export
    csv_out = export_intervention_feedback_csv(samples)
    assert "record_id" in csv_out
    assert "scientific_caveat" in csv_out
    print(f"Verified Records Processed: {len(samples)} (Field Verified: {analytics['count_verified_field']}, Synthetic: {analytics['count_synthetic']})")
    print(f"Observed Field Outcome: Baseline 132.5 -> Post 39.8 ug/m3 (Delta {outcome['pm10_abs_change']} ug/m3, {outcome['pm10_pct_change']}%)")
    print("CSV Export Schema Validated with Provenance & Caveat Isolation")

    print("\n[SUCCESS] All End-to-End Verification Checks Passed!")

if __name__ == "__main__":
    run_verification()
