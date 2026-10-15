import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from streamlit.testing.v1 import AppTest

VIEWS = [
    "Overview",
    "Spatial Map & Plume",
    "Source Attribution",
    "Mitigation Playbook",
    "Intervention Sandbox",
    "Intervention Feedback",
    "Sensor Analytics",
    "Data Health & Pipeline",
    "Reports & Export",
    "Privacy Policy",
    "Terms of Use"
]

@pytest.mark.parametrize("view_name", VIEWS)
def test_each_view_renders_without_exceptions(view_name):
    """
    AppTest verification: Navigate to each dashboard view and verify
    zero exceptions or unhandled crashes occur during rendering.
    """
    app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app.py"))
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run()
    assert len(at.exception) == 0, f"Exception on initial load: {[e.value for e in at.exception]}"
    
    # Radio 0 is the Navigation Menu
    nav_radio = at.sidebar.radio[0]
    assert nav_radio.label == "Navigation Menu"
    
    nav_radio.set_value(view_name).run()
    assert len(at.exception) == 0, f"Exception rendering view '{view_name}': {[e.value for e in at.exception]}"


def test_sensor_analytics_valid_data_shows_positive_metrics():
    """
    AppTest verification: With default synthetic 24h stream, Sensor Analytics
    must display real, non-zero out-of-sample forecast metrics and zero lookahead split.
    """
    app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app.py"))
    at = AppTest.from_file(app_path, default_timeout=30).run()
    at.sidebar.radio[0].set_value("Sensor Analytics").run()
    
    assert len(at.exception) == 0
    metric_labels = [m.label for m in at.metric]
    assert "Test MAE (Model vs Persistence)" in metric_labels
    assert "Test RMSE (Model vs Persistence)" in metric_labels
    
    # Check that the test MAE metric value is positive and not 0.00
    test_mae_metric = next(m for m in at.metric if "Test MAE" in m.label)
    assert "0.00" not in test_mae_metric.value
    assert "µg/m³" in test_mae_metric.value


def test_sensor_analytics_insufficient_data_withholds_zero_metrics():
    """
    AppTest verification: With insufficient historical records (<8 train or <2 test),
    Sensor Analytics must display an explicit 'Insufficient data for evaluation' warning
    and must NOT render fake 0.00 metrics.
    """
    import pandas as pd
    app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app.py"))
    at = AppTest.from_file(app_path, default_timeout=30).run()
    
    times = pd.date_range("2026-10-14 10:00:00", periods=6, freq="15min")
    small_df = pd.DataFrame({
        "timestamp": times,
        "zone": ["Excavation"] * 6,
        "pm10": [50.0, 55.0, 60.0, 65.0, 70.0, 75.0],
        "pm2_5": [12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
        "wind_speed": [5.0] * 6,
        "wind_deg": [180.0] * 6,
        "humidity": [40.0] * 6,
        "temperature": [22.0] * 6
    })
    
    at.session_state["raw_df"] = small_df
    at.sidebar.radio[0].set_value("Sensor Analytics").run()
    
    assert len(at.exception) == 0
    warning_texts = [w.value for w in at.warning]
    assert any("Insufficient data for evaluation" in t for t in warning_texts)
    # Zero metrics must be withheld
    metric_labels = [m.label for m in at.metric]
    assert "Test MAE (Model vs Persistence)" not in metric_labels
