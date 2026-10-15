"""
DustSentinel: AI-Assisted Fugitive Dust Monitoring & Mitigation Dashboard
Tech For Earth Hackathon 2026 - Clean SaaS Environmental Intelligence Redesign
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
import math
import textwrap
import html
import sys
import os
import importlib

# Ensure application directory is in sys.path for robust module resolution
_app_dir = os.path.dirname(os.path.abspath(__file__))
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

# Ensure core_engine module cache is up to date across live reloads
import core_engine
importlib.reload(core_engine)

# Import pure testable domain logic from core engine
from core_engine import (
    SITE_ZONES,
    ZONE_COORDINATES,
    THRESHOLDS,
    BACKGROUND_FLOOR,
    get_cardinal_direction,
    compute_plume_heading,
    compute_angular_offset,
    validate_and_clean_data,
    generate_synthetic_stream,
    evaluate_site_risk,
    compute_rule_based_attribution,
    calculate_mitigation_simulation,
    split_chronological,
    train_short_horizon_forecast,
    generate_forward_forecast,
    evaluate_actionable_recommendations,
    compare_mitigation_scenarios,
    create_feedback_record,
    PROVENANCE_OPTIONS,
    VERIFICATION_STATUSES,
    ZONE_RECOMMENDED_ACTIONS,
    calculate_observed_outcome,
    validate_intervention_input,
    create_intervention_record,
    filter_intervention_records,
    aggregate_feedback_analytics,
    export_intervention_feedback_csv,
    generate_sample_feedback_records
)

# ---------------------------------------------------------
# Streamlit App Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="DustSentinel | Construction Dust Intelligence",
    page_icon="assets/favicon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

def render_html(html_str: str) -> None:
    """
    Renders application HTML cleanly in Streamlit without triggering Markdown
    indented code block parsing (<pre><code>).
    Strips all leading and trailing whitespace from every line, eliminating blank lines
    and leading spaces that trigger CommonMark code-block interpretations.
    Uses st.html for direct DOM insertion with DOMPurify sanitization.
    """
    cleaned = " ".join(line.strip() for line in html_str.splitlines() if line.strip())
    if hasattr(st, "html"):
        st.html(cleaned)
    else:
        st.markdown(cleaned, unsafe_allow_html=True)

# Custom CSS: Modern Clean SaaS Analytics Dashboard (Light Palette & Responsive)
st.markdown("""
<style>
    /* Global Typography & Light SaaS Base */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: #1E293B;
    }
    
    .stApp {
        background-color: #F8FAFC;
    }
    
    /* Main Content Container Padding Optimization */
    .main .block-container {
        padding-top: 1.25rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        max-width: 100% !important;
    }
    @media (max-width: 1024px) {
        .main .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            padding-top: 1rem !important;
        }
    }
    @media (max-width: 640px) {
        .main .block-container {
            padding-left: 0.65rem !important;
            padding-right: 0.65rem !important;
            padding-top: 0.75rem !important;
        }
    }
    
    /* Sidebar Surface & Constrained Width */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }
    @media (min-width: 769px) {
        [data-testid="stSidebar"] {
            min-width: 250px !important;
            max-width: 290px !important;
        }
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 1.25rem;
        padding-bottom: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    /* Brand Mark Box */
    .brand-mark-box {
        width: 36px;
        height: 36px;
        background: #0284C7;
        color: #FFFFFF;
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: -0.02em;
        flex-shrink: 0;
    }
    .brand-title {
        color: #0F172A;
        font-size: 1.2rem;
        font-weight: 700;
        letter-spacing: -0.015em;
        line-height: 1.2;
    }
    .brand-subtitle {
        color: #64748B;
        font-size: 0.8rem;
        font-weight: 400;
        margin-top: 0.15rem;
    }
    
    /* Header & Top Bar */
    .brand-header-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.85rem 1.15rem;
        background: #FFFFFF;
        border-radius: 6px;
        margin-bottom: 1.15rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px 0 rgba(15, 23, 42, 0.04);
        flex-wrap: wrap;
        gap: 0.75rem;
    }
    .brand-header-left {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        flex-shrink: 0;
        min-width: 240px;
    }
    .brand-header-right {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        flex-wrap: wrap;
    }
    .header-meta-pill {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        padding: 0.25rem 0.55rem;
        border-radius: 4px;
        font-size: 0.725rem;
        color: #475569;
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        white-space: nowrap;
    }
    .header-meta-pill .meta-label {
        color: #94A3B8;
        font-weight: 500;
    }
    .header-meta-pill .meta-val {
        color: #0F172A;
        font-weight: 600;
    }
    @media (max-width: 768px) {
        .brand-header-bar {
            flex-direction: column;
            align-items: flex-start;
            gap: 0.6rem;
            padding: 0.75rem 0.85rem;
        }
        .brand-header-right {
            width: 100%;
        }
    }
    
    /* Provenance Badges */
    .badge-status {
        padding: 0.25rem 0.65rem;
        border-radius: 4px;
        font-size: 0.725rem;
        font-weight: 600;
        font-family: monospace;
        letter-spacing: 0.03em;
        display: inline-flex;
        align-items: center;
        white-space: nowrap;
    }
    .badge-synthetic {
        background-color: #F0F9FF;
        color: #0369A1;
        border: 1px solid #BAE6FD;
    }
    .badge-uploaded {
        background-color: #F0FDFA;
        color: #0F766E;
        border: 1px solid #99F6E4;
    }
    .badge-sample {
        background-color: #F8FAFC;
        color: #475569;
        border: 1px solid #CBD5E1;
    }
    .badge-verified {
        background-color: #F0FDF4;
        color: #166534;
        border: 1px solid #BBF7D0;
    }
    .badge-warning {
        background-color: #FFFBEB;
        color: #B45309;
        border: 1px solid #FDE68A;
    }
    .badge-danger {
        background-color: #FEF2F2;
        color: #991B1B;
        border: 1px solid #FECACA;
    }
    .badge-info {
        background-color: #F0F9FF;
        color: #0369A1;
        border: 1px solid #BAE6FD;
    }
    
    /* Metric KPI Cards & Grid */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.85rem;
        margin-bottom: 1.15rem;
    }
    @media (max-width: 1200px) {
        .kpi-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }
    @media (max-width: 600px) {
        .kpi-grid {
            grid-template-columns: 1fr;
        }
    }
    
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 0.85rem 1rem;
        min-height: 106px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-sizing: border-box;
        box-shadow: 0 1px 3px 0 rgba(15, 23, 42, 0.04);
        transition: border-color 0.15s ease;
    }
    .kpi-card:hover {
        border-color: #CBD5E1;
    }
    .kpi-label {
        font-size: 0.725rem;
        text-transform: uppercase;
        color: #64748B;
        letter-spacing: 0.04em;
        font-weight: 600;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.35rem;
        line-height: 1.2;
    }
    .kpi-val {
        font-size: 1.55rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.2;
        margin: 0.25rem 0;
    }
    .kpi-unit {
        font-size: 0.8rem;
        font-weight: 500;
        color: #64748B;
    }
    .kpi-sub {
        font-size: 0.75rem;
        color: #64748B;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.25rem;
        line-height: 1.3;
    }
    
    /* Standard Operational Risk Color Tokens */
    .risk-low { color: #0D9488; }
    .risk-moderate { color: #D97706; }
    .risk-high { color: #EA580C; }
    .risk-severe { color: #DC2626; }
    
    .pill-teal {
        background: #F0FDFA;
        color: #0F766E;
        border: 1px solid #99F6E4;
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 0.675rem;
        font-weight: 600;
        white-space: nowrap;
        display: inline-flex;
        align-items: center;
    }
    .pill-amber {
        background: #FFFBEB;
        color: #B45309;
        border: 1px solid #FDE68A;
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 0.675rem;
        font-weight: 600;
        white-space: nowrap;
        display: inline-flex;
        align-items: center;
    }
    .pill-orange {
        background: #FFF7ED;
        color: #C2410C;
        border: 1px solid #FED7AA;
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 0.675rem;
        font-weight: 600;
        white-space: nowrap;
        display: inline-flex;
        align-items: center;
    }
    .pill-red {
        background: #FEF2F2;
        color: #B91C1C;
        border: 1px solid #FECACA;
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 0.675rem;
        font-weight: 600;
        white-space: nowrap;
        display: inline-flex;
        align-items: center;
    }
    
    /* Information Panels */
    .dashboard-panel {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 1rem 1.15rem;
        box-shadow: 0 1px 3px 0 rgba(15, 23, 42, 0.04);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-sizing: border-box;
    }
    .panel-title {
        color: #0F172A;
        font-size: 0.95rem;
        font-weight: 600;
        margin: 0;
    }
    .panel-subtitle {
        color: #64748B;
        font-size: 0.75rem;
        margin: 0.2rem 0 0.8rem 0;
    }
    
    /* Priority Mitigation Action Cards & Grid */
    .priority-recs-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
        gap: 0.75rem;
        margin-bottom: 0.75rem;
    }
    .priority-rec-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 5px;
        padding: 0.85rem 1rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-sizing: border-box;
        height: 100%;
    }
    
    /* Zone Status Cards & Grid */
    .zone-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.85rem;
        margin-top: 0.5rem;
    }
    @media (max-width: 1200px) {
        .zone-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }
    @media (max-width: 600px) {
        .zone-grid {
            grid-template-columns: 1fr;
        }
    }
    
    .zone-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 0.95rem 1.05rem;
        box-shadow: 0 1px 3px 0 rgba(15, 23, 42, 0.04);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-sizing: border-box;
        transition: border-color 0.15s ease;
    }
    .zone-card:hover {
        border-color: #CBD5E1;
    }
    
    /* Technical Notice Banners */
    .banner-simulation {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #0284C7;
        padding: 0.75rem 1.1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
        color: #334155;
        font-size: 0.825rem;
        line-height: 1.5;
    }
    .banner-warning {
        background: #FFFBEB;
        border: 1px solid #FDE68A;
        border-left: 4px solid #D97706;
        padding: 0.75rem 1.1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
        color: #92400E;
        font-size: 0.825rem;
        line-height: 1.5;
    }
    
    /* Sidebar Navigation Item SaaS Styling */
    [data-testid="stSidebar"] [data-testid="stRadio"] > div {
        gap: 3px;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        padding: 0.45rem 0.7rem;
        border-radius: 5px;
        margin: 0;
        cursor: pointer;
        font-size: 0.85rem;
        color: #475569;
        font-weight: 500;
        border-left: 3px solid transparent;
        transition: all 0.15s ease;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background-color: #F1F5F9;
        color: #0F172A;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
        background-color: #F0F9FF;
        color: #0284C7;
        font-weight: 600;
        border-left: 3px solid #0284C7;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] input[type="radio"],
    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child {
        display: none;
    }
    
    /* Responsive Streamlit Columns Behavior */
    @media (max-width: 1280px) {
        /* Overview 3-panel row: 2 columns on top (Panel A & Panel B), Panel C full-width below */
        [data-testid="stHorizontalBlock"]:has(.priority-recs-grid) {
            flex-wrap: wrap !important;
            gap: 1rem !important;
        }
        [data-testid="stHorizontalBlock"]:has(.priority-recs-grid) > [data-testid="stColumn"]:nth-child(1),
        [data-testid="stHorizontalBlock"]:has(.priority-recs-grid) > [data-testid="column"]:nth-child(1),
        [data-testid="stHorizontalBlock"]:has(.priority-recs-grid) > .stColumn:nth-child(1) {
            flex: 1 1 calc(54% - 0.5rem) !important;
            width: calc(54% - 0.5rem) !important;
            min-width: 320px !important;
        }
        [data-testid="stHorizontalBlock"]:has(.priority-recs-grid) > [data-testid="stColumn"]:nth-child(2),
        [data-testid="stHorizontalBlock"]:has(.priority-recs-grid) > [data-testid="column"]:nth-child(2),
        [data-testid="stHorizontalBlock"]:has(.priority-recs-grid) > .stColumn:nth-child(2) {
            flex: 1 1 calc(46% - 0.5rem) !important;
            width: calc(46% - 0.5rem) !important;
            min-width: 280px !important;
        }
        [data-testid="stHorizontalBlock"]:has(.priority-recs-grid) > [data-testid="stColumn"]:nth-child(3),
        [data-testid="stHorizontalBlock"]:has(.priority-recs-grid) > [data-testid="column"]:nth-child(3),
        [data-testid="stHorizontalBlock"]:has(.priority-recs-grid) > .stColumn:nth-child(3) {
            flex: 1 1 100% !important;
            width: 100% !important;
            min-width: 100% !important;
            margin-top: 0.25rem;
        }
        
        /* General 3-column blocks (e.g. Playbook cards, Sandbox scenarios) wrap evenly */
        [data-testid="stHorizontalBlock"]:not(:has(.priority-recs-grid)):has(> [data-testid="stColumn"]:nth-child(3):last-child),
        [data-testid="stHorizontalBlock"]:not(:has(.priority-recs-grid)):has(> [data-testid="column"]:nth-child(3):last-child),
        [data-testid="stHorizontalBlock"]:not(:has(.priority-recs-grid)):has(> .stColumn:nth-child(3):last-child) {
            flex-wrap: wrap !important;
            gap: 0.85rem !important;
        }
        [data-testid="stHorizontalBlock"]:not(:has(.priority-recs-grid)):has(> [data-testid="stColumn"]:nth-child(3):last-child) > [data-testid="stColumn"],
        [data-testid="stHorizontalBlock"]:not(:has(.priority-recs-grid)):has(> [data-testid="column"]:nth-child(3):last-child) > [data-testid="column"],
        [data-testid="stHorizontalBlock"]:not(:has(.priority-recs-grid)):has(> .stColumn:nth-child(3):last-child) > .stColumn {
            flex: 1 1 calc(50% - 0.5rem) !important;
            width: calc(50% - 0.5rem) !important;
            min-width: 260px !important;
        }
        
        /* 4-column blocks (e.g. Sliders, Metrics) wrap to 2x2 */
        [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"]:nth-child(4):last-child),
        [data-testid="stHorizontalBlock"]:has(> [data-testid="column"]:nth-child(4):last-child),
        [data-testid="stHorizontalBlock"]:has(> .stColumn:nth-child(4):last-child) {
            flex-wrap: wrap !important;
            gap: 0.75rem !important;
        }
        [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"]:nth-child(4):last-child) > [data-testid="stColumn"],
        [data-testid="stHorizontalBlock"]:has(> [data-testid="column"]:nth-child(4):last-child) > [data-testid="column"],
        [data-testid="stHorizontalBlock"]:has(> .stColumn:nth-child(4):last-child) > .stColumn {
            flex: 1 1 calc(50% - 0.5rem) !important;
            width: calc(50% - 0.5rem) !important;
            min-width: 220px !important;
        }
    }
    
    @media (max-width: 860px) {
        /* Switch multi-column layouts to vertical single-column stack on tablets/mobile */
        [data-testid="stHorizontalBlock"] {
            flex-direction: column !important;
            flex-wrap: wrap !important;
            gap: 1rem !important;
        }
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"],
        [data-testid="stHorizontalBlock"] > [data-testid="column"],
        [data-testid="stHorizontalBlock"] > .stColumn,
        .stColumn {
            width: 100% !important;
            min-width: 100% !important;
            flex: 1 1 100% !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Helper Functions for Light Theme Plotly Graphics
# ---------------------------------------------------------
def apply_light_theme(fig, height=360, title=""):
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=30 if title else 20, b=20),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#F8FAFC",
        font=dict(family="Inter, sans-serif", color="#334155", size=11),
        xaxis=dict(showgrid=True, gridcolor="#E2E8F0", linecolor="#CBD5E1", tickfont=dict(color="#64748B")),
        yaxis=dict(showgrid=True, gridcolor="#E2E8F0", linecolor="#CBD5E1", tickfont=dict(color="#64748B")),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10, color="#475569")
        )
    )
    return fig

# Distinct, high-contrast zone palette for white background
ZONE_COLORS = {
    "Excavation": "#D97706",       # Ochre / Amber
    "Haul Road": "#DC2626",        # Crimson Red
    "Material Storage": "#0284C7", # Slate Sky Blue
    "Loading Point": "#0D9488"     # Teal Green
}

# ---------------------------------------------------------
# Mitigation Recommendation Matrix (Industrial Playbooks)
# ---------------------------------------------------------
MITIGATION_PLAYBOOK = {
    "Excavation": [
        {
            "action": "Excavator Boom/Bucket Tip Wetting",
            "type": "Primary Source Control",
            "efficiency": "60% - 75% PM10 reduction",
            "eff_val": 0.68,
            "priority": "Immediate",
            "time_to_deploy": "5 mins",
            "cost": "Low ($)",
            "detail": "Activate pressurized fine-mist spray nozzles mounted directly on excavator stick to suppress dust at the active cut boundary."
        },
        {
            "action": "Mobile Geotextile Wind Barriers (3m)",
            "type": "Perimeter Control",
            "efficiency": "30% - 40% PM10 reduction",
            "eff_val": 0.35,
            "priority": "Preventative",
            "time_to_deploy": "30 mins",
            "cost": "Medium ($$)",
            "detail": "Erect portable semi-permeable windbreak screens on the upwind boundary to decelerate incoming surface wind gusts."
        },
        {
            "action": "Soil Stripping Phasing & Restricting Cut Area",
            "type": "Operational Practice",
            "efficiency": "25% - 35% PM10 reduction",
            "eff_val": 0.30,
            "priority": "Scheduled",
            "time_to_deploy": "15 mins",
            "cost": "Low ($)",
            "detail": "Pause secondary cut zones; only disturb soil areas that will be backfilled or compacted within the immediate shift."
        }
    ],
    "Haul Road": [
        {
            "action": "Automated Water Bowser Misting Cycle",
            "type": "Surface Moisture Retention",
            "efficiency": "65% - 75% PM10 reduction",
            "eff_val": 0.70,
            "priority": "Immediate",
            "time_to_deploy": "10 mins",
            "cost": "Medium ($$)",
            "detail": "Dispatch 10,000L water truck equipped with low-pressure atomizing rear spray bar across unpaved transit lanes at 20-min cycles."
        },
        {
            "action": "Vehicle Speed Restriction Enforcement (15 km/h)",
            "type": "Operational Control",
            "efficiency": "40% - 50% PM10 reduction",
            "eff_val": 0.45,
            "priority": "Immediate",
            "time_to_deploy": "Instant",
            "cost": "Zero ($0)",
            "detail": "Enforce strict 15 km/h limit on unpaved roads via site telemetry; dust generation scales quadratically with vehicular speed."
        },
        {
            "action": "Environmentally Benign Polymer Crusting Agent",
            "type": "Chemical Stabilization",
            "efficiency": "80% - 90% PM10 reduction",
            "eff_val": 0.85,
            "priority": "Long-term",
            "time_to_deploy": "2 hours",
            "cost": "High ($$$)",
            "detail": "Apply biodegradable lignosulfonate or acrylic polymer emulsion to bind fine silts and establish a durable road crust."
        }
    ],
    "Material Storage": [
        {
            "action": "Automated High-Angle Atomized Mist Cannons",
            "type": "Airborne Suppression",
            "efficiency": "65% - 75% PM10 reduction",
            "eff_val": 0.70,
            "priority": "Immediate",
            "time_to_deploy": "5 mins",
            "cost": "Medium ($$)",
            "detail": "Activate dual 30m oscillation mist cannons positioned upwind of aggregate stockpiles with 30-50 micron droplet size."
        },
        {
            "action": "Continuous Geotextile Tarpaulin Stockpile Cover",
            "type": "Physical Containment",
            "efficiency": "85% - 95% PM10 reduction",
            "eff_val": 0.90,
            "priority": "Scheduled",
            "time_to_deploy": "45 mins",
            "cost": "Low ($)",
            "detail": "Anchor heavy-duty weighted covers over dormant sand and fine aggregate stockpiles to prevent aerodynamic wind pick-up."
        },
        {
            "action": "Stockpile Height Management & Crest Levelling",
            "type": "Operational Practice",
            "efficiency": "25% - 35% PM10 reduction",
            "eff_val": 0.30,
            "priority": "Preventative",
            "time_to_deploy": "20 mins",
            "cost": "Low ($)",
            "detail": "Maintain pile heights below 3m to avoid vortex turbulence at the crest under elevated wind velocities."
        }
    ],
    "Loading Point": [
        {
            "action": "Atomized Dry Fog Enclosure at Transfer Chute",
            "type": "Engineered Containment",
            "efficiency": "75% - 85% PM10 reduction",
            "eff_val": 0.80,
            "priority": "Immediate",
            "time_to_deploy": "Instant",
            "cost": "Medium ($$)",
            "detail": "Trigger ultrasonic dry fog nozzle array inside the hopper throat to agglomerate sub-10 micron dust without wetting bulk material."
        },
        {
            "action": "Minimize Free-Fall Drop Height (< 1.2m)",
            "type": "Operational Control",
            "efficiency": "45% - 55% PM10 reduction",
            "eff_val": 0.50,
            "priority": "Immediate",
            "time_to_deploy": "Instant",
            "cost": "Zero ($0)",
            "detail": "Lower loader bucket to within 1 meter of truck bed before releasing aggregate to prevent turbulent displacement air drafts."
        },
        {
            "action": "Three-Sided Flexible Heavy Curtain Enclosure",
            "type": "Structural Barrier",
            "efficiency": "60% - 70% PM10 reduction",
            "eff_val": 0.65,
            "priority": "Scheduled",
            "time_to_deploy": "1 hour",
            "cost": "Medium ($$)",
            "detail": "Deploy heavy PVC hanging strip curtains around the hopper loading bay to isolate wind gusts from the dropping material."
        }
    ]
}

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "dispatch_log" not in st.session_state:
    st.session_state["dispatch_log"] = []

if "feedback_records" not in st.session_state:
    st.session_state["feedback_records"] = generate_sample_feedback_records()

if "data_mode" not in st.session_state:
    st.session_state["data_mode"] = "synthetic"

# Navigation definition
NAV_OPTIONS = [
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

NAV_ICONS = {
    "Overview": "▣  Overview",
    "Spatial Map & Plume": "◈  Spatial Map & Plume",
    "Source Attribution": "☵  Source Attribution",
    "Mitigation Playbook": "⛨  Mitigation Playbook",
    "Intervention Sandbox": "☷  Intervention Sandbox",
    "Intervention Feedback": "⟲  Intervention Feedback",
    "Sensor Analytics": "∿  Sensor Analytics",
    "Data Health & Pipeline": "✓  Data Health & Pipeline",
    "Reports & Export": "⤓  Reports & Export",
    "Privacy Policy": "§  Privacy Policy",
    "Terms of Use": "⚖  Terms of Use"
}

# Query parameter routing
if "view" in st.query_params:
    qp_view = str(st.query_params.get("view", "")).lower()
    if qp_view == "privacy":
        st.session_state["nav_view"] = "Privacy Policy"
    elif qp_view in ["terms", "tos"]:
        st.session_state["nav_view"] = "Terms of Use"
    elif qp_view in ["spatial", "map"]:
        st.session_state["nav_view"] = "Spatial Map & Plume"
    elif qp_view in ["attribution", "source"]:
        st.session_state["nav_view"] = "Source Attribution"
    elif qp_view in ["playbook", "mitigation"]:
        st.session_state["nav_view"] = "Mitigation Playbook"
    elif qp_view in ["sandbox", "simulation"]:
        st.session_state["nav_view"] = "Intervention Sandbox"
    elif qp_view in ["feedback", "intervention", "evaluation"]:
        st.session_state["nav_view"] = "Intervention Feedback"
    elif qp_view in ["analytics", "forecast"]:
        st.session_state["nav_view"] = "Sensor Analytics"
    elif qp_view in ["health", "pipeline"]:
        st.session_state["nav_view"] = "Data Health & Pipeline"
    elif qp_view in ["export", "reports"]:
        st.session_state["nav_view"] = "Reports & Export"
    elif qp_view in ["overview", "operations", "dashboard"]:
        st.session_state["nav_view"] = "Overview"

if "nav_view" not in st.session_state or st.session_state["nav_view"] not in NAV_OPTIONS:
    st.session_state["nav_view"] = "Overview"

# ---------------------------------------------------------
# Sidebar Navigation & Controls
# ---------------------------------------------------------
with st.sidebar:
    render_html("""
    <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:1.25rem;">
        <div class="brand-mark-box">DS</div>
        <div>
            <div style="font-size:1.15rem; font-weight:700; color:#0F172A; line-height:1.2;">DustSentinel</div>
            <div style="font-size:0.75rem; color:#64748B;">Environmental Intelligence</div>
        </div>
    </div>
    """)
    
    render_html("<div style='font-size:0.675rem; font-weight:600; text-transform:uppercase; color:#94A3B8; letter-spacing:0.06em; margin-bottom:0.4rem;'>Dashboard Navigation</div>")
    
    current_nav = st.session_state.get("nav_view", "Overview")
    current_idx = NAV_OPTIONS.index(current_nav) if current_nav in NAV_OPTIONS else 0
    
    nav_selection = st.radio(
        "Navigation Menu",
        NAV_OPTIONS,
        index=current_idx,
        format_func=lambda x: NAV_ICONS.get(x, x),
        label_visibility="collapsed"
    )
    if nav_selection != st.session_state["nav_view"]:
        st.session_state["nav_view"] = nav_selection
        st.rerun()
        
    st.markdown("---")
    
    with st.expander("Data Ingestion & Presets", expanded=False):
        render_html("<div style='font-size:0.75rem; font-weight:600; color:#475569; margin-bottom:0.3rem;'>Ingestion Mode:</div>")
        data_source_mode = st.radio(
            "Select Ingestion Mode:",
            ["Synthetic Sensor Stream", "Upload CSV Dataset"],
            index=0 if st.session_state.get("data_mode") == "synthetic" else 1,
            label_visibility="collapsed"
        )
        if data_source_mode == "Synthetic Sensor Stream":
            scenario_preset = st.selectbox(
                "Preset Scenario:",
                ["Active Peak Shift", "Dry Windy Alert Condition", "Standard Operations"],
                index=0
            )
            sim_hours = st.slider("Window (Hours):", min_value=6, max_value=48, value=24, step=6)
            if st.session_state.get("data_mode") != "synthetic" or "raw_df" not in st.session_state:
                st.session_state["raw_df"] = generate_synthetic_stream(scenario_preset, hours=sim_hours, seed=101)
                st.session_state["data_mode"] = "synthetic"
            if st.button("Regenerate Synthetic Data", use_container_width=True):
                st.session_state["raw_df"] = generate_synthetic_stream(scenario_preset, hours=sim_hours, seed=101)
                st.session_state["data_mode"] = "synthetic"
                st.success(f"Generated {len(st.session_state['raw_df'])} synthetic readings.")
        else:
            uploaded_file = st.file_uploader(
                "Upload Sensor CSV (Max 15 MB):",
                type=["csv"],
                help="Required columns: timestamp, zone, pm2_5, pm10, wind_speed, wind_deg, humidity, temperature"
            )
            if uploaded_file is not None:
                if uploaded_file.size > 15 * 1024 * 1024:
                    st.error("File size exceeds 15 MB safety limit.")
                else:
                    try:
                        raw_upload = pd.read_csv(uploaded_file)
                        st.session_state["raw_df"] = raw_upload
                        st.session_state["data_mode"] = "uploaded"
                    except Exception as e:
                        st.error(f"Error reading file: {e}")
            elif st.session_state.get("data_mode") not in ["uploaded", "sample_file"]:
                try:
                    st.session_state["raw_df"] = pd.read_csv("sample_sensor_data.csv")
                    st.session_state["data_mode"] = "sample_file"
                except Exception:
                    pass
            if st.button("Load Built-In sample_sensor_data.csv", use_container_width=True):
                try:
                    sample_df = pd.read_csv("sample_sensor_data.csv")
                    st.session_state["raw_df"] = sample_df
                    st.session_state["data_mode"] = "sample_file"
                    st.success("Loaded baseline sample_sensor_data.csv.")
                except Exception as e:
                    st.error(f"Failed to load sample_sensor_data.csv: {e}")

    with st.expander("Guided Demo Walkthrough", expanded=False):
        st.markdown("""
        **End-to-End Decision-Support Tour:**
        1. **Overview**: Executive KPIs, real-time plume schematic, priority recommendations, and zone cards.
        2. **Spatial Map**: Live plume dispersion vectors & receptor boundary.
        3. **Source Attribution**: Rule-based heuristic ranking & causality evidence.
        4. **Mitigation Playbook**: Inspect transparent rule firings & authorize work orders.
        5. **Intervention Sandbox**: Compare Baseline vs Targeted vs Comprehensive packages.
        6. **Sensor Analytics**: Chronological forecast baseline & persistence benchmark.
        7. **Data Health & Pipeline**: Ingestion schema diagnostics & anomaly flags.
        8. **Reports & Export**: Download verified logs, work orders & audit trails.
        """)
        if st.button("Pre-Load Demo State", use_container_width=True, help="Populate sample work orders & audit records for rapid evaluation."):
            st.session_state["raw_df"] = generate_synthetic_stream("Active Peak Shift", hours=24, seed=101)
            st.session_state["data_mode"] = "synthetic"
            if not st.session_state["dispatch_log"]:
                st.session_state["dispatch_log"].append({
                    "order_id": "ORD-DEMO-001",
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "target_zone": "Haul Road",
                    "assigned_crew": "Water Bowser Unit WB-04",
                    "action": "Automated Water Bowser Misting Cycle",
                    "authorizer": "M. Santos (Senior Site Engineer)",
                    "current_pm10": 128.4,
                    "status": "Authorized",
                    "linked_record_id": "INT-DEMO-VERIFIED-001"
                })
            st.session_state["feedback_records"] = generate_sample_feedback_records()
            st.success("Demo state initialized with active scenario and intervention audit records.")
            st.rerun()

    with st.expander("Air Quality Reference Limits", expanded=False):
        st.markdown("""
        • **Site Action Trigger (15-Min Peak)**: 100 µg/m³ PM10  
        • **US EPA NAAQS (24-Hour Average)**: 150 µg/m³ PM10 / 35 µg/m³ PM2.5  
        • **WHO Air Quality Guideline (24-Hour)**: 45 µg/m³ PM10 / 15 µg/m³ PM2.5  
        • **Aerodynamic Lift Threshold**: Wind speed ≥ 18 km/h with dry air (RH < 40%)
        """)

    render_html("""
    <div style="margin-top:1.5rem; padding-top:1rem; border-top:1px solid #E2E8F0; font-size:0.725rem; color:#94A3B8; line-height:1.5;">
        <div><b style="color:#475569;">DustSentinel v1.1</b></div>
        <div>Tech For Earth Hackathon 2026</div>
        <div>Advisory Decision Support System</div>
    </div>
    """)

# ---------------------------------------------------------
# Ensure Active DataFrame Exists & Clean
# ---------------------------------------------------------
if "raw_df" not in st.session_state or st.session_state["raw_df"] is None:
    st.session_state["raw_df"] = generate_synthetic_stream("Active Peak Shift", hours=24, seed=101)
    st.session_state["data_mode"] = "synthetic"

df_clean, val_report = validate_and_clean_data(st.session_state["raw_df"])

if not val_report["valid"]:
    st.error(f"Data Validation Error: {', '.join(val_report['issues'])}")
    st.stop()

# Slices & Key Environmental Metrics
latest_timestamp = df_clean["timestamp"].max()
latest_snapshot = df_clean[df_clean["timestamp"] == latest_timestamp].copy()
if latest_snapshot.empty:
    latest_snapshot = df_clean.tail(4)

risk_summary = evaluate_site_risk(latest_snapshot, df_clean)
attribution_df = compute_rule_based_attribution(latest_snapshot, df_clean)
actionable_recs = evaluate_actionable_recommendations(latest_snapshot, df_clean, attribution_df)

# Provenance Badge
if st.session_state.get("data_mode") == "synthetic":
    provenance_badge = "<span class='badge-status badge-synthetic'>[SYNTHETIC TEST DATA]</span>"
elif st.session_state.get("data_mode") == "uploaded":
    provenance_badge = "<span class='badge-status badge-uploaded'>[UPLOADED DATASET]</span>"
else:
    provenance_badge = "<span class='badge-status badge-sample'>[SAMPLE DATASET]</span>"

session_time_str = latest_timestamp.strftime('%Y-%m-%d %H:%M') if hasattr(latest_timestamp, 'strftime') else str(latest_timestamp)[:16]

# ---------------------------------------------------------
# VIEW 1: Overview (Main SaaS Analytics Dashboard)
# ---------------------------------------------------------
def render_overview():
    # Top Header & Metadata Bar
    render_html(f"""
    <div class="brand-header-bar">
        <div class="brand-header-left">
            <div class="brand-mark-box">DS</div>
            <div>
                <div class="brand-title">DustSentinel</div>
                <div class="brand-subtitle">Construction dust monitoring & mitigation decision support</div>
            </div>
        </div>
        <div class="brand-header-right">
            <div class="header-meta-pill">
                <span class="meta-label">Site:</span>
                <span class="meta-val">Sector 4 North Perimeter</span>
            </div>
            <div class="header-meta-pill">
                <span class="meta-label">Session:</span>
                <span class="meta-val">{session_time_str}</span>
            </div>
            <div class="header-meta-pill">
                <span class="meta-label">Role:</span>
                <span class="meta-val">Environmental Officer</span>
            </div>
            {provenance_badge}
        </div>
    </div>
    """)
    
    # 4 Responsive KPI Cards
    peak_pm10 = risk_summary['peak_pm10']
    peak_zone = risk_summary['peak_zone']
    peak_pm25 = risk_summary['peak_pm25']
    avg_pm25 = float(latest_snapshot["pm2_5"].mean()) if not latest_snapshot.empty else 0.0
    pm_ratio = (avg_pm25 / peak_pm10) if peak_pm10 > 0 else 0.0
    
    current_wind_spd = float(latest_snapshot["wind_speed"].iloc[0]) if not latest_snapshot.empty else 0.0
    current_wind_deg = float(latest_snapshot["wind_deg"].iloc[0]) if not latest_snapshot.empty else 0.0
    current_wind_cardinal = get_cardinal_direction(current_wind_deg)
    current_humidity = float(latest_snapshot["humidity"].iloc[0]) if not latest_snapshot.empty else 0.0
    plume_heading = compute_plume_heading(current_wind_deg)
    
    high_risk_zones = int((latest_snapshot["pm10"] >= THRESHOLDS["PM10_SITE_ACTION"]).sum()) if not latest_snapshot.empty else 0
    total_zones = len(latest_snapshot["zone"].unique()) if not latest_snapshot.empty else 4
    
    if peak_pm10 >= 150:
        kpi1_badge = '<span class="pill-red">Severe (≥150)</span>'
        kpi1_color = "risk-severe"
    elif peak_pm10 >= 100:
        kpi1_badge = '<span class="pill-orange">Action (≥100)</span>'
        kpi1_color = "risk-high"
    elif peak_pm10 >= 45:
        kpi1_badge = '<span class="pill-amber">Moderate (≥45)</span>'
        kpi1_color = "risk-moderate"
    else:
        kpi1_badge = '<span class="pill-teal">Normal (<45)</span>'
        kpi1_color = "risk-low"
        
    if avg_pm25 >= 35.0:
        kpi2_badge = '<span class="pill-red">Above EPA 24h</span>'
    elif avg_pm25 >= 15.0:
        kpi2_badge = '<span class="pill-amber">Above WHO 24h</span>'
    else:
        kpi2_badge = '<span class="pill-teal">Baseline Fine</span>'
        
    is_wind_lift = current_wind_spd >= 18.0 and current_humidity < 40.0
    if is_wind_lift:
        kpi3_badge = '<span class="pill-amber">Wind Lift Risk</span>'
    elif current_wind_spd >= 18.0:
        kpi3_badge = '<span class="pill-orange">Elevated Wind</span>'
    else:
        kpi3_badge = '<span class="pill-teal">Calm / Low Drift</span>'
        
    if high_risk_zones >= 2:
        kpi4_badge = '<span class="pill-red">Multi-Zone Action</span>'
    elif high_risk_zones == 1:
        kpi4_badge = '<span class="pill-orange">Targeted Action</span>'
    else:
        kpi4_badge = '<span class="pill-teal">All Nominal</span>'
        
    render_html(f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">
                <span>Peak PM10</span>
                {kpi1_badge}
            </div>
            <div class="kpi-val {kpi1_color}">{peak_pm10:.1f} <span class="kpi-unit">µg/m³</span></div>
            <div class="kpi-sub">
                <span>15-Min Instantaneous</span>
                <span style="font-weight:600; color:#334155;">{html.escape(peak_zone)}</span>
            </div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">
                <span>Average PM2.5</span>
                {kpi2_badge}
            </div>
            <div class="kpi-val">{avg_pm25:.1f} <span class="kpi-unit">µg/m³</span></div>
            <div class="kpi-sub">
                <span>Site-Wide Mean</span>
                <span style="font-weight:600; color:#334155;">Ratio: {pm_ratio:.2f} (Crustal)</span>
            </div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">
                <span>Wind Velocity</span>
                {kpi3_badge}
            </div>
            <div class="kpi-val">{current_wind_spd:.1f} <span class="kpi-unit">km/h</span></div>
            <div class="kpi-sub">
                <span>{current_wind_cardinal} ({current_wind_deg:.0f}°)</span>
                <span style="font-weight:600; color:#334155;">Plume: {plume_heading:.0f}°</span>
            </div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">
                <span>High-Risk Zones</span>
                {kpi4_badge}
            </div>
            <div class="kpi-val">{high_risk_zones} <span class="kpi-unit">/ {total_zones} active</span></div>
            <div class="kpi-sub">
                <span>Trigger: ≥ 100 µg/m³</span>
                <span style="font-weight:600; color:#334155;">{total_zones - high_risk_zones} Normal</span>
            </div>
        </div>
    </div>
    """)
    
    # 3-Column Content Layout (Panels A, B, C)
    col_panel_a, col_panel_b, col_panel_c = st.columns([1.35, 1.15, 1.05])
    
    # PANEL A: PM10 Concentration
    with col_panel_a:
        render_html("""
        <div class="panel-title">PM10 Particulate Concentration</div>
        <div class="panel-subtitle">15-minute multi-zone telemetry vs regulatory & site action thresholds</div>
        """)
        
        fig_ts = go.Figure()
        for z in SITE_ZONES:
            zdf = df_clean[df_clean["zone"] == z]
            fig_ts.add_trace(go.Scatter(
                x=zdf["timestamp"],
                y=zdf["pm10"],
                mode="lines",
                line=dict(color=ZONE_COLORS.get(z, "#64748B"), width=2),
                name=z
            ))
            
        fig_ts.add_hline(
            y=THRESHOLDS["PM10_SITE_ACTION"],
            line_dash="dash",
            line_color="#EA580C",
            annotation_text="Site Action Trigger (15-Min: 100 µg/m³)",
            annotation_position="top left",
            annotation_font=dict(color="#EA580C", size=9)
        )
        fig_ts.add_hline(
            y=THRESHOLDS["PM10_EPA_24H_LIMIT"],
            line_dash="dot",
            line_color="#DC2626",
            annotation_text="EPA Standard (24h: 150 µg/m³)",
            annotation_position="top right",
            annotation_font=dict(color="#DC2626", size=9)
        )
        apply_light_theme(fig_ts, height=350)
        fig_ts.update_layout(margin=dict(l=35, r=15, t=25, b=25))
        st.plotly_chart(fig_ts, use_container_width=True)
        
    # PANEL B: Wind Direction & Plume (Schematic)
    with col_panel_b:
        render_html("""
        <div class="panel-title">Wind Direction & Plume Vector</div>
        <div class="panel-subtitle">[SCHEMATIC DISPERSION MODEL] 2D perimeter layout & sensitive boundary</div>
        """)
        
        fig_map = go.Figure()
        
        # Perimeter fence boundary
        fig_map.add_trace(go.Scatter(
            x=[0, 100, 100, 0, 0],
            y=[0, 0, 100, 100, 0],
            mode="lines",
            line=dict(color="#94A3B8", width=1.5, dash="dash"),
            name="Site Perimeter Fence",
            hoverinfo="name"
        ))
        
        # Sensitive Receptor Zone (Residential boundary at SE quadrant)
        fig_map.add_trace(go.Scatter(
            x=[85, 100, 100, 85, 85],
            y=[0, 0, 25, 25, 0],
            fill="toself",
            fillcolor="rgba(220, 38, 38, 0.12)",
            line=dict(color="#DC2626", width=1.5),
            name="Sensitive Receptor (Residential)",
            hoverinfo="name"
        ))
        
        # Active sensor nodes
        for z in SITE_ZONES:
            coords = ZONE_COORDINATES[z]
            z_row = latest_snapshot[latest_snapshot["zone"] == z]
            pm_val = float(z_row["pm10"].iloc[0]) if not z_row.empty else 0.0
            color = "#DC2626" if pm_val >= 150 else ("#EA580C" if pm_val >= 100 else ("#D97706" if pm_val >= 45 else "#0D9488"))
            fig_map.add_trace(go.Scatter(
                x=[coords["x"]],
                y=[coords["y"]],
                mode="markers+text",
                marker=dict(size=14, color=color, line=dict(color="#FFFFFF", width=2)),
                text=[z],
                textposition="top center",
                name=z,
                hovertext=f"{z}: {pm_val:.1f} µg/m³"
            ))
            
        # Plume vector from peak source
        peak_coords = ZONE_COORDINATES.get(peak_zone, {"x": 50, "y": 50})
        rad = math.radians(plume_heading)
        length = min(35.0, 12.0 + current_wind_spd * 0.9)
        target_x = peak_coords["x"] + length * math.sin(rad)
        target_y = peak_coords["y"] + length * math.cos(rad)
        fig_map.add_trace(go.Scatter(
            x=[peak_coords["x"], target_x],
            y=[peak_coords["y"], target_y],
            mode="lines+markers",
            line=dict(color="#EA580C", width=3, dash="dot"),
            marker=dict(size=[4, 10], symbol=["circle", "arrow-up"], angleref="previous"),
            name="Plume Vector"
        ))
        
        fig_map.update_layout(
            height=350,
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#F8FAFC",
            xaxis=dict(range=[-5, 105], showgrid=True, gridcolor="#E2E8F0", zeroline=False, showticklabels=False),
            yaxis=dict(range=[-5, 105], showgrid=True, gridcolor="#E2E8F0", zeroline=False, showticklabels=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=9))
        )
        st.plotly_chart(fig_map, use_container_width=True)
        
        # Calculate angle offset to sensitive receptor
        dx = 95.0 - peak_coords["x"]
        dy = 10.0 - peak_coords["y"]
        bearing_rec = math.degrees(math.atan2(dx, dy)) % 360.0
        angle_diff = compute_angular_offset(plume_heading, bearing_rec)
        is_threat = angle_diff <= 40.0
        threat_pill = '<span class="pill-red">Receptor Downwind Threat</span>' if is_threat else '<span class="pill-teal">Off-Axis Drift</span>'
        
        render_html(f"""
        <div style="background:#F8FAFC; border:1px solid #E2E8F0; padding:0.45rem 0.65rem; border-radius:4px; font-size:0.75rem; color:#475569; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.3rem;">
            <span>Plume: <b>{plume_heading:.0f}°</b> • Offset: <b>{angle_diff:.0f}°</b></span>
            {threat_pill}
        </div>
        """)
        
    # PANEL C: Priority Actions
    with col_panel_c:
        render_html("""
        <div class="panel-title">Priority Mitigation Actions</div>
        <div class="panel-subtitle">Transparent rules mapped to top emission sources</div>
        """)
        
        priority_recs = actionable_recs[:2] if actionable_recs else []
        
        if priority_recs:
            cards_html = []
            for act in priority_recs:
                p_pill = '<span class="pill-red">Immediate Action</span>' if "Immediate" in act["priority"] else '<span class="pill-amber">Preventative</span>'
                reasons_html = "".join([f"<li style='margin-bottom:2px;'>{html.escape(r)}</li>" for r in act.get("selection_reasons", [])[:2]])
                
                cards_html.append(f"""
                <div class="priority-rec-card">
                    <div>
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.35rem; flex-wrap:wrap; gap:0.25rem;">
                            <span style="font-size:0.8rem; font-weight:700; color:#0F172A;">{html.escape(act['zone'])}</span>
                            {p_pill}
                        </div>
                        <div style="font-size:0.825rem; font-weight:600; color:#0284C7; margin-bottom:0.35rem; line-height:1.3;">
                            {html.escape(act['action_title'])}
                        </div>
                        <div style="background:#F8FAFC; border-left:3px solid #0284C7; padding:0.4rem 0.6rem; border-radius:2px; font-size:0.725rem; color:#334155; margin-bottom:0.45rem;">
                            <ul style="margin:0; padding-left:1rem; line-height:1.35;">{reasons_html}</ul>
                        </div>
                    </div>
                    <div style="font-size:0.725rem; color:#64748B; display:flex; justify-content:space-between; flex-wrap:wrap; gap:0.25rem;">
                        <span>Efficacy: <b style="color:#0D9488;">{html.escape(str(act['expected_efficacy_pct']))}</b></span>
                        <span>Lead: <b style="color:#1E293B;">{html.escape(str(act['deployment_latency']))}</b></span>
                    </div>
                </div>
                """)
            render_html(f"""
            <div class="priority-recs-grid">
                {"".join(cards_html)}
            </div>
            """)
        else:
            st.info("All construction zones are operating within nominal baseline parameters.")
            
        if st.button("Open Full Mitigation Playbook →", use_container_width=True):
            st.session_state["nav_view"] = "Mitigation Playbook"
            st.rerun()
            
    render_html("<div style='height:0.85rem;'></div>")
    
    # 4 Aligned Construction Zone Status Cards in Responsive Grid
    st.markdown("#### Construction Zone Status")
    st.caption("Real-time telemetry and 30-minute particulate velocity across active site operational sectors.")
    
    zone_cards_html = []
    for idx, z in enumerate(SITE_ZONES):
        z_snap = latest_snapshot[latest_snapshot["zone"] == z]
        z_pm10 = float(z_snap["pm10"].iloc[0]) if not z_snap.empty else 0.0
        z_act = str(z_snap["activity_level"].iloc[0]) if not z_snap.empty and "activity_level" in z_snap else "Moderate"
        z_coords = ZONE_COORDINATES.get(z, {"area": "Active Sector", "desc": "Standard operations"})
        
        # 30-min trend delta
        z_hist = df_clean[df_clean["zone"] == z].sort_values("timestamp")
        if len(z_hist) >= 3:
            delta_pm = z_pm10 - float(z_hist["pm10"].iloc[-3])
            if delta_pm > 15.0:
                trend_badge = f'<span class="pill-red">▲ +{delta_pm:.1f} µg/m³ (30m)</span>'
            elif delta_pm > 5.0:
                trend_badge = f'<span class="pill-orange">▲ +{delta_pm:.1f} µg/m³ (30m)</span>'
            elif delta_pm < -5.0:
                trend_badge = f'<span class="pill-teal">▼ {delta_pm:.1f} µg/m³ (30m)</span>'
            else:
                trend_badge = '<span class="pill-teal">▬ Stable (±5 µg/m³)</span>'
        else:
            trend_badge = '<span class="pill-teal">▬ Baseline</span>'
            
        if z_pm10 >= 150:
            z_risk_pill = '<span class="pill-red">Severe Alert</span>'
            z_pm_color = "risk-severe"
        elif z_pm10 >= 100:
            z_risk_pill = '<span class="pill-orange">Action Trigger</span>'
            z_pm_color = "risk-high"
        elif z_pm10 >= 45:
            z_risk_pill = '<span class="pill-amber">Moderate Advisory</span>'
            z_pm_color = "risk-moderate"
        else:
            z_risk_pill = '<span class="pill-teal">Normal Baseline</span>'
            z_pm_color = "risk-low"
            
        zone_cards_html.append(f"""
        <div class="zone-card">
            <div>
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem; flex-wrap:wrap; gap:0.25rem;">
                    <span style="font-size:0.85rem; font-weight:700; color:#0F172A;">{html.escape(z)}</span>
                    {z_risk_pill}
                </div>
                <div style="font-size:0.725rem; color:#64748B; margin-bottom:0.6rem;">{html.escape(z_coords['area'])}</div>
                
                <div style="display:flex; justify-content:space-between; align-items:baseline; margin-bottom:0.4rem; flex-wrap:wrap; gap:0.25rem;">
                    <div style="font-size:1.65rem; font-weight:700;" class="{z_pm_color}">{z_pm10:.1f} <span class="kpi-unit">µg/m³</span></div>
                    {trend_badge}
                </div>
                
                <div style="font-size:0.775rem; color:#475569; line-height:1.4; margin-bottom:0.6rem;">
                    {html.escape(z_coords['desc'])}
                </div>
            </div>
            
            <div style="border-top:1px solid #E2E8F0; padding-top:0.5rem; font-size:0.7rem; color:#94A3B8; display:flex; justify-content:space-between; flex-wrap:wrap; gap:0.25rem;">
                <span>Activity: <b style="color:#0284C7;">{html.escape(z_act)}</b></span>
                <span>Updated: {session_time_str[-5:]}</span>
            </div>
        </div>
        """)
        
    render_html(f"""
    <div class="zone-grid">
        {"".join(zone_cards_html)}
    </div>
    """)
    render_html("<div style='height:1rem;'></div>")
    render_html("""
    <div class="banner-simulation">
        <b>DECISION SUPPORT INTEGRITY NOTICE:</b> DustSentinel is an experimental operational prototype developed for construction site dust screening and early mitigation. 
        Readings represent indicative sensor telemetry and empirical engineering transfer equations (US EPA AP-42 & WRAP Handbook). All physical mitigation work orders require human site engineer authorization.
    </div>
    """)

# ---------------------------------------------------------
# VIEW 2: Spatial Map & Plume (Detailed Inspection)
# ---------------------------------------------------------
def render_spatial_map():
    render_html(f"""
    <div class="brand-header-bar">
        <div>
            <div class="brand-title">Spatial Map & Plume Vector Diagnostics</div>
            <div class="brand-subtitle">2D coordinate layout, perimeter boundaries, and downwind dispersion modeling</div>
        </div>
        <div>{provenance_badge}</div>
    </div>
    """)
    
    col_map, col_details = st.columns([1.6, 1.0])
    
    with col_map:
        fig_map = go.Figure()
        
        # Perimeter fence boundary
        fig_map.add_trace(go.Scatter(
            x=[0, 100, 100, 0, 0],
            y=[0, 0, 100, 100, 0],
            mode="lines",
            line=dict(color="#94A3B8", width=2, dash="dash"),
            name="Site Perimeter Fence",
            hoverinfo="name"
        ))
        
        # Sensitive Receptor Zone
        fig_map.add_trace(go.Scatter(
            x=[85, 100, 100, 85, 85],
            y=[0, 0, 25, 25, 0],
            fill="toself",
            fillcolor="rgba(220, 38, 38, 0.12)",
            line=dict(color="#DC2626", width=1.5),
            name="Sensitive Receptor (Residential Boundary)",
            hoverinfo="name"
        ))
        
        current_wind_deg = float(latest_snapshot["wind_deg"].iloc[0]) if not latest_snapshot.empty else 0.0
        current_wind_spd = float(latest_snapshot["wind_speed"].iloc[0]) if not latest_snapshot.empty else 0.0
        plume_heading = compute_plume_heading(current_wind_deg)
        peak_zone = risk_summary["peak_zone"]
        
        # Active sensor nodes
        for z in SITE_ZONES:
            coords = ZONE_COORDINATES[z]
            z_row = latest_snapshot[latest_snapshot["zone"] == z]
            pm_val = float(z_row["pm10"].iloc[0]) if not z_row.empty else 0.0
            color = "#DC2626" if pm_val >= 150 else ("#EA580C" if pm_val >= 100 else ("#D97706" if pm_val >= 45 else "#0D9488"))
            fig_map.add_trace(go.Scatter(
                x=[coords["x"]],
                y=[coords["y"]],
                mode="markers+text",
                marker=dict(size=16, color=color, line=dict(color="#FFFFFF", width=2)),
                text=[f"<b>{z}</b><br>{pm_val:.1f} µg/m³"],
                textposition="top center",
                name=z,
                hovertext=f"{z}: {pm_val:.1f} µg/m³"
            ))
            
        # Plume Vector Cone
        peak_coords = ZONE_COORDINATES.get(peak_zone, {"x": 50, "y": 50})
        rad = math.radians(plume_heading)
        length = min(40.0, 15.0 + current_wind_spd * 0.9)
        target_x = peak_coords["x"] + length * math.sin(rad)
        target_y = peak_coords["y"] + length * math.cos(rad)
        
        fig_map.add_trace(go.Scatter(
            x=[peak_coords["x"], target_x],
            y=[peak_coords["y"], target_y],
            mode="lines+markers",
            line=dict(color="#EA580C", width=3, dash="dot"),
            marker=dict(size=[6, 12], symbol=["circle", "arrow-up"], angleref="previous"),
            name="Downwind Plume Vector"
        ))
        
        fig_map.update_layout(
            height=460,
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#F8FAFC",
            xaxis=dict(range=[-5, 105], showgrid=True, gridcolor="#E2E8F0", zeroline=False, title="Perimeter X (meters)"),
            yaxis=dict(range=[-5, 105], showgrid=True, gridcolor="#E2E8F0", zeroline=False, title="Perimeter Y (meters)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_map, use_container_width=True)
        
    with col_details:
        st.markdown("#### Sensor Node Telemetry & Spatial Layout")
        for z in SITE_ZONES:
            coords = ZONE_COORDINATES[z]
            z_row = latest_snapshot[latest_snapshot["zone"] == z]
            pm10_val = float(z_row["pm10"].iloc[0]) if not z_row.empty else 0.0
            pm25_val = float(z_row["pm2_5"].iloc[0]) if not z_row.empty else 0.0
            z_act = str(z_row["activity_level"].iloc[0]) if not z_row.empty and "activity_level" in z_row else "Moderate"
            
            render_html(f"""
            <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:5px; padding:0.75rem 1rem; margin-bottom:0.6rem; box-shadow:0 1px 2px rgba(0,0,0,0.03);">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.25rem;">
                    <b style="color:#0F172A; font-size:0.875rem;">{html.escape(z)}</b>
                    <span style="font-size:0.725rem; color:#64748B;">({coords['x']:.0f}m, {coords['y']:.0f}m)</span>
                </div>
                <div style="display:flex; justify-content:space-between; font-size:0.8rem; margin-top:0.3rem; flex-wrap:wrap; gap:0.35rem;">
                    <span>PM10: <b>{pm10_val:.1f} µg/m³</b></span>
                    <span>PM2.5: <b>{pm25_val:.1f} µg/m³</b></span>
                    <span>Activity: <b style="color:#0284C7;">{html.escape(z_act)}</b></span>
                </div>
                <div style="font-size:0.725rem; color:#64748B; margin-top:0.25rem;">
                    {html.escape(coords['area'])} • {html.escape(coords['desc'])}
                </div>
            </div>
            """)
            
        dx = 95.0 - peak_coords["x"]
        dy = 10.0 - peak_coords["y"]
        bearing_rec = math.degrees(math.atan2(dx, dy)) % 360.0
        angle_diff = compute_angular_offset(plume_heading, bearing_rec)
        
        render_html(f"""
        <div class="banner-simulation" style="margin-top:0.5rem;">
            <b>Plume Vector Geometry:</b><br>
            • Prevailing Wind: <b>{current_wind_spd:.1f} km/h</b> from {current_wind_deg:.0f}°<br>
            • Plume Dispersion Heading: <b>{plume_heading:.0f}°</b><br>
            • Bearing to Sensitive Receptor: <b>{bearing_rec:.0f}°</b><br>
            • Angular Offset: <b>{angle_diff:.0f}°</b> ({"< 40° — Active Threat Corridor" if angle_diff <= 40 else "Off-Axis / Safe Drift"})
        </div>
        """)

# ---------------------------------------------------------
# VIEW 3: Source Attribution
# ---------------------------------------------------------
def render_source_attribution():
    render_html(f"""
    <div class="brand-header-bar">
        <div>
            <div class="brand-title">Explainable Rule-Based Source Attribution</div>
            <div class="brand-subtitle">Multi-criteria deterministic ranking engine combining particulate ratios, wind dispersion, and site activity</div>
        </div>
        <div>{provenance_badge}</div>
    </div>
    """)
    
    render_html("""
    <div class="banner-simulation">
        <b>SCIENTIFIC INTEGRITY NOTICE:</b> Source attribution uses a deterministic <b>Multi-Criteria Environmental Rule Engine</b> 
        combining particulate size fractioning (PM2.5 / PM10), aerodynamic wind vector dispersion, and activity correlations. 
        Scores represent a <b>Heuristic Ranking Index (0–100)</b> for screening, not a calibrated Bayesian causal probability.
    </div>
    """)
    
    col_rank, col_factors = st.columns([1.2, 1.0])
    
    with col_rank:
        st.markdown("#### Primary Source Attribution Ranking")
        for _, arow in attribution_df.iterrows():
            rnk = arow["rank"]
            zone = arow["zone"]
            score = arow["ranking_score"]
            pm10 = arow["pm10"]
            exps = arow["explanations"]
            
            with st.expander(f"Rank #{rnk}: {zone} — Attribution Score: {score:.1f}/100", expanded=(rnk <= 2)):
                st.markdown(f"**Current PM10:** `{pm10:.1f} µg/m³` | **Activity Level:** `{arow['activity']}`")
                st.markdown("**Evidence & Rule Firings:**")
                for e in exps:
                    st.markdown(f"- {e}")
                    
    with col_factors:
        st.markdown("#### Attribution Ranking Score Breakdown")
        fig_conf = px.bar(
            attribution_df,
            x="ranking_score",
            y="zone",
            orientation="h",
            color="ranking_score",
            color_continuous_scale="Reds",
            labels={"ranking_score": "Attribution Index (0-100)", "zone": "Zone"},
            text="ranking_score"
        )
        fig_conf.update_layout(
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#F8FAFC",
            xaxis=dict(range=[0, 105], showgrid=True, gridcolor="#E2E8F0"),
            yaxis=dict(autorange="reversed")
        )
        fig_conf.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        st.plotly_chart(fig_conf, use_container_width=True)
        
        render_html("""
        <div style="background:#FFFFFF; border:1px solid #E2E8F0; padding:0.85rem; border-radius:5px; font-size:0.775rem; color:#475569; line-height:1.5;">
            <b style="color:#0F172A;">Causality Weight Matrix:</b><br>
            • <b>Coarse Signature Check</b>: Ratio PM2.5/PM10 &lt; 0.35 confirms mechanical crustal dust.<br>
            • <b>Downwind Vector Error</b>: Penalty applied if prevailing wind carries plume away from boundaries.<br>
            • <b>Erosion Threshold</b>: Triggered when Wind &gt; 18 km/h and Humidity &lt; 40%.
        </div>
        """)

# ---------------------------------------------------------
# VIEW 4: Mitigation Playbook & Work Orders
# ---------------------------------------------------------
def render_mitigation_playbook():
    render_html(f"""
    <div class="brand-header-bar">
        <div>
            <div class="brand-title">Targeted Mitigation Playbook & Work Orders</div>
            <div class="brand-subtitle">Actionable engineering countermeasures mapped to top-ranked dust emission sources</div>
        </div>
        <div>{provenance_badge}</div>
    </div>
    """)
    
    primary_source = attribution_df.iloc[0]["zone"] if not attribution_df.empty else "Excavation"
    primary_pm10 = attribution_df.iloc[0]["pm10"] if not attribution_df.empty else 0.0

    render_html(f"""
    <div class="banner-warning">
        <b>Priority Dispatch Recommendation:</b> Focus suppression resources on <b>{html.escape(primary_source)}</b> 
        (Current Reading: <b>{primary_pm10:.1f} µg/m³</b>, Ranked #1 Emitter). 
        <i>Recommendations are advisory decision support and require human engineering authorization.</i>
    </div>
    """)
    
    col_filter, col_auth_info = st.columns([1.5, 1.0])
    with col_filter:
        zone_filter = st.selectbox(
            "Filter Recommendations by Construction Zone:",
            ["All Active Zones"] + SITE_ZONES,
            index=0
        )
    with col_auth_info:
        render_html("""
        <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:4px; padding:0.6rem 0.8rem; font-size:0.75rem; color:#475569; margin-top:0.4rem;">
            <b>Advisory Governance Notice:</b> All recommendations must be authorized by the designated site engineer before physical equipment deployment.
        </div>
        """)

    filtered_recs = actionable_recs if zone_filter == "All Active Zones" else [r for r in actionable_recs if r["zone"] == zone_filter]

    if not filtered_recs:
        st.info(f"No active priority recommendations for {zone_filter}. Particulate levels are currently within baseline parameters.")
    else:
        rec_cols = st.columns(min(len(filtered_recs), 3) if len(filtered_recs) > 0 else 1)
        for i, act in enumerate(filtered_recs):
            with rec_cols[i % len(rec_cols)]:
                p_pill = '<span class="pill-red">Immediate Action</span>' if "Immediate" in act["priority"] else '<span class="pill-amber">Preventative</span>'
                reasons_html = "".join([f"<li style='margin-bottom:0.25rem;'>{html.escape(r)}</li>" for r in act.get("selection_reasons", [])])
                
                render_html(f"""
                <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:6px; padding:1.1rem; height:100%; display:flex; flex-direction:column; justify-content:space-between; margin-bottom:1rem; box-shadow:0 1px 3px rgba(0,0,0,0.04);">
                    <div>
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem; flex-wrap:wrap; gap:0.25rem;">
                            <span style="font-size:0.85rem; font-weight:700; color:#0F172A;">{html.escape(act['zone'])}</span>
                            {p_pill}
                        </div>
                        <h4 style="color:#0284C7; margin:0.3rem 0; font-size:0.975rem;">{html.escape(act['action_title'])}</h4>
                        <p style="font-size:0.8rem; color:#475569; line-height:1.4; margin-bottom:0.6rem;">{html.escape(act['operational_detail'])}</p>
                        
                        <div style="background:#F8FAFC; border-left:3px solid #0284C7; padding:0.5rem 0.7rem; border-radius:2px; margin-bottom:0.6rem;">
                            <div style="font-size:0.7rem; font-weight:600; text-transform:uppercase; color:#64748B; letter-spacing:0.04em; margin-bottom:0.2rem;">Why Selected (Causality Rationale):</div>
                            <ul style="margin:0; padding-left:1.1rem; font-size:0.75rem; color:#334155; line-height:1.4;">
                                {reasons_html}
                            </ul>
                        </div>
                    </div>
                    <div style="border-top:1px solid #E2E8F0; padding-top:0.6rem; font-size:0.775rem; color:#64748B;">
                        <div>Estimated Efficacy: <b style="color:#0D9488;">{html.escape(str(act['expected_efficacy_pct']))}</b> <span style="font-size:0.675rem; color:#94A3B8;">(Handbook)</span></div>
                        <div>Deployment Lead Time: <b style="color:#0F172A;">{html.escape(str(act['deployment_latency']))}</b> • Cost: <b style="color:#0F172A;">{html.escape(str(act['cost_tier']))}</b></div>
                        <div style="margin-top:0.3rem;"><span style="color:#0284C7; font-size:0.7rem; font-weight:600;">[ADVISORY — REQUIRES HUMAN APPROVAL]</span></div>
                    </div>
                </div>
                """)
                
    st.markdown("---")
    # Action Dispatch Ticket Generator & Human Approval Workflow
    st.markdown("#### Operational Action Dispatch & Human Approval Work Order")
    
    ticket_col1, ticket_col2 = st.columns([1.6, 1.2])
    with ticket_col1:
        order_zone = st.selectbox("Designate Target Construction Zone:", SITE_ZONES, index=SITE_ZONES.index(primary_source) if primary_source in SITE_ZONES else 0)
        zone_matching_actions = [r["action_title"] for r in actionable_recs if r["zone"] == order_zone]
        if not zone_matching_actions:
            zone_matching_actions = [a["action"] for a in MITIGATION_PLAYBOOK.get(order_zone, [])]
        order_action = st.selectbox("Select Action Protocol:", zone_matching_actions, index=0)
        
        ticket_target = st.text_input("Assigned Site Crew / Equipment ID:", value="Water Bowser Unit WB-04 & Excavator Team Bravo")
        operator_authorizer = st.text_input("Authorizing Site Engineer / Operator:", value="M. Santos (Senior Site Engineer)")
        ticket_notes = st.text_area("Operational Site Instructions:", value=f"Deploy immediate suppression protocol to {order_zone}. Target sustained PM10 reduction below 50 µg/m³ before shift review.")
        human_approval_ack = st.checkbox("I authorize this mitigation dispatch based on current sensor telemetry and on-site operational review.", value=True)

    with ticket_col2:
        render_html("<div style='height: 1.8rem;'></div>")
        if st.button("Authorize & Issue Operational Dispatch Order", use_container_width=True, disabled=not human_approval_ack):
            current_zone_pm10 = float(latest_snapshot[latest_snapshot["zone"] == order_zone]["pm10"].iloc[0]) if not latest_snapshot[latest_snapshot["zone"] == order_zone].empty else primary_pm10
            
            order_id = f"ORD-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{int(datetime.now().microsecond / 1000):03d}"
            ticket_content = f"""======================================================
DUSTSENTINEL MITIGATION DISPATCH WORK ORDER
Tech For Earth 2026 - Dust Mitigation Plan
======================================================
Generated At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Work Order ID: {order_id}
Target Zone: {order_zone}
Current PM10: {current_zone_pm10:.1f} ug/m3
Status: AUTHORIZED (Human Site Approval Confirmed)

AUTHORIZING ENGINEER: {operator_authorizer}
ASSIGNED CREW / UNIT: {ticket_target}
SELECTED ACTION: {order_action}
EXPECTED NOMINAL EFFICACY: 60% - 75% PM10 reduction (Literature Reference)

OPERATIONAL DISPATCH INSTRUCTIONS:
{ticket_notes}

DECISION SUPPORT VERIFICATION TIMELINE:
- Initial suppression deployment: T+5 to T+10 minutes
- Sensor node telemetry recheck: T+15 minutes
- Operator physical outcome inspection: T+30 minutes
======================================================"""
            new_feedback = create_intervention_record(
                zone=order_zone,
                recommended_action=order_action,
                performed_action=order_action,
                provenance="Field Measurement" if st.session_state.get("data_mode") == "uploaded" else "Synthetic Demonstration",
                pm10_before=current_zone_pm10,
                pm10_after=None,
                verification_status="Pending Verification",
                is_authorized=True,
                authorizer_name=operator_authorizer,
                operator_notes=f"Dispatched by {operator_authorizer}. Pending physical field confirmation.",
                linked_order_id=order_id
            )
            dispatch_entry = {
                "order_id": order_id,
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "target_zone": order_zone,
                "assigned_crew": ticket_target,
                "action": order_action,
                "authorizer": operator_authorizer,
                "current_pm10": current_zone_pm10,
                "status": "Authorized",
                "linked_record_id": new_feedback["record_id"]
            }
            st.session_state["dispatch_log"].append(dispatch_entry)
            st.session_state["feedback_records"].insert(0, new_feedback)
            st.success(f"Work order {order_id} authorized and registered in operational audit log for {order_zone}.")
            
            st.download_button(
                label="Download Dispatch Ticket (.txt)",
                data=ticket_content,
                file_name=f"dust_dispatch_{order_zone.lower().replace(' ', '_')}.txt",
                mime="text/plain",
                use_container_width=True
            )

    if st.session_state["dispatch_log"]:
        st.markdown("##### Authorized Shift Dispatch Log")
        st.dataframe(pd.DataFrame(st.session_state["dispatch_log"]), use_container_width=True)

    st.markdown("---")
    # Closed-Loop Feedback & Audit Trail
    st.markdown("#### Closed-Loop Intervention Feedback & Operational Audit Trail")
    render_html("""
    <div class="banner-simulation">
        <b>OPERATIONAL AUDIT TRAIL NOTICE:</b> Operator field evaluations are recorded for shift quality assurance, internal audit trails, and manual engineering review. 
        <b>DustSentinel does not claim continuous automated AI retraining without an offline validated training-and-evaluation loop.</b>
    </div>
    """)
    
    fb_col1, fb_col2 = st.columns([1.5, 1.0])
    with fb_col1:
        if st.session_state["feedback_records"]:
            record_options = [f"{r['record_id']} — {r['zone']} ({r.get('performed_action', r.get('action_title', 'Action'))[:30]}...) [{r.get('verification_status', r.get('evaluation_status'))}]" for r in st.session_state["feedback_records"]]
            selected_fb_idx = st.selectbox("Select Active Record to Evaluate:", range(len(record_options)), format_func=lambda x: record_options[x])
            active_fb = st.session_state["feedback_records"][selected_fb_idx]
            
            eval_evaluator = st.text_input("Reviewing Site Engineer / Inspector:", value=active_fb.get("authorizer_name", active_fb.get("operator_name", "M. Santos (Senior Site Engineer)")))
            eval_status = st.selectbox(
                "Observed Physical Outcome Status:",
                VERIFICATION_STATUSES,
                index=VERIFICATION_STATUSES.index(active_fb.get("verification_status", "Pending Verification")) if active_fb.get("verification_status") in VERIFICATION_STATUSES else 1
            )
            init_eval_pm10 = float(active_fb.get("pm10_after")) if active_fb.get("pm10_after") is not None else 0.0
            eval_obs_pm10 = st.number_input("Observed Post-Intervention PM10 (µg/m³):", min_value=0.0, max_value=500.0, value=init_eval_pm10, step=0.5)
            eval_notes = st.text_area("Field Engineering Observations / Shift Notes:", value=active_fb.get("operator_notes", active_fb.get("field_notes", "Water misting pass completed; visible dust boundary receded. Perimeter readings dropped within 20 mins.")))
            
            if st.button("Save Physical Field Evaluation", use_container_width=True):
                is_valid, v_errs, _ = validate_intervention_input(
                    zone=active_fb.get("zone", "Haul Road"),
                    recommended_action=active_fb.get("recommended_action", active_fb.get("action_title", "Mitigation Action")),
                    performed_action=active_fb.get("performed_action", active_fb.get("action_title", "Mitigation Action")),
                    provenance=active_fb.get("provenance", "Synthetic Demonstration"),
                    pm10_before=active_fb.get("pm10_before"),
                    pm10_after=eval_obs_pm10,
                    verification_status=eval_status,
                    is_authorized=True,
                    authorizer_name=eval_evaluator
                )
                if not is_valid:
                    for err in v_errs:
                        st.error(f"Cannot save evaluation: {err}")
                else:
                    active_fb["authorizer_name"] = eval_evaluator
                    active_fb["operator_name"] = eval_evaluator
                    active_fb["verification_status"] = eval_status
                    active_fb["evaluation_status"] = eval_status
                    active_fb["pm10_after"] = eval_obs_pm10
                    active_fb["operator_notes"] = eval_notes
                    active_fb["field_notes"] = eval_notes
                    active_fb["timestamp_after"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    # Recompute outcome
                    b10 = active_fb.get("pm10_before")
                    outcome = calculate_observed_outcome(b10, eval_obs_pm10)
                    active_fb["pm10_abs_change"] = outcome["pm10_abs_change"]
                    active_fb["pm10_pct_change"] = outcome["pm10_pct_change"]
                    active_fb["pm10_direction"] = outcome["pm10_direction"]
                    active_fb["is_complete"] = outcome["is_complete"]
                    active_fb["is_verified_field_outcome"] = (
                        active_fb.get("provenance") == "Field Measurement" and
                        eval_status == "Field Verified" and
                        outcome["is_complete"] and
                        b10 is not None and b10 > 0
                    )
                    st.success(f"Audit record {active_fb['record_id']} updated with outcome: '{eval_status}'.")
                    st.rerun()
        else:
            st.info("No work orders logged yet this session. Authorize a dispatch ticket above or click 'Pre-Load Demo State' in the sidebar.")
            
    with fb_col2:
        render_html("""
        <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:5px; padding:0.85rem; font-size:0.775rem; color:#475569; line-height:1.5;">
            <b style="color:#0F172A;">Feedback System Governance:</b><br>
            • <b>Human-in-the-Loop</b>: Algorithms provide advisory recommendations; operators confirm physical abatement.<br>
            • <b>Non-Continuous Learning</b>: Feedback records prevent data drift and catastrophic forgetting by avoiding unchecked online parameter mutations.<br>
            • <b>Operational Audit Logging</b>: Timestamped records log operator interventions and field observations for internal shift review and QA tracking. System logs do not constitute formal regulatory compliance filings.
        </div>
        """)
        if st.button("Go to Full Intervention Feedback & Analytics View ➔", use_container_width=True, key="btn_goto_feedback_tab"):
            st.session_state["nav_view"] = "Intervention Feedback"
            st.rerun()
        fb_records = st.session_state.get("feedback_records", [])
        fb_csv = export_intervention_feedback_csv(fb_records).encode('utf-8')
        st.download_button(
            label=f"Download Intervention Feedback Audit ({len(fb_records)} records) (.csv)",
            data=fb_csv,
            file_name=f"dustsentinel_feedback_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
            key="btn_playbook_download_feedback_csv"
        )

    if st.session_state["feedback_records"]:
        st.markdown("##### Structured Feedback & Operational Audit Trail")
        st.dataframe(pd.DataFrame(st.session_state["feedback_records"]), use_container_width=True)

# ---------------------------------------------------------
# VIEW 5: Intervention Sandbox & Multi-Scenario Verification
# ---------------------------------------------------------
def render_intervention_sandbox():
    render_html(f"""
    <div class="brand-header-bar">
        <div>
            <div class="brand-title">Intervention Sandbox & Scenario Verification</div>
            <div class="brand-subtitle">Simulated before-and-after intervention modeling and side-by-side scenario verification</div>
        </div>
        <div>{provenance_badge}</div>
    </div>
    """)

    render_html("""
    <div class="banner-simulation">
        <b>DECISION SUPPORT NOTICE (HYPOTHETICAL SIMULATION MODELS):</b> 
        All metrics, scenario comparisons, and time-series displayed below are <b>SIMULATED PROJECTIONS</b> calculated using empirical engineering dust transfer equations 
        (derived from US EPA AP-42 Compilation of Air Pollutant Emission Factors & WRAP Fugitive Dust Handbook). 
        <b>Simulated reductions do NOT prove or guarantee physical real-world effectiveness under variable site conditions.</b>
    </div>
    """)

    # Multi-Scenario Verification Engine
    st.markdown("#### Multi-Scenario Verification Matrix (Decision Support Comparison)")
    st.caption("Side-by-side comparison of baseline site conditions against pre-configured operational response packages using unique clock-hour exceedance calculations.")

    multi_scenario = compare_mitigation_scenarios(df_clean, interval_hours=0.25)
    scenarios_list = multi_scenario.get("scenarios", [])

    if scenarios_list:
        sc_cols = st.columns(3)
        for idx, sc in enumerate(scenarios_list):
            with sc_cols[idx]:
                border_color = "#E2E8F0" if idx == 0 else ("#BAE6FD" if idx == 1 else "#A7F3D0")
                badge_bg = "#64748B" if idx == 0 else ("#0284C7" if idx == 1 else "#0D9488")
                badge_text = "BASELINE" if idx == 0 else ("RECOMMENDED" if idx == 1 else "MAXIMUM CONTROL")
                
                render_html(f"""
                <div style="background:#FFFFFF; border:1px solid {border_color}; border-radius:6px; padding:1.1rem; height:100%; display:flex; flex-direction:column; justify-content:space-between; box-shadow:0 1px 3px rgba(0,0,0,0.04);">
                    <div>
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem; flex-wrap:wrap; gap:0.25rem;">
                            <span style="background:{badge_bg}; color:#FFFFFF; padding:2px 6px; border-radius:3px; font-size:0.7rem; font-weight:600;">{badge_text}</span>
                            <span style="font-size:0.725rem; color:#64748B;">{html.escape(str(sc['cost_tier']))}</span>
                        </div>
                        <h4 style="color:#0F172A; margin:0.3rem 0; font-size:0.95rem;">{html.escape(str(sc['name']))}</h4>
                        <p style="font-size:0.775rem; color:#64748B; margin-bottom:0.8rem; line-height:1.4;">{html.escape(str(sc['scope']))}</p>
                        
                        <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.5rem; background:#F8FAFC; border:1px solid #E2E8F0; padding:0.6rem; border-radius:4px; margin-bottom:0.6rem;">
                            <div>
                                <div style="font-size:0.675rem; color:#64748B; text-transform:uppercase;">Peak PM10</div>
                                <div style="font-size:1.15rem; font-weight:700; color:{'#DC2626' if sc['peak_pm10'] >= 100 else '#0D9488'};">{sc['peak_pm10']:.1f} <span style="font-size:0.7rem; color:#64748B;">µg/m³</span></div>
                            </div>
                            <div>
                                <div style="font-size:0.675rem; color:#64748B; text-transform:uppercase;">Peak Cut</div>
                                <div style="font-size:1.15rem; font-weight:700; color:{'#64748B' if sc['peak_reduction_pct'] == 0 else '#0D9488'};">-{sc['peak_reduction_pct']:.1f}%</div>
                            </div>
                            <div>
                                <div style="font-size:0.675rem; color:#64748B; text-transform:uppercase;">Exceedance Hrs</div>
                                <div style="font-size:1.15rem; font-weight:700; color:#0F172A;">{sc['exceedance_hours']:.2f} <span style="font-size:0.7rem; color:#64748B;">hrs</span></div>
                            </div>
                            <div>
                                <div style="font-size:0.675rem; color:#64748B; text-transform:uppercase;">Avoided Hrs</div>
                                <div style="font-size:1.15rem; font-weight:700; color:{'#64748B' if sc['avoided_hours'] == 0 else '#0284C7'};">+{sc['avoided_hours']:.2f} <span style="font-size:0.7rem; color:#64748B;">hrs</span></div>
                            </div>
                        </div>
                    </div>
                    <div style="font-size:0.725rem; color:#64748B; border-top:1px solid #E2E8F0; padding-top:0.4rem;">
                        Deployment Time: <b style="color:#0F172A;">{html.escape(str(sc['deployment_time']))}</b> • Model: <span style="color:#64748B;">EPA AP-42</span>
                    </div>
                </div>
                """)

        sc_names = [s["name"].split(":")[0] for s in scenarios_list]
        sc_peaks = [s["peak_pm10"] for s in scenarios_list]
        sc_exceeds = [s["exceedance_hours"] for s in scenarios_list]
        
        fig_sc_bar = go.Figure()
        fig_sc_bar.add_trace(go.Bar(
            name="Peak PM10 (µg/m³)",
            x=sc_names,
            y=sc_peaks,
            marker_color=["#DC2626", "#0284C7", "#0D9488"],
            text=[f"{p:.1f} µg/m³" for p in sc_peaks],
            textposition="auto"
        ))
        fig_sc_bar.add_trace(go.Bar(
            name="EPA Standard Exceedance Hours (hrs)",
            x=sc_names,
            y=sc_exceeds,
            marker_color=["#94A3B8", "#64748B", "#475569"],
            text=[f"{e:.2f} hrs" for e in sc_exceeds],
            textposition="auto"
        ))
        apply_light_theme(fig_sc_bar, height=280)
        fig_sc_bar.update_layout(barmode="group")
        st.plotly_chart(fig_sc_bar, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Custom Scenario Exploration Sandbox")
    
    sim_ctrl1, sim_ctrl2, sim_ctrl3, sim_ctrl4 = st.columns(4)
    with sim_ctrl1:
        st.markdown("**Excavation Zone**")
        exc_spray = st.checkbox("Boom Wetting Sprayers", value=True, help="-68% nominal handbook efficacy")
        exc_barrier = st.checkbox("Geotextile Windbreak", value=False, help="-35% nominal handbook efficacy")
        
    with sim_ctrl2:
        st.markdown("**Haul Road**")
        road_bowser = st.checkbox("Water Bowser 20m Cycle", value=True, help="-70% nominal handbook efficacy")
        road_speed = st.checkbox("15 km/h Speed Limit", value=True, help="-45% nominal handbook efficacy")
        
    with sim_ctrl3:
        st.markdown("**Material Storage**")
        stor_mist = st.checkbox("Oscillating Mist Cannons", value=True, help="-70% nominal handbook efficacy")
        stor_tarp = st.checkbox("Stockpile Geotextile Tarp", value=False, help="-90% nominal handbook efficacy")
        
    with sim_ctrl4:
        st.markdown("**Loading Point**")
        load_fog = st.checkbox("Dry Fog Chute Sprayers", value=True, help="-80% nominal handbook efficacy")
        load_drop = st.checkbox("Drop Height < 1.2m", value=True, help="-50% nominal handbook efficacy")
        
    sim_intensity = st.slider("Intervention Operational Compliance Level (%):", min_value=30, max_value=100, value=85, step=5)
    
    def calc_zone_eff(toggles: list[tuple[bool, float]], compliance: float) -> float:
        rem = 1.0
        for active, val in toggles:
            if active:
                rem *= (1.0 - (val * (compliance / 100.0)))
        return 1.0 - rem

    zone_eff_map = {
        "Excavation": calc_zone_eff([(exc_spray, 0.68), (exc_barrier, 0.35)], sim_intensity),
        "Haul Road": calc_zone_eff([(road_bowser, 0.70), (road_speed, 0.45)], sim_intensity),
        "Material Storage": calc_zone_eff([(stor_mist, 0.70), (stor_tarp, 0.90)], sim_intensity),
        "Loading Point": calc_zone_eff([(load_fog, 0.80), (load_drop, 0.50)], sim_intensity)
    }
    
    df_sim, sim_metrics = calculate_mitigation_simulation(df_clean, zone_eff_map, interval_hours=0.25)
    
    st.markdown("#### Modeled Impact Metrics (Simulated Projections)")
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        st.metric(
            label="Simulated Peak PM10",
            value=f"{sim_metrics['simulated_post_peak_pm10']:.1f} µg/m³",
            delta=f"-{sim_metrics['simulated_reduction_pct']:.1f}% peak cut",
            delta_color="inverse"
        )
    with mcol2:
        st.metric(
            label="Avoided Exceedance Hours",
            value=f"{sim_metrics['simulated_avoided_exceedance_hours']:.2f} hrs",
            delta=f"Baseline: {sim_metrics['simulated_baseline_exceedance_hours']:.2f} hrs",
            delta_color="normal"
        )
    with mcol3:
        st.metric(
            label="Average Site Abatement",
            value=f"{sim_metrics['simulated_avg_site_efficacy']:.1f}%",
            delta="Modeled Transfer"
        )
    with mcol4:
        post_compliance = "[NORMAL] < 100 µg/m³" if sim_metrics['simulated_post_peak_pm10'] < 100 else "[ELEVATED] ≥ 100 µg/m³"
        st.metric(
            label="Projected vs Action Level (100 µg/m³)",
            value=post_compliance,
            delta="15-Min Modeled Level"
        )
        
    sim_zone_select = st.selectbox("Inspect Time-Series Comparison for Zone:", SITE_ZONES, index=1)
    df_zone_pre = df_clean[df_clean["zone"] == sim_zone_select]
    df_zone_sim = df_sim[df_sim["zone"] == sim_zone_select]
    
    fig_sim = go.Figure()
    
    fig_sim.add_trace(go.Scatter(
        x=df_zone_pre["timestamp"],
        y=df_zone_pre["pm10"],
        mode="lines",
        line=dict(color="#DC2626", width=2.5),
        name=f"Observed Baseline PM10 ({sim_zone_select})"
    ))
    
    fig_sim.add_trace(go.Scatter(
        x=df_zone_sim["timestamp"],
        y=df_zone_sim["pm10_simulated"],
        mode="lines",
        line=dict(color="#0D9488", width=2.5, dash="dash"),
        name=f"Simulated Projected PM10 ({sim_zone_select})"
    ))
    
    fig_sim.add_trace(go.Scatter(
        x=pd.concat([df_zone_pre["timestamp"], df_zone_sim["timestamp"][::-1]]),
        y=pd.concat([df_zone_pre["pm10"], df_zone_sim["pm10_simulated"][::-1]]),
        fill="toself",
        fillcolor="rgba(13, 148, 136, 0.12)",
        line=dict(color="rgba(255,255,255,0)"),
        name="Avoided Dust Exposure Burden",
        hoverinfo="none"
    ))
    
    fig_sim.add_hline(
        y=THRESHOLDS["PM10_SITE_ACTION"],
        line_dash="dash",
        line_color="#EA580C",
        annotation_text="Site Action Trigger (15-Min: 100 µg/m³)",
        annotation_position="top left",
        annotation_font=dict(color="#EA580C", size=10)
    )

    fig_sim.add_hline(
        y=THRESHOLDS["PM10_EPA_24H_LIMIT"],
        line_dash="dot",
        line_color="#DC2626",
        annotation_text="EPA Standard (150 µg/m³)",
        annotation_position="top right",
        annotation_font=dict(color="#DC2626", size=10)
    )
    
    apply_light_theme(fig_sim, height=380)
    st.plotly_chart(fig_sim, use_container_width=True)
    
    sim_csv_buffer = io.StringIO()
    df_sim.to_csv(sim_csv_buffer, index=False)
    st.download_button(
        label="Download Simulated Post-Mitigation Dataset (.csv)",
        data=sim_csv_buffer.getvalue().encode('utf-8'),
        file_name=f"dustsentinel_simulated_projection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )

# ---------------------------------------------------------
# VIEW 5B: Intervention Feedback & Outcome Evaluation Workflow
# ---------------------------------------------------------
def render_intervention_feedback():
    render_html(f"""
    <div class="brand-header-bar">
        <div class="brand-header-left">
            <div class="brand-mark-box" style="background:#0F766E;">⟲</div>
            <div>
                <div class="brand-title">Intervention Feedback & Outcome Evaluation</div>
                <div class="brand-subtitle">Closed-loop field verification connecting mitigation actions, ambient sensor readings, and verified outcomes</div>
            </div>
        </div>
        <div class="brand-header-right">
            <div class="header-meta-pill">
                <span class="meta-label">WORKFLOW:</span>
                <span class="meta-val">CLOSED-LOOP AUDIT</span>
            </div>
            <div class="header-meta-pill">
                <span class="meta-label">GOVERNANCE:</span>
                <span class="meta-val">EVIDENCE-INFORMED</span>
            </div>
            <span class="badge-status badge-verified">FIELD VERIFIED PIPELINE</span>
        </div>
    </div>
    """)

    render_html("""
    <div class="banner-simulation">
        <b>OPERATIONAL AUDIT TRAIL & EMPIRICAL EVIDENCE GOVERNANCE:</b><br>
        • <b>Strict Data Separation</b>: Synthetic demonstration benchmarks and field measurements are never mixed in aggregate performance metrics.<br>
        • <b>Non-Continuous Online Learning</b>: Feedback records provide shift audit logging and empirical review; algorithms do not undergo unchecked online parameter mutations.<br>
        • <b>Advisory Decision Support</b>: All recommendations remain decision-support heuristics subject to mandatory human authorization.
    </div>
    """)

    records = st.session_state.get("feedback_records", [])
    analytics = aggregate_feedback_analytics(records, min_observations=3)
    
    # 4-KPI Row with mutually exclusive reconciled counts:
    # Card 2 (Verified Field) + Card 3 (Pending & Incomplete) == Card 1 (Total Interventions)
    vf_stats = analytics["field_verified_metrics"]
    mean_pct_str = f"{vf_stats['mean_pct_reduction']:.1f}%" if vf_stats["mean_pct_reduction"] is not None else "No Field Data"
    mean_abs_sub = f"{vf_stats['mean_abs_reduction_ugm3']:+.1f} µg/m³ PM10" if vf_stats["mean_abs_reduction_ugm3"] is not None else "Awaiting field readings"
    
    val_pending_incomplete = analytics['count_pending'] + analytics['count_unverified_incomplete'] + analytics['count_rejected']
    sub_rejected = f" • {analytics['count_rejected']} Rejected" if analytics['count_rejected'] > 0 else ""
    
    render_html(f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">
                <span>Total Interventions</span>
                <span class="badge-status badge-sample">N = {analytics['total_recorded']}</span>
            </div>
            <div class="kpi-val">{analytics['total_recorded']}</div>
            <div class="kpi-sub">
                <span>Human Authorized:</span>
                <span style="font-weight:600; color:#0F172A;">{analytics['count_authorized']} of {analytics['total_recorded']}</span>
            </div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">
                <span>Verified Field Outcomes</span>
                <span class="badge-status badge-verified">VERIFIED</span>
            </div>
            <div class="kpi-val" style="color:#059669;">{analytics['count_verified_field']}</div>
            <div class="kpi-sub">
                <span>Field Measurements:</span>
                <span style="font-weight:600; color:#0F172A;">{analytics['count_field_total']} logged</span>
            </div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">
                <span>Pending / Incomplete</span>
                <span class="badge-status badge-warning">IN REVIEW</span>
            </div>
            <div class="kpi-val" style="color:#D97706;">{val_pending_incomplete}</div>
            <div class="kpi-sub">
                <span>{analytics['count_pending']} Pending</span>
                <span>• {analytics['count_unverified_incomplete']} Incomplete{sub_rejected}</span>
            </div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">
                <span>Mean Verified Reduction</span>
                <span class="badge-status badge-info">PM10 OBSERVED</span>
            </div>
            <div class="kpi-val" style="color:#1E40AF;">{mean_pct_str}</div>
            <div class="kpi-sub">
                <span>{mean_abs_sub}</span>
                <span>(N={analytics['count_verified_field']})</span>
            </div>
        </div>
    </div>
    """)

    tab_record, tab_evidence, tab_history = st.tabs([
        "📝 Record & Evaluate Intervention",
        "📊 Evidence & Feedback Analytics",
        "📋 Intervention History & Audit Log"
    ])

    with tab_record:
        render_html("""
        <div style="margin-bottom:0.85rem;">
            <div style="font-size:0.95rem; font-weight:600; color:#0F172A;">Log Mitigation Intervention & Sensor Outcome</div>
            <div style="font-size:0.775rem; color:#64748B;">Enter baseline and post-intervention air quality readings, operational details, and human sign-off.</div>
        </div>
        """)

        existing_records = st.session_state.get("feedback_records", [])
        mode_options = ["Record New Intervention"]
        if existing_records:
            mode_options.append("Update Existing Work Order / Inspection")
            
        entry_mode = st.radio("Workflow Entry Mode:", mode_options, horizontal=True, key="entry_mode_selector")
        
        target_record = None
        if entry_mode == "Update Existing Work Order / Inspection":
            rec_labels = [f"{r['record_id']} — {r.get('zone', 'Zone')} ({r.get('performed_action', r.get('action_title', 'Action'))[:26]}...) [{r.get('verification_status', r.get('evaluation_status'))}]" for r in existing_records]
            target_idx = st.selectbox("Select Active Record to Inspect & Update:", range(len(rec_labels)), format_func=lambda i: rec_labels[i], key="target_rec_idx")
            target_record = existing_records[target_idx]

        form_col, preview_col = st.columns([1.55, 1.0])

        with form_col:
            init_zone_idx = SITE_ZONES.index(target_record.get("zone")) if target_record and target_record.get("zone") in SITE_ZONES else 1
            f_zone = st.selectbox("Construction Zone:", SITE_ZONES, index=init_zone_idx, key=f"f_zone_{entry_mode}")
            rec_options = ZONE_RECOMMENDED_ACTIONS.get(f_zone, ["Standard Water Sprinkler Suppression"])
            
            init_rec_action = target_record.get("recommended_action", target_record.get("action_title", rec_options[0])) if target_record else rec_options[0]
            rec_action_idx = rec_options.index(init_rec_action) if init_rec_action in rec_options else 0
            f_recommended = st.selectbox("Recommended Mitigation Action:", rec_options, index=rec_action_idx, key=f"f_recommended_{entry_mode}")
            
            init_performed = target_record.get("performed_action", target_record.get("action_title", f_recommended)) if target_record else f_recommended
            f_performed = st.text_input("Actually Performed Mitigation Action:", value=init_performed, key=f"f_performed_{entry_mode}", help="Confirm if the recommended action was deployed or an alternate operational action was taken.")
            
            row2_col1, row2_col2 = st.columns([1, 1])
            with row2_col1:
                init_prov = target_record.get("provenance", "Field Measurement") if target_record else "Field Measurement"
                prov_idx = PROVENANCE_OPTIONS.index(init_prov) if init_prov in PROVENANCE_OPTIONS else 2
                f_provenance = st.selectbox("Data Provenance:", PROVENANCE_OPTIONS, index=prov_idx, key=f"f_provenance_{entry_mode}", help="Strictly categorize the origin of these observations.")
            with row2_col2:
                # Default new intervention to Pending Verification to prevent unperformed actions from being marked Field Verified
                init_verif = target_record.get("verification_status", "Field Verified") if target_record else "Pending Verification"
                verif_idx = VERIFICATION_STATUSES.index(init_verif) if init_verif in VERIFICATION_STATUSES else 1
                f_verification = st.selectbox("Verification Status:", VERIFICATION_STATUSES, index=verif_idx, key=f"f_verification_{entry_mode}", help="Field Verified requires actual field measurements and complete valid baseline readings.")

            row3_col1, row3_col2 = st.columns([1, 1])
            with row3_col1:
                init_auth = bool(target_record.get("is_authorized", True)) if target_record else True
                f_is_authorized = st.checkbox("Authorized by Responsible Human", value=init_auth, key=f"f_is_auth_{entry_mode}")
            with row3_col2:
                init_auth_name = target_record.get("authorizer_name", target_record.get("operator_name", "M. Santos (Senior Site Engineer)")) if target_record else "M. Santos (Senior Site Engineer)"
                f_authorizer = st.text_input("Authorizing Engineer / Operator:", value=init_auth_name, key=f"f_authorizer_{entry_mode}")

            st.markdown("##### Particulate Matter Telemetry Readings")
            render_html("<div style='font-size:0.75rem; color:#64748B; margin-bottom:0.5rem;'>Do not convert missing readings to zero. If post-intervention reading is pending, uncheck the box.</div>")
            
            pm_col1, pm_col2 = st.columns([1, 1])
            with pm_col1:
                init_has_b10 = (target_record.get("pm10_before") is not None) if target_record else True
                has_pm10_before = st.checkbox("Baseline PM10 Recorded", value=init_has_b10, key=f"has_pm10_before_{entry_mode}")
                if has_pm10_before:
                    if target_record and target_record.get("pm10_before") is not None:
                        init_b10_val = float(target_record.get("pm10_before"))
                    else:
                        active_zone_pts = df_clean[df_clean["zone"] == f_zone]["pm10"] if ("df_clean" in locals() and "pm10" in df_clean.columns) else []
                        init_b10_val = round(float(active_zone_pts.iloc[-1]), 1) if len(active_zone_pts) > 0 else 0.0
                    f_pm10_before = st.number_input("PM10 Before Intervention (µg/m³):", min_value=0.0, max_value=800.0, value=init_b10_val, step=0.5, key=f"f_pm10_b_{entry_mode}")
                else:
                    f_pm10_before = None

            with pm_col2:
                # Default unchecked for new interventions to avoid stale cached/magic readings being presented as saved field outcomes
                init_has_a10 = (target_record.get("pm10_after") is not None) if target_record else False
                has_pm10_after = st.checkbox("Post-Intervention PM10 Recorded", value=init_has_a10, key=f"has_pm10_after_{entry_mode}")
                if has_pm10_after:
                    init_a10_val = float(target_record.get("pm10_after")) if (target_record and target_record.get("pm10_after") is not None) else 0.0
                    f_pm10_after = st.number_input("PM10 After Intervention (µg/m³):", min_value=0.0, max_value=800.0, value=init_a10_val, step=0.5, key=f"f_pm10_a_{entry_mode}")
                else:
                    f_pm10_after = None

            with st.expander("PM2.5 Telemetry (Optional)", expanded=False):
                init_has_pm25 = (target_record and (target_record.get("pm25_before") is not None or target_record.get("pm25_after") is not None))
                has_pm25 = st.checkbox("Include PM2.5 Telemetry", value=init_has_pm25, key=f"has_pm25_{entry_mode}")
                if has_pm25:
                    pm25_c1, pm25_c2 = st.columns(2)
                    with pm25_c1:
                        init_pm25_b = float(target_record.get("pm25_before") or 28.0) if target_record else 28.0
                        f_pm25_before = st.number_input("PM2.5 Before (µg/m³):", min_value=0.0, max_value=400.0, value=init_pm25_b, step=0.5, key=f"f_pm25_b_{entry_mode}")
                    with pm25_c2:
                        init_pm25_a = float(target_record.get("pm25_after") or 12.5) if target_record else 12.5
                        f_pm25_after = st.number_input("PM2.5 After (µg/m³):", min_value=0.0, max_value=400.0, value=init_pm25_a, step=0.5, key=f"f_pm25_a_{entry_mode}")
                else:
                    f_pm25_before = None
                    f_pm25_after = None

            st.markdown("##### Observation Interval & Meteorological Observations")
            env_c1, env_c2, env_c3 = st.columns([1, 1, 1])
            with env_c1:
                init_interval = int(target_record.get("observation_interval_mins") or 30) if target_record else 30
                f_interval = st.number_input("Interval (Minutes):", min_value=5, max_value=240, value=init_interval, step=5, key=f"f_interval_{entry_mode}")
            with env_c2:
                init_ws = float(target_record.get("wind_speed_kmh") or 14.2) if target_record else 14.2
                f_wind_speed = st.number_input("Wind Speed (km/h):", min_value=0.0, max_value=120.0, value=init_ws, step=0.5, key=f"f_ws_{entry_mode}")
            with env_c3:
                init_wd = target_record.get("wind_direction", "NW") if target_record else "NW"
                wd_options = ["NW", "NNW", "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "Variable"]
                wd_idx = wd_options.index(init_wd) if init_wd in wd_options else 0
                f_wind_dir = st.selectbox("Wind Direction:", wd_options, index=wd_idx, key=f"f_wd_{entry_mode}")

            ts_now = datetime.now()
            ts_before_default = target_record.get("timestamp_before") if target_record and target_record.get("timestamp_before") else (ts_now - timedelta(minutes=int(f_interval))).strftime("%Y-%m-%d %H:%M:%S")
            ts_after_default = target_record.get("timestamp_after") if target_record and target_record.get("timestamp_after") else ts_now.strftime("%Y-%m-%d %H:%M:%S")
            
            ts_c1, ts_c2 = st.columns([1, 1])
            with ts_c1:
                f_ts_before = st.text_input("Baseline Timestamp:", value=ts_before_default, key=f"f_ts_b_{entry_mode}")
            with ts_c2:
                f_ts_after = st.text_input("Post-Intervention Timestamp:", value=ts_after_default, key=f"f_ts_a_{entry_mode}")

            init_wn = target_record.get("weather_notes", "Dry soil, moderate gusting from NW.") if target_record else "Dry soil, moderate gusting from NW."
            f_weather_notes = st.text_input("Atmospheric Conditions:", value=init_wn, key=f"f_wn_{entry_mode}")
            
            init_on = target_record.get("operator_notes", target_record.get("field_notes", "Water bowser completed 2 circuits; surface silt stabilized. Plume receded below action limit within 20 mins.")) if target_record else "Water bowser completed 2 circuits; surface silt stabilized. Plume receded below action limit within 20 mins."
            f_notes = st.text_area("Operator Shift Notes & Observations:", value=init_on, key=f"f_notes_{entry_mode}")

            btn_label = "💾 Update Existing Intervention Record" if target_record else "💾 Record New Intervention Outcome to Session"
            if st.button(btn_label, use_container_width=True, key=f"btn_save_record_{entry_mode}"):
                is_valid, validation_errors, cleaned = validate_intervention_input(
                    zone=f_zone,
                    recommended_action=f_recommended,
                    performed_action=f_performed,
                    provenance=f_provenance,
                    pm10_before=f_pm10_before,
                    pm10_after=f_pm10_after,
                    timestamp_before=f_ts_before,
                    timestamp_after=f_ts_after,
                    verification_status=f_verification,
                    is_authorized=f_is_authorized,
                    authorizer_name=f_authorizer,
                    pm25_before=f_pm25_before if has_pm25 else None,
                    pm25_after=f_pm25_after if has_pm25 else None,
                    wind_speed=f_wind_speed,
                    wind_direction=f_wind_dir
                )
                if not is_valid:
                    for err in validation_errors:
                        st.error(f"Validation Error: {err}")
                else:
                    if target_record:
                        target_record["zone"] = cleaned["zone"]
                        target_record["recommended_action"] = cleaned["recommended_action"]
                        target_record["performed_action"] = cleaned["performed_action"]
                        target_record["action_title"] = cleaned["performed_action"]
                        target_record["action_alignment"] = "Matches Recommendation" if cleaned["recommended_action"].strip().lower() == cleaned["performed_action"].strip().lower() else "Alternative Action Performed"
                        target_record["provenance"] = cleaned["provenance"]
                        target_record["is_field_measurement"] = (cleaned["provenance"] == "Field Measurement")
                        target_record["verification_status"] = cleaned["verification_status"]
                        target_record["evaluation_status"] = cleaned["verification_status"]
                        target_record["is_authorized"] = cleaned["is_authorized"]
                        target_record["authorizer_name"] = cleaned["authorizer_name"]
                        target_record["operator_name"] = cleaned["authorizer_name"]
                        target_record["pm10_before"] = cleaned["pm10_before"]
                        target_record["pm10_after"] = cleaned["pm10_after"]
                        target_record["pm25_before"] = cleaned["pm25_before"]
                        target_record["pm25_after"] = cleaned["pm25_after"]
                        target_record["timestamp_before"] = cleaned["timestamp_before"]
                        target_record["timestamp_after"] = cleaned["timestamp_after"]
                        target_record["observation_interval_mins"] = int(f_interval)
                        target_record["wind_speed_kmh"] = cleaned["wind_speed"]
                        target_record["wind_direction"] = cleaned["wind_direction"]
                        target_record["weather_notes"] = f_weather_notes
                        target_record["operator_notes"] = f_notes
                        target_record["field_notes"] = f_notes
                        
                        outcome = calculate_observed_outcome(cleaned["pm10_before"], cleaned["pm10_after"], cleaned["pm25_before"], cleaned["pm25_after"])
                        target_record["pm10_abs_change"] = outcome["pm10_abs_change"]
                        target_record["pm10_pct_change"] = outcome["pm10_pct_change"]
                        target_record["pm10_direction"] = outcome["pm10_direction"]
                        target_record["is_complete"] = outcome["is_complete"]
                        target_record["scientific_caveat"] = outcome["scientific_caveat"]
                        target_record["is_verified_field_outcome"] = (
                            cleaned["provenance"] == "Field Measurement" and
                            cleaned["verification_status"] == "Field Verified" and
                            outcome["is_complete"] and
                            (cleaned["pm10_before"] or 0) > 0 and
                            bool(cleaned["is_authorized"])
                        )
                        st.success(f"Intervention record '{target_record['record_id']}' updated with status '{cleaned['verification_status']}'.")
                    else:
                        new_rec = create_intervention_record(
                            zone=cleaned["zone"],
                            recommended_action=cleaned["recommended_action"],
                            performed_action=cleaned["performed_action"],
                            provenance=cleaned["provenance"],
                            pm10_before=cleaned["pm10_before"],
                            pm10_after=cleaned["pm10_after"],
                            pm25_before=cleaned["pm25_before"],
                            pm25_after=cleaned["pm25_after"],
                            timestamp_before=cleaned["timestamp_before"],
                            timestamp_after=cleaned["timestamp_after"],
                            observation_interval_mins=int(f_interval),
                            wind_speed=cleaned["wind_speed"],
                            wind_direction=cleaned["wind_direction"],
                            weather_notes=f_weather_notes,
                            is_authorized=cleaned["is_authorized"],
                            authorizer_name=cleaned["authorizer_name"],
                            operator_notes=f_notes,
                            verification_status=cleaned["verification_status"]
                        )
                        st.session_state["feedback_records"].insert(0, new_rec)
                        st.success(f"Intervention record '{new_rec['record_id']}' successfully logged to session audit trail.")
                    st.rerun()

        with preview_col:
            # Real-time preview calculation
            live_outcome = calculate_observed_outcome(
                pm10_before=f_pm10_before,
                pm10_after=f_pm10_after,
                pm25_before=f_pm25_before if has_pm25 else None,
                pm25_after=f_pm25_after if has_pm25 else None
            )

            completeness_badge = '<span class="badge-status badge-verified">COMPLETE READINGS</span>' if live_outcome["is_complete"] else '<span class="badge-status badge-warning">INCOMPLETE DATA</span>'
            
            dir_color = "#059669" if live_outcome["pm10_direction"] == "Decreased" else ("#DC2626" if live_outcome["pm10_direction"] == "Increased" else "#64748B")
            dir_badge = f'<span class="badge-status" style="background:#F8FAFC; color:{dir_color}; border:1px solid #CBD5E1;">{live_outcome["pm10_direction"].upper()}</span>'

            b10_str = f"{live_outcome['pm10_before']:.1f} µg/m³" if live_outcome["pm10_before"] is not None else "Not Recorded"
            a10_str = f"{live_outcome['pm10_after']:.1f} µg/m³" if live_outcome["pm10_after"] is not None else "Pending / Missing"
            
            abs_str = f"{live_outcome['pm10_abs_change']:+.1f} µg/m³" if live_outcome["pm10_abs_change"] is not None else "N/A"
            pct_str = f"{live_outcome['pm10_pct_change']:.1f}%" if live_outcome["pm10_pct_change"] is not None else ("N/A (Zero Baseline)" if live_outcome["pm10_before"] == 0.0 else "N/A")

            # Verification eligibility warning
            eligibility_html = ""
            if f_verification == "Field Verified":
                if f_provenance != "Field Measurement":
                    eligibility_html = "<div style='color:#DC2626; font-size:0.75rem; font-weight:600; margin-top:0.5rem;'>❌ Ineligible for Field Verified: Data Provenance must be 'Field Measurement'.</div>"
                elif not live_outcome["is_complete"] or (live_outcome["pm10_before"] or 0) <= 0:
                    eligibility_html = "<div style='color:#DC2626; font-size:0.75rem; font-weight:600; margin-top:0.5rem;'>❌ Ineligible for Field Verified: Complete before & after PM10 measurements (baseline > 0) required.</div>"
                elif not f_is_authorized:
                    eligibility_html = "<div style='color:#DC2626; font-size:0.75rem; font-weight:600; margin-top:0.5rem;'>❌ Ineligible for Field Verified: Responsible human authorization required.</div>"
                else:
                    eligibility_html = "<div style='color:#059669; font-size:0.75rem; font-weight:600; margin-top:0.5rem;'>✓ Meets criteria for Field Verified status</div>"

            render_html(f"""
            <div class="dashboard-panel" style="margin-bottom:1rem;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.75rem;">
                    <div class="panel-title">Observed Outcome Preview</div>
                    {completeness_badge}
                </div>
                
                <div style="font-size:0.775rem; color:#64748B; margin-bottom:0.6rem;">
                    Target: <b style="color:#0F172A;">{f_zone}</b> • Action: <b style="color:#0F172A;">{f_performed[:28]}...</b>
                </div>

                <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.5rem; margin-bottom:0.75rem;">
                    <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:4px; padding:0.5rem 0.65rem;">
                        <div style="font-size:0.675rem; color:#94A3B8; text-transform:uppercase; font-weight:600;">Baseline PM10</div>
                        <div style="font-size:1.1rem; font-weight:700; color:#0F172A;">{b10_str}</div>
                    </div>
                    <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:4px; padding:0.5rem 0.65rem;">
                        <div style="font-size:0.675rem; color:#94A3B8; text-transform:uppercase; font-weight:600;">Post-Intervention</div>
                        <div style="font-size:1.1rem; font-weight:700; color:#0F172A;">{a10_str}</div>
                    </div>
                </div>

                <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.5rem; margin-bottom:0.75rem;">
                    <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:4px; padding:0.5rem 0.65rem;">
                        <div style="font-size:0.675rem; color:#94A3B8; text-transform:uppercase; font-weight:600;">Absolute Delta</div>
                        <div style="font-size:1.1rem; font-weight:700; color:{dir_color};">{abs_str}</div>
                        <div style="font-size:0.65rem; color:#64748B;">(+ indicates reduction)</div>
                    </div>
                    <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:4px; padding:0.5rem 0.65rem;">
                        <div style="font-size:0.675rem; color:#94A3B8; text-transform:uppercase; font-weight:600;">Relative Delta</div>
                        <div style="font-size:1.1rem; font-weight:700; color:{dir_color};">{pct_str}</div>
                        <div style="font-size:0.65rem; color:#64748B;">((before-after)/before)</div>
                    </div>
                </div>

                <div style="display:flex; justify-content:space-between; align-items:center; padding:0.5rem 0; border-top:1px solid #E2E8F0;">
                    <span style="font-size:0.75rem; color:#475569; font-weight:600;">Observed Trend:</span>
                    {dir_badge}
                </div>

                {eligibility_html}

                <div style="font-size:0.725rem; color:#64748B; font-style:italic; padding:0.55rem; background:#F8FAFC; border:1px solid #E2E8F0; border-radius:4px; margin-top:0.75rem; line-height:1.4;">
                    <b>Scientific Disclaimer:</b> {live_outcome["scientific_caveat"]}
                </div>
            </div>
            """)

    with tab_evidence:
        render_html("""
        <div style="margin-bottom:0.85rem;">
            <div style="font-size:0.95rem; font-weight:600; color:#0F172A;">Evidence-Informed Feedback Analytics & Recommendations</div>
            <div style="font-size:0.775rem; color:#64748B;">Summarizes verified field outcomes by mitigation action. Strictly isolates synthetic simulations from field performance.</div>
        </div>
        """)

        ctrl_col1, ctrl_col2 = st.columns([1.5, 1.0])
        with ctrl_col1:
            evidence_min_obs = st.slider(
                "Minimum Verified Field Observations Required for Empirical Guidance (N_min):",
                min_value=1,
                max_value=6,
                value=3,
                key="evidence_min_obs_slider",
                help="Evidence threshold before an action is classified as empirically verified in the field."
            )
        with ctrl_col2:
            if st.button("➕ Inject 3 Verified Field Observations for Demonstration", key="btn_inject_verified_demo", help="Demonstrates how the analytics engine transitions from 'Insufficient verified field data' to 'Sufficient Verified Field Data' once N >= 3."):
                now_demo = datetime.now()
                for i, (b, a, ws) in enumerate([(138.0, 42.0, 15.0), (124.0, 36.0, 12.0), (145.0, 45.0, 17.5)], start=1):
                    st.session_state["feedback_records"].insert(0, create_intervention_record(
                        zone="Haul Road",
                        recommended_action="Automated Water Bowser Misting Cycle",
                        performed_action="Automated Water Bowser Misting Cycle",
                        provenance="Field Measurement",
                        pm10_before=b,
                        pm10_after=a,
                        timestamp_before=(now_demo - timedelta(hours=i*2, minutes=30)).strftime("%Y-%m-%d %H:%M:%S"),
                        timestamp_after=(now_demo - timedelta(hours=i*2)).strftime("%Y-%m-%d %H:%M:%S"),
                        wind_speed=ws,
                        wind_direction="NW",
                        is_authorized=True,
                        authorizer_name="M. Santos (Senior Site Engineer)",
                        operator_notes=f"Demo verified pass #{i}: moisture dampening confirmed by field inspector.",
                        verification_status="Field Verified",
                        record_id=f"INT-DEMO-VERIFIED-00{i}"
                    ))
                st.success("Injected 3 verified field records for Haul Road misting. Threshold met!")
                st.rerun()

        # Re-run aggregation with slider threshold
        evidence_analytics = aggregate_feedback_analytics(st.session_state["feedback_records"], min_observations=evidence_min_obs)
        action_map = evidence_analytics["action_summaries"]

        st.markdown("##### Action-by-Action Field Verification Analytics")
        if not action_map:
            st.info("No recorded interventions found. Record interventions in Tab 1 or click 'Pre-Load Demo State' in the sidebar.")
        else:
            for act_name, act_info in action_map.items():
                is_suff = act_info["has_sufficient_evidence"]
                badge_html = f'<span class="badge-status badge-verified">SUFFICIENT EVIDENCE (N = {act_info["verified_field_count"]})</span>' if is_suff else f'<span class="badge-status badge-warning">INSUFFICIENT VERIFIED FIELD DATA (N = {act_info["verified_field_count"]} < {evidence_min_obs})</span>'
                card_border = "#BBF7D0" if is_suff else "#FDE68A"
                card_bg = "#F0FDF4" if is_suff else "#FFFBEB"

                metric_sub = ""
                if is_suff:
                    metric_sub = f"""
                    <div style="display:flex; gap:1.5rem; margin-top:0.4rem; margin-bottom:0.4rem;">
                        <div><span style="color:#64748B; font-size:0.75rem;">Observed Mean Reduction:</span> <b style="color:#059669; font-size:0.95rem;">{act_info['mean_pct_reduction']:.1f}%</b></div>
                        <div><span style="color:#64748B; font-size:0.75rem;">Absolute PM10 Drop:</span> <b style="color:#059669; font-size:0.95rem;">{act_info['mean_abs_reduction_ugm3']:+.1f} µg/m³</b></div>
                    </div>
                    """

                render_html(f"""
                <div style="background:{card_bg}; border:1px solid {card_border}; border-radius:6px; padding:0.85rem 1rem; margin-bottom:0.75rem;">
                    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem; margin-bottom:0.35rem;">
                        <div style="font-weight:700; color:#0F172A; font-size:0.925rem;">{act_name}</div>
                        {badge_html}
                    </div>
                    {metric_sub}
                    <div style="font-size:0.8rem; color:#334155; line-height:1.4; margin-bottom:0.25rem;">
                        {act_info['summary_text']}
                    </div>
                    <div style="font-size:0.75rem; color:#64748B; font-style:italic; border-top:1px dashed #CBD5E1; padding-top:0.35rem; margin-top:0.35rem;">
                        <b>Advisory Guidance:</b> {act_info['guidance_note']}
                    </div>
                </div>
                """)

        st.markdown("##### Theoretical Simulation Benchmarks (Strictly Excluded from Field Metrics)")
        render_html(f"""
        <div style="background:#F8FAFC; border:1px solid #BAE6FD; border-left:4px solid #0284C7; padding:0.85rem 1rem; border-radius:4px; margin-bottom:1rem;">
            <b style="color:#0369A1;">PROVENANCE SEPARATION SAFEGUARD:</b><br>
            <p style="font-size:0.8rem; color:#334155; line-height:1.4; margin:0.3rem 0 0.5rem 0;">
                The metrics below reflect {evidence_analytics['synthetic_metrics']['count']} synthetic demonstration models and uploaded benchmark files only. 
                They are mathematically isolated and never pooled into field-verified empirical performance.
            </p>
            <div style="display:flex; gap:1.5rem; font-size:0.775rem;">
                <div>Synthetic Benchmarks: <b>{evidence_analytics['synthetic_metrics']['count']} records</b></div>
                <div>Mean Simulated Reduction: <b>{evidence_analytics['synthetic_metrics']['mean_pct_reduction'] if evidence_analytics['synthetic_metrics']['mean_pct_reduction'] is not None else 'N/A'}%</b></div>
                <div>Mean Simulated Delta: <b>{evidence_analytics['synthetic_metrics']['mean_abs_reduction_ugm3'] if evidence_analytics['synthetic_metrics']['mean_abs_reduction_ugm3'] is not None else 'N/A'} µg/m³</b></div>
            </div>
        </div>
        """)

    with tab_history:
        render_html("""
        <div style="margin-bottom:0.85rem;">
            <div style="font-size:0.95rem; font-weight:600; color:#0F172A;">Intervention History & Operational Audit Trail</div>
            <div style="font-size:0.775rem; color:#64748B;">Filterable history log with data provenance, verification status, and one-click CSV export.</div>
        </div>
        """)

        fil_c1, fil_c2, fil_c3, fil_c4 = st.columns([1, 1, 1, 1.2])
        with fil_c1:
            h_zone = st.selectbox("Filter Zone:", ["All Zones"] + SITE_ZONES, index=0, key="h_zone")
        with fil_c2:
            h_prov = st.selectbox("Filter Provenance:", ["All Provenances"] + PROVENANCE_OPTIONS, index=0, key="h_prov")
        with fil_c3:
            h_stat = st.selectbox("Filter Status:", ["All Statuses"] + VERIFICATION_STATUSES, index=0, key="h_stat")
        with fil_c4:
            h_query = st.text_input("Search ID / Notes / Authorizer:", value="", key="h_query")

        filtered_recs = filter_intervention_records(
            st.session_state["feedback_records"],
            zone=h_zone,
            provenance=h_prov,
            verification_status=h_stat,
            search_query=h_query
        )

        st.caption(f"Displaying **{len(filtered_recs)}** of **{len(st.session_state['feedback_records'])}** recorded interventions")

        if filtered_recs:
            table_rows = []
            for r in filtered_recs:
                abs_ch = f"{r.get('pm10_abs_change'):+.1f}" if r.get("pm10_abs_change") is not None else "—"
                pct_ch = f"{r.get('pm10_pct_change'):.1f}%" if r.get("pm10_pct_change") is not None else "—"
                b10 = f"{r.get('pm10_before'):.1f}" if r.get("pm10_before") is not None else "—"
                a10 = f"{r.get('pm10_after'):.1f}" if r.get("pm10_after") is not None else "Pending"
                
                table_rows.append({
                    "Record ID": r.get("record_id", ""),
                    "Linked Order": r.get("linked_order_id", "—") or "—",
                    "Logged At": r.get("timestamp", ""),
                    "Zone": r.get("zone", ""),
                    "Mitigation Action": r.get("performed_action", r.get("action_title", "")),
                    "Provenance": r.get("provenance", "Synthetic Demonstration"),
                    "Verification Status": r.get("verification_status", r.get("evaluation_status", "Pending Verification")),
                    "Authorized": "Yes" if r.get("is_authorized") else "No",
                    "Authorizer": r.get("authorizer_name", r.get("operator_name", "")),
                    "PM10 Before": b10,
                    "PM10 After": a10,
                    "Δ Abs (µg/m³)": abs_ch,
                    "Δ Pct (%)": pct_ch,
                    "Trend": r.get("pm10_direction", "—"),
                    "Wind (km/h)": r.get("wind_speed_kmh", "—"),
                    "Notes": r.get("operator_notes", r.get("field_notes", ""))[:50] + ("..." if len(r.get("operator_notes", r.get("field_notes", ""))) > 50 else "")
                })
            
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True)
        else:
            st.info("No intervention records match the active filter criteria or session feedback log is empty.")

        exp_col1, exp_col2, exp_col3 = st.columns([1.5, 1.0, 1.0])
        with exp_col1:
            csv_bytes = export_intervention_feedback_csv(filtered_recs).encode('utf-8')
            st.download_button(
                label=f"Download Filtered Audit Trail ({len(filtered_recs)} records) (.csv)",
                data=csv_bytes,
                file_name=f"dustsentinel_intervention_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
                key="btn_download_feedback_csv"
            )
        with exp_col2:
            if st.button("Reset to Demo Feedback Records", use_container_width=True, key="btn_reset_demo_records"):
                st.session_state["feedback_records"] = generate_sample_feedback_records()
                st.rerun()
        with exp_col3:
            with st.expander("⚠️ Clear Feedback Records", expanded=False):
                st.caption("Clears only in-memory intervention feedback records. Does **not** delete dispatch work orders, telemetry, or uploads.")
                confirm_chk = st.checkbox("Confirm clearing feedback logs", key="chk_confirm_clear_feedback")
                if st.button("Confirm Clear", disabled=not confirm_chk, use_container_width=True, key="btn_clear_all_records"):
                    st.session_state["feedback_records"] = []
                    st.rerun()

        render_html("""
        <div style="background:#F1F5F9; border:1px solid #CBD5E1; border-radius:6px; padding:0.85rem; margin-top:1.25rem; font-size:0.775rem; color:#475569; line-height:1.5;">
            <b>Storage & Export Notice:</b> Records are retained exclusively in volatile in-memory session state (<code>st.session_state["feedback_records"]</code>) for shift review and demonstration purposes. Records do not persist across server or browser restarts. Exports are unauthenticated, non-cryptographic CSV files; download exports promptly for offline recordkeeping.
        </div>
        """)

# ---------------------------------------------------------
# VIEW 6: Sensor Analytics & Forecasting Baseline
# ---------------------------------------------------------
def render_sensor_analytics():
    render_html(f"""
    <div class="brand-header-bar">
        <div>
            <div class="brand-title">Sensor Analytics & Short-Horizon Risk Forecasting</div>
            <div class="brand-subtitle">Chronological train/test validation, 2-hour forward projection, and multi-zone telemetry analysis</div>
        </div>
        <div>{provenance_badge}</div>
    </div>
    """)
    
    # FEATURE 1: Dust-Risk Forecasting Baseline (Experimental)
    st.markdown("#### Short-Horizon Dust-Risk Forecasting Baseline (Experimental)")
    render_html("""
    <div class="banner-simulation">
        <b>DECISION SUPPORT NOTICE (EXPERIMENTAL SHORT-HORIZON PROJECTION):</b> 
        This baseline model is trained strictly on chronologically split historical time-series without lookahead data leakage. 
        Predictions are benchmarked against a naive persistence baseline ($y_t = y_{t-1}$). 
        Because 15-minute particulate time series exhibit aerodynamic turbulence and synthetic/sample provenance, this feature is labeled <b>EXPERIMENTAL</b> rather than claiming validated predictive performance.
    </div>
    """)
    
    train_df, test_df = split_chronological(df_clean, test_ratio=0.25)
    forecast_eval = train_short_horizon_forecast(train_df, test_df, horizon_steps=1)
    
    if forecast_eval.get("status") == "evaluated" and forecast_eval.get("test_mae") is not None:
        fc_col1, fc_col2, fc_col3, fc_col4 = st.columns(4)
        with fc_col1:
            st.metric(
                label="Test MAE (Model vs Persistence)",
                value=f"{forecast_eval['test_mae']:.2f} µg/m³",
                delta=f"Baseline: {forecast_eval['baseline_persistence_mae']:.2f} µg/m³",
                delta_color="inverse"
            )
        with fc_col2:
            st.metric(
                label="Test RMSE (Model vs Persistence)",
                value=f"{forecast_eval['test_rmse']:.2f} µg/m³",
                delta=f"Baseline: {forecast_eval['baseline_persistence_rmse']:.2f} µg/m³",
                delta_color="inverse"
            )
        with fc_col3:
            skill = forecast_eval["relative_skill_score_pct"]
            st.metric(
                label="Relative Skill Score vs Naive",
                value=f"{skill:+.1f}%",
                delta="Skill Score" if skill >= 0 else "Persistence Ahead",
                delta_color="normal" if skill >= 0 else "inverse"
            )
        with fc_col4:
            split_meta = forecast_eval.get("split_metadata", {})
            render_html(f"""
            <div class="kpi-card" style="min-height:86px;">
                <div class="kpi-label">Chronological Split</div>
                <div style="font-size:0.85rem; font-weight:600; color:#0F172A;">Zero Lookahead Leakage</div>
                <div class="kpi-sub">Train: {split_meta.get('train_records', len(train_df))} • Test: {split_meta.get('test_records', len(test_df))}</div>
            </div>
            """)
    else:
        split_meta = forecast_eval.get("split_metadata", {})
        tr_cnt = split_meta.get("train_records", len(train_df))
        te_cnt = split_meta.get("test_records", len(test_df))
        st.warning(
            f"⚠️ **Insufficient data for evaluation:** Insufficient historical time-series records "
            f"(Train: {tr_cnt} • Test: {te_cnt}) to compute valid out-of-sample forecast metrics. "
            f"A minimum of 8 historical training observations and 2 test observations per zone are required. "
            f"Zero metrics are withheld to prevent misleading performance claims."
        )
        
    # Forward 2-Hour Projection
    st.markdown("##### 2-Hour Forward Particulate Risk Projection")
    forward_forecast = generate_forward_forecast(df_clean, horizon_steps=8, interval_mins=15)
    
    col_fc_zone, col_fc_status = st.columns([1.2, 1.8])
    with col_fc_zone:
        peak_z = risk_summary["peak_zone"] if risk_summary["peak_zone"] in SITE_ZONES else SITE_ZONES[0]
        fc_zone_select = st.selectbox("Select Zone for Forward Horizon Projection:", SITE_ZONES, index=SITE_ZONES.index(peak_z))
        
    df_fc_zone = forward_forecast[forward_forecast["zone"] == fc_zone_select]
    df_hist_zone = df_clean[df_clean["zone"] == fc_zone_select].tail(16)
    
    max_proj_pm10 = float(df_fc_zone["pm10_forecast"].max()) if not df_fc_zone.empty else 0.0
    will_exceed = max_proj_pm10 >= THRESHOLDS["PM10_SITE_ACTION"]
    
    with col_fc_status:
        if will_exceed:
            render_html(f"""
            <div class="banner-warning" style="margin-top:0.4rem; padding:0.6rem 1rem;">
                <b>ACTION ALERT TRIGGER PROJECTED:</b> Projected peak of <b>{max_proj_pm10:.1f} µg/m³</b> within +2 hours for {html.escape(fc_zone_select)}. 
                Recommend proactive misting dispatch prior to threshold exceedance.
            </div>
            """)
        else:
            render_html(f"""
            <div class="banner-simulation" style="margin-top:0.4rem; padding:0.6rem 1rem;">
                <b>STABLE PROJECTION:</b> Projected peak of <b>{max_proj_pm10:.1f} µg/m³</b> remains within normal operational envelope (&lt; 100 µg/m³).
            </div>
            """)

    fig_fwd = go.Figure()
    
    fig_fwd.add_trace(go.Scatter(
        x=df_hist_zone["timestamp"],
        y=df_hist_zone["pm10"],
        mode="lines+markers",
        line=dict(color="#0284C7", width=2),
        marker=dict(size=4),
        name=f"Observed Recent PM10 ({fc_zone_select})"
    ))
    
    fig_fwd.add_trace(go.Scatter(
        x=df_fc_zone["timestamp"],
        y=df_fc_zone["pm10_forecast"],
        mode="lines+markers",
        line=dict(color="#D97706", width=2.5, dash="dash"),
        marker=dict(size=6, symbol="diamond"),
        name="Experimental Forward Projection"
    ))
    
    fig_fwd.add_trace(go.Scatter(
        x=pd.concat([df_fc_zone["timestamp"], df_fc_zone["timestamp"][::-1]]),
        y=pd.concat([df_fc_zone["pm10_upper"], df_fc_zone["pm10_lower"][::-1]]),
        fill="toself",
        fillcolor="rgba(217, 119, 6, 0.12)",
        line=dict(color="rgba(255,255,255,0)"),
        name="Projection Uncertainty Envelope (95% CI)",
        hoverinfo="none"
    ))
    
    fig_fwd.add_hline(
        y=THRESHOLDS["PM10_SITE_ACTION"],
        line_dash="dash",
        line_color="#EA580C",
        annotation_text="Site Action Trigger (100 µg/m³)",
        annotation_position="top left",
        annotation_font=dict(color="#EA580C", size=10)
    )

    apply_light_theme(fig_fwd, height=350)
    st.plotly_chart(fig_fwd, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Multi-Zone Historical Sensor Analytics")
    
    metric_choice = st.radio("Select Pollutant Metric to Plot:", ["PM10 (Coarse + Fine)", "PM2.5 (Fine)", "PM2.5 / PM10 Ratio"], horizontal=True)
    
    fig_hist = go.Figure()
    
    for z in SITE_ZONES:
        zdf = df_clean[df_clean["zone"] == z]
        if metric_choice == "PM10 (Coarse + Fine)":
            y_val = zdf["pm10"]
            y_title = "PM10 (µg/m³)"
        elif metric_choice == "PM2.5 (Fine)":
            y_val = zdf["pm2_5"]
            y_title = "PM2.5 (µg/m³)"
        else:
            y_val = zdf["pm2_5"] / zdf["pm10"].replace(0, np.nan)
            y_title = "PM2.5 / PM10 Ratio"
            
        fig_hist.add_trace(go.Scatter(
            x=zdf["timestamp"],
            y=y_val,
            mode="lines",
            line=dict(color=ZONE_COLORS.get(z, "#64748B"), width=2),
            name=z
        ))
        
    apply_light_theme(fig_hist, height=360)
    fig_hist.update_layout(yaxis=dict(title=y_title))

    if metric_choice == "PM10 (Coarse + Fine)":
        fig_hist.add_hline(
            y=100.0,
            line_dash="dash",
            line_color="#EA580C",
            annotation_text="Site Action Trigger (15-Min: 100 µg/m³)",
            annotation_position="top left",
            annotation_font=dict(color="#EA580C", size=10)
        )
        fig_hist.add_hline(
            y=150.0,
            line_dash="dot",
            line_color="#DC2626",
            annotation_text="EPA Standard (150 µg/m³)",
            annotation_position="top right",
            annotation_font=dict(color="#DC2626", size=10)
        )
    elif metric_choice == "PM2.5 (Fine)":
        fig_hist.add_hline(
            y=35.0,
            line_dash="dot",
            line_color="#DC2626",
            annotation_text="EPA Standard (35 µg/m³)",
            annotation_position="top right",
            annotation_font=dict(color="#DC2626", size=10)
        )
    elif metric_choice == "PM2.5 / PM10 Ratio":
        fig_hist.add_hline(
            y=0.35,
            line_dash="dot",
            line_color="#0284C7",
            annotation_text="Crustal Dust Ceiling (< 0.35)",
            annotation_position="top right",
            annotation_font=dict(color="#0284C7", size=10)
        )

    st.plotly_chart(fig_hist, use_container_width=True)
    
    col_rose, col_scatter = st.columns(2)
    
    with col_rose:
        st.markdown("#### Wind Rose / Dust by Direction")
        df_rose = df_clean.copy()
        df_rose["cardinal"] = df_rose["wind_deg"].apply(get_cardinal_direction)
        rose_agg = df_rose.groupby(["cardinal", "zone"])["pm10"].mean().reset_index()
        
        fig_polar = px.bar_polar(
            rose_agg,
            r="pm10",
            theta="cardinal",
            color="zone",
            color_discrete_map=ZONE_COLORS
        )
        fig_polar.update_layout(
            height=340,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#F8FAFC",
            font=dict(family="Inter, sans-serif", color="#334155"),
            legend=dict(orientation="h", y=-0.1)
        )
        st.plotly_chart(fig_polar, use_container_width=True)
        
    with col_scatter:
        st.markdown("#### Particulate Size Scatter (PM2.5 vs PM10)")
        fig_scatter = px.scatter(
            df_clean,
            x="pm10",
            y="pm2_5",
            color="zone",
            color_discrete_map=ZONE_COLORS,
            hover_data=["timestamp", "wind_speed", "humidity", "is_anomalous"],
            labels={"pm10": "PM10 (µg/m³)", "pm2_5": "PM2.5 (µg/m³)"}
        )
        max_pm10 = float(df_clean["pm10"].max()) if not df_clean.empty else 100.0
        fig_scatter.add_trace(go.Scatter(
            x=[0, max_pm10],
            y=[0, max_pm10 * 0.25],
            mode="lines",
            line=dict(color="#94A3B8", dash="dot"),
            name="Crustal Baseline (Ratio 0.25)",
            hoverinfo="name"
        ))
        apply_light_theme(fig_scatter, height=340)
        st.plotly_chart(fig_scatter, use_container_width=True)

# ---------------------------------------------------------
# VIEW 7: Data Health & Pipeline Diagnostics
# ---------------------------------------------------------
def render_data_health():
    render_html(f"""
    <div class="brand-header-bar">
        <div>
            <div class="brand-title">Data Health & Pipeline Ingestion Diagnostics</div>
            <div class="brand-subtitle">Schema verification, anomaly isolation, and automated cleansing audit records</div>
        </div>
        <div>{provenance_badge}</div>
    </div>
    """)
    
    col_v1, col_v2 = st.columns([1.2, 1.0])
    
    with col_v1:
        st.markdown("#### Validation Audit Summary")
        st.markdown(f"""
        - **Validation Status:** {"[PASSED ALL CHECKS]" if val_report['valid'] else "[FAILED CHECKS]"}
        - **Total Records Ingested:** `{val_report['total_rows']}`
        - **Time Span Covered:** `{val_report['time_span']}`
        - **Missing Values Handled:** `{val_report['missing_values_handled']}` (Zone-Scoped)
        - **Duplicate Records Detected:** `{val_report['duplicate_count']}`
        - **Identified Construction Zones:** `{', '.join(df_clean['zone'].unique())}`
        """)
        
        if val_report["anomalies"]:
            with st.expander("Flagged Data Quality Anomalies (Retained without silent alteration):", expanded=True):
                for a in val_report["anomalies"]:
                    st.markdown(f"- {a}")

        if val_report["fixes"]:
            with st.expander("Automatic Data Cleansing Actions Applied:", expanded=True):
                for f in val_report["fixes"]:
                    st.markdown(f"- {f}")
        elif not val_report["anomalies"]:
            st.info("No missing or malformed records detected in current stream.")
            
    with col_v2:
        st.markdown("#### Export Clean Sensor Dataset")
        csv_buffer = io.StringIO()
        df_clean.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode('utf-8')
        
        st.download_button(
            label="Download Clean Active Sensor Data (.csv)",
            data=csv_bytes,
            file_name=f"dustsentinel_sensor_clean_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        st.markdown("#### Expected CSV Schema Format")
        st.code("""timestamp,zone,pm2_5,pm10,wind_speed,wind_deg,humidity,temperature,activity_level
2026-10-13 06:00:00,Excavation,22.0,51.0,8.5,239.5,61.1,25.1,Low
2026-10-13 06:00:00,Haul Road,20.6,35.0,8.5,239.5,61.1,25.1,None
2026-10-13 06:00:00,Material Storage,19.3,41.6,8.5,239.5,61.1,25.1,None
2026-10-13 06:00:00,Loading Point,19.1,46.8,8.5,239.5,61.1,25.1,None""", language="csv")
        
    st.markdown("---")
    st.markdown("#### Raw Data Preview (First 20 Records)")
    st.dataframe(df_clean.head(20), use_container_width=True)

# ---------------------------------------------------------
# VIEW 8: Reports & Export Hub
# ---------------------------------------------------------
def render_reports_and_export():
    render_html(f"""
    <div class="brand-header-bar">
        <div>
            <div class="brand-title">Audit Reports & Data Export Center</div>
            <div class="brand-subtitle">Consolidated operational records, telemetry archives, and compliance logs</div>
        </div>
        <div>{provenance_badge}</div>
    </div>
    """)
    
    render_html("""
    <div class="banner-simulation">
        <b>EXPORT FORMATTING & OPERATIONAL AUDIT NOTICE:</b> All exportable CSV files generated below are structured, formatted, and timestamped tabular text files for internal operational review and engineering shift audits. 
        <b>DustSentinel exports are non-cryptographic, unauthenticated text files; they do not provide cryptographic hashing, digital signatures, or certified proof of regulatory compliance.</b>
    </div>
    """)
    
    exp_col1, exp_col2 = st.columns(2)
    
    with exp_col1:
        render_html("""
        <div class="dashboard-panel" style="margin-bottom:0.75rem;">
            <div class="panel-title">1. Clean Active Sensor Telemetry Dataset</div>
            <div class="panel-subtitle">Fully validated, zone-isolated particulate and meteorological records</div>
            <p style="font-size:0.8rem; color:#475569; line-height:1.4; margin-top:0.4rem; margin-bottom:0;">
                Contains timestamps, zone identifiers, PM10, PM2.5, wind parameters, and data quality anomaly flags. 
                Cleansed using zone-scoped interpolation without cross-zone contamination.
            </p>
        </div>
        """)
        csv_buffer = io.StringIO()
        df_clean.to_csv(csv_buffer, index=False)
        st.download_button(
            label=f"Download Clean Telemetry CSV ({len(df_clean)} records) (.csv)",
            data=csv_buffer.getvalue().encode('utf-8'),
            file_name=f"dustsentinel_telemetry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        render_html("""
        <div class="dashboard-panel" style="margin-top:1rem; margin-bottom:0.75rem;">
            <div class="panel-title">2. Operational Mitigation Work Orders</div>
            <div class="panel-subtitle">Authorized field dispatch directives and assigned crew instructions</div>
            <p style="font-size:0.8rem; color:#475569; line-height:1.4; margin-top:0.4rem; margin-bottom:0;">
                Formal work-order directives signed off by site environmental leads. Includes target zones, assigned equipment units, and handbook efficacy estimates.
            </p>
        </div>
        """)
        dispatch_orders = st.session_state.get("dispatch_log", [])
        dispatch_headers = [
            "order_id", "timestamp", "target_zone", "assigned_crew",
            "action", "authorizer", "current_pm10", "status", "linked_record_id"
        ]
        if dispatch_orders:
            disp_df = pd.DataFrame(dispatch_orders)
            for col in dispatch_headers:
                if col not in disp_df.columns:
                    disp_df[col] = ""
            disp_csv = disp_df[dispatch_headers].to_csv(index=False).encode('utf-8')
        else:
            disp_csv = (",".join(dispatch_headers) + "\n").encode('utf-8')

        st.download_button(
            label=f"Download Dispatch Log ({len(dispatch_orders)} orders) (.csv)",
            data=disp_csv,
            file_name=f"dustsentinel_dispatch_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    with exp_col2:
        render_html("""
        <div class="dashboard-panel" style="margin-bottom:0.75rem;">
            <div class="panel-title">3. Multi-Scenario Mitigation Projections</div>
            <div class="panel-subtitle">Simulated post-mitigation particulate series and avoided burden calculations</div>
            <p style="font-size:0.8rem; color:#475569; line-height:1.4; margin-top:0.4rem; margin-bottom:0;">
                Full mathematical comparison of Baseline, Targeted Fast-Response, and Comprehensive Controls packages. 
                Derived using empirical engineering transfer functions (US EPA AP-42 & WRAP Handbook).
            </p>
        </div>
        """)
        sim_df, _ = calculate_mitigation_simulation(df_clean, {"Haul Road": 0.70, "Excavation": 0.68}, interval_hours=0.25)
        if "provenance" not in sim_df.columns:
            sim_df["provenance"] = "Synthetic Demonstration Projection"
        sim_csv = sim_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"Download Simulated Projections ({len(sim_df)} records) (.csv)",
            data=sim_csv,
            file_name=f"dustsentinel_mitigation_sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        render_html("""
        <div class="dashboard-panel" style="margin-top:1rem; margin-bottom:0.75rem;">
            <div class="panel-title">4. Closed-Loop Intervention Feedback Audit Trail</div>
            <div class="panel-subtitle">Human-verified physical field outcomes, calculated deltas, and operational notes</div>
            <p style="font-size:0.8rem; color:#475569; line-height:1.4; margin-top:0.4rem; margin-bottom:0;">
                Field verification records connecting mitigation actions with before/after PM10 readings,
                data provenance, authorizer sign-off, and scientific non-causal disclaimers.
            </p>
        </div>
        """)
        fb_records = st.session_state.get("feedback_records", [])
        fb_csv = export_intervention_feedback_csv(fb_records).encode('utf-8')
        st.download_button(
            label=f"Download Feedback Audit Trail ({len(fb_records)} records) (.csv)",
            data=fb_csv,
            file_name=f"dustsentinel_feedback_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
            key="btn_reports_download_fb_csv"
        )
        if not fb_records:
            st.caption("ℹ️ Session feedback log is currently empty. Download will export a schema-compliant header-only CSV.")
        if st.button("Open Dedicated Intervention Feedback Interface ➔", use_container_width=True, key="btn_reports_goto_feedback"):
            st.session_state["nav_view"] = "Intervention Feedback"
            st.rerun()

# ---------------------------------------------------------
# VIEW 9 & 10: Legal Governance Disclosures
# ---------------------------------------------------------
def render_privacy_policy():
    render_html(f"""
    <div class="brand-header-bar">
        <div>
            <div class="brand-title">DustSentinel | Privacy & Data Governance Policy</div>
            <div class="brand-subtitle">Technical disclosure of environmental data ingestion, processing, and memory boundaries for the Tech For Earth 2026 hackathon prototype</div>
        </div>
        <div><span class="badge-status badge-sample">[DRAFT DATA GOVERNANCE v1.2]</span></div>
    </div>
    """)

    render_html("""
    <div style="background:#FFFBEB; border:1px solid #FCD34D; border-left:4px solid #D97706; padding:0.85rem 1rem; border-radius:4px; font-size:0.825rem; color:#92400E; margin-top:0.75rem; margin-bottom:0.75rem;">
        <b>DRAFT DISCLOSURE — PRE-PUBLICATION OWNER REQUIREMENTS:</b><br>
        This document is an experimental prototype disclosure prepared for the Tech For Earth Hackathon 2026. 
        It does not constitute finalized legal advice. Prior to public release or commercial deployment, the project owner and qualified legal counsel must formalize the following items:
        <ul style="margin:0.4rem 0 0.2rem 1.2rem; padding:0;">
            <li><b>Entity / Licensor Name</b>: Operating organization or corporate entity</li>
            <li><b>Governing Jurisdiction</b>: Applicable state, national, or regional laws</li>
            <li><b>Formal Effective Date</b>: Operational commencement date</li>
            <li><b>Legal & Privacy Contact Email</b>: Monitored compliance inbox</li>
            <li><b>Physical Business Address</b>: Registered office or principal place of business</li>
            <li><b>Data Protection Officer (DPO)</b>: Contact details (if mandated under applicable data protection laws)</li>
        </ul>
    </div>
    """)

    if st.button("⬅ Return to Overview Dashboard", key="btn_return_priv_top"):
        st.session_state["nav_view"] = "Overview"
        st.rerun()

    render_html("""
    <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:6px; padding:1.25rem 1.5rem; margin-top:1rem; margin-bottom:1.5rem; box-shadow:0 1px 3px rgba(0,0,0,0.04);">
        <div style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.05em; color:#0284C7; font-weight:600; margin-bottom:0.4rem;">
            Document Governance & Draft Metadata
        </div>
        <div style="font-size:0.85rem; color:#475569; line-height:1.6;">
            <b>Entity / Licensor</b>: <i>Not yet specified (Draft Hackathon Prototype — Operating entity name to be supplied prior to publication)</i><br>
            <b>Project Context</b>: DustSentinel — Tech For Earth Hackathon 2026 Decision Support Prototype<br>
            <b>Governing Jurisdiction</b>: <i>Not yet specified (Governing jurisdiction to be determined by project owner prior to release)</i><br>
            <b>Effective Date</b>: <i>Draft for Review (Formal effective date to be established upon release)</i><br>
            <b>Privacy & Compliance Contact</b>: <i>Not yet configured (Compliance contact email to be supplied by project owner)</i><br>
            <b>Data Protection Officer</b>: <i>Not appointed (To be evaluated under local data protection laws prior to commercialization)</i><br>
            <b>Physical Business Address</b>: <i>Not yet specified (Principal place of business to be supplied by operating entity)</i>
        </div>
    </div>
    """)

    st.markdown("""
    ### 1. Prototype Scope & System Nature
    DustSentinel is an **experimental software prototype** developed for the Tech For Earth Hackathon 2026. The application demonstrates an AI-assisted environmental instrumentation and mitigation decision-support workflow for fugitive dust monitoring on construction sites.

    **No "Zero-Tracking" Claim**: Because DustSentinel operates as a client-server web application on the Streamlit framework, standard HTTP request metadata and WebSocket communication channels are required for network transport. We do not claim zero tracking or client-only isolation. This disclosure describes our actual technical data architecture honestly and transparently.

    ---

    ### 2. Categories of Ingested Information
    | Data Category | Ingestion Method | Technical Handling | Personal Data (PII) Implication |
    | :--- | :--- | :--- | :--- |
    | **Synthetic Test Streams** | Algorithmic Generator (`generate_synthetic_stream`) | Diurnal Gaussian and log-normal particulate distributions generated in volatile memory | **Zero Personal Data**. Fully synthetic parameters. |
    | **Sample Datasets** | Bundled CSV (`sample_sensor_data.csv`) | Pre-packaged static test data for demonstration walkthroughs | **Zero Personal Data**. Non-identifying environmental readings. |
    | **User-Uploaded CSVs** | File Uploader (`st.file_uploader`) | Timestamp, zone identifier, PM2.5, PM10, wind speed, wind direction, humidity, temperature | **Environmental Telemetry Only**. Users must not upload PII or confidential site data. |
    | **Mitigation Work Orders** | Interactive Dashboard Forms | Assigned crew ID, authorizing engineer name/title, operational instructions, dispatch timestamp | **Volatile Operational Logs** retained only in session state (`st.session_state["dispatch_log"]`). |
    | **Technical Access Logs** | Hosting Server Layer (Streamlit / Cloud Host) | Client IP address, browser User-Agent, HTTP headers, request timestamps, WebSocket session state | **Standard Infrastructure Logs** necessary for web service operation. |

    ---

    ### 3. Server-Side Data Handling, Volatile Memory & Retention
    1. **In-Memory Volatile Processing**:
       - Uploaded CSV telemetry, synthetic data streams, and active simulation parameters are held exclusively in volatile server process memory (`st.session_state["raw_df"]`) for the duration of the user's active session.
       - Dispatched mitigation work orders and human operator feedback logs are stored in in-memory session arrays (`st.session_state["dispatch_log"]` and `st.session_state["feedback_records"]`).
       - **No External Database Persistence**: The hackathon prototype does not connect to an external persistent database, cloud storage bucket, or third-party CRM.
    2. **Session Termination**:
       - When the browser tab is closed or the WebSocket connection disconnects, Streamlit automatically tears down the session context and deallocates memory.
       - Temporary file buffers created during CSV downloads are flushed immediately after transmission.

    ---

    ### 4. Algorithmic Processing, Heuristics & Forecasting Boundaries
    1. **Deterministic Heuristic Source Attribution**:
       - The Source Attribution engine computes a heuristic ranking index (0–100) using transparent multi-criteria rules (PM2.5/PM10 ratio, wind vector alignment, activity level).
       - This process runs in-memory without training black-box models on user data and without sending telemetry to external AI services.
    2. **Experimental Short-Horizon Forecasting**:
       - Time-series dust forecasts are generated locally using chronological feature splits and baseline persistence models. No user telemetry is shared with external model vendors.
    3. **No Continuous Model Retraining**:
       - Operator field feedback records are retained for shift audit logging and demonstration purposes only. DustSentinel **does not perform continuous online retraining** without validated offline evaluation loops.

    ---

    ### 5. Third-Party Hosting & External Dependencies
    1. **Hosting Environment**:
       - When hosted on Streamlit Community Cloud or similar demonstration platforms, server infrastructure, container orchestration, and network transport are governed by the respective hosting provider's data handling policies.
    2. **External Assets**:
       - Typography is loaded from Google Fonts (`Inter`). Plotly charting libraries run client-side in browser WebGL/Canvas contexts.
       - DustSentinel does not incorporate external tracking trackers, advertising pixels, or telemetry brokers.

    ---

    ### 6. User Rights & Contact Details
    For inquiries regarding this prototype disclosure, or to request removal of session logs upon commercial release, contact:
    - **Draft Privacy Contact**: *Not yet configured (Contact email to be provided by project owner prior to release)*
    - **Designated DPO Placeholder**: *Not appointed (To be evaluated under local data protection laws prior to commercialization)*
    - **Governing Entity Placeholder**: *Not yet specified (Operating entity and physical address to be supplied by project owner)*
    """)

    if st.button("⬅ Return to Overview Dashboard", key="btn_return_priv_bottom"):
        st.session_state["nav_view"] = "Overview"
        st.rerun()

def render_terms_and_conditions():
    render_html(f"""
    <div class="brand-header-bar">
        <div>
            <div class="brand-title">DustSentinel | Terms of Use & Operational Boundaries</div>
            <div class="brand-subtitle">Operational decision-support framework, engineering boundaries, and legal disclaimers for the Tech For Earth 2026 hackathon prototype</div>
        </div>
        <div><span class="badge-status badge-sample">[DRAFT TERMS v1.2]</span></div>
    </div>
    """)

    render_html("""
    <div style="background:#FFFBEB; border:1px solid #FCD34D; border-left:4px solid #D97706; padding:0.85rem 1rem; border-radius:4px; font-size:0.825rem; color:#92400E; margin-top:0.75rem; margin-bottom:0.75rem;">
        <b>DRAFT TERMS — PRE-PUBLICATION OWNER REQUIREMENTS:</b><br>
        These terms govern an experimental decision-support prototype developed for the Tech For Earth Hackathon 2026. 
        They do not constitute finalized legal advice. Prior to public production deployment, the project owner and qualified legal counsel must formalize:
        <ul style="margin:0.4rem 0 0.2rem 1.2rem; padding:0;">
            <li><b>Operating Corporate Entity / Licensor Name</b></li>
            <li><b>Governing Jurisdiction & Choice of Law</b></li>
            <li><b>Formal Effective Date</b></li>
            <li><b>Legal Notice Channel & Monitored Email</b></li>
            <li><b>Principal Place of Business</b></li>
            <li><b>Designated Dispute Resolution Forum / Arbitration Venue</b></li>
        </ul>
    </div>
    """)

    if st.button("⬅ Return to Overview Dashboard", key="btn_return_terms_top"):
        st.session_state["nav_view"] = "Overview"
        st.rerun()

    render_html("""
    <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:6px; padding:1.25rem 1.5rem; margin-top:1rem; margin-bottom:1.5rem; box-shadow:0 1px 3px rgba(0,0,0,0.04);">
        <div style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.05em; color:#0284C7; font-weight:600; margin-bottom:0.4rem;">
            Governing Legal Framework & Prototype Status
        </div>
        <div style="font-size:0.85rem; color:#475569; line-height:1.6;">
            <b>Licensor / Provider</b>: <i>Not yet specified (Draft Hackathon Prototype — Corporate entity to be supplied prior to publication)</i><br>
            <b>Project Context</b>: DustSentinel — Tech For Earth Hackathon 2026 Decision Support Prototype<br>
            <b>Governing Jurisdiction</b>: <i>Not yet specified (Governing jurisdiction to be determined by project owner prior to release)</i><br>
            <b>Effective Date</b>: <i>Draft for Review (Formal effective date to be established upon release)</i><br>
            <b>Legal Notice Channel</b>: <i>Not yet configured (Legal contact email to be provided by project owner)</i><br>
            <b>Principal Business Address</b>: <i>Not yet specified (Principal place of business to be supplied by operating entity)</i>
        </div>
    </div>
    """)

    st.markdown("""
    ### 1. Prototype Status & Acceptance of Terms
    By accessing, viewing, or interacting with DustSentinel, you acknowledge that you are evaluating an **experimental decision-support prototype** developed for the Tech For Earth Hackathon 2026. If you do not agree to these terms, do not use the application.

    ---

    ### 2. Operational Nature: Advisory Decision Support Only
    1. **No Regulatory Certification (Non-FRM/FEM)**:
       - DustSentinel is **NOT** a Federal Reference Method (FRM) or Federal Equivalent Method (FEM) air monitoring device under United States EPA 40 CFR Part 58 or equivalent international statutory bodies.
       - Telemetry indicators, 15-minute action thresholds (100 µg/m³), and 24-hour benchmark references are operational screening tools only. Outputs do **not** constitute official legal proof of regulatory compliance or statutory violation.
    2. **Synthetic Data and Unverified Sensor Telemetry**:
       - Default telemetry streams are synthetically simulated or loaded from sample test records.
       - User-uploaded CSV datasets are processed as raw inputs without independent laboratory or calibration verification. The prototype makes no warranty regarding the physical accuracy of sensor inputs.
    3. **Heuristic Source Attribution Limitations**:
       - Source attribution rankings (0–100) are explainable, deterministic multi-criteria screening heuristics (combining PM2.5/PM10 ratios, wind vector dispersion, and activity tags).
       - Scores represent screening indicators, **not** formal Bayesian causal probabilities, dispersion proofs, or legal fault attributions.
    4. **Experimental Forecasting Boundaries**:
       - Short-horizon forecasting models are experimental baselines using time-series rolling trends. They do not represent certified meteorological predictions or atmospheric dispersion simulations.
    5. **Simulation Projections vs. Physical Field Reality**:
       - The Intervention Sandbox projects potential particulate cuts using empirical transfer literature (US EPA AP-42 Compilation of Air Pollutant Emission Factors & WRAP Fugitive Dust Handbook).
       - Projected percentage reductions (e.g. 60%–85%) and avoided exceedance hours are **mathematical approximations**. Physical mitigation outcomes vary significantly with soil compaction, moisture retention, vehicle speed, and atmospheric wind turbulence. Mitigation results are **not guaranteed**.

    ---

    ### 3. Human Authorization Requirement & User Responsibilities
    1. **Mandatory Human-in-the-Loop Authorization**:
       - All mitigation work orders, equipment dispatch tickets, and site operational adjustments generated by DustSentinel are **advisory recommendations only**.
       - No stop-work order, water bowser dispatch, or speed restriction may be executed without independent physical verification and formal authorization by a designated site engineer or environmental officer.
    2. **Operational Duty of Care**:
       - Site personnel maintain exclusive responsibility for workplace safety, statutory compliance, and environmental management on construction sites.

    ---

    ### 4. Disclaimer of Warranties & Limitation of Liability
    1. **"As-Is" Demonstration Prototype**:
       - DustSentinel is provided **"AS IS" and "AS AVAILABLE"**, without warranty of any kind, express or implied, including warranties of merchantability, fitness for a particular purpose, non-infringement, or uninterrupted availability.
    2. **Limitation of Liability**:
       - To the maximum extent permitted by applicable law in the governing jurisdiction to be determined by the project owner, the authors, contributors, and prototype developers shall not be liable for any direct, indirect, incidental, special, consequential, or punitive damages arising from the use of, or inability to use, this prototype, including environmental fines, site shutdowns, or equipment costs.

    ---

    ### 5. Governing Law & Dispute Resolution Placeholder
    - These terms shall be governed by and construed in accordance with the laws of the applicable jurisdiction to be designated by the operating entity prior to publication.
    - Any unresolved disputes shall be submitted to the competent courts or arbitration forum designated by legal counsel upon formal release.
    - Legal correspondence: *Contact details to be supplied upon publication* | *Physical address to be supplied upon publication*
    """)

    if st.button("⬅ Return to Overview Dashboard", key="btn_return_terms_bottom"):
        st.session_state["nav_view"] = "Overview"
        st.rerun()

# ---------------------------------------------------------
# Dynamic View Router
# ---------------------------------------------------------
current_view = st.session_state.get("nav_view", "Overview")

if current_view == "Overview":
    render_overview()
elif current_view == "Spatial Map & Plume":
    render_spatial_map()
elif current_view == "Source Attribution":
    render_source_attribution()
elif current_view == "Mitigation Playbook":
    render_mitigation_playbook()
elif current_view == "Intervention Sandbox":
    render_intervention_sandbox()
elif current_view == "Intervention Feedback":
    render_intervention_feedback()
elif current_view == "Sensor Analytics":
    render_sensor_analytics()
elif current_view == "Data Health & Pipeline":
    render_data_health()
elif current_view == "Reports & Export":
    render_reports_and_export()
elif current_view == "Privacy Policy":
    render_privacy_policy()
elif current_view == "Terms of Use":
    render_terms_and_conditions()
else:
    render_overview()

render_html("""
<div style="margin-top:2.5rem; padding:1rem 1.25rem; border-top:1px solid #E2E8F0; display:flex; justify-content:space-between; align-items:center; font-size:0.75rem; color:#64748B; background:#FFFFFF; border-radius:6px; box-shadow:0 1px 2px rgba(0,0,0,0.02); flex-wrap:wrap; gap:0.5rem;">
    <div><b>DustSentinel v1.1</b> • Construction Dust Monitoring & Mitigation Decision Support Prototype</div>
    <div>Tech For Earth Hackathon 2026 • Experimental Decision Support Prototype • Draft — Not Finalized Legal Advice</div>
</div>
""")
