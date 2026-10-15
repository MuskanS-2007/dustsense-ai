# DustSentinel — Fugitive Dust Monitoring & Mitigation Platform

> **Tech For Earth 2026 — Innovative Dust Mitigation Solutions**  
> *Track: Construction Air Quality & Fugitive Dust Management*

---

## Executive Summary

**DustSentinel** is an explainable construction site fugitive dust monitoring and mitigation platform. Traditional construction air quality systems are purely reactive—they register an air quality index (AQI) exceedance only after particulate matter has already dispersed into adjacent communities and worker zones.

DustSentinel transforms monitoring from passive alerting into **proactive source attribution and targeted engineering intervention**. By continuously analyzing particulate size fractions ($PM_{2.5} / PM_{10}$ ratios), meteorological aerodynamics (wind vector gradients, humidity desiccation, gust thresholds), and site operational schedules, DustSentinel:
1. **Identifies and ranks the most likely fugitive dust sources** across four site zones using explainable heuristic scores ($0\text{–}100$).
2. **Visualizes spatial dispersion plumes** drifting toward sensitive site perimeter receptors.
3. **Recommends source-specific mitigation playbooks** (water bowser misting cycles, vehicle speed caps, atomized dry fog chutes, and surface tackifiers).
4. **Simulates before-and-after intervention results** to model potential exceedance hour reductions prior to equipment deployment, using mathematically sound unique-timestamp clock hour accounting.
5. **Audits and validates sensor CSV data**, isolating missing-value imputation within zones and flagging physical anomalies ($PM_{2.5} > PM_{10}$, negative readings) without silent data alteration.

---

## Monitored Site Zones

DustSentinel monitors four distinct operational zones, each possessing a characteristic aerodynamic particulate signature:

| Site Zone | Spatial Sector | Characteristic Particulate Dynamics | Primary Emission Mechanism |
| :--- | :--- | :--- | :--- |
| **Excavation** | North-West Quadrant | High $PM_{10}$, low $PM_{2.5}/PM_{10}$ ratio (0.18–0.28) | Mechanical cut-and-fill, topsoil stripping, bulldozer blade shear |
| **Haul Road** | Central Unpaved Corridor | High episodic spikes, high short-term variance ($CV > 0.35$) | Heavy dump truck tire-road shear, silt kicking from unpaved surfaces |
| **Material Storage** | North-East Quadrant | Strongly wind-correlated ($U > 18\text{ km/h}$) & low humidity | Aerodynamic wind erosion and shear lift off exposed aggregate mounds |
| **Loading Point** | South-East Quadrant | Cyclical hopper and drop-batch particulate plumes | Bulk aggregate drop into haulers, transfer chute displacement drafts |

---

## Key Dashboard Capabilities

### 1. Transparent Synthetic Data Generator & Real-Time Metrics
- Generates physically realistic, diurnal time-series for $PM_{2.5}$, $PM_{10}$, wind speed, wind direction, ambient temperature, and relative humidity.
- Selectable scenario presets: **Active Peak Shift (High Dust Event)**, **Dry Windy Alert Condition**, and **Standard Operations**.
- **Prominently labelled with badges** to distinguish synthetic test streams from uploaded CSV datasets.
- Deterministic random seed and normalized timestamp floored to 15-minute intervals for test reproducibility.

### 2. 2D Interactive Site Layout & Dispersion Map
- Plotly-powered schematic of the $100\text{m} \times 100\text{m}$ construction parcel.
- Shows perimeter fences, the central haul corridor, and adjacent **vulnerable residential/highway receptor boundaries**.
- **Dynamic Wind Vector Arrow** indicating prevailing wind velocity and compass heading.
- **Simulated Dispersion Cone** tracing dust drift from the highest-emitting zone toward perimeter receptors.
- Color-coded zone sensor nodes (Green < 45, Yellow 45–100, Orange 100–150, Red > 150 µg/m³).

### 3. Explainable Rule-Based Source Attribution (Zero Black-Box Deception)
- Adheres strictly to scientific integrity: **no unverified machine learning models are claimed**.
- Multi-criteria deterministic rule engine evaluates:
  - **Particulate Granularity**: Mechanical crustal dust exhibits coarse fraction dominance ($PM_{10} \gg PM_{2.5}$), while diesel exhaust skews fine ($PM_{2.5}/PM_{10} > 0.45$).
  - **Receptor Alignment**: Computes angular delta between downwind plume heading and sensitive boundary receptors.
  - **Aerodynamic Erosion Thresholds**: Identifies stockpile lift when sustained winds exceed $18\text{ km/h}$ combined with relative humidity $< 40\%$.
  - **Turbulence Spikiness Index**: Detects haul road truck pass transients via localized coefficient of variation.
- Outputs transparent **Attribution Ranking Scores ($0\text{–}100$)** alongside bulleted causality explanations.

### 4. Targeted Mitigation Playbook & Operational Work Orders
- Source-specific playbook detailing intervention type, deployment timeframe, estimated cost tier, and expected reduction efficacy.
- **Operational Dispatch Generator**: Generates and exports downloadable mitigation work orders (`.txt`) for site equipment operators and water bowser teams.
- **Session Dispatch Log**: Persists issued work orders in an active table during the operator's session.

### 5. Simulated Before-and-After Intervention Sandbox
- Interactive scenario testing: toggle specific physical and operational controls per zone (e.g., *Boom Mist Sprayers*, *Water Bowser Cycles*, *Speed Enforcement*, *Mist Cannons*, *Dry Fog Enclosures*).
- Multi-stage abatement modeling based on empirical transfer efficiencies from **US EPA AP-42 Compilation of Air Pollutant Emission Factors** and the **WRAP Fugitive Dust Handbook**.
- **Mathematically Sound Clock Hour Accounting**: Calculates avoided EPA exceedance hours across *unique elapsed timestamps*, preventing multi-zone quadruple counting.
- **Downloadable Projections**: Export the complete simulated post-mitigation dataset (`df_sim`) as CSV.
- **Clearly labelled**: All results are explicitly marked as *Simulated Projections* for decision support, not measured outcomes.

### 6. CSV Data Ingestion & Validation Pipeline
- Drag-and-drop custom site CSV uploads with a safety limit of $15\text{ MB}$.
- **Zone-Scoped Imputation**: Missing values are interpolated strictly within individual zones, never bleeding across zone boundaries.
- **Duplicate Detection**: Flags conflicting records sharing identical `(timestamp, zone)` keys.
- **Anomaly Preservation**: Inverted particulate readings ($PM_{2.5} > PM_{10}$) and negative values are preserved and flagged with audit notes rather than silently altered.

---

## Technology Stack

- **Python 3.10+** (Validated on Python 3.14)
- **Streamlit**: Interactive web dashboard and reactive state management
- **Pandas**: Time-series manipulation, data validation, and aggregation
- **NumPy**: Physical calculations, circular angle mathematics, and synthetic signal processing
- **Plotly**: Dynamic 2D spatial layouts, polar wind roses, and comparative before/after time-series charts
- **Pytest**: Automated unit testing suite

---

## Installation & Running on Windows

### 1. Clone or Open Workspace
Open PowerShell in the workspace root directory:
```powershell
cd c:\Projects\dustsense-ai
```

### 2. Verify Python Installation
```powershell
python --version
pip --version
```

### 3. Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Run Automated Unit Tests
```powershell
pytest -v
```

### 5. Run the Streamlit Application
```powershell
streamlit run app.py
```

### 6. Access the Dashboard
Streamlit will automatically open your default browser. If not, navigate to:
```
http://localhost:8501
```

---

## Deployment to Streamlit Community Cloud (Hackathon Demonstration)

DustSentinel is optimized for zero-cost, serverless deployment on **Streamlit Community Cloud** (`https://<app-name>.streamlit.app`).

### Step-by-Step Cloud Deployment

1. **Push Repository to GitHub**:
   Ensure all local changes are pushed to your GitHub repository:
   - Remote: `https://github.com/MuskanS-2007/dustsense-ai.git`
   - Branch: `main`

2. **Sign In to Streamlit Community Cloud**:
   - Visit [share.streamlit.io](https://share.streamlit.io).
   - Sign in using your GitHub account that has access to the repository.

3. **Deploy the App**:
   - Click **"Create app"** (or **"New app"**).
   - Configure the following parameters:
     - **Repository**: `MuskanS-2007/dustsense-ai`
     - **Branch**: `main`
     - **Main file path**: `app.py`
     - **App URL**: Optionally customize the subdomain prefix (e.g. `https://dustsentinel.streamlit.app`).
   - (Optional) Under **Advanced settings**, confirm Python version is set to 3.10, 3.11, or 3.12.
   - Click **"Deploy!"**.

4. **Test the Live Cloud URL**:
   Streamlit Community Cloud will automatically install dependencies from `requirements.txt`, load theme and configuration from `.streamlit/config.toml`, and serve the application with managed HTTPS.
   - Verify live operations dashboard, 2D plume vectors, and all five KPI cards.
   - Test synthetic stream regeneration across the three scenario presets.
   - Test CSV validation with the built-in sample or custom uploads.
   - Verify that the custom favicon appears in the browser tab.
   - Switch to **Privacy Policy** and **Terms & Conditions** via the sidebar navigation.

---

## Repository File Structure

```text
dustsense-ai/
│
├── PRD.md                      # Official Product Requirements Document (Source of Truth)
├── app.py                      # Streamlit dashboard, navigation, and UI presentation layer
├── core_engine.py              # Pure, testable domain logic (validation, attribution, simulation)
├── requirements.txt            # Reproducible pinned dependencies (including Pillow)
├── sample_sensor_data.csv      # Baseline 24-hour realistic multi-zone sensor readings
├── generate_data.py            # Standalone generator utility for synthetic datasets
├── README.md                   # Comprehensive project documentation & guide
├── .gitignore                  # Git ignore rules for virtualenvs, caches, and secrets
├── favicon.ico                 # Root favicon asset
├── favicon.png                 # Root favicon asset (64x64)
│
├── .streamlit/
│   └── config.toml             # Streamlit Cloud configuration (theme, upload limits, privacy)
│
├── assets/
│   ├── favicon.png             # Original particulate sensor icon (PNG format)
│   └── favicon.ico             # Multi-resolution favicon icon
│
├── docs/
│   ├── PRD_GAP_ANALYSIS.md     # Post-remediation audit and verification report
│   └── HOSTING_OPTIONS.md      # Cloud hosting options and Streamlit Cloud recommendation
│
└── tests/
    ├── test_engine.py          # 15 automated pytest unit tests covering core math & edge cases
    └── verify_e2e.py           # End-to-end integration verification script
```

---

## CSV File Schema Format

When uploading custom site sensor data, ensure your CSV conforms to the following schema:

| Column | Type | Unit / Format | Description |
| :--- | :--- | :--- | :--- |
| `timestamp` | string / datetime | `YYYY-MM-DD HH:MM:SS` | Timestamp of sensor reading |
| `zone` | string | Categorical | One of: `Excavation`, `Haul Road`, `Material Storage`, `Loading Point` |
| `pm2_5` | float | $\mu\text{g/m}^3$ | Fine particulate matter concentration |
| `pm10` | float | $\mu\text{g/m}^3$ | Coarse + fine particulate matter concentration |
| `wind_speed` | float | $\text{km/h}$ | Ambient wind speed |
| `wind_deg` | float | Degrees ($0\text{–}360^\circ$) | Direction wind is blowing *from* |
| `humidity` | float | $\%$ ($0\text{–}100$) | Ambient relative humidity |
| `temperature`| float | $^\circ\text{C}$ | Ambient dry-bulb temperature |
| `activity_level` | string (optional) | Categorical | `High`, `Moderate`, `Low`, or `None` |

---

## Integrity, Scientific Disclaimers & Known Limitations

1. **Synthetic Data**: The baseline data is synthetically generated using diurnal meteorological cycles and empirical particulate emission factors. It is intended for software evaluation and decision-support demonstrations.
2. **Attribution Mechanism**: The source attribution engine is based on deterministic, explainable multi-criteria engineering rules. It does not claim an unverified black-box ML model.
3. **Simulation vs Measurement**: The before-and-after intervention simulator calculates theoretical abatement based on published empirical control efficiencies (US EPA AP-42 & WRAP Fugitive Dust Handbook). It does not represent site-measured physical reductions.
4. **Hardware Connectivity**: This prototype does not interface with live physical IoT hardware (e.g., optical particle counters or ultrasonic anemometers). Future roadmaps include integrating MQTT/LoRaWAN gateways.

---

*Developed for the Tech For Earth 2026 • Dust Mitigation Track*
