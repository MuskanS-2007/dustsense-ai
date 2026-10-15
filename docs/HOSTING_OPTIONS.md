# DustSentinel: Hosting Architecture & Deployment Strategy

## Executive Summary & Hackathon Recommendation

- **Primary Recommended Platform**: **Streamlit Community Cloud** (`https://<app-name>.streamlit.app`)
- **Target Event**: Tech For Earth Hackathon 2026 — Innovative Dust Mitigation Solutions
- **Status**: Ready for immediate deployment upon GitHub repository synchronization
- **Cost**: **$0.00** (Free tier; no custom domain purchase or VPS provisioning required)

For the Tech For Earth Hackathon demonstration, **Streamlit Community Cloud is the optimal deployment target**. It eliminates infrastructure management, requires no external DNS or server provisioning, natively supports persistent WebSockets, auto-provisions TLS/HTTPS certificates, and deploys directly from the team's GitHub repository with continuous integration.

---

## 1. Primary Path: Streamlit Community Cloud Deployment Guide

### Why Streamlit Community Cloud Fits DustSentinel
1. **Zero Infrastructure Overhead**: No Linux server provisioning, SSH key management, firewall tuning, or OS security patching.
2. **Native WebSocket Handling**: Continuous bi-directional WebSockets (`/_stcore/stream`) work out of the box with zero proxy timeout configuration.
3. **Automatic TLS / HTTPS**: Every app receives an automatic, valid SSL certificate under the `*.streamlit.app` domain.
4. **Automated Continuous Deployment**: Any update pushed to the `main` branch of `dustsense-ai` triggers an automatic rebuild and hot reload.
5. **Memory Profile Compatibility**: Community Cloud provides 1 GB of container RAM. DustSentinel's in-memory footprint is ~180 MB idle and ~250–320 MB under active multi-zone scenario simulation, well within resource bounds.
6. **No File System Persistence Required**: Because DustSentinel operates as an in-memory decision-support prototype with volatile session state (`st.session_state`), it runs seamlessly on Community Cloud containers.

### Step-by-Step Deployment Procedure

```
[Local Workspace] ---- git push ----> [GitHub: MuskanS-2007/dustsense-ai]
                                              |
                                              | Webhook Trigger
                                              v
                              [Streamlit Community Cloud]
                                              |
                                              | Builds from requirements.txt
                                              v
                              [https://<app-name>.streamlit.app]
```

1. **Push Repository to GitHub**:
   Ensure all local files (`app.py`, `core_engine.py`, `requirements.txt`, `sample_sensor_data.csv`, `assets/`, `.streamlit/config.toml`, `README.md`) are pushed to the team repository:
   - Repository: `MuskanS-2007/dustsense-ai` (or team fork)
   - Branch: `main`

2. **Authenticate with Streamlit Community Cloud**:
   - Navigate to [share.streamlit.io](https://share.streamlit.io).
   - Sign in using your GitHub account that has access to the repository.

3. **Configure New Application**:
   - Click the **"Create app"** (or **"New app"**) button.
   - Fill in the deployment parameters:
     - **Repository**: `MuskanS-2007/dustsense-ai`
     - **Branch**: `main`
     - **Main file path**: `app.py`
     - **App URL**: Customize the subdomain prefix (e.g., `dustsentinel` or `dustsentinel-ai` yielding `https://dustsentinel.streamlit.app`).

4. **Verify Advanced Settings (Optional)**:
   - Ensure the Python version is set to 3.10, 3.11, or 3.12 (supported by pinned dependencies in `requirements.txt`).
   - No environment variables or secrets are required for the standard demonstration.

5. **Deploy & Validate**:
   - Click **"Deploy!"**.
   - Streamlit will clone the repository, install pinned dependencies from `requirements.txt`, apply theme settings from `.streamlit/config.toml`, and launch `app.py`.
   - Access the live URL and test all six operational tabs, scenario toggles, CSV validation, and legal pages.

---

## 2. Platform Comparison Matrix

| Evaluation Criterion | Streamlit Community Cloud (Recommended) | Dedicated Linux VPS + NGINX (Future) | Managed Cloud Run / ECS (Enterprise) |
| :--- | :--- | :--- | :--- |
| **Suitability for Hackathon** | **Top Recommendation (Primary)** | Over-engineered for demo | High configuration overhead |
| **Domain & URL** | Subdomain (`*.streamlit.app`) | Custom domain required (`.com`, etc.) | Custom domain or cloud URL |
| **Infrastructure Cost** | **$0.00 / month** | $6 – $24 / month | $25 – $65 / month |
| **Setup Time** | **< 3 minutes** | 2 – 4 hours | 2 – 3 hours |
| **SSL / TLS Certificate** | Automatic & managed | Let's Encrypt Certbot setup | Cloud Load Balancer / ACM |
| **WebSocket Configuration** | Native, zero config | Manual NGINX proxy header tuning | Requires session affinity |
| **Git Integration** | Automatic rebuild on `git push` | Manual pull or CI/CD runner | Docker build & registry push |
| **Data Residency** | Shared cloud (US/EU) | Full private ownership | Dedicated VPC |

---

## 3. Future Enterprise Roadmap Options (Post-Hackathon)

If a commercial organization or municipal air quality district adopts DustSentinel for permanent field installations requiring proprietary hardware connectivity, corporate domain branding, or on-premises data residency, the following production architectures may be deployed:

### Option B: Dedicated Linux VPS with NGINX Reverse Proxy
- **When to Use**: When a custom enterprise domain (e.g., `dustsentinel.company.com`) or isolated single-tenant data residency is mandated.
- **Components**:
  - Ubuntu 22.04/24.04 LTS on AWS EC2, DigitalOcean, or Hetzner.
  - NGINX reverse proxy with WebSocket upgrade headers (`Upgrade $http_upgrade`, `Connection "upgrade"`), `proxy_read_timeout 86400s;`, and `client_max_body_size 20M;`.
  - Let's Encrypt TLS certificate via Certbot.
  - Systemd daemon service (`dustsentinel.service`) for process supervision.

### Option C: Containerized Cloud PaaS (Google Cloud Run / AWS ECS Fargate)
- **When to Use**: High-traffic multi-project deployments requiring container portability and automated auto-scaling.
- **Key Requirement**: Must configure cookie-based session affinity and minimum 1 warm instance to maintain persistent WebSocket streams.

### Option D: Industrial On-Premises Kubernetes (Air-Gapped Site LAN)
- **When to Use**: Isolated mining or heavy civil infrastructure parcels where sensor telemetry cannot traverse the public internet.

---

## 4. Pre-Deployment Verification Checklist for Streamlit Community Cloud

- [x] Application entry point confirmed: `app.py`
- [x] Domain engine isolated and verified: `core_engine.py`
- [x] Pinned dependencies complete in `requirements.txt` (including `Pillow` for favicon processing)
- [x] Baseline sample data present: `sample_sensor_data.csv`
- [x] Local original favicon assets present in `assets/` and root
- [x] Cloud-compatible configuration added: `.streamlit/config.toml` (dark theme, 15 MB limit, disabled telemetry, no localhost binding)
- [x] `.gitignore` verified to exclude secrets (`.streamlit/secrets.toml`), caches, and virtual environments while tracking `config.toml`
- [x] Automated test suite passing (`pytest -v` 15/15, `verify_e2e.py` passed)
- [x] Legal pages include transparent data handling disclosures with clearly marked placeholders
- [x] Zero API keys, passwords, or live hardware claims present
