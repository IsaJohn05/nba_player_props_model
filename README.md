# 🏀 NBA Player Player Stats Model

An end-to-end machine learning pipeline for predicting **NBA player stats props** using:
- A **minutes prediction model (XGBoost)**
- A **rate-based scoring model**
- **Vegas odds comparison** to identify betting edges
- Automated **daily inference with styled Excel output**

---
DISCLAIMER:
This project is for educational and analytical purposes only.
No betting advice is provided.

## 🔍 Model Overview

This system estimates player statistical outcomes using the following structure:

For example: **Expected Points**

Expected Points = Predicted Minutes x Points Per Minute


### Why this works
- Minutes determine **opportunity**
- Points per minute captures **efficiency**
- Separating the two avoids overfitting and reacts quickly to role changes

---

## 🧠 Modeling Approach

### 1️⃣ Minutes Model (XGBoost)
Predicts expected minutes played using:
- Recent minutes trends
- Starter/bench role
- Rest days / back-to-backs
- Home vs away
- Recent usage indicators

This is the **highest-leverage model** in the system.

---

### 2️⃣ Scoring Rate (Statistical)
Points per minute is computed from rolling historical performance:

pts_per_min_last_10 = total_points_last10/total_minutes_last10
This is the most basic way to calculate this


This provides a stable efficiency baseline.

---

### 3️⃣ Probability & Edge Calculation
- Convert expected points into a probability distribution
- Compute:
P(Over) = 1 - beta((line- mean) / alpha)

- Compare model probability vs sportsbook implied probability
- Rank plays using:
AI Rating = Edge x 100


---

## 📊 Output

The pipeline produces a **styled Excel sheet** featuring:
- Top **11 player props**
- Max **5 unders** (no minimum)
- Max **1 pick per player**
- Separate **OVERS / UNDERS** sections
- FanDuel & Bet365 pricing
- Daily timestamped header

📁 Output file:
data/processed/today_processed_prop_predictions.xlsx


---

## ⚙️ Daily Usage

### Run the full pipeline (one command):
```bash
python run_today_pipeline.py
'''

This performs:
Roster updates (current teams)
Odds fetching (Odds API)
Odds normalization
Model inference + Excel output

Tech Stack
Python
XGBoost
pandas / numpy
nba_api
The Odds API
openpyxl (Excel formatting)

Future Improvements
Minutes volatility modeling (σ_minutes)
Efficiency adjustment ML model (rate deltas)
Injury-aware rotation modeling
ROI tracking & calibration
Automated result labeling
