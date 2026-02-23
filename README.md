# 🏠 London Property Intelligence

A machine learning-powered property valuation and risk profiling tool 
built on 418,000+ London residential property records.

Built as a portfolio project to demonstrate applied data science skills 
relevant to property intelligence, valuation modelling, and risk assessment.

🔗 **[Live Dashboard →]# 🏠 London Property Intelligence

A machine learning-powered property valuation and risk profiling tool 
built on 418,000+ London residential property records.

Built as a portfolio project to demonstrate applied data science skills 
relevant to property intelligence, valuation modelling, and risk assessment.

🔗 **[Live Dashboard → (https://london-property-intelligence-oxsj2ewudzwnfow9suf7qs.streamlit.app/)**

---

## Project Overview

This project mirrors the core work of a property intelligence platform —
combining open property data, machine learning, and geospatial analysis to 
generate insights for valuation, risk, and market analysis.

It was built using a real London property dataset with features including 
floor area, property type, tenure, energy rating, and geographic location.

---

## What It Does

**1. Property Price Prediction**
Predicts estimated sale value using a Random Forest Regressor trained on 
325,758 cleaned property records.
- Mean Absolute Error: £57,560
- R² Score: 0.9597

**2. Valuation Confidence Classifier**
Predicts whether a property valuation is HIGH, MEDIUM, or LOW confidence —
directly applicable to mortgage lending and insurance risk decisions.
- Overall accuracy: 90%
- HIGH confidence precision: 93%

**3. Geospatial Heatmap**
Interactive London property price heatmap built with Folium, showing 
price concentration across postcodes and boroughs.

**4. Interactive Dashboard**
Streamlit web app allowing users to input property details and receive 
instant price predictions and confidence scores.

---

## Key Findings

- **Location dominates:** Outcode is the strongest price predictor, 
  with Central and West London (SW1, W1, WC) commanding the highest values
- **Size matters:** Floor area and bedroom count are the top property-level 
  features after location
- **Energy efficiency premium:** A/B rated properties command a clear price 
  premium over D/E rated equivalents — consistent with growing buyer focus 
  on running costs
- **Risk profiling:** The classifier identifies LOW confidence valuations 
  with 83% precision, useful for flagging properties requiring manual review

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Pandas & NumPy | Data cleaning and manipulation |
| Scikit-learn | Random Forest models |
| Matplotlib & Seaborn | EDA visualisations |
| Folium | Geospatial heatmap |
| Streamlit | Interactive dashboard |

---

## Dataset

**London Property Dataset** — 418,201 residential properties with sale 
estimates, rental estimates, energy ratings, floor area, and geographic 
coordinates. Sourced from Kaggle.

Cleaned to 325,758 records after removing missing values and outliers 
above £10 million.

---

## Project Structure
```
london-property-intelligence/
│
├── London_Dataset.ipynb        # Full analysis notebook
├── app.py                      # Streamlit dashboard
├── price_distribution.png      # EDA charts
├── price_by_type.png
├── price_by_energy.png
├── top_areas.png
├── feature_importance.png
├── actual_vs_predicted.png
└── london_property_heatmap.html
```

---

## Next Steps

- Incorporate transaction history trends for time-series analysis
- Add postcode-level deprivation index as an additional feature
- Test XGBoost and gradient boosting for accuracy comparison
- Expand to cover UK-wide data beyond London

---

## Author

Built by Vijay — CS student at Swansea University  
[GitHub](https://github.com/Rai59))**

---

## Project Overview

This project mirrors the core work of a property intelligence platform —
combining open property data, machine learning, and geospatial analysis to 
generate insights for valuation, risk, and market analysis.

It was built using a real London property dataset with features including 
floor area, property type, tenure, energy rating, and geographic location.

---

## What It Does

**1. Property Price Prediction**
Predicts estimated sale value using a Random Forest Regressor trained on 
325,758 cleaned property records.
- Mean Absolute Error: £57,560
- R² Score: 0.9597

**2. Valuation Confidence Classifier**
Predicts whether a property valuation is HIGH, MEDIUM, or LOW confidence —
directly applicable to mortgage lending and insurance risk decisions.
- Overall accuracy: 90%
- HIGH confidence precision: 93%

**3. Geospatial Heatmap**
Interactive London property price heatmap built with Folium, showing 
price concentration across postcodes and boroughs.

**4. Interactive Dashboard**
Streamlit web app allowing users to input property details and receive 
instant price predictions and confidence scores.

---

## Key Findings

- **Location dominates:** Outcode is the strongest price predictor, 
  with Central and West London (SW1, W1, WC) commanding the highest values
- **Size matters:** Floor area and bedroom count are the top property-level 
  features after location
- **Energy efficiency premium:** A/B rated properties command a clear price 
  premium over D/E rated equivalents — consistent with growing buyer focus 
  on running costs
- **Risk profiling:** The classifier identifies LOW confidence valuations 
  with 83% precision, useful for flagging properties requiring manual review

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Pandas & NumPy | Data cleaning and manipulation |
| Scikit-learn | Random Forest models |
| Matplotlib & Seaborn | EDA visualisations |
| Folium | Geospatial heatmap |
| Streamlit | Interactive dashboard |

---

## Dataset

**London Property Dataset** — 418,201 residential properties with sale 
estimates, rental estimates, energy ratings, floor area, and geographic 
coordinates. Sourced from Kaggle.

Cleaned to 325,758 records after removing missing values and outliers 
above £10 million.

---

## Project Structure
```
london-property-intelligence/
│
├── London_Dataset.ipynb        # Full analysis notebook
├── app.py                      # Streamlit dashboard
├── price_distribution.png      # EDA charts
├── price_by_type.png
├── price_by_energy.png
├── top_areas.png
├── feature_importance.png
├── actual_vs_predicted.png
└── london_property_heatmap.html
```

---

## Next Steps

- Incorporate transaction history trends for time-series analysis
- Add postcode-level deprivation index as an additional feature
- Test XGBoost and gradient boosting for accuracy comparison
- Expand to cover UK-wide data beyond London

---

## Author

Built by Vijay — CS student at Swansea University  
[GitHub](https://github.com/Rai59)
