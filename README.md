# 🏠 London Property Intelligence

A machine learning web app that predicts London property prices based on location, size, and features.

**[Live Demo](https://london-property-intelligence-oxsj2ewudzwnfow9suf7qs.streamlit.app/)**

## Features

- **Price Prediction** — Enter property details to get an estimated value
- **Confidence Scoring** — See how reliable each prediction is
- **Area Comparison** — Compare predictions against local median prices
- **Interactive Heatmap** — Visualize price distribution across London

## Tech Stack

- **ML Model**: Random Forest (scikit-learn)
- **Frontend**: Streamlit
- **Data**: 30,000+ London property records
- **Visualization**: Folium heatmaps

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dataset

Property data sourced from [Kaggle London Property Dataset](https://www.kaggle.com/), including features like bedrooms, bathrooms, floor area, tenure, energy rating, and postcode.
