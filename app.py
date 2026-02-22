import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import folium
from streamlit_folium import st_folium
import warnings
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="London Property Intelligence",
    page_icon="🏠",
    layout="wide"
)

# ── Load & cache data ─────────────────────────────────────
@st.cache_data
def load_and_train():
    data = pd.read_parquet("kaggle_london_house_price_data.parquet")

    cols = ['bathrooms', 'bedrooms', 'floorAreaSqM', 'livingRooms',
            'tenure', 'propertyType', 'currentEnergyRating', 'outcode',
            'latitude', 'longitude', 'saleEstimate_currentPrice',
            'saleEstimate_confidenceLevel']

    df = data[cols].copy()
    df = df.dropna(subset=['saleEstimate_currentPrice'])
    df = df.dropna(subset=['tenure', 'propertyType', 'currentEnergyRating', 'outcode'])

    num_cols = ['bathrooms', 'bedrooms', 'floorAreaSqM', 'livingRooms']
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df = df[df['saleEstimate_currentPrice'] < 10_000_000]

    df_model = df.copy()

    le_dict = {}
    cat_cols = ['tenure', 'propertyType', 'currentEnergyRating', 'outcode']
    for col in cat_cols:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col].astype(str))
        le_dict[col] = le

    features = ['bathrooms', 'bedrooms', 'floorAreaSqM', 'livingRooms',
                'tenure', 'propertyType', 'currentEnergyRating', 'outcode']

    X = df_model[features]
    y = df_model['saleEstimate_currentPrice']

    # Price model
    price_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    price_model.fit(X, y)

    # Confidence classifier
    df_conf = df[df['saleEstimate_confidenceLevel'].isin(['HIGH', 'MEDIUM', 'LOW'])].copy()
    for col in cat_cols:
        df_conf[col] = LabelEncoder().fit_transform(df_conf[col].astype(str))
    y_conf = LabelEncoder().fit_transform(df_conf['saleEstimate_confidenceLevel'])
    conf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    conf_model.fit(df_conf[features], y_conf)

    return df, price_model, conf_model, le_dict, features

df, price_model, conf_model, le_dict, features = load_and_train()

# ── Header ────────────────────────────────────────────────
st.title("🏠 London Property Intelligence")
st.markdown("*Predict property values and risk profiles across London — powered by 418,000+ property records*")
st.divider()

# ── Layout: two columns ───────────────────────────────────
col1, col2 = st.columns([1, 1.6])

with col1:
    st.subheader("Property Details")

    outcode = st.selectbox(
        "Location (Outcode)",
        sorted(df['outcode'].dropna().unique()),
        index=list(sorted(df['outcode'].dropna().unique())).index('SW1P') if 'SW1P' in df['outcode'].values else 0
    )

    property_type = st.selectbox(
        "Property Type",
        sorted(df['propertyType'].dropna().unique())
    )

    tenure = st.selectbox(
        "Tenure",
        sorted(df['tenure'].dropna().unique())
    )

    energy_rating = st.selectbox(
        "Energy Rating",
        ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
        index=2
    )

    col_a, col_b = st.columns(2)
    with col_a:
        bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=2)
        bathrooms = st.number_input("Bathrooms", min_value=0, max_value=10, value=1)
    with col_b:
        living_rooms = st.number_input("Living Rooms", min_value=0, max_value=10, value=1)
        floor_area = st.number_input("Floor Area (sqm)", min_value=10, max_value=1000, value=75)

    predict_btn = st.button("🔍 Predict Property Value", use_container_width=True)

with col2:
    if predict_btn:
        # Encode inputs
        def safe_encode(le, val):
            if val in le.classes_:
                return le.transform([val])[0]
            return 0

        input_data = pd.DataFrame([{
            'bathrooms': bathrooms,
            'bedrooms': bedrooms,
            'floorAreaSqM': floor_area,
            'livingRooms': living_rooms,
            'tenure': safe_encode(le_dict['tenure'], tenure),
            'propertyType': safe_encode(le_dict['propertyType'], property_type),
            'currentEnergyRating': safe_encode(le_dict['currentEnergyRating'], energy_rating),
            'outcode': safe_encode(le_dict['outcode'], outcode),
        }])

        predicted_price = price_model.predict(input_data)[0]
        confidence_raw = conf_model.predict(input_data)[0]
        confidence_map = {0: 'HIGH', 1: 'LOW', 2: 'MEDIUM'}
        confidence = confidence_map.get(confidence_raw, 'MEDIUM')
        confidence_color = {'HIGH': '🟢', 'MEDIUM': '🟡', 'LOW': '🔴'}

        # Results
        st.subheader("Prediction Results")

        m1, m2, m3 = st.columns(3)
        m1.metric("Estimated Value", f"£{predicted_price:,.0f}")
        m2.metric("Confidence", f"{confidence_color[confidence]} {confidence}")
        m3.metric("Location", outcode)

        st.divider()

        # Area context
        area_data = df[df['outcode'] == outcode]['saleEstimate_currentPrice']
        if len(area_data) > 0:
            area_median = area_data.median()
            diff = predicted_price - area_median
            diff_pct = (diff / area_median) * 100

            st.markdown(f"**Area Context — {outcode}**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Area Median", f"£{area_median:,.0f}")
            c2.metric("vs Area Median", f"{'▲' if diff > 0 else '▼'} £{abs(diff):,.0f}", f"{diff_pct:+.1f}%")
            c3.metric("Properties in Area", f"{len(area_data):,}")

        st.divider()

        # Energy rating insight
        energy_prices = df.groupby('currentEnergyRating')['saleEstimate_currentPrice'].median()
        st.markdown("**Energy Rating Market Context**")
        if energy_rating in energy_prices.index:
            rating_median = energy_prices[energy_rating]
            st.markdown(f"Median price for **{energy_rating}-rated** properties in London: **£{rating_median:,.0f}**")

    else:
        st.info("👈 Fill in the property details and click **Predict Property Value** to get started.")

        # Show market overview while waiting
        st.subheader("London Market Overview")
        col_x, col_y = st.columns(2)
        col_x.metric("Total Properties", f"{len(df):,}")
        col_x.metric("Median Price", f"£{df['saleEstimate_currentPrice'].median():,.0f}")
        col_y.metric("Most Common Type", df['propertyType'].mode()[0])
        col_y.metric("Price Range", f"£89k — £10M")

st.divider()

# ── Heatmap ───────────────────────────────────────────────
st.subheader("🗺️ London Property Price Heatmap")

try:
    from folium.plugins import HeatMap
    sample = df.dropna(subset=['latitude', 'longitude']).sample(5000, random_state=42)
    m = folium.Map(location=[51.5074, -0.1278], zoom_start=10)
    heat_data = [[r['latitude'], r['longitude'], r['saleEstimate_currentPrice']]
                 for _, r in sample.iterrows()]
    HeatMap(heat_data, radius=10, blur=15, max_zoom=13).add_to(m)
    st_folium(m, width=1200, height=450)
except Exception as e:
    st.warning(f"Map unavailable: {e}")

# ── Footer ────────────────────────────────────────────────
st.divider()
st.caption("Built using 418,201 London property records | Random Forest ML | Data: Kaggle London Property Dataset")