import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date
from sqlalchemy import create_engine
import os
import streamlit.components.v1 as components

# CHARGER MODELE ET ENCODEUR
model = joblib.load("models/xgb_model.pkl")
encoder = joblib.load("models/ordinal_encoder.pkl")

# VALEURS FIXES
categories = ['Beverages', 'Bakery', 'Produce', 'Dairy', 'Snacks']
subcategories = {
    'Beverages': ['Soda', 'Juice', 'Water'],
    'Bakery': ['Bread', 'Croissant', 'Cake'],
    'Produce': ['Apple', 'Banana', 'Carrot'],
    'Dairy': ['Milk', 'Cheese', 'Yogurt'],
    'Snacks': ['Chips', 'Chocolate', 'Cookies']
}
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
seasons = ['Winter', 'Spring', 'Summer', 'Autumn']

# TITRE & MODE
st.title("📦 Prédiction de la Demande avec XGBoost")
mode = st.radio("Choisissez un mode de prédiction :", ("👭 Produit spécifique (manuel)", "📄 Tous les produits (automatique)"))

# MODE MANUEL
if mode == "👭 Produit spécifique (manuel)":
    selected_date = st.date_input("Date", value=date.today())
    product_category = st.selectbox("Catégorie du produit", categories)
    product_subcategory = st.selectbox("Sous-catégorie", subcategories[product_category])
    weekday = st.selectbox("Jour de la semaine", weekdays)
    season = st.selectbox("Saison", seasons)

    unit_price = st.number_input("Prix unitaire (€)", min_value=0.1, value=5.0)
    available_stock = st.number_input("Stock disponible", min_value=0, value=100)
    supplier_lead_time = st.slider("Délai fournisseur (jours)", 1, 30, 7)
    on_promotion = st.checkbox("Produit en promotion")
    is_holiday = st.checkbox("Jour férié")
    temperature = st.slider("Température (°C)", -10.0, 40.0, 20.0)
    rainfall = st.slider("Pluviométrie (mm)", 0.0, 50.0, 5.0)

    lag_1 = st.number_input("Units sold 1 jour avant", min_value=0, value=20)
    lag_2 = st.number_input("Units sold 2 jours avant", min_value=0, value=20)
    lag_3 = st.number_input("Units sold 3 jours avant", min_value=0, value=20)

    rolling_mean = np.mean([lag_1, lag_2, lag_3])
    rolling_std = np.std([lag_1, lag_2, lag_3])
    rolling_min = np.min([lag_1, lag_2, lag_3])
    rolling_max = np.max([lag_1, lag_2, lag_3])
    trend = lag_1 - rolling_mean

    month = selected_date.month
    year = selected_date.year
    day = selected_date.day
    week = selected_date.isocalendar()[1]
    is_weekend = 1 if weekday in ['Saturday', 'Sunday'] else 0
    price_per_stock = unit_price / (available_stock + 1)
    promo_and_holiday = int(on_promotion and is_holiday)

    input_df = pd.DataFrame([{
        "product_category": product_category,
        "product_subcategory": product_subcategory,
        "weekday": weekday,
        "season": season,
        "unit_price": unit_price,
        "available_stock": available_stock,
        "supplier_lead_time_days": supplier_lead_time,
        "on_promotion": int(on_promotion),
        "is_holiday": int(is_holiday),
        "temperature_c": temperature,
        "rainfall_mm": rainfall,
        "month": month,
        "year": year,
        "day": day,
        "week": week,
        "is_weekend": is_weekend,
        "price_per_stock": price_per_stock,
        "promo_and_holiday": promo_and_holiday,
        "units_sold_lag_1": lag_1,
        "units_sold_lag_2": lag_2,
        "units_sold_lag_3": lag_3,
        "units_sold_avg_3": rolling_mean,
        "units_sold_rolling_mean_3": rolling_mean,
        "units_sold_rolling_std_3": rolling_std,
        "units_sold_rolling_min_3": rolling_min,
        "units_sold_rolling_max_3": rolling_max,
        "units_sold_trend_3": trend
    }])

    if st.button("📈 Prédire la demande"):
        cat_input = encoder.transform(input_df[["product_category", "product_subcategory", "weekday", "season"]])
        num_input = input_df.drop(columns=["product_category", "product_subcategory", "weekday", "season"]).values
        final_input = np.hstack([cat_input, num_input])
        prediction = model.predict(final_input)[0]
        st.success(f"📊 Prévision : {prediction:.2f} unités vendues")

elif mode == "📄 Tous les produits (automatique)":
    st.info("Chargement des dernières lignes connues depuis Azure (PostgreSQL)...")

    if st.button("⚡ Générer les prévisions"):
        db_user = st.secrets["postgres"]["user"]
        db_password = st.secrets["postgres"]["password"]
        db_host = st.secrets["postgres"]["host"]
        db_port = st.secrets["postgres"]["port"]
        db_name = st.secrets["postgres"]["dbname"]
        sslmode = st.secrets["postgres"].get("sslmode", "require") 

        engine = create_engine(
            f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?sslmode={sslmode}"
        )
        query = "SELECT * FROM simulated_sales"
        df = pd.read_sql(query, engine, parse_dates=["date"])

        df["month"] = df["date"].dt.month
        df["year"] = df["date"].dt.year
        df["day"] = df["date"].dt.day
        df["week"] = df["date"].dt.isocalendar().week
        df["is_weekend"] = df["weekday"].isin(["Saturday", "Sunday"]).astype(int)
        df["price_per_stock"] = df["unit_price"] / (df["available_stock"] + 1)
        df["promo_and_holiday"] = df["on_promotion"] & df["is_holiday"]

        df = df.sort_values(by=["product_category", "product_subcategory", "date"])
        df["units_sold_lag_1"] = df.groupby(["product_category", "product_subcategory"])['units_sold'].shift(1)
        df["units_sold_lag_2"] = df.groupby(["product_category", "product_subcategory"])['units_sold'].shift(2)
        df["units_sold_lag_3"] = df.groupby(["product_category", "product_subcategory"])['units_sold'].shift(3)
        df["units_sold_avg_3"] = df[["units_sold_lag_1", "units_sold_lag_2", "units_sold_lag_3"]].mean(axis=1)
        df["units_sold_rolling_mean_3"] = df.groupby(["product_category", "product_subcategory"])["units_sold"].transform(lambda x: x.rolling(window=3).mean())
        df["units_sold_rolling_std_3"] = df.groupby(["product_category", "product_subcategory"])["units_sold"].transform(lambda x: x.rolling(window=3).std())
        df["units_sold_rolling_min_3"] = df.groupby(["product_category", "product_subcategory"])["units_sold"].transform(lambda x: x.rolling(window=3).min())
        df["units_sold_rolling_max_3"] = df.groupby(["product_category", "product_subcategory"])["units_sold"].transform(lambda x: x.rolling(window=3).max())
        df["units_sold_trend_3"] = df["units_sold_lag_1"] - df["units_sold_rolling_mean_3"]

        df_latest = df.sort_values("date").groupby(["product_category", "product_subcategory"]).tail(1)
        df_latest = df_latest.dropna()

        categorical_cols = ["product_category", "product_subcategory", "weekday", "season"]
        numerical_cols = [
            "unit_price", "available_stock", "supplier_lead_time_days",
            "on_promotion", "is_holiday", "temperature_c", "rainfall_mm",
            "month", "year", "day", "week", "is_weekend", "price_per_stock", "promo_and_holiday",
            "units_sold_lag_1", "units_sold_lag_2", "units_sold_lag_3", "units_sold_avg_3",
            "units_sold_rolling_mean_3", "units_sold_rolling_std_3", "units_sold_rolling_min_3",
            "units_sold_rolling_max_3", "units_sold_trend_3"
        ]

        X_cat = encoder.transform(df_latest[categorical_cols])
        X_num = df_latest[numerical_cols].values
        final_input = np.hstack([X_cat, X_num])
        preds = model.predict(final_input)
        df_latest["units_sold_predicted"] = preds

        st.subheader("📋 Récapitulatif des données à prédire")
        st.write(f"Nombre total de produits : {df_latest.shape[0]}")
        st.write(f"Période des données : {df_latest['date'].min().date()} ➡️ {df_latest['date'].max().date()}")
        st.write("Produits par catégorie :")
        st.dataframe(df_latest["product_category"].value_counts().reset_index().rename(columns={"index": "Catégorie", "product_category": "Nombre de produits"}))
        st.success(f"✅ {len(preds)} prédictions générées sur les lignes les plus récentes de chaque produit !")

        to_insert = df_latest[
            [
                "date", "product_subcategory", "product_category", "weekday", "season",
                "unit_price", "available_stock", "supplier_lead_time_days",
                "on_promotion", "is_holiday", "temperature_c", "rainfall_mm",
                "price_per_stock", "promo_and_holiday",
                "units_sold_lag_1", "units_sold_lag_2", "units_sold_lag_3", "units_sold_avg_3",
                "units_sold_rolling_mean_3", "units_sold_rolling_std_3",
                "units_sold_rolling_min_3", "units_sold_rolling_max_3", "units_sold_trend_3",
                "units_sold_predicted"
            ]
        ].copy()
        to_insert = to_insert.rename(columns={
            "product_subcategory": "product",
            "units_sold_predicted": "predicted_units_sold"
        })
        to_insert["prediction_date"] = to_insert["date"] + pd.Timedelta(days=1)
        to_insert["prediction_method"] = "XGBoost"

        engine = create_engine(
            f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?sslmode={sslmode}"
        )
        try:
            to_insert.to_sql("forecasts", engine, if_exists="append", index=False)
        except Exception as e:
            st.error(f"Erreur lors de l'insertion : {e}")

    st.header("📊 Suivi des stocks & Recommandations via Power BI")
    powerbi_url = "https://app.powerbi.com/links/2j-oNtN4Zx?ctid=108bc864-cdf5-4ec3-8b7c-4eb06be1b41d&pbi_source=linkShare"
    st.link_button("Accéder au rapport Power BI", powerbi_url)
