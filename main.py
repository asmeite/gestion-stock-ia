import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import joblib
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configuration de la page
st.set_page_config(page_title="Prévision IA & Actions", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("data/simulated_sales_data_en.csv", parse_dates=["Date"])

# Chargement données et modèle
df = load_data()
model = joblib.load("models/model_global.pkl")
encoder = joblib.load("models/product_encoder.pkl")

st.sidebar.title("🛠 Configuration")
selected_product = st.sidebar.selectbox("Produit", df["Product"].unique())
page = st.sidebar.radio("🔖 Pages", [
    "Comparaison IA vs Classique",
    "Analyse du modèle",
    "Conséquences & Recommandations"
])

hist = df[df["Product"] == selected_product].sort_values("Date")
code = encoder.transform([selected_product])[0]
last_date = hist["Date"].max()
last_rows = hist.tail(3)

future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=7)
future = pd.DataFrame({
    "Date": future_dates,
    "Product_Code": code,
    "Day_of_Week": [(d.weekday() + 1) for d in future_dates],
    "External_Temperature": hist["External_Temperature"].mean(),
    "Promotion_Active": 0,
    "Sales_Lag_1": last_rows["Daily_Sales"].iloc[-1],
    "Sales_Lag_2": last_rows["Daily_Sales"].iloc[-2],
    "Sales_Lag_3": last_rows["Daily_Sales"].iloc[-3],
})
X_f = future[[
    "Product_Code", "Day_of_Week", "External_Temperature",
    "Promotion_Active", "Sales_Lag_1", "Sales_Lag_2", "Sales_Lag_3"
]]
future["IA"] = model.predict(X_f)
ma = hist["Daily_Sales"].rolling(window=7).mean().iloc[-1]
future["Classique"] = ma

current_stock = hist["Current_Stock"].iloc[-1]
dmd_7j = future["IA"].sum()
to_order = max(0, int(dmd_7j - current_stock))

st.title("📦 Prévision de la demande et recommandations")

if page == "Comparaison IA vs Classique":
    st.header("🔍 1) Comparaison IA vs Moyenne Mobile")

    st.subheader("⚙️ Scénario What-If")
    p_promo = st.selectbox("Promo active ?", [0, 1])
    p_temp = st.slider(
        "Température (°C)",
        int(df["External_Temperature"].min()),
        int(df["External_Temperature"].max()),
        int(df["External_Temperature"].mean()),
        step=1
    )
    p_day = st.slider("Jour de semaine", 1, 7, 1, step=1)

    X_sc = pd.DataFrame([{
        "Product_Code": code,
        "Day_of_Week": p_day,
        "External_Temperature": p_temp,
        "Promotion_Active": p_promo,
        "Sales_Lag_1": last_rows["Daily_Sales"].iloc[-1],
        "Sales_Lag_2": last_rows["Daily_Sales"].iloc[-2],
        "Sales_Lag_3": last_rows["Daily_Sales"].iloc[-3],
    }])
    ia_sc = model.predict(X_sc)[0]

    st.markdown(f"- **Prévision IA** (scénario) : **{ia_sc:.0f}** unités")
    st.markdown(f"- **Moyenne mobile**       : **{ma:.0f}** unités")

    st.subheader("📋 Prévisions comparées (7 jours)")
    st.dataframe(future[["Date", "Classique", "IA"]])

    fig = px.line(
        future, x="Date",
        y=["Classique", "IA"],
        title=f"Comparaison prévisions pour {selected_product}",
        labels={"value": "Ventes", "variable": "Méthode"},
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)

elif page == "Analyse du modèle":
    st.header("📊 2) Analyse du modèle IA")

    rmse = 9.34
    r2   = 0.87
    st.markdown("**Performances du modèle global :**")
    st.write(f"- RMSE : **{rmse:.2f}**")
    st.write(f"- R²   : **{r2:.2f}**")
    
    st.subheader("📈 Prédiction IA vs Réel")
    st.image("plots/prediction_vs_reel.png",)

    st.subheader("📊 Réel vs Prédit")
    st.image("plots/reel_vs_predit.png")

elif page == "Conséquences & Recommandations":
    st.header("⚙️ 3) Conséquences et recommandations")

    st.markdown(f"- Stock actuel : **{current_stock}** unités")
    st.markdown(f"- Demande IA 7 jours : **{dmd_7j:.0f}** unités")

    avg_daily = dmd_7j / 7
    cover_days = current_stock / avg_daily if avg_daily>0 else np.nan
    st.metric("📅 Couverture (jours)", f"{cover_days:.1f}")

    stock = current_stock
    dmd7  = dmd_7j

    coverage = stock / dmd7 if dmd7>0 else 0
    coverage_pct = coverage * 100
    risk_pct = max(0, 100 - coverage_pct)

    st.metric("📊 Taux de couverture (%)", f"{coverage_pct:.0f}%")
    st.metric("⚠️ Risque de rupture (%)", f"{risk_pct:.0f}%")
    
    if to_order > 0:
        st.error(f"⚠️ Commander **{to_order}** unités")
    else:
        st.success("✅ Stock suffisant")


