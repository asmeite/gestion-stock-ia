import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
import os
import joblib

df = pd.read_csv("data.csv", parse_dates=["date"])

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



X = df.drop(columns=["units_sold"])
y = df["units_sold"]

categorical_cols = ["product_category", "product_subcategory", "weekday", "season"]
numerical_cols = [
    "unit_price", "available_stock", "supplier_lead_time_days",
    "on_promotion", "is_holiday", "temperature_c", "rainfall_mm",
    "month", "year", "day", "week", "is_weekend", "price_per_stock", "promo_and_holiday",
    "units_sold_lag_1", "units_sold_lag_2", "units_sold_lag_3",
    "units_sold_avg_3"
]


numerical_cols += [
    "units_sold_rolling_mean_3",
    "units_sold_rolling_std_3",
    "units_sold_rolling_min_3",
    "units_sold_rolling_max_3",
    "units_sold_trend_3"
]

encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_cat = encoder.fit_transform(X[categorical_cols])
X_num = X[numerical_cols].values
X_all = np.hstack([X_cat, X_num])

X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=0.2, random_state=42)
model = XGBRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R² Score       : {r2:.4f}")
print(f"MAE            : {mae:.2f} units")
print(f"RMSE           : {rmse:.2f} units")

plt.figure(figsize=(10,5))
plt.plot(y_test.values[:30], label="Réel", marker="o")
plt.plot(y_pred[:30], label="XGBoost", marker="x")
plt.title("Prévision IA vs Réel")
plt.xlabel("Index")
plt.ylabel("Units Sold")
plt.legend()
plt.tight_layout()
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/pred_vs_real.png")
plt.close()

feature_names = categorical_cols + numerical_cols
importances = model.feature_importances_
plt.figure(figsize=(10,6))
indices = np.argsort(importances)[::-1]
plt.barh(np.array(feature_names)[indices], importances[indices])
plt.xlabel("Importance")
plt.title("Feature Importances (XGBoost)")
plt.tight_layout()
plt.savefig("plots/feature_importances.png")
plt.close()

history = list(y_train)
mm_preds = []
for val in y_test:
    if len(history) >= 7:
        mm_preds.append(np.mean(history[-7:]))
    else:
        mm_preds.append(np.mean(history))
    history.append(val)

mae_mm = mean_absolute_error(y_test, mm_preds)
rmse_mm = np.sqrt(mean_squared_error(y_test, mm_preds))
r2_mm = r2_score(y_test, mm_preds)

print("\n--- Moyenne Mobile (7 jours) ---")
print(f"MAE            : {mae_mm:.2f} units")
print(f"RMSE           : {rmse_mm:.2f} units")
print(f"R² Score       : {r2_mm:.4f}")

plt.figure(figsize=(10,5))
plt.plot(y_test.values[:30], label="Réel", marker="o")
plt.plot(y_pred[:30], label="XGBoost", marker="x")
plt.plot(mm_preds[:30], label="Moyenne Mobile (7j)", marker="s")
plt.title("Comparaison XGBoost vs Moyenne Mobile vs Réel")
plt.xlabel("Index")
plt.ylabel("Units Sold")
plt.legend()
plt.tight_layout()
plt.savefig("plots/comparaison_xgb_vs_mm.png")
plt.close()

metrics = ["MAE", "RMSE", "R²"]
xgb_scores = [mae, rmse, r2]
mm_scores = [mae_mm, rmse_mm, r2_mm]
plt.figure(figsize=(8,5))
bar_width = 0.35
x = np.arange(len(metrics))
plt.bar(x, xgb_scores, width=bar_width, label="XGBoost")
plt.bar(x + bar_width, mm_scores, width=bar_width, label="Moyenne Mobile (7j)")
plt.xticks(x + bar_width/2, metrics)
plt.ylabel("Score")
plt.title("Comparaison des métriques XGBoost vs Moyenne Mobile")
plt.legend()
plt.tight_layout()
plt.savefig("plots/comparaison_metrics_xgb_vs_mm.png")
plt.close()

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/xgb_model.pkl")
joblib.dump(encoder, "models/ordinal_encoder.pkl")
print("✅ Modèle et encodeur sauvegardés dans le dossier models/")