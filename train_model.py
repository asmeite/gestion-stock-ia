import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import matplotlib.pyplot as plt

# Chargement et préparation des données
df = pd.read_csv("data/simulated_sales_data_en.csv", parse_dates=["Date"])
df = df.sort_values(["Product", "Date"])
le = LabelEncoder()
df["Product_Code"] = le.fit_transform(df["Product"])
for lag in [1, 2, 3]:
    df[f"Sales_Lag_{lag}"] = df.groupby("Product")["Daily_Sales"].shift(lag)
df = df.dropna()

# Features & target
features = [
    "Product_Code", "Day_of_Week", "External_Temperature",
    "Promotion_Active", "Sales_Lag_1", "Sales_Lag_2", "Sales_Lag_3"
]
X = df[features]
y = df["Daily_Sales"]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Entraînement
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prédiction et métriques
y_pred = model.predict(X_test)
mae  = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2   = r2_score(y_test, y_pred)

print("\n📊 Performances du modèle global :")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.2f}")

# Création du répertoire 'plots' s'il n'existe pas
os.makedirs("plots", exist_ok=True)

# 1) Graphique échantillon réel vs prédit (50 points)
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(y_test.values[:100], label="Ventes réelles", marker='o')
ax1.plot(y_pred[:100],       label="Ventes prédites", marker='x')
ax1.set_xlabel("Index")
ax1.set_ylabel("Ventes")
ax1.legend()
ax1.grid(True)
plt.tight_layout()
# Sauvegarde
fig1.savefig("plots/prediction_vs_reel.png")
plt.close(fig1)

# 2) Nuage réel vs prédit
fig2, ax2 = plt.subplots(figsize=(5, 5))
ax2.scatter(y_test, y_pred, alpha=0.6)
mn, mx = y_test.min(), y_test.max()
ax2.plot([mn, mx], [mn, mx], 'r--')
ax2.set_xlabel("Ventes réelles")
ax2.set_ylabel("Ventes prédites")
ax2.grid(True)
plt.tight_layout()
# Sauvegarde
fig2.savefig("plots/reel_vs_predit.png")
plt.close(fig2)

print("\n✅ Graphiques sauvegardés dans le dossier 'plots/'")

# # (Optionnel) Sauvegarde du modèle et de l'encodeur
# os.makedirs("models", exist_ok=True)
# joblib.dump(model, "models/model_global.pkl")
# joblib.dump(le,    "models/product_encoder.pkl")
# print("✅ Modèle et encodeur sauvegardés dans 'models/'")
