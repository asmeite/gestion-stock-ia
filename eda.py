# 1. IMPORTS
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

# 2. LECTURE DU DATASET
df = pd.read_csv("data.csv")

print("df.shape:", df.shape)

# Créer un dossier pour stocker les images
os.makedirs("analysis_outputs", exist_ok=True)

# 3. CORRÉLATION NUMÉRIQUE
numerical_cols = [
    "unit_price", "available_stock", "supplier_lead_time_days",
    "on_promotion", "is_holiday", "temperature_c", "rainfall_mm", "units_sold"
]
corr_matrix = df[numerical_cols].corr()
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix - Demand Prediction (Units Sold)")
plt.tight_layout()
plt.savefig("analysis_outputs/correlation_matrix.png")
plt.close()

# 4. DISTRIBUTION DE LA DEMANDE (TARGET)
plt.figure(figsize=(8, 5))
sns.histplot(df["units_sold"], kde=True, bins=30)
plt.title("Distribution of Units Sold")
plt.xlabel("Units Sold")
plt.savefig("analysis_outputs/units_sold_distribution.png")
plt.close()

# 5. EFFET DES PROMOS SUR LA DEMANDE
plt.figure(figsize=(7, 5))
sns.boxplot(x="on_promotion", y="units_sold", data=df)
plt.title("Effect of Promotion on Units Sold")
plt.xlabel("On Promotion (0 = No, 1 = Yes)")
plt.ylabel("Units Sold")
plt.savefig("analysis_outputs/promotion_effect.png")
plt.close()

# 6. EFFET DU PRIX
plt.figure(figsize=(8, 5))
sns.scatterplot(x="unit_price", y="units_sold", data=df)
plt.title("Effect of Unit Price on Units Sold")
plt.xlabel("Unit Price (€)")
plt.ylabel("Units Sold")
plt.savefig("analysis_outputs/price_effect.png")
plt.close()

# 7. DEMANDE PAR JOUR DE LA SEMAINE
plt.figure(figsize=(10, 5))
order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
sns.boxplot(x="weekday", y="units_sold", data=df, order=order)
plt.title("Units Sold by Weekday")
plt.xlabel("Weekday")
plt.ylabel("Units Sold")
plt.savefig("analysis_outputs/weekday_effect.png")
plt.close()

# 8. DEMANDE PAR SAISON
plt.figure(figsize=(8, 5))
sns.boxplot(x="season", y="units_sold", data=df, order=["Winter", "Spring", "Summer", "Autumn"])
plt.title("Units Sold by Season")
plt.xlabel("Season")
plt.ylabel("Units Sold")
plt.savefig("analysis_outputs/seasonal_demand.png")
plt.close()

# 9. HEATMAP DES STOCKS DISPONIBLES vs DEMANDE
plt.figure(figsize=(8, 5))
sns.scatterplot(x="available_stock", y="units_sold", data=df)
plt.title("Stock Level vs Units Sold")
plt.xlabel("Available Stock")
plt.ylabel("Units Sold")
plt.savefig("analysis_outputs/stock_vs_demand.png")
plt.close()

print("✅ All analysis images saved in: 'analysis_outputs/'")
