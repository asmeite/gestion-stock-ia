import pandas as pd
import psycopg2
from sqlalchemy import create_engine

# 📁 Remplace par ton chemin local si besoin
csv_path = "data.csv"

# 🔐 Connexion PostgreSQL Azure
host = "stockai-postgres.postgres.database.azure.com"
dbname = "postgres"
user = "postgres"
password = "azure_stock_8"
sslmode = "require"

# Chargement du fichier CSV
df = pd.read_csv(csv_path, parse_dates=["date"])

# Connexion avec SQLAlchemy
engine = create_engine(
    f"postgresql+psycopg2://{user}:{password}@{host}:5432/{dbname}?sslmode={sslmode}"
)

# Import dans la table
df.to_sql("simulated_sales", engine, if_exists="append", index=False)

print("✅ Données importées avec succès dans simulated_sales.")
