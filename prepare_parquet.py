"""
Script à lancer une seule fois (ou à chaque mise à jour de df_final_clean.parquet).

Réécrit le parquet en petits groupes de lignes (row groups) triés par SK_ID_CURR,
pour que customer.py puisse lire uniquement le groupe concerné par un id au lieu
de charger les 307 511 lignes en mémoire à chaque démarrage de l'API.
Reste compressé sur disque (~même taille que l'original) : pas d'explosion d'espace disque.
"""
import pandas as pd
from preprocessing import clean_col_names, downcast_floats

SOURCE = "df_final_clean.parquet"
DEST = "customers_indexed.parquet"
ROW_GROUP_SIZE = 5000  # ~62 groupes pour 307 511 lignes

df = pd.read_parquet(SOURCE)
df = clean_col_names(df)
df = downcast_floats(df)
df = df.sort_values("SK_ID_CURR")  # garantit des groupes avec des plages d'id disjointes

df.to_parquet(DEST, row_group_size=ROW_GROUP_SIZE, index=False)

print(f"Écrit {DEST} : {len(df)} lignes, {df.shape[1]} colonnes")
