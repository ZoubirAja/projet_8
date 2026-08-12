import re

# Utilisé à la fois à l'entraînement (config.py) et au moment de la prédiction (customer.py) :
# le modèle doit voir exactement les mêmes noms/types de colonnes des deux côtés.

def clean_col_names(df):
    df.columns = [re.sub(r'[^A-Za-z0-9_]', '_', col) for col in df.columns]
    return df


def downcast_floats(df):
    float64_cols = df.select_dtypes(include='float64').columns
    df[float64_cols] = df[float64_cols].astype('float32')
    return df
