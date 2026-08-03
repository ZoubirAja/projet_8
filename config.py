import pandas as pd
import numpy as np
from preprocessing import clean_col_names, downcast_floats

df = pd.read_parquet('df_final_clean.parquet')
df = clean_col_names(df)
df = downcast_floats(df)

X = df.drop(columns=['TARGET'])
y = df['TARGET']
