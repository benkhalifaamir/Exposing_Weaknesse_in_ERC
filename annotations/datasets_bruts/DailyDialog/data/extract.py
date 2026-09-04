import pandas as pd

# Charger le fichier extrait
df = pd.read_csv("dailydialog_sentiment.csv")

# Vérifie que la colonne 'data_split' est bien là
if 'data_split' not in df.columns:
    raise ValueError("La colonne 'data_split' est manquante dans le fichier CSV.")

# Filtrer par split
df_train = df[df['data_split'] == 'train']
df_val   = df[df['data_split'] == 'validation']
df_test  = df[df['data_split'] == 'test']

# Sauvegarde des fichiers
df_train.to_csv("dailydialog_sentiment_train.csv", index=False)
df_val.to_csv("dailydialog_sentiment_validation.csv", index=False)
df_test.to_csv("dailydialog_sentiment_test.csv", index=False)

print("✅ Fichiers créés :")
print(f"  - Train      : {len(df_train)} lignes")
print(f"  - Validation : {len(df_val)} lignes")
print(f"  - Test       : {len(df_test)} lignes")
