import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import acf, pacf

# Importer les données
df_init = pd.read_csv(r'C:\Projets_personnels\Loto\data\historique-loto.csv', sep=';')

df_init['date'] = pd.to_datetime(df_init['date'], format='%d/%m/%Y')
df_init = df_init.sort_values(by='date', ascending=True)
df_init.set_index('numero_tirage', inplace=True)

df = df_init.copy()

# Créer les variables one-hot
boules = ['boule_1', 'boule_2', 'boule_3', 'boule_4', 'boule_5']

# Crée les colonnes 1 à 49
for numero in range(1, 50):
    df[numero] = df[boules].eq(numero).any(axis=1).astype(int)

df.columns = df.columns.astype(str)


def conditional_probability(X, number, lag=1):
    x = X[number].to_numpy()

    a = x[:-lag]
    b = x[lag:]

    # P(X[t+lag] = 1 | X[t] = 1)
    mask = a == 1

    return b[mask].mean()

# Obtetnir la liste des colonnes binaires (1 à 49)
num_cols = [str(i) for i in range(1, 50)]

results = []

for number in num_cols:

    p = df[number].mean()

    for lag in range(1, 21):

        p_cond = conditional_probability(df, number, lag)

        results.append({
            "numero": int(number),
            "lag": lag,
            "p": p,
            "p_conditionnelle": p_cond,
            "difference": p_cond - p,
            "ratio": p_cond / p if p > 0 else np.nan
        })

acf_analysis = pd.DataFrame(results)
print(acf_analysis)