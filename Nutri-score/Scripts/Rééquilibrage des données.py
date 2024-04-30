# Packages
import pandas as pd
from imblearn.over_sampling import SMOTE

# Importation de la base
chemin_fichier = r"C:\Projets-personnels\Nutri-score\Données\4Data_no_miss_unbalanced.csv"

df = pd.read_csv(chemin_fichier, sep=',')

percentages = round(df['nutriscore_grade'].value_counts(normalize=True) * 100,1)
print(percentages)

# Appliquer SMOTE
X = df.drop(columns=['nutriscore_grade'])
y = df['nutriscore_grade']  # Cible

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Convertir en DataFrame si nécessaire
df_resampled = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame({'nutriscore_grade': y_resampled})], axis=1)

# Vérification des nouveaux pourcentages
percentages_resampled = round(df_resampled['nutriscore_grade'].value_counts(normalize=True) * 100,1)
print(percentages_resampled)

# Enregistrer les données
df_resampled.to_csv(r"C:\Projets-personnels\Nutri-score\Données\5Data_no_miss_balanced.csv", index=False)
