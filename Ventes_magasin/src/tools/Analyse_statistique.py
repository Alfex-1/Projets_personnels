import pandas as pd
from scipy.stats import chi2_contingency
from scipy.stats import shapiro, ansari, mannwhitneyu, fligner, kruskal, spearmanr
import pingouin as pg

df = pd.read_csv("Ventes.csv", sep=";")

del df['Transaction ID'], df['Customer ID']

df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# =============================================================================
# Questions
# =============================================================================

# 1. Est-ce qu'il y a une catgéorie de produit privilégiée par les hommes ou les femmes ?
table_contin = pd.crosstab(df['Gender'], df['Product Category'])

_, p, _, _ = chi2_contingency(table_contin)
del table_contin

# Réponse : Non, les hommes et les femmes ont tendance à acheter les mêmes catgéories de produits


# 2. Est-ce que les hommes achètent en moyenne plus que les femmes ou inversement ?

df_man = df[df['Gender'] == 'Male']
df_woman = df[df['Gender'] == 'Female']

# Test de normalité
_, p_value_man = shapiro(df_man['Quantity'])
_, p_value_woman = shapiro(df_woman['Quantity'])
# Les deux groupes ne suivent pas une loi normale

# Test d'égalité des variances
_, p_value_ansari = ansari(df_man['Quantity'], df_woman['Quantity'])
# Les deux groupes ont statistiquemet la même variance

# Test d'égalité des médianes
_, p_value_mediane = mannwhitneyu(df_man['Quantity'], df_woman['Quantity'])
# Les deux groupes ont la même médiane statistiquement

# Réponse : Aucun n'achète en moyenne plus que l'autre en quantité.


# 3. Est-ce que l'âge des clients a un impact sur leurs préférences, en uanitité et en produit ?

df_clothes = df[df['Product Category'] == 'Clothing']
df_electro = df[df['Product Category'] == 'Electronics']
df_beauty = df[df['Product Category'] == 'Beauty']
# Aucun groupe ne suit une distribution normale

# Test de normalité
_, p_value_clothes = shapiro(df_clothes['Age'])
_, p_value_electro = shapiro(df_electro['Age'])
_, p_value_beauty = shapiro(df_beauty['Age'])
# Les trois groupes ne suivent pas une loi normale

# Test d'homogénéité des variances
_, p_value_homoge = fligner(
    df_clothes['Age'], df_electro['Age'], df_beauty['Age'])
# Les variances des trois groupes sont égales

# Test d'égalité des médianes
_, p_value_mediane_age = kruskal(
    df_clothes['Age'], df_electro['Age'], df_beauty['Age'])
# Les médianes sont statistiquement égales


# Test de corrélation
corr, pvalue, test = correlation(df, 'Age', 'Quantity', alpha=0.05)

# Il n'existe donc aucune relation monotone entre l'âge des clients et les produits et quantités vendues


# 4. Quels sont les produits les plus achetés ?

# Test de normalité
_, p_value_clothes_qte = shapiro(df_clothes['Quantity'])
_, p_value_electro_qte = shapiro(df_electro['Quantity'])
_, p_value_beauty_qte = shapiro(df_beauty['Quantity'])

# Aucun ne conserve la normalité

# Test d'homogénéité des variances
_, p_value_homoge_qte = fligner(
    df_clothes['Quantity'], df_electro['Quantity'], df_beauty['Quantity'])
# Les variances des trois groupes sont égales

# Test d'égalité des médianes
_, p_value_mediane_qte = kruskal(
    df_clothes['Quantity'], df_electro['Quantity'], df_beauty['Quantity'])
# Les médianes sont statistiquement égales
