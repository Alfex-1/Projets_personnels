import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Données_nutriscore_v3\6Data_no_miss_balanced.csv")

sns.pairplot(df, hue="NutriScore")

g = sns.FacetGrid(df, col="NutriScore")
g.map_dataframe(sns.relplot, x="Protéines", y="Graisses")



