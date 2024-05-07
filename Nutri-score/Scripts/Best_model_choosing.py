# Packages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,f1_score,recall_score,roc_curve,auc
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier

# Fixer la graine pour la reproductibilité
np.random.seed(42)

# Importation de la base
chemin_fichier = r"C:\5Data_no_miss_balanced.csv"

df = pd.read_csv(chemin_fichier, sep=',')

# Vérification
df.info()
print(df.isnull().sum())
df.describe()

df, infos = encode_data(df,reverse=True)

# Division des données
X = df.drop('nutriscore_grade', axis=1) # Variables prédictives
y = df['nutriscore_grade'] # Variable cible

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Initialiser tous les modèles
xg, ada, cat, lg = XGBClassifier(random_state=42), AdaBoostClassifier(random_state=42), CatBoostClassifier(random_state=42), lgb.LGBMClassifier(random_state=42)

models = [xg,ada,cat,lg]

models_dict = {'XGBClassifier': xg,'AdaBoostClassifier': ada,
               'CatBoostClassifier': cat,'LGBMClassifier': lg}

best_params_per_model,results_per_model,max_depth_RF = compare_boosting(models=models,
                                      nb_estimators=np.arange(50, 550, 50),
                                      learn_rate=np.arange(0.1, 1.1, 0.1),
                                      gamma=np.arange(0, 1.05, 0.05),
                                      l1=np.arange(0, 1.05, 0.05),
                                      l2=np.arange(0, 1.05, 0.05),
                                      max_depth=np.arange(1, 21, 1),
                                      metric='accuracy',
                                      average_metric='macro')

resultats = entrainement_model_optimal(models[3],average_metric='macro')