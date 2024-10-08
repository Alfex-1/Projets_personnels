# =============================================================================
# Packages et modules
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest, RandomForestRegressor, RandomForestClassifier
import missingno as msno
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import chi2, VarianceThreshold, RFECV, SelectKBest, f_classif, f_regression
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, accuracy_score, recall_score, f1_score

# =============================================================================
# Fonctions
# =============================================================================
def vif_selection(data, target, vif_value=5):
    """
    Effectue une sélection de caractéristiques en utilisant le facteur d'inflation de la variance (VIF) pour détecter
    les variables présentant une colinéarité élevée. Les variables avec un VIF supérieur ou égal au seuil spécifié 
    sont supprimées itérativement.

    Args:
        data (pd.DataFrame): Le DataFrame contenant les variables explicatives et la variable cible.
        target (str): Le nom de la colonne cible dans `data` à exclure de l'analyse.
        vif_value (float, optional): Seuil du facteur d'inflation de la variance pour identifier la colinéarité. Par défaut, 5.

    Raises:
        ValueError: Si le DataFrame ne contient aucune colonne numérique continue (type `int` ou `float`).
        KeyError: Si la variable cible `target` n'existe pas dans `data`.

    Returns:
        list: Liste des noms de colonnes rejetées en raison de leur colinéarité élevée.
    """
    # Séparer les variables explicatives et la variable cible
    X = data.drop(columns=[target])
    X = X.select_dtypes(include=['int', 'float'])
    
    rejected_variables = []
    
    while True:
        # Calculer le VIF pour chaque variable
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        
        # Trouver la variable avec le VIF le plus élevé
        max_vif = vif_data["VIF"].max()
        
        # Vérifier si le VIF maximal est supérieur ou égal au seuil
        if max_vif >= vif_value:
            # Identifier la variable à supprimer
            variable_to_remove = vif_data.loc[vif_data["VIF"].idxmax(), "Variable"]
            rejected_variables.append(variable_to_remove)
            # Supprimer la variable du DataFrame
            X = X.drop(columns=[variable_to_remove])
        else:
            break  # Aucune variable n'a un VIF supérieur au seuil
            
    return rejected_variables

def chi2_selection(data, target, threshold,alpha=0.05):
    """
    Effectue une sélection de caractéristiques discrètes en utilisant le test du chi-deux pour déterminer les variables
    qui ne sont pas statistiquement associées à la variable cible. Retourne la liste des variables rejetées.

    Args:
        data (pd.DataFrame): Le DataFrame contenant les variables explicatives et la variable cible.
        target (str): Le nom de la colonne cible dans `data`.
        threshold (int): Le nombre maximum de modalités pour qu'une variable soit considérée comme discrète.
        alpha (float, optional): Niveau de signification statistique (seuil de p-value) pour le test chi-deux. Par défaut, 0.05.
        
    Raises:
        ValueError: Si la variable cible n'existe pas dans le DataFrame ou si elle n'est pas catégorielle.

    Returns:
        list: Liste des noms de colonnes qui ne sont pas significativement associées à la variable cible selon le test chi-deux.
    """
    data = data.dropna()
    
    # Séparer la variable cible
    y = data[target]
    
    # Sélectionner les colonnes catégorielles
    df = data.select_dtypes(include=['object', 'category'])
    
    # Encoder toutes les variables catégorielles avec LabelEncoder
    le = LabelEncoder()
    encoded_data = df.apply(lambda col: le.fit_transform(col))
    
    # Détecter les variables discrètes
    discrete_vars = [col for col in df.columns
                     if pd.api.types.is_integer_dtype(df[col]) 
                     and df[col].nunique() <= threshold]
    
    df_discrete = data[discrete_vars]
    
    # Combiner les deux jeux de données
    combined_data = pd.concat([encoded_data.reset_index(drop=True), df_discrete.reset_index(drop=True)], axis=1)
    
    # Appliquer le test du chi-deux
    _, p_values = chi2(combined_data, y)
    
    # Créer une liste des noms de colonnes (variables)
    all_columns = combined_data.columns
    
    # Séparer les variables rejetées et conservées
    rejected_variables = list([col for col, p_val in zip(all_columns, p_values) if p_val >= alpha])
    
    return rejected_variables

def knn_impute_categorical(df, column_name, n_neighbors=3, weights='distance'):
    """
    Impute les valeurs manquantes d'une colonne catégorielle en utilisant l'algorithme des K-plus-proches-voisins (KNN).

    Args:
        df (pd.DataFrame): Le DataFrame contenant les données.
        column_name (str): Le nom de la colonne catégorielle à imputer.
        n_neighbors (int, optional): Le nombre de voisins à utiliser pour l'imputation. Par défaut, 3.
        weights (str, optional): La stratégie de pondération des voisins. Peut prendre les valeurs 'uniform' 
                                 ou 'distance'. Par défaut, 'distance'.

    Raises:
        ValueError: Si la colonne spécifiée n'existe pas dans le DataFrame.
    
    Returns:
        pd.Series: La colonne imputée avec les valeurs manquantes remplacées.
    
    Notes:
        - Cette fonction utilise `LabelEncoder` pour encoder les variables catégorielles en entiers avant l'imputation.
        - Après l'imputation, les valeurs sont converties aux labels originaux.
    """
    # Vérifie que la colonne existe
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
    
    # Étape 1: Encoder la variable non ordinale
    le = LabelEncoder()
    encoded_column = le.fit_transform(df[column_name].astype(str))  # Convertir en str pour gérer les NaN
    df[column_name + '_Encoded'] = encoded_column
    
    # Étape 2: Appliquer KNNImputer
    imputer = KNNImputer(n_neighbors=n_neighbors,weights=weights)
    df[column_name + '_Encoded'] = imputer.fit_transform(df[[column_name + '_Encoded']])
    
    # Étape 3: Rétablir les modalités d'origine
    df[column_name + '_Encoded'] = df[column_name + '_Encoded'].round().astype(int)
    df[column_name] = le.inverse_transform(df[column_name + '_Encoded'])
    
    # Suppression de la colonne encodée
    df.drop(columns=[column_name + '_Encoded'], inplace=True)
    
    return df[column_name]

def feature_elimination_cv(data, target, method="regression", scoring="rmse",min_features_to_select=1,test_size=1/3, cv=5):
    """
    Effectue une sélection de caractéristiques par élimination itérative avec validation croisée, 
    en utilisant un modèle de forêt aléatoire pour la régression ou la classification.

    Args:
        data (pd.DataFrame): Les données d'entrée sous forme de DataFrame, comprenant les variables explicatives et la cible.
        target (str): Le nom de la colonne cible dans le DataFrame `data`.
        method (str, optional): Spécifie le type de modèle à utiliser, "regression" ou "classification". 
                                Par défaut, utilise la régression.
        scoring (str, optional): La métrique de performance à optimiser lors de la sélection. Pour la régression, 
                                 choisir parmi "rmse" ou "mae". Pour la classification, choisir parmi "accuracy", 
                                 "recall" ou "f1". Par défaut, "rmse" pour la régression.
        min_features_to_select (int, optional): Le nombre minimal de caractéristiques à conserver dans le modèle final. 
                                                Par défaut, 1.
        test_size (float, optional): Proportion des données à utiliser pour le jeu de test. Par défaut, 1/3.
        cv (int, optional): Nombre de plis pour la validation croisée. Par défaut, 5.

    Raises:
        ValueError: Si le type de modèle (`method`) n'est pas "regression" ou "classification", ou si la métrique (`scoring`) 
                    n'est pas valide pour le type de modèle spécifié.
    
    Returns:
        list: Une liste des variables qui ne participent pas à l'amélioration de la métrique choisie.
    """
    # Encodage des colonnes catégorielles
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    # Séparer les variables explicatives (features) et la cible
    X = data.drop(columns=[target])
    y = data[target]

    # Séparer les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Initialiser le modèle RandomForest selon la méthode
    if method == "regression":
        model = RandomForestRegressor()
        
        # Définir les scoring pour la régression
        if scoring == "rmse":
            scoring_func = make_scorer(lambda y_true, y_pred: mean_squared_error(y_true, y_pred, squared=False))
        elif scoring == "mae":
            scoring_func = make_scorer(mean_absolute_error)
        else:
            return "Invalid scoring method for regression. Use 'rmse' or 'mae'."

    elif method == "classification":
        model = RandomForestClassifier()
        
        # Définir les scoring pour la classification
        if scoring == "accuracy":
            scoring_func = make_scorer(accuracy_score)
        elif scoring == "recall":
            scoring_func = make_scorer(recall_score)
        elif scoring == "f1":
            scoring_func = make_scorer(f1_score)
        else:
            return "Invalid scoring method for classification. Use 'accuracy', 'recall', or 'f1'."

    else:
        return "Method must be 'regression' or 'classification'."

    # Initialiser RFECV avec le modèle et le scoring appropriés
    rfecv = RFECV(estimator=model, step=1, cv=cv, scoring=scoring_func,
                  min_features_to_select=min_features_to_select,n_jobs=3)

    # Ajuster RFECV sur les données d'entraînement
    rfecv.fit(X_train, y_train)

    # Variables conservées et rejetées
    rejected_variables = list(X.columns[~rfecv.support_])

    return rejected_variables

def estimate_forecasting_error(model, X, y, k=5, metric='MSE'):
    """
    Estime l'erreur théorique de prévision sur l'ensemble des données (train + test)
    à l'aide de la méthode de validation croisée K-Fold, avec choix de la métrique.

    Parameters:
    - model: modèle déjà entraîné (doit implémenter `predict`).
    - X: caractéristiques (features) de l'ensemble de données.
    - y: valeurs cibles de l'ensemble de données.
    - k: nombre de plis (folds) pour la validation croisée.
    - metric: métrique d'évaluation ('mse', 'rmse', 'mae' ou 'mape').

    Returns:
    - mean_error: erreur moyenne selon la métrique spécifiée.
    """
    # Initialisation du K-Fold
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    error_list = []

    # Choix de la fonction d'erreur selon la métrique spécifiée
    if metric == 'mse':
        error_fn = mean_squared_error
    elif metric == 'rmse':
        error_fn = lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))
    elif metric == 'mae':
        error_fn = mean_absolute_error
    elif metric == 'mape':
        error_fn = mean_absolute_percentage_error
    else:
        raise ValueError("Métrique non reconnue. Utilisez 'mse', 'rmse', 'mae' ou 'mape'.")

    # Validation croisée
    for train_index, test_index in kfold.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Prédictions sur le pli de test avec le modèle pré-entraîné
        y_pred = model.predict(X_test)

        # Calcul de l'erreur sur ce pli avec la fonction choisie
        error = error_fn(y_test, y_pred)
        error_list.append(error)

    # Moyenne de l'erreur sur tous les plis
    mean_error = round(np.mean(error_list),4)
    return mean_error