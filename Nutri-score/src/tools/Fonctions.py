import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
from sklearn.neighbors import KNeighborsRegressor
from imblearn.over_sampling import SMOTE
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.miscmodels.ordinal_model import OrderedModel

def find_optimal_contamination(data, target_count, tol=1):
    """
    Trouve la contamination optimale pour obtenir un nombre pr�cis d'individus apr�s nettoyage.

    Args:
        data (DataFrame) : DataFrame contenant les données à nettoyer.
        target_count (int) : Nombre souhaité d'individus après nettoyage.
        tol (int or float, optional) : Tolérance pour le nombre d'individus (par défaut 1).

    Returns
    -------
    best_contamination (float) : contamination optimale.

    """

    low, high = 0.0, 0.5  # Les valeurs limites pour la contamination
    best_contamination = 0.0
    best_diff = float('inf')

    while low <= high:
        contamination = (low + high) / 2
        iso_forest = IsolationForest(
            contamination=contamination, random_state=42)
        iso_forest.fit(data)
        predictions = iso_forest.predict(data)

        cleaned_data = data[predictions == 1]
        current_count = len(cleaned_data)
        diff = abs(current_count - target_count)

        if diff < best_diff:
            best_diff = diff
            best_contamination = contamination

        if current_count < target_count:
            high = contamination - tol / len(data)
        else:
            low = contamination + tol / len(data)

    return best_contamination

def encoding_all_data(data, reverse=False):
    """
    Encode les colonnes non numériques d'un DataFrame à l'aide de l'encodage par étiquette.

    Cette fonction prend un DataFrame en entrée, encode toutes les colonnes non numériques 
    au format numérique en utilisant l'encodage par étiquette, et retourne à la fois le DataFrame 
    encodé et un DataFrame contenant des informations sur le processus d'encodage.

    Args:
        data (pd.DataFrame): Le DataFrame d'entrée contenant à la fois des colonnes numériques et non numériques.
        reverse (bool, optional): Si défini sur True, l'encodage des classes non numériques sera inversé. Par défaut, il est False.

    Returns:
        pd.DataFrame: Un nouveau DataFrame contenant les colonnes non numériques encodées ainsi que les colonnes numériques d'origine.
        pd.DataFrame: Un DataFrame avec des informations sur l'encodage, y compris les noms des variables, 
                      leurs codes correspondants et les modalités des classes.
    """
    # Séparer les variables numériques
    numeric_columns = data.select_dtypes(include=[float, int]).columns
    non_numeric_data = data.drop(columns=numeric_columns)

    # Encodage des variables non numériques
    class_info = {}  # Dictionnaire pour stocker les informations sur les classes
    infos = pd.DataFrame(columns=["Variable", "Code", "Modalité"])

    for column in non_numeric_data.columns:
        le = LabelEncoder()
        non_numeric_data[column] = le.fit_transform(non_numeric_data[column])

        # Stocker les informations sur les classes dans le dictionnaire
        class_info[column] = {
            'label_encoder': le,
            'classes': list(le.classes_)
        }

        # Récupérer les classes correspondantes
        if reverse:
            classes = list(reversed(le.classes_))
        else:
            classes = le.classes_

        # Afficher les correspondances entre les codes et les classes
        for code, classe in enumerate(classes):
            print(f"Colonnes '{column}', Classe {code} : {classe}")
            variable = column
            code = code
            classe = classe
            
            infos = infos._append({
                "Variable": variable,
                "Code": code,
                "Modalité": classe
            }, ignore_index=True)      

    # Réintégrer les variables numériques dans le jeu de données encodé
    encoded_data = pd.concat([non_numeric_data, data[numeric_columns]], axis=1)

    return encoded_data, infos

def vif_selection(data, target, vif_value=5):
    """
    Sélectionne les variables explicatives en fonction de leur facteur d'inflation de la variance (VIF).

    Cette fonction calcule le VIF pour chaque variable explicative d'un DataFrame 
    et supprime celles dont le VIF dépasse un seuil donné, indiquant une colinéarité élevée.

    Args:
        data (pd.DataFrame): Le DataFrame contenant les variables explicatives et la variable cible.
        target (str): Le nom de la variable cible que l'on souhaite exclure de l'analyse.
        vif_value (int, optional): La valeur seuil du VIF au-delà de laquelle une variable est considérée comme colinéaire. 
                                   Par défaut, il est fixé à 5.

    Returns:
        list: Une liste des noms des variables rejetées en raison d'un VIF élevé.
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

def convergence_error_OrderedModel(data, target, distr_logistic, method_logistic, iterations, test_size, scoring='accuracy', average='macro'):
    """
    Calcule la convergence de la performance moyenne cumulative d'une régression logistique ordinale sur un certain nombre d'itérations
    et affiche la courbe de convergence en fonction de la métrique spécifiée.

    Args:
        data (pd.DataFrame): Le jeu de données contenant les caractéristiques et la cible.
        target (str): Le nom de la colonne de la variable cible à prédire.
        distr_logistic (str): La distribution utilisée pour la régression logistique ordinale (par ex., 'logit').
        method_logistic (str): La méthode d'optimisation pour l'entraînement du modèle (par ex., 'bfgs').
        iterations (int): Le nombre d'itérations pour entraîner et évaluer le modèle.
        test_size (float): La proportion des données utilisées pour le jeu de test lors de chaque itération (entre 0 et 1).
        scoring (str, optional): La métrique utilisée pour évaluer les performances du modèle. Peut être 'accuracy', 'precision',
                                 'recall', ou 'f1'. Par défaut 'accuracy'.
        average (str, optional): La méthode d'agrégation des scores pour les métriques multiclasses. Utilisé pour les métriques
                                 'precision', 'recall', et 'f1'. Par défaut 'macro'.

    Returns:
        float: La moyenne cumulative de la métrique choisie calculée sur toutes les itérations.
    """
    # Initialisations
    X = data.drop(columns=target, axis=1)
    y = data[target]
    model_err = []

    # Boucle pour entraîner le modèle sur plusieurs divisions de données et calculer l'erreur
    for _ in range(iterations):
        # Split de l'ensemble en apprentissage et validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True, random_state=None)
        
        # Entraînement du modèle de classification
        model = OrderedModel(y_train, X_train, distr=distr_logistic).fit(method=method_logistic, disp=False)
        
        # Prédictions
        y_pred_prob = model.predict(X_test)
        predicted_classes = np.argmax(y_pred_prob, axis=1)

        # Calcul de l'erreur en fonction de la méthode et la métrique choisie       
        if scoring == 'accuracy' :
            model_err.append(accuracy_score(y_test, predicted_classes))
            metric = 'Accuracy'
            title = f"\nConvergence de l'accuracy sur {iterations} itérations\n"
        
        elif scoring == 'precision':
            model_err.append(precision_score(y_test, predicted_classes, average=average))
            metric = 'Precision'
            title = f"\nConvergence de la precision sur {iterations} itérations\n"
        
        elif scoring == 'recall':
            model_err.append(recall_score(y_test, predicted_classes, average=average))
            metric = 'Recall'
            title = f"\nConvergence du reacll sur {iterations} itérations\n"
        
        elif scoring == 'f1':
            model_err.append(f1_score(y_test, predicted_classes, average=average))
            metric = 'F1-Score'
            title = f"\nConvergence du F1-Score sur {iterations} itérations\n"

        else:
            # Afficher un avertissement et utiliser l'accuracy par défaut
            warnings.warn(f"Métrique '{scoring}' non reconnue. Veuillez choisir 'accuracy', 'precision', 'recall' ou 'f1'. Utilisation de l'accuracy par défaut.")
            model_err.append(accuracy_score(y_test, predicted_classes))
            metric = 'Accuracy'
            title = f"\nConvergence de l'accuracy sur {iterations-1} itérations\n"

    # Calcul de l'erreur cumulative moyenne sur toutes les itérations
    perform_list = [np.mean(model_err[:i+1]) for i in range(iterations)]
    perform = round(np.mean(perform_list), 4)

    # Affichage du graphique de la convergence
    plt.plot(perform_list, label='Erreur moyenne cumulative')
    plt.xlabel('Nombre d\'itérations')
    plt.ylabel(metric)
    plt.title(title)
    plt.show()

    return perform