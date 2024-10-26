import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE

def encoding_binary(data, list_variables, reverse=False, reverse_variables=None):
    """
    Effectue l'encodage binaire des variables catégorielles spécifiées dans un DataFrame.

    Cette fonction encode les variables catégorielles en utilisant un encodage binaire (0 et 1), 
    tout en stockant les informations sur les classes originales. Elle offre également la possibilité 
    d'inverser l'encodage pour certaines variables.

    Args:
        data (pd.DataFrame): Le DataFrame contenant les données à encoder.
        list_variables (list of str): La liste des noms de colonnes à encoder.
        reverse (bool, optional): Indique si l'encodage doit être inversé pour certaines variables. 
                                  Par défaut à False.
        reverse_variables (list of str, optional): La liste des variables pour lesquelles l'encodage 
                                                   doit être inversé. Ignoré si reverse est False. 
                                                   Par défaut à None.

    Returns:
        pd.DataFrame: Le DataFrame avec les variables encodées.
        pd.DataFrame: Un DataFrame contenant les informations sur l'encodage (variable, modalité, code).
    """
    # Copie du DataFrame pour éviter les modifications inattendues
    encoded_data = data.copy()

    # Dictionnaire pour stocker les informations sur les classes
    class_info = {}

    # DataFrame pour stocker les informations sur l'encodage
    infos = pd.DataFrame(columns=["Variable","Modalité","Code"])

    for variable in list_variables:
        # Initialisation de l'encodeur binaire
        le = LabelEncoder()

        # Encodage des variables
        encoded_data[variable] = le.fit_transform(encoded_data[variable])

        # Stocker les informations sur les classes dans le dictionnaire
        class_info[variable] = {
            'label_encoder': le,
            'classes': list(le.classes_)}

        # Récupérer les classes correspondantes
        if reverse and variable in reverse_variables:
            # Récupérer les modalités originales
            original_modalities = class_info[variable]['classes']
            # Inverser les classes
            encoded_data[variable] = 1 - encoded_data[variable]
            # Définir les classes aprés inversion
            classes = original_modalities[::-1]
        else:
            classes = le.classes_

        # Afficher les correspondances entre les codes et les classes
        for code, classe in enumerate(classes):
            print(f"Colonnes '{variable}', Classe {code} : {classe}")

            infos = infos._append({
                "Variable": variable,
                "Modalité": classe,
                "Code": code},
                ignore_index=True)

    return encoded_data, infos

def exhaustive_logistic_regression_search(X, y, penalties, C_values, l1_ratios, solvers, 
                                          metric='accuracy', average='macro', cv=10):
    """
    Effectue une recherche exhaustive des hyperparamètres pour une régression logistique en 
    utilisant une validation croisée.

    Cette fonction teste différentes combinaisons de pénalités, de valeurs de C, de ratios l1 
    et de solveurs afin de trouver la meilleure configuration en fonction d'une métrique donnée. 
    Les solveurs et les pénalités sont vérifiés pour compatibilité, et la validation croisée est 
    utilisée pour estimer les performances moyennes du modèle.

    Args:
        X (array-like): Les données d'entraînement, de forme (n_samples, n_features).
        y (array-like): Les étiquettes ou cibles correspondantes, de forme (n_samples,).
        penalties (list of str): Liste des types de pénalités à tester ('l1', 'l2', 'elasticnet').
        C_values (list of float): Liste des valeurs de régularisation C à tester.
        l1_ratios (list of float): Liste des ratios l1 à tester (utilisé uniquement avec Elastic-Net).
        solvers (list of str): Liste des solveurs à tester ('liblinear', 'saga', etc.).
        metric (str, optional): La métrique d'évaluation à utiliser ('accuracy', 'precision', 'recall', 
                                'f1', 'log_loss'). Par défaut 'accuracy'.
        average (str, optional): Le type de moyennage à utiliser pour les métriques multiclasses 
                                 ('weighted', 'macro', etc.). Par défaut 'macro'.
        cv (int, optional): Le nombre de plis pour la validation croisée. Par défaut 20.

    Raises:
        ValueError: Si la métrique spécifiée n'est pas reconnue.

    Returns:
        dict: Un dictionnaire contenant les meilleurs paramètres trouvés ('penalty', 'C', 'solver', 
              'l1_ratio') et la meilleure performance atteinte.
    """
    best_score = -np.inf
    best_params = {}
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)

    # Parcourir toutes les combinaisons de pénalité, C, l1_ratio et solveur
    for penalty in penalties:
        for solver in solvers:
            # Vérifier la compatibilité entre le solveur et le type de pénalité
            if penalty == 'l1' and solver not in ['liblinear', 'saga']:
                continue  # Les solveurs 'liblinear' et 'saga' prennent en charge L1
            if penalty == 'l2' and solver not in ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']:
                continue  # Tous ces solveurs prennent en charge L2
            if penalty == 'elasticnet' and solver != 'saga':
                continue  # 'saga' est le seul solveur qui prend en charge Elastic-Net

            for C in C_values:
                for l1_ratio in l1_ratios:
                    # Vérifier si l1_ratio est applicable
                    if penalty in ['l1', 'l2'] and l1_ratio is not None:
                        continue  # l1_ratio ne doit pas être utilisé avec 'l1' ou 'l2'

                    # Calculer le score moyen pour la configuration actuelle
                    scores = []
                    for train_index, test_index in kf.split(X):
                        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                        # Initialiser le modèle avec les paramètres actuels
                        if penalty == 'elasticnet':
                            model = LogisticRegression(penalty=penalty, C=C, solver=solver, l1_ratio=l1_ratio, 
                                                       max_iter=1000, random_state=42)
                        else:
                            model = LogisticRegression(penalty=penalty, C=C, solver=solver, 
                                                       max_iter=1000, random_state=42)

                        # Entraîner le modèle
                        model.fit(X_train, y_train)

                        # Prédire et calculer le score selon la métrique choisie
                        y_pred = model.predict(X_test)
                        if metric == 'accuracy':
                            score = accuracy_score(y_test, y_pred)
                        elif metric == 'precision':
                            score = precision_score(y_test, y_pred, average=average)
                        elif metric == 'f1':
                            score = f1_score(y_test, y_pred, average=average)
                        elif metric == 'recall':
                            score = recall_score(y_test, y_pred, average=average)
                        elif metric == 'log_loss':
                            y_prob = model.predict_proba(X_test)
                            score = -log_loss(y_test, y_prob)  # Log-loss est une erreur, donc l'inverser pour maximisation
                        elif metric == 'roc_auc':
                            y_prob = model.predict_proba(X_test)[:, 1]
                            score = roc_auc_score(y_test, y_prob)
                        else:
                            raise ValueError(f"Métrique non reconnue: {metric}")
                        
                        scores.append(score)

                    # Calculer le score moyen sur les plis
                    mean_score = np.mean(scores)

                    # Mettre à jour les meilleurs paramètres si le score actuel est meilleur
                    if mean_score > best_score:
                        best_score = mean_score
                        best_params = {
                            'penalty': penalty,
                            'C': C,
                            'solver': solver,
                            'l1_ratio': l1_ratio if penalty == 'elasticnet' else None
                        }

    # Retourner les meilleurs paramètres et le score associé
    return best_params

def forecasting_error(data, target, model, k=5, method='regression', scoring=None, average=None):
    """Calcule l'erreur moyenne de prévision d'un modèle en utilisant la validation croisée
    sur k plis. Gère les métriques de régression et de classification.

    Args:
        data (pd.DataFrame): 
            Le DataFrame contenant les données d'entrée avec toutes les variables explicatives
            et la variable cible.
        target (str): 
            Nom de la colonne de la variable cible dans le DataFrame.
        model (sklearn.base.BaseEstimator): 
            Modèle de régression ou de classification à utiliser pour l'entraînement.
        k (int, optional): 
            Nombre de plis pour la validation croisée. Par défaut à 5.
        method (str, optional): Méthode à utiliser pour le modèle, soit 'regression' pour un modèle de régression, 
            soit 'classification' pour un modèle de classification. Défaut : 'regression'.
        scoring (str, optional): 
            Métrique d'évaluation à utiliser pour mesurer la performance du modèle. 
            Les options incluent 'accuracy', 'precision', 'recall', 'f1', 'log_loss', 'roc_auc' pour 
            les modèles de classification et 'rmse', 'mae', 'mape' pour les modèles de régression.
            Par défaut à None.
        average (str, optional): 
            Méthode d'agrégation pour les métriques multiclasses. 
            Les options incluent 'micro', 'macro', 'weighted'. Nécessaire pour certaines 
            métriques de classification. Par défaut à None.

    Raises:
        ValueError: 
            Si une métrique de régression non reconnue est spécifiée.
        ValueError: 
            Si une métrique de classification non reconnue est spécifiée.
        ValueError: 
            Si la méthode spécifiée n'est ni 'regression' ni 'classification'.
        ValueError: 
            Si le modèle sélectionné ne supporte pas 'predict_proba', nécessaire pour
            calculer 'log_loss' et 'roc_auc'.

    Returns:
        float: 
            Erreur moyenne sur tous les plis de validation croisée.
        list: 
            Liste des erreurs calculées pour chaque pli.
    """
    # Initialisation du K-Fold
    if k == 'all':
        kfold = KFold(n_splits=len(data), shuffle=True)
    else:
        kfold = KFold(n_splits=k, shuffle=True)

    # Choix de la fonction d'erreur selon la méthode et la métrique spécifiée
    if method == 'regression':
        if scoring is None or scoring == 'rmse':
            error_fn = lambda y_true, y_pred: root_mean_squared_error(y_true, y_pred)
        elif scoring == 'mae':
            error_fn = lambda y_true, y_pred: mean_absolute_error(y_true, y_pred)
        elif scoring == 'mape':
            error_fn = lambda y_true, y_pred: mean_absolute_percentage_error(y_true, y_pred)
        else:
            raise ValueError("Métrique de régression non reconnue. Utilisez 'rmse', 'mae' ou 'mape'.")

    elif method == 'classification':
        if average is None:
            average = 'macro'
            
        if scoring is None or scoring == 'accuracy':
            error_fn = lambda y_true, y_pred: accuracy_score(y_true, y_pred)
        elif scoring == 'precision':
            error_fn = lambda y_true, y_pred: precision_score(y_true, y_pred, average=average)
        elif scoring == 'recall':
            error_fn = lambda y_true, y_pred: recall_score(y_true, y_pred, average=average)
        elif scoring == 'f1':
            error_fn = lambda y_true, y_pred: f1_score(y_true, y_pred, average=average)
        elif scoring == 'log_loss':
            error_fn = lambda y_true, y_prob: log_loss(y_true, y_prob)
        elif scoring == 'roc_auc':
            error_fn = lambda y_true, y_prob: roc_auc_score(y_true, y_prob)
        else:
            raise ValueError("Métrique de classification non reconnue. Utilisez 'accuracy', 'precision', 'recall', 'f1', 'log_loss' ou 'roc_auc'.")
    else:
        raise ValueError("La méthode doit être 'regression' ou 'classification'.")
    
    # Validation croisée
    X = data.drop(columns=target, axis=1)
    y = data[target]
    error_list = []
    
    for train_index, test_index in kfold.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Entraînement du modèle
        model.fit(X_train, y_train)

        # Prédictions sur le pli de test
        y_pred = model.predict(X_test)

        # Pour log_loss et roc_auc, utiliser `predict_proba` si disponible
        if method == 'classification' and (scoring == 'log_loss' or scoring == 'roc_auc'):
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)[:, 1]  # Probabilités pour la classe positive
                error = error_fn(y_test, y_prob)
            else:
                raise ValueError("Le modèle sélectionné ne supporte pas 'predict_proba', nécessaire pour calculer 'log_loss' et 'roc_auc'.")
        else:
            error = round(error_fn(y_test, y_pred), 4)

        error_list.append(error)

    # Moyenne de l'erreur sur tous les plis
    mean_error = round(np.mean(error_list), 4)
    
    return mean_error, error_list

def evaluation_model(data, target, model, scores, k=5, average=None):
    """
    Évalue les performances d'un modèle de classification de machine learning par validation croisée.
    Calcul également la matrice de confusion à partir du moèle entraîné sur l'ensemble des données 

    Args:
        data (pd.DataFrame): Les données complètes.
        target (str): Le nom de la colonne cible.
        model (estimator): Le modèle de machine learning à évaluer.
        scores (list of str): Les métriques d'évaluation : 'accuracy', 'precision', 'recall', 'f1'.
        k (int, optional): Nombre de plis pour la validation croisée. Defaults to 5.
        average (str, optional): Moyenne utilisée pour les métriques multiclasses. Defaults to None.

    Returns:
        Un DataFrame contenant les scores et la matrice de confusion.
    """
    # Evaluation des performances
    list_scores, lists = [],[]
    for i in scores:
        score, metric_list = forecasting_error(data=data, target=target,
                                        model=model, k=k, method='classification',
                                        scoring=i, average=average)
        list_scores.append(score)
        lists.append(metric_list)

    df_results = pd.DataFrame({
        'Metric': scores,
        'Score': list_scores,
        'Details': lists
    })
    
    # Matrice de confusion
    X = data.drop(columns=target, axis=1)
    y = data[target]
        
    model.fit(X,y)
    y_pred = model.predict(X)
    conf_matrix = confusion_matrix(y, y_pred)
    
    return df_results, conf_matrix