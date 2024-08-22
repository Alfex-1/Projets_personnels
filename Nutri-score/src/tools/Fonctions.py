import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score,accuracy_score, f1_score, recall_score, roc_curve, auc,classification_report
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import MissingIndicator, KNNImputer, SimpleImputer,IterativeImputer
from sklearn.ensemble import IsolationForest,RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier

def find_optimal_contamination(data, target_count, tol=1):
    """Trouve la contamination optimale pour obtenir un nombre pr�cis d'individus apr�s nettoyage.

    Args:
        data (DataFrame) : DataFrame contenant les donn�es � nettoyer.
        target_count (int) : Nombre souhait� d'individus apr�s nettoyage.
        tol (int or float, optional) : Tol�rance pour le nombre d'individus (par d�faut 1).

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

def xgboost_models(model, nb_estimators, learn_rate, l1, l2, gamma, max_depth, X_train,y_train,y_test,metric='accuracy', average='weighted', selected_models=3,cv=5):
    """Fonction qui effectue une validation croisée pour déterminer la meilleurs combinaisons d'hyperparamètres initialisés pour optimiser le développement d'un modèle XGBoost.

    Args:
        model (XGBClassifier): le modèle XGBoost instancié.
        nb_estimators (int): nombre d'estimateurs.
        learn_rate (int or float): taux d'apprentissage (valeurs possibles : de 0 à 1).
        l1 (int or float): coefficient de la pénalité L1.
        l2 (int or float): coefficient de la pénalité L2.
        gamma (int or float): coefficient pour la simplification du modèle.
        max_depth (int): profondeur maximale de chaque estimateur.
        metric (str, optional): Métrique pour  l'évaluation des modèles. Defaults to 'accuracy'.
        average (str, optional): Type de calcul du moyenne effectué sur les données. La valeur par défaut est 'weighted'.
        selected_models (int, optional): Nombre de modèle sélectionné pour le top. La valeur par défaut est 3 (top 3 des meilleurs modèles).
        X_train (DataFrame, optional): Valeurs d'entraînement des variables explicatives. La DataFrame par défaut est X_train.
        y_train (Series, optional): Valeurs d'entraînement de la variable cible. La Série par défaut est y_train.
        X_test (DataFrame, optional): Valeurs de test des variables explicatives. La DataFrame par défaut est X_test.
        y_test (Series, optional): Valeurs de test de la variable cible. La Série par défaut est y_test.
        cv (int,optional) : Nombre de Folds. La valeur par défaut est 5.

    Returns:
        DataFrame : Tableau affichant les performances des k meilleurs modèles sur la base de train et celle de test.
    """
    param_grid = {
        'n_estimators': nb_estimators,
        'learning_rate': learn_rate,
        'reg_alpha': l1,
        'reg_lambda': l2,
        'gamma': gamma,
        'max_depth': max_depth
    }

    grid = GridSearchCV(estimator=model, param_grid=param_grid,
                        cv=cv, scoring=metric, n_jobs=1)
    grid_result = grid.fit(X_train, y_train)

    # Conserver les données dans un DataFrame
    top_models_details = pd.DataFrame({
        'Nb_estimators': grid_result.cv_results_['param_n_estimators'],
        'Learning_Rate': grid_result.cv_results_['param_learning_rate'],
        'L1': grid_result.cv_results_['param_reg_alpha'],
        'L2': grid_result.cv_results_['param_reg_lambda'],
        'Gamma': grid_result.cv_results_['param_gamma'],
        'Max_depth': grid_result.cv_results_['param_max_depth'],
        'Rank': grid_result.cv_results_['rank_test_score'],
        'Std Test Score': grid_result.cv_results_['std_test_score']
    })

    # Trier les résultats par rang et on ne garde qu'un certain nombre de modèles
    top_models_details = top_models_details.sort_values(
        by='Rank', ascending=True).head(selected_models)

    metrics = pd.DataFrame(columns=["Nb_estimators", "Learning_Rate", "L1", "L2",
                                    "Gamma", "Max_depth",
                                    "Accuracy", "Precision", "F1", "Recall"])

    # Boucle sur les meilleurs modèles
    for idx, row in top_models_details.iterrows():
        # Construire le modèle avec les paramètres du modèle actuel
        model = XGBClassifier(
            n_estimators=int(row['Nb_estimators']),
            learning_rate=float(row['Learning_Rate']),
            reg_alpha=float(row['L1']),
            reg_lambda=float(row['L2']),
            gamma=float(row['Gamma']),
            depth=int(row['Max_depth']),
            random_state=42,
            verbose=-1
        )

        # Entraîner le modèle
        model.fit(X_train, y_train)

        # Prédictions sur l'ensemble de test
        y_pred = model.predict(X_test)

        # Calculer et stocker les métriques
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average=average)
        f1 = f1_score(y_test, y_pred, average=average)
        recall = recall_score(y_test, y_pred, average=average)

        # Ajouter les résultats à la DataFrame
        metrics = metrics._append({
            "Nb_estimators": int(row['Nb_estimators']),
            "Learning_Rate": float(row['Learning_Rate']),
            "L1": float(row['L1']),
            "L2": float(row['L2']),
            "Gamma": float(row['Gamma']),
            "Max_depth": int(row['Max_depth']),
            "Accuracy": accuracy,
            "Precision": precision,
            "F1": f1,
            "Recall": recall
        }, ignore_index=True)

        # Prédictions sur l'ensemble de test
        # Probabilité des classes positives pour la courbe ROC
        y_pred_prob = model.predict_proba(X_test)

        plt.figure(figsize=(8, 8))

        # Plot ROC curve pour chaque classe
        for class_index in range(model.n_classes_):
            fpr, tpr, _ = roc_curve(
                y_test == class_index, y_pred_prob[:, class_index])
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, lw=2,
                     label=f'Classe {class_index} (AUC = {roc_auc:.2f})')

        # Plot diagonal line
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('Taux de faux positifs')
        plt.ylabel('Taux de vrais positifs')
        plt.title(f'Courbes ROC - Modèle {idx}')
        plt.legend(loc="lower right")
        plt.show()

    models_results = pd.merge(top_models_details, metrics, on=[
                              'Nb_estimators', 'Learning_Rate', 'L1', 'L2', 'Gamma', 'Max_depth'])

    # Améliorer l'affichage des métriques
    models_results[['Accuracy', 'Precision', 'F1', 'Recall']] *= 100
    models_results[['Accuracy', 'Precision', 'F1', 'Recall']] = models_results[[
        'Accuracy', 'Precision', 'F1', 'Recall']].round(2)

    return models_results

def adaboost_models(model,nb_estimators, learn_rate, max_depth_RF,X_train, y_train,X_test,y_test,metric='accuracy', average="weighted", selected_models=3, cv=5):
    """Fonction qui effectue une validation croisée pour déterminer la meilleurs combinaisons d'hyperparamètres initialisés pour optimiser le développement d'un modèle Adaboost.

    Args:
        model (AdaBoostClassifier): le modèle Adaboost instancié.
        nb_estimators (int): nombre d'estimateurs.
        learn_rate (int or float): taux d'apprentissage (valeurs possibles : de 0 à 1).
        max_depth (int): profondeur maximale de chaque estimateur.
        metric (str, optional): Métrique pour  l'évaluation des modèles. Defaults to 'accuracy'.
        average (str, optional): Type de calcul du moyenne effectué sur les données. La valeur par défaut est 'weighted'.
        selected_models (int, optional): Nombre de modèle sélectionné pour le top. La valeur par défaut est 3 (top 3 des meilleurs modèles).
        X_train (DataFrame, optional): Valeurs d'entraînement des variables explicatives. La DataFrame par défaut est X_train.
        y_train (Series, optional): Valeurs d'entraînement de la variable cible. La Série par défaut est y_train.
        X_test (DataFrame, optional): Valeurs de test des variables explicatives. La DataFrame par défaut est X_test.
        y_test (Series, optional): Valeurs de test de la variable cible. La Série par défaut est y_test.
        cv (int,optional) : Nombre de Folds. La valeur par défaut est 5.

    Returns:
        DataFrame : Tableau affichant les performances des k meilleurs modèles sur la base de train et celle de test.
    """
    results = []

    for depth in max_depth_RF:
        base_estimator = RandomForestClassifier(
            max_depth=depth, random_state=42)
        model = AdaBoostClassifier(estimator=base_estimator, random_state=42)

        param_grid = {
            'n_estimators': nb_estimators,
            'learning_rate': learn_rate
        }

        grid = GridSearchCV(
            estimator=model, param_grid=param_grid, cv=5, scoring=metric, n_jobs=8)
        grid_result = grid.fit(X_train, y_train)

        # Ajouter les résultats à la liste
        for i in range(len(grid_result.cv_results_['rank_test_score'])):
            results.append({
                'Nb_estimators': grid_result.cv_results_['param_n_estimators'][i],
                'Learning_Rate': grid_result.cv_results_['param_learning_rate'][i],
                'Max_Depth_RF': depth,
                'Rank': grid_result.cv_results_['rank_test_score'][i],
                'Std_Test_Score': grid_result.cv_results_['std_test_score'][i]
            })

    # Créer une DataFrame à partir des résultats
    top_models_details = pd.DataFrame(results)

    # Trier les résultats par rang et on ne garde qu'un certian nombre de modèle
    top_models_details = top_models_details.sort_values(
        by='Rank', ascending=True).head(selected_models)

    metrics = pd.DataFrame(columns=["Nb_estimators", "Learning_Rate",
                           "Max_Depth_RF", "Accuracy", "Precision", "F1", "Recall"])

    # Boucle sur les 3 meilleurs modèles
    for idx, row in top_models_details.iterrows():
        # Construire le modèle avec les paramètres du modèle actuel
        model = AdaBoostClassifier(
            estimator=RandomForestClassifier(
                max_depth=int(row['Max_Depth_RF']), random_state=42),
            n_estimators=int(row['Nb_estimators']),
            learning_rate=float(row['Learning_Rate']),
            random_state=42)

        # Entraîner le modèle
        model.fit(X_train, y_train)

        # Prédictions sur l'ensemble de test
        y_pred_prob = model.predict(X_test)

        # Sélectionner la classe prédite (celle avec la probabilité la plus élevée)
        y_pred = y_pred_prob

        # Calculer et stocker les métriques
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average=average)
        f1 = f1_score(y_test, y_pred, average=average)
        recall = recall_score(y_test, y_pred, average=average)

        # Ajouter les résultats à la DataFrame
        metrics = metrics._append({
            "Nb_estimators": int(row['Nb_estimators']),
            "Learning_Rate": float(row['Learning_Rate']),
            "Max_Depth_RF": int(row['Max_Depth_RF']),
            "Accuracy": accuracy,
            "Precision": precision,
            "F1": f1,
            "Recall": recall
        }, ignore_index=True)

        # Prédictions sur l'ensemble de test
        # Probabilité des classes positives pour la courbe ROC
        y_pred_prob = model.predict_proba(X_test)

        plt.figure(figsize=(8, 8))

        # Plot ROC curve pour chaque classe
        for class_index in range(model.n_classes_):
            fpr, tpr, _ = roc_curve(
                y_test == class_index, y_pred_prob[:, class_index])
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, lw=2,
                     label=f'Classe {class_index} (AUC = {roc_auc:.2f})')

        # Plot diagonal line
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('Taux de faux positifs')
        plt.ylabel('Taux de vrais positifs')
        plt.title(f'Courbes ROC - Modèle {idx}')
        plt.legend(loc="lower right")
        plt.show()

    models_results = pd.merge(top_models_details, metrics, on=[
                              'Nb_estimators', 'Learning_Rate', 'Max_Depth_RF'])

    # Améliorer l'affichage des métriques
    models_results[['Accuracy', 'Precision', 'F1', 'Recall']] *= 100
    models_results[['Accuracy', 'Precision', 'F1', 'Recall']] = models_results[[
        'Accuracy', 'Precision', 'F1', 'Recall']].round(2)

    return models_results

def catboost_models(model,nb_estimators, learn_rate, l2, max_depth, X_train, y_train, X_test, y_test,metric='accuracy', average="weighted", selected_models=3, cv=5):
    """Fonction qui effectue une validation croisée pour déterminer la meilleurs combinaisons d'hyperparamètres initialisés pour optimiser le développement d'un modèle catboost.

    Args:
        model (CatBoostClassifier): le modèle Catboost instancié.
        nb_estimators (int): nombre d'estimateurs.
        learn_rate (int or float): taux d'apprentissage (valeurs possibles : de 0 à 1).
        l2 (int or float): coefficient de la pénalité L2.
        max_depth (int): profondeur maximale de chaque estimateur.
        metric (str, optional): Métrique pour  l'évaluation des modèles. Defaults to 'accuracy'.
        average (str, optional): Type de calcul du moyenne effectué sur les données. La valeur par défaut est 'weighted'.
        selected_models (int, optional): Nombre de modèle sélectionné pour le top. La valeur par défaut est 3 (top 3 des meilleurs modèles).
        X_train (DataFrame, optional): Valeurs d'entraînement des variables explicatives. La DataFrame par défaut est X_train.
        y_train (Series, optional): Valeurs d'entraînement de la variable cible. La Série par défaut est y_train.
        X_test (DataFrame, optional): Valeurs de test des variables explicatives. La DataFrame par défaut est X_test.
        y_test (Series, optional): Valeurs de test de la variable cible. La Série par défaut est y_test.
        cv (int,optional) : Nombre de Folds. La valeur par défaut est 5.

    Returns:
        DataFrame : Tableau affichant les performances des k meilleurs modèles sur la base de train et celle de test.
    """
    param_grid = {
        'iterations': nb_estimators,
        'learning_rate': learn_rate,
        'l2_leaf_reg': l2,
        'depth': max_depth
    }

    grid = GridSearchCV(estimator=model, param_grid=param_grid,
                        cv=cv, scoring=metric, n_jobs=8)
    grid_result = grid.fit(X_train, y_train)

    # Conserver les données dans un DataFrame
    top_models_details = pd.DataFrame({
        'Nb_estimators': grid_result.cv_results_['param_iterations'],
        'Learning_Rate': grid_result.cv_results_['param_learning_rate'],
        'L2': grid_result.cv_results_['param_l2_leaf_reg'],
        'Max_depth': grid_result.cv_results_['param_depth'],
        'Rank': grid_result.cv_results_['rank_test_score'],
        'Std Test Score': grid_result.cv_results_['std_test_score']
    })

    # Trier les résultats par rang et on ne garde qu'un certain nombre de modèles
    top_models_details = top_models_details.sort_values(
        by='Rank', ascending=True).head(selected_models)

    metrics = pd.DataFrame(columns=["Nb_estimators", "Learning_Rate",
                           "L2", "Max_depth", "Accuracy", "Precision", "F1", "Recall"])

    # Boucle sur les meilleurs modèles
    for idx, row in top_models_details.iterrows():
        # Construire le modèle avec les paramètres du modèle actuel
        model = CatBoostClassifier(
            iterations=int(row['Nb_estimators']),
            learning_rate=float(row['Learning_Rate']),
            l2_leaf_reg=float(row['L2']),
            depth=int(row['Max_depth']),
            random_state=42,
            verbose=False
        )

        # Entraîner le modèle
        model.fit(X_train, y_train)

        # Prédictions sur l'ensemble de test
        y_pred = model.predict(X_test)

        # Calculer et stocker les métriques
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average=average)
        f1 = f1_score(y_test, y_pred, average=average)
        recall = recall_score(y_test, y_pred, average=average)

        # Ajouter les résultats à la DataFrame
        metrics = metrics._append({
            "Nb_estimators": int(row['Nb_estimators']),
            "Learning_Rate": float(row['Learning_Rate']),
            "L2": float(row['L2']),
            "Max_depth": int(row['Max_depth']),
            "Accuracy": accuracy,
            "Precision": precision,
            "F1": f1,
            "Recall": recall
        }, ignore_index=True)

        # Prédictions sur l'ensemble de test
        # Probabilité des classes positives pour la courbe ROC
        y_pred_prob = model.predict_proba(X_test)

        plt.figure(figsize=(8, 8))

        # Plot ROC curve pour chaque classe
        num_classes = len(np.unique(y_train))
        for class_index in range(num_classes):
            fpr, tpr, _ = roc_curve(
                y_test == class_index, y_pred_prob[:, class_index])
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, lw=2,
                     label=f'Classe {class_index} (AUC = {roc_auc:.2f})')

        # Plot diagonal line
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('Taux de faux positifs')
        plt.ylabel('Taux de vrais positifs')
        plt.title(f'Courbes ROC - Modèle {idx}')
        plt.legend(loc="lower right")
        plt.show()

    models_results = pd.merge(top_models_details, metrics, on=[
                              'Nb_estimators', 'Learning_Rate', 'L2', 'Max_depth'])

    # Améliorer l'affichage des métriques
    models_results[['Accuracy', 'Precision', 'F1', 'Recall']] *= 100
    models_results[['Accuracy', 'Precision', 'F1', 'Recall']] = models_results[[
        'Accuracy', 'Precision', 'F1', 'Recall']].round(2)

    return models_results

def lightgbm_models(model,nb_estimators, learn_rate, l1, l2, max_depth, X_train, y_train, X_test, y_test,metric='accuracy', average='weighted', selected_models=3, cv=5):
    """Fonction qui effectue une validation croisée pour déterminer la meilleurs combinaisons d'hyperparamètres initialisés pour optimiser le développement d'un modèle LightGBM.

    Args:
        model (LGBMClassifier): le modèle LightGBM instancié.
        nb_estimators (int): nombre d'estimateurs.
        learn_rate (int or float): taux d'apprentissage (valeurs possibles : de 0 à 1).
        l1 (int or float): coefficient de la pénalité L1.
        l2 (int or float): coefficient de la pénalité L2.
        max_depth (int): profondeur maximale de chaque estimateur.
        metric (str, optional): Métrique pour  l'évaluation des modèles. Defaults to 'accuracy'.
        average (str, optional): Type de calcul du moyenne effectué sur les données. La valeur par défaut est 'weighted'.
        selected_models (int, optional): Nombre de modèle sélectionné pour le top. La valeur par défaut est 3 (top 3 des meilleurs modèles).
        X_train (DataFrame, optional): Valeurs d'entraînement des variables explicatives. La DataFrame par défaut est X_train.
        y_train (Series, optional): Valeurs d'entraînement de la variable cible. La Série par défaut est y_train.
        X_test (DataFrame, optional): Valeurs de test des variables explicatives. La DataFrame par défaut est X_test.
        y_test (Series, optional): Valeurs de test de la variable cible. La Série par défaut est y_test.
        cv (int,optional) : Nombre de Folds. La valeur par défaut est 5.

    Returns:
        DataFrame : Tableau affichant les performances des k meilleurs modèles sur la base de train et celle de test.
    """
    param_grid = {
        'n_estimators': nb_estimators,
        'learning_rate': learn_rate,
        'reg_alpha': l1,
        'reg_lambda': l2,
        'max_depth': max_depth
    }

    grid = GridSearchCV(estimator=model, param_grid=param_grid,
                        cv=cv, scoring=metric, n_jobs=8)
    grid_result = grid.fit(X_train, y_train)

    # Conserver les données dans un DataFrame
    top_models_details = pd.DataFrame({
        'Nb_estimators': grid_result.cv_results_['param_n_estimators'],
        'Learning_Rate': grid_result.cv_results_['param_learning_rate'],
        'L1': grid_result.cv_results_['param_reg_alpha'],
        'L2': grid_result.cv_results_['param_reg_lambda'],
        'Max_depth': grid_result.cv_results_['param_max_depth'],
        'Rank': grid_result.cv_results_['rank_test_score'],
        'Std Test Score': grid_result.cv_results_['std_test_score']
    })

    # Trier les résultats par rang et on ne garde qu'un certain nombre de modèles
    top_models_details = top_models_details.sort_values(
        by='Rank', ascending=True).head(selected_models)

    metrics = pd.DataFrame(columns=["Nb_estimators", "Learning_Rate", "L1",
                           "L2", "Max_depth", "Accuracy", "Precision", "F1", "Recall"])

    # Boucle sur les meilleurs modèles
    for idx, row in top_models_details.iterrows():
        # Construire le modèle avec les paramètres du modèle actuel
        model = lgb.LGBMClassifier(
            n_estimators=int(row['Nb_estimators']),
            learning_rate=float(row['Learning_Rate']),
            reg_alpha=float(row['L1']),
            reg_lambda=float(row['L2']),
            depth=int(row['Max_depth']),
            random_state=42,
            verbose=-1
        )

        # Entraîner le modèle
        model.fit(X_train, y_train)

        # Prédictions sur l'ensemble de test
        y_pred = model.predict(X_test)

        # Calculer et stocker les métriques
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average=average)
        f1 = f1_score(y_test, y_pred, average=average)
        recall = recall_score(y_test, y_pred, average=average)

        # Ajouter les résultats à la DataFrame
        metrics = metrics._append({
            "Nb_estimators": int(row['Nb_estimators']),
            "Learning_Rate": float(row['Learning_Rate']),
            "L1": float(row['L1']),
            "L2": float(row['L2']),
            "Max_depth": int(row['Max_depth']),
            "Accuracy": accuracy,
            "Precision": precision,
            "F1": f1,
            "Recall": recall
        }, ignore_index=True)

        # Prédictions sur l'ensemble de test
        # Probabilité des classes positives pour la courbe ROC
        y_pred_prob = model.predict_proba(X_test)

        plt.figure(figsize=(8, 8))

        # Plot ROC curve pour chaque classe
        for class_index in range(model.n_classes_):
            fpr, tpr, _ = roc_curve(
                y_test == class_index, y_pred_prob[:, class_index])
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, lw=2,
                     label=f'Classe {class_index} (AUC = {roc_auc:.2f})')

        # Plot diagonal line
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('Taux de faux positifs')
        plt.ylabel('Taux de vrais positifs')
        plt.title(f'Courbes ROC - Modèle {idx}')
        plt.legend(loc="lower right")
        plt.show()

    models_results = pd.merge(top_models_details, metrics, on=[
                              'Nb_estimators', 'Learning_Rate', 'L1', 'L2', 'Max_depth'])

    # Améliorer l'affichage des métriques
    models_results[['Accuracy', 'Precision', 'F1', 'Recall']] *= 100
    models_results[['Accuracy', 'Precision', 'F1', 'Recall']] = models_results[[
        'Accuracy', 'Precision', 'F1', 'Recall']].round(2)

    return models_results

def model_opti(model,n_estimators, learning_rate,max_depth, X_train, y_train, X_test, y_test,l1=0, l2=0, gamma=0, average="weighted"):
    """Entraînement d'un modèle de boosting à partir des hyperparamètres choisit.

    Args:
        model (str) : Le nom du modèle choisit. Valeurs possibles : "xgb", "ada", "cat", "lgb".
        n_estimators (int): Nombre d'estimateurs choisit
        learning_rate (int or float): Taux d'apprentissage choisit.
        l1 (int or float): Coefficient de la pénalité L1 choisit. La valeur par défaut est 0.
        l2 (int or float): Le coeffcient de la pénalité L2 choisit. La valeur par défaut est 0.
        gamma (int or float): Le coefficient pour la simplicité du modèle choisit. La valeur par défaut est 0.
        max_depth (int): La profondeur de chaque estimateur choisit.
        average (str, optional): Le type de calcul de moyenne choisit. La valeur par défaut est "weighted".
        X_train (DataFrame, optional): Valeurs d'entraînement des variables explicatives. La DataFrame par défaut est X_train.
        y_train (Series, optional): Valeurs d'entraînement de la variable cible. La Série par défaut est y_train.
        X_test (DataFrame, optional): Valeurs de test des variables explicatives. La DataFrame par défaut est X_test.
        y_test (Series, optional): Valeurs de test de la variable cible. La Série par défaut est y_test.
        X (DataFrame, optional): L'ensemble des valeurs des variables explicatives. La DataFrame par défaut est X.

    Returns:
        DataFrame and model : Modèle sélectionné et tableau affichant ses performances sur la base de train et de test.
    """
    # Construire le modèle choisit avec les hyperparamètres sélectionnés
    if model == "xgb":
        selected_model = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        reg_alpha=l1,
        reg_lambda=l2,
        gamma=gamma,
        max_depth=max_depth,
        random_state=42)
    
    elif model == "ada":
            selected_model = AdaBoostClassifier(estimator=RandomForestClassifier(max_depth=max_depth, random_state=42),
                                                n_estimators=n_estimators,
                                                learning_rate=learning_rate,
                                                random_state=42)
    
    elif model == "cat":
            selected_model = CatBoostClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                l2_leaf_reg=l2,
                depth=max_depth,
                random_state=42)
    elif model == "lgb":
            selected_model = lgb.LGBMClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                reg_alpha=l1,
                reg_lambda=l2,
                max_depth=max_depth,
                random_state=42,
                verbose=-1)
    else:
        print("Veuillez choisir un nom de modèle parmi 'xgb', 'ada', 'cat' et 'lgb'")       

    # Entraîner le modèle sur les données d'entraînement
    selected_model.fit(X_train, y_train)

    # Utiliser le modèle pour faire des prédictions sur la base d'apprentissage et de test
    y_train_pred = selected_model.predict(X_train)
    y_test_pred = selected_model.predict(X_test)

    # Calculer les métriques pour la base d'apprentissage
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred, average=average)
    train_f1 = f1_score(y_train, y_train_pred, average=average)
    train_recall = recall_score(y_train, y_train_pred, average=average)

    # Calculer les métriques pour la base de test
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average=average)
    test_f1 = f1_score(y_test, y_test_pred, average=average)
    test_recall = recall_score(y_test, y_test_pred, average=average)

    # Créer une DataFrame pour stocker les résultats
    metrics_df = pd.DataFrame({
        'Base': ['Apprentissage', 'Validation'],
        'Accuracy': [train_accuracy, test_accuracy],
        'Precision': [train_precision, test_precision],
        'F1': [train_f1, test_f1],
        'Recall': [train_recall, test_recall]
    })

    metrics_df[['Accuracy', 'Precision', 'F1', 'Recall']] *= 100
    metrics_df[['Accuracy', 'Precision', 'F1', 'Recall']] = metrics_df[[
        'Accuracy', 'Precision', 'F1', 'Recall']].round(1)

    # Récupérer l'importance des variables
    feature_importance = selected_model.feature_importances_

    # Trier les variables par ordre d'importance
    feature_names = X_train.columns
    sorted_idx = feature_importance.argsort()[::-1]
    sorted_feature_names = feature_names[sorted_idx]

    # Visualisation de l'importance des variables
    plt.figure(figsize=(10, 6))
    barplot = sns.barplot(
        x=feature_importance[sorted_idx], y=sorted_feature_names, palette="viridis")
    plt.title("Importance des variables")
    plt.xlabel("Importance")
    plt.ylabel("Variables")

    # Ajouter des étiquettes de données
    for i, val in enumerate(feature_importance[sorted_idx]):
        barplot.text(val, i, f'{val:.2f}', va='center')

    plt.show()

    return metrics_df,selected_model