import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score,accuracy_score, f1_score, recall_score, roc_curve, auc,classification_report
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import MissingIndicator, KNNImputer, SimpleImputer,IterativeImputer
from sklearn.ensemble import IsolationForest

def find_optimal_contamination(data, target_count, tol=1):
    """Trouve la contamination optimale pour obtenir un nombre pr�cis d'individus apr�s nettoyage.

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

def pred_metrics(model,X_train,X_test,y_train,y_test,method="Regression",average='weighted'):
    # Prédictions sur l'ensemble d'apprentissage et de test
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    if method == 'Regression':
        # Calcul des métriques de régression
        train_rmse = round(root_mean_squared_error(y_train, y_pred_train),2)
        test_rmse = round(root_mean_squared_error(y_test, y_pred_test),2)
        diff_rmse = round(abs(train_rmse-test_rmse),2)
    
        train_mae = round(mean_absolute_error(y_train, y_pred_train),2)
        test_mae = round(mean_absolute_error(y_test, y_pred_test),2)
        diff_mae = round(abs(train_mae-test_mae),2)
        
        train_mae_pct = round(mean_absolute_percentage_error(y_train, y_pred_train)*100,2)
        test_mae_pct = round(mean_absolute_percentage_error(y_test, y_pred_test)*100,2)
        diff_mae_pct = round(abs(train_mae_pct-test_mae_pct),2)
    
        train_r2 = round(r2_score(y_train, y_pred_train)*100,2)
        test_r2 = round(r2_score(y_test, y_pred_test)*100,2)
        diff_r2 = round(abs(train_r2-test_r2),2)
        
        metrics = pd.DataFrame({
            'RMSE_apprentissage': [train_rmse],
            'RMSE_validation': [test_rmse],
            'RMSE_diff_apprentissage_validation': [diff_rmse],
            'MAE_apprentissage': [train_mae],
            'MAE_validation': [test_mae],
            'MAE_diff_apprentissage_validation': [diff_mae],
            'MAE_pct_apprentissage': [train_mae_pct],
            'MAE_pct_validation': [test_mae_pct],
            'MAE_pct_diff_apprentissage_validation': [diff_mae_pct],
            'R2_apprentissage': [train_r2],
            'R2_validation': [test_r2],
            'R2_diff_apprentissage_validation': [diff_r2]
            }, index=[0])

        
    elif method == 'Classification':
        # Calcul des métriques de classification
        train_accuracy = round(accuracy_score(y_train, y_pred_train) * 100, 2)
        test_accuracy = round(accuracy_score(y_test, y_pred_test) * 100, 2)
        diff_accuracy = round(abs(train_accuracy - test_accuracy), 2)

        train_precision = round(precision_score(y_train, y_pred_train, average=average) * 100, 2)
        test_precision = round(precision_score(y_test, y_pred_test, average=average) * 100, 2)
        diff_precision = round(abs(train_precision - test_precision), 2)

        train_recall = round(recall_score(y_train, y_pred_train, average=average) * 100, 2)
        test_recall = round(recall_score(y_test, y_pred_test, average=average) * 100, 2)
        diff_recall = round(abs(train_recall - test_recall), 2)

        train_f1 = round(f1_score(y_train, y_pred_train, average=average) * 100, 2)
        test_f1 = round(f1_score(y_test, y_pred_test, average=average) * 100, 2)
        diff_f1 = round(abs(train_f1 - test_f1), 2)
        
        metrics = pd.DataFrame({
            'Accuracy_apprentissage': [train_accuracy],
            'Accuracy_validation': [test_accuracy],
            'Accuracy_diff_apprentissage_validation': [diff_accuracy],
            'Precision_apprentissage': [train_precision],
            'Precision_validation': [test_precision],
            'Precision_diff_apprentissage_validation': [diff_precision],
            'Recall_apprentissage': [train_recall],
            'Recall_validation': [test_recall],
            'Recall_diff_apprentissage_validation': [diff_recall],
            'F1_apprentissage': [train_f1],
            'F1_validation': [test_f1],
            'F1_diff_apprentissage_validation': [diff_f1]}, index=[0]) 
        
    else:
        print("La méthode choisie doit être de 'Regression' ou de 'Classification' seulement.")

def CV_parameters_classif(model,hidden_layer_sizes, activation, alpha, learning_rate, X_train,X_test,y_train,y_test,metric='neg_mean_absolute_error',average='weighted',selected_model=3):
    # Définir les paramètres à optimiser et leurs valeurs possibles
    param_grid = {
    'hidden_layer_sizes': hidden_layer_sizes,
    'activation': activation,
    'alpha': alpha,
    'learning_rate_init': learning_rate
}

    # Utiliser GridSearchCV
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring=metric,n_jobs=8)
    grid_result = grid.fit(X_train, y_train)

    # Créer une DataFrame à partir des résultats de la grille
    results_df = pd.DataFrame({
        'Hidden Layers': grid_result.cv_results_['param_hidden_layer_sizes'],
        'Activation': grid_result.cv_results_['param_activation'],
        'Alpha': grid_result.cv_results_['param_alpha'],
        'Learning_rate' : grid_result.cv_results_['param_learning_rate_init'],        
        'Rank': grid_result.cv_results_['rank_test_score'],
        'Std Test Score': grid_result.cv_results_['std_test_score']
    })

    # Trier les résultats par rang et on ne garde qu'un certain nombre de modèles
    results_df['Std Test Score'] = round(results_df['Std Test Score'], 2)
    results_df = results_df.sort_values(by='Rank', ascending=True).head(selected_model)
    
    # DataFrame pour stocker les hyperparamètres métriques des modèles sélectionnés
    results_df_classif = pd.DataFrame()
    
    # Boucle sur les meilleurs modèles
    for idx, row in results_df.iterrows():
        # Construire le modèle avec les paramètres du modèle actuel
        model = MLPClassifier(
            hidden_layer_sizes=row['Hidden Layers'],
            activation=row['Activation'],
            alpha=row['Alpha'],
            solver='adam',
            learning_rate_init=float(row['Learning_rate']),
            max_iter = 1000)
        
        model.fit(X_train, y_train)
        
        # Calcul des métriques
        metrics = pred_metrics(model, X_train, X_test, y_train, y_test, method='Classification',average=average)
        
        # Ajouter les hyperparamètres au DataFrame des métriques
        metrics['Hidden Layers'] = [row['Hidden Layers']]
        metrics['Activation'] = [row['Activation']]
        metrics['Alpha'] = [row['Alpha']]
        metrics['Learning_rate'] = [row['Learning_rate']]
        metrics['Rank'] = [row['Rank']]
        
        # Ajouter les résultats au DataFrame général
        results_df_classif = pd.concat([results_df_classif, metrics], ignore_index=True)
        
    return results_df_classif


