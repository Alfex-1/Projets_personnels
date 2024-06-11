# Packages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_curve, auc
import seaborn as sns
from sklearn.ensemble import AdaBoostClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier

# Fixer la graine pour la reproductibilité
np.random.seed(42)

# Importation de la base
chemin_fichier = r"\\172.20.3.5\vol_modelisation_001\modelisation\MOD_DONNEES_SATELLITAIRES\Stage\Alex\Projets_personnels\Nutri-score\Scripts\6Data_no_miss_balanced.csv"

df = pd.read_csv(chemin_fichier, sep=',')

# Vérification
df.info()
print(df.isnull().sum())
df.describe()

def encoding_all_data(data,reverse=False):
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
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

df, infos = encoding_all_data(df, reverse=True)

# Division des données
X = df.drop('NutriScore', axis=1)  # Variables prédictives
y = df['NutriScore']  # Variable cible

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Initialiser tous les modèles
xg, ada, cat, lg = XGBClassifier(random_state=42), AdaBoostClassifier(random_state=42), CatBoostClassifier(
    random_state=42, verbose=False), lgb.LGBMClassifier(random_state=42, verbosity=-1)

models = [xg, ada, cat, lg]

models_dict = {'XGBClassifier': xg, 'AdaBoostClassifier': ada,
               'CatBoostClassifier': cat, 'LGBMClassifier': lg}

def compare_boosting(models, nb_estimators, learn_rate, gamma, l1, l2, max_depth,
                     metric, cv=5, average_metric='macro',
                     X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test):

    # Dictionnaire pour stocker les meilleurs paramètres de chaque modèle
    best_params_per_model = {}

    for model in models:
        # Définir les paramètres à optimiser et leurs valeurs possibles
        if model is models[0]:
            print('XGBoost en préparation...')
            param_grid = {'n_estimators': nb_estimators,
                          'learning_rate': learn_rate,
                          'max_depth': max_depth,
                          'gamma': gamma,
                          'alpha': l1,
                          'lambda': l2}

        elif model is models[1]:
            print('AdaBoost en préparation...')
            max_depth_RF = max_depth.tolist().append(None)
            param_grid = {'estimator': [RandomForestClassifier(max_depth=max_depth_RF)],
                          'n_estimators': nb_estimators,
                          'learning_rate': learn_rate}

        elif model is models[2]:
            print('CatBoost en préparation...')
            param_grid = {'iterations': nb_estimators,
                          'learning_rate': learn_rate,
                          'l2_leaf_reg': l2}

        elif model is models[3]:
            print('LightGBM en préparation...')
            param_grid = {'n_estimators': nb_estimators,
                          'learning_rate': learn_rate,
                          'max_depth': max_depth,
                          'reg_alpha': l1,
                          'reg_lambda': l2}

        # Utiliser GridSearchCV
        grid = GridSearchCV(
            estimator=model, param_grid=param_grid, cv=cv, scoring=metric, n_jobs=16)
        grid_result = grid.fit(X_train, y_train)

        # Obtenir les meilleurs paramètres
        best_params_per_model[model.__class__.__name__] = grid_result.best_params_

    # Créer une liste pour stocker les données des métriques pour chaque modèle
    metrics_data = []

    # Boucle sur les meilleurs paramètres de chaque modèle
    for model_name, model_params in best_params_per_model.items():
        # Obtenir le modèle avec les meilleurs paramètres
        model = [mdl for mdl in models if mdl.__class__.__name__ == model_name][0]

        # Entraîner le modèle avec les meilleurs paramètres sur l'ensemble d'entraînement
        model.set_params(**model_params)
        model.fit(X_train, y_train)

        # Prédictions sur l'ensemble d'entraînement et de test
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Calculer les métriques pour l'ensemble d'entraînement
        accuracy_train = round(accuracy_score(y_train, y_pred_train), 2)
        recall_train = round(recall_score(
            y_train, y_pred_train, average=average_metric), 2)
        f1_train = round(f1_score(y_train, y_pred_train,
                         average=average_metric), 2)

        # Calculer les métriques pour l'ensemble de test
        accuracy_test = round(accuracy_score(y_test, y_pred_test), 2)
        recall_test = round(recall_score(
            y_test, y_pred_test, average=average_metric), 2)
        f1_test = round(f1_score(y_test, y_pred_test,
                        average=average_metric), 2)

        # Différence entre apprentissage et validation
        accuracy_diff = abs(accuracy_train - accuracy_test)
        recall_diff = abs(recall_train - recall_test)
        f1_diff = abs(f1_train - f1_test)

        # Stocker les métriques dans un dictionnaire
        metrics_dict = {
            'Modèle': model_name,
            'Accuracy (train)': accuracy_train,
            'Recall (train)': recall_train,
            'F1-score (train)': f1_train,
            'Accuracy (test)': accuracy_test,
            'Recall (test)': recall_test,
            'F1-score (test)': f1_test,
            'Accuracy (train-test)': accuracy_diff,
            'Recall (train-test)': recall_diff,
            'F1-score (train-test)': f1_diff}

        # Ajouter les données des métriques à la liste
        metrics_data.append(metrics_dict)

        # Prédictions sur l'ensemble de test
        y_pred_prob = model.predict_proba(X_test)

        plt.figure(figsize=(8, 8))

        # Plot ROC curve pour chaque classe
        num_classes = len(np.unique(y_test))
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
        plt.title(f'Courbes ROC - {model_name}')
        plt.legend(loc="lower right")
        plt.show()

        # Mettre en forme les données
        metrics_df = pd.DataFrame(metrics_data)
        best_params_per_model = pd.DataFrame(best_params_per_model)

    # Accéder à l'estimateur du modèle AdaBoostClassifier
    estimator = best_params_per_model.loc['estimator', 'AdaBoostClassifier']

    # Vérifier si l'estimateur est un RandomForestClassifier
    if isinstance(estimator, RandomForestClassifier):
        # Extraire la profondeur maximale
        max_depth_rf = estimator.max_depth
        print("Profondeur maximale du RandomForestClassifier dans AdaBoost:", max_depth_rf)
    else:
        print("L'estimateur n'est pas un RandomForestClassifier.")

    new_index_names = {
        'alpha': 'L1', 'reg_alpha': 'L1',
        'lambda': 'L2', 'reg_lambda': 'L2', 'l2_leaf_reg': 'L2',
        'n_estimators': 'Nb_estimators', 'iterations': 'Nb_estimators'}

    mod_dfs = []
    for i in range(4):
        mod_df = pd.DataFrame(
            best_params_per_model.iloc[:, i].dropna().rename(index=new_index_names))
        mod_dfs.append(mod_df)

    # Joindre les DataFrames
    best_params_per_model = mod_dfs[0]
    for df in mod_dfs[1:]:
        best_params_per_model = best_params_per_model.join(df, how='outer')

    return best_params_per_model, metrics_df, max_depth_rf


best_params_per_model, results_per_model, max_depth_RF = compare_boosting(models=models,
                                                                          nb_estimators=np.arange(
                                                                              50, 90, 5),
                                                                          learn_rate=np.arange(
                                                                              0.1, 1.1, 0.1),
                                                                          gamma=np.arange(
                                                                              0, 16, 1),
                                                                          l1=np.arange(
                                                                              0, 16, 1),
                                                                          l2=np.arange(
                                                                              0, 16, 1),
                                                                          max_depth=np.arange(
                                                                              0, 16, 1),
                                                                          metric='accuracy',
                                                                          average_metric='macro')

resultats = entrainement_model_optimal(models[3], average_metric='macro')
