# Packages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score,f1_score,recall_score,roc_curve, auc
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from scikeras.wrappers import KerasClassifier
import itertools
import tensorflow as tf
from tensorflow.keras.initializers import GlorotUniform

# Fixer la graine pour la reproductibilité
np.random.seed(42)

# Importation de la base
chemin_fichier = r"C:\Projets-personnels\Nutri-score\Données\5Data_no_miss_balanced.csv"

df = pd.read_csv(chemin_fichier, sep=',')

# Vérification
df.info()
print(df.isnull().sum())
df.describe()

df = df.drop('sodium_100g', axis=1)
## Processing

# Division des données
X = df.drop('nutriscore_grade', axis=1) # Variables prédictives
y = df['nutriscore_grade']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Reagrder si les classes sont déséquilibrer
class_presence_percentage = (pd.DataFrame({'classes': y})['classes'].value_counts() / len(y)) * 100
# Calculer la différence entre les pourcentages de présence des classes
class_pairs = [(c1, c2) for i, c1 in enumerate(class_presence_percentage.index) for c2 in class_presence_percentage.index[i+1:]]
differences = []
for c1, c2 in class_pairs:
    difference = abs(class_presence_percentage[c1] - class_presence_percentage[c2])
    differences.append(difference)
mean_difference = sum(differences) / len(differences)
# Déterminer la valeur de l'argument "average"
average = 'weighted' if mean_difference > 10 else 'macro'

num_classes = len(np.unique(y))

# Division des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/8, random_state=42)


"""
Savoir combien j'ai de processeurs dans ma machine pour tous les utiliser'
import multiprocessing as mp

print("Number of processors: ", mp.cpu_count())
Number of processors:  8
"""

def get_model(hidden_layer_sizes, dropout, optimizer='adam', seed=42,metric='accuracy'):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=(X_train.shape[1],)))
    for hidden_layer_size in hidden_layer_sizes:
        model.add(keras.layers.Dense(hidden_layer_size, activation="relu"))
        model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(num_classes, activation="softmax", kernel_initializer=GlorotUniform(seed=42)))
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=[metric])
    return model

model = KerasClassifier(
    model=get_model,
    hidden_layer_sizes=[10,50,100],
    dropout=0.5,
    optimizer='adam')

def hidden_layer_size(nb_hidden_layers, nb_neurons_per_layer):
    hidden_layer_sizes_combinations = []
    for num_layers in nb_hidden_layers:
        for layer_sizes in itertools.combinations_with_replacement(nb_neurons_per_layer, num_layers):
            hidden_layer_sizes_combinations.append(layer_sizes)
    
    return hidden_layer_sizes_combinations

# Utiliser la validation croisée pour déterminer les meilleurs hyperparamètres
def CV_parameters_classification(hidden_layer_dims,
                             optimizer = ["adam","sdg"],
                             dropout = np.arange(0.1,0.5,0.1),
                             metric='sparse_categorical_crossentropy',
                             X_train=X_train,y_train=y_train):
    
    # Définir les paramètres à optimiser et leurs valeurs possibles
    params = {"hidden_layer_sizes": hidden_layer_dims,
        "loss": ["sparse_categorical_crossentropy"],
        "optimizer": optimizer,
        'model__dropout': dropout}

    # Utiliser GridSearchCV
    grid_regression = GridSearchCV(model,params,refit=False, cv=5, scoring=metric,n_jobs=7)
    grid_result_regression = grid_regression.fit(X_train, y_train)
    
    # Créer une DataFrame à partir des résultats de la grille
    results_df = pd.DataFrame({'Rank' : grid_result_regression.cv_results_["rank_test_score"],
        'Hidden Layer Sizes': grid_result_regression.cv_results_['param_hidden_layer_sizes'],
        'Loss': grid_result_regression.cv_results_['param_loss'],
        'Optimizer' : grid_result_regression.cv_results_['param_optimizer'],
        'Dropout' : grid_result_regression.cv_results_['param_model__dropout'],
        'Mean Test Score': grid_result_regression.cv_results_['mean_test_score'],
        'Std Test Score': grid_result_regression.cv_results_['std_test_score']})
    
    # Trier les données
    results_df = results_df.sort_values('Rank',axis=0,ascending=True)
    
    # Traiter les données
    results_df['Std Test Score'] = round(results_df['Std Test Score'], 2)
    results_df['Hidden Layer Sizes'] = tuple(results_df['Hidden Layer Sizes'])
    del results_df['Loss']
    results_df['Optimizer'] = results_df['Optimizer'].astype(str)
    results_df['Dropout'] = results_df['Dropout'].astype(float)
    
    return results_df

results_df = CV_parameters_classification(hidden_layer_dims=hidden_layer_size(range(1, 50), range(1, 50)),
                                      optimizer = ["adam","Nadam"],
                                      dropout = np.arange(0.05,1.05,0.05),
                                      metric='accuracy')

top_models_results = results_df.head(3)