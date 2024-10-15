# =============================================================================
# Importation de la base puis traitement
# =============================================================================

df = pd.read_csv(r"C:\Projets_personnels\Ventes_magasin\src\data\Ventes.csv", sep=";")

# Conservation que des colonnes utiles pour la mod�lisation
df.columns  # Pour avoir la liste des variables
col_conserv = ['Date', 'Total_Amount']  # Liste des colonenes � conserver
df = df[col_conserv]
del col_conserv

# Transformation de la variable Date en Datetime
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df = df.sort_values(by='Date')

# Avoir la moyenne des ventes par jour
df = df.groupby('Date')['Total_Amount'].mean().reset_index()
df.set_index("Date", inplace=True)
df = df.dropna()

# =============================================================================
# Visualisation de l'�volution des ventes
# =============================================================================

# Visualisation graphique des ventes au fil du temps
plt.figure(figsize=(10, 6))
plt.plot(df.index, df.values)
plt.title('Evolution des ventes moyennes au cours du temps')
plt.xlabel('Date')
plt.ylabel('Ventes moyennes')
plt.grid(False)
plt.show()

# =============================================================================
# Evaluation de la stationnarit�
# =============================================================================

# Division de la base en apprentissage/validation
train_data, test_data = temporal_train_test_split(df, test_size=1/3)

# Test de DickerFuller
DickeyFuller(train_data, 'Total_Amount', 0.05)
# La s�rie est stationnaire, donc nous n'avons pas besoin de diff�rencier

# =============================================================================
# Etude de l'autocorr�lation
# =============================================================================

p, q = pq_param(lags=25, data=train_data)

# =============================================================================
# Construction du mod�le et cr�ation des pr�visions
# =============================================================================

# Cr�ation et ajustement du mod�le ARMA sur les donn�es d'entra�nement
# Comme la s�rie est stationnaire, d=0 donc le mod�le d�velopp� est un ARMA(p,q)
model = ARIMA(train_data, order=(0, 0, 0))
model_fit = model.fit()

# R�sum� du mod�le
print(model_fit.summary())

# Pr�visions pour l'ensemble de test
n_periods = len(test_data)
forecast = model_fit.forecast(steps=n_periods)

# Cr�ez un index pour les pr�visions bas� sur l'index de test_data
forecast_index = pd.date_range(
    start=test_data.index[0], periods=n_periods, freq='D')

# Calcul de l'erreur (MAPE)
mape = mean_absolute_percentage_error(test_data, forecast)
print(f'Mean Absolute Percentage Error : {mape:.2f}%')

# Pr�dictions sur l'ensemble d'entraînement
train_predictions = model_fit.predict(
    start=train_data.index[0], end=train_data.index[-1])

# Tracer les pr�dictions et les donn�es r�elles avec les courbes coll�es
plt.figure(figsize=(12, 6))
plt.plot(train_data.index, train_data.values,
         label="Ensemble d'entra�nement", color='blue')
plt.plot(test_data.index, test_data.values,
         label='Ensemble de test (r�el)', color='blue', linestyle='--')
plt.plot(train_data.index, train_predictions,
         label='Pr�dictions (Entra�nement)', color='red')
plt.plot(forecast_index, forecast, label='Pr�visions (Test)', color='green')

# D�finir les limites des axes x et y pour une continuit� visuelle
plt.xlim(train_data.index[0], test_data.index[-1])
y_min = min(train_data.min().values[0], test_data.min().values[0])
y_max = max(train_data.max().values[0], test_data.max().values[0])
plt.ylim(y_min, y_max)

plt.xlabel('Date')
plt.ylabel('Valeur')
plt.title('Pr�dictions du mod�le ARIMA')
plt.legend()
plt.show()

# =============================================================================
# Validation du mod�le ou non
# =============================================================================

# Test de Ljung-Box
ljung_box_results = sm.stats.acorr_ljungbox(
    model_fit.resid, lags=[1], return_df=True)

# Test de Jarque-Bera
jb_test_stat, jb_test_p_value, skew, kurtosis = sm.stats.jarque_bera(
    model_fit.resid)

# Test de H�t�rosc�dasticit� (Breusch-Pagan)
exog = add_constant(train_data)
bp_test_stat, bp_test_p_value, _, _ = het_breuschpagan(model_fit.resid, exog)

results_df = pd.DataFrame({
    'Test': ['Ljung-Box', 'Jarque-Bera', 'Breusch-Pagan'],
    'p-value': [ljung_box_results['lb_pvalue'].iloc[0], jb_test_p_value, bp_test_p_value],
    'Hypoth�se': [
        'Pas d\'autocorr�lation dans les r�sidus',
        'Les r�sidus sont distribu�s normalement',
        'Pas d\'h�t�rosc�dasticit� dans les r�sidus'],
    'Hypoth�se valid�e': [ljung_box_results['lb_pvalue'].iloc[0] > 0.05, jb_test_p_value > 0.05, bp_test_p_value > 0.05]
})

# Afficher les r�sultats
print(results_df.iloc[:, -2:])

# Enseignements :
# Les r�sidus sont un bruit blanc
# L� s�rie n'est que du bruit blanc : p = q = 0
# Donc il n'y a rien � mod�liser
# Les r�sidus sont h�t�rosc�dastiques
# D�veloppement d'un mod�le ARCH (ou GARCH) pour capturer cette h�t�rosc�dasticit� des r�sidus

# =============================================================================
# D�veloppement d'un mod�le ARCH ou GARCH
# =============================================================================

df_results = GARCH_search(train_data, "Total_Amount", 7, 7)
print(df_results.head(3))

# Appliquer le meilleur model
model = arch_model(df["Total_Amount"], vol='Garch', p=6, q=0)
results = model.fit(disp='off')
summary = results.summary()

# =============================================================================
# Visualisation des pr�visions
# =============================================================================

horizon_date = 30

pred = results.forecast(horizon=horizon_date)
future_dates = [df.index[-1] + timedelta(days=i)
                for i in range(1, horizon_date+1)]
pred = pd.Series(np.sqrt(pred.variance.values[-1, :]), index=future_dates)

plt.figure(figsize=(10, 4))
plt.plot(pred)
plt.title('Pr�diction de Volatilit� - 30 Prochains Jours', fontsize=20)

# =============================================================================
# V�rification des hypoth�ses
# =============================================================================

# Nous ne devons pas avoir d'h�t�rosc�dasticit� conditionnelle

# Residuals plot
plt.figure(figsize=(10, 4))
plt.plot(results.resid)
plt.title("Residuals of the GARCH model")

# ACF and PACF
plot_acf(results.resid, lags=20)
# S'il y a des pics significatifs, il y a h�t�rosc�dasticit� conditionnelle
plot_pacf(results.resid, lags=20)

# Shapiro-Wilk test for normality
stat, p = shapiro(results.resid)
print(f'Shapiro-Wilk statistic: {stat}, p-value: {p}')
# La normalit� des r�sidus est rejett�e

# LM test for ARCH effects pour v�rifier la pr�sence d'h�t�rosc�dasticit� conditionnelle
lm_test = het_arch(results.resid)
print('LM Statistical Test: %.3f, p-value: %.3f' % (lm_test[0], lm_test[1]))
# Le mod�le ne capture pas correctement la structure de volatilit� (car la P-Value est trop grande)

# Globalement, le mod�le ARMA et ARCH ne sont pas de bons mod�les pour
# mod�liser les ventes de produits

# =============================================================================
# Utilisation d'un réseau de neuronnes
# =============================================================================

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from itertools import combinations, permutations
from keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import TimeSeriesSplit

train_data, test_data = temporal_train_test_split(df, test_size=1/3)

# Normalisation des données
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.fit_transform(test_data)

# Création des générateurs de séquences
length = 2  # Longueur des séquences
train_generator = TimeseriesGenerator(train_scaled, train_scaled, length=length, batch_size=1)
test_generator = TimeseriesGenerator(test_scaled, test_scaled, length=length, batch_size=1)

def generate_lstm_layer_combinations(max_layers, max_neurons):
    """
    Génère toutes les combinaisons possibles de tailles de couches pour un modèle LSTM.
    
    Parameters:
    - max_layers: le nombre maximum de couches.
    - max_neurons: le nombre maximum de neurones dans une couche.
    
    Returns:
    - unique_combinations: une liste contenant toutes les combinaisons uniques d'architectures.
    """
    # Chiffres à utiliser pour les tailles de couches (de 2 à max_neurons)
    sizes = list(range(2, max_neurons + 1))
    
    # Liste pour stocker les combinaisons
    layer_combinations = []

    # Une seule couche
    layer_combinations.extend([(size,) for size in sizes])

    # Plusieurs couches
    for n_layers in range(2, max_layers + 1):
        for combo in combinations(sizes, n_layers):
            layer_combinations.extend(permutations(combo, n_layers))
    
    # Supprimer les doublons (les permutations peuvent créer des doublons)
    unique_combinations = list(set(layer_combinations))
    
    # Convertir chaque tuple en liste
    unique_combinations = [list(combo) for combo in unique_combinations]
    
    return unique_combinations

# Générer des combinaisons avec jusqu'à 4 couches cachées et un maximum de 10 neurones par couche
combinations_list = generate_lstm_layer_combinations(max_layers=3, max_neurons=3)

def create_flexible_model(lstm_units_list, dense_units, l1, l2, input_shape,optimizer='adam', activation='relu', loss='mean_squared_error'):
    
    # Initier le modèle
    model = Sequential()
    
    # Ajouter la première couche LSTM
    model.add(LSTM(lstm_units_list[0], activation=activation, return_sequences=(len(lstm_units_list) > 1), input_shape=input_shape))
    
    # Ajouter les couches LSTM suivantes
    for i in range(1, len(lstm_units_list)):
        return_seq = (i < len(lstm_units_list) - 1)
        model.add(LSTM(lstm_units_list[i],
                       activation=activation,
                       return_sequences=return_seq,
                       bias_regularizer=regularizers.L2(l2),
                       kernel_regularizer=regularizers.L1L2(l1=l1, l2=l2)))
    
    # Vérifier si dense_units est une liste, sinon le convertir en liste
    if isinstance(dense_units, int):
        dense_units = [dense_units]
    
    # Ajouter les couches Dense
    for units in dense_units:
        model.add(Dense(units, activation=activation))
    
    # Couche de sortie
    model.add(Dense(1))
    
    # Compiler le modèle
    model.compile(optimizer=optimizer, loss=loss)
    
    return model

# Fonction de validation croisée personnalisée
def custom_cross_val_search(train_generator, test_generator, lstm_units_list_combinations, dense_units_list_combinations, l1_values, l2_values, activation_list, input_shape,optimizer='adam',epochs=500):
    results = []
    
    for lstm_units in lstm_units_list_combinations:
        for dense_units in dense_units_list_combinations:
            for l1 in l1_values:
                for l2 in l2_values:
                    for activation in activation_list:
                        
                        # Créer le modèle
                        model = create_flexible_model(lstm_units, dense_units, l1, l2, optimizer=optimizer, activation=activation, input_shape=input_shape)
                        
                        # Entraînement avec EarlyStopping pour éviter l'overfitting
                        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                        history = model.fit(train_generator, validation_data=test_generator, epochs=epochs, callbacks=[early_stopping], verbose=0)
                        
                        # Récupérer la perte sur les données de validation
                        val_loss = history.history['val_loss'][-1]
                        results.append({
                            'lstm_units': lstm_units,
                            'dense_units': dense_units,
                            'l1': l1,
                            'l2': l2,
                            'optimizer': optimizer,
                            'activation': activation,
                            'val_loss': val_loss
                        })
    
    # Trier les résultats en fonction de la perte de validation
    results = pd.DataFrame(sorted(results, key=lambda x: x['val_loss']))
    
    return results

layer_combinations = combinations_list

dense_units_list = [10,20]
l1_values = [0.01, 0.5]
l2_values = [0.01, 0.5]
activation_list = ['relu', 'tanh']
input_shape = (3,1)

results_df = custom_cross_val_search(train_generator, test_generator,
                                     lstm_units_list_combinations=layer_combinations,
                                     dense_units_list_combinations=dense_units_list,
                                     l1_values=l1_values, l2_values=l2_values,
                                     optimizer='adam',
                                     activation_list=activation_list,
                                     input_shape=input_shape)

print(results_df.head(1))


def evaluate_model_with_cv(data, lstm_units, dense_units, l1, l2, optimizer, activation, input_shape, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    errors = []

    for train_index, test_index in tscv.split(data):
        # Utiliser iloc pour accéder aux données par indice
        train_data, test_data = data.iloc[train_index], data.iloc[test_index]
        
        # Normalisation des données
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_data)
        test_scaled = scaler.transform(test_data)

        # Création des générateurs de séquences
        train_generator = TimeseriesGenerator(train_scaled, train_scaled, length=3, batch_size=1)
        test_generator = TimeseriesGenerator(test_scaled, test_scaled, length=3, batch_size=1)

        # Créer le modèle avec les meilleurs hyperparamètres
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model = create_flexible_model(lstm_units, dense_units, l1, l2, optimizer=optimizer, activation=activation, input_shape=input_shape)

        # Entraînement du modèle
        model.fit(train_generator, epochs=500, callbacks=[early_stopping], verbose=0)

        # Faire des prévisions
        predictions = model.predict(test_generator)

        # Calculer l'erreur (par exemple, l'erreur quadratique moyenne)
        actual = test_scaled[3:, 0]
        error = np.sqrt(np.mean((predictions.flatten() - actual) ** 2))
        errors.append(error)

    # Retourner la moyenne des erreurs
    return np.mean(errors)

lstm_units = [2,3]
dense_units=20
l1=0.01
l2=0.5
optimizer='adam'
activation='tanh'
input_shape = (3,1)

mean_error = evaluate_model_with_cv(data=df, lstm_units = lstm_units,
                                    dense_units=dense_units, l1=l1, l2=l2,
                                    optimizer=optimizer, activation=activation, input_shape=input_shape)

print(f'Erreur théorique de prévision (RMSE) : {mean_error}')


def visualize_predictions(lstm_units, dense_units, l1, l2, optimizer, activation, input_shape, train_data, test_data, k_periods,loss='mean_squared_error'):
    # Normalisation des données de test
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    # Création du générateur de séquences pour la base de test
    train_generator = TimeseriesGenerator(train_scaled, train_scaled, length=2, batch_size=1)

    # Création du modèle
    model = Sequential()
    model.add(LSTM(lstm_units[0], activation=activation, return_sequences=(len(lstm_units) > 1), input_shape=input_shape))
    
    for i in range(1, len(lstm_units)):
        return_seq = (i < len(lstm_units) - 1)
        model.add(LSTM(lstm_units[i],
                       activation=activation,
                       return_sequences=return_seq,
                       bias_regularizer=regularizers.L2(l2),
                       kernel_regularizer=regularizers.L1L2(l1=l1, l2=l2)))
    if isinstance(dense_units, int):
        dense_units = [dense_units]
    for units in dense_units:
        model.add(Dense(units, activation=activation))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss=loss)
    
    # Faire des prévisions sur la base de test
    predictions = model.predict(test_generator)

    # Inverser la normalisation pour obtenir les valeurs réelles
    predictions_inverse = scaler.inverse_transform(predictions)
    actual_inverse = scaler.inverse_transform(test_scaled[2:])

    # Prévisions sur K périodes
    forecast = []
    last_sequence = test_scaled[-2:]  # Dernière séquence de test

    for _ in range(k_periods):
        # Prédire la prochaine période
        next_pred = model.predict(np.expand_dims(last_sequence, axis=0))
        forecast.append(next_pred[0, 0])  # Stocker la prédiction
        # Mettre à jour la séquence en incluant la dernière prédiction
        last_sequence = np.roll(last_sequence, -1, axis=0)
        last_sequence[-1] = next_pred

    # Inverser la normalisation pour les prévisions
    forecast_inverse = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

    # Visualiser les résultats
    # Obtenir les dates du test_data
    dates = test_data.index

    # Créer des dates pour les prévisions futures
    future_dates = pd.date_range(dates[-1], periods=k_periods+1, freq='D')[1:]  # Si fréquence quotidienne

    # Visualiser les résultats
    plt.figure(figsize=(14, 7))
    plt.plot(dates[2:], actual_inverse, label='Valeurs réelles', color='blue')
    plt.plot(dates[2:], predictions_inverse, label='Prédictions sur la base de test', color='orange')
    plt.plot(future_dates, forecast_inverse, label='Prévisions sur K périodes', color='green', linestyle='--')
    plt.gcf().autofmt_xdate()
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))  # Format des dates
    plt.title('Visualisation des Prédictions')
    plt.xlabel('Date')
    plt.ylabel('Valeur')
    plt.legend()
    plt.show()
    
    
visualize_predictions(lstm_units, dense_units, l1, l2, optimizer, activation, input_shape=input_shape, train_data=train_data,test_data=test_data, k_periods=10,loss='mean_squared_error')

