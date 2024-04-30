############################# Importing liberaries #############################

from datetime import timedelta
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import pandas as pd
import arch
from statsmodels.tsa.stattools import adfuller
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_arch,acorr_ljungbox
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

############################### Data processing ###############################

# Reading the dataset and formatting the date
file_path = "C:/Projets-personnels/Séries temporelles/BTC_Data.csv"
df = pd.read_csv(file_path, parse_dates=['Date'], date_format='%Y/%m/%d')
df['Date'] = pd.to_datetime(df['Date']).dt.date
df.set_index("Date", inplace=True)

# Checking data set contains null values
df.isnull().values.any()  # We don't have any missing values

########################## Visualization of the price ##########################


def visualisation(data,feature,title_y_evolution,ylabel):
    # Select the y serie
    y = data[feature]

    # Display the y's evolution
    plt.figure(figsize=(10,5))
    plt.plot(y.index,y,color="red")
    plt.title(title_y_evolution)
    plt.xlabel("Date")
    plt.ylabel("Price in USD")
    plt.grid(True)
    plt.show()

    # Plot the ACF and PACF
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    plot_acf(y, lags=50, ax=ax1)
    ax1.set_title('ACF')
    plot_pacf(y, lags=50, ax=ax2)
    ax2.set_title('PACF')
    plt.tight_layout()
    
    return plt.show()

visualisation(df, "priceUSD", "Bitcoin's price evolution in USD", "Price in USD")

######################### Visualization of the returns #########################

# Select the price series to predict
returns = df["priceUSD"].pct_change().dropna()

# Analysis of the evolution of yields
returns.plot(figsize=(10, 6))
plt.title('Returns Bitcoin evolution')
plt.xlabel('Date')
plt.ylabel('Returns')
plt.show()

# Calculation of ACF and PACF
fig, ax = plt.subplots(nrows=2, figsize=(10, 8))
plot_acf(returns, ax=ax[0], lags=20, alpha=0.05)
plot_pacf(returns, ax=ax[1], lags=20, alpha=0.05)

plt.tight_layout()
plt.show()

################################# Stationarity #################################

# Stationarity test - Augmented Dickey-Fuller test
result = adfuller(returns)
print('P-value:', result[1])
print('Is stationary?', 'Yes' if result[1] < 0.05 else 'No')
# The time serie is stationary

####################### Analysis of the square of return #######################

# Analysis of the square of returns
returns_squared = returns ** 2
returns_squared.plot(figsize=(10, 6))
plt.title("Evolution of the square of CAC 40 returns")
plt.xlabel("Date")
plt.ylabel("Square of returns")
plt.show()

# Calculation of the ACF and the PACF of the square of returns
fig, ax = plt.subplots(nrows=2, figsize=(10, 8))
plot_acf(returns_squared, ax=ax[0], lags=20, alpha=0.05)
# There are significant peaks, then there is conditional heteroscedascity
plot_pacf(returns_squared, ax=ax[1], lags=20, alpha=0.05)
plt.tight_layout()
plt.show()

####################### Construction of the GARCH model #######################

df['Return'] = df["priceUSD"].pct_change()
df['Squared_Return'] = df['Return'] ** 2
df = df.dropna()

def GARCH_search(data, feature, p_max, q_max):
    # Define the rank of p and q
    p_range = range(1, p_max)  
    q_range = range(0, q_max) 

    # Initialize the DataFrame to store the results
    results_df = pd.DataFrame(columns=['p', 'q', 'aic'])

    # Loop over all possible combinations of p and q
    for p in p_range:
        for q in q_range:
            # Specify GARCH model
            model = arch.arch_model(data[feature], vol='Garch', p=p, q=q)
            
            # Use the model
            results = model.fit(disp='off')  # turn off convergence messages
            
            # Obtain the AIC
            aic = results.aic
            
            # Add the results to the DataFrame
            results_df = results_df._append({'p': p, 'q': q, 'aic': aic}, ignore_index=True)

    # Find the optimal values
    result_df_aic = results_df.sort_values(by='aic').head(1)
    optimal_p = int(result_df_aic['p'].values[0])
    optimal_q = int(result_df_aic['q'].values[0])
    
    # Display the results
    return optimal_p, optimal_q

GARCH_search(df, "Return", 20, 20)

# Display the optimal model
model = arch.arch_model(df["Return"], vol='Garch', p=1, q=7)
results = model.fit()
summary = results.summary()

############################### Model validation ###############################
# Nous ne devons pas avoir d'hétéroscédasticité conditionnelle

# Residuals plot
plt.figure(figsize=(10,4))
plt.plot(results.resid)
plt.title("Residuals of the GARCH model")

# ACF and PACF
plot_acf(results.resid, lags=20)
# S'il y a des pics significatifs, il y a hétéroscédasticité conditionnelle
plot_pacf(results.resid, lags=20)

# Shapiro-Wilk test for normality
stat, p = shapiro(results.resid)
print(f'Shapiro-Wilk statistic: {stat}, p-value: {p}')
#La normalité des résidus est rejettée

# LM test for ARCH effects pour vérifier la présence d'hétéroscédasticité conditionnelle
lm_test = het_arch(results.resid)
print('LM Statistical Test: %.3f, p-value: %.3f' % (lm_test[0], lm_test[1]))
# Le modèle ne capture pas correctement la structure de volatilité

############################## Rolling forecasts ##############################

# Objectif : Avec seulement les données d'apprentissage
# Je prévois la volatilité du jour suivant,
# je rajoute ce jour là pour ensuite prévoir le jour d'apres, etc.
# jusqu'à ce que les deux ans se terminent (365*2)

rolling_predictions = []
test_size = 365*2

for i in range(test_size):
     train = df['Return'][:-(test_size-i)]
     model = arch.arch_model(train, p=1, q=7)
     model_fit = model.fit(disp='off')
     pred = model_fit.forecast(horizon=1)
     rolling_predictions.append(np.sqrt(pred.variance.values[-1,:][0]))

rolling_predictions = pd.Series(rolling_predictions, index=returns.index[-test_size:])

plt.figure(figsize=(10,4))
true, = plt.plot(returns[-test_size:])
preds, = plt.plot(rolling_predictions)
plt.title('Volatility of the Bitcoin price with the rolling forecast', fontsize=20)
plt.legend(['Bitcoin yield', 'Predicted volatility'], fontsize=16)

############################### Using the model ###############################

train = df['Return']
model = arch.arch_model(train, p=1, q=7)
model_fit = model.fit(disp='off')

horizon_date=7

pred = model_fit.forecast(horizon=horizon_date)
future_dates = [df.index[-1] + timedelta(days=i) for i in range(1,horizon_date+1)]
pred = pd.Series(np.sqrt(pred.variance.values[-1,:]), index=future_dates)

plt.figure(figsize=(10,4))
plt.plot(pred)
plt.title('Prédiction de Volatilité - 7 Prochains Jours', fontsize=20)

############################### Cross validation ###############################

# Créer une DataFrame pour stocker les résultats
results_df = pd.DataFrame(columns=['p', 'q', 'Mean_MSE', 'Mean_RMSE'])

# Définir les valeurs de p et q
for p in tqdm(range(1, 5)):
    for q in range(1, 5):
        # Initialisez le modèle GARCH avec les paramètres choisis
        model = arch.arch_model(df['Return'], vol='Garch', p=p, q=q)

        # Initialisez une liste pour stocker les résultats de la validation croisée
        cross_val_results = []

        # Initialisez TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=5)

        # Effectuez la validation croisée
        for train_index, test_index in tscv.split(df['Return']):
            train_set, test_set = df['Return'].iloc[train_index], df['Return'].iloc[test_index]

            # Ajustez le modèle sur l'ensemble d'entraînement
            results = model.fit(disp='off', last_obs=train_index[-1])

            # Effectuez des prévisions sur l'ensemble de test
            forecasts = results.forecast(horizon=len(test_set))

            # Évaluez les performances du modèle sur l'ensemble de test
            mse = ((test_set - forecasts.variance.values[-1])**2).mean()
            rmse = np.sqrt(mse)

            # Stockez les résultats de la validation croisée
            cross_val_results.append({'MSE': mse, 'RMSE': rmse})

        # Calculer la moyenne MSE et RMSE pour cette combinaison de p et q
        mean_mse = np.mean([result['MSE'] for result in cross_val_results])
        mean_rmse = np.mean([result['RMSE'] for result in cross_val_results])

        # Ajouter les résultats à la DataFrame
        results_df = results_df._append({'p': p, 'q': q, 'Mean_MSE': mean_mse, 'Mean_RMSE': mean_rmse}, ignore_index=True)

# Trier la DataFrame par Mean_MSE et Mean_RMSE dans l'ordre décroissant et afficher les 10 premières observations
results_sort_df = results_df.sort_values(by=['Mean_MSE', 'Mean_RMSE'], ascending=True).head(3)

# Afficher la DataFrame finale
print(results_sort_df)

############################# Using the new model #############################

model = arch.arch_model(train, p=4, q=1)
model_fit = model.fit(disp='off')

horizon_date=7

pred = model_fit.forecast(horizon=horizon_date)
future_dates = [df.index[-1] + timedelta(days=i) for i in range(1,horizon_date+1)]
pred = pd.Series(np.sqrt(pred.variance.values[-1,:]), index=future_dates)

plt.figure(figsize=(10,4))
plt.plot(pred)
plt.title('Prédiction de Volatilité - 7 Prochains Jours', fontsize=20)