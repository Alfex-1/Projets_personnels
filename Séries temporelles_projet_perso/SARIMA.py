############################# Importing liberaries #############################

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from pylab import rcParams
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

#Filter and ignore warning messages about statsmodels.tsa.arima_model.ARMA class
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA',FutureWarning)


############################### Data processing ###############################

#Reading the dataset and formating the date
file_path = "C:/Projets-personnels/Séries temporelles/valeurs_mensuelles.csv"
df = pd.read_csv(file_path, sep=";")
df['Date'] = pd.to_datetime(df['Date'])
df['Date'] = df['Date'].dt.to_period('M')
df.set_index("Date", inplace = True)

#To verify if we have a correct database
df.describe()
df.info()

#Checking data set contains null values
df.isnull().values.any()


############################## Correlation study ##############################

df1=df.reset_index(drop=True)
X = df1.drop('temp', axis=1)
y=df1[["temp"]]

#Dropping features which is highly correlated each other.

#Create correlation matrix
corr_matrix = X.corr().abs()

#Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

#Find index of feature columns with correlation greater than 0.75
to_drop = [column for column in upper.columns if any(upper[column] > 0.75)]

#Drop features 
X.drop(X[to_drop], axis=1,inplace=True)
X_columns=list(X.columns)
y_columns=["temp"]

#Pearson correlation between each feature in a set of features and the target variable
correlation_result={}
for i in range(len(X_columns)):
    correlation = X[X_columns[i]].corr(y["temp"])
    correlation_result[X_columns[i]]=correlation
correlation_result=sorted(correlation_result.items(), key = lambda kv:(kv[1], kv[0]),reverse=True)

#Creating a new DataFrame by selecting only the features based on their correlation
temp=[]
for i in correlation_result:
    temp.append(i[0])
X_train=X[temp]
X_train

#Correlation HeatMap
plt.figure(figsize=(12,8))
sns.heatmap(X.corr(), annot = True,fmt= '.2f')
plt.title('Correlation Heatmap about features')
plt.show()

#Scaling values
estimators=[]
estimators.append(['minmax',MinMaxScaler(feature_range=(-1,1))])
scale=Pipeline(estimators)
X_min_max=scale.fit_transform(X)
y_min_max=scale.fit_transform(y)

train_data = df[list(X_train.columns)].copy()
train_data['temp'] = df['temp'].copy()

############################ Creating new variables ############################

#Creating the variable Percent change = priceUSD_t/priceUSD_(t-1)
train_data['Change'] = train_data.priceUSD.div(train_data.priceUSD.shift())
train_data = train_data.assign(Change=pd.Series(train_data.priceUSD.div(train_data.priceUSD.shift())))
train_data['Change'].plot(figsize=(20,8))
plt.show()


#Creating additional columns representing lagged values of the priceUSD,
#enabling the inclusion of past temporal information in the dataset
train_data['lag_1'] = train_data['priceUSD'].shift(1)
train_data['lag_2'] = train_data['priceUSD'].shift(2)
train_data['lag_3'] = train_data['priceUSD'].shift(3)
train_data['lag_4'] = train_data['priceUSD'].shift(4)
train_data['lag_5'] = train_data['priceUSD'].shift(5)
train_data['lag_6'] = train_data['priceUSD'].shift(6)
train_data['lag_7'] = train_data['priceUSD'].shift(7)


#Calculating the percentage return based on the Change and adds it as a new column named 'Return' in the dataset
train_data = train_data.assign(Return=pd.Series(train_data.Change.sub(1).mul(100)))

############################# Time series seasons #############################

rcParams['figure.figsize'] = 50, 35
decomposed_train_data = sm.tsa.seasonal_decompose(train_data["priceUSD"],period=365)
figure = decomposed_train_data.plot()
plt.show()
#1 : Original
#2 : Trend
#3 : Seasonal
#4 : Residual


#Verication that no missed values are here and drop them if it's the case
train_data.isnull().values.any()
train_data.dropna(axis = 0, how ='any',inplace=True)
train_data["priceUSD"].describe()


#Identification significative lags for the p parameter : 3
plot_pacf(train_data["priceUSD"],lags=20)
plt.show()

#Plotting autocorrelation of white noise for the q parameter : 0
plot_acf(train_data["priceUSD"],lags=150,alpha=0.05)
plt.show()

############################### Statistic tests ###############################

#Stationary condition test

def DickeyFuller(data,feature,pvalue):
    """
    Perform the Augmented Dickey-Fuller test.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing the time series data.
    feature : str
        The name of the column in the DataFrame representing the time series.
    pvalue : float
        The significance level for the test.

    Returns
    -------
    A message indicates if the time series is stationary or not.

    """
    #Perform the Augmented Dickey-Fuller test
    result = adfuller(data[feature], autolag='AIC')
    
    #Extract the test results
    test_statistic, p_value, lags, _, critical_values, _ = result
    
    #Display the time serie graphic
    df[feature].plot(title='Time series')
    plt.ylabel(f'{feature} Values')
    plt.show()
    
    #Interpret the results
    if p_value <= pvalue:
        return print("The time series is stationary")
    else:
        return print("The time series is not stationary")

DickeyFuller(df,'priceUSD', 0.05) #Not stationary

def KPSS(data,feature,pvalue):
    """
    Perform the KPSS test.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing the time series data.
    feature : str
        The name of the column in the DataFrame representing the time series.
    pvalue : float
        The significance level for the test.

    Returns
    -------
    A message indicates if the time series is stationary or not.

    """
    result = kpss(data[feature], regression='c')
    kpss_statistic, p_value, lags, critical_values = result
    
    # Interpret the results
    alpha = pvalue
    if p_value <= alpha:
        return print("The time series is not stationary")
    else:
        return print("The time series is stationary")

KPSS(df,'priceUSD', 0.05) #Not stationary

#In this case, we are going te compare the ARIMA AND SARIMA models


################################# Random Walk #################################

#Generating a random walk

#Yi = Y_t-Y_(t-1)
diff = train_data["priceUSD"].diff()
#Adding the "difference" variable
train_data = train_data.assign(difference=pd.Series(diff))
#Drop the missed value in the 1st row
diff = diff.dropna()
diff.plot()
plt.show()
#Puropose : To make a time series stationary, facilitating
#the application of models

#To verify if this new time series is stationary
df_diff = diff.to_frame()
DickeyFuller(df_diff,'priceUSD', 0.001) #Stationary
KPSS(df_diff,'priceUSD', 0.001) #Stationary

#In this case d=1

"""
plt.figure(figsize=(12, 6))
plt.plot(diff, label='Différenciation 1')
plt.legend()
plt.show()

from pmdarima import auto_arima

model_d0 = auto_arima(train_data["priceUSD"], start_p=1, start_q=1, max_p=3, max_q=3, d=0, seasonal=False, trace=True)
model_d1 = auto_arima(train_data["priceUSD"], start_p=1, start_q=1, max_p=3, max_q=3, d=1, seasonal=False, trace=True)
model_d2 = auto_arima(train_data["priceUSD"], start_p=1, start_q=1, max_p=3, max_q=3, d=2, seasonal=False, trace=True)

print("AIC d=0:", model_d0.aic())
print("AIC d=1:", model_d1.aic())
print("AIC d=2:", model_d2.aic())
"""


########################### Modeling and forecasting ###########################

#ARIMA model
def ARIMA_forecast(p, d, q, data, feature, ylabel, enddate):
    """
    Perform ARIMA forecasting.

    Parameters
    ----------
    p : int
        The autoregressive order.
    d : int
        The differencing order.
    q : int
        The moving average order.
    data : pd.DataFrame
        The DataFrame containing the time series data.
    feature : str
        The name of the column in the DataFrame representing the time series.
    ylabel : str
        The label for the y-axis in the plot.
     enddate : str
        The end date until which the forecasting is performed in the format 'YYYY-MM-DD'. 

    Returns
    -------
    Displays a plot with the actual prices, ARIMA predictions, and future predictions.
    Additionally, prints the RMSE, MAE, and R²metrics inside the plot.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
    from statsmodels.tsa.arima.model import ARIMA

    df_train = data[:round(len(data)*(2/3))]
    df_valid = data[round(len(data)*(2/3)):]

    # Create ARIMA model
    model = ARIMA(data[feature], order=(p, d, q))
    model_fit = model.fit()

    # Make predictions
    prediction = model_fit.predict(start=df_valid.index[0], end=df_valid.index[-1])

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(df_valid[feature], prediction))
    mae = mean_absolute_error(df_valid[feature], prediction)
    r_squared = r2_score(df_valid[feature], prediction)

    # Set the figure size
    plt.figure(figsize=(10, 6))

    ax = plt.gca()

    train_data["Arima"] = prediction
    train_data.dropna()

    futureDates = pd.DataFrame(pd.date_range(start="2022-12-22", end=enddate, freq="B"), columns=["Dates"])
    futureDates.set_index('Dates', inplace=True)
    model_fit.predict(start=futureDates.index[0], end=futureDates.index[-1])

    sns.lineplot(data=data, x=data.index, y=feature, label='Actual Price')
    sns.lineplot(data=data, x=data.index, y='Arima', label='Arima Prediction')
    model_fit.predict(start=futureDates.index[0], end=futureDates.index[-1]).plot(color="purple", label='Future Prediction')

    plt.legend(loc='best', bbox_to_anchor=(1, 1))

    # Place metrics in top left corner
    ax.text(0.02, 0.95, f'RMSE: {rmse:.2f}', transform=ax.transAxes, fontsize=10, color='red')
    ax.text(0.02, 0.90, f'MAE: {mae:.2f}', transform=ax.transAxes, fontsize=10, color='blue')
    ax.text(0.02, 0.85, f'R²: {r_squared:.2f}', transform=ax.transAxes, fontsize=10, color='green')

    return plt.show()

ARIMA_forecast(3, 1, 0, train_data, "priceUSD", 'USD price', "2024-02-28")


#SARIMA model
#We keep p,d,q parameters that we have found for ARIMA
#We need to find P,D,Q and m parameters

df_diff = df_diff.dropna()
plot_acf(df_diff["priceUSD"], lags=20)
plt.show()
plot_pacf(df_diff["priceUSD"],lags=20)
plt.show()



def SARIMA_forecast(p, d, q, P, D, Q, m, data, feature, ylabel, enddate):
    """
    Perform SARIMA forecasting.

    Parameters
    ----------
    p : int
        The autoregressive order.
    d : int
        The differencing order.
    q : int
        The moving average order.
    P : int
        The seasonal autoregressive order.
    D : int
        The seasonal differencing order.
    Q : int
        The seasonal moving average order.
    m : int
        The number of time steps for a seasonal period.
    data : pd.DataFrame
        The DataFrame containing the time series data.
    feature : str
        The name of the column in the DataFrame representing the time series.
    ylabel : str
        The label for the y-axis in the plot.
    enddate : str
        The end date until which the forecasting is performed in the format 'YYYY-MM-DD'.

    Returns
    -------
    Displays a plot with the actual prices, SARIMA predictions, and future predictions.
    Additionally, prints the RMSE, MAE and R² metrics inside the plot.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
    import seaborn as sns

    # Create SARIMA model
    model = SARIMAX(data[feature], order=(p, d, q), seasonal_order=(P, D, Q, m))

    # Fit the model to the training data
    model_fit = model.fit(disp=False)

    df_train = data[:round(len(data)*(2/3))]
    df_valid = data[round(len(data)*(2/3)):]

    # Make predictions
    prediction = model_fit.predict(start=df_valid.index[0], end=df_valid.index[-1])

    data["Sarimax"] = prediction
    data.dropna()

    # Calculate RMSE, MAE, and R²
    rmse = np.sqrt(mean_squared_error(df_valid[feature], prediction))
    mae = mean_absolute_error(df_valid[feature], prediction)
    r_squared = r2_score(df_valid[feature], prediction)

    futureDates = pd.DataFrame(pd.date_range(start="2022-12-22", end=enddate, freq="B"), columns=["Dates"])
    futureDates.set_index('Dates', inplace=True)

    # Set the figure size
    plt.figure(figsize=(10, 6))

    sns.lineplot(data=data, x=data.index, y=feature, label='Actual Price')
    sns.lineplot(data=data, x=data.index, y='Sarimax', label='Sarima Prediction')
    model_fit.predict(start=futureDates.index[0], end=futureDates.index[-1]).plot(color="purple", label='Future Prediction')

    plt.legend(loc='best', bbox_to_anchor=(1, 1))

    # Place metrics in top left corner
    plt.text(0.02, 0.95, f'RMSE: {rmse:.2f}', transform=plt.gca().transAxes, fontsize=10, color='red')
    plt.text(0.02, 0.90, f'MAE: {mae:.2f}', transform=plt.gca().transAxes, fontsize=10, color='blue')
    plt.text(0.02, 0.85, f'R²: {r_squared:.2f}', transform=plt.gca().transAxes, fontsize=10, color='green')

    plt.xlabel('Date')
    plt.ylabel(ylabel)
    plt.title('SARIMA Forecasting')

    return plt.show()


SARIMA_forecast(3, 1, 0, 1, 1, 1, 4, train_data, "priceUSD", 'USD price', "2026-02-01")