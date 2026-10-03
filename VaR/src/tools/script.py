# Importation
tickers=['BNP.PA','RNO.PA','ORA.PA','AIR.PA','BN.PA']
weights=np.array([0.2,0.2,0.2,0.2,0.2])
start="2020-01-01"
end=pd.to_datetime("today")
df=yf.download(tickers, start, end)['Close']

# Visualisation
fig_prices = px.line(df, title='Prix de Clôture des Actions', labels={'value': 'Prix (€)', 'variable': 'Ticker'})
fig_prices.update_layout(template='plotly_dark')
fig_prices.show()

# Calcul des rendements
returns= df.pct_change().dropna()*100
returns

# Visualisation
fig_returns = px.line(returns, title='Rendements des Actions', labels={'value': 'Rendement (%)', 'variable': 'Ticker'})
fig_returns.update_layout(template='plotly_dark')
fig_returns.show()

px.histogram(returns)

# Calcul de la VaR            
var_histo, var_future_histo=var_calculation(returns, method='historical', multi=True, weights=[0.03,0.17,0.05,0.15,0.6], confidence_level=95, days=10)
var_param, var_future_param=var_calculation(returns, method='parametric', multi=True, weights=[0.03,0.17,0.05,0.15,0.6], confidence_level=95, days=10)
var_simu, var_future_simu=var_calculation(returns, method='simulation', multi=True, weights=[0.03,0.17,0.05,0.15,0.6], confidence_level=95, num_simulations=100000, days=10)

def stress_test(data, shocks, weights=None, method='historical', confidence_level=0.95, num_simulations=1000):
    # Mise en forme des poids
    if weights is None:
        weights = np.full(len(data.columns), 1/len(data.columns))
    weights = np.array(weights)
    
    # Calculer les rendements quotidiens
    returns = data.pct_change().dropna()
    
    # Calculer la valeur initiale du portefeuille
    portfolio_returns = returns.dot(weights)
    portfolio_value = (1 + portfolio_returns).cumprod()
    
    stressed_portfolio_values = portfolio_value.copy()
    
    # Appliquer les différents scénarios de stress selon la méthode
    if method == 'historical':
        stressed_returns = portfolio_returns.copy()
        for date, shock in shocks.items():
            if date in stressed_returns.index:
                stressed_returns.loc[date] *= (1 + shock)  # Appliquer le choc historique
        stressed_portfolio_values = (1 + stressed_returns).cumprod()
    
    elif method == 'hypothetical':
        stressed_returns = portfolio_returns.copy()
        for ticker, shock in shocks.items():
            stressed_returns[ticker] *= (1 + shock)
        stressed_portfolio_values = (1 + stressed_returns).cumprod()
    
    elif method == 'monte_carlo':
        # Simulations de Monte Carlo
        simulated_portfolio_values = []
        mean_return = np.mean(portfolio_returns)
        std_dev = np.std(portfolio_returns)
        
        for _ in range(num_simulations):
            random_shocks = np.random.normal(mean_return, std_dev, len(portfolio_returns))
            simulated_returns = portfolio_returns + random_shocks
            simulated_value = (1 + simulated_returns).cumprod()
            simulated_portfolio_values.append(simulated_value)
        
        # Calculer le percentile correspondant au niveau de confiance
        simulated_portfolio_values = np.array(simulated_portfolio_values)
        stressed_portfolio_values = np.percentile(simulated_portfolio_values, (1 - confidence_level) * 100, axis=0)
    
    # Calcul de la VaR et de l'Expected Shortfall (ES)
    var = np.percentile(stressed_portfolio_values, (1 - confidence_level) * 100)
    es = stressed_portfolio_values[stressed_portfolio_values <= var].mean()

    # Affichage des indicateurs de performance
    print(f"VaR à {confidence_level * 100}% de confiance : {var:.2f}")
    print(f"Expected Shortfall (ES) à {confidence_level * 100}% de confiance : {es:.2f}")

    # Feedback loop (exemple simple)
    if var < -0.15:  # Si la VaR est plus basse qu'un seuil (par exemple, -15%)
        print("La VaR est trop basse, ajustement nécessaire des chocs ou du portefeuille.")
        # Ici, on pourrait ajuster les chocs ou les poids en fonction des résultats

    # Tracer le graphique
    fig = go.Figure()

    # Ajout de la trace pour la valeur du portefeuille sans stress
    fig.add_trace(go.Scatter(
        x=portfolio_value.index,
        y=portfolio_value,
        mode='lines',
        name='Valeur du portefeuille (sans stress)'
    ))

    # Ajout de la trace pour la valeur du portefeuille avec ou sans stress selon la méthode
    fig.add_trace(go.Scatter(
        x=stressed_portfolio_values.index,  # Assuming this is a pandas Series or similar
        y=stressed_portfolio_values,
        mode='lines',
        name='Valeur du portefeuille (avec stress)'
    ))

    # Mise à jour de la mise en page
    fig.update_layout(
        title='Résultat du Stress Test',
        xaxis_title='Date',
        yaxis_title='Valeur du Portefeuille',
        legend_title='Légende',
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True)
    )

    # Affichage de la figure
    fig.show()
    
    # Retourner les résultats
    return stressed_portfolio_values, fig, var, es
    
chocs = {
    '2020-03-09': -0.5, '2020-03-16': -0.5, '2020-03-23': -0.5,
    '2021-02-15': -0.05, '2021-06-21': -0.06, '2021-09-20': -0.04,
    '2022-01-24': -0.03, '2022-02-24': -0.10, '2022-03-01': -0.05,
    '2022-05-05': -0.5, '2022-07-14': -0.92, '2022-09-13': -0.05,
    '2022-11-10': -0.04, '2023-03-15': -0.09, '2023-05-15': -0.10,
    '2023-07-25': -0.06, '2023-10-13': -0.05
}
stress_histo=stress_test(data=df, shocks=chocs, method='historical', confidence_level=0.95, num_simulations=1000)


def calculate_returns(data, weights=None):
    # Calculer les rendements quotidiens
    returns = data.pct_change().dropna()
    
    # Si des poids sont fournis, calculer les rendements du portefeuille
    if weights is not None:
        portfolio_returns = returns.dot(weights)
        return portfolio_returns
    return returns

def validate_stationarity(returns):
    # Test ADF (Augmented Dickey-Fuller) pour la stationnarité
    adf_test = adfuller(returns)
    p_value = adf_test[1]
    stationarity_status = 'Stationnaire' if p_value < 0.05 else 'Non stationnaire'
    return stationarity_status, p_value

def validate_volatility_condition(returns):
    # Ajuster un modèle GARCH pour vérifier la volatilité conditionnelle
    model = arch.arch_model(returns, vol='Garch', p=1, q=1)
    results = model.fit(disp="off")
    return results

def calculate_risk_metrics(portfolio_value, confidence_level=0.95):
    # Calcul de la VaR à partir des rendements historiques
    portfolio_returns = portfolio_value.pct_change().dropna()
    var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
    
    # Calcul de l'Expected Shortfall (ES)
    es = portfolio_returns[portfolio_returns <= var].mean()
    
    # Calcul du Maximum Drawdown
    cumulative_returns = (1 + portfolio_returns).cumprod()
    max_drawdown = (cumulative_returns.cummax() - cumulative_returns).max()
    
    # Calcul du Tail VaR
    tail_var = np.mean(portfolio_returns[portfolio_returns <= var])
    
    return {'VaR': var, 'Expected Shortfall': es, 'Tail VaR': tail_var, 'Max Drawdown': max_drawdown}

def calculate_volatility(returns, method='historical', p=1, q=1):
    if method == 'historical':
        # Volatilité historique (écart-type des rendements)
        volatility = np.std(returns) * np.sqrt(252)  # annualisée (252 jours de bourse par an)
    elif method == 'conditional':
        # Volatilité conditionnelle avec modèle GARCH
        model = arch.arch_model(returns, vol='Garch', p=p, q=q)
        results = model.fit(disp="off")
        volatility = results.conditional_volatility[-1] * np.sqrt(252)  # annualisée
    return volatility

def test_robustness(data, shocks, method='historical', num_simulations=1000, confidence_level=0.95):
    # Appliquer les chocs sur différentes configurations de données
    stressed_returns = apply_shocks(data, shocks, method, num_simulations=num_simulations, confidence_level=confidence_level)
    
    # Calculer les métriques de risque pour chaque simulation
    risk_metrics = calculate_risk_metrics(stressed_returns, confidence_level)
    
    return risk_metrics

def monte_carlo_simulation(returns, num_simulations=1000, volatility=None, correlation_matrix=None, confidence_level=0.95):
    mean_return = np.mean(returns)
    
    if volatility is None:
        volatility = np.std(returns)
    
    # Simulations Monte Carlo
    simulated_portfolio_values = []
    
    for _ in range(num_simulations):
        random_returns = np.random.normal(mean_return, volatility, len(returns))
        if correlation_matrix is not None:
            random_returns = np.dot(correlation_matrix, random_returns)
        simulated_portfolio_values.append(np.cumprod(1 + random_returns))
    
    # Calculer le percentile correspondant au niveau de confiance
    simulated_portfolio_values = np.array(simulated_portfolio_values)
    stressed_portfolio_values = np.percentile(simulated_portfolio_values, (1 - confidence_level) * 100, axis=0)
    
    return stressed_portfolio_values

def stress_test(data, shocks, weights=None, method='historical', confidence_level=0.95, num_simulations=1000):
    # Calculer les rendements du portefeuille
    returns = calculate_returns(data, weights)
    
    # Calculer la valeur initiale du portefeuille
    portfolio_value = (1 + returns).cumprod()
    
    # Appliquer les chocs
    stressed_returns = apply_shocks(returns, shocks, method, weights, num_simulations, confidence_level)
    
    # Calculer la valeur du portefeuille après chocs
    stressed_portfolio_value = (1 + stressed_returns).cumprod()
    
    # Validation des hypothèses (stationnarité et volatilité conditionnelle)
    stationarity_status, p_value = validate_stationarity(returns)
    garch_results = validate_volatility_condition(returns)
    
    # Calcul de la volatilité
    historical_volatility = calculate_volatility(returns, method='historical')
    conditional_volatility = calculate_volatility(returns, method='conditional')
    
    # Calcul des métriques de risque
    risk_metrics = calculate_risk_metrics(stressed_portfolio_value, confidence_level)
    
    # Tracer les courbes et afficher les résultats
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=portfolio_value.index, y=portfolio_value, mode='lines', name='Valeur du portefeuille (sans stress)'))
    fig.add_trace(go.Scatter(x=stressed_portfolio_value.index, y=stressed_portfolio_value, mode='lines', name='Valeur du portefeuille (avec stress)'))
    
    fig.update_layout(title='Stress Test', xaxis_title='Date', yaxis_title='Valeur du Portefeuille')
    fig.show()
    
    # Retourner tous les résultats pour un feedback complet
    return {
        'stressed_portfolio_value': stressed_portfolio_value,
        'risk_metrics': risk_metrics,
        'stationarity_status': stationarity_status,
        'stationarity_p_value': p_value,
        'garch_results': garch_results,
        'historical_volatility': historical_volatility,
        'conditional_volatility': conditional_volatility
    }