import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import yfinance as yf
import plotly.express as px
from scipy.stats import norm
import seaborn as sns
import plotly.graph_objects as go
from statsmodels.tsa.stattools import adfuller
import arch

def var_calculation(data, method, multi=True, weights=None, confidence_level=95, num_simulations=10000, days=10):
    data = data.dropna()
    
    # Calcul dans le cas où il y a plusieurs actifs
    if multi:
        # Mise en forme des poids
        if weights is None:
            weights = np.full(len(data.columns), 1/len(data.columns))
        weights = np.array(weights)
        
        if method == 'historical':
            # Pondération des actifs
            weighted_returns_portfolio = data.mul(weights, axis=1)
            
            # Sommer les rendements pondérés du portefeuille
            portfolio = weighted_returns_portfolio.sum(axis=1)
            
            # Calcul de la VaR à 1 jour (historique)
            var_historical = round(np.percentile(portfolio, (100 - confidence_level)), 2)
            
            # Calcul de la VaR future pour 10 jours (ajustée par la racine carrée du nombre de jours)
            portfolio_std = portfolio.std()  # écart-type des rendements
            var_future = round(np.percentile(portfolio, (100 - confidence_level)) * np.sqrt(days), 2)
            
            # Visualisation
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=portfolio,
                nbinsx=50,
                name='Rendements du portefeuille',
                opacity=0.75,
                histnorm='probability density',
            ))
            fig.add_trace(go.Scatter(
                x=[var_historical] * 2,
                mode='lines',
                line=dict(color='red', dash='dash'),
                name=f'VaR 1 jour à {confidence_level}%',
            ))
            fig.add_trace(go.Scatter(
                x=[var_future] * 2,
                mode='lines',
                line=dict(color='purple', dash='dash'),
                name=f'VaR future ({days} jours) à {confidence_level}%',
            ))
            fig.update_layout(
                title='Distribution des rendements du portefeuille',
                xaxis_title='Rendements (%)',
                yaxis_title='Densité de Fréquence',
                template='plotly_dark',
                showlegend=True,
                width=800,
                height=700
            )
            fig.show()
            return var_historical, var_future

        elif method == 'parametric':
            # Matrice de covariance
            cov_matrix = data.cov()
            # Moyenne des rendements
            avg_returns = data.mean()
            # Pondération des moyennes
            portfolio_mean = avg_returns @ weights
            # Ecart-type des rendements pondérés
            portfolio_std = np.sqrt(weights.T @ cov_matrix @ weights)

            # Calcul de la VaR à 1 jour (paramétrique)
            var_parametric = round(norm.ppf((100 - confidence_level) / 100, portfolio_mean, portfolio_std), 2)

            # Calcul de la VaR future pour 10 jours (ajustée par la racine carrée du nombre de jours)
            var_future = round(var_parametric * np.sqrt(days), 2)
            
            # Visualisation
            plt.figure(figsize=(10, 6))
            x = np.linspace(portfolio_mean - 3 * portfolio_std, portfolio_mean + 3 * portfolio_std, 1000)
            y = norm.pdf(x, portfolio_mean, portfolio_std)
            plt.plot(x, y, label='Distribution normale des rendements')
            plt.axvline(var_parametric, color='red', linestyle='--', label=f'VaR 1 jour ({confidence_level}%): {var_parametric}%')
            plt.axvline(var_future, color='purple', linestyle='--', label=f'VaR future ({days} jours) ({confidence_level}%): {var_future}%')
            plt.fill_between(x, 0, y, where=(x <= var_parametric), color='red', alpha=0.5)
            plt.fill_between(x, 0, y, where=(x <= var_future), color='purple', alpha=0.5)
            plt.title(f'Distribution des rendements avec VaR à 1 jour et VaR future ({days} jours)')
            plt.xlabel('Rendements (%)')
            plt.ylabel('Densité de Probabilité')
            plt.legend()
            plt.show()
            return var_parametric, var_future

        elif method == 'simulation':
            # Simulation de Monte Carlo
            simulated_returns = np.random.multivariate_normal(data.mean(), data.cov(), size=num_simulations)
            
            # Pondération des rendements simulés
            weighted_simulated_returns = np.dot(simulated_returns, weights)
            
            # Calcul de la VaR à 1 jour (simulation)
            var_simulation = round(np.percentile(weighted_simulated_returns, (100 - confidence_level)), 2)
            
            # Calcul de la VaR future pour 10 jours (ajustée par la racine carrée du nombre de jours)
            var_future = round(var_simulation * np.sqrt(days), 2)
            
            # Visualisation
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=weighted_simulated_returns,
                nbinsx=50,
                name='Simulations du portefeuille',
                opacity=0.75,
                histnorm='probability density',
            ))
            fig.add_trace(go.Scatter(
                x=[var_simulation] * 2,
                mode='lines',
                line=dict(color='red', dash='dash'),
                name=f'VaR 1 jour à {confidence_level}%',
            ))
            fig.add_trace(go.Scatter(
                x=[var_future] * 2,
                mode='lines',
                line=dict(color='purple', dash='dash'),
                name=f'VaR future ({days} jours) à {confidence_level}%',
            ))
            fig.update_layout(
                title='Distribution des simulations du portefeuille',
                xaxis_title='Rendements simulés (%)',
                yaxis_title='Densité de Fréquence',
                template='plotly_dark',
                showlegend=True,
                width=800,
                height=700
            )
            fig.show()
            return var_simulation, var_future
        
    else:
        # Cas où il n'y a qu'un seul actif
        if method == 'historical':
            # Calcul des rendements du portefeuille (ici, c'est directement les rendements de l'actif)
            portfolio = data
            # Calcul de la VaR à 1 jour (historique)
            var_historical = round(np.percentile(portfolio, (100 - confidence_level)), 2)
            
            # Calcul de la VaR future pour 10 jours
            portfolio_std = portfolio.std()
            var_future = round(var_historical * np.sqrt(days), 2)
            
            # Visualisation
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=portfolio,
                nbinsx=50,
                name='Rendements de l\'Actif',
                opacity=0.75,
                histnorm='probability density',
            ))
            fig.add_trace(go.Scatter(
                x=[var_historical] * 2,
                mode='lines',
                line=dict(color='red', dash='dash'),
                name=f'VaR 1 jour à {confidence_level}%',
            ))
            fig.add_trace(go.Scatter(
                x=[var_future] * 2,
                mode='lines',
                line=dict(color='purple', dash='dash'),
                name=f'VaR future ({days} jours) à {confidence_level}%',
            ))
            fig.update_layout(
                title='Distribution des Rendements de l\'Actif',
                xaxis_title='Rendement (%)',
                yaxis_title='Densité de Fréquence',
                template='plotly_dark',
                showlegend=True,
                width=800,
                height=700
            )
            fig.show()
            return var_historical, var_future

        elif method == 'parametric':
            # Calcul de la moyenne et de l'écart-type de l'actif
            mean_return = data.mean()
            std_dev = data.std()

            # Calcul de la VaR à 1 jour (paramétrique)
            var_parametric = round(norm.ppf((100 - confidence_level) / 100, mean_return, std_dev), 2)

            # Calcul de la VaR future pour 10 jours
            var_future = round(var_parametric * np.sqrt(days), 2)
            
            # Visualisation
            plt.figure(figsize=(10, 6))
            x = np.linspace(mean_return - 3 * std_dev, mean_return + 3 * std_dev, 1000)
            y = norm.pdf(x, mean_return, std_dev)
            plt.plot(x, y, label='Distribution normale des rendements')
            plt.axvline(var_parametric, color='red', linestyle='--', label=f'VaR 1 jour ({confidence_level}%): {var_parametric}%')
            plt.axvline(var_future, color='purple', linestyle='--', label=f'VaR future ({days} jours) ({confidence_level}%): {var_future}%')
            plt.fill_between(x, 0, y, where=(x <= var_parametric), color='red', alpha=0.5)
            plt.fill_between(x, 0, y, where=(x <= var_future), color='purple', alpha=0.5)
            plt.title(f'Distribution des rendements avec VaR à 1 jour et VaR future ({days} jours)')
            plt.xlabel('Rendements (%)')
            plt.ylabel('Densité de Probabilité')
            plt.legend()
            plt.show()
            return var_parametric, var_future

        elif method == 'simulation':
            # Simulation de Monte Carlo pour un seul actif
            simulated_returns = np.random.normal(data.mean(), data.std(), size=num_simulations)
            
            # Calcul de la VaR à 1 jour (simulation)
            var_simulation = round(np.percentile(simulated_returns, (100 - confidence_level)), 2)
            
            # Calcul de la VaR future pour 10 jours
            var_future = round(var_simulation * np.sqrt(days), 2)
            
            # Visualisation
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=simulated_returns,
                nbinsx=50,
                name='Simulations de l\'Actif',
                opacity=0.75,
                histnorm='probability density',
            ))
            fig.add_trace(go.Scatter(
                x=[var_simulation] * 2,
                mode='lines',
                line=dict(color='red', dash='dash'),
                name=f'VaR 1 jour à {confidence_level}%',
            ))
            fig.add_trace(go.Scatter(
                x=[var_future] * 2,
                mode='lines',
                line=dict(color='purple', dash='dash'),
                name=f'VaR future ({days} jours) à {confidence_level}%',
            ))
            fig.update_layout(
                title='Distribution des simulations de l\'Actif',
                xaxis_title='Rendements simulés (%)',
                yaxis_title='Densité de Fréquence',
                template='plotly_dark',
                showlegend=True,
                width=800,
                height=700
            )
            fig.show()
            return var_simulation, var_future