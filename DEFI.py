import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf

tickers = ['BABA', 'MELI', 'NU', 'AMT', 'QCOM', 'INTC', 'AAL', 'RTX']

# Descargar datos históricos
data = yf.download(tickers, start="2021-12-09", end="2024-09-18", group_by="ticker")

# Guardar los datos en un archivo CSV
data.to_csv("data_defi.csv")
data = pd.read_csv("data_defi.csv", header=[0, 1], index_col=0, parse_dates=True)

# Función para graficar precios históricos usando Plotly
def plot_closing_prices(ticker):
    closing_prices = data.xs('Close', axis=1, level=1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=closing_prices.index, y=closing_prices[ticker],
                             mode='lines', name=ticker))
    fig.update_layout(title=f'Precio de Cierre de {ticker}',
                      xaxis_title='Fecha', yaxis_title='Precio',
                      template='plotly_dark')
    return fig

# Función para graficar densidades de precios usando Plotly
def plot_density(ticker):
    closing_prices = data.xs('Close', axis=1, level=1)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=closing_prices[ticker].dropna(), nbinsx=50,
                               name='Densidad', histnorm='probability'))
    fig.update_layout(title=f'Densidad de Precios de Cierre de {ticker}',
                      xaxis_title='Precio', yaxis_title='Densidad',
                      template='plotly_dark')
    return fig

# Función para graficar la matriz de correlación usando Plotly
def plot_correlation():
    closing_prices = data.xs('Close', axis=1, level=1)
    returns = closing_prices.pct_change().apply(lambda x: np.log(1 + x)).dropna()
    corr_matrix = returns.corr()
    
    fig = go.Figure(data=go.Heatmap(z=corr_matrix.values, 
                                    x=corr_matrix.columns, 
                                    y=corr_matrix.columns, 
                                    colorscale='RdBu', 
                                    zmin=-1, zmax=1))
    fig.update_layout(title='Matriz de Correlación de Retornos Logarítmicos',
                      template='plotly_dark')
    return fig

# Función de simulación para el Modelo Geométrico Browniano
def simulacion(So, mu, sigma, days=90):
    escenario = [So]
    z = np.random.standard_normal(size=days)
    for i in range(days):
        St = So * np.exp(mu + sigma * z[i])
        escenario.append(St)
        So = St
    return escenario

# Función para generar escenarios por ticker
def generar_escenarios(ticker, num_escenarios=1000, days=90):
    closing_prices = data.xs('Close', axis=1, level=1)
    returns = closing_prices.pct_change().apply(lambda x: np.log(1 + x)).dropna()
    So = closing_prices[ticker].iloc[-1]
    mu = returns[ticker].mean()
    sigma = returns[ticker].std()
    escenarios = []
    for _ in range(num_escenarios):
        escenarios.append(simulacion(So, mu, sigma, days))
    return escenarios

# Función para graficar los escenarios simulados usando Plotly con animación
def plot_escenarios(ticker, num_escenarios=10000, days=90):
    escenarios = generar_escenarios(ticker, num_escenarios=num_escenarios, days=days)
    df = pd.DataFrame(escenarios).T
    
    fig = go.Figure()

    # Añadir traza para cada escenario
    for i in range(min(50, num_escenarios)):  # Graficar 50 escenarios para evitar sobrecargar el gráfico
        fig.add_trace(go.Scatter(x=list(range(days + 1)), y=df[i],
                                 mode='lines', line=dict(width=0.5), opacity=0.5))

    fig.update_layout(title=f'Escenarios Simulados para {ticker} (Modelo Geométrico Browniano)',
                      xaxis_title='Días', yaxis_title='Precio',
                      template='plotly_dark')
    return fig

# Función para graficar precio esperado, máximo y mínimo usando Plotly
def plot_precio_esperado(ticker, num_escenarios=1000, days=90):
    escenarios = generar_escenarios(ticker, num_escenarios=num_escenarios, days=days)
    df = pd.DataFrame(escenarios).T
    
    media = df.mean(axis=1)
    upper = df.quantile(0.95, axis=1)
    lower = df.quantile(0.05, axis=1)

    fig = go.Figure()

    # Precio esperado (media)
    fig.add_trace(go.Scatter(x=list(range(days + 1)), y=media,
                             mode='lines', name="Precio Esperado", line=dict(color='blue')))
    
    # Límite superior (95%)
    fig.add_trace(go.Scatter(x=list(range(days + 1)), y=upper,
                             mode='lines', name="Límite Superior (95%)", 
                             line=dict(color='green', dash='dash')))
    
    # Límite inferior (5%)
    fig.add_trace(go.Scatter(x=list(range(days + 1)), y=lower,
                             mode='lines', name="Límite Inferior (5%)", 
                             line=dict(color='red', dash='dash')))

    fig.update_layout(title=f'Predicción de Precios Esperados para {ticker}',
                      xaxis_title='Días', yaxis_title='Precio',
                      template='plotly_dark')
    
    return fig

# Streamlit Dashboard
st.title("Dashboard Interactivo de Acciones")
st.sidebar.title("Seleccionar Ticker")
ticker = st.sidebar.selectbox("Elige un ticker", tickers)

st.header(f"Análisis del Ticker {ticker}")

# Precios históricos
st.subheader("Precios de Cierre")
st.plotly_chart(plot_closing_prices(ticker))

# Densidad de precios
st.subheader("Densidad de Precios")
st.plotly_chart(plot_density(ticker))

# Matriz de correlación
st.subheader("Matriz de Correlación")
st.plotly_chart(plot_correlation())

# Simulación de Modelo Geométrico Browniano
st.subheader("Simulación de Escenarios (Modelo Geométrico Browniano)")
st.plotly_chart(plot_escenarios(ticker))

# Precios esperados, máximo y mínimo
st.subheader("Precio Esperado, Máximo y Mínimo")
st.plotly_chart(plot_precio_esperado(ticker))
