import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from ucimlrepo import fetch_ucirepo
from anfis_model import ANFIS

st.set_page_config(layout="wide")
st.title("🔧 Neuro-Fuzzy ANFIS vs MLP")

# --- DADOS ---
st.header("📊 Configuração dos Dados")
dataset = fetch_ucirepo(id=235)
df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
df.set_index('Datetime', inplace=True)
df.drop(columns=['Date', 'Time'], inplace=True)
df = df.apply(pd.to_numeric, errors='coerce').dropna()

start_date = st.date_input("Data inicial", datetime(2009, 1, 1))
end_date = st.date_input("Data final", datetime(2009, 1, 7))
df = df.loc[str(start_date):str(end_date)]

features = st.multiselect("Variáveis de entrada", df.columns.tolist(), default=['Global_reactive_power', 'Voltage', 'Global_intensity'])
target = st.selectbox("Variável alvo", df.columns, index=df.columns.get_loc('Global_active_power'))

# --- CONFIGURAÇÃO DO MODELO ---
st.header("⚙️ Parâmetros da ANFIS")
col1, col2 = st.columns(2)
with col1:
    n_rules = st.slider("Número de regras fuzzy", 2, 10, 4)
    mf_type = st.selectbox("Tipo de função de pertinência", ["gaussiana"])  # Futuro: triangular, trapezoidal
with col2:
    epochs = st.slider("Épocas", 10, 500, 100, step=10)
    learning_rate = st.slider("Taxa de aprendizado", 0.001, 0.1, 0.01)

# --- PROCESSAMENTO ---
X = df[features].values
y = df[target].values
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

if st.button("🚀 Treinar modelos"):
    with st.spinner("Treinando ANFIS..."):
        anfis = ANFIS(X_train, y_train, n_rules, len(features))
        anfis.train(epochs=epochs, learning_rate=learning_rate)
        y_pred_anfis_scaled = anfis.predict(X_test)
        y_pred_anfis = scaler_y.inverse_transform(y_pred_anfis_scaled.reshape(-1, 1)).flatten()
        y_test_real = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    with st.spinner("Treinando MLP..."):
        mlp = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=epochs, learning_rate_init=learning_rate, random_state=42)
        mlp.fit(X_train, y_train)
        y_pred_mlp_scaled = mlp.predict(X_test)
        y_pred_mlp = scaler_y.inverse_transform(y_pred_mlp_scaled.reshape(-1, 1)).flatten()

    # --- Resultados ---
    st.subheader("📈 Resultados")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ANFIS")
        st.write(f"MAE: {mean_absolute_error(y_test_real, y_pred_anfis):.4f}")
        st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test_real, y_pred_anfis)):.4f}")
        st.write(f"R²: {r2_score(y_test_real, y_pred_anfis):.4f}")
    with col2:
        st.markdown("### MLP")
        st.write(f"MAE: {mean_absolute_error(y_test_real, y_pred_mlp):.4f}")
        st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test_real, y_pred_mlp)):.4f}")
        st.write(f"R²: {r2_score(y_test_real, y_pred_mlp):.4f}")

    # --- Gráfico comparativo ---
    st.subheader("🔍 Comparativo Visual")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(y_test_real, label="Real", alpha=0.7)
    ax.plot(y_pred_anfis, label="ANFIS", alpha=0.7)
    ax.plot(y_pred_mlp, label="MLP", alpha=0.7)
    ax.set_xlabel("Amostras")
    ax.set_ylabel("Consumo de Energia (kW)")
    ax.legend()
    st.pyplot(fig)

    # --- Curva de Convergência ---
    st.subheader("📉 Curva de Convergência do ANFIS")
    fig2, ax2 = plt.subplots()
    ax2.plot(anfis.loss_history, marker='o')
    ax2.set_xlabel("Épocas")
    ax2.set_ylabel("MAE")
    ax2.set_title("Convergência do ANFIS")
    st.pyplot(fig2)