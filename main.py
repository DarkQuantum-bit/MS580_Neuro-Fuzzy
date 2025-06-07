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
st.title("📈 Comparação: Neuro-Fuzzy ANFIS vs MLP")

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
st.header("⚙️ Configuração do Modelo ANFIS")
preset = st.selectbox("Escolha um perfil de configuração:", [
    "Rápido (2 regras, 50 épocas, LR=0.05)",
    "Balanceado (4 regras, 100 épocas, LR=0.01)",
    "Preciso (6 regras, 200 épocas, LR=0.005)"
])

if preset == "Rápido (2 regras, 50 épocas, LR=0.05)":
    n_rules = 2
    epochs = 50
    learning_rate = 0.05
elif preset == "Balanceado (4 regras, 100 épocas, LR=0.01)":
    n_rules = 4
    epochs = 100
    learning_rate = 0.01
else:
    n_rules = 6
    epochs = 200
    learning_rate = 0.005

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
        mlp = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=1, warm_start=True,
                           learning_rate_init=learning_rate, random_state=42)
        mlp_loss = []
        for _ in range(epochs):
            mlp.fit(X_train, y_train)
            y_train_pred = mlp.predict(X_train)
            loss = mean_absolute_error(y_train, y_train_pred)
            mlp_loss.append(loss)
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

    # --- Comparativos Real vs Previsao ---
    st.subheader("📊 Real vs ANFIS")
    fig_real_anfis, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(y_test_real, label="Real", alpha=0.7)
    ax1.plot(y_pred_anfis, label="ANFIS", alpha=0.7)
    ax1.set_title("Previsão com ANFIS", fontsize=14)
    ax1.set_xlabel("Amostras")
    ax1.set_ylabel("Consumo de Energia (kW)")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig_real_anfis)

    st.subheader("📊 Real vs MLP")
    fig_real_mlp, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(y_test_real, label="Real", alpha=0.7)
    ax2.plot(y_pred_mlp, label="MLP", alpha=0.7)
    ax2.set_title("Previsão com MLP", fontsize=14)
    ax2.set_xlabel("Amostras")
    ax2.set_ylabel("Consumo de Energia (kW)")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig_real_mlp)

    # --- Curvas de Convergência ---
    st.subheader("📉 Curva de Convergência do ANFIS")
    fig2, ax2 = plt.subplots(figsize=(8, 3))
    ax2.plot(anfis.loss_history, marker='o', color='blue')
    ax2.set_xlabel("Épocas")
    ax2.set_ylabel("MAE")
    ax2.set_title("Convergência do ANFIS")
    ax2.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig2)

    st.subheader("📉 Curva de Convergência do MLP")
    fig3, ax3 = plt.subplots(figsize=(8, 3))
    ax3.plot(mlp_loss, marker='s', color='orange')
    ax3.set_xlabel("Épocas")
    ax3.set_ylabel("MAE")
    ax3.set_title("Convergência do MLP")
    ax3.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig3)
