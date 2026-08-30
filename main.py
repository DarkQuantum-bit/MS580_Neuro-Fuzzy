# Bibliotecas principais para interface, manipulação de dados e modelagem
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
from anfis_model import ANFIS  # Modelo ANFIS customizado
import time

# Configuração da página Streamlit
st.set_page_config(layout="wide")
st.title("Comparação: Neuro-Fuzzy ANFIS vs MLP")

# === Seção de carregamento e preparação dos dados ===
st.header("Configuração dos Dados")
dataset = fetch_ucirepo(id=235)  # Carrega dataset de energia elétrica da UCI (Individual household electric power consumption)
df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)

# Converte colunas de data/hora para datetime
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
df.set_index('Datetime', inplace=True)
df.drop(columns=['Date', 'Time'], inplace=True)

# Remove entradas não numéricas (NaNs)
df = df.apply(pd.to_numeric, errors='coerce').dropna()

# Filtros de data definidos pelo usuário
start_date = st.date_input("Data inicial", datetime(2009, 1, 1))
end_date = st.date_input("Data final", datetime(2009, 1, 7))
df = df.loc[str(start_date):str(end_date)]  # Filtra o dataframe pelas datas

# Seleção de variáveis de entrada e alvo
features = st.multiselect("Variáveis de entrada", df.columns.tolist(), default=['Global_reactive_power', 'Voltage', 'Global_intensity'])
target = st.selectbox("Variável alvo", df.columns, index=df.columns.get_loc('Global_active_power'))

# === Configuração do modelo ANFIS ===
st.header("Configuração do Modelo ANFIS")
preset = st.selectbox("Escolha um perfil de configuração:", [
    "Rápido (2 regras, 50 épocas, LR=0.05)",
    "Balanceado (4 regras, 100 épocas, LR=0.01)",
    "Preciso (6 regras, 200 épocas, LR=0.005)"
])

# Define hiperparâmetros com base na escolha do usuário
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

# === Processamento dos dados ===
X = df[features].values
y = df[target].values

# Normalização dos dados com MinMaxScaler
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# Divisão em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Exibe estatísticas do conjunto de dados
st.write(f"Total de amostras utilizadas: {len(X)}")
st.write(f"Amostras de treino: {len(X_train)}")
st.write(f"Amostras de teste: {len(X_test)}")

# === Treinamento dos modelos ANFIS e MLP ===
if st.button("Treinar modelos"):
    # --- Treinamento ANFIS ---
    with st.spinner("Treinando ANFIS..."):
        start_anfis = time.time()
        anfis = ANFIS(X_train, y_train, n_rules, len(features))
        anfis.train(epochs=epochs, learning_rate=learning_rate)
        end_anfis = time.time()
        time_anfis = end_anfis - start_anfis

        # Previsões e desnormalização
        y_pred_anfis_scaled = anfis.predict(X_test)
        y_pred_anfis = scaler_y.inverse_transform(y_pred_anfis_scaled.reshape(-1, 1)).flatten()
        y_test_real = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # --- Treinamento MLP ---
    with st.spinner("Treinando MLP..."):
        start_mlp = time.time()
        mlp = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=1, warm_start=True,
                           learning_rate_init=learning_rate, random_state=42)
        mlp_loss = []

        # Treinamento manual por época (usando warm_start)
        for _ in range(epochs):
            mlp.fit(X_train, y_train)
            y_train_pred = mlp.predict(X_train)
            loss = mean_absolute_error(y_train, y_train_pred)
            mlp_loss.append(loss)

        end_mlp = time.time()
        time_mlp = end_mlp - start_mlp

        # Previsões e desnormalização
        y_pred_mlp_scaled = mlp.predict(X_test)
        y_pred_mlp = scaler_y.inverse_transform(y_pred_mlp_scaled.reshape(-1, 1)).flatten()

    # === Avaliação dos modelos ===
    st.subheader("Resultados")
    st.success(f"Tempo de treinamento do ANFIS: {time_anfis:.2f} segundos")
    st.success(f"Tempo de treinamento do MLP: {time_mlp:.2f} segundos")

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

    # === Visualização: Real vs Previsão ===
    st.subheader("Real vs ANFIS")
    fig_real_anfis, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(y_test_real, label="Real", alpha=0.7)
    ax1.plot(y_pred_anfis, label="ANFIS", alpha=0.7)
    ax1.set_title("Previsão com ANFIS", fontsize=14)
    ax1.set_xlabel("Amostras")
    ax1.set_ylabel("Consumo de Energia (kW)")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig_real_anfis)

    st.subheader("Real vs MLP")
    fig_real_mlp, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(y_test_real, label="Real", alpha=0.7)
    ax2.plot(y_pred_mlp, label="MLP", alpha=0.7)
    ax2.set_title("Previsão com MLP", fontsize=14)
    ax2.set_xlabel("Amostras")
    ax2.set_ylabel("Consumo de Energia (kW)")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig_real_mlp)

    # === Visualização: Curvas de convergência ===
    st.subheader("Curva de Convergência do ANFIS")
    fig2, ax2 = plt.subplots(figsize=(8, 3))
    ax2.plot(anfis.loss_history, marker='o', color='blue')
    ax2.set_xlabel("Épocas")
    ax2.set_ylabel("MAE")
    ax2.set_title("Convergência do ANFIS")
    ax2.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig2)

    st.subheader("Curva de Convergência do MLP")
    fig3, ax3 = plt.subplots(figsize=(8, 3))
    ax3.plot(mlp_loss, marker='s', color='orange')
    ax3.set_xlabel("Épocas")
    ax3.set_ylabel("MAE")
    ax3.set_title("Convergência do MLP")
    ax3.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig3)

    # === Gráficos de resíduos (erros) ===
    st.subheader("Resíduos do ANFIS")
    residuals_anfis = y_test_real - y_pred_anfis
    fig4, ax4 = plt.subplots(figsize=(8, 3))
    ax4.plot(residuals_anfis, color='blue', label='Resíduo')
    ax4.axhline(0, linestyle='--', color='gray')
    ax4.set_title("Resíduos ANFIS")
    ax4.set_xlabel("Amostras")
    ax4.set_ylabel("Erro (real - previsto)")
    ax4.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig4)

    st.subheader("Resíduos do MLP")
    residuals_mlp = y_test_real - y_pred_mlp
    fig5, ax5 = plt.subplots(figsize=(8, 3))
    ax5.plot(residuals_mlp, color='orange', label='Resíduo')
    ax5.axhline(0, linestyle='--', color='gray')
    ax5.set_title("Resíduos MLP")
    ax5.set_xlabel("Amostras")
    ax5.set_ylabel("Erro (real - previsto)")
    ax5.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig5)
