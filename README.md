
# MS580 — Sistemas Neuro-Fuzzy: ANFIS vs. MLP

**Implementação e avaliação comparativa de um sistema de inferência neuro-fuzzy adaptativo e uma rede neural Multilayer Perceptron para regressão de dados de consumo energético.**

Projeto acadêmico desenvolvido para a disciplina **MS580 — Lógica Fuzzy**, com foco no estudo de sistemas de inferência fuzzy, redes neurais artificiais e arquiteturas híbridas de inteligência computacional.

O trabalho apresenta uma implementação própria de um **Adaptive Neuro-Fuzzy Inference System (ANFIS)**, utilizando Python e NumPy, e uma comparação experimental com uma rede neural **Multilayer Perceptron (MLP)** implementada com scikit-learn.

A aplicação utiliza dados reais de consumo de energia elétrica residencial disponibilizados pelo **UCI Machine Learning Repository**, permitindo configurar modelos, realizar treinamentos e comparar desempenho preditivo, convergência e tempo computacional por meio de uma interface Streamlit.

## 1. Introdução

Sistemas neuro-fuzzy são modelos híbridos de inteligência computacional que integram conceitos de **lógica fuzzy** e **redes neurais artificiais**.

A lógica fuzzy permite representar graus de pertinência e relações aproximadas, oferecendo uma estrutura matemática para modelar conceitos imprecisos e regras linguísticas.

As redes neurais artificiais, por sua vez, utilizam processos de aprendizado baseados em dados para ajustar parâmetros e construir aproximações de funções.

A combinação dessas abordagens permite construir sistemas capazes de aprender parâmetros de regras fuzzy a partir de exemplos.

Entre as arquiteturas neuro-fuzzy, destaca-se o **Adaptive Neuro-Fuzzy Inference System (ANFIS)**, introduzido por Jang (1993).

O ANFIS combina mecanismos de inferência fuzzy com procedimentos de aprendizagem paramétrica, permitindo adaptar funções de pertinência e consequentes de regras durante o treinamento.

Neste projeto, essa arquitetura é utilizada para resolver um problema de regressão e comparada a uma rede neural MLP convencional.

Os principais aspectos investigados são:

- Representação de conhecimento por regras fuzzy;
- Aprendizagem de funções de pertinência;
- Inferência fuzzy do tipo Takagi-Sugeno;
- Ajuste de parâmetros por métodos baseados em gradientes;
- Comparação de modelos neuro-fuzzy e redes neurais convencionais;
- Avaliação de desempenho preditivo e custo computacional.

## 2. Objetivos

O objetivo geral do projeto é implementar e avaliar um sistema neuro-fuzzy adaptativo aplicado à regressão de dados reais.

Os objetivos específicos incluem:

1. Estudar os fundamentos matemáticos da lógica fuzzy e dos sistemas de inferência;
2. Implementar uma arquitetura ANFIS utilizando NumPy;
3. Desenvolver funções de pertinência gaussianas parametrizadas;
4. Implementar regras fuzzy com consequentes lineares;
5. Construir um procedimento de treinamento para atualização dos parâmetros;
6. Implementar uma rede MLP como modelo de comparação;
7. Avaliar os modelos com métricas de regressão;
8. Investigar curvas de convergência, resíduos e tempos de treinamento;
9. Disponibilizar uma interface interativa para experimentação.

## 3. Fundamentação teórica

### 3.1. Lógica fuzzy

A lógica fuzzy generaliza a lógica clássica ao permitir que proposições apresentem graus de verdade intermediários entre falso e verdadeiro.

Na teoria clássica dos conjuntos, uma função característica determina se um elemento pertence ou não a determinado conjunto.

Na teoria dos conjuntos fuzzy, a pertinência é representada por uma função:

```math
\mu_A:X\rightarrow[0,1]
```

onde:

- `X` representa o universo de discurso;
- `A` representa um conjunto fuzzy;
- `μ_A(x)` representa o grau de pertinência de `x` ao conjunto `A`.

Valores próximos de zero indicam baixa pertinência, enquanto valores próximos de um indicam alta pertinência.

Essa abordagem permite representar categorias graduais e relações não estritamente binárias.

### 3.2. Funções de pertinência gaussianas

O modelo ANFIS implementado utiliza funções de pertinência gaussianas.

Sua formulação matemática usual é:

```math
\mu(x;c,\sigma)=
\exp\left(
-\frac{(x-c)^2}{2\sigma^2}
\right)
```

onde:

- `c` representa o centro da função;
- `σ` representa o parâmetro de dispersão;
- `x` representa o valor da variável de entrada.

Na implementação, um termo de estabilização numérica é acrescentado ao denominador:

```math
\mu(x;c,\sigma)=
\exp\left(
-\frac{(x-c)^2}{2\sigma^2+\varepsilon}
\right)
```

com:

```math
\varepsilon=10^{-6}
```

O parâmetro `c` controla a posição da função de pertinência, enquanto `σ` influencia sua largura.

Esses parâmetros são ajustados durante o treinamento do modelo.

### 3.3. Sistemas de inferência Takagi-Sugeno

Os sistemas de inferência fuzzy do tipo Takagi-Sugeno utilizam regras cujos consequentes são funções das variáveis de entrada.

Uma regra genérica pode ser representada por:

**Regra r:**

SE `x_1` pertence a `A_1r` E `x_2` pertence a `A_2r` E ... E `x_n` pertence a `A_nr`,

ENTÃO:

```math
f_r(\mathbf{x})=
\sum_{j=1}^{n}a_{rj}x_j+b_r
```

onde:

- `x_j` representa a j-ésima entrada;
- `a_rj` representa um coeficiente linear da regra;
- `b_r` representa o termo constante;
- `f_r` representa o consequente da regra.

A saída global é calculada por uma combinação ponderada das saídas das regras.

### 3.4. Adaptive Neuro-Fuzzy Inference System

O ANFIS integra sistemas de inferência fuzzy com mecanismos de aprendizagem paramétrica.

Sua estrutura pode ser descrita em cinco etapas conceituais:

1. Fuzzificação das entradas;
2. Cálculo das forças de ativação das regras;
3. Normalização das forças de ativação;
4. Avaliação dos consequentes;
5. Agregação das saídas das regras.

Essa organização permite separar os parâmetros em duas categorias principais.

**Parâmetros antecedentes:** controlam as funções de pertinência utilizadas na avaliação das condições das regras.

**Parâmetros consequentes:** controlam as funções utilizadas para produzir as saídas das regras.

A implementação deste projeto utiliza atualização iterativa de ambas as categorias.

## 4. Arquitetura ANFIS implementada

O modelo está definido na classe `ANFIS`, disponível no arquivo [`anfis_model.py`](anfis_model.py).

A implementação utiliza NumPy para realizar as operações numéricas e construir o processo de inferência e aprendizagem.

### 4.1. Inicialização dos parâmetros

A arquitetura recebe os seguintes argumentos:

```python
ANFIS(
    input_data,
    output_data,
    n_rules,
    n_inputs
)
```

onde:

- `input_data`: matriz de atributos de treinamento;
- `output_data`: valores-alvo;
- `n_rules`: quantidade de regras fuzzy;
- `n_inputs`: quantidade de variáveis de entrada.

Os parâmetros das funções de pertinência são armazenados em uma matriz de dimensão:

```math
n_{\mathrm{inputs}}\times n_{\mathrm{rules}}\times 2
```

O último eixo contém o centro e o parâmetro de dispersão de cada função gaussiana.

Os parâmetros consequentes possuem dimensão:

```math
n_{\mathrm{rules}}\times(n_{\mathrm{inputs}}+1)
```

Cada regra possui um coeficiente para cada entrada e um termo independente.

Os parâmetros são inicialmente gerados de forma aleatória.

### 4.2. Fuzzificação

Para cada entrada e regra, calcula-se o grau de pertinência correspondente.

```math
\mu_{rj}(x_j)=
\exp\left(
-\frac{(x_j-c_{rj})^2}
{2\sigma_{rj}^{2}+\varepsilon}
\right)
```

onde:

- `r` identifica a regra fuzzy;
- `j` identifica a variável de entrada;
- `c_rj` representa o centro;
- `σ_rj` representa o parâmetro de dispersão.

A função `gaussian_mf()` implementa essa operação.

### 4.3. Ativação das regras

A força de ativação de uma regra é calculada pelo produto dos graus de pertinência de todas as variáveis de entrada.

```math
w_r=
\prod_{j=1}^{n}\mu_{rj}(x_j)
```

Essa operação corresponde à utilização do produto como operador de conjunção fuzzy.

A função `rule_evaluation()` calcula as forças de ativação de todas as regras.

### 4.4. Normalização

As forças de ativação são normalizadas pela soma dos valores calculados.

```math
\bar{w}_r=
\frac{w_r}{\sum_{k=1}^{R}w_k}
```

onde `R` representa a quantidade de regras.

A normalização faz com que, quando o denominador é positivo, as forças normalizadas somem um.

Na implementação, quando a soma das forças de ativação é exatamente zero, o método `normalize()` retorna o vetor original.

### 4.5. Consequentes das regras

Cada regra possui um consequente linear:

```math
f_r(\mathbf{x})=
\sum_{j=1}^{n}a_{rj}x_j+b_r
```

Os coeficientes são armazenados em `rule_params`.

Essa estrutura corresponde a um sistema Takagi-Sugeno de primeira ordem.

### 4.6. Agregação e saída

A predição final é obtida pela soma ponderada dos consequentes, utilizando as forças de ativação normalizadas.

```math
\hat{y}=
\sum_{r=1}^{R}\bar{w}_r f_r(\mathbf{x})
```

De forma expandida:

```math
\hat{y}=
\sum_{r=1}^{R}
\bar{w}_r
\left(
\sum_{j=1}^{n}a_{rj}x_j+b_r
\right)
```

A implementação utiliza os métodos:

- `defuzzify()`: combinação das saídas das regras;
- `forward_pass()`: inferência para uma amostra;
- `predict()`: inferência para um conjunto de amostras.

## 5. Treinamento do ANFIS

O treinamento é implementado diretamente na classe `ANFIS`, por meio do método `train()`.

O procedimento percorre as amostras de treinamento e realiza atualizações iterativas dos parâmetros do modelo.

### 5.1. Erro de predição

Para uma amostra, o erro residual é definido por:

```math
e_i=y_i-\hat{y}_i
```

onde:

- `y_i` representa o valor observado;
- `ŷ_i` representa o valor predito.

O histórico de treinamento utiliza o erro absoluto médio por época:

```math
\mathrm{MAE}=
\frac{1}{N}
\sum_{i=1}^{N}
|y_i-\hat{y}_i|
```

### 5.2. Atualização dos consequentes

Os coeficientes lineares das regras são atualizados a partir do erro, das entradas e das forças de ativação normalizadas.

Para o coeficiente associado à entrada `j` da regra `r`:

```math
a_{rj}^{\mathrm{novo}}=
a_{rj}+
\eta e_i\bar{w}_r x_{ij}
```

Para o termo constante:

```math
b_r^{\mathrm{novo}}=
b_r+\eta e_i\bar{w}_r
```

onde `η` representa a taxa de aprendizagem.

Essas expressões correspondem às atualizações explícitas implementadas no código.

### 5.3. Atualização dos antecedentes

Os centros e parâmetros de dispersão das funções gaussianas também são atualizados durante o treinamento.

De maneira geral, essas atualizações podem ser representadas por:

```math
c_{rj}^{\mathrm{novo}}=
c_{rj}+
\eta e_i g_{c_{rj}}
```

```math
\sigma_{rj}^{\mathrm{novo}}=
\sigma_{rj}+
\eta e_i g_{\sigma_{rj}}
```

onde os termos `g` representam as expressões de derivadas aproximadas utilizadas no procedimento de atualização.

A implementação calcula essas expressões explicitamente em NumPy.

**Observação técnica:** o treinamento utiliza um procedimento próprio de atualização de parâmetros, inspirado em métodos baseados em gradientes. Ele não implementa o algoritmo híbrido clássico de ANFIS com mínimos quadrados para os consequentes e gradiente para os antecedentes.

Além disso, as expressões de atualização dos antecedentes merecem validação por diferenciação numérica, especialmente quanto aos termos de normalização e estabilidade dos parâmetros de dispersão.

### 5.4. Histórico de convergência

Ao final de cada época, o erro absoluto médio de treinamento é armazenado em:

```python
anfis.loss_history
```

Esse histórico permite visualizar a evolução do erro durante a aprendizagem.

## 6. Modelo de comparação: Multilayer Perceptron

O projeto utiliza uma rede neural **Multilayer Perceptron (MLP)** como referência para comparação com o ANFIS.

A implementação é baseada em `MLPRegressor`, da biblioteca scikit-learn.

A arquitetura possui:

- Uma camada de entrada com dimensão determinada pelo número de atributos;
- Duas camadas ocultas com dez neurônios cada;
- Uma camada de saída para regressão.

A configuração da rede inclui:

```python
MLPRegressor(
    hidden_layer_sizes=(10, 10),
    max_iter=1,
    warm_start=True,
    learning_rate_init=learning_rate,
    random_state=42
)
```

O parâmetro `warm_start=True` permite reutilizar os parâmetros aprendidos em chamadas sucessivas de treinamento.

A aplicação realiza um laço com o número de épocas selecionado, executando `fit()` sucessivamente.

### 6.1. Formulação matemática

Para uma rede neural feedforward, a transformação de uma camada pode ser representada por:

```math
\mathbf{h}^{(l)}=
\phi\left(
W^{(l)}\mathbf{h}^{(l-1)}+
\mathbf{b}^{(l)}
\right)
```

onde:

- `W` representa a matriz de pesos;
- `b` representa o vetor de vieses;
- `φ` representa a função de ativação;
- `h` representa a saída de uma camada.

A saída final é produzida pelas transformações sucessivas da rede.

Diferentemente do ANFIS, a MLP não utiliza explicitamente regras fuzzy nem funções de pertinência na construção de sua inferência.

## 7. Base de dados

O projeto utiliza o dataset **Individual Household Electric Power Consumption**, disponibilizado pelo UCI Machine Learning Repository.

**Fonte:** [UCI Machine Learning Repository — Dataset 235](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)

O conjunto contém registros de medições elétricas de uma residência, incluindo variáveis de potência, tensão, intensidade de corrente e consumo em diferentes circuitos.

A base apresenta medições realizadas aproximadamente em intervalos de um minuto, abrangendo o período entre dezembro de 2006 e novembro de 2010.

### 7.1. Variáveis

Entre as variáveis disponíveis estão:

| Variável | Descrição |
|---|---|
| `Global_active_power` | Potência ativa global, em kW |
| `Global_reactive_power` | Potência reativa global, em kVAr |
| `Voltage` | Tensão elétrica, em V |
| `Global_intensity` | Intensidade de corrente global, em A |
| `Sub_metering_1` | Energia medida no primeiro circuito |
| `Sub_metering_2` | Energia medida no segundo circuito |
| `Sub_metering_3` | Energia medida no terceiro circuito |

As três variáveis de submedição são expressas em Wh de energia ativa por minuto.

### 7.2. Variáveis padrão

A configuração inicial utiliza como entradas:

- `Global_reactive_power`;
- `Voltage`;
- `Global_intensity`.

A variável-alvo padrão é:

```text
Global_active_power
```

A interface permite selecionar outras combinações de atributos e variável-alvo.

### 7.3. Recorte temporal

O período inicialmente selecionado na aplicação é:

- Data inicial: 01/01/2009;
- Data final: 07/01/2009.

As datas podem ser modificadas pelo usuário.

O recorte permite reduzir o volume de dados utilizado em cada experimento e investigar diferentes intervalos da série.

### 7.4. Pré-processamento

A preparação dos dados inclui:

1. Carregamento pelo pacote `ucimlrepo`;
2. Conversão das colunas de data e hora;
3. Construção de um índice temporal;
4. Conversão dos atributos para valores numéricos;
5. Remoção de registros com valores ausentes;
6. Filtragem pelo intervalo de datas;
7. Seleção dos atributos e da variável-alvo;
8. Normalização com `MinMaxScaler`;
9. Divisão em conjuntos de treinamento e teste.

A normalização Min-Max pode ser representada por:

```math
x'=
\frac{x-x_{\min}}
{x_{\max}-x_{\min}}
```

para atributos com amplitude não nula.

O código utiliza normalizadores separados para entradas e saída.

A divisão treino-teste é realizada com:

```python
train_test_split(
    X_scaled,
    y_scaled,
    test_size=0.2,
    random_state=42
)
```

Portanto, aproximadamente 80% das amostras são destinadas ao treinamento e 20% à avaliação.

**Limitação metodológica:** o código ajusta os normalizadores antes da separação treino-teste. Isso pode introduzir vazamento de informações. Além disso, a divisão aleatória não preserva a ordem cronológica, característica relevante em problemas de previsão temporal.

A implementação deve ser interpretada como um experimento comparativo de regressão, não como uma validação rigorosa de previsão futura de séries temporais.

## 8. Configurações experimentais

A interface disponibiliza três perfis de treinamento.

| Configuração | Regras ANFIS | Épocas | Taxa de aprendizagem |
|---|---:|---:|---:|
| Rápido | 2 | 50 | 0,05 |
| Balanceado | 4 | 100 | 0,01 |
| Preciso | 6 | 200 | 0,005 |

A nomenclatura dos perfis identifica configurações de complexidade e treinamento. Ela não garante que o perfil denominado "Preciso" obtenha o menor erro em todos os experimentos.

A quantidade de regras afeta diretamente o número de parâmetros e a estrutura do sistema de inferência.

Para `n` variáveis de entrada e `R` regras, o número de parâmetros do ANFIS implementado é:

```math
P=2nR+R(n+1)
```

Equivalentemente:

```math
P=R(3n+1)
```

Essa contagem inclui os parâmetros gaussianos e os coeficientes dos consequentes lineares.

Para três variáveis de entrada:

| Regras | Parâmetros ANFIS |
|---|---:|
| 2 | 20 |
| 4 | 40 |
| 6 | 60 |

A MLP mantém a arquitetura de duas camadas ocultas com dez neurônios, enquanto a quantidade de regras do ANFIS varia de acordo com o perfil selecionado.

## 9. Métricas de avaliação

A aplicação compara os modelos utilizando três métricas de regressão.

### 9.1. Mean Absolute Error (MAE)

O erro absoluto médio é definido por:

```math
\mathrm{MAE}=
\frac{1}{N}
\sum_{i=1}^{N}
|y_i-\hat{y}_i|
```

Essa métrica representa a magnitude média dos erros de predição.

Valores menores indicam melhor desempenho segundo esse critério.

### 9.2. Root Mean Squared Error (RMSE)

A raiz do erro quadrático médio é definida por:

```math
\mathrm{RMSE}=
\sqrt{
\frac{1}{N}
\sum_{i=1}^{N}
(y_i-\hat{y}_i)^2
}
```

Por utilizar erros elevados ao quadrado, essa métrica atribui maior peso a desvios de grande magnitude.

### 9.3. Coeficiente de determinação

O coeficiente de determinação é dado por:

```math
R^2=
1-
\frac{
\sum_{i=1}^{N}(y_i-\hat{y}_i)^2
}{
\sum_{i=1}^{N}(y_i-\bar{y})^2
}
```

onde `ȳ` representa a média dos valores observados.

Valores próximos de um indicam melhor ajuste segundo esse critério. Valores negativos também são possíveis quando as predições são piores que a referência baseada na média.

### 9.4. Tempo de treinamento

O tempo de execução dos procedimentos de treinamento é medido utilizando `time.time()`.

Esse indicador permite comparar o custo computacional observado nas condições de execução.

Os tempos não representam medidas universais de eficiência, pois dependem do hardware, do ambiente e dos procedimentos de treinamento utilizados.

## 10. Visualizações e análise dos resultados

A aplicação apresenta visualizações produzidas com Matplotlib.

### 10.1. Valores reais e preditos

São produzidos gráficos para comparar os valores observados com as predições de cada modelo.

As visualizações permitem examinar diferenças entre:

- Dados reais e resultados do ANFIS;
- Dados reais e resultados da MLP.

### 10.2. Curvas de convergência

A interface apresenta a evolução do erro de treinamento ao longo das épocas.

Para o ANFIS, o histórico é registrado em `loss_history`.

Para a MLP, a aplicação calcula o MAE no conjunto de treinamento após cada chamada de `fit()`.

As curvas permitem investigar o comportamento dos procedimentos de treinamento.

### 10.3. Análise de resíduos

Os resíduos são definidos por:

```math
e_i=y_i-\hat{y}_i
```

A aplicação apresenta gráficos de resíduos para ambos os modelos.

Esses gráficos permitem investigar a distribuição dos erros, identificar padrões sistemáticos e analisar possíveis discrepâncias entre as predições e os valores observados.

### 10.4. Comparação computacional

Além das métricas de regressão, são apresentados os tempos de treinamento dos dois modelos.

Essa análise permite investigar compromissos entre desempenho preditivo e custo computacional.

## 11. Interface interativa

O projeto utiliza Streamlit para disponibilizar um ambiente de experimentação com os modelos.

A interface está organizada em etapas.

### 11.1. Configuração dos dados

Permite selecionar:

- Intervalo de datas;
- Variáveis de entrada;
- Variável-alvo.

A aplicação apresenta a quantidade total de amostras utilizadas e a distribuição entre treinamento e teste.

### 11.2. Configuração do ANFIS

Permite escolher entre os perfis:

- Rápido;
- Balanceado;
- Preciso.

Cada perfil determina a quantidade de regras, as épocas e a taxa de aprendizagem.

### 11.3. Treinamento

O botão **Treinar modelos** inicia o treinamento do ANFIS e da MLP.

Ao final, são apresentados os tempos de treinamento, as métricas de regressão e as visualizações dos resultados.

## 12. Estrutura do repositório

```text
MS580_Neuro-Fuzzy/
│
├── anfis_model.py
├── main.py
├── requirements.txt
│
├── Monografia_MS580.pdf
├── Documentação do código.pdf
│
└── README.md
```

### Arquivos principais

| Arquivo | Descrição |
|---|---|
| `anfis_model.py` | Implementação própria da arquitetura ANFIS |
| `main.py` | Interface Streamlit, preparação dos dados, treinamento e avaliação |
| `requirements.txt` | Dependências Python |
| `Monografia_MS580.pdf` | Monografia acadêmica associada ao projeto |
| `Documentação do código.pdf` | Documentação complementar da implementação |
| `README.md` | Apresentação e documentação geral do repositório |

Para informações adicionais sobre o desenvolvimento acadêmico, consulte a [monografia](Monografia_MS580.pdf) e a [documentação do código](Documenta%C3%A7%C3%A3o%20do%20c%C3%B3digo.pdf).

## 13. Tecnologias utilizadas

| Tecnologia | Aplicação |
|---|---|
| Python | Linguagem principal |
| NumPy | Implementação matemática do ANFIS |
| scikit-learn | MLP, normalização, divisão dos dados e métricas |
| Pandas | Manipulação e preparação de dados |
| Streamlit | Interface interativa |
| Matplotlib | Visualização de resultados |
| ucimlrepo | Carregamento do dataset da UCI |

A implementação do ANFIS utiliza NumPy diretamente, sem depender de um framework de diferenciação automática.

## 14. Instalação e execução

### 14.1. Pré-requisitos

- Python 3;
- Pip;
- Ambiente virtual recomendado;
- Acesso à internet para baixar o dataset da UCI.

### 14.2. Clonar o repositório

```bash
git clone https://github.com/DarkQuantum-bit/MS580_Neuro-Fuzzy.git
```

Acesse o diretório:

```bash
cd MS580_Neuro-Fuzzy
```

### 14.3. Criar ambiente virtual

```bash
python -m venv .venv
```

No Windows:

```powershell
.venv\Scripts\activate
```

No Linux ou macOS:

```bash
source .venv/bin/activate
```

### 14.4. Instalar dependências

```bash
python -m pip install -r requirements.txt
```

### 14.5. Executar a aplicação

```bash
streamlit run main.py
```

O Streamlit disponibilizará a interface em um endereço local, normalmente:

http://localhost:8501

### 14.6. Utilização

1. Selecione o intervalo de datas;
2. Configure as variáveis de entrada e a variável-alvo;
3. Escolha o perfil de treinamento;
4. Clique em **Treinar modelos**;
5. Aguarde a execução dos dois modelos;
6. Compare as métricas e os tempos;
7. Analise as curvas de convergência e os resíduos.

O carregamento da base depende da disponibilidade do repositório de dados da UCI e da conexão com a internet.

## 15. Experimentos possíveis

O sistema permite investigar diferentes configurações de aprendizagem e representação.

### 15.1. Quantidade de regras fuzzy

Comparar a influência da quantidade de regras sobre o erro de regressão e o tempo computacional.

### 15.2. Taxa de aprendizagem

Analisar o comportamento do treinamento sob diferentes taxas de atualização dos parâmetros.

Na implementação atual, a taxa de aprendizagem é selecionada por meio dos perfis predefinidos.

### 15.3. Seleção de atributos

Investigar como diferentes combinações de variáveis elétricas afetam a capacidade preditiva dos modelos.

### 15.4. Recortes temporais

Executar experimentos com diferentes intervalos de datas, comparando o comportamento dos modelos em subconjuntos distintos do dataset.

### 15.5. ANFIS vs. MLP

Comparar:

- MAE;
- RMSE;
- R²;
- Tempo de treinamento;
- Convergência;
- Distribuição dos resíduos.

Esses experimentos permitem explorar diferenças entre uma arquitetura baseada em regras fuzzy e uma rede neural convencional.

Não são apresentados resultados numéricos fixos neste README, pois eles dependem da configuração escolhida e das condições de execução.

## 16. Limitações e considerações metodológicas

O projeto possui caráter acadêmico e experimental.

Algumas limitações devem ser consideradas na interpretação dos resultados.

**Implementação autoral do ANFIS:** o procedimento de aprendizagem é desenvolvido diretamente em NumPy e não corresponde integralmente ao treinamento híbrido clássico de ANFIS.

**Estabilidade numérica:** os parâmetros gaussianos são atualizados sem uma restrição explícita que garanta valores positivos e afastados de zero para os parâmetros de dispersão.

**Inicialização aleatória:** o ANFIS utiliza inicialização aleatória sem uma semente fixa na classe, o que pode produzir variações entre execuções.

**Gradientes:** as expressões utilizadas para atualizar os antecedentes precisam de validação numérica para assegurar correspondência com as derivadas da saída normalizada.

**Vazamento de informações:** os normalizadores são ajustados antes da separação entre treinamento e teste.

**Divisão temporal:** o método utilizado realiza divisão aleatória das amostras, não uma validação cronológica.

**Comparação de arquiteturas:** a quantidade de parâmetros e os métodos de treinamento não são necessariamente equivalentes entre ANFIS e MLP.

**Convergência:** as curvas apresentadas utilizam MAE, mas as regras de atualização dos modelos não são idênticas.

**Generalização:** os resultados são calculados sobre o recorte selecionado e não constituem evidência suficiente de desempenho em outros domínios ou períodos.

Essas limitações não impedem a utilização do projeto como estudo de sistemas neuro-fuzzy, mas devem ser consideradas em análises comparativas rigorosas.

## 17. Possíveis extensões

O projeto pode ser ampliado em diferentes direções:

- Implementação do algoritmo híbrido clássico de ANFIS;
- Validação dos gradientes por diferenças finitas;
- Parametrização positiva dos desvios das funções gaussianas;
- Inicialização reproduzível por sementes aleatórias;
- Divisão cronológica de treinamento, validação e teste;
- Ajuste dos normalizadores exclusivamente no conjunto de treinamento;
- Validação cruzada apropriada para séries temporais;
- Busca sistemática de hiperparâmetros;
- Comparação com modelos adicionais de regressão;
- Análise de interpretabilidade das regras fuzzy;
- Visualização das funções de pertinência antes e depois do treinamento;
- Análise estatística dos resultados em múltiplas execuções;
- Implementação com diferenciação automática em PyTorch;
- Exportação de resultados e modelos treinados.

Essas extensões permitiriam aprofundar a avaliação das arquiteturas neuro-fuzzy e melhorar a robustez experimental da comparação.

## 18. Referências

### Adaptive Neuro-Fuzzy Inference Systems

**Jang, J.-S. R. (1993).**

*ANFIS: Adaptive-Network-Based Fuzzy Inference System.*

IEEE Transactions on Systems, Man, and Cybernetics, 23(3), 665–685.

https://doi.org/10.1109/21.256541

### Lógica Fuzzy

**Zadeh, L. A. (1965).**

*Fuzzy Sets.*

Information and Control, 8(3), 338–353.

https://doi.org/10.1016/S0019-9958(65)90241-X

### Sistemas de Inferência Fuzzy

**Takagi, T., & Sugeno, M. (1985).**

*Fuzzy Identification of Systems and Its Applications to Modeling and Control.*

IEEE Transactions on Systems, Man, and Cybernetics, SMC-15(1), 116–132.

https://doi.org/10.1109/TSMC.1985.6313399

### Dataset

**Hebrail, G., & Berard, A. (2012).**

*Individual Household Electric Power Consumption.*

UCI Machine Learning Repository.

https://doi.org/10.24432/C58K54

### scikit-learn

**Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011).**

*Scikit-learn: Machine Learning in Python.*

Journal of Machine Learning Research, 12, 2825–2830.

https://jmlr.org/papers/v12/pedregosa11a.html

As referências apresentadas fornecem fundamentação teórica e metodológica complementar. Para a discussão acadêmica específica do trabalho, consulte a monografia disponibilizada neste repositório.

## 19. Contexto acadêmico

Este projeto foi desenvolvido para a disciplina **MS580 — Lógica Fuzzy**, explorando a integração entre sistemas de inferência fuzzy e aprendizado de máquina.

O trabalho reúne conceitos de:

- Lógica Fuzzy;
- Inteligência Computacional;
- Redes Neurais Artificiais;
- Sistemas Neuro-Fuzzy;
- Sistemas de Inferência Takagi-Sugeno;
- Aprendizado Supervisionado;
- Métodos Numéricos de Otimização;
- Regressão e Avaliação de Modelos;
- Computação Científica.

A implementação combina fundamentação matemática, desenvolvimento de algoritmos, processamento de dados reais e análise experimental.

O repositório possui finalidade acadêmica, educacional e demonstrativa.

---

**MS580 — Sistemas Neuro-Fuzzy: ANFIS vs. MLP**

*Lógica Fuzzy · Inteligência Computacional · Machine Learning · Python*
