import numpy as np

class ANFIS:
    def __init__(self, input_data, output_data, n_rules, n_inputs):
        self.input_data = input_data
        self.output_data = output_data
        self.n_rules = n_rules  # Número de regras fuzzy
        self.n_inputs = n_inputs  # Número de variáveis de entrada
        self.mf_params = np.random.rand(self.n_inputs, self.n_rules, 2)  # [c, sigma]  # Parâmetros das funções de pertinência Gaussianas: (centro c e desvio sigma)
        self.rule_params = np.random.rand(self.n_rules, self.n_inputs + 1) # Parâmetros das funções lineares (parte consequente): [a1, ..., an, b]


    def gaussian_mf(self, x, c, sigma): # Define a função de pertinência Gaussiana
        return np.exp(-((x - c) ** 2) / (2 * sigma ** 2 + 1e-6))

    def rule_evaluation(self, x):
        firing_strengths = []
        for i in range(self.n_rules): # Para cada regra
            strength = 1
            for j in range(self.n_inputs): # Para cada entrada
                c, sigma = self.mf_params[j, i]
                strength *= self.gaussian_mf(x[j], c, sigma) # Produto das pertinências
            firing_strengths.append(strength)
        return np.array(firing_strengths)

    def normalize(self, w):
        total = np.sum(w)
        return w / total if total != 0 else w # Normaliza para que a soma das forças seja 1

    def defuzzify(self, x, normalized_w): # Realiza a defuzzificação
        output = 0
        for i in range(self.n_rules):
            a = self.rule_params[i, :-1] # Coeficientes das entradas
            b = self.rule_params[i, -1] # Termo constante
            z = np.dot(a, x) + b # Saída linear da regra i
            output += normalized_w[i] * z # Combinação ponderada das saídas das regras
        return output

    def forward_pass(self, x): #Avalia uma entrada x, normaliza as forças de disparo e calcula a saída do sistema ANFIS
        w = self.rule_evaluation(x)
        normalized_w = self.normalize(w)
        return self.defuzzify(x, normalized_w)

    def train(self, epochs=100, learning_rate=0.01):
        self.loss_history = [] # Inicializa histórico de erros
        for epoch in range(epochs): # Loop de épocas
            total_error = 0
            for idx in range(self.input_data.shape[0]): # Loop sobre o dataset
                x = self.input_data[idx]
                y_true = self.output_data[idx]

                w = self.rule_evaluation(x)
                norm_w = self.normalize(w)
                y_pred = self.defuzzify(x, norm_w)
                error = y_true - y_pred
                total_error += abs(error)

                for i in range(self.n_rules): # Atualiza os coeficientes lineares a e o viés b das regras com base no erro e na força de disparo normalizada
                    a = self.rule_params[i, :-1]
                    b = self.rule_params[i, -1]
                    z = np.dot(a, x) + b

                    for j in range(self.n_inputs):
                        self.rule_params[i, j] += learning_rate * error * norm_w[i] * x[j]
                    self.rule_params[i, -1] += learning_rate * error * norm_w[i]
 
                for i in range(self.n_rules): # Atualiza os parâmetros das funções de pertinência
                    for j in range(self.n_inputs):
                        c, sigma = self.mf_params[j, i]
                        xj = x[j]

                        gauss = self.gaussian_mf(xj, c, sigma)
                        d_gauss_dc = gauss * ((xj - c) / (sigma ** 2 + 1e-6))
                        d_gauss_dsigma = gauss * (((xj - c) ** 2) / ((sigma ** 3 + 1e-6)))

                        other_mfs_product = 1 # Calcula o produto das outras funções de pertinência da regra (exceto a j-ésima), necessário para o cálculo da derivada da força de disparo total
                        for k in range(self.n_inputs):
                            if k != j:
                                ck, sigmak = self.mf_params[k, i]
                                other_mfs_product *= self.gaussian_mf(x[k], ck, sigmak)

                        dw_dc = d_gauss_dc * other_mfs_product # Calcula as derivadas do output final em relação aos parâmetros c e sigma da função de pertinência da entrada j da regra i
                        dw_dsigma = d_gauss_dsigma * other_mfs_product

                        sum_w = np.sum(w) + 1e-6
                        z_i = np.dot(self.rule_params[i, :-1], x) + self.rule_params[i, -1]
                        d_output_dc = ((sum_w * z_i - w[i] * z_i) / (sum_w ** 2)) * dw_dc
                        d_output_dsigma = ((sum_w * z_i - w[i] * z_i) / (sum_w ** 2)) * dw_dsigma

                        self.mf_params[j, i, 0] += learning_rate * error * d_output_dc # Atualiza os parâmetros c e sigma com gradiente descendente
                        self.mf_params[j, i, 1] += learning_rate * error * d_output_dsigma

            self.loss_history.append(total_error / len(self.input_data)) # Guarda o erro médio da época para visualização posterior 

    def predict(self, inputs): # Executa o forward pass para todas as entradas fornecidas, retornando as predições
        return np.array([self.forward_pass(x) for x in inputs])
