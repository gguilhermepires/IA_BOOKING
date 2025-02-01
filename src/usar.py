import numpy as np
import torch
from datetime import datetime

# Função para converter "DD/MM/YYYY HH:MM" em timestamp
def data_para_timestamp(data_str):
    """
    Converte uma string de data no formato 'DD/MM/YYYY HH:MM' para timestamp.
    
    :param data_str: String da data no formato 'DD/MM/YYYY HH:MM'.
    :return: Timestamp correspondente.
    """
    formato = "%d/%m/%Y %H:%M"
    data_obj = datetime.strptime(data_str, formato)
    return data_obj.timestamp()

# Função para transformar os dados de entrada em um vetor
def transform_input(input_data):
    # Vetor de características (features)
    features = []
    
    # Adiciona idServicoTarget
    features.append(input_data["idServicoTarget"])
    
    # Adiciona dataTarget
    features.append(input_data["dataTarget"])
    
    # Adiciona duracao
    features.append(input_data["duracao"])
    
    # Adiciona informações da listabooking
    for booking in input_data["listabooking"]:
        features.append(booking["servicoId"])
        features.append(booking["duration"])
        features.append(booking["data"])
    
    # Adiciona informações da listaRoom
    for room in input_data["listaRoom"]:
        features.append(room["idRoom"])
        features.extend(room["listaServicoId"])
    
    # Adiciona informações da listaProfisional
    for profissional in input_data["listaProfisional"]:
        features.append(profissional["idProfissional"])
        features.extend(profissional["listaServico"])
        features.append(profissional["inicioTrabalho"])
        features.append(profissional["fimTrabalho"])
    
    # Adiciona informações da listaShop
    for shop in input_data["listaShop"]:
        features.append(shop["idShop"])
        features.append(shop["aberturaInicio"])
        features.append(shop["aberturaFechamento"])
    
    return np.array(features, dtype=np.float32)

# Define a rede neural (a mesma arquitetura usada no treinamento)
class SimpleNN(torch.nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 64)  # Camada de entrada
        self.fc2 = torch.nn.Linear(64, 32)          # Camada oculta
        self.fc3 = torch.nn.Linear(32, 1)           # Camada de saída
        self.relu = torch.nn.ReLU()                 # Função de ativação
        self.sigmoid = torch.nn.Sigmoid()           # Função de ativação para saída

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Carrega o modelo treinado
input_size = 28  # Substitua pelo número de características usadas no treinamento
model = SimpleNN(input_size)
model.load_state_dict(torch.load("modelo_treinado.pth"))
model.eval()  # Coloca o modelo em modo de avaliação

# Exemplo de caso de teste
caso_teste = {
    "idServicoTarget": 12,
    "dataTarget": data_para_timestamp("26/10/2023 14:00"),  # Nova data
    "duracao": 120,  # Nova duração
    "listabooking": [
        {
            "servicoId": 13,
            "duration": 60,
            "data": data_para_timestamp("26/10/2023 10:00"),
        }
    ],
    "listaRoom": [
        {
            "idRoom": 1,
            "listaServicoId": [12, 13, 15]
        },
        {
            "idRoom": 2,
            "listaServicoId": [12, 13, 15]
        }
    ],
    "listaProfisional": [
        {
            "idProfissional": 24,
            "listaServico": [12, 13, 15],
            "inicioTrabalho": data_para_timestamp("26/10/2023 08:00"),
            "fimTrabalho": data_para_timestamp("26/10/2023 22:00"),
        },
        {
            "idProfissional": 25,
            "listaServico": [13, 15],
            "inicioTrabalho": data_para_timestamp("26/10/2023 08:00"),
            "fimTrabalho": data_para_timestamp("26/10/2023 22:00"),
        }
    ],
    "listaShop": [
        {
            "idShop": 12,
            "aberturaInicio": data_para_timestamp("26/10/2023 08:00"),
            "aberturaFechamento": data_para_timestamp("26/10/2023 22:00"),
        }
    ]
}

# Passo 1: Transforma o caso de teste em um vetor de características
X_teste = transform_input(caso_teste)

# Passo 2: Normaliza o caso de teste (usando a mesma normalização dos dados de treinamento)
# Aqui, você precisa usar a média e o desvio padrão dos dados de treinamento
# Exemplo: X_teste = (X_teste - mean_train) / std_train
# Substitua mean_train e std_train pelos valores usados no treinamento
mean_train = np.mean(X_teste, axis=0)  # Exemplo (substitua pelos valores reais)
std_train = np.std(X_teste, axis=0)    # Exemplo (substitua pelos valores reais)
X_teste = (X_teste - mean_train) / std_train

# Passo 3: Converte o vetor de características em um tensor do PyTorch
X_teste_tensor = torch.tensor(X_teste, dtype=torch.float32).unsqueeze(0)  # Adiciona dimensão de lote

# Passo 4: Faz a previsão
with torch.no_grad():  # Desabilita o cálculo de gradientes (não é necessário para inferência)
    prediction_teste = model(X_teste_tensor)
    print(f"Previsão para o caso de teste: {prediction_teste.item()}")

# Passo 5 (Opcional): Converte a previsão em uma classificação binária
threshold = 0.5
binary_prediction_teste = 1 if prediction_teste.item() > threshold else 0
print(f"Classificação binária: {binary_prediction_teste}")