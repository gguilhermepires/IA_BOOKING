import torch
import torch.nn as nn
import numpy as np
from util import data_para_timestamp

# Função para obter os valores de normalização a partir dos dados de treinamento
def get_normalization_values(X_train):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    return mean, std

# Função para transformar os dados de entrada em um vetor
def transform_input(input_data, max_bookings=2, max_rooms=2, max_profissionais=2, max_shops=1):
    features = []
    features.append(input_data["idServicoTarget"])
    features.append(input_data["dataTarget"])
    features.append(input_data["duracao"])
    
    for i in range(max_bookings):
        if i < len(input_data["listabooking"]):
            booking = input_data["listabooking"][i]
            features.append(booking["servicoId"])
            features.append(booking["duration"])
            features.append(booking["data"])
        else:
            features.extend([0, 0, 0])
    
    for i in range(max_rooms):
        if i < len(input_data["listaRoom"]):
            room = input_data["listaRoom"][i]
            features.append(room["idRoom"])
            features.extend(room["listaServicoId"])
            features.extend([0] * (3 - len(room["listaServicoId"])))
        else:
            features.extend([0] * 4)
    
    for i in range(max_profissionais):
        if i < len(input_data["listaProfisional"]):
            profissional = input_data["listaProfisional"][i]
            features.append(profissional["idProfissional"])
            features.extend(profissional["listaServico"])
            features.extend([0] * (3 - len(profissional["listaServico"])))
            features.append(profissional["inicioTrabalho"])
            features.append(profissional["fimTrabalho"])
        else:
            features.extend([0] * 6)
    
    for i in range(max_shops):
        if i < len(input_data["listaShop"]):
            shop = input_data["listaShop"][i]
            features.append(shop["idShop"])
            features.append(shop["aberturaInicio"])
            features.append(shop["aberturaFechamento"])
        else:
            features.extend([0, 0, 0])
    
    return np.array(features, dtype=np.float32)

# Caso de teste
caso_teste = {
    "idServicoTarget": 12,
    "dataTarget": data_para_timestamp("26/10/2023 14:00"),
    "duracao": 120,
    "listabooking": [
        {"servicoId": 13, "duration": 60, "data": data_para_timestamp("26/10/2023 10:00")}],
    "listaRoom": [
        {"idRoom": 1, "listaServicoId": [12, 13, 15]},
        {"idRoom": 2, "listaServicoId": [12, 13, 15]}],
    "listaProfisional": [
        {"idProfissional": 24, "listaServico": [12, 13, 15], "inicioTrabalho": data_para_timestamp("26/10/2023 08:00"), "fimTrabalho": data_para_timestamp("26/10/2023 22:00")},
        {"idProfissional": 25, "listaServico": [13, 15], "inicioTrabalho": data_para_timestamp("26/10/2023 08:00"), "fimTrabalho": data_para_timestamp("26/10/2023 22:00")}],
    "listaShop": [{"idShop": 12, "aberturaInicio": data_para_timestamp("26/10/2023 08:00"), "aberturaFechamento": data_para_timestamp("26/10/2023 22:00")}]
}

# Modelo de rede neural
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Carregar o modelo treinado
input_size = 32  # Certifique-se de que o tamanho da entrada esteja correto
model = SimpleNN(input_size)
model.load_state_dict(torch.load("modelo_treinado.pth"))
model.eval()

# Transformar o dado de entrada
X_teste = transform_input(caso_teste)

# Aqui você deve passar o conjunto de dados de treino para calcular a média e o desvio padrão
# Vou assumir que você já tem esses valores do treinamento. Caso contrário, você precisa de um conjunto de treinamento X_train.
X_train = np.random.randn(100, input_size)  # Exemplo de dados de treino (ajuste conforme necessário)
mean, std = get_normalization_values(X_train)

# Normalizar o dado de teste
X_teste = (X_teste - mean) / (std + 1e-8)
X_teste_tensor = torch.tensor(X_teste, dtype=torch.float32).unsqueeze(0)

# Fazer a previsão
with torch.no_grad():
    prediction = torch.sigmoid(model(X_teste_tensor))
    print(f"Previsão: {prediction.item()}")
