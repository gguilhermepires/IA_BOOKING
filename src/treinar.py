import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from TC.TC1 import input_data as TC1_input, output_data as TC1_output

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

# Lista de input_datas e output_datas
input_datas = [
    TC1_input,
    {
        "idServicoTarget": 12,
        "dataTarget": data_para_timestamp("25/10/2023 14:00"),
        "duracao": 110,
        "listabooking": [
            {
                "servicoId": 13,
                "duration": 50,
                "data": data_para_timestamp("25/10/2023 09:00"),
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
                "inicioTrabalho": data_para_timestamp("25/10/2023 08:00"),
                "fimTrabalho": data_para_timestamp("25/10/2023 22:00"),
            },
            {
                "idProfissional": 25,
                "listaServico": [13, 15],
                "inicioTrabalho": data_para_timestamp("25/10/2023 08:00"),
                "fimTrabalho": data_para_timestamp("25/10/2023 22:00"),
            }
        ],
        "listaShop": [
            {
                "idShop": 12,
                "aberturaInicio": data_para_timestamp("25/10/2023 08:00"),
                "aberturaFechamento": data_para_timestamp("25/10/2023 22:00"),
            }
        ]
    },
    {
        "idServicoTarget": 13,
        "dataTarget": data_para_timestamp("26/10/2023 15:00"),
        "duracao": 90,
        "listabooking": [
            {
                "servicoId": 14,
                "duration": 60,
                "data": data_para_timestamp("26/10/2023 10:00"),
            }
        ],
        "listaRoom": [
            {
                "idRoom": 1,
                "listaServicoId": [13, 14, 16]
            }
        ],
        "listaProfisional": [
            {
                "idProfissional": 26,
                "listaServico": [13, 14],
                "inicioTrabalho": data_para_timestamp("26/10/2023 08:00"),
                "fimTrabalho": data_para_timestamp("26/10/2023 22:00"),
            }
        ],
        "listaShop": [
            {
                "idShop": 13,
                "aberturaInicio": data_para_timestamp("26/10/2023 08:00"),
                "aberturaFechamento": data_para_timestamp("26/10/2023 22:00"),
            }
        ]
    },
    # Adicione mais exemplos aqui...
]

output_datas = [
    TC1_output,
    {"match": 1},  # Saída correspondente ao primeiro input_data
    {"match": 0},  # Saída correspondente ao segundo input_data
    # Adicione mais saídas aqui...
]

# Transforma todos os input_datas em vetores de características
X_list = [transform_input(input_data) for input_data in input_datas]
X = np.array(X_list, dtype=np.float32)

# Prepara os rótulos (outputs)
y = np.array([output["match"] for output in output_datas], dtype=np.float32)

# Normaliza os dados (opcional, mas recomendado)
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Verifica as formas dos dados
print(f"Forma de X: {X.shape}")  # Deve ser (n_amostras, n_features)
print(f"Forma de y: {y.shape}")  # Deve ser (n_amostras,)

# Converte os dados para tensores do PyTorch
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # Adiciona dimensão de lote (batch)

# Verifica as formas dos tensores
print(f"Forma de X_tensor: {X_tensor.shape}")  # Deve ser (n_amostras, n_features)
print(f"Forma de y_tensor: {y_tensor.shape}")  # Deve ser (n_amostras, 1)

# Cria um DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)  # Ajuste o batch_size conforme necessário

# Define a rede neural
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)  # Camada de entrada
        self.fc2 = nn.Linear(64, 32)          # Camada oculta
        self.fc3 = nn.Linear(32, 1)           # Camada de saída
        self.relu = nn.ReLU()                 # Função de ativação
        self.sigmoid = nn.Sigmoid()           # Função de ativação para saída

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Instancia o modelo
input_size = X.shape[1]  # Número de características
model = SimpleNN(input_size)

# Define a função de perda e o otimizador
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Treina o modelo
num_epochs = 10
for epoch in range(num_epochs):
    for inputs, labels in dataloader:
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass e otimização
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Faz uma previsão com o modelo treinado
with torch.no_grad():
    for i, input_data in enumerate(input_datas):
        X_teste = transform_input(input_data)
        X_teste = (X_teste - np.mean(X, axis=0)) / np.std(X, axis=0)  # Normaliza
        X_teste_tensor = torch.tensor(X_teste, dtype=torch.float32).unsqueeze(0)
        prediction = model(X_teste_tensor)
        print(f"Previsão para input_data {i+1}: {prediction.item()}")

# Salva o modelo treinado
torch.save(model.state_dict(), "modelo_treinado.pth")