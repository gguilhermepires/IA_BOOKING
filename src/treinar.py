import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from TC.TC0 import input_data as TC0_input, output_data as TC0_output
from TC.TC0B import input_data as TC0B_input, output_data as TC0B_output
from TC.TC1 import input_data as TC1_input, output_data as TC1_output
from TC.TC2 import input_data as TC2_input, output_data as TC2_output
from TC.TC3 import input_data as TC3_input, output_data as TC3_output
from TC.TC4 import input_data as TC4_input, output_data as TC4_output
from TC.TC5 import input_data as TC5_input, output_data as TC5_output
from TC.TC6 import input_data as TC6_input, output_data as TC6_output
from TC.TC7 import input_data as TC7_input, output_data as TC7_output
from TC.constants import ALL_ROOMS

# Parâmetros de padding (ajustados conforme solicitado)
MAX_BOOKINGS = 50       # Aumente conforme necessário
MAX_ROOMS = 40          # Aumente conforme necessário
MAX_PROFISSIONAIS = 30  # Aumente conforme necessário
MAX_SHOPS = 10          # Aumente conforme necessário

def transform_input(input_data):
    features = []
    
    # Adiciona informações básicas
    features.extend([
        input_data["idServicoTarget"],
        input_data["dataTarget"],
        input_data["duracao"]
    ])
    
    # Listabooking (agora com mais slots)
    print(f"Listabooking tamanho original: {len(input_data['listabooking'])}")
    for i in range(MAX_BOOKINGS):
        if i < len(input_data["listabooking"]):
            booking = input_data["listabooking"][i]
            features.extend([booking["servicoId"], booking["duration"], booking["data"]])
        else:
            features.extend([0, 0, 0])  # Padding
    print(f"Total de features após Listabooking: {len(features)}")
    
    # ListaRoom (agora com mais salas)
    print(f"ListaRoom tamanho original: {len(input_data['listaRoom'])}")
    for i in range(MAX_ROOMS):
        if i < len(input_data["listaRoom"]):
            room = input_data["listaRoom"][i]
            features.append(room["idRoom"])
            features.extend(room["listaServicoId"])
            features.extend([0] * (3 - len(room["listaServicoId"])))  # Padding
        else:
            features.extend([0] * 4)  # idRoom + 3 serviços
    print(f"Total de features após ListaRoom: {len(features)}")
    
    # listaProfisional (agora com mais profissionais)
    print(f"guilherme")
    print(input_data)
    print(f"listaProfisional tamanho original: {len(input_data['listaProfisional'])}")
    for i in range(MAX_PROFISSIONAIS):
        if i < len(input_data["listaProfisional"]):
            profissional = input_data["listaProfisional"][i]
            features.append(profissional["idProfissional"])
            features.extend(profissional["listaServico"])
            features.extend([0] * (3 - len(profissional["listaServico"])))  # Padding
            features.extend([
                profissional["inicioTrabalho"],
                profissional["fimTrabalho"]
            ])
        else:
            features.extend([0] * 6)  # id + 3 serviços + 2 horários
    print(f"Total de features após listaProfisional: {len(features)}")
    
    # ListaShop (agora com mais lojas)
    print(f"ListaShop tamanho original: {len(input_data['listaShop'])}")
    for i in range(MAX_SHOPS):
        if i < len(input_data["listaShop"]):
            shop = input_data["listaShop"][i]
            features.extend([
                shop["idShop"],
                shop["aberturaInicio"],
                shop["aberturaFechamento"]
            ])
        else:
            features.extend([0, 0, 0])  # Padding
    print(f"Total de features após ListaShop: {len(features)}")
    
    # Verificação de tamanho
    expected_size = (MAX_BOOKINGS * 3) + (MAX_ROOMS * 4) + (MAX_PROFISSIONAIS * 6) + (MAX_SHOPS * 3) + 3  # Base size (informações básicas)

    if len(features) != expected_size:
        print(f"Tamanho atual: {len(features)}, Esperado: {expected_size}")
        print("Ajustando o tamanho para o valor esperado.")
        features = features[:expected_size]  # Ajusta o tamanho se necessário

    return np.array(features, dtype=np.float32)


# Lista de input_datas e output_datas
input_datas = [
    TC1_input,
    TC2_input,
    TC3_input,
    TC4_input,
    TC5_input,
    TC6_input,
    TC7_input,
]
output_datas = [
    TC1_output,
    TC2_output,
    TC3_output,
    TC4_output,
    TC5_output,
    TC6_output,
    TC7_output,
]

# Transforma todos os input_datas em vetores de características
X_list = []
i =0
for data in input_datas:  # Use enumerate para obter índice e valor
    print(f'***Transformando o input {i}')
    transformed_data = transform_input(data)
    X_list.append(transformed_data)
    i =i+1

# Verifica o tamanho de cada vetor
for i, x in enumerate(X_list):
    print(f"Tamanho do vetor {i+1}: {len(x)}")

# Converte para array NumPy
X = np.array(X_list, dtype=np.float32)

# Prepara os rótulos (outputs)
y = np.array([output["match"] for output in output_datas], dtype=np.float32)

# Salvar parâmetros de normalização
mean = np.mean(X, axis=0)
std = np.std(X, axis=0) + 1e-8
np.save("mean.npy", mean)
np.save("std.npy", std)

# Normalização
X = (X - mean) / std

# Verificação da forma
print(f"Forma de X: {X.shape}, Forma de y: {y.shape}")

# Criação do modelo
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

model = SimpleNN(X.shape[1])
model.apply(lambda m: (isinstance(m, nn.Linear) and 
                      (nn.init.xavier_uniform_(m.weight), 
                       m.bias.data.fill_(0.01))))

# Treinamento
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

dataset = TensorDataset(torch.tensor(X), torch.tensor(y).unsqueeze(1))
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

for epoch in range(100):
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Salvar modelo
torch.save({
    "model_state": model.state_dict(),
    "input_size": X.shape[1]
}, "modelo_treinado.pth")
