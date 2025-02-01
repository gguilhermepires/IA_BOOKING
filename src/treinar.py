import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from TC.TC1 import input_data as TC1_input, output_data as TC1_output

def transform_input(input_data, max_bookings=2, max_rooms=2, max_profissionais=2, max_shops=1):
    # Vetor de características (features)
    features = []
    
    # Adiciona idServicoTarget
    features.append(input_data["idServicoTarget"])
    
    # Adiciona dataTarget
    features.append(input_data["dataTarget"])
    
    # Adiciona duracao
    features.append(input_data["duracao"])
    
    # Adiciona informações da listabooking (pad with zeros if necessary)
    for i in range(max_bookings):
        if i < len(input_data["listabooking"]):
            booking = input_data["listabooking"][i]
            features.append(booking["servicoId"])
            features.append(booking["duration"])
            features.append(booking["data"])
        else:
            features.extend([0, 0, 0])  # Pad with zeros
    
    # Adiciona informações da listaRoom (pad with zeros if necessary)
    for i in range(max_rooms):
        if i < len(input_data["listaRoom"]):
            room = input_data["listaRoom"][i]
            features.append(room["idRoom"])
            features.extend(room["listaServicoId"])
            features.extend([0] * (3 - len(room["listaServicoId"])))  # Pad with zeros if listaServicoId has fewer than 3 elements
        else:
            features.extend([0] * 4)  # Pad with zeros (1 for idRoom + 3 for listaServicoId)
    
    # Adiciona informações da listaProfisional (pad with zeros if necessary)
    for i in range(max_profissionais):
        if i < len(input_data["listaProfisional"]):
            profissional = input_data["listaProfisional"][i]
            features.append(profissional["idProfissional"])
            features.extend(profissional["listaServico"])
            features.extend([0] * (3 - len(profissional["listaServico"])))  # Pad with zeros if listaServico has fewer than 3 elements
            features.append(profissional["inicioTrabalho"])
            features.append(profissional["fimTrabalho"])
        else:
            features.extend([0] * 6)  # Pad with zeros (1 for idProfissional + 3 for listaServico + 2 for timestamps)
    
    # Adiciona informações da listaShop (pad with zeros if necessary)
    for i in range(max_shops):
        if i < len(input_data["listaShop"]):
            shop = input_data["listaShop"][i]
            features.append(shop["idShop"])
            features.append(shop["aberturaInicio"])
            features.append(shop["aberturaFechamento"])
        else:
            features.extend([0, 0, 0])  # Pad with zeros
    
    return np.array(features, dtype=np.float32)

# Lista de input_datas e output_datas
input_datas = [
    TC1_input,
]

output_datas = [
    TC1_output,
]

# Transforma todos os input_datas em vetores de características
X_list = [transform_input(input_data) for input_data in input_datas]
X = np.array(X_list, dtype=np.float32)

# Prepara os rótulos (outputs)
y = np.array([output["match"] for output in output_datas], dtype=np.float32)

# Normaliza os dados com epsilon para evitar divisão por zero
X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)

# Verifica as formas dos dados
print(f"Forma de X: {X.shape}")  # Deve ser (n_amostras, n_features)
print(f"Forma de y: {y.shape}")  # Deve ser (n_amostras,)

# Converte os dados para tensores do PyTorch
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # Adiciona dimensão de lote (batch)

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

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # Sem sigmoid aqui
        return x

# Instancia o modelo
input_size = X.shape[1]  # Número de características
model = SimpleNN(input_size)

# Inicializa os pesos
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

model.apply(init_weights)

# Define a função de perda e o otimizador
criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy with Logits Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Treina o modelo
num_epochs = 10
for epoch in range(num_epochs):
    for inputs, labels in dataloader:
        # Forward pass
        outputs = model(inputs)
        
        # Compute loss
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
        X_teste = (X_teste - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)  # Normaliza
        X_teste_tensor = torch.tensor(X_teste, dtype=torch.float32).unsqueeze(0)
        prediction = torch.sigmoid(model(X_teste_tensor))  # Aplica sigmoid para obter probabilidades
        print(f"Previsão para input_data {i+1}: {prediction.item()}")

# Salva o modelo treinado
torch.save(model.state_dict(), "modelo_treinado.pth")