import numpy as np
import torch
import torch.nn as nn
from TC.TC0 import input_data as TC0_input
from TC.TC0B import input_data as TC0B_input
from TC.TC1 import input_data as TC1_input
from TC.TC2 import input_data as TC2_input
from TC.TC3 import input_data as TC3_input
from TC.TC4 import input_data as TC4_input
from TC.TC5 import input_data as TC5_input
from TC.TC6 import input_data as TC6_input
from treinar import MAX_BOOKINGS, MAX_PROFISSIONAIS, MAX_ROOMS, MAX_SHOPS, transform_input

# Carregar os parâmetros de normalização
mean = np.load("mean.npy")
std = np.load("std.npy")

# Carregar o modelo treinado
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

# Carregar o modelo e o estado salvo
model = SimpleNN(input_size=mean.shape[0])
model.load_state_dict(torch.load("modelo_treinado.pth")["model_state"])
model.eval()  # Coloca o modelo em modo de avaliação

# Carregar um exemplo de input_data
input_data = TC1_input  # Você pode escolher qualquer input_data aqui

# Transformar o input
X_input = transform_input(input_data)

# Normalizar o input
X_input = (X_input - mean) / std

# Transformar para tensor
X_tensor = torch.tensor(X_input, dtype=torch.float32).unsqueeze(0)  # Adiciona uma dimensão para batch size

# Fazer a previsão
with torch.no_grad():  # Desativa o cálculo do gradiente
    prediction = model(X_tensor)

# A previsão será um valor contínuo, converta para um valor binário
prediction = torch.sigmoid(prediction).item()  # A função sigmoide retorna valores entre 0 e 1

# Mostrar o resultado
print(f"Previsão: {prediction:.4f}")
