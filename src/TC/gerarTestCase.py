import random

def data_para_timestamp(data_str):
    # Simulando a função de conversão de string para timestamp
    return data_str

ALL_SERVICES = list(range(1, 11))
ALL_ROOMS = list(range(11, 17))
ALL_PROFESSIONALS = list(range(21, 27))
SHOP_ID = 32

def generate_case(idServicoTarget, dataTarget, duracao, bookings, rooms, professionals, match, case_number):
    filename = f"TCA_{case_number}.py"
    with open(filename, "w") as f:
        f.write(f"""
from TC.constants import ALL_PROFISSIONALS, ALL_ROOMS, ALL_SHOPS
from util import data_para_timestamp

input_data = {{
    "idServicoTarget": {idServicoTarget},
    "dataTarget": data_para_timestamp("{dataTarget}"),
    "duracao": {duracao},
    "listabooking": {bookings},
    "listaRoom": ALL_ROOMS,
    "listaProfisional": ALL_PROFISSIONALS,
    "listaShop": ALL_SHOPS
}}

output_data = {{"match": {match}}}
        """)

cases = []

for i in range(1):
    idServicoTarget = random.choice(ALL_SERVICES)
    dataTarget = f"25/10/2023 {random.randint(8, 21)}:00"
    duracao = random.randint(30, 180)
    
    bookings = [
        {
            "servicoId": random.choice(ALL_SERVICES),
            "duration": random.randint(30, 120),
            "data": data_para_timestamp(f"25/10/2023 {random.randint(8, 21)}:00"),
        }
        for _ in range(random.randint(0, 3))
    ]
    
    rooms = ALL_ROOMS
    professionals = ALL_PROFESSIONALS
    
    match = random.choice([0, 1])
    generate_case(idServicoTarget, dataTarget, duracao, bookings, rooms, professionals, match, i+1)

print("Test cases generated successfully.")
