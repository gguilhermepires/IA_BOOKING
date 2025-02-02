import random

def data_para_timestamp(data_str):
    # Simulando a função de conversão de string para timestamp
    return data_str

ALL_SERVICES = list(range(1, 11))
ALL_ROOMS = list(range(11, 17))
ALL_PROFESSIONALS = list(range(21, 27))
SHOP_ID = 32

def generate_case(idServicoTarget, dataTarget, duracao, bookings, rooms, professionals, match):
    return {
        "input_data": {
            "idServicoTarget": idServicoTarget,
            "dataTarget": data_para_timestamp(dataTarget),
            "duracao": duracao,
            "listabooking": bookings,
            "listaRoom": rooms,
            "listaProfisional": professionals,
            "listaShop": [
                {
                    "idShop": SHOP_ID,
                    "aberturaInicio": data_para_timestamp("25/10/2023 08:00"),
                    "aberturaFechamento": data_para_timestamp("25/10/2023 22:00"),
                }
            ],
        },
        "output_data": {"match": match},
    }

cases = []

for i in range(30):
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
    
    rooms = [
        {
            "idRoom": room_id,
            "listaServicoId": random.sample(ALL_SERVICES, random.randint(1, len(ALL_SERVICES)))
        }
        for room_id in ALL_ROOMS
    ]
    
    professionals = [
        {
            "idProfissional": prof_id,
            "listaServico": random.sample(ALL_SERVICES, random.randint(1, len(ALL_SERVICES))),
            "inicioTrabalho": data_para_timestamp("25/10/2023 08:00"),
            "fimTrabalho": data_para_timestamp("25/10/2023 22:00"),
        }
        for prof_id in ALL_PROFESSIONALS
    ]
    
    match = random.choice([0, 1])
    cases.append(generate_case(idServicoTarget, dataTarget, duracao, bookings, rooms, professionals, match))

# Exibir um exemplo de caso
test_cases = {"cases": cases}
print(test_cases)
