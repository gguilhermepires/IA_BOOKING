from datetime import datetime
from TC.constants import ALL_PROFISSIONALS, ALL_SERVICES, ALL_SHOPS
from util import data_para_timestamp

#TC3: Múltiplos profissionais disponíveis para o serviço no horário solicitado.

input_data = {
    "idServicoTarget": 12,
    "dataTarget": data_para_timestamp("25/10/2023 14:00"),
    "duracao": 110,
    "listabooking": [],
    "listaRoom": [
        {
            "idRoom": 1,
            "listaServicoId": [12, 13, 15]
        }
    ],
    "listaProfisional": [
        {
            "idProfissional": 24,
            "listaServico": ALL_SERVICES,
            "inicioTrabalho": data_para_timestamp("25/10/2023 08:00"),
            "fimTrabalho": data_para_timestamp("25/10/2023 22:00"),
        },
        {
            "idProfissional": 25,
            "listaServico": [12, 13, 15],
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
}
output_data = {"match": 1}
