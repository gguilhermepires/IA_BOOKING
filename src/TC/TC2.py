# Input e Output que serão exportados
from util import data_para_timestamp


input_data = {
    "idServicoTarget": 12,
    "dataTarget": data_para_timestamp("25/10/2023 14:00"),
    "duracao": 110,
    "listabooking": [
        {
            "servicoId": 12,
            "duration": 50,
            "data": data_para_timestamp("25/10/2023 14:00"),
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
}

output_data = {"match": 0}