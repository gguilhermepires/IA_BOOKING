from TC.constants import ALL_PROFISSIONALS, ALL_ROOMS, ALL_SHOPS
from util import data_para_timestamp

input_data = {
    "idServicoTarget": 12,
    "dataTarget": data_para_timestamp("25/10/2023 14:00"),
    "duracao": 110,
    "listabooking": [
        {
            "servicoId": 1,
            "duration": 50,
            "data": data_para_timestamp("25/10/2023 14:00"),
        }
    ],
    "listaRoom": ALL_ROOMS,
    "listaProfisional": ALL_PROFISSIONALS,
    "listaShop": ALL_SHOPS
}

output_data = {"match": 0}