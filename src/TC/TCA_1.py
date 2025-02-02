
from TC.constants import ALL_PROFISSIONALS, ALL_ROOMS, ALL_SHOPS
from util import data_para_timestamp

input_data = {
    "idServicoTarget": 7,
    "dataTarget": data_para_timestamp("25/10/2023 16:00"),
    "duracao": 94,
    "listabooking": [{'servicoId': 10, 'duration': 62, 'data': '25/10/2023 11:00'}, {'servicoId': 4, 'duration': 51, 'data': '25/10/2023 13:00'}],
    "listaRoom": ALL_ROOMS,
    "listaProfisional": ALL_PROFISSIONALS,
    "listaShop": ALL_SHOPS
}

output_data = {"match": 0}
        