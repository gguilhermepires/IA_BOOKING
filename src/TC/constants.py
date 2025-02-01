from util import data_para_timestamp

ALL_SERVICES= [1,2,3,4,5,6,7,8,9,10]

ALL_ROOMS=[
        {
            "idRoom": 11,
            "listaServicoId": ALL_SERVICES
        },
        {
            "idRoom": 12,
            "listaServicoId": [2,3]
        },
        {
            "idRoom": 13,
            "listaServicoId": [3,4]
        },
        {
            "idRoom": 14,
            "listaServicoId": ALL_SERVICES
        },
        {
            "idRoom": 15,
            "listaServicoId": [5,6,7,8]
        },
        {
            "idRoom": 16,
            "listaServicoId": [1]
        } 
    ]

ALL_PROFISSIONALS=[
     {
            "idProfissional": 21,
            "listaServico": ALL_SERVICES,
            "inicioTrabalho": data_para_timestamp("25/10/2023 08:00"),
            "fimTrabalho": data_para_timestamp("25/10/2023 22:00"),
        },
        {
            "idProfissional": 22,
            "listaServico": [1,2,3],
            "inicioTrabalho": data_para_timestamp("25/10/2023 08:00"),
            "fimTrabalho": data_para_timestamp("25/10/2023 22:00"),
        },
          {
            "idProfissional": 23,
            "listaServico": ALL_SERVICES,
            "inicioTrabalho": data_para_timestamp("25/10/2023 08:00"),
            "fimTrabalho": data_para_timestamp("25/10/2023 22:00"),
        },
          {
            "idProfissional": 24,
            "listaServico": [5,6,7,8],
            "inicioTrabalho": data_para_timestamp("25/10/2023 08:00"),
            "fimTrabalho": data_para_timestamp("25/10/2023 22:00"),
        },
          {
            "idProfissional": 25,
            "listaServico": ALL_SERVICES,
            "inicioTrabalho": data_para_timestamp("25/10/2023 08:00"),
            "fimTrabalho": data_para_timestamp("25/10/2023 22:00"),
        },
          {
            "idProfissional": 26,
            "listaServico": [1,5,8,9],
            "inicioTrabalho": data_para_timestamp("25/10/2023 08:00"),
            "fimTrabalho": data_para_timestamp("25/10/2023 22:00"),
        }
]

ALL_SHOPS=[
        {
            "idShop": 12,
            "aberturaInicio": data_para_timestamp("25/10/2023 08:00"),
            "aberturaFechamento": data_para_timestamp("25/10/2023 22:00"),
        }
    ]