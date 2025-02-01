from datetime import datetime
def data_para_timestamp(data_str):
    """
    Converte uma string de data no formato 'DD/MM/YYYY HH:MM' para timestamp.
    
    :param data_str: String da data no formato 'DD/MM/YYYY HH:MM'.
    :return: Timestamp correspondente.
    """
    formato = "%d/%m/%Y %H:%M"
    data_obj = datetime.strptime(data_str, formato)
    return data_obj.timestamp()