import pandas as pd
from ..db.repository import amostra
def get_classes(data, class_names) -> pd.DataFrame:
    class_list = []
    for i,class_name in enumerate(class_names):
        x = data[class_name].copy()
        x['Target'] = [i]*x.shape[0]
        class_list.append(x)
    
    return pd.concat(class_list,ignore_index=True)

def get_data(data, class_names) -> (pd.DataFrame, pd.DataFrame):
    data_base = get_classes(data, class_names)
    _data = data_base.drop(labels='Target',axis = 1)
    target = data_base['Target']
    #print(_data)
    return _data, target

def init_data(class_names) -> pd.DataFrame:
    data = {}
    for i in class_names:
        data[i] = amostra.get_amostras_by_classe_dt(i)
    return data

def used_names(names, data):
    data = data[names]
    return data

def change_name(names, data):
    data = pd.Series(data.values, index=names, dtype=float)
    return data
    
