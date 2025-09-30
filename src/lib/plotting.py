import matplotlib.pyplot as plt
import numpy as np

import os

from lib.data_manager import load_data
from lib.utils import convert_data

# data = {name : convert_data(load_data(base_path+name[:-5])) # -5 since names already have .json
#             for name in directory 
#                 if (search_string in name and 'Adam' not in name
#                     and 'Stateful' not in name)}

# desired_order = ['_Hinge', '_CrossEntropy', '_VeryLazyCrossEntropy', '_MSE', '_VeryLazyMSE']

# target_order = []
# i = 0
# for _ in data.keys():
#     for k in data.keys():
#         if desired_order[i] in k:
#             target_order.append(k)
#             i += 1
#             break

# data = {k: data[k] for k in target_order}

def get_palettes(colors, n_colors, names=None):
    if names is None:
        return [plt.get_cmap(c, n_colors) for n, c in enumerate(colors)]
    else:
        return {names[n] : plt.get_cmap(c, n_colors) for n, c in enumerate(colors)}
    
    
def get_and_filter_data(include=None, exclude=None, path='./', extension='.json'):
    directory = os.listdir(path)  
    return {name : convert_data(load_data(path+name, extension=''))
                for name in directory
                    if name.endswith(extension)
                    if not include or     all(yes in name for yes in include)
                    if not exclude or not any(nope in name for nope in exclude)}

def order_data(data, desired_name_order):
    target_order = []
    i = 0
    for _ in data.keys():
        for k in data.keys():
            if desired_name_order[i] in k:
                target_order.append(k)
                i += 1
                break
        if i == len(desired_name_order):
            break
    return {k: data[k] for k in target_order}
    
    
# base_path = './experiments/'
# d = get_and_filter_data(['MAIN'], ['Adam', 'Stateful'])