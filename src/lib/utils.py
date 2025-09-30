import matplotlib.pyplot as plt
import numpy as np
import json
import os

# Gemini-2.5 Pro
def pluck_masked_values(search_array, # filter items against
                        return_array, # pull items from
                        condition_func,  # filter to apply 
                        axis: int = 0):
    condition_met = condition_func(search_array)
    first_match_indices = np.argmax(condition_met, axis=axis)
    plot_mask = ~np.any(condition_met, axis=axis)
    plucked_values = np.take_along_axis(
        return_array, np.expand_dims(first_match_indices, axis=axis), axis=axis
    ).squeeze(axis=axis)
    result = np.ma.masked_where(plot_mask, plucked_values)
    return result

# Gemini-Flash 2.5
def print_state_data_structure(state, level=0):
    """
    Recursively prints the keys and types of values at the leaf nodes 
    of a nested dictionary using indentation to show the path.
    
    Args:
        d: The dictionary to traverse.
        level: The current indentation level (depth).
    """
    indent = '    ' * level  # Four spaces per level for tabbing
    
    for key, value in state.items():
        if isinstance(value, dict):
            # If the value is a dictionary, print the key with current indentation
            # and then recurse with an increased level.
            print(f"{indent}{key}:")
            print_state_data_structure(value, level + 1)
        else:
            # If it's a leaf node, print the key and type with the current indentation.
            print(f"{indent}{key}: {type(value).__name__}")

# Gemini-Flash 2.5
def moving_mean(data: np.ndarray, window_size, axis = -1):
    """
    Calculates the moving mean of a NumPy array along a specified axis using convolution.

    The moving mean is calculated by convolving the data with a window of ones,
    and then dividing by the window size. The 'same' mode of convolution is used,
    which means the output will have the same length as the input along the
    specified axis. This method adds padding to both ends of the input so the
    convolution result is centered and maintains the original size.

    Args:
        data (np.ndarray): A NumPy array of numerical data.
        window_size (int): The size of the moving window. Must be a positive integer.
        axis (int, optional): The axis along which to compute the moving mean.
                              Defaults to the last axis (-1).

    Returns:
        np.ndarray: A new NumPy array containing the moving mean values, padded to
                    the original dimension.

    Raises:
        ValueError: If window_size is not a positive integer, is larger than the
                    data length along the specified axis, or if the axis is out of bounds.
    """
    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError("window_size must be a positive integer.")
        
    if axis >= data.ndim or axis < -data.ndim:
        raise ValueError(f"Axis {axis} is out of bounds for an array with {data.ndim} dimensions.")

    data_length = data.shape[axis]
    if window_size > data_length:
        raise ValueError(f"window_size ({window_size}) cannot be larger than the data length ({data_length}) along axis {axis}.")

    # Define the convolution kernel (a window of ones)
    weights = np.ones(window_size)
    
    # Define a helper function to apply the 1D convolution with 'same' mode
    def _convolve_1d(arr):
        return np.convolve(arr, weights, 'same') / window_size

    # Apply the 1D convolution function along the specified axis
    return np.apply_along_axis(_convolve_1d, axis, data)

def convert_data(data_dict):
    """
    Recursively converts lists within a nested dictionary to NumPy arrays.

    Args:
        data_dict: The dictionary to traverse and convert.

    Returns:
        The modified dictionary with lists converted to NumPy arrays.
    """
    if isinstance(data_dict, dict):
        for key, value in data_dict.items():
            # If the value is a dictionary, recurse on it.
            if isinstance(value, dict):
                data_dict[key] = convert_data(value)
            # If the value is a list, convert it to a NumPy array.
            elif isinstance(value, list):
                data_dict[key] = np.array(value)
    return data_dict

def save_data(file_name, data):
    with open(file_name + ".json", "w") as f:
        json.dump(data, f, indent=4, sort_keys=True)

# loads data into default python types
def load_data(file_name, extension='.json'):
    with open(file_name + extension, "r") as f:
        data = json.load(f)
    return data

def get_palettes(colors, n_colors, names=None):
    if names is None:
        return [plt.get_cmap(c, n_colors) for n, c in enumerate(colors)]
    else:
        return {names[n] : plt.get_cmap(c, n_colors) for n, c in enumerate(colors)}
    
    
def get_and_filter_data(include, exclude, path='./', extension='.json'):
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