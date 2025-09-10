import numpy as np

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
