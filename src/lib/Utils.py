import numpy as np

def pluck_masked_values(search_array, 
                        return_array, 
                        condition_func, 
                        axis: int = -1):
    condition_met = condition_func(search_array)
    first_match_indices = np.argmax(condition_met, axis=axis)
    plot_mask = ~np.any(condition_met, axis=axis)
    plucked_values = np.take_along_axis(
        return_array, np.expand_dims(first_match_indices, axis=axis), axis=axis
    ).squeeze(axis=axis)
    result = np.ma.masked_where(plot_mask, plucked_values)
    return result