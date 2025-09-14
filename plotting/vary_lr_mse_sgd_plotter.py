from lib import data_manager as dm
from lib.utils import pluck_masked_values, print_state_data_structure, moving_mean

import numpy as np
import matplotlib.pyplot as plt

file_name = 'VARY_LR_MSELoss_SGD_2025-09-14'

data = dm.load_data(file_name)

acc = moving_mean(np.array(data['tracked_data']['test_accuracies']['test']), 3)
en = np.array(data['tracked_data']['backward_pass_counts'])
lr = data['experimental_setup']['lr']

filtered = pluck_masked_values(acc, en, lambda x : x > 9700)

plt.plot(lr, filtered/60_000)
plt.yscale('log')
plt.xscale('log')