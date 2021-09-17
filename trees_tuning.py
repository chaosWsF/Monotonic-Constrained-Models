import numpy as np
import pandas as pd
from data_loading import Data
from monotonic_constraints import get_monotonicity_string
from tree_trainer import Model

import sys
import time

args = sys.argv[1:]
model_name = args[0]
data_name = args[1]
n_iterations = int(args[2])

## Specify dataset and model
data_root = './Data/'

train_ratio = 0.7
seed = 3024

if data_name == 'housing':
    model_type = 'regression'
else:
    model_type = 'classification'

X_train, Y_train, X_val, Y_val, feature_names = Data(data_name, data_root).split(train_ratio)

## Set parameters grid
n_tree_low = 50
n_tree_high = 1000

max_depth_low = 3
max_depth_high = 15

tree_method = 'hist'
# tree_method = 'gpu_hist'

n_jobs = 4

monotonicity_string = get_monotonicity_string(feature_names, data_name)

min_child_weight_low = 1
min_child_weight_high = 10

gamma_10_low = 0
gamma_10_high = 90

eta_100_low = 1
eta_100_high = 30

# The last note on https://xgboost.readthedocs.io/en/latest/tutorials/monotonic.html
max_bin_low = 256
max_bin_high = 512

## tuning hyper-parameters
model = Model(model_name, model_type)
records = []
log_file = f'{data_name}_{model_name}.csv'

for i in range(n_iterations):
    t_0 = time.perf_counter()

    # Sample hyperparameters from uniform grid
    hyper_parameters = {
        'n_estimators': np.random.randint(n_tree_low, n_tree_high),
        'max_depth': np.random.randint(max_depth_low, max_depth_high),
        'tree_method': tree_method,
        'n_jobs': n_jobs,
        'monotone_constraints': monotonicity_string, 
        'min_child_weight': np.random.randint(min_child_weight_low, min_child_weight_high), 
        'gamma': np.random.randint(gamma_10_low, gamma_10_high) / 10, 
        'eta': np.random.randint(eta_100_low, eta_100_high) / 100, 
        'max_bin': np.random.randint(max_bin_low, max_bin_high), 
        'random_state': seed
    }
    
    model.record(hyper_parameters, X_train, Y_train, X_val, Y_val, records, log_file)

    dt = time.perf_counter() - t_0
    print(f'The iteration {i+1} used {dt:.2f} seconds')
