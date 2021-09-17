import numpy as np
import pandas as pd
from monotonic_constraints import *

# read heart dataset
df = pd.read_csv('./Data/heart_train.csv')

# Define variable types from preprocessing script
categoricals = ['sex', 'cp',
                 'fbs', 'restecg', 'exang', 'slope', 'ca', 
                 'thal']
numericals = ['age', 'trestbps','chol',
            'thalach', 'oldpeak']
output_col = 'target'

lattice_heart = LatticeDataset(numericals, categoricals, output_col)

lattice_heart.fit(df)

configs, features, target = lattice_heart.transform(df,heart_monotonicity) # heart_monotonicity is from monotonic_constraints.py file

# test that random lattice ensemble compiles and trains a single epoch successfully
model = random_lattice_ensemble_model(configs,target)

model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.016), 
    metrics = ['accuracy'])

model.fit(x=features, y = target, batch_size = 512)


# test that lattice model compiles and trains a single epoch successfully
model = lattice_model(configs,target)

model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.016), 
    metrics = ['accuracy'])

model.fit(x=features, y = target, batch_size = 512)
