import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_lattice as tfl


## Dataset Monotonicities

# adult dataset
adult_monotonicity = {
    # 'fnlwgt': '-1', 
    'EducationNum': '1', 
    'CapitalGain':'1'
}

# residential housing
housing_monotonicity = {
    'Lot Size': '1'
}

# heart dataset
heart_monotonicity = {
    # 'cp': '1',
    # 'slope': '1',
    # 'ca': '1',
    'thalach': '-1',
    'oldpeak': '1'
}

# synthetic test dataset
test_monotonicity = {
    'x1': '1',
    'x2': '-1'
}


## Generate synthetic test dataset
def generate_synthetic_dataset(size = 800):    
    x1 = np.random.uniform(size=size)
    x2 = np.random.uniform(size=size)
    noise = np.random.normal(scale = 0.01, size = size)
    
    y = 5.0*x1 + np.sin(10.0*np.pi*x1) - 5.0*x2 -np.cos(10.0*np.pi*x2) + noise
    
    return pd.DataFrame({'x1':x1, 'x2':x2, 'y':y})


## Xgboost helpers
def xgboost_monotonicity_string(feature_list, monotonicities): 
    
    return '(' + ','.join([monotonicities[x] if x in monotonicities else '0' for x in feature_list]) + ')'


def get_monotonicity_string(feature_list, dataset = None):
    
    if dataset == 'adult':
        return xgboost_monotonicity_string(feature_list, adult_monotonicity)
    
    elif dataset == 'housing':
        # return "(0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0)"
        return xgboost_monotonicity_string(feature_list, housing_monotonicity)
    
    elif dataset == 'heart':
        return xgboost_monotonicity_string(feature_list, heart_monotonicity)
    
    elif dataset == 'synthetic':
        return xgboost_monotonicity_string(feature_list, test_monotonicity)
    
    else:
         return xgboost_monotonicity_string(feature_list, {})


## Tensorflow lattice helpers

# quantile function from tutorial
def compute_quantiles(features,
                      num_keypoints=10,
                      clip_min=None,
                      clip_max=None,
                      missing_value=None):
    # Clip min and max if desired.
    if clip_min is not None:
        features = np.maximum(features, clip_min)
        features = np.append(features, clip_min)
    if clip_max is not None:
        features = np.minimum(features, clip_max)
        features = np.append(features, clip_max)
        # Make features unique.
        #unique_features = np.unique(features)
    # Remove missing values if specified.
    #if missing_value is not None:
        #unique_features = np.delete(unique_features,
       #                             np.where(unique_features == missing_value))
    # Compute and return quantiles over unique non-missing feature values.
    return np.quantile(
      features,
      np.linspace(0., 1., num=num_keypoints),
      interpolation='nearest').astype(float)


class LatticeDataset:    
    def __init__(self, num_cols, cat_cols, output_col, keypoints = 10, lattice_size = 3):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.output_col = output_col
        
        self.num_quantiles = {}
        self.cat_sizes = {}
        
        
        self.keypoints = keypoints        
        self.lattice_size = lattice_size # I don't think lattice_size does anything
        
    def fit(self,df):
        for c in self.num_cols:
            self.num_quantiles[c] = compute_quantiles(
            np.unique(df[c].values),
            num_keypoints=self.keypoints)
            
        for c in self.cat_cols:
            self.cat_sizes[c] = df[c].nunique()
            
    def transform(self,df,monotonicities):
        assert( len(self.num_cols) == len(self.num_quantiles))
        assert( len(self.cat_cols) == len(self.cat_sizes))
        
        feature_configs = []
        input_list = []
        
        for c in df.columns.to_list():
            
            # Process numerical
            if c in self.num_cols:
                feature_monotonicity = 'none'
                
                if c in monotonicities:
                    feature_monotonicity = 'increasing' if monotonicities[c] == '1' else 'decreasing'
                
                feature_configs.append(tfl.configs.FeatureConfig(
                                        name= c,
                                        lattice_size= self.lattice_size,
                                        monotonicity= feature_monotonicity,                                        
                                        pwl_calibration_num_keypoints= self.keypoints,
                                        pwl_calibration_input_keypoints= self.num_quantiles[c],                                        
                                        regularizer_configs= None
                                    ))
                input_list.append(df[c].values)
                
            # Process categorical
            if c in self.cat_cols:
                feature_configs.append(tfl.configs.FeatureConfig(
                                        name=c,
                                        num_buckets=self.cat_sizes[c],
                                    ))
                input_list.append(df[c].values)
                
        return feature_configs, input_list, df[self.output_col].values


def lattice_model(feature_configs, label, numerical_error_epsilon = 1e-5, random_seed = 42):
    min_label, max_label = float(np.min(label)), float(np.max(label))
    
    lattice_model_config = tfl.configs.CalibratedLatticeConfig(
    feature_configs=feature_configs,
    output_min= min_label,
    output_max=max_label - numerical_error_epsilon,
    output_initialization=[min_label, max_label],
    random_seed=random_seed)

    lattice_model = tfl.premade.CalibratedLattice(
    lattice_model_config)
    
    return lattice_model


def random_lattice_ensemble_model(feature_configs, label, num_lattices = 5, lattice_rank = 3, numerical_error_epsilon = 1e-5, random_seed = 42):
    min_label, max_label = float(np.min(label)), float(np.max(label))
    
    random_ensemble_model_config = tfl.configs.CalibratedLatticeEnsembleConfig(
    feature_configs=feature_configs,
    lattices='random',
    num_lattices= num_lattices,
    lattice_rank= lattice_rank,
    output_min= min_label,
    output_max=max_label - numerical_error_epsilon,
    output_initialization=[min_label, max_label],
    random_seed=random_seed)
    
    tfl.premade_lib.set_random_lattice_ensemble(random_ensemble_model_config)
    
    random_ensemble_model = tfl.premade.CalibratedLatticeEnsemble(
    random_ensemble_model_config)   
    
    
    return random_ensemble_model


## feature config for housing dataset
def housing_config(X,Y):
    min_label, max_label = float(np.min(Y)), float(np.max(Y))
    numerical_error_epsilon = 1e-5
    
    feature_configs = [ 
    tfl.configs.FeatureConfig(
        name='lat',
        lattice_size=3,
        monotonicity='none',
        # We must set the keypoints manually.
        pwl_calibration_num_keypoints=10,
        pwl_calibration_input_keypoints=compute_quantiles(
            X[:,0],
            num_keypoints=10),
        # Per feature regularization.
        regularizer_configs=None
    ),
    tfl.configs.FeatureConfig(
        name='long',
        lattice_size=3,
        monotonicity='none',
        # We must set the keypoints manually.
        pwl_calibration_num_keypoints=10,
        pwl_calibration_input_keypoints=compute_quantiles(
            X[:,1],
            num_keypoints=10),
        # Per feature regularization.
        regularizer_configs=None
    ),
    
    tfl.configs.FeatureConfig(
        name='neighbourhood',
        lattice_size=3,
        monotonicity='none',
        # We must set the keypoints manually.
        pwl_calibration_num_keypoints=10,
        pwl_calibration_input_keypoints=compute_quantiles(
            X[:,2],
            num_keypoints=10),
        # Per feature regularization.
        regularizer_configs=None
    ),
    
    tfl.configs.FeatureConfig(
        name='year',
        lattice_size=3,
        monotonicity='none',
        # We must set the keypoints manually.
        pwl_calibration_num_keypoints=10,
        pwl_calibration_input_keypoints=compute_quantiles(
            X[:,3],
            num_keypoints=10),
        # Per feature regularization.
        regularizer_configs=None
    ),
    tfl.configs.FeatureConfig(
        name='Garage',
        lattice_size=3,
        monotonicity='none',
        # We must set the keypoints manually.
        pwl_calibration_num_keypoints=10,
        pwl_calibration_input_keypoints=compute_quantiles(
            X[:,4],
            num_keypoints=10),
        # Per feature regularization.
        regularizer_configs=None
    ),
    tfl.configs.FeatureConfig(
        name='zone',
        lattice_size=3,
        monotonicity='none',
        # We must set the keypoints manually.
        pwl_calibration_num_keypoints=10,
        pwl_calibration_input_keypoints=compute_quantiles(
            X[:,5],
            num_keypoints=10),
        # Per feature regularization.
        regularizer_configs=None
    ),
    tfl.configs.FeatureConfig(
        name='Lot_size',
        lattice_size=3,
        monotonicity='increasing',
        # We must set the keypoints manually.
        pwl_calibration_num_keypoints=10,
        pwl_calibration_input_keypoints=compute_quantiles(
            X[:,6],
            num_keypoints=10),
        # Per feature regularization.
        regularizer_configs=None
    )
    
    
]
    # Model
    random_ensemble_model_config = tfl.configs.CalibratedLatticeConfig(
    feature_configs=feature_configs,
    output_min=min_label,
    output_max=max_label - numerical_error_epsilon,
    output_initialization=[min_label, max_label],
    random_seed=42)
    
    random_ensemble_model = tfl.premade.CalibratedLattice(
    random_ensemble_model_config)
    
    features = [X[:,0],X[:,1],X[:,2],X[:,3],X[:,4],X[:,5],X[:,6]]
    random_ensemble_model.compile(
    loss='mse',
    optimizer=tf.keras.optimizers.Adam(0.016), 
    metrics = ['mean_absolute_percentage_error'])
    
    return features, random_ensemble_model
