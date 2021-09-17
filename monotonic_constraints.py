import numpy as np
import pandas as pd


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

