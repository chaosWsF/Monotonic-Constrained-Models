# Monotonic-Constrained-Models

## Data

| dataset | link |
| --- | --- |   
| heart | <http://storage.googleapis.com/download.tensorflow.org/data/heart.csv> |
| adult | <https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv> |
| housing | <https://data.edmonton.ca/City-Administration/Property-Assessment-Data-2012-2019-/qi6a-xuwt> |

- data_cleaning.py
  ```
  import numpy as np
  import pandas as pd
  from sklearn.model_selection import train_test_split, ShuffleSplit
  from sklearn.preprocessing import StandardScaler, Normalizer, RobustScaler, PowerTransformer, QuantileTransformer
  from sklearn.compose import ColumnTransformer
  from sklearn.pipeline import Pipeline
  from sklearn.impute import SimpleImputer, MissingIndicator
  from category_encoders import LeaveOneOutEncoder    # library Catergory Encoders
  ```

- data_loading.py
  - works for train/test data
  - spliting into X & Y
  - get names of features/labels

- data_not_good.ipynb
  - the adult dataset from [here](https://archive.ics.uci.edu/ml/datasets/census+income) has a bad format

## Monotonicity

- explore_monotonicity.ipynb
  - `interpret` from interpretml/interpret
  - `scipy.stats.spearmanr`
  - `pymannkendall.original_test`
    - slow for large data
    - need to be combined with random selection

## Models

- trees_trainer.py
- trees_tuning.py
