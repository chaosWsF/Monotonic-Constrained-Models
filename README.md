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

---
## Project Outline

### The Inital Work & New Topic

Predict the probability of loan default given previous loan performance

+ Used transaction based data to enrich the existing loan specific fields to significantly increase the performance of our default predictions

**Introducing Monotonic Constraints**

1. Force a ML model to only learn **monotonic** relationships in the data (*like regularization*)
2. *Lattice regression* for NN -- `lattice` in `tensorflow`
3. Imposing constraints on each decision tree in tree ensembles -- `XGBoost`

### Objective

Compare the performance of 5 kinds of models (*RF, XGBoost with/without monotonicity, NN with/without monotonicity*) on 1-3 (time permitting) datasets to answer the following questions:

1. Given monotonic constraints, how much does the **performance drop** for neural networks and xgboost, can it still **outperform a standard random forest**?
2. Are the monotonically constrained **neural networks** competitive with the constrained version of **xgboost**? This was not compared in the original *reference* paper.

### Methodology

**Performance Measure**

1. For the classification datasets, the area under the ROC curve or area under the precision recall curve can be used for evaluation.
2. For regression, mean squared error, mean absolute error and root mean squared error should be used.

**Validation**

Any validation strategy can be used to tune model hyperparameters.

**Coding**

Ideally, aside from the initial data preprocessing **each dataset should use a very similar training framework** to tune and evaluate the models.

Q: How to label monotonicity?

A: Just use the arguments in training framework. (Go to reference)

### Further

+ Monotonicity vs other shape constraints vs interaction constraints
+ Financial dataset common sense variables that should be constrained.

### Schedule

| Task | Timelines |
| --- | --- |
| Initial environment setup, download datasets, test packages, review as needed | 2 weeks |
| Develop Model Pipeline | 2-3 weeks |
| Dataset preprocessing + experiments | 3-5 weeks |

### Reference

1. `XGBoost`
   + https://xgboost.readthedocs.io/en/latest/tutorials/monotonic.html
2. `Tensorflow`
   + tensorflow.org/lattice/overview
   + Example: tensorflow.org/lattice/tutorials/premade_models
3. Papers
   1. Gupta2016
      + Slightly better than RF
      + No comparisons between calibrated and uncalibrated NN or compared to xgboost with constraints
      + A *lattice* is an interpolated look-up table that can approximate arbitrary input-output relationships in your data.
      + Lattice regression
      + `Tensorflow`
   2. Appendix of Popov2019
      + tabular data
      + ensembles of oblivious decision trees
      + `Pytorch`
