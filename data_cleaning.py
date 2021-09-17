import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.preprocessing import StandardScaler, Normalizer, RobustScaler, PowerTransformer, QuantileTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, MissingIndicator
from category_encoders import LeaveOneOutEncoder    # library Category Encoders


data_root = './Data/'    # location of datasets
test_size = 0.2
seed = 684

## Housing Data set
df = pd.read_csv(f'{data_root}Property_Assessment_Data__2012_-_2019_.csv')

df = df[(df['Assessment Year'] == 2019) & (df['Assessment Class 1'] == 'RESIDENTIAL') & (df['Assessed Value'] > 0)].copy()

input_cols = ['Latitude', 'Longitude', 'Neighbourhood', 'Actual Year Built', 'Garage', 'Zoning', 'Lot Size']
output_col = ['Assessed Value']

# Drop columns not used for input or output
df = df[input_cols + output_col].copy()

# Scale output
output_scaler = PowerTransformer(method='yeo-johnson').fit(df['Assessed Value'].values.reshape(-1,1))    # Needs to be reshaped before transform
df['Assessed_value_log'] = output_scaler.transform(df['Assessed Value'].values.reshape(-1,1))

# Convert categoricals to integer
categoricals = ['Neighbourhood', 'Garage', 'Zoning']

for column in categoricals:
    df[column] = df[column].astype('category')
    df[column] = df[column].cat.codes    # Will assign an integer for each category, assigning -1 to missing entries

# Scale numericals
numericals = ['Latitude', 'Longitude', 'Actual Year Built', 'Lot Size']
for column in numericals:
    df[column] = PowerTransformer(method='yeo-johnson').fit_transform(df[column].values.reshape(-1,1))

df.replace(to_replace=[np.inf,-np.inf], value=[np.nan,np.nan], inplace=True)

df = df.reset_index(drop=True)

train_idx, test_idx = train_test_split(df.index.values, test_size=test_size, random_state=seed)

train_df = df.iloc[train_idx].copy()
test_df = df.iloc[test_idx].copy()

train_df.to_csv(f'{data_root}housing_train.csv')
test_df.to_csv(f'{data_root}housing_test.csv')


## Census / Adult dataset
df = pd.read_csv(f'{data_root}adult.data', header=None)
df.columns = [
    "Age", "WorkClass", "fnlwgt", "Education", "EducationNum",
    "MaritalStatus", "Occupation", "Relationship", "Race", "Gender",
    "CapitalGain", "CapitalLoss", "HoursPerWeek", "NativeCountry", "Income"
]

df['target'] = df['Income'].astype('category')
df['target'] = df['target'].cat.codes

input_cols = [
    'Age', 'WorkClass', 'fnlwgt', 'Education', 'EducationNum',
    'MaritalStatus', 'Occupation', 'Relationship', 'Race', 'Gender', 
    'CapitalGain', 'CapitalLoss', 'HoursPerWeek', 'NativeCountry'
]

output_col = ['Income', 'target']

# Drop columns not used for input or output
df = df[input_cols + output_col].copy()

# Convert categoricals to integer
categoricals = [
    'WorkClass', 'Education',
    'MaritalStatus', 'Occupation', 'Relationship', 'Race', 'Gender', 
    'NativeCountry'
]
for column in categoricals:
    df[column] = df[column].astype('category')
    df[column] = df[column].cat.codes    # Will assign an integer for each category, assigning -1 to missing entries

# Scale numericals
numericals = [
    'Age', 'fnlwgt', 'EducationNum',
    'CapitalGain', 'CapitalLoss', 'HoursPerWeek'
]
for column in numericals:
    df[column] = PowerTransformer(method='yeo-johnson').fit_transform(df[column].values.reshape(-1,1))
    
df.replace(to_replace=[np.inf,-np.inf], value=[np.nan,np.nan], inplace=True)

df = df.reset_index(drop=True)
train_idx, test_idx = train_test_split(df.index.values, test_size=test_size, random_state=seed)

train_df = df.iloc[train_idx].copy()
test_df = df.iloc[test_idx].copy()

train_df.to_csv(f'{data_root}adult_train.csv')
test_df.to_csv(f'{data_root}adult_test.csv')


## Heart
df = pd.read_csv(f'{data_root}heart.csv')

input_cols = [
    'age',
    'sex',
    'cp',
    'trestbps',
    'chol',
    'fbs',
    'restecg',
    'thalach',
    'exang',
    'oldpeak',
    'slope',
    'ca',
    'thal'
]

output_col = ['target']

# Drop columns not used for input or output
df = df[input_cols + output_col].copy()

# Convert categoricals to integer
categoricals = [
    'sex', 'cp',
    'fbs', 'restecg', 'exang', 'slope', 'ca', 
    'thal'
]
for column in categoricals:
    df[column] = df[column].astype('category')
    df[column] = df[column].cat.codes    # Will assign an integer for each category, assigning -1 to missing entries

# Scale numericals
numericals = [
    'age', 'trestbps', 'chol',
    'thalach', 'oldpeak'
]
for column in numericals:
    df[column] = StandardScaler().fit_transform(df[column].values.reshape(-1,1))
    
df.replace(to_replace=[np.inf,-np.inf], value=[np.nan,np.nan], inplace=True)

df = df.reset_index(drop=True)
train_idx,test_idx = train_test_split(df.index.values, test_size=test_size, random_state=seed)

train_df = df.iloc[train_idx].copy()
test_df = df.iloc[test_idx].copy()

train_df.to_csv(f'{data_root}heart_train.csv')
test_df.to_csv(f'{data_root}heart_test.csv')
