import pandas as pd


class Data:
    def __init__(self, data_name, data_root, data_type='train'):
        self.data_name = data_name
        self.data_loc = f'{data_root}{data_name}_{data_type}.csv'

    def load(self):
        df = pd.read_csv(self.data_loc)
        df.dropna(inplace=True)

        data_name = self.data_name
        if data_name == 'adult':
            feature_names = [
                'Age', 'WorkClass', 'fnlwgt', 'Education',
                'EducationNum', 'MaritalStatus', 'Occupation',
                'Relationship', 'Race', 'Gender', 'CapitalGain',
                'CapitalLoss', 'HoursPerWeek', 'NativeCountry'
            ]
            target_var = 'target'
        elif data_name == 'heart':
            feature_names = [
                'age', 'sex', 'cp', 'trestbps', 'chol', 
                'fbs', 'restecg', 'thalach', 'exang', 
                'oldpeak', 'slope', 'ca', 'thal'
            ]
            target_var = 'target'
        elif data_name == 'housing':
            feature_names = [
                'Latitude', 'Longitude', 'Neighbourhood', 
                'Actual Year Built', 'Garage', 'Zoning', 'Lot Size'
            ]
            target_var = 'Assessed_value_log'
        else:    # TODO 'synthetic'
            print('Waiting for synthetic demos')

        X = df[feature_names].to_numpy()
        Y = df[target_var].to_numpy()
        self.feature_names = feature_names

        return X, Y

    def split(self, train_ratio):    # used for training
        X, Y = self.load()

        valid_cutoff = int(train_ratio * X.shape[0])

        X_train = X[0:valid_cutoff, :]
        Y_train = Y[0:valid_cutoff]

        X_val = X[valid_cutoff:, :]
        Y_val = Y[valid_cutoff:]

        return X_train, Y_train, X_val, Y_val, self.feature_names
