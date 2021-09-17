import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, average_precision_score, roc_auc_score, brier_score_loss
from xgboost import XGBRFClassifier, XGBRFRegressor, XGBClassifier, XGBRegressor


def classification_evaluator(model, X_train, y_train, X_test, y_test):
    # Fit train, evaluate on test set, return model and metrics
    model.fit(X_train, y_train, eval_metric='logloss')
    
    probabilities = model.predict_proba(X_test)[:,1]
    
    metrics = {}
    metrics['AUC'] = roc_auc_score(y_test, probabilities)
    metrics['AP'] = average_precision_score(y_test, probabilities)
    metrics['brier_score'] = brier_score_loss(y_test, probabilities)
    
    return model, metrics


def regression_evaluator(model, X_train, y_train, X_test, y_test):
    # Fit train, evaluate on test set, return model and metrics
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    metrics = {}
    metrics['MSE'] = mean_squared_error(y_test, y_pred)
    metrics['MAE'] = mean_absolute_error(y_test, y_pred)
    metrics['RMSE'] = mean_squared_error(y_test, y_pred, squared=False)
    metrics['MAPE'] = mean_absolute_percentage_error(y_test, y_pred)
    
    return model, metrics


models = {
    'RF': {'classification': XGBRFClassifier, 'regression': XGBRFRegressor}, 
    'XGB': {'classification': XGBClassifier, 'regression': XGBRegressor}
}
evaluators = {'classification': classification_evaluator, 'regression': regression_evaluator}


class Model:
    def __init__(self, model_name, model_type):
        self.model_name = model_name.split('_')
        self.model_type = model_type
    
    def setup(self, hyper_parameters):
        mname = self.model_name
        mtype = self.model_type

        mfunc = models[mname[0]][mtype]

        if len(mname) == 1:    # no constraints
            hyper_parameters['monotone_constraints'] = hyper_parameters['monotone_constraints'].replace('-1', '0')
            hyper_parameters['monotone_constraints'] = hyper_parameters['monotone_constraints'].replace('1', '0')
        
        if mname[0] == 'RF':
            hyper_parameters['booster'] = 'gbtree'
            if mtype == 'regression':
                hyper_parameters['eta'] = 1

        if mtype == 'classification':
            hyper_parameters['use_label_encoder'] = False

        model = mfunc(**hyper_parameters)

        evaluator = evaluators[mtype]

        return model, evaluator

    def fit_eval(self, params, X_train, Y_train, X_val, Y_val):
        model, evaluator = self.setup(params)
        fitted_model, metrics = evaluator(model, X_train, Y_train, X_val, Y_val)
        return metrics, fitted_model
    
    def record(self, params, X_train, Y_train, X_val, Y_val, records, log_file):
        metrics, _ = self.fit_eval(params, X_train, Y_train, X_val, Y_val)

        records.append({**params, **metrics})

        pd.DataFrame.from_records(records, index = range(len(records))).to_csv(log_file, index = False)
