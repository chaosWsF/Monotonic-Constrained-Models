import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, average_precision_score, roc_auc_score, brier_score_loss

from monotonic_constraints import *

import tensorflow as tf
import tensorflow_lattice as tfl


def classification_evaluator(model, X_train, y_train, X_test, y_test, train_parameters):    
    history = model.fit(X_train,y_train, validation_data= (X_test, y_test), **train_parameters)    
    
    probabilities = model.predict(X_test)
    
    metrics = {}
    metrics['early_stopping_epochs'] = np.argmin(history.history['val_loss']) + 1
    metrics['AUC'] = roc_auc_score(y_test, probabilities)
    metrics['AP'] = average_precision_score(y_test, probabilities)
    metrics['brier_score'] = brier_score_loss(y_test, probabilities)    
    
    return model, metrics


def regression_evaluator(model, X_train, y_train, X_test, y_test, train_parameters):
    history = model.fit(X_train,y_train, validation_data= (X_test, y_test), **train_parameters)
    
    y_pred = model.predict(X_test)
    
    metrics = {}
    metrics['early_stopping_epochs'] = np.argmin(history.history['val_loss']) + 1
    metrics['MSE'] = mean_squared_error(y_test, y_pred)
    metrics['MAE'] = mean_absolute_error(y_test, y_pred)
    metrics['RMSE'] = mean_squared_error(y_test, y_pred, squared=False)
    metrics['MAPE'] = mean_absolute_percentage_error(y_test, y_pred)
    
    return model, metrics
