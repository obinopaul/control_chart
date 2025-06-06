import numpy as np
import optuna
import warnings
import os
import pandas as pd
from ol_train import ol_train

def CV_algorithm(Y, X, options, dataset_name):
    method = options.method.upper()

    if method in ['PERCEPTRON', 'PA_L1', 'PA_L2', 'PA', 'CPA', 'PA_L1', 'PA_L2']:
        # Directly train without hyperparameter tuning
        return options  # No need to modify options

    elif method in ['PA1', 'PA1_L1', 'PA1_L2', 'PA2', 'PA2_L1', 'PA2_L2', 'OGD', 'OGD_1', 'OGD_2', 'CSOGD_1', 'CSOGD_2', 'CPA1', 'CPA2', 'PA1_Csplit', 'PA2_Csplit', 'PA1_CSPLIT', 'PA2_CSPLIT', 'PA_I_L1', 'PA_I_L2', 'PA_II_L1', 'PA_II_L2']:
        # Perform hyperparameter tuning and update options
        options = tune_hyperparameter(Y, X, options, ['C'], dataset_name)

    # Return only the options with the best hyperparameters
    return options

def tune_hyperparameter(Y, X, options, params, dataset_name, metric='gmeans'):
    optuna.logging.set_verbosity(optuna.logging.WARN)
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    def objective(trial):
        for param in params:
            if param == 'C':
                options.C = trial.suggest_loguniform('C', options.range_C[0], options.range_C[-1])
            elif param == 'eta':
                options.eta = trial.suggest_float('eta', options.range_eta[0], options.range_eta[-1], step=0.05)
            elif param == 'b':
                options.b = trial.suggest_float('b', options.range_b[0], options.range_b[-1], step=0.1)
            elif param == 'p':
                options.p = trial.suggest_int('p', options.range_p[0], options.range_p[-1], step=2)
            elif param == 'C_pos':
                options.C_pos = trial.suggest_loguniform('C_pos', options.range_C_pos[0], options.range_C_pos[-1])
            elif param == 'C_neg':
                options.C_neg = trial.suggest_loguniform('C_neg', options.range_C_neg[0], options.range_C_neg[-1])
                
        _, result = ol_train(Y, X, options)
        result_metric_value = np.mean(result[metric])

        if metric == 'err_count':
            return result['err_count']
        elif metric == 'f1':
            f1_scores = 2 * (result['precisions'] * result['sensitivities']) / (result['precisions'] + result['sensitivities'])
            return -np.nanmean(f1_scores)
        elif metric == 'gmeans':
            return -np.nanmean(result['gmeans'])
        elif metric == 'mccs':
            return -np.nanmean(result['mccs'])
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)

    best_trial = study.best_trial

    for param in params:
        if param == 'C' and 'C' in best_trial.params:
            options.C = best_trial.params['C']
        elif param == 'eta' and 'eta' in best_trial.params:
            options.eta = best_trial.params['eta']
        elif param == 'b' and 'b' in best_trial.params:
            options.b = best_trial.params['b']
        elif param == 'p' and 'p' in best_trial.params:
            options.p = best_trial.params['p']

            
    return options  # Return options with the best hyperparameters
