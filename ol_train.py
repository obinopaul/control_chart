from init_model import Model
from math import floor
import numpy as np
import imp
import importlib
import importlib.util 
import time
import copy 
from pyswarm import pso  # PSO library for optimization
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score, matthews_corrcoef
import warnings 

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    with np.errstate(divide='ignore', invalid='ignore'):
        specificity = np.mean([cm[i, i] / np.sum(cm[:, i]) if np.sum(cm[:, i]) != 0 else 0 for i in range(len(cm))])
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    sensitivity = recall_score(y_true, y_pred, average='macro', zero_division=0)
    gmean = np.sqrt(sensitivity * specificity)
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    return accuracy, sensitivity, specificity, precision, gmean, kappa, mcc, cm

def ol_train(Y, X, options):
    ID = options.id_list.astype(int)
    n = len(ID)
    d = X.shape[1]
    t_tick = options.t_tick
    num_ticks = (n // t_tick) + (1 if n % t_tick != 0 else 0) # Number of ticks 
    
    # Initialize storage for metrics
    mistakes = np.zeros(num_ticks)
    nb_SV = np.zeros(num_ticks)
    ticks = np.zeros(num_ticks)
    captured_t = np.zeros(num_ticks)
    accuracies = np.zeros(num_ticks)
    sensitivities = np.zeros(num_ticks)
    specificities = np.zeros(num_ticks)
    precisions = np.zeros(num_ticks)
    gmeans = np.zeros(num_ticks)
    kappas = np.zeros(num_ticks)
    mccs = np.zeros(num_ticks)
    cumulative_errors = np.zeros(num_ticks)
    confusion_matrices = []

    num_SV = 0
    err_count = 0
    if options.task_type == 'bc':
        nb_class = 2
    elif options.task_type == 'mc':
        nb_class = len(np.unique(Y))

    # Dynamically load the algorithm module
    method = options.method
    module_name = f'algorithms.{method}'
    module_spec = importlib.util.find_spec(module_name)
    if module_spec is None:
        raise ImportError(f'Could not find module {module_name}')
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    func = getattr(module, method)

    model = Model(options, d, nb_class)
    
    start_time = time.time() 
    idx = 0
    y_pred_all = []
    

    # Initialize cost values for the cost matrix
    total_samples = len(Y)
    num_positive = np.sum(Y == 1)
    num_negative = np.sum(Y == -1)
   
    for t in range(n):
        _id = ID[t]
        y_t = Y[_id]
        x_t = X[_id].reshape(1, -1)

        if y_t.size == 0 or x_t.size == 0:
            print(f"Skipping sample {_id} due to empty y_t or x_t")
            continue
       
        # Get prediction and loss for the sample
        eta_p = 0.5
        eta_n = 0.5

        # Calculate ratio, handling the division-by-zero case
        if num_negative != 0:
            ratio_Tp_Tn = float(num_positive / num_negative)
        else:
            ratio_Tp_Tn = float('inf')  # or some fallback value
        
        model, hat_y_t, l_t = func(y_t, x_t, model, eta_p, eta_n, num_positive, num_negative)  # Pass weights to the algorithm
        
        y_pred_all.append(hat_y_t)
        if hat_y_t != y_t:
            err_count += 1

        if l_t > 0:
            num_SV += 1

        run_time = time.time() - start_time

        if (t + 1) % t_tick == 0 or (t + 1) == n:
            y_true = Y[ID[:t + 1]]
            y_pred = np.array(y_pred_all)

            if y_true.size == 0 or y_pred.size == 0:
                print(f"Skipping metrics calculation at sample {t + 1} due to empty y_true or y_pred")
                continue

            accuracy, sensitivity, specificity, precision, gmean, kappa, mcc, cm = calculate_metrics(y_true, y_pred)

            if idx >= len(mistakes):
                mistakes = np.append(mistakes, np.zeros(1))
                nb_SV = np.append(nb_SV, np.zeros(1))
                ticks = np.append(ticks, np.zeros(1))
                captured_t = np.append(captured_t, np.zeros(1))
                accuracies = np.append(accuracies, np.zeros(1))
                sensitivities = np.append(sensitivities, np.zeros(1))
                specificities = np.append(specificities, np.zeros(1))
                precisions = np.append(precisions, np.zeros(1))
                gmeans = np.append(gmeans, np.zeros(1))
                kappas = np.append(kappas, np.zeros(1))
                mccs = np.append(mccs, np.zeros(1))
                cumulative_errors = np.append(cumulative_errors, np.zeros(1))
                confusion_matrices.append(np.zeros(cm.shape))

            mistakes[idx] = err_count / (t + 1)
            nb_SV[idx] = num_SV
            ticks[idx] = run_time
            captured_t[idx] = t + 1
            accuracies[idx] = accuracy
            sensitivities[idx] = sensitivity
            specificities[idx] = specificity
            precisions[idx] = precision
            gmeans[idx] = gmean
            kappas[idx] = kappa
            mccs[idx] = mcc
            cumulative_errors[idx] = err_count / (t + 1)
            confusion_matrices.append(cm)
            
            idx += 1

    run_time = time.time() - start_time
    result = {
        'run_time': run_time,
        'err_count': err_count,
        'mistakes': mistakes,
        'ticks': ticks,
        'nb_SV': nb_SV,
        'captured_t': captured_t,
        'accuracies': accuracies,
        'sensitivities': sensitivities,
        'specificities': specificities,
        'precisions': precisions,
        'gmeans': gmeans,
        'kappas': kappas,
        'mccs': mccs,
        'cumulative_errors': cumulative_errors,
        'confusion_matrices': confusion_matrices
    }
    model.final_nb_SV = num_SV
    return model, result
