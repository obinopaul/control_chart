from init_model import Model
from math import floor
import numpy as np
import imp
import time
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
    batch_size = options.batch_size  # Add a new option for batch size
    
    mistakes = np.zeros(floor(n/t_tick))
    nb_SV = np.zeros(floor(n/t_tick))
    ticks = np.zeros(floor(n/t_tick))
    captured_t = np.zeros(floor(n/t_tick))
    accuracies = np.zeros(floor(n/t_tick))
    sensitivities = np.zeros(floor(n/t_tick))
    specificities = np.zeros(floor(n/t_tick))
    precisions = np.zeros(floor(n/t_tick))
    gmeans = np.zeros(floor(n/t_tick))
    kappas = np.zeros(floor(n/t_tick))
    mccs = np.zeros(floor(n/t_tick))
    cumulative_errors = np.zeros(floor(n/t_tick))
    confusion_matrices = []
    
    num_SV = 0
    err_count = 0
    if options.task_type == 'bc':
        nb_class = 2
    elif options.task_type == 'mc':
        nb_class = len(np.unique(Y))

    model = Model(options, d, nb_class)

    start_time = time.time()
    
    f_ol = options.method

    module = imp.load_source(f_ol, './algorithms/' + f_ol + '.py')
    func = getattr(module, f_ol)
    
    idx = 0
    y_pred_all = []

    for t in range(0, len(ID), batch_size):
        batch_ids = ID[t:t+batch_size]
        y_t_batch = Y[batch_ids]
        x_t_batch = X[batch_ids]
        
        for y_t, x_t in zip(y_t_batch, x_t_batch):
            model, hat_y_t, l_t = func(y_t, x_t, model)
            y_pred_all.append(hat_y_t)
        
            if hat_y_t != y_t:
                err_count += 1

            if l_t > 0:
                num_SV += 1

        run_time = time.time() - start_time
        
        if (t + batch_size) % t_tick == 0:
            y_true = Y[ID[:t+batch_size]]
            y_pred = np.array(y_pred_all)
            accuracy, sensitivity, specificity, precision, gmean, kappa, mcc, cm = calculate_metrics(y_true, y_pred)
            
            mistakes[idx] = err_count / (t + batch_size)
            nb_SV[idx] = num_SV
            ticks[idx] = run_time
            captured_t[idx] = t + batch_size
            accuracies[idx] = accuracy
            sensitivities[idx] = sensitivity
            specificities[idx] = specificity
            precisions[idx] = precision
            gmeans[idx] = gmean
            kappas[idx] = kappa
            mccs[idx] = mcc
            cumulative_errors[idx] = err_count / (t + batch_size)
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
