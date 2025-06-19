import argparse
import numpy as np
import sys 
import os 
import pandas as pd 
import matplotlib.pyplot as plt 
from numpy.random import RandomState
from math import floor
from load_data import load_data
from init_options import Options
from test_options import TestOptions
from ol_train import ol_train
from CV_algorithm import CV_algorithm
from arg_check import arg_check
from handle_parameters import handle_parameters

def run(task_type, algorithm_name, dataset_name, file_format, nb_runs=1, shuffle_data=True, print_results=True, test_parameters=False, loss_type=None):
    # Initialize the options for each method
    return_vals = load_data(dataset_name, file_format, task_type)

    if return_vals is None:
        sys.exit("Argument error.")
    
    xt, y, n = return_vals

    # Shuffle the data
    indices = np.random.permutation(n)  # Generate a random permutation of indices
    xt = xt[indices]  # Shuffle xt
    y = y[indices]    # Shuffle y
    
    # Check argument
    if arg_check(task_type, y) != 0:
        print("Error: Dataset is not for ", task_type, " task.")
        return

    # Perform hyperparameter optimization once across all runs
    _options = Options(algorithm_name, n, task_type)
    _options = CV_algorithm(y, xt, _options, dataset_name)
    
    # Collect only the hyperparameters that are defined in the options object
    best_hyperparameters = {param: getattr(_options, param, None) for param in ['C', 'eta', 'b', 'p', 'C_pos', 'C_neg'] if hasattr(_options, param)}

    # Save hyperparameters once before the loop
    save_hyperparameters(algorithm_name, best_hyperparameters, dataset_name)  # Pass best_hyperparameters here

    # Generate test ID sequence
    ID_list = np.zeros((nb_runs, n), dtype=int)
    for i in range(nb_runs):
        if shuffle_data:
            ID_list[i] = np.random.permutation(n)
        else:
            ID_list[i] = np.arange(n) 
    
    # Initialize arrays with algorithm stats
    err_count_arr = np.zeros(nb_runs)
    nSV_arr = np.zeros(nb_runs)
    time_arr = np.zeros(nb_runs)

    # Performance metric arrays
    max_ticks = floor(n / _options.t_tick) + 1
    mistakes_arr = np.zeros((max_ticks, nb_runs))
    nb_SV_cum_arr = np.zeros((max_ticks, nb_runs))
    time_cum_arr = np.zeros((max_ticks, nb_runs))
    accuracies_arr = np.zeros((max_ticks, nb_runs))
    sensitivities_arr = np.zeros((max_ticks, nb_runs))
    specificities_arr = np.zeros((max_ticks, nb_runs))
    precisions_arr = np.zeros((max_ticks, nb_runs))
    gmeans_arr = np.zeros((max_ticks, nb_runs))
    kappas_arr = np.zeros((max_ticks, nb_runs))
    mccs_arr = np.zeros((max_ticks, nb_runs))
    cumulative_errors_arr = np.zeros((max_ticks, nb_runs))
    sum_arr = np.zeros((max_ticks, nb_runs))  # Initialize the sum array


    for i in range(nb_runs):
        # Initialize options for each run with the best hyperparameters
        #_options = Options(algorithm_name, n, task_type)
        for param, value in best_hyperparameters.items():
            setattr(_options, param, value) 
        _options.id_list = ID_list[i] # Set the ID list for this run
            
        # Train the model using the best hyperparameters
        model, result = ol_train(y, xt, _options)

        # Save the model parameters
        if hasattr(model, 'w') and model.w is not None:
            model_parameters = {
                'Algorithm': algorithm_name,
                'Run': i + 1,
                'W': model.w.flatten().tolist(),
                'C': model.C
            }
        elif hasattr(model, 'alpha') and hasattr(model, 'SV'):
            model_parameters = {
                'Algorithm': algorithm_name,
                'Run': i + 1,
                'Alpha': model.alpha.tolist(),
                'SupportVectors': model.SV.tolist(),
                'C': model.C,
                'Sigma': getattr(model, 'sigma', None),
                'Kernel': getattr(model, 'kernel', None)
            }
        else:
            # print("Warning: Model does not have recognizable attributes for saving.")
            model_parameters = {
                'Algorithm': algorithm_name,
                'Run': i + 1,
                'W': None,
                'C': None
            } 
    
        # elif hasattr(model, 'alpha'):
        #     model_parameters = {
        #         'Algorithm': algorithm_name,
        #         'Run': i + 1,
        #         'Alpha': model.alpha.flatten().tolist(),  # For kernel methods
        #         'SV': model.SV.flatten().tolist(),
        #         'C': model.C,
        #         'Sigma': model.sigma,
        #         'Kernel': model.kernel
        #     }
            
        save_model_parameters(model_parameters, dataset_name, run_number = i + 1)  # Pass the run number
                
        # Save the best model and results across all runs
        run_time = result['run_time']
        err_count = result['err_count']
        mistakes = result['mistakes']
        ticks = result['ticks']
        nb_SV = result['nb_SV']
        captured_t = result['captured_t']
        accuracies = result['accuracies']
        sensitivities = result['sensitivities']
        specificities = result['specificities']
        precisions = result['precisions']
        gmeans = result['gmeans']
        kappas = result['kappas']
        mccs = result['mccs']
        cumulative_errors = result['cumulative_errors']
        
        err_count_arr[i] = err_count
        nSV_arr[i] = model.final_nb_SV
        time_arr[i] = run_time

        for idx in range(len(ticks)):
            mistakes_arr[idx, i] = mistakes[idx]
            nb_SV_cum_arr[idx, i] = nb_SV[idx]
            time_cum_arr[idx, i] = ticks[idx]
            accuracies_arr[idx, i] = accuracies[idx]
            sensitivities_arr[idx, i] = sensitivities[idx]
            specificities_arr[idx, i] = specificities[idx]
            precisions_arr[idx, i] = precisions[idx]
            gmeans_arr[idx, i] = gmeans[idx]
            kappas_arr[idx, i] = kappas[idx]
            mccs_arr[idx, i] = mccs[idx]
            cumulative_errors_arr[idx, i] = cumulative_errors[idx]
            
            # Calculate sum metric for each time point
            eta_p = eta_n = 0.5  
            sum_arr[idx, i] = eta_p * sensitivities[idx] + eta_n * specificities[idx]
            

    mean_error_count = round(np.mean(err_count_arr) / n, 6)
    mean_update_count = round(np.mean(nSV_arr), 6)
    mean_time = round(np.mean(time_arr), 6)
    mean_mistakes = np.mean(mistakes_arr, axis=1)
    mean_accuracies = np.mean(accuracies_arr, axis=1)
    mean_sensitivities = np.mean(sensitivities_arr, axis=1)
    mean_specificities = np.mean(specificities_arr, axis=1)
    mean_precisions = np.mean(precisions_arr, axis=1)
    mean_gmeans = np.mean(gmeans_arr, axis=1)
    mean_kappas = np.mean(kappas_arr, axis=1)
    mean_mccs = np.mean(mccs_arr, axis=1)
    mean_cumulative_errors = np.mean(cumulative_errors_arr, axis=1)
    mean_sum = np.mean(sum_arr, axis=1)  # Calculate the mean sum metric

    
    # Calculate means and standard deviations
    return {
        'mean_error_count': round(np.mean(err_count_arr) / n, 6),
        'mean_update_count': round(np.mean(nSV_arr), 6),
        'mean_time': round(np.mean(time_arr), 6),
        'mean_nb_SV': np.mean(nb_SV_cum_arr, axis=1),
        'mean_ticks': np.mean(time_cum_arr, axis=1),
        'captured_t': captured_t,
        'mean_mistakes': np.mean(mistakes_arr, axis=1),
        'std_mistakes': np.std(mistakes_arr, axis=1),
        'mean_accuracies': np.mean(accuracies_arr, axis=1),
        'std_accuracies': np.std(accuracies_arr, axis=1),
        'mean_sensitivities': np.mean(sensitivities_arr, axis=1),
        'std_sensitivities': np.std(sensitivities_arr, axis=1),
        'mean_specificities': np.mean(specificities_arr, axis=1),
        'std_specificities': np.std(specificities_arr, axis=1),
        'mean_precisions': np.mean(precisions_arr, axis=1),
        'std_precisions': np.std(precisions_arr, axis=1),
        'mean_gmeans': np.mean(gmeans_arr, axis=1),
        'std_gmeans': np.std(gmeans_arr, axis=1),
        'mean_kappas': np.mean(kappas_arr, axis=1),
        'std_kappas': np.std(kappas_arr, axis=1),
        'mean_mccs': np.mean(mccs_arr, axis=1),
        'std_mccs': np.std(mccs_arr, axis=1),
        'mean_cumulative_errors': np.mean(cumulative_errors_arr, axis=1),
        'std_cumulative_errors': np.std(cumulative_errors_arr, axis=1),
        'mean_sum': np.mean(sum_arr, axis=1),
        'std_sum': np.std(sum_arr, axis=1)
    }
    
    
def save_hyperparameters(algorithm_name, best_hyperparameters, dataset_name):
    # Save the best hyperparameters to a CSV file
    best_hyperparameters['Algorithm'] = algorithm_name

    # base_name = os.path.splitext(dataset_name)[0].split('\\')[-1]  # Use forward slashes 
    # Extract base_name from the last part of the dataset_name (file name only, without path)
    base_name = os.path.splitext(os.path.basename(dataset_name))[0]  # Get only the file name without directories 
    

    # main_path = os.path.join("best_hyperparameters", os.path.dirname(dataset_name).replace("data", "").strip("\\")) 

    # Correctly handle the relative path by removing only "data/" and leaving the rest
    middle_part = os.path.dirname(dataset_name).replace("data/", "")
    # main_path = os.path.join("best_hyperparameters", middle_part)
    main_path = os.path.join("/ourdisk/hpc/disc/obinopaul/auto_archive_notyet/tape_2copies/best_hyperparameters", middle_part)

    # Create the directory if it does not exist, and suppress error if it already exists
    os.makedirs(main_path, exist_ok=True)

    # Construct full path for the CSV file
    csv_file_path = os.path.join(main_path, f"{base_name}.csv")

    # Check if CSV file already exists
    if os.path.exists(csv_file_path):
        df = pd.read_csv(csv_file_path)
    else:
        df = pd.DataFrame(columns=best_hyperparameters.keys())

    # Save the updated DataFrame and confirm the operation
    df = pd.concat([df, pd.DataFrame([best_hyperparameters])], ignore_index=True)
    df.to_csv(csv_file_path, index=False)



def save_model_parameters(model_parameters, dataset_name, run_number):


    # Extract the base name of the dataset (without path or extension)
    base_name = os.path.splitext(os.path.basename(dataset_name))[0]
    
    # Handle the relative path for the parameters
    middle_part = os.path.dirname(dataset_name).replace("data/", "")
    main_path = os.path.join("model_parameters", middle_part)

    # Create the directory if it does not exist
    os.makedirs(main_path, exist_ok=True)

    # Construct full path for the CSV file
    csv_file_path = os.path.join(main_path, f"{base_name}.csv")

    # Check if the file exists
    if os.path.exists(csv_file_path):
        # Load existing data
        df = pd.read_csv(csv_file_path)
    else:
        # Create a fresh DataFrame with the appropriate columns
        df = pd.DataFrame(columns=model_parameters.keys())

    # Append the new data to the DataFrame
    df = pd.concat([df, pd.DataFrame([model_parameters])], ignore_index=True)

    # Save the updated DataFrame to the CSV file
    df.to_csv(csv_file_path, index=False) 
        

if __name__ == '__main__':
    task_type, algorithm_name, dataset_name, file_format, n, _ = handle_parameters()
    run(task_type, algorithm_name, dataset_name, file_format, nb_runs=n)
