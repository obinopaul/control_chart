import matplotlib.pyplot as plt
from math import floor
import os 
import pandas as pd 

print_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
markers = ['o', 'x']

def plot(algorithms, run_stats, dataset_name, save_path):
    
    # Extracting values from run_stats dictionaries
    mistakes = [x['mean_mistakes'] for x in run_stats]  # Taking the final value
    updates = [x['mean_nb_SV'] for x in run_stats]  # Taking the final value
    time = [x['mean_ticks'] for x in run_stats]     # Taking the final value
    captured_t = [x['captured_t'] for x in run_stats]   
    accuracies = [x['mean_accuracies'] for x in run_stats]      # Taking the final value
    sensitivities = [x['mean_sensitivities'] for x in run_stats]        # Taking the final value
    specificities = [x['mean_specificities'] for x in run_stats]    # Taking the final value
    precisions = [x['mean_precisions'] for x in run_stats]  # Taking the final value
    gmeans = [x['mean_gmeans'] for x in run_stats]  # Taking the final value
    kappas = [x['mean_kappas'] for x in run_stats]  # Taking the final value
    mccs = [x['mean_mccs'] for x in run_stats]  # Taking the final value
    cumulative_errors = [x['mean_cumulative_errors'] for x in run_stats]    # Taking the final value
    sums = [x['mean_sum'] for x in run_stats]  # Extract the mean sum metric

    # Saving the metrics to CSV
    save_metrics_to_csv(captured_t, mistakes, updates, time, accuracies, sensitivities, specificities, precisions, gmeans, kappas, mccs, cumulative_errors, sums, algorithms, dataset_name, save_path)

# Function to save the metrics to a CSV file
def save_metrics_to_csv(captured_t, mistakes, updates, time, accuracies, sensitivities, specificities, precisions, gmeans, kappas, mccs, cumulative_errors, sums, algorithms, dataset_name, save_path):
    # data_name_use = os.path.splitext(dataset_name)[0].split('\\')[-1]

    
    # Extract base name without extension
    base_name = os.path.splitext(os.path.basename(dataset_name))[0]
    main_path = save_path

    data = {
        'Algorithm': [],
        'Captured Time': [],
        'Mistakes': [],
        'Updates': [],
        'Time': [],
        'Accuracies': [],
        'Sensitivities': [],
        'Specificities': [],
        'Precisions': [],
        'G-Mean': [],
        'Kappas': [],
        'MCCs': [],
        'Cumulative Errors': [],
        'Sum': [],  # Add sum to the data dictionary
        'Dataset': []
    }
    
    for i, algorithm in enumerate(algorithms):
    
        min_length = min([len(metrics) for metrics in [
            mistakes[i], updates[i], time[i], accuracies[i], sensitivities[i],
            specificities[i], precisions[i], gmeans[i], kappas[i], mccs[i], cumulative_errors[i], sums[i]
        ]])

    
        for j in range(min_length):
            data['Algorithm'].append(algorithm)
            data['Captured Time'].append(captured_t[i][j])
            data['Mistakes'].append(mistakes[i][j])
            data['Updates'].append(updates[i][j])
            data['Time'].append(time[i][j])
            data['Accuracies'].append(accuracies[i][j])
            data['Sensitivities'].append(sensitivities[i][j])
            data['Specificities'].append(specificities[i][j])
            data['Precisions'].append(precisions[i][j])
            data['G-Mean'].append(gmeans[i][j])
            data['Kappas'].append(kappas[i][j])
            data['MCCs'].append(mccs[i][j])
            data['Cumulative Errors'].append(cumulative_errors[i][j])
            data['Sum'].append(sums[i][j])  # Add sum to the CSV output
            data['Dataset'].append(base_name)
    
    df = pd.DataFrame(data)
    if not os.path.exists(main_path):
        os.makedirs(main_path)

    df.to_csv(f"{main_path}/{base_name}.csv", index=False)