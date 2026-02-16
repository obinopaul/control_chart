# import matplotlib.pyplot as plt
# from math import floor
# import os 
# import pandas as pd 

# print_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
# markers = ['o', 'x']

# def plot(algorithms, run_stats, dataset_name, save_path):
    
#     # Extracting values from run_stats dictionaries
#     mistakes = [x['mean_mistakes'] for x in run_stats]  # Taking the final value
#     updates = [x['mean_nb_SV'] for x in run_stats]  # Taking the final value
#     time = [x['mean_ticks'] for x in run_stats]     # Taking the final value
#     captured_t = [x['captured_t'] for x in run_stats]   
#     accuracies = [x['mean_accuracies'] for x in run_stats]      # Taking the final value
#     sensitivities = [x['mean_sensitivities'] for x in run_stats]        # Taking the final value
#     specificities = [x['mean_specificities'] for x in run_stats]    # Taking the final value
#     precisions = [x['mean_precisions'] for x in run_stats]  # Taking the final value
#     gmeans = [x['mean_gmeans'] for x in run_stats]  # Taking the final value
#     kappas = [x['mean_kappas'] for x in run_stats]  # Taking the final value
#     mccs = [x['mean_mccs'] for x in run_stats]  # Taking the final value
#     cumulative_errors = [x['mean_cumulative_errors'] for x in run_stats]    # Taking the final value
#     sums = [x['mean_sum'] for x in run_stats]  # Extract the mean sum metric

#     # Saving the metrics to CSV
#     save_metrics_to_csv(captured_t, mistakes, updates, time, accuracies, sensitivities, specificities, precisions, gmeans, kappas, mccs, cumulative_errors, sums, algorithms, dataset_name, save_path)

# # Function to save the metrics to a CSV file
# def save_metrics_to_csv(captured_t, mistakes, updates, time, accuracies, sensitivities, specificities, precisions, gmeans, kappas, mccs, cumulative_errors, sums, algorithms, dataset_name, save_path):
#     # data_name_use = os.path.splitext(dataset_name)[0].split('\\')[-1]

    
#     # Extract base name without extension
#     base_name = os.path.splitext(os.path.basename(dataset_name))[0]
#     main_path = save_path

#     data = {
#         'Algorithm': [],
#         'Captured Time': [],
#         'Mistakes': [],
#         'Updates': [],
#         'Time': [],
#         'Accuracies': [],
#         'Sensitivities': [],
#         'Specificities': [],
#         'Precisions': [],
#         'G-Mean': [],
#         'Kappas': [],
#         'MCCs': [],
#         'Cumulative Errors': [],
#         'Sum': [],  # Add sum to the data dictionary
#         'Dataset': []
#     }
    
#     for i, algorithm in enumerate(algorithms):
    
#         min_length = min([len(metrics) for metrics in [
#             mistakes[i], updates[i], time[i], accuracies[i], sensitivities[i],
#             specificities[i], precisions[i], gmeans[i], kappas[i], mccs[i], cumulative_errors[i], sums[i]
#         ]])

    
#         for j in range(min_length):
#             data['Algorithm'].append(algorithm)
#             data['Captured Time'].append(captured_t[i][j])
#             data['Mistakes'].append(mistakes[i][j])
#             data['Updates'].append(updates[i][j])
#             data['Time'].append(time[i][j])
#             data['Accuracies'].append(accuracies[i][j])
#             data['Sensitivities'].append(sensitivities[i][j])
#             data['Specificities'].append(specificities[i][j])
#             data['Precisions'].append(precisions[i][j])
#             data['G-Mean'].append(gmeans[i][j])
#             data['Kappas'].append(kappas[i][j])
#             data['MCCs'].append(mccs[i][j])
#             data['Cumulative Errors'].append(cumulative_errors[i][j])
#             data['Sum'].append(sums[i][j])  # Add sum to the CSV output
#             data['Dataset'].append(base_name)
    
#     df = pd.DataFrame(data)
#     if not os.path.exists(main_path):
#         os.makedirs(main_path)

#     df.to_csv(f"{main_path}/{base_name}.csv", index=False)






import matplotlib.pyplot as plt
from math import floor
import os
import pandas as pd

print_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
markers = ['o', 'x']

def plot(algorithms, run_stats, dataset_name, save_path):
    # Extracting means and standard deviations from run_stats
    mistakes = [x['mean_mistakes'] for x in run_stats]
    std_mistakes = [x['std_mistakes'] for x in run_stats]

    updates = [x['mean_nb_SV'] for x in run_stats]
    # std_updates = [x['std_nb_SV'] for x in run_stats]

    time = [x['mean_ticks'] for x in run_stats]
    # std_time = [x['std_ticks'] for x in run_stats]

    captured_t = [x['captured_t'] for x in run_stats]

    accuracies = [x['mean_accuracies'] for x in run_stats]
    std_accuracies = [x['std_accuracies'] for x in run_stats]

    sensitivities = [x['mean_sensitivities'] for x in run_stats]
    std_sensitivities = [x['std_sensitivities'] for x in run_stats]

    specificities = [x['mean_specificities'] for x in run_stats]
    std_specificities = [x['std_specificities'] for x in run_stats]

    precisions = [x['mean_precisions'] for x in run_stats]
    std_precisions = [x['std_precisions'] for x in run_stats]

    gmeans = [x['mean_gmeans'] for x in run_stats]
    std_gmeans = [x['std_gmeans'] for x in run_stats]

    kappas = [x['mean_kappas'] for x in run_stats]
    std_kappas = [x['std_kappas'] for x in run_stats]

    mccs = [x['mean_mccs'] for x in run_stats]
    std_mccs = [x['std_mccs'] for x in run_stats]

    cumulative_errors = [x['mean_cumulative_errors'] for x in run_stats]
    std_cumulative_errors = [x['std_cumulative_errors'] for x in run_stats]

    sums = [x['mean_sum'] for x in run_stats]
    std_sums = [x['std_sum'] for x in run_stats]

    # Save the metrics to CSV
    save_metrics_to_csv(
        captured_t, mistakes, std_mistakes, updates, time,
        accuracies, std_accuracies, sensitivities, std_sensitivities,
        specificities, std_specificities, precisions, std_precisions,
        gmeans, std_gmeans, kappas, std_kappas, mccs, std_mccs,
        cumulative_errors, std_cumulative_errors, sums, std_sums,
        algorithms, dataset_name, save_path
    )


def get_base_path(dataset_name):
    """
    Extract the path before the 'data' folder from the given dataset path.

    Parameters:
    - dataset_name (str): Full path to the dataset.

    Returns:
    - str: Base path before the 'data' folder.
    """
    # Normalize path separators for compatibility
    normalized_path = os.path.normpath(dataset_name)
    
    # Split the path into parts
    parts = normalized_path.split(os.sep)
    
    # Find the index of 'data' and join everything before it
    if 'data' in parts:
        data_index = parts.index('data')
        base_path = os.sep.join(parts[:data_index])  # Join parts before 'data'
        return base_path
    else:
        # 'data' not found in path
        raise ValueError("'data' folder not found in the dataset path.")
        
        
# Function to save metrics and their standard deviations to a CSV file
def save_metrics_to_csv(
    captured_t, mistakes, std_mistakes, updates, time,
    accuracies, std_accuracies, sensitivities, std_sensitivities,
    specificities, std_specificities, precisions, std_precisions,
    gmeans, std_gmeans, kappas, std_kappas, mccs, std_mccs,
    cumulative_errors, std_cumulative_errors, sums, std_sums,
    algorithms, dataset_name, save_path
):
    base_name = os.path.splitext(os.path.basename(dataset_name))[0]
    main_path = save_path

    data = {
        'Algorithm': [],
        'Captured Time': [],
        'Mistakes': [],
        'Mistakes (Std)': [],
        'Updates': [],
        # 'Updates (Std)': [],
        'Time': [],
        # 'Time (Std)': [],
        'Accuracies': [],
        'Accuracies (Std)': [],
        'Sensitivities': [],
        'Sensitivities (Std)': [],
        'Specificities': [],
        'Specificities (Std)': [],
        'Precisions': [],
        'Precisions (Std)': [],
        'G-Mean': [],
        'G-Mean (Std)': [],
        'Kappas': [],
        'Kappas (Std)': [],
        'MCCs': [],
        'MCCs (Std)': [],
        'Cumulative Errors': [],
        'Cumulative Errors (Std)': [],
        'Sum': [],
        'Sum (Std)': [],
        'Dataset': []
    }

    for i, algorithm in enumerate(algorithms):
        min_length = min([len(metrics) for metrics in [
            mistakes[i], updates[i], time[i], accuracies[i], sensitivities[i],
            specificities[i], precisions[i], gmeans[i], kappas[i], mccs[i],
            cumulative_errors[i], sums[i]
        ]])

        for j in range(min_length):
            data['Algorithm'].append(algorithm)
            data['Captured Time'].append(captured_t[i][j])
            data['Mistakes'].append(mistakes[i][j])
            data['Mistakes (Std)'].append(std_mistakes[i][j])
            data['Updates'].append(updates[i][j])
            # data['Updates (Std)'].append(std_updates[i][j])
            data['Time'].append(time[i][j])
            # data['Time (Std)'].append(std_time[i][j])
            data['Accuracies'].append(accuracies[i][j])
            data['Accuracies (Std)'].append(std_accuracies[i][j])
            data['Sensitivities'].append(sensitivities[i][j])
            data['Sensitivities (Std)'].append(std_sensitivities[i][j])
            data['Specificities'].append(specificities[i][j])
            data['Specificities (Std)'].append(std_specificities[i][j])
            data['Precisions'].append(precisions[i][j])
            data['Precisions (Std)'].append(std_precisions[i][j])
            data['G-Mean'].append(gmeans[i][j])
            data['G-Mean (Std)'].append(std_gmeans[i][j])
            data['Kappas'].append(kappas[i][j])
            data['Kappas (Std)'].append(std_kappas[i][j])
            data['MCCs'].append(mccs[i][j])
            data['MCCs (Std)'].append(std_mccs[i][j])
            data['Cumulative Errors'].append(cumulative_errors[i][j])
            data['Cumulative Errors (Std)'].append(std_cumulative_errors[i][j])
            data['Sum'].append(sums[i][j])
            data['Sum (Std)'].append(std_sums[i][j])
            data['Dataset'].append(base_name)

    df = pd.DataFrame(data)
    if not os.path.exists(main_path):
        os.makedirs(main_path)
        #os.makedirs(os.path.join(get_base_path(dataset_name), main_path))

    df.to_csv(f"{main_path}/{base_name}.csv", index=False)

