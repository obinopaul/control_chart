import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Algorithms to plot
algorithms_to_plot = ['PA', 'PA1', 'PA2', 'OGD', 'OGD_1', 'OGD_2', 'PA1_Csplit', 'PA2_Csplit', 'PA1_L1', 'PA2_L1', 'PA1_L2', 'PA2_L2',
                      'PA_L1', 'PA_L2'] # 'PA_I_L1', 'PA_I_L2', 'PA_II_L1', 'PA_II_L2']

# Create a mapping for renaming algorithms in the legend
algorithm_name_map = {
    'PA1_Csplit': 'CSPA-I',
    'PA2_Csplit': 'CSPA-II',
    'OGD_1': 'CSOGD-I',
    'OGD_2': 'CSOGD-II',
    'PA1': 'PA-I',
    'PA2': 'PA-II',
    'PA1_L1': 'CSPA-I-L1',
    'PA2_L1': 'CSPA-II-L1',
    'PA1_L2': 'CSPA-I-L2',
    'PA2_L2': 'CSPA-II-L2',

    # New algorithms without C+ and C- 
    'PA_L1': 'CSPA-L1',
    'PA_L2': 'CSPA-L2',
    
    #'PA_I_L1': 'PA-I-L1',
    #'PA_I_L2': 'PA-I-L2',
    #'PA_II_L1': 'PA-II-L1',
    #'PA_II_L2': 'PA-II-L2'
}

# Define window lengths to plot
window_lengths_to_plot = [10, 30, 50, 70, 100]

# Define function to extract t values from filenames for each pattern and window length
def extract_t_values_for_pattern(result_dir, abtype):
    t_values_by_w = {}
    
    abtype_dir = os.path.join(result_dir, f'abtype{abtype}')
    if os.path.exists(abtype_dir):
        for file in os.listdir(abtype_dir):
            if file.endswith('.csv'):
                try:
                    # Extract window length (w) and t value from filename (e.g., 'abtype1_w100_t1.2.csv')
                    parts = file.split('_')
                    w_value_str = parts[1][1:]  # Extract w value from 'w100'
                    t_value_str = parts[2][1:].replace('.csv', '')  # Extract t value from 't1.2'

                    w_value = int(w_value_str)
                    t_value = float(t_value_str)

                    # Only consider the window lengths we are interested in
                    if w_value in window_lengths_to_plot:
                        if w_value not in t_values_by_w:
                            t_values_by_w[w_value] = []
                        t_values_by_w[w_value].append(t_value)

                except ValueError:
                    continue

    # Remove duplicates and sort the t values for each window length
    for w_value in t_values_by_w:
        t_values_by_w[w_value] = sorted(set(t_values_by_w[w_value]))

    return t_values_by_w

# Plot the performance metric for a specific abtype, algorithm, and metric
def plot_performance_metric(abtype, algorithm, metric):
    # Extract t values for all window lengths for the current pattern (abtype)
    t_values_by_w = extract_t_values_for_pattern('results', abtype)
    
    plt.figure(figsize=(10, 6))
    
    for w, abnormal_params_to_plot in t_values_by_w.items():
        y_values = []
        for t in abnormal_params_to_plot:
            # Construct the file path
            file_path = os.path.join('results', f'abtype{abtype}', f'abtype{abtype}_w{w}_t{t}.csv')
            
            # Check if file exists
            if os.path.exists(file_path):
                # Read the CSV file
                df = pd.read_csv(file_path)
                
                # Filter rows where Captured Time is 1000 and Algorithm matches
                subset = df[(df['Captured Time'] == 1000) & (df['Algorithm'] == algorithm)]
                
                # Check if the metric exists and calculate the mean value for that metric
                if metric in df.columns and not subset.empty:
                    y_values.append(subset[metric].mean())
                else:
                    y_values.append(np.nan)  # Append NaN if data not found
            else:
                y_values.append(np.nan)  # Append NaN if file not found
        
        # Use the algorithm_name_map to rename the algorithm in the legend, if applicable
        legend_name = algorithm_name_map.get(algorithm, algorithm)  # Rename if in the map
        plt.plot(abnormal_params_to_plot, y_values, marker='o', label=f'{legend_name} (Window {w})')
    
    plt.xlabel('Abnormal Parameter (t)', fontsize=20)
    plt.ylabel(metric, fontsize=20)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)  # Move legend outside plot
    plt.grid(True)

    # Create save directory if it doesn't exist
    save_dir = os.path.join('plot_time', f'abtype{abtype}')  # Simplified save directory 
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the plot
    plot_filename = os.path.join(save_dir, f'abtype{abtype}_{algorithm}_{metric}.png')
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close()

# Function to plot all metrics for all algorithms
def plot_all_metrics_for_all_algorithms():
    metric = 'Time'  # Example of performance metric to plot, you can loop through other metrics too

    for abtype in range(1, 8):  # Loop over abtypes from 1 to 7
        for algorithm in algorithms_to_plot:
            plot_performance_metric(abtype, algorithm, metric)

if __name__ == "__main__":
    plot_all_metrics_for_all_algorithms()
