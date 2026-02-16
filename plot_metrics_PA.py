import os
import pandas as pd
import matplotlib.pyplot as plt

# Define abtypes and their names
abtypes = range(1, 8)  # Abtypes from 1 to 7

# Define performance metrics
performance_metrics = [
    'Captured Time', 'Mistakes', 'Updates', 'Time', 'Accuracies', 'Sensitivities',
    'Specificities', 'Precisions', 'G-Mean', 'Kappas', 'MCCs', 'Cumulative Errors'
]

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

# Define marker styles for PA, CSPA, and OGD groups
algorithm_marker_map = {
    'PA': 'x',
    'CSPA': 'o',
    'CSPA_I_II': '+',  # Marker for CSPA-I and CSPA-II
    'OGD': 's',

    # New Algorithms without C+ and C- 
    'PA_L1': {'color': 'orange', 'marker': 'd'},  # Orange and diamond for 'PA-L1'
    'PA_L2': {'color': 'orange', 'marker': '^'},  # Orange and upward triangle for 'PA-L2'
    
    'PA_I_L1': {'color': 'purple', 'marker': 'v'}, # Purple and inverted triangle for 'PA-I-L1'
    'PA_I_L2': {'color': 'purple', 'marker': '<'}, # Purple and left triangle for 'PA-I-L2'
    
    'PA_II_L1': {'color': 'brown', 'marker': '>'}, # Brown and right triangle for 'PA-II-L1'
    'PA_II_L2': {'color': 'brown', 'marker': '*'}  # Brown and star for 'PA-II-L2'
}

# Function to get marker based on algorithm type
def get_marker(algorithm):
    if algorithm in ['PA1_Csplit', 'PA2_Csplit']:
        return algorithm_marker_map['CSPA_I_II']
    elif 'PA1_L' in algorithm or 'PA2_L' in algorithm:  # Other CSPA algorithms
        return algorithm_marker_map['CSPA']
    elif 'OGD' in algorithm:
        return algorithm_marker_map['OGD']
    # New algorithms without C+ and C-
    elif algorithm == 'PA_L1':
        return algorithm_marker_map['PA_L1']['marker']
    elif algorithm == 'PA_L2':
        return algorithm_marker_map['PA_L2']['marker']
    elif algorithm == 'PA_I_L1':
        return algorithm_marker_map['PA_I_L1']['marker']
    elif algorithm == 'PA_I_L2':
        return algorithm_marker_map['PA_I_L2']['marker']
    elif algorithm == 'PA_II_L1':
        return algorithm_marker_map['PA_II_L1']['marker']
    elif algorithm == 'PA_II_L2':
        return algorithm_marker_map['PA_II_L2']['marker']
    else:
        return algorithm_marker_map['PA']

# Define function to list all dataset files
def list_datasets(result_dir, abtype):
    datasets = []
    abtype_dir = os.path.join(result_dir, f'abtype{abtype}')
    if os.path.exists(abtype_dir):
        for file in os.listdir(abtype_dir):
            if file.endswith('.csv'):
                datasets.append(file.replace('.csv', ''))
    else:
        print(f"Directory not found: {abtype_dir}")  # Directory does not exist
    return datasets


def plot_performance_metric(abtype, dataset_name, metric):
    # Construct the file path
    file_path = os.path.join('results', f'abtype{abtype}', f'{dataset_name}.csv')
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: The file {file_path} does not exist.")
        return

    # Read the CSV file
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return
    
    # Check if the metric exists in the dataframe
    if metric not in df.columns:
        print(f"Error: The metric {metric} does not exist in the dataset {file_path}.")
        return
    
    # Extract unique algorithms
    algorithms = df['Algorithm'].unique()
  
    # Create a plot for the specified metric
    plt.figure(figsize=(10, 6))
    
    # Store legend entries by group
    legend_entries = {'PA': [], 'CSPA': [], 'OGD': []}
    
    for algorithm in algorithms:
        if algorithm in algorithms_to_plot:
            subset = df[df['Algorithm'] == algorithm]
            # Rename the algorithm in the legend using the map
            legend_name = algorithm_name_map.get(algorithm, algorithm)
            marker_style = get_marker(algorithm)  # Get the appropriate marker
            
            # Ensure there is data to plot
            if subset.empty:
                print(f"No data for algorithm {algorithm} in dataset {dataset_name} for abtype{abtype}.")
                continue
            
            # Plot data
            line, = plt.plot(subset['Captured Time'], subset[metric], marker=marker_style, label=legend_name)
            
            # Store legend handles by type
            if 'PA' in legend_name and 'CSPA' not in legend_name:
                legend_entries['PA'].append(line)
            elif 'CSPA' in legend_name:
                legend_entries['CSPA'].append(line)
            else:
                legend_entries['OGD'].append(line)
    
    # Ensure there are entries to plot
    if not any(legend_entries.values()):
        print(f"No valid data to plot for abtype{abtype}, dataset {dataset_name}, metric {metric}.")
        return

    # Reorder legend as per user preference
    handles = legend_entries['PA'] + legend_entries['CSPA'] + legend_entries['OGD']
    labels = [handle.get_label() for handle in handles]
    
    plt.xlabel('Number of Samples', fontsize=20)
    plt.ylabel(metric, fontsize=20)
    
    plt.xticks(fontsize=20)  # Increase tick size
    plt.yticks(fontsize=20)  # Increase tick size
    # Set maximum of 7 ticks for the y-axis
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(7))
    
    plt.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)  # Move legend outside plot
    plt.grid(False)

    # Save the plot
    save_dir = os.path.join('plot_metrics_PA', f'abtype{abtype}', metric)
    os.makedirs(save_dir, exist_ok=True)
    plot_filename = os.path.join(save_dir, f'{dataset_name}.png')
    plt.savefig(plot_filename, bbox_inches='tight')  # Save with legend outside
    plt.close()

def plot_all_metrics():
    result_dir = 'results'

    for abtype in abtypes:
        datasets = list_datasets(result_dir, abtype)
        for dataset_name in datasets:
            for metric in performance_metrics:
                plot_performance_metric(abtype, dataset_name, metric)

if __name__ == "__main__":
    plot_all_metrics()

