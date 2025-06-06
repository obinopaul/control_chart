import os
import pandas as pd
import matplotlib.pyplot as plt

# Define abtypes and their names
abtypes = range(1, 8)  # Abtypes from 1 to 7

# Define performance metrics
performance_metrics = [
    'Captured Time', 'Mistakes', 'Updates', 'Time', 'Accuracies', 'Sensitivities',
    'Specificities', 'Precisions', 'G-Means', 'Kappas', 'MCCs', 'Cumulative Errors'
]

# Define function to list all dataset files
def list_datasets(result_dir, abtype):
    datasets = []
    abtype_dir = os.path.join(result_dir, f'abtype{abtype}')
    if os.path.exists(abtype_dir):
        for file in os.listdir(abtype_dir):
            if file.endswith('.csv'):
                datasets.append(file.replace('.csv', ''))
    return datasets

def plot_performance_metric(abtype, dataset_name, metric):
    # Construct the file path
    file_path = os.path.join('results', f'abtype{abtype}', f'{dataset_name}.csv')
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: The file {file_path} does not exist.")
        return

    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Check if the metric exists in the dataframe
    if metric not in df.columns:
        print(f"Error: The metric {metric} does not exist in the dataset.")
        return
    
    # Extract unique algorithms
    algorithms = df['Algorithm'].unique()
    
    # Create a plot for the specified metric
    plt.figure(figsize=(10, 6))
    
    for algorithm in algorithms:
        subset = df[df['Algorithm'] == algorithm]
        plt.plot(subset['Captured Time'], subset[metric], marker='o', label=algorithm)
    
    plt.title(f'{metric} vs Number of Samples ({dataset_name})')
    plt.xlabel('Number of Samples')
    plt.ylabel(metric)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # Move legend outside plot
    plt.legend()
    plt.grid(True)

    # Save the plot
    save_dir = os.path.join('plots_metrics', f'abtype{abtype}', metric)
    os.makedirs(save_dir, exist_ok=True)
    plot_filename = os.path.join(save_dir, f'{dataset_name}.png')
    plt.savefig(plot_filename, bbox_inches='tight')
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
