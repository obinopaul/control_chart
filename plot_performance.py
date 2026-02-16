# import os
# import sys
# import pandas as pd
# import matplotlib.pyplot as plt

# def plot_performance_metric(metric, dataset_name):
#     # Construct the file path
#     file_path = os.path.join('results', dataset_name, 'performance_results.csv')
    
#     # Check if file exists
#     if not os.path.exists(file_path):
#         print(f"Error: The file {file_path} does not exist.")
#         return

#     # Read the CSV file
#     df = pd.read_csv(file_path)
    
#     # Check if the metric exists in the dataframe
#     if metric not in df.columns:
#         print(f"Error: The metric {metric} does not exist in the dataset.")
#         return
    
#     # Extract unique algorithms
#     algorithms = df['Algorithm'].unique()
    
#     # Create a plot for the specified metric
#     plt.figure(figsize=(10, 6))
    
#     for algorithm in algorithms:
#         subset = df[df['Algorithm'] == algorithm]
#         plt.plot(subset['Captured Time'], subset[metric], marker='o', label=algorithm)
    
#     plt.title(f'{metric} vs Number of Samples ({dataset_name})')
#     plt.xlabel('Number of Samples')
#     plt.ylabel(metric)
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# if __name__ == "__main__":
#     if len(sys.argv) != 3:
#         print("Usage: python plot_performance.py <metric> <dataset_name>")
#         sys.exit(1)
    
#     metric = sys.argv[1]
#     dataset_name = sys.argv[2]
    
#     plot_performance_metric(metric, dataset_name)


import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

def plot_performance_metric(abtype, category, dataset_name, metric):
    # Construct the file path
    file_path = os.path.join('results', f'abtype_{abtype}', category, f'{dataset_name}.csv')
    
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
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python plot_performance.py <abtype> <category> <dataset_name> <metric>")
        sys.exit(1)
    
    abtype = sys.argv[1]
    category = sys.argv[2]
    dataset_name = sys.argv[3]
    metric = sys.argv[4]
    
    plot_performance_metric(abtype, category, dataset_name, metric)
