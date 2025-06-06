import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Define abtypes and their names
abtypes = {
    1: 'Uptrend',
    2: 'Downtrend',
    3: 'Upshift',
    4: 'Downshift',
    5: 'Systematic',
    6: 'Cyclic',
    7: 'Stratification'
}

# Define binary classification algorithms
algorithms = [
    'OGD','OGD_1', 'OGD_2', 'PA', 'PA1', 'PA2', 'PA1_Csplit', 'PA2_Csplit', 'PA1_L1', 'PA2_L1', 'PA1_L2', 'PA2_L2', 
    'PA_L1', 'PA_L2', 'PA_I_L1', 'PA_I_L2', 'PA_II_L1','PA_II_L2']

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
    
    'PA_I_L1': 'PA-I-L1',
    'PA_I_L2': 'PA-I-L2',
    'PA_II_L1': 'PA-II-L1',
    'PA_II_L2': 'PA-II-L2'
}

# Define performance metrics
performance_metrics = [
    'Captured Time', 'Mistakes', 'Updates', 'Time', 'Accuracies', 'Sensitivities',
    'Specificities', 'Precisions', 'G-Mean', 'Kappas', 'MCCs', 'Cumulative Errors'
]

# Define metrics that should use vmin=0 and vmax=1
bounded_metrics = {'Accuracies', 'Sensitivities', 'Specificities', 'Precisions', 'G-Mean', 'Sum'}


def read_performance_results(result_dir, abtype):
    abtype_dir = f'abtype{abtype}'
    category_dir = os.path.join(result_dir, abtype_dir)

    all_performance_results = pd.DataFrame()

    # Read and concatenate all performance result CSV files in the directory
    for file in os.listdir(category_dir):
        if file.endswith('.csv'):
            file_path = os.path.join(category_dir, file)
            df = pd.read_csv(file_path)
            all_performance_results = pd.concat([all_performance_results, df], ignore_index=True)

    # Filter for rows where "Captured Time" is 871
    filtered_performance_results = all_performance_results[all_performance_results['Captured Time'] == 1000]
    
    return filtered_performance_results

def read_metadata(base_dir, abtype):
    metadata = pd.DataFrame()
    metadata_file = os.path.join(base_dir, f'abtype{abtype}', 'datasets_metadata.csv')
    if os.path.exists(metadata_file):
        metadata_df = pd.read_csv(metadata_file)
        metadata = pd.concat([metadata, metadata_df], ignore_index=True)
    
    metadata['filename'] = metadata['filename'].str.replace('.libsvm', '', regex=False)
    
    return metadata

def merge_data(filtered_performance_results, metadata):
    return pd.merge(filtered_performance_results, metadata, left_on='Dataset', right_on='filename')

def plot_data(merged_data, algorithm, performance_metric, abtype_name, save_dir):
    algo_data = merged_data[merged_data['Algorithm'] == algorithm]
    
    pivot_data = algo_data.pivot_table(index='w', columns='t', values=performance_metric)  # Use the specified performance metric

    # Reorder the pivot data index to have 'w' in descending order
    pivot_data = pivot_data.sort_index(ascending=False)
    
    plt.figure(figsize=(10, 8))
    
    # Set vmin and vmax based on the performance metric
    if performance_metric in bounded_metrics:
        vmin = 0
        vmax = 1
    else:
        vmin = None
        vmax = None

    # Rename the algorithm in the legend using the map
    legend_name = algorithm_name_map.get(algorithm, algorithm)
    
    # sns.heatmap(pivot_data, annot=False, cmap='binary_r', cbar_kws={'label': performance_metric}, linewidths=0, yticklabels=pivot_data.index[::-1], vmin=vmin, vmax=vmax)
    sns.heatmap(pivot_data, annot=False, cmap='binary_r', cbar_kws={'label': performance_metric}, linewidths=0, yticklabels=True, vmin=vmin, vmax=vmax)
    # Access the color bar and set the label font size
    cbar = plt.gcf().axes[-1]  # Access the color bar
    cbar.set_ylabel(performance_metric, fontsize=20)  # Increase the color bar label size
    cbar.tick_params(labelsize=20)  # Set tick label size for color bar
    
    # plt.title(f'Window Length (w) vs Abnormal Parameter Value (t) for {legend_name} - {performance_metric}\n{abtype_name}')
    # plt.title(f'Window Length (i) vs Abnormal Parameter Value (t) for {legend_name}', fontsize=20)
    plt.xlabel('Abnormal Parameter', fontsize=20)
    plt.ylabel('Window Length', fontsize=20)

       # Set the ticks to reduce their number
    x_ticks = np.arange(0, len(pivot_data.columns), max(1, len(pivot_data.columns)//5))
    y_ticks = np.arange(0, len(pivot_data.index), max(1, len(pivot_data.index)//5))

    plt.xticks(ticks=x_ticks, labels=pivot_data.columns[x_ticks], fontsize=20)
    plt.yticks(ticks=y_ticks, labels=pivot_data.index[y_ticks], fontsize=20)
    
    
    # Save the plot
    plot_filename = os.path.join(save_dir, f'{performance_metric}.png')
    plt.savefig(plot_filename)
    plt.close()




def plot_performance(base_dir, result_dir):
    for abtype, abtype_name in abtypes.items():
        for algorithm in algorithms:
            for performance_metric in performance_metrics:
                filtered_performance_results = read_performance_results(result_dir, abtype)
                metadata = read_metadata(base_dir, abtype)
                merged_data = merge_data(filtered_performance_results, metadata)

                # Create the directory to save plots if it does not exist
                save_dir = os.path.join('plots_HeatMap', f'abtype{abtype}', algorithm)
                os.makedirs(save_dir, exist_ok=True)

                plot_data(merged_data, algorithm, performance_metric, abtype_name, save_dir)

if __name__ == "__main__":
    base_dir = os.path.join(r'/home/obinopaul/LIBOL-python_CS_2', 'data')
    result_dir = os.path.join(r'/home/obinopaul/LIBOL-python_CS_2', 'results') 

    plot_performance(base_dir, result_dir)
