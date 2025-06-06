import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def read_performance_results(result_dir, abtype, category):
    abtype_dir = f'abtype_{abtype}'
    category_dir = os.path.join(result_dir, abtype_dir, category)

    all_performance_results = pd.DataFrame()

    # Read and concatenate all performance result CSV files in the directory
    for file in os.listdir(category_dir):
        if file.endswith('.csv'):
            file_path = os.path.join(category_dir, file)
            df = pd.read_csv(file_path)
            all_performance_results = pd.concat([all_performance_results, df], ignore_index=True)

    # Filter for rows where "Captured Time" is 871
    filtered_performance_results = all_performance_results[all_performance_results['Captured Time'] == 871]
    
    return filtered_performance_results

def read_metadata(base_dir, abtype, category):
    metadata = pd.DataFrame()
    metadata_file = os.path.join(base_dir, f'abtype_{abtype}', category, 'datasets_metadata.csv')
    if os.path.exists(metadata_file):
        metadata_df = pd.read_csv(metadata_file)
        metadata = pd.concat([metadata, metadata_df], ignore_index=True)
    
    metadata['filename'] = metadata['filename'].str.replace('.libsvm', '', regex=False)
    
    return metadata

def merge_data(filtered_performance_results, metadata):
    return pd.merge(filtered_performance_results, metadata, left_on='Dataset', right_on='filename')

def plot_data(merged_data, algorithm, performance_metric, category, abtype_name):
    algo_data = merged_data[merged_data['Algorithm'] == algorithm]
    pivot_data = algo_data.pivot_table(index='w', columns='t', values=performance_metric)  # Use the specified performance metric

    # # Ensure all window lengths from 10 to 100 are represented
    # pivot_data = pivot_data.reindex(index=range(10, 101, 10), columns=sorted(pivot_data.columns), fill_value=0)

    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_data, annot=False, cmap='binary_r', cbar_kws={'label': performance_metric}, linewidths=0, yticklabels=pivot_data.index[::-1])
    plt.title(f'Window Length (w) vs Abnormal Parameter Value (t) for {algorithm} ({category}) - {performance_metric}\n{abtype_name}')
    plt.xlabel('Abnormal Parameter Value (t)')
    plt.ylabel('Window Length (w)')
    plt.show()

def plot_performance(base_dir, result_dir, abtype, algorithm, performance_metric):

    abtype_names = {
        1: 'Uptrend',
        2: 'Downtrend',
        3: 'Upshift',
        4: 'Downshift',
        5: 'Systematic',
        6: 'Cyclic',
        7: 'Stratification'
    }
    
    abtype_name = abtype_names.get(abtype, f'Unknown Abtype {abtype}')

    for category in ['base', 'cost_sensitive']:
        filtered_performance_results = read_performance_results(result_dir, abtype, category)
        metadata = read_metadata(base_dir, abtype, category)
        merged_data = merge_data(filtered_performance_results, metadata)
        plot_data(merged_data, algorithm, performance_metric, category, abtype_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot performance metrics for specified abtype and algorithm')
    parser.add_argument('--abtype', type=int, required=True, help='Specify the abtype you want to plot (1-8)')
    parser.add_argument('--algorithm', type=str, required=True, help='Specify the algorithm you want to plot')
    parser.add_argument('--performance_metric', type=str, required=True, help='Specify the performance metric you want to plot')

    args = parser.parse_args()

    base_dir = 'data'
    result_dir = 'results'

    plot_performance(base_dir, result_dir, args.abtype, args.algorithm, args.performance_metric)
