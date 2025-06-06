import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Define window lengths to plot
window_lengths_to_plot = [10, 30, 50, 70, 100]

# Algorithms to plot
algorithms_to_plot = ['PA', 'PA1', 'PA2', 'OGD', 'OGD_1', 'OGD_2', 
                      'PA1_Csplit', 'PA2_Csplit', 'PA1_L1', 'PA2_L1', 
                      'PA1_L2', 'PA2_L2', 'PA_L1', 'PA_L2']

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
    'PA_L1': 'CSPA-L1',
    'PA_L2': 'CSPA-L2',
}

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


# Define function to extract parameter values from filenames
def extract_t_values_for_abtype(result_dir, abtype):
    """Extract all t values for the given abtype."""
    t_values = set()

    abtype_dir = os.path.join(result_dir, f'abtype{abtype}')
    if os.path.exists(abtype_dir):
        for file in os.listdir(abtype_dir):
            if file.endswith('.csv'):
                try:
                    # Extract t value from filename (e.g., 'abtype1_w100_t1.2.csv')
                    t_value_str = file.split('_')[2][1:].replace('.csv', '')
                    t_value = float(t_value_str)
                    t_values.add(t_value)
                except (IndexError, ValueError):
                    continue

    return sorted(t_values)

def plot_time_vs_window_length(abtype, t_value):
    """Plot Time vs Window Length for all algorithms in one plot."""
    plt.figure(figsize=(10, 6))  # Initialize the figure

    # Iterate over each algorithm
    for algorithm in algorithms_to_plot:
        window_lengths = []
        time_values = []

        # Iterate over window lengths and read the corresponding data
        for w in window_lengths_to_plot:
            file_path = os.path.join('results', f'abtype{abtype}', f'abtype{abtype}_w{w}_t{t_value}.csv')

            if os.path.exists(file_path):
                # Read the CSV file
                df = pd.read_csv(file_path)

                # Filter for the specific algorithm
                df_algorithm = df[df['Algorithm'] == algorithm]

                # Check if Time is available
                if not df_algorithm.empty:
                    time = df_algorithm['Time'].values[0]
                    window_lengths.append(w)
                    time_values.append(time)

        # If data is available, plot it
        if window_lengths:
            # Rename the algorithm for the legend
            marker_style = get_marker(algorithm)  # Get the appropriate marker
            legend_name = algorithm_name_map.get(algorithm, algorithm)
            # plt.plot(window_lengths, time_values, marker=marker_style, label=legend_name)
            plt.plot(window_lengths, time_values, marker=marker_style)

    # Customize the plot
    plt.xlabel('Window Length', fontsize=20)
    plt.ylabel('Time', fontsize=20)
    
    # Format the y-axis to show 3 decimal places
    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.4f}'))
    
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(5, integer=True))
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(5, integer=True))
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)  # Legend outside plot
    plt.grid(False)

    # Save the plot
    save_dir = os.path.join('plot_time2', f'abtype{abtype}')
    os.makedirs(save_dir, exist_ok=True)
    plot_filename = os.path.join(save_dir, f'abtype{abtype}_t{t_value}_Time_vs_Window_Length.png')
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close()

def plot_all_time_vs_window_length():
    """Plot Time vs Window Length for all abtypes and t values."""
    result_dir = 'results'

    for abtype in range(1, 8):  # Loop over abtypes from 1 to 7
        t_values = extract_t_values_for_abtype(result_dir, abtype)

        for t_value in t_values:
            plot_time_vs_window_length(abtype, t_value)

if __name__ == "__main__":
    plot_all_time_vs_window_length()
