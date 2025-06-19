# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # Define window lengths to plot
# window_lengths_to_plot = [10, 30, 50, 70, 100]

# # Algorithms to plot
# algorithms_to_plot = ['PA', 'PA1', 'PA2', 'OGD', 'OGD_1', 'OGD_2', 'PA1_Csplit', 'PA2_Csplit', 
#                       'PA1_L1', 'PA2_L1', 'PA1_L2', 'PA2_L2', 'PA_L1', 'PA_L2',] 
#                       # 'PA_I_L1', 'PA_I_L2', 'PA_II_L1', 'PA_II_L2']

# # Create a mapping for renaming algorithms in the legend
# algorithm_name_map = {
#     'PA1_Csplit': 'CSPA-I',
#     'PA2_Csplit': 'CSPA-II',
#     'OGD_1': 'CSOGD-I',
#     'OGD_2': 'CSOGD-II',
#     'PA1': 'PA-I',
#     'PA2': 'PA-II',
#     'PA1_L1': 'CSPA-I-L1',
#     'PA2_L1': 'CSPA-II-L1',
#     'PA1_L2': 'CSPA-I-L2',
#     'PA2_L2': 'CSPA-II-L2',
#     'PA_L1': 'CSPA-L1',
#     'PA_L2': 'CSPA-L2',
    
#     #'PA_I_L1': 'PA-I-L1',
#     #'PA_I_L2': 'PA-I-L2',
#     #'PA_II_L1': 'PA-II-L1',
#     #'PA_II_L2': 'PA-II-L2'
# }

# # Define function to extract t values from filenames for each abtype and window length
# def extract_t_values_for_abtype(result_dir, abtype):
#     t_values_by_w = {}

#     abtype_dir = os.path.join(result_dir, f'abtype{abtype}')
#     if os.path.exists(abtype_dir):
#         for file in os.listdir(abtype_dir):
#             if file.endswith('.csv'):
#                 try:
#                     # Extract window length (w) and t value from filename (e.g., 'abtype1_w100_t1.2.csv')
#                     parts = file.split('_')
#                     w_value_str = parts[1][1:]  # Extract w value from 'w100'
#                     t_value_str = parts[2][1:].replace('.csv', '')  # Extract t value from 't1.2'

#                     w_value = int(w_value_str)
#                     t_value = float(t_value_str)

#                     # Only consider the window lengths we are interested in
#                     if w_value in window_lengths_to_plot:
#                         if w_value not in t_values_by_w:
#                             t_values_by_w[w_value] = []
#                         t_values_by_w[w_value].append(t_value)

#                 except ValueError:
#                     continue

#     # Remove duplicates and sort the t values for each window length
#     for w_value in t_values_by_w:
#         t_values_by_w[w_value] = sorted(set(t_values_by_w[w_value]))

#     return t_values_by_w


# def plot_performance_metric(abtype, algorithm, metric):
#     # Extract t values for all window lengths for the current pattern (abtype)
#     t_values_by_w = extract_t_values_for_abtype('results', abtype)
    
#     # Sort the window lengths to ensure plotting in increasing order
#     sorted_window_lengths = sorted(t_values_by_w.keys())
    
#     # Create a plot for the specified metric across window lengths and extracted t values
#     plt.figure(figsize=(10, 6))
    
#     for w in sorted_window_lengths:
#         t_values = t_values_by_w[w]
#         y_values = []
#         for t in t_values:
#             # Construct the file path
#             file_path = os.path.join('results', f'abtype{abtype}', f'abtype{abtype}_w{w}_t{t}.csv')
            
#             # Check if file exists
#             if os.path.exists(file_path):
#                 # Read the CSV file
#                 df = pd.read_csv(file_path)
                
#                 # Filter rows where Captured Time is 1000 and Algorithm matches
#                 subset = df[(df['Captured Time'] == 1000) & (df['Algorithm'] == algorithm)]
                
#                 # Check if the metric exists and calculate the mean value for that metric
#                 if metric in df.columns and not subset.empty:
#                     y_values.append(subset[metric].mean())
#                 else:
#                     y_values.append(np.nan)  # Append NaN if data not found
#             else:
#                 y_values.append(np.nan)  # Append NaN if file not found
        
#         # Use the algorithm_name_map to rename the algorithm in the legend, if applicable
#         legend_name = algorithm_name_map.get(algorithm, algorithm)  # Rename if in the map
        
#         # plt.plot(t_values, y_values, marker='o', label=f'{legend_name} (Window {w})')
#         plt.plot(t_values, y_values, marker='o') 
    
#     plt.xlabel('Abnormal Parameter', fontsize=20)
#     plt.ylabel(metric, fontsize=20)

#     # Set the size of the x and y ticks
#     plt.xticks(fontsize=20)
#     plt.yticks(fontsize=20)
    
#     # Display the legend in the bottom right, sorted by window lengths
#     # plt.legend(loc='lower right', fontsize=20)  
#     # plt.legend(loc='center left', fontsize=20)

#     plt.grid(False)

#     # Create save directory if it doesn't exist
#     save_dir = os.path.join('plot_G-Means', f'abtype{abtype}') # Simplified save directory
#     os.makedirs(save_dir, exist_ok=True)
    
#     # Save the plot
#     plot_filename = os.path.join(save_dir, f'abtype{abtype}_{algorithm}_{metric}.png')
#     plt.savefig(plot_filename, bbox_inches='tight')
#     plt.close()


# def plot_all_metrics_for_all_algorithms():
#     result_dir = 'results'
#     metric = 'G-Mean'  # Example of performance metric to plot, you can loop through other metrics too

#     for abtype in range(1, 8):  # Loop over abtypes from 1 to 7
#         for algorithm in algorithms_to_plot:
#             plot_performance_metric(abtype, algorithm, metric)

# if __name__ == "__main__":
#     plot_all_metrics_for_all_algorithms()



import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define window lengths and algorithms to plot
window_lengths_to_plot = [10, 30, 50, 70, 100]
algorithms_to_plot = ['PA', 'PA1', 'PA2', 'OGD', 'OGD_1', 'OGD_2', 'PA1_Csplit', 'PA2_Csplit',
                      'PA1_L1', 'PA2_L1', 'PA1_L2', 'PA2_L2', 'PA_L1', 'PA_L2']

# Mapping for algorithm names in the legend (if used)
algorithm_name_map = {
    'PA1_Csplit': 'CSPA-I', 'PA2_Csplit': 'CSPA-II', 'OGD_1': 'CSOGD-I', 'OGD_2': 'CSOGD-II',
    'PA1': 'PA-I', 'PA2': 'PA-II', 'PA1_L1': 'CSPA-I-L1', 'PA2_L1': 'CSPA-II-L1',
    'PA1_L2': 'CSPA-I-L2', 'PA2_L2': 'CSPA-II-L2', 'PA_L1': 'CSPA-L1', 'PA_L2': 'CSPA-L2',
}

def extract_t_values_for_abtype(result_dir, abtype):
    """Extracts all unique 't' values for a given abtype from filenames."""
    t_values_by_w = {}
    abtype_dir = os.path.join(result_dir, f'abtype{abtype}')
    if os.path.exists(abtype_dir):
        for file in os.listdir(abtype_dir):
            if file.endswith('.csv'):
                try:
                    parts = file.split('_')
                    w_value = int(parts[1][1:])
                    t_value = float(parts[2][1:].replace('.csv', ''))
                    if w_value in window_lengths_to_plot:
                        t_values_by_w.setdefault(w_value, []).append(t_value)
                except (ValueError, IndexError):
                    continue
    for w_value in t_values_by_w:
        t_values_by_w[w_value] = sorted(list(set(t_values_by_w[w_value])))
    return t_values_by_w

# --- CORRECTED FUNCTION ---
def calculate_xaxis_range_for_abtype(result_dir, abtype):
    """
    Calculates the exact min/max x-axis (t-value) range for a specific abtype.
    This ensures all plots for the same abtype have an identical x-axis
    based ONLY on the min and max data points.
    """
    t_values_by_w = extract_t_values_for_abtype(result_dir, abtype)
    all_t_values = [t for w_list in t_values_by_w.values() for t in w_list]

    if not all_t_values:
        return None

    # FIX: Return the absolute min and max without any padding calculation.
    x_min, x_max = min(all_t_values), max(all_t_values)
    
    # Handle the edge case where there's only one data point.
    if x_min == x_max:
        return x_min - 0.1, x_max + 0.1 # Create a small range for plotting

    return x_min, x_max

# --- MODIFIED PLOTTING FUNCTION ---
def plot_performance_metric(abtype, algorithm, metric, x_lim, y_lim):
    """
    Generates and saves a single plot with predefined axis limits and a specific number of ticks.
    """
    t_values_by_w = extract_t_values_for_abtype('results', abtype)
    sorted_window_lengths = sorted(t_values_by_w.keys())
    
    plt.figure(figsize=(10, 6))
    
    for w in sorted_window_lengths:
        t_values = t_values_by_w.get(w, [])
        y_values = []
        for t in t_values:
            file_path = os.path.join('results', f'abtype{abtype}', f'abtype{abtype}_w{w}_t{t}.csv')
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                subset = df[(df['Captured Time'] == 1000) & (df['Algorithm'] == algorithm)]
                if metric in df.columns and not subset.empty:
                    y_values.append(subset[metric].mean())
                else:
                    y_values.append(np.nan)
            else:
                y_values.append(np.nan)
        
        plt.plot(t_values, y_values, marker='o', label=f'W={w}')

    plt.xlabel('Abnormal Parameter', fontsize=20)
    plt.ylabel(metric, fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    # --- AXIS CONTROL AS PER INSTRUCTIONS ---
    # Apply the calculated and fixed limits and tick counts.
    if x_lim:
        plt.xlim(x_lim)
        # Generate exactly 6 ticks from the calculated min to max.
        plt.xticks(np.linspace(start=x_lim[0], stop=x_lim[1], num=6))
    
    plt.ylim(y_lim)
    # Generate exactly 5 ticks from 0.45 to 0.70.
    plt.yticks(np.linspace(start=y_lim[0], stop=y_lim[1], num=5))
    # --- END AXIS CONTROL ---
    
    plt.grid(False)

    save_dir = os.path.join('plot_G-Means', f'abtype{abtype}')
    os.makedirs(save_dir, exist_ok=True)
    
    plot_filename = os.path.join(save_dir, f'abtype{abtype}_{algorithm}_{metric}.png')
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close()

# --- MAIN EXECUTION BLOCK ---
def plot_all_metrics_for_all_algorithms():
    """
    Main function to generate all plots. It now calculates the x-axis range
    per abtype and applies a fixed y-axis range to all plots.
    """
    result_dir = 'results'
    metric = 'G-Mean'
    abtypes_to_plot = range(1, 8)

    # --- SET FIXED Y-AXIS RANGE AND TICKS ---
    # As requested: Y-axis from 0.45 to 0.70 with 5 ticks.
    y_axis_limits = (0.45, 0.70)
    
    print("✅ Starting plot generation with corrected axis settings...")

    for abtype in abtypes_to_plot:
        print(f"\nProcessing pattern: abtype{abtype}")
        
        # Calculate the precise x-axis range for the current abtype.
        x_axis_limits = calculate_xaxis_range_for_abtype(result_dir, abtype)
        
        if x_axis_limits is None:
            print(f"--> ⚠️ Warning: No data found for abtype{abtype}. Skipping.")
            continue
            
        print(f"--> Set uniform x-axis for abtype{abtype} to: ({x_axis_limits[0]:.2f}, {x_axis_limits[1]:.2f})")

        for algorithm in algorithms_to_plot:
            plot_performance_metric(abtype, algorithm, metric, x_axis_limits, y_axis_limits)
            
    print("\n✅ Finished generating all plots.")

if __name__ == "__main__":
    plot_all_metrics_for_all_algorithms()