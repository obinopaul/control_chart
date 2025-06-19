import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# Algorithms to plot, ensure these names match the 'Algorithm' column in your CSV files
algorithms_to_plot = ['PA', 'PA1', 'PA2', 'OGD', 'OGD_1', 'OGD_2', 'PA1_Csplit', 'PA2_Csplit', 
                      'PA1_L1', 'PA2_L1', 'PA1_L2', 'PA2_L2', 'PA_L1', 'PA_L2']

# Mapping for clear, publication-ready names in the plot legend and filenames
algorithm_name_map = {
    'PA1_Csplit': 'CSPA_1',
    'PA2_Csplit': 'CSPA_2',
    'OGD_1': 'CSOGD-I',
    'OGD_2': 'CSOGD-II',
    'PA1': 'PA-I',
    'PA2': 'PA-II',
    'PA1_L1': 'CSPA_1-ℓ¹',
    'PA2_L1': 'CSPA_2-ℓ¹',
    'PA1_L2': 'CSPA_1-ℓ²',
    'PA2_L2': 'CSPA_2-ℓ²',
    'PA_L1': 'CSPA_ℓ¹',
    'PA_L2': 'CSPA_ℓ²',
    'PA': 'PA',
    'OGD': 'OGD'
}

# --- CONFIGURATION ---
# Define which window lengths to show on each plot
window_lengths_for_plot = [10, 30, 70, 100] # Using 100 as 90 might not exist
abtypes_to_plot = range(1, 8)
NUM_T_VALUES_TO_SELECT = 4
BASE_RESULT_DIR = 'results'
BASE_OUTPUT_DIR = 'plot_time'

# --- HELPER FUNCTION ---
def find_all_t_values_for_abtype(result_dir, abtype):
    """Scans all filenames for an abtype and returns a list of unique t-values."""
    t_values = set()
    abtype_dir = os.path.join(result_dir, f'abtype{abtype}')
    if not os.path.exists(abtype_dir):
        return []
    
    for file in os.listdir(abtype_dir):
        if file.endswith('.csv'):
            try:
                # e.g., abtype1_w10_t1.2.csv
                parts = file.split('_')
                t_str = parts[2][1:].replace('.csv', '')
                t_values.add(float(t_str))
            except (ValueError, IndexError):
                continue
    return sorted(list(t_values))

# --- PLOTTING FUNCTION ---
def plot_algorithm_time_performance(abtype, t_value, algorithm, save_dir):
    """
    Generates a single plot for one algorithm, showing multiple window lengths.
    Each line on the plot represents a different window length.
    """
    plt.figure(figsize=(12, 7))
    
    # Use distinct line styles for visual clarity
    line_styles = ['-', '--', '-.', ':']
    max_time_overall = 0  # To set the y-axis limit dynamically

    # Plot a line for each specified window length
    for i, w in enumerate(window_lengths_for_plot):
        file_path = os.path.join(BASE_RESULT_DIR, f'abtype{abtype}', f'abtype{abtype}_w{w}_t{t_value}.csv')
        
        if not os.path.exists(file_path):
            print(f"    - Skipping w={w}, file not found: {file_path}")
            continue
            
        df = pd.read_csv(file_path)
        algo_df = df[df['Algorithm'] == algorithm]
        
        if not algo_df.empty:
            # Update the max time for y-axis scaling
            max_time_overall = max(max_time_overall, algo_df['Time'].max())
            
            style = line_styles[i % len(line_styles)]
            plt.plot(algo_df['Captured Time'], algo_df['Time'], label=f'w={w}', lw=2.5, linestyle=style)

    # --- Apply Styling and Formatting ---
    legend_name = algorithm_name_map.get(algorithm, algorithm)
    # plt.title(f'Time Complexity of {legend_name}\n(Pattern {abtype}, Abnormal Parameter t={t_value})', fontsize=22, fontweight='bold')
    plt.xlabel('Training Instances', fontsize=20)
    plt.ylabel('Time (seconds)', fontsize=20)
    
    # Set axis limits and ticks
    plt.xlim(0, 1000)
    if max_time_overall > 0:
        plt.ylim(0, max_time_overall * 1.1) # Add 10% padding
    
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    plt.grid(False) # Remove grid lines as requested
    
    # if plt.gca().has_data(): # Only add legend if something was plotted
    #     plt.legend(loc='upper left', fontsize=14)
    
    # Save the plot
    plot_filename = f"{legend_name}_time_plot.png"
    save_path = os.path.join(save_dir, plot_filename)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


# --- MAIN EXECUTION BLOCK ---
def generate_all_plots():
    """
    Main function to find t-values, create directories, and generate all plots.
    """
    print("🚀 Starting plot generation...")

    for abtype in abtypes_to_plot:
        print(f"\nProcessing Pattern: abtype{abtype}")
        
        # 1. Find all available t-values for this pattern
        all_t = find_all_t_values_for_abtype(BASE_RESULT_DIR, abtype)
        if not all_t:
            print(f"--> No data found for abtype{abtype}. Skipping.")
            continue
        
        # 2. Randomly select 4 (or fewer if not enough are available)
        num_to_sample = min(len(all_t), NUM_T_VALUES_TO_SELECT)
        selected_t_values = random.sample(all_t, num_to_sample)
        print(f"--> Found {len(all_t)} t-values. Randomly selected: {selected_t_values}")

        # 3. Loop through selected t-values and create plots
        for t in selected_t_values:
            # Create the specific subdirectory for this t-value
            save_dir = os.path.join(BASE_OUTPUT_DIR, f'abtype{abtype}', f't_{t}')
            os.makedirs(save_dir, exist_ok=True)
            print(f"  -> Generating plots for t={t} in folder: {save_dir}")

            # 4. Generate a plot for each algorithm within this t-value's folder
            for algo in algorithms_to_plot:
                plot_algorithm_time_performance(abtype, t, algo, save_dir)
                
    print("\n✅ All plots have been generated successfully!")

if __name__ == "__main__":
    generate_all_plots()
