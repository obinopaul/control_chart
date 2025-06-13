# import os
# import csv
# import sys

# # Define the range for window length (w) and abnormal parameter value (t)
# w_values = range(10, 101, 5)
# t_values = [round(x * 0.1, 3) for x in range(1, int(1.8 / 0.1) + 1)]

# # Define the abtype value from command-line argument
# abtype = int(sys.argv[1])

# # Base command template
# base_command = "python data_generator.py -t bc -d {folder}/{filename} -w {w} --t {t} -a 900 -b 100 -m 1 --abtype {abtype} --normalize_abnormal"

# # Generate commands for all combinations of w and t for the specified abtype
# commands = []
# folder = f"data/abtype{abtype}"
# for w in w_values:
#     for t in t_values:
#         filename = f"abtype{abtype}_w{w}_t{t}.libsvm"
#         command = base_command.format(folder=folder, filename=filename, w=w, t=t, abtype=abtype)
#         commands.append((command, folder, filename, w, t, 900, 100, abtype))

# # Create directories, execute the commands, and store metadata
# for command, folder, filename, w, t, a, b, abtype in commands:
#     os.makedirs(folder, exist_ok=True)
#     os.system(command)
    
#     # Write metadata to CSV file in the corresponding abtype folder
#     csv_file_path = os.path.join(folder, 'datasets_metadata.csv')
#     file_exists = os.path.isfile(csv_file_path)
    
#     with open(csv_file_path, mode='a', newline='') as file:
#         writer = csv.writer(file)
#         if not file_exists:
#             # Write the header
#             writer.writerow(['filename', 'w', 't', 'a', 'b', 'abtype'])
#         # Write the dataset metadata
#         writer.writerow([filename, w, t, a, b, abtype])

# print(f"Generated datasets for abtype {abtype}.")



import numpy as np
import os
import csv
import sys

# Function to generate exactly num_values unique t values, ensuring EXACT start and stop values
def generate_exact_unique_t_values(start, stop, num_values, precision=3):
    if num_values == 2:
        return np.array([start, stop])

    # Generate values between start and stop
    raw_values = np.linspace(start, stop, num_values)
    
    # Round the values to the required precision
    rounded_values = np.round(raw_values, precision)
    
    # Force exact start and stop values (no rounding errors)
    rounded_values[0] = start
    rounded_values[-1] = stop
    
    # If there are duplicates due to rounding, increase the precision
    if len(np.unique(rounded_values)) < num_values:
        return generate_exact_unique_t_values(start, stop, num_values, precision + 1)
    
    return rounded_values

# Define the range for window length (w)
w_values = range(10, 101, 5)

# Define the range of t for each abtype using the unique t generation function
# EXACT ranges based on advisor's paper specifications with r=1:
r = 1  # Standard deviation
t_ranges = {
    1: generate_exact_unique_t_values(0.005*r, 0.605*r, 40),  # Slope k (Uptrend): [0.005r, 0.605r]
    2: generate_exact_unique_t_values(0.005*r, 0.605*r, 40),  # Slope k (Downtrend): [0.005r, 0.605r] 
    3: generate_exact_unique_t_values(0.005*r, 1.805*r, 40),  # Shift x (Upshift): [0.005r, 1.805r]
    4: generate_exact_unique_t_values(0.005*r, 1.805*r, 40),  # Shift x (Downshift): [0.005r, 1.805r]
    5: generate_exact_unique_t_values(0.005*r, 1.805*r, 40),  # Systematic k: [0.005r, 1.805r]
    6: generate_exact_unique_t_values(0.005*r, 1.805*r, 40),  # Cyclic a: [0.005r, 1.805r]
    7: generate_exact_unique_t_values(0.005*r, 0.8*r, 40)     # Standard deviation e_t: [0.005r, 0.8r]
}

# Define the abtype value from command-line argument
abtype = int(sys.argv[1])

# Check if abtype is valid
if abtype not in t_ranges:
    print(f"Invalid abtype {abtype}. Please use an abtype from 1 to 7.")
    sys.exit(1)

# Get the appropriate t_values for the specified abtype
t_values = t_ranges[abtype]

# Base command template
base_command = "python data_generator.py -t bc -d {folder}/{filename} -w {w} --t {t} -a 900 -b 100 -m 1 --abtype {abtype} --normalize_abnormal"

# Generate commands for all combinations of w and t for the specified abtype
commands = []
folder = f"data/abtype{abtype}"
for w in w_values:
    for t in t_values:
        filename = f"abtype{abtype}_w{w}_t{t:.{len(str(t).split('.')[1])}f}.libsvm"
        command = base_command.format(folder=folder, filename=filename, w=w, t=f"{t:.{len(str(t).split('.')[1])}f}", abtype=abtype)
        commands.append((command, folder, filename, w, t, 900, 100, abtype))

# Create directories, execute the commands, and store metadata
for command, folder, filename, w, t, a, b, abtype in commands:
    os.makedirs(folder, exist_ok=True)
    os.system(command)
    
    # Write metadata to CSV file in the corresponding abtype folder
    csv_file_path = os.path.join(folder, 'datasets_metadata.csv')
    file_exists = os.path.isfile(csv_file_path)
    
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            # Write the header
            writer.writerow(['filename', 'w', 't', 'a', 'b', 'abtype'])
        # Write the dataset metadata
        writer.writerow([filename, w, f"{t:.{len(str(t).split('.')[1])}f}", a, b, abtype])

print(f"Generated datasets for abtype {abtype}.")
