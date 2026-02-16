import numpy as np
import os

def normalize_sample(sample):
    """
    Normalizes a single time series sample to have a unit L2 norm.
    This function is identical to the one in your data_generator.py
    to ensure consistency.

    Args:
        sample (np.ndarray): A 1D numpy array representing a time series.

    Returns:
        np.ndarray: The normalized time series.
    """
    # Calculate the L2 norm (Euclidean norm) of the sample.
    norm = np.linalg.norm(sample)
    
    # If the norm is zero (i.e., the sample is all zeros),
    # return the sample as is to avoid division by zero.
    if norm == 0:
        return sample
        
    # Divide each element by the norm to scale it to unit length.
    return sample / norm

def convert_to_libsvm(input_file_path, output_file_path, normalize=True):
    """
    Converts a dense, CSV-like dataset to the LIBSVM format.

    Args:
        input_file_path (str): The path to the input wafer data file.
        output_file_path (str): The path where the converted LIBSVM file will be saved.
        normalize (bool): If True, each time series sample will be L2 normalized.
    """
    print(f"Starting conversion for: {input_file_path}")
    
    # Check if the input file exists before proceeding.
    if not os.path.exists(input_file_path):
        print(f"Error: Input file not found at '{input_file_path}'")
        return

    # Open the output file in write mode. The 'with' statement ensures it's properly closed.
    with open(output_file_path, 'w') as f_out:
        # Open the input file in read mode.
        with open(input_file_path, 'r') as f_in:
            # Process each line in the input file.
            for line in f_in:
                # Strip leading/trailing whitespace and skip empty lines.
                line = line.strip()
                if not line:
                    continue
                
                # Split the line by commas to separate the values.
                parts = line.split(',')
                
                # The first element is the class label. Convert it to an integer.
                # The Wafer dataset uses 1 (normal) and -1 (abnormal).
                label = int(float(parts[0]))
                
                # The rest of the elements are the time series values.
                # Convert them from strings to a numpy array of floats.
                features = np.array([float(p) for p in parts[1:]])
                
                # If normalization is enabled, apply it to the features.
                if normalize:
                    features = normalize_sample(features)
                
                # Create the feature string in LIBSVM format.
                # This format is "index:value", and indices are 1-based.
                # We use enumerate to get both the index (j) and the value.
                feature_string = ' '.join(f"{j+1}:{features[j]:.6f}" for j in range(len(features)))
                
                # Write the final LIBSVM formatted line to the output file.
                f_out.write(f"{label} {feature_string}\n")
                
    print(f"Successfully converted and saved to: {output_file_path}")
    print("-" * 30)


# --- Main Execution ---
if __name__ == "__main__":
    # Define the input and output file paths based on your directory structure.
    # Assumes the script is run from the same directory containing wafer_TRAIN and wafer_TEST.
    
    # --- Process the Training Data ---
    input_train_file = 'wafer_TRAIN'
    output_train_file = 'wafer_train.libsvm'
    convert_to_libsvm(input_train_file, output_train_file, normalize=True)
    
    # --- Process the Test Data ---
    input_test_file = 'wafer_TEST'
    output_test_file = 'wafer_test.libsvm'
    convert_to_libsvm(input_test_file, output_test_file, normalize=True)
    
    print("All conversions complete.")