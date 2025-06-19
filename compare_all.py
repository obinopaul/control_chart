import os
# import sys
# import subprocess

# def get_datasets(abtype):
#     datasets = []
#     folder_path = f"data/abtype{abtype}"
#     if os.path.exists(folder_path):
#         for file in os.listdir(folder_path):
#             if file.endswith(".libsvm"):
#                 datasets.append(os.path.join(folder_path, file))
#     return datasets

# def run_model(dataset):
#     dataset_name = os.path.basename(dataset)
#     save_path = os.path.join("results", os.path.basename(os.path.dirname(dataset)))
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#     command = ["python", "compare.py", "-t", "bc", "-d", dataset, "-f", "libsvm", "-n", "20", "-s", save_path]
#     try:
#         subprocess.run(command, check=True)
#     except subprocess.CalledProcessError as e:
#         print(f"Failed to run model on dataset {dataset_name}: {e}")

# def run_models_on_datasets(datasets):
#     for dataset in datasets:
#         run_model(dataset)

# def main():
#     try:
#         abtype = int(sys.argv[1])
#     except (IndexError, ValueError):
#         print("Usage: python compare_all.py <abtype>")
#         sys.exit(1)

#     os.chdir("/home/obinopaul/LIBOL-python_CS_2")
#     datasets = get_datasets(abtype)
#     if datasets:
#         run_models_on_datasets(datasets)
#     else:
#         print(f"No datasets found for abtype_{abtype}")

# if __name__ == '__main__':
#     main()



import os
import sys
import subprocess
import concurrent.futures

def get_datasets(abtype):
    datasets = []
    # folder_path = f"data/abtype{abtype}"
    folder_path = f"/ourdisk/hpc/disc/obinopaul/auto_archive_notyet/tape_2copies/data/abtype{abtype}"
    if os.path.exists(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith(".libsvm"):
                datasets.append(os.path.join(folder_path, file))
    return datasets

def split_datasets_by_window_length(datasets):
    chunks = {1: [], 2: [], 3: [], 4: [], 5: []}
    
    for dataset in datasets:
        # Extract window length (w) from the filename
        filename = os.path.basename(dataset)
        try:
            w_value = int(filename.split('_')[1][1:])  # Extract the value of 'w'
        except (IndexError, ValueError):
            print(f"Could not parse window length from filename: {filename}")
            continue
        
        # Assign to the appropriate chunk based on window length
        if 10 <= w_value <= 25:
            chunks[1].append(dataset)
        elif 30 <= w_value <= 45:
            chunks[2].append(dataset)
        elif 50 <= w_value <= 65:
            chunks[3].append(dataset)
        elif 70 <= w_value <= 85:
            chunks[4].append(dataset)
        elif 90 <= w_value <= 100:
            chunks[5].append(dataset)
    
    return list(chunks.values())  # Return list of chunks


def run_model(dataset):
    dataset_name = os.path.basename(dataset)
    # save_path = os.path.join("results", os.path.basename(os.path.dirname(dataset)))
    save_path = os.path.join("/ourdisk/hpc/disc/obinopaul/auto_archive_notyet/tape_2copies/results", os.path.basename(os.path.dirname(dataset)))
    os.makedirs(save_path, exist_ok=True)
    # if not os.path.exists(save_path):
    #    os.makedirs(save_path)
    command = ["python", "compare.py", "-t", "bc", "-d", dataset, "-f", "libsvm", "-n", "20", "-s", save_path]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to run model on dataset {dataset_name}: {e}")

def run_models_in_chunk(chunk):
    for dataset in chunk:
        run_model(dataset)

def run_models_on_datasets(datasets):
    # Split datasets into chunks of 5
    chunk_size = 5
    chunks = [datasets[i:i + chunk_size] for i in range(0, len(datasets), chunk_size)]
    
    # chunks = split_datasets_by_window_length(datasets)
    
    # Run each chunk in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit each chunk of datasets to be processed in parallel
        futures = [executor.submit(run_models_in_chunk, chunk) for chunk in chunks]

        # Wait for all futures to complete
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error running model on a chunk: {e}")

def main():
    try:
        abtype = int(sys.argv[1])
    except (IndexError, ValueError):
        print("Usage: python compare_all.py <abtype>")
        sys.exit(1)

    # os.chdir("/home/obinopaul/LIBOL-python_CS_2")
    datasets = get_datasets(abtype)
    if datasets:
        run_models_on_datasets(datasets)
    else:
        print(f"No datasets found for abtype_{abtype}")

if __name__ == '__main__':
    main()
