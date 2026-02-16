import os
import sys
import subprocess
import concurrent.futures
import time

def generate_dataset_type(abtype, python_executable):
    """
    Worker function to generate datasets for a specific abnormal type.
    """
    print(f"Starting generation for Abnormal Type {abtype}...")
    start_time = time.time()
    
    try:
        # We capture output to prevent messy interleaved console logs, 
        # but you can remove capture_output=True if you want to see everything.
        result = subprocess.run(
            [python_executable, "generate_datasets.py", str(abtype)], 
            check=True,
            capture_output=True,
            text=True
        )
        elapsed = time.time() - start_time
        print(f"Completed Abnormal Type {abtype} in {elapsed:.2f}s")
        return True, abtype, None
    except subprocess.CalledProcessError as e:
        print(f"Error generating type {abtype}: {e.stderr}")
        return False, abtype, e.stderr
    except Exception as e:
        print(f"Unexpected error for type {abtype}: {e}")
        return False, abtype, str(e)

def main():
    print("Starting PARALLEL generation of all datasets (Abnormal Types 1-7)...")
    print("This will utilize multiple CPU cores.")
    
    python_executable = sys.executable
    start_total = time.time()
    
    # We use ThreadPoolExecutor because the subprocesses release the GIL 
    # and run as independent system processes.
    # max_workers=7 ensures we run all types simultaneously if the system allows.
    with concurrent.futures.ThreadPoolExecutor(max_workers=7) as executor:
        # Submit all tasks
        futures = [
            executor.submit(generate_dataset_type, i, python_executable) 
            for i in range(1, 8)
        ]
        
        # Wait for completion
        for future in concurrent.futures.as_completed(futures):
            success, abtype, error = future.result()
            if not success:
                print(f"!!! Failed to generate datasets for Type {abtype} !!!")

    total_elapsed = time.time() - start_total
    print(f"\nAll dataset generation tasks completed in {total_elapsed:.2f}s.")

if __name__ == "__main__":
    main()
