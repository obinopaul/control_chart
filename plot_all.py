import subprocess
from multiprocessing import Pool, cpu_count

# List of scripts to run
scripts = [
    # "python plot_heatmaps.py",
    "python plot_metrics_PA.py",
    # "python plot_collage.py",
    "python plot_time.py",
    "python plot_gmean.py",
    "python plot_time2.py",
    "python plot_gmean2.py",
]

# Function to run a script
def run_script(script):
    try:
        print(f"Running {script}...")
        subprocess.run(script, check=True, shell=True)
        print(f"{script} completed successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script}: {e}\n")

# Running the scripts in parallel using multiprocessing Pool
if __name__ == "__main__":
    # Use the number of available CPU cores, or you can set it explicitly based on the Slurm request
    num_workers = min(6, cpu_count())  # You requested 6 CPUs in the Slurm script

    with Pool(processes=num_workers) as pool:
        pool.map(run_script, scripts)
