#!/bin/bash
#SBATCH --job-name=plot_all        # Job name
#SBATCH --output=plots_%A_%a.out   # Standard output log, unique per job array task
#SBATCH --error=plots_%A_%a.err    # Standard error log, unique per job array task
#SBATCH --ntasks=1                 # Number of tasks (jobs)
#SBATCH --cpus-per-task=1          # Number of CPU cores per task
#SBATCH --mem=20G                  # Memory limit
#SBATCH --time=2-00:00:00          # Time limit days-hrs:min:sec
#SBATCH --array=0-5                # Job array for 6 scripts (0 to 5)
#SBATCH --partition=normal         # Partition name

# Load the necessary Python and CUDA modules
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.3.0

# List of scripts to run
scripts=(
    "python plot_heatmaps.py"
    "python plot_metrics_PA.py"
    # "python plot_collage.py"
    "python plot_time.py"
    "python plot_gmean.py"
    "python plot_time2.py"
    "python plot_gmean2.py"
)

# Run the script corresponding to the array task ID
script_to_run=${scripts[$SLURM_ARRAY_TASK_ID]}
echo "Running script: $script_to_run"
$script_to_run