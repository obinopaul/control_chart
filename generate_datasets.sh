#!/bin/bash
#SBATCH --job-name=dataset_arrray        # Job name
#SBATCH --output=dataset_%a.out      # Standard output and error log
#SBATCH --error=dataset_%a.err       # Standard error log
#SBATCH --ntasks=1                    # Number of tasks (jobs)
#SBATCH --mem=50G                    # Memory limit
#SBATCH --time=03:30:00             # Time limit days-hrs:min:sec
#SBATCH --partition=normal               # Partition name for GPUs
#SBATCH --cpus-per-task=8               # Set 8 CPUs per task
#SBATCH --array=1-7


# Load the necessary Python module
module load Python/3.11.3-GCCcore-12.3.0 
module load CUDA/12.3.0 

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source /home/obinopaul/LIBOL-python_CS_2/libol_env/bin/activate 

# Run the Python script
python -u /home/obinopaul/LIBOL-python_CS_2/generate_datasets.py $SLURM_ARRAY_TASK_ID
