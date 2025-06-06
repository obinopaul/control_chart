#!/bin/bash
#SBATCH --job-name=plot_metrics        # Job name
#SBATCH --output=metrics_%a.out         # Standard output log (overwrites existing)
#SBATCH --error=metrics_%a.err          # Standard error log (overwrites existing) 
#SBATCH --ntasks=1                    # Number of tasks (jobs)
#SBATCH --mem=20G                    # Memory limit
#SBATCH --time=2-00:00:00             # Time limit days-hrs:min:sec
#SBATCH --partition=normal               # Partition name for GPUs


# Load the necessary Python module
module load Python/3.11.3-GCCcore-12.3.0 
module load CUDA/12.3.0 

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source /home/obinopaul/LIBOL-python_CS_2/libol_env/bin/activate 

# Run the Python script
python -u /home/obinopaul/LIBOL-python_CS_2/plot_metrics.py 

