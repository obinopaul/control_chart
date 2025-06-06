#!/bin/bash
#SBATCH --job-name=plot_collage        # Job name
#SBATCH --output=collage_%a.out         # Standard output log (overwrites existing)
#SBATCH --error=collage_%a.err          # Standard error log (overwrites existing) 
#SBATCH --ntasks=1                    # Number of tasks (jobs)
#SBATCH --mem=20G                    # Memory limit
#SBATCH --time=2-00:00:00             # Time limit days-hrs:min:sec
#SBATCH --partition=normal               # Partition name for GPUs


# Load the necessary Python module
module load Python/3.11.3-GCCcore-12.3.0 
module load CUDA/12.3.0 


# Run the Python script
python -u /home/obinopaul/LIBOL-python_CS_2/plot_collage.py

