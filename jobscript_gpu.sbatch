#!/bin/sh
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=04:00:00 #If I assume it would be a long running job
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=2 
#SBATCH --mem-per-cpu=1024 # I have assigned 2GB to each CPU instead of the overall 1GB
#SBATCH --mail-type=END 


# previous=$(/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2') 

/usr/bin/scontrol show job -d "$SLURM_JOB_ID"
# previous=$(/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2') 
# /usr/bin/nvidia-smi # Check sbatch settings are working (it should show the GPU that you requested)


# Assuming you have a Python module to load
module use /opt/insy/modulefiles
module load miniconda/3.9


# Your python script instead of "my_program.py"
srun python evaluation/train_mimic_sidebyside_multi.py --method mcts --num_seeds 1 --weight_con 1 --weight_sin 1