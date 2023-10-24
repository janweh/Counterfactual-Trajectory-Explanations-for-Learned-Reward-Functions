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
# /usr/bin/nvidia-smi

# Assuming you have a Python module to load
module use /opt/insy/modulefiles
module load miniconda/3.9

# Your python script instead of "my_program.py"
# best step step: 
# srun python interpretability/counterfactual_trajectories.py --method step --weights 29 --num_runs 1000 --ending_meeting False --ending_prob 0.55 --num_deviations 3 --follow_policy True
# srun python interpretability/counterfactual_trajectories.py --method step --special_weights scrambled --num_runs 1000 --ending_meeting False --ending_prob 0.65 --num_deviations 3 --follow_policy True

# srun python interpretability/counterfactual_trajectories.py --method mcts --special_weights scrambled --num_runs 1000 --starting_points 70 --action_threshold 0.003 --discount_factor 1.0 --num_simulations 1 --num_iterations 10 --likelihood_terminal 0.45 --policy_simulation False --branching_heuristic False
# srun python interpretability/counterfactual_trajectories.py --method mcts --special_weights scrambled --num_runs 1000 --starting_points 70 --action_threshold 0.003 --discount_factor 1.0 --num_simulations 1 --num_iterations 30 --likelihood_terminal 0.35 --policy_simulation False --branching_heuristic False
# --policy_simulation False --starting_points 20 --num_iterations 5 --discount_factor 1.0 --num_simulations 10
# --num_iterations 10 --likelihood_terminal 1.0 --starting_points 1 
# srun python evaluation/train_mimic_sidebyside_multi.py --weight 29 --num_seeds 10 --method mcts
# /usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"

# python interpretability/generation_methods/random_cf_generator.py  --num_runs 1000 --likelihood_terminal 0.65

srun python evaluation/train_mimic_sidebyside_multi.py --method mcts --num_seeds 10 --weight_con 1 --weight_sin 100