import sys
import os
from pathlib import Path
adjacent_folder = Path(__file__).parent.parent.parent
sys.path.append(str(adjacent_folder))
from tqdm import tqdm
from moral.ppo import PPO, TrajectoryDataset, update_policy
import torch
from moral.airl import *
from moral.active_learning import *
import numpy as np
from envs.gym_wrapper import *
from moral.preference_giver import *
import argparse
import sys
from copy import *
from helpers.util_functions import *
import random 
import time
from quality_metrics.distance_measures import distance_all as distance_all
import pickle
from helpers.parsing import sort_args, parse_attributes
from interpretability.generation_methods.counterfactual_mcts import *
from interpretability.generation_methods.counterfactual_step import *
from interpretability.generation_methods.counterfactual_random import *
from quality_metrics.critical_state_measures import critical_state_all as critical_state

# Use GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class config:
    env_id= 'randomized_v2'
    env_steps= 8e6
    batchsize_ppo= 12
    n_queries= 50
    preference_noise= 0
    n_workers= 1
    lr_ppo= 3e-4
    entropy_reg= 0.25
    gamma= 0.999
    epsilon= 0.1
    ppo_epochs= 5
    max_steps = 75
    measure_statistics = True
    num_runs = 1000
    criteria = ['validity', 'diversity', 'proximity', 'critical_state', 'realisticness', 'sparsity']
    # criteria = ['baseline']
    # criteria = ['validity']
    cf_method = 'mcts' # 'mcts' or 'step'

if __name__ == '__main__':

    # Parse Arguments
    parser = argparse.ArgumentParser(description='Parse attributes string')
    parser.add_argument('--likelihood_terminal', type=float, default=0.2, help='likelihood of terminal action per step')
    parser.add_argument('--num_runs', type=int, default=1000, help='number of CTEs to generate')
    args = parser.parse_args()

    config.num_runs = args.num_runs
    likelihood_terminal = args.likelihood_terminal


    # determine whether this is a baseline run or not
    baseline = 'baseline' in config.criteria

    print('Criteria: ', config.criteria, baseline)
    
    # make a random number based on the time
    random.seed(7)
    seed_env = random.randint(0, 100000)
    torch.manual_seed(seed_env)
    np.random.seed(seed_env)
    
    # Create Environment
    vec_env = VecEnv(config.env_id, config.n_workers, seed=seed_env)
    states = vec_env.reset()
    states_tensor = torch.tensor(states).float().to(device)

    # Fetch Shapes
    n_actions = vec_env.action_space.n
    obs_shape = vec_env.observation_space.shape
    state_shape = obs_shape[:-1]
    in_channels = obs_shape[-1]

    # Initialize Models
    print('Initializing and Normalizing Rewards...')
    ppo = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
    optimizer = torch.optim.Adam(ppo.parameters(), lr=config.lr_ppo)
    ppo.load_state_dict(torch.load(os.path.join('saved_models', 'ppo_airl_v2_[1,10]_new.pt'), map_location=torch.device('cpu')))
    discriminator = DiscriminatorMLP(state_shape=state_shape, in_channels=in_channels).to(device)
    discriminator.load_state_dict(torch.load(os.path.join('saved_models', 'discriminator_v2_[1,10]_new.pt'), map_location=torch.device('cpu')))

    all_org_trajs, all_cf_trajs, all_starts, all_end_orgs, all_end_cfs, all_part_orgs, all_part_cfs, random_baseline_cfs, random_baseline_orgs = [], [], [], [], [], [], [], [], []
    lengths_org, lengths_cf, start_points, quality_criteria, effiencies, qc_statistics = [], [], [], [], [], []

    # load the original trajectories
    org_traj_seed = pickle.load(open(os.path.join('demonstrations', 'original_trajectories_new_maxsteps75_airl_1000_new.pkl'), 'rb'))

    run = 0
    sumss = []
    for org_traj, seed_env in org_traj_seed:
        print(run)
        if run >= config.num_runs: break
        run += 1

        time_start = time.time()

        random_org, random_cf, random_start, sums = generate_counterfactual_random(org_traj, ppo, discriminator, seed_env, likelihood_terminal, config)
        sumss.append(sums)
        
        time_end = time.time()
        effiencies.append(time_end - time_start)
        lengths_org.append(len(random_org['states']))
        lengths_cf.append(len(random_cf['states']))
        start_points.append(random_start)

        random_rewards = sum(random_org['rewards'])
        all_part_orgs.append((random_org, random_rewards))
        random_rewards_cf = sum(random_cf['rewards'])
        all_part_cfs.append((random_cf, random_rewards_cf))
        all_starts.append(random_start)

    #  check if folder exists
    if not os.path.exists(os.path.join('datasets','ablations_norm','random' + str(likelihood_terminal))):
        os.makedirs(os.path.join('datasets','ablations_norm','random' + str(likelihood_terminal)))
        os.makedirs(os.path.join('datasets','ablations_norm','random' + str(likelihood_terminal), 'statistics'))

    print(np.mean(sumss))

    # #save the trajectories
    # with open(os.path.join('datasets','ablations_norm','random' + str(likelihood_terminal), 'org_trajectories.pkl'), 'wb') as f:
    #     pickle.dump(all_part_orgs, f)

    # with open(os.path.join('datasets','ablations_norm','random' + str(likelihood_terminal), 'cf_trajectories.pkl'), 'wb') as f:
    #     pickle.dump(all_part_cfs, f)

    # with open(os.path.join('datasets', 'ablations_norm', 'random' + str(likelihood_terminal), 'statistics', 'start_points.pkl'), 'wb') as f:
    #     pickle.dump(start_points, f)

    # with open(os.path.join('datasets','ablations_norm','random' + str(likelihood_terminal), 'statistics','effiencies.pkl'), 'wb') as f:
    #     pickle.dump(effiencies, f)

    # with open(os.path.join('datasets','ablations_norm','random' + str(likelihood_terminal), 'statistics','lengths_org.pkl'), 'wb') as f:
    #     pickle.dump(lengths_org, f)

    # with open(os.path.join('datasets','ablations_norm','random' + str(likelihood_terminal), 'statistics','lengths_cf.pkl'), 'wb') as f:
    #     pickle.dump(lengths_cf, f)

    print(np.mean(effiencies))