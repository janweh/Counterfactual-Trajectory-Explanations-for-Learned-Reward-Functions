import sys
import os
from pathlib import Path
adjacent_folder = Path(__file__).parent.parent
sys.path.append(str(adjacent_folder))
from quality_metrics.distance_measures import distance_subtrajectories
from quality_metrics.diversity_measures import diversity_single
from quality_metrics.validity_measures import validity_single_partial
from quality_metrics.critical_state_measures import critical_state_single
from quality_metrics.realisticness_measures import realisticness_single_partial
from quality_metrics.sparsity_measure import sparsitiy_single_partial
from interpretability.generation_methods.counterfactual_random import generate_counterfactual_random
from interpretability.generation_methods.counterfactual_mcts import generate_counterfactual_mcts
from interpretability.generation_methods.counterfactual_step import generate_counterfactual_step
from quality_metrics.quality_metrics import measure_quality
import numpy as np
import pickle
from helpers.util_functions import partial_trajectory
import torch
from moral.ppo import PPO, TrajectoryDataset, update_policy
import random
from envs.gym_wrapper import *
from moral.airl import *
from moral.active_learning import *

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

## LOADING ALL NECESSARY THINGS
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

random.seed(4)
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

ppo = PPO(state_shape=state_shape, in_channels=in_channels, n_actions=n_actions).to(device)
optimizer = torch.optim.Adam(ppo.parameters(), lr=config.lr_ppo)
ppo.load_state_dict(torch.load(os.path.join('saved_models', 'ppo_airl_v2_[1,10]_new.pt'), map_location=torch.device('cpu')))
discriminator = DiscriminatorMLP(state_shape=state_shape, in_channels=in_channels).to(device)
discriminator.load_state_dict(torch.load(os.path.join('saved_models', 'discriminator_v2_[1,10]_new.pt'), map_location=torch.device('cpu')))
org_traj_seed = pickle.load(open(os.path.join('demonstrations', 'original_trajectories_new_maxsteps75_airl_1000_new.pkl'), 'rb'))

# # load weights
# with open(os.path.join('quality_metrics','qc_weights.pkl'), 'rb') as f:
#     weights = pickle.load(f)

# weight = {'validity': 1, 'proximity': 1, 'critical_state': 1, 'diversity': 1, 'realisticness': 1, 'sparsity': 1}

with open('quality_metrics\\1000weights.pkl', 'rb') as f:
    loaded_weights = pickle.load(f)

# # used for resetting the normalisation
# with open('interpretability\\normalisation_values.pkl', 'wb') as f:
#     normalisation = {'validity': [0, 1], 'diversity': [0, 1], 'proximity': [0, 1], 'critical_state': [0, 1], 'realisticness': [0, 1], 'sparsity': [0, 1]}
#     pickle.dump(normalisation, f)

with open('interpretability\\normalisation_values.pkl', 'rb') as f:
    old_normalisation = pickle.load(f)

# CALCULATE NORMALISATION VALUES
# load original trajectories
proxs, vals, divs, crits, spars, reals = [], [], [], [], [], []
prev_org_trajs, prev_cf_trajs, prev_starts = [], [], []
length = 5
it = 11
for i in range(2*it*length, (2*it+1)*length):
    weight = loaded_weights[i]
    org_traj, seed_env = org_traj_seed[i]
    print(i)
    random_org, random_cf, random_start = generate_counterfactual_mcts(org_traj, ppo, discriminator, seed_env, prev_org_trajs, prev_cf_trajs, prev_starts, config, weights=weight)
    proxs.append(distance_subtrajectories(random_org, random_cf))
    vals.append(validity_single_partial(random_org, random_cf))
    divs.append(diversity_single(random_org, random_cf, random_start, prev_org_trajs, prev_cf_trajs, prev_starts))
    crits.append(critical_state_single(ppo, random_org['states'][0]))
    spars.append(sparsitiy_single_partial(random_org, random_cf))
    reals.append(realisticness_single_partial(random_org, random_cf))
    prev_org_trajs.append(random_org)
    prev_cf_trajs.append(random_cf)
    prev_starts.append(random_start)

# normalisation = {'validity': [min(vals), max(vals)], 'diversity': [min(divs), max(divs)], 'proximity': [min(proxs), max(proxs)], 'critical_state': [min(crits), max(crits)], 'realisticness': [min(reals), max(reals)], 'sparsity': [min(spars), max(spars)]}
# print(normalisation)
prev_org_trajs, prev_cf_trajs, prev_starts = [], [], []
for i in range((2*it+1)*length, length*(2*it+2)):
    weight = loaded_weights[i]
    print(i)
    org_traj, seed_env = org_traj_seed[i]
    counterfactual_trajs, counterfactual_rewards, starts, end_cfs, end_orgs = generate_counterfactual_step(org_traj, ppo, discriminator, seed_env, config)
    sort_index, qc_stats = measure_quality(org_traj, counterfactual_trajs, counterfactual_rewards, starts, end_cfs, end_orgs, ppo, prev_org_trajs, prev_cf_trajs, prev_starts, config.criteria, weights=weight)
    chosen_counterfactual_trajectory = counterfactual_trajs[sort_index]
    chosen_start = starts[sort_index]
    chosen_end_cf = end_cfs[sort_index]
    chosen_end_org = end_orgs[sort_index]
    step_org = partial_trajectory(org_traj, chosen_start, chosen_end_org)
    step_cf = partial_trajectory(chosen_counterfactual_trajectory, chosen_start, chosen_end_cf)
    prev_org_trajs.append(step_org)
    prev_cf_trajs.append(step_cf)
    prev_starts.append(chosen_start)

    proxs.append(distance_subtrajectories(step_org, step_cf))
    vals.append(validity_single_partial(step_org, step_cf))
    divs.append(diversity_single(step_org, step_cf, random_start, prev_org_trajs, prev_cf_trajs, prev_starts))
    crits.append(critical_state_single(ppo, step_org['states'][0]))
    spars.append(sparsitiy_single_partial(step_org, step_cf))
    reals.append(realisticness_single_partial(step_org, step_cf))

normalisation = {'validity': [min(vals), max(vals)], 'diversity': [min(divs), max(divs)], 'proximity': [min(proxs), max(proxs)], 'critical_state': [min(crits), max(crits)], 'realisticness': [min(reals), max(reals)], 'sparsity': [min(spars), max(spars)]}
print(normalisation)
print(old_normalisation)
# take the mins and maxes across the new and old normalisation values
for key in normalisation.keys():
    normalisation[key][0] = min(normalisation[key][0], old_normalisation[key][0])
    normalisation[key][1] = max(normalisation[key][1], old_normalisation[key][1])
print(normalisation)
# check if normalisation is the same as old normalisation
same = True
for key in normalisation.keys():
    if normalisation[key][0] != old_normalisation[key][0] or normalisation[key][1] != old_normalisation[key][1]:
        same = False
        break
print('same:', same)
# write into pickle

with open(os.path.join('interpretability','normalisation_values.pkl'), 'wb') as f:
    pickle.dump(normalisation, f)