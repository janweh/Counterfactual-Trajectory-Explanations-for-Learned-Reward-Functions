import sys
import os
from pathlib import Path
adjacent_folder = Path(__file__).parent.parent
sys.path.append(str(adjacent_folder))
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
from helpers.visualize_trajectory import visualize_two_part_trajectories, visualize_two_part_trajectories_part
from helpers.util_functions import *
import random 
import time
from quality_metrics.quality_metrics import measure_quality, evaluate_qcs_for_cte
from quality_metrics.distance_measures import distance_all as distance_all
import pickle
from helpers.parsing import sort_args, parse_attributes
from generation_methods.counterfactual_mcts import *
from generation_methods.counterfactual_step import *
from generation_methods.counterfactual_random import *
from quality_metrics.critical_state_measures import critical_state_all as critical_state
from distutils.util import strtobool

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
    base_path = os.path.join('datasets', 'weights_norm')
    measure_statistics = True
    num_runs = 1000
    criteria = ['validity', 'diversity', 'proximity', 'critical_state', 'realisticness', 'sparsity']
    # criteria = ['baseline']
    # criteria = ['validity']
    cf_method = 'mcts' # 'mcts' or 'step'
    scrambled_weights = False
    unit_weights = False
    start_run = 0

def str2bool(v):
    return bool(strtobool(v))

def read_out_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='mcts')
    parser.add_argument('--num_iterations', type=int, default=5)
    parser.add_argument('--action_threshold', type=float, default=0.003)
    parser.add_argument('--likelihood_terminal', type=float, default=0.2)
    parser.add_argument('--discount_factor', type=float, default=1.0)
    parser.add_argument('--starting_points', type=int, default=40)
    parser.add_argument('--branching_heuristic', type=strtobool, default=False)
    parser.add_argument('--policy_simulation', type=strtobool, default=False)
    parser.add_argument('--num_simulations', type=int, default=1)

    parser.add_argument('--ending_meeting', type=strtobool, default=False)
    parser.add_argument('--ending_prob', type=float, default=0.5)
    parser.add_argument('--num_deviations', type=int, default=3)
    parser.add_argument('--follow_policy', type=strtobool, default=True)

    parser.add_argument('--weights', type=int, default=0)
    parser.add_argument('--special_weights', type=str, default='')

    parser.add_argument('--num_runs', type=int, default=config.num_runs)
    parser.add_argument('--start_run', type=int, default=config.start_run)

    args = parser.parse_args()
    conf = {}
    file_name = 'a'
    if args.method == 'mcts':
        config.cf_method = 'mcts'
        conf['num_iterations'] = args.num_iterations
        if args.num_iterations != 40:
            file_name += '_num_iterations' + str(args.num_iterations)
        conf['action_threshold'] = args.action_threshold
        if args.action_threshold != 0.003:
            file_name += '_action_threshold' + str(args.action_threshold)
        conf['likelihood_terminal'] = args.likelihood_terminal
        if args.likelihood_terminal != 0.2:
            file_name += '_likelihood_terminal' + str(args.likelihood_terminal)
        conf['discount_factor'] = args.discount_factor
        if args.discount_factor != 0.85:
            file_name += '_discount_factor' + str(args.discount_factor)
        conf['starting_points'] = args.starting_points
        if args.starting_points != 2:
            file_name += '_starting_points' + str(args.starting_points)
        conf['branching_heuristic'] = args.branching_heuristic
        if args.branching_heuristic != True:
            file_name += '_branching_heuristic' + str(args.branching_heuristic)
        conf['policy_simulation'] = args.policy_simulation
        if args.policy_simulation != True:
            file_name += '_policy_simulation' + str(args.policy_simulation)
        conf['num_simulations'] = args.num_simulations
        if args.num_simulations != 1:
            file_name += '_num_simulations' + str(args.num_simulations)

    elif args.method == 'step':
        config.cf_method = 'step'
        conf['ending_meeting'] = args.ending_meeting
        if args.ending_meeting != False:
            file_name += '_ending_meeting' + str(args.ending_meeting)
        conf['ending_prob'] = args.ending_prob
        if args.ending_prob != 0.2:
            file_name += '_ending_prob' + str(args.ending_prob)
        conf['num_deviations'] = args.num_deviations
        if args.num_deviations != 1:
            file_name += '_num_deviations' + str(args.num_deviations)
        conf['follow_policy'] = args.follow_policy
        if args.follow_policy != True:
            file_name += '_follow_policy' + str(args.follow_policy)

    if args.weights != 0:
        file_name = 'weight_' + str(args.weights)
    if args.special_weights != '':
        # file_name += '_special_weights' + str(args.special_weights)
        if args.special_weights == 'scrambled':
            config.scrambled_weights = True
        if args.special_weights == 'ones':
            config.unit_weights = True

    config.num_runs = args.num_runs
    print(conf, file_name)
    return conf, os.path.join(args.method, file_name), args.weights



if __name__ == '__main__':
    print('Starting...')
    conf, filename, weight_num = read_out_args()

    # determine whether this is a baseline run or not
    baseline = 'baseline' in config.criteria
    if not baseline:
        path_folder = os.path.join(config.base_path, filename)
    else:
        path_folder = os.path.join(config.base_path, 'baseline', str(config.num_runs))
    

    print(path_folder)
    print('Criteria: ', config.criteria, baseline)
    
    # make a random number based on the time
    random.seed(6)
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
    weights = pickle.load(open(os.path.join('quality_metrics', '1000weights.pkl'), 'rb'))

    # normalising_qcs(ppo, discriminator, org_traj_seed, config)

    # stop after the number of runs config.num_runs or iterate through all original trajectories
    run = 0
    for org_traj, seed_env in org_traj_seed:
        if config.scrambled_weights:
            weight = weights[run]
        elif config.unit_weights:
            weight = {'validity': 1, 'diversity': 1, 'proximity': 1, 'critical_state': 1, 'realisticness': 1, 'sparsity': 1}
        else:
            weight = weights[weight_num]
        print(run)
        if run < config.start_run:
            run += 1
            continue
        if run >= config.num_runs: break
        run += 1

        time_start = time.time()

        # generate the counterfactual trajectories
        if config.cf_method == 'mcts':
            # Method 1: MCTS            
            traj_org, traj_cf, traj_start = generate_counterfactual_mcts(org_traj, ppo, discriminator, seed_env, all_org_trajs, all_cf_trajs, all_starts, config, conf=conf, weights=weight)
            efficiency = time.time() - time_start
            # visualize_two_part_trajectories_part(traj_org, traj_cf)


            mcts_rewards = sum(traj_org['rewards'])
            all_part_orgs.append((traj_org, mcts_rewards))
            mcts_rewards_cf = sum(traj_cf['rewards'])
            all_part_cfs.append((traj_cf, mcts_rewards_cf))
            print(len(all_part_cfs))

            # # # append this to 'datasets\1000mcts\1000\cf_trajectories_tmp.pkl'
            # try:
            #     with open(os.path.join(path_folder, 'cf_trajectories_tmp.pkl'), 'rb') as f:
            #         data = pickle.load(f)
            # except:
            #     data = []
            # data.append((traj_cf, mcts_rewards_cf))
            # with open(os.path.join(path_folder, 'cf_trajectories_tmp.pkl'), 'wb') as f:
            #     pickle.dump(data, f)
            # try:
            #     with open(os.path.join(path_folder, 'org_trajectories_tmp.pkl'), 'rb') as f:
            #         data = pickle.load(f)
            # except:
            #     data = []
            # data.append((traj_org, mcts_rewards))
            # print(len(data))
            # with open(os.path.join(path_folder, 'org_trajectories_tmp.pkl'), 'wb') as f:
            #     pickle.dump(data, f)
            # try:
            #     with open(os.path.join(path_folder, 'start_points_tmp.pkl'), 'rb') as f:
            #         data = pickle.load(f)
            # except:
            #     data = []
            # data.append(traj_start)
            # print(len(data))
            # with open(os.path.join(path_folder, 'start_points_tmp.pkl'), 'wb') as f:
            #     pickle.dump(data, f)


        if config.cf_method == 'step':
            # Method 2: 1-step deviation
            counterfactual_trajs, counterfactual_rewards, starts, end_cfs, end_orgs = generate_counterfactual_step(org_traj, ppo, discriminator, seed_env, config, conf=conf)
            if not baseline:
                # use the quality criteria to determine the best counterfactual trajectory
                sort_index, qc_stats = measure_quality(org_traj, counterfactual_trajs, counterfactual_rewards, starts, end_cfs, end_orgs, ppo, all_org_trajs, all_cf_trajs, all_starts, config.criteria, weights=weight)
                qc_statistics.append(qc_stats)
            else:
                # use a random baseline to determine the best counterfactual trajectory
                sort_index = random.randint(0, len(counterfactual_trajs)-1)

            chosen_counterfactual_trajectory = counterfactual_trajs[sort_index]
            traj_start = starts[sort_index]
            chosen_end_cf = end_cfs[sort_index]
            chosen_end_org = end_orgs[sort_index]

            efficiency = time.time() - time_start

            traj_org = partial_trajectory(org_traj, traj_start, chosen_end_org)
            step_rewards = sum(traj_org['rewards'])
            all_part_orgs.append((traj_org, step_rewards))

            traj_cf = partial_trajectory(chosen_counterfactual_trajectory, traj_start, chosen_end_cf)
            step_rewards_cf = sum(traj_cf['rewards'])
            all_part_cfs.append((traj_cf, step_rewards_cf))

            # print(len(traj_org['states']), len(traj_org['actions']))
            # visualize_two_part_trajectories_part(traj_org, traj_cf)

            # # Method 3: Random
            # random_org, random_cf, random_start = generate_counterfactual_random(org_traj, ppo, discriminator, seed_env, config)

        # uncomment below if the trajectories should be visualized:
        # visualize_two_part_trajectories(org_traj, chosen_counterfactual_trajectory, chosen_start, chosen_end_cf,  chosen_end_org)



        if config.measure_statistics:
            # record stastics
            lengths_org.append(len(traj_org['states']))
            lengths_cf.append(len(traj_cf['states']))
            start_points.append(traj_start)
            chosen_val, chosen_prox, chosen_crit, chosen_dive, chosen_real, chosen_spar = evaluate_qcs_for_cte(traj_org, traj_cf, traj_start, ppo, all_org_trajs, all_cf_trajs, all_starts)
            quality_criteria.append((chosen_val, chosen_prox, chosen_crit, chosen_dive, chosen_real, chosen_spar))
            effiencies.append(efficiency)

        # add the original trajectory and the counterfactual trajectory to the list of all trajectories
        all_org_trajs.append(traj_org)
        all_cf_trajs.append(traj_cf)
        all_starts.append(traj_start)
        print(len(all_starts))


    print('avg length org: ', np.mean(lengths_org))
    print('avg length cf: ', np.mean(lengths_cf))
    print('avg start point: ', np.mean(start_points))
    print('avg quality criteria: ', np.mean(quality_criteria, axis=0))
    print('avg generation time: ', np.mean(effiencies))

    if not os.path.exists(path_folder):
        os.makedirs(path_folder)

    #save the trajectories
    with open(os.path.join(path_folder, 'org_trajectories.pkl'), 'wb') as f:
        pickle.dump(all_part_orgs, f)
    with open(os.path.join(path_folder, 'cf_trajectories.pkl'), 'wb') as f:
        pickle.dump(all_part_cfs, f)

    if config.measure_statistics:
        # saving statistics
        # check if folder statistics exists
        if not os.path.exists(os.path.join(path_folder, 'statistics')):
            os.makedirs(os.path.join(path_folder, 'statistics'))

        with open(os.path.join(path_folder, 'statistics', 'lengths_org.pkl'), 'wb') as f:
            pickle.dump(lengths_org, f)
        with open(os.path.join(path_folder, 'statistics', 'lengths_cf.pkl'), 'wb') as f:
            pickle.dump(lengths_cf, f)
        with open(os.path.join(path_folder, 'statistics', 'start_points.pkl'), 'wb') as f:
            pickle.dump(start_points, f)
        with open(os.path.join(path_folder, 'statistics', 'quality_criteria.pkl'), 'wb') as f:
            pickle.dump(quality_criteria, f)
        with open(os.path.join(path_folder, 'statistics', 'effiencies.pkl'), 'wb') as f:
            pickle.dump(effiencies, f)
        with open(os.path.join(path_folder, 'statistics', 'qc_statistics.pkl'), 'wb') as f:
            pickle.dump(qc_statistics, f)