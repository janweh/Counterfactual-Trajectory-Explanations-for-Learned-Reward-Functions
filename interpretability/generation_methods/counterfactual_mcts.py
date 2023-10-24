import sys
import os
from pathlib import Path
adjacent_folder = Path(__file__).parent.parent
sys.path.append(str(adjacent_folder))
from tqdm import tqdm
from moral.ppo import PPO, TrajectoryDataset, update_policy
import torch
from moral.airl import *
from moral.active_learning import *
import numpy as np
from envs.gym_wrapper import *
from moral.preference_giver import *
import sys
from copy import *
from helpers.visualize_trajectory import visualize_two_part_trajectories
from helpers.util_functions import *
import random 
import time
from quality_metrics.quality_metrics import measure_quality, evaluate_qcs_for_cte, evaluate_qc
from quality_metrics.distance_measures import distance_all as distance_all
from quality_metrics.critical_state_measures import critical_state_all as critical_state
from quality_metrics.diversity_measures import diversity_single
import pickle
from helpers.parsing import sort_args, parse_attributes
from collections import defaultdict
from helpers.util_functions import extract_player_position
from moral.ppo import PPO
from torch.distributions import Categorical
from helpers.util_functions import partial_trajectory
from envs.randomized_v2_reimplementation import step_v2, MAX_STEPS


class settings:
    # 0-8 are the normal actions. 9 is an action which goes to the terminal state
    set_of_actions = [0,1,2,3,4,5,6,7,8]
    action_threshold = 0.003
    likelihood_terminal = 0.2
    discount_factor = 1.0
    qc_criteria_to_use = ['proximity', 'sparsity', 'validity', 'realisticness', 'diversity']
    num_iterations = 10
    starting_points = 70
    branching_heuristic = True
    policy_simulation = True
    num_simulations = 1

def generate_counterfactual_mcts(org_traj, ppo, discriminator, seed_env, prev_org_trajs, prev_cf_trajs, prev_starts, config, weights=None, conf=None):
    if not conf is None:
        settings.discount_factor = conf['discount_factor']
        settings.starting_points = conf['starting_points']
        settings.num_iterations = conf['num_iterations']
        settings.likelihood_terminal = conf['likelihood_terminal']
        settings.branching_heuristic = conf['branching_heuristic']
        settings.policy_simulation = conf['policy_simulation']
        settings.action_threshold = conf['action_threshold']
        settings.num_simulations = conf['num_simulations']
    
    critical_states = critical_state(ppo, org_traj['states'][:-1])
    # get the index of the 5 states with the highest critical state
    critical_states = [(i,j) for i,j in zip(critical_states, range(len(critical_states)))]
    critical_states.sort(key=lambda x: x[0], reverse=True)
    if settings.starting_points!=70:
        critical_states = critical_states[:settings.starting_points]

    part_orgs, part_cfs, q_values, qc_values = [], [], [], []
    for (i, starting_position) in critical_states:
        if starting_position < 72:
            part_org, part_cf, q_value = run_mcts_from(org_traj, starting_position, ppo, discriminator, seed_env, prev_org_trajs, prev_cf_trajs, prev_starts, config, weights)
            part_orgs.append(part_org)
            part_cfs.append(part_cf)
            q_values.append(q_value)
            qc_value = evaluate_qc(part_org, part_cf, starting_position, settings.qc_criteria_to_use + ['critical_state'], prev_org_trajs, prev_cf_trajs, prev_starts, ppo, weights)
            qc_values.append(qc_value)

    # select the best counterfactual trajectory
    sort_index_q = np.argmax(q_values)
    sort_index_qc = np.argmax(qc_values)
    chosen_part_org = part_orgs[sort_index_qc]
    chosen_part_cf = part_cfs[sort_index_qc]
    chosen_start = critical_states[sort_index_qc][1]
    return chosen_part_org, chosen_part_cf, chosen_start

def run_mcts_from(org_traj, starting_position, ppo, discriminator, seed_env, prev_org_trajs, prev_cf_trajs, prev_starts, config, weights=None):
    chosen_action = -1
    done = False
    root_node = None
    cf_trajectory = []
    qfunction = QTable()
    num_step = starting_position+1

    i = 0
    while chosen_action!=9 and num_step < MAX_STEPS-2:
        mdp = MDP(discriminator, ppo, starting_position, org_traj, prev_org_trajs, prev_cf_trajs, prev_starts, weights)
        root_node = MCTS(mdp, qfunction, UpperConfidenceBounds(), ppo, num_step).mcts(root_node=root_node)

        # choose the child node with the highest q value
        max_val = -np.inf
        string = ""
        for action in root_node.children:
            string += str(action) + ": " + str(qfunction.get_q_value(root_node.trajectory['actions'], action)) + "; "
            value = qfunction.get_q_value(root_node.trajectory['actions'], action)
            if value > max_val:
                max_val = value
                chosen_action = action

        # print("Depth:", i, "; Chosen action:", chosen_action, string)
        # This try statement is a bandaid for a bug, where the root_node.children can somtimes not contain the chosen action. In that case I currently just skip this step and rerun MCTS for the current root_node
        cf_trajectory.append(chosen_action)
        for (child, _) in root_node.children[chosen_action]:
            if cf_trajectory == child.trajectory['actions']:
                root_node = child
        i +=1
        num_step += 1
        
        # action = qfunction.get_max_q(root_Node.trajectory, mdp.get_actions(root_Node.state))[0]
        # cf_trajectory.append(action)
        # (next_state, reward, done) = mdp.execute(action)

    if 9 in cf_trajectory:
        cf_trajectory.remove(9)

    full_trajectory = {'states': [], 'actions': [], 'rewards': []}
    vec_env = VecEnv(config.env_id, config.n_workers, seed=seed_env)
    states = vec_env.reset()
    states_tensor = torch.tensor(states).float().to(device)
    full_trajectory, states_tensor = retrace_original(0, starting_position, full_trajectory, org_traj, vec_env, states_tensor, discriminator)
    cf_traj = {'states': [states_tensor], 'actions': [], 'rewards': [discriminator.g(states_tensor)[0][0].item()]}
    for action in cf_trajectory:
        cf_traj['actions'].append(action)
        state, reward, done, info = vec_env.step([action])
        states_tensor = torch.tensor(state).float().to(device)
        cf_traj['states'].append(states_tensor)
        cf_traj['rewards'].append(discriminator.g(states_tensor)[0][0].item())
    org_traj = partial_trajectory(org_traj, starting_position, len(cf_traj['states'])-1+starting_position)
    org_traj['actions'] = org_traj['actions'][:-1]
    
    return org_traj, cf_traj, max_val


        

def get_viable_actions(state, ppo):
    # TODO: if it's the first state of the trajectory, it has to be a different action than the original action
    viable_actions = []
    actions, log_probs = ppo.act(state)
    for i in range(len(actions)):
        # only include actions above the threshold of ppo loglikelihood
        if log_probs[i] > settings.action_threshold:
            # divied by 1-settings.likelihood_terminal to account for the fact that the terminal state is not included in the ppo policy. Now likelihoods sum of to 1 again :)
            viable_actions.append(actions[i], log_probs[i]*(1-settings.likelihood_terminal))
    viable_actions.append(9, settings.likelihood_terminal)
    return viable_actions

class MDP():

    def __init__(self, discriminator, ppo, starting_step, original_trajectory, prev_org_trajs, prev_cf_trajs, prev_starts, weights=None):
        self.discriminator = discriminator
        self.ppo = ppo
        self.starting_step = starting_step
        self.original_trajectory = original_trajectory
        self.prev_org_trajs = prev_org_trajs
        self.prev_cf_trajs = prev_cf_trajs
        self.prev_starts = prev_starts
        self.weights = weights

    def execute(self, trajectory, action, num_step):
        state = trajectory['states'][-1]
        if action == 9:
            next_state, done, num_step = step_v2(state, action, num_step)
        else:
            next_state, done, num_step = step_v2(state, action, num_step)
            state_tensor = next_state.clone().detach()
            trajectory['states'].append(state_tensor)
            trajectory['actions'].append(action)
            trajectory['rewards'].append(self.discriminator.g(state_tensor)[0][0].item())
            reward = 0
        if done:
            reward = self.get_reward(trajectory)
            if len(trajectory['states']) <= 2:
                a=0
        return trajectory, reward, done, num_step

        
    def get_reward(self, trajectory):
        org_traj = partial_trajectory(self.original_trajectory, self.starting_step, len(trajectory['states'])-1+self.starting_step)
        if len(org_traj['states']) != len(trajectory['states']):
            a=0
        return evaluate_qc(org_traj, trajectory, self.starting_step, settings.qc_criteria_to_use, self.prev_org_trajs, self.prev_cf_trajs, self.prev_starts, self.ppo, self.weights)
    

    def get_actions(self, state, traj_length):
        valid_actions = []
        actions = self.ppo.action_probabilities(state)
        act_tresh = settings.action_threshold
        # the first action of the coutnerfactual trajectory should not be the same as the first action of the original trajectory 
        if traj_length == 0:
            org_action = self.original_trajectory['actions'][self.starting_step]
            actions[org_action] = 0
        while len(valid_actions) < 3:
            valid_actions = []
            for action in settings.set_of_actions:
                if actions[action] > act_tresh:
                    valid_actions.append(action)
            act_tresh = act_tresh * 0.1
        # the first action cannot be 9, since this would immediately end the counterfactual without making a difference
        if traj_length > 0:
            valid_actions.append(9)
        return valid_actions
    
    def get_initial_state(self):
        return self.original_trajectory['states'][self.starting_step]

    def is_terminal(self, traj, num_step):
        if len(traj) >= 1:
            return traj[-1]==9 or num_step >= MAX_STEPS-2
        return False
    

class QTable():
    # stores the values assigned to trajectory-action pairs
    # a trajectory is represented as the sequence of actions from the deviation until a point

    def __init__(self, default=0.0):
        self.trajectory_action_values = defaultdict(lambda: default)

    def get_q_value(self, action_sequence, action):
        return self.trajectory_action_values[(tuple(action_sequence), action)]
    
    def get_max_q(self, action_trajectory, actions):
        max_q_value = -np.inf
        max_action = None
        for action in actions:
            q_value = self.get_q_value(action_trajectory, action)
            if q_value > max_q_value:
                max_q_value = q_value
                max_action = action
        return (max_action, max_q_value)

    def update(self, action_trajectory, action, delta):
        self.trajectory_action_values[(tuple(action_trajectory), action)] = self.trajectory_action_values[(tuple(action_trajectory), action)] + delta

class Node():
    
    # Record a unique node id to distinguish duplicated states
    next_node_id = 0

    # Records the number of times states have been visited
    visits = defaultdict(lambda: 0)

    def __init__(self, mdp, parent, state, trajectory, qfunction, bandit, num_step, action=None):
        self.mdp = mdp
        self.parent = parent
        self.state = state
        self.num_step = num_step
        self.trajectory = trajectory
        self.id = Node.next_node_id
        Node.next_node_id += 1

        # The Q function used to store state-action values
        self.qfunction = qfunction

        # A multi-armed bandit for this node
        self.bandit = bandit

        # The action that generated this node
        self.action = action

        # A dictionary from actions to a set of node-probability pairs
        self.children = {}

    """ Return the value of this node """
    def get_value(self):
        (_, max_q_value) = self.qfunction.get_max_q(
            self.trajectory['actions'], self.mdp.get_actions(self.state, len(self.trajectory['actions']))
        )
        return max_q_value
    
    """ Get the number of visits to this state """
    def get_visits(self):
        return Node.visits[tuple(self.trajectory['actions'])]
    
    """ Return true if and only if all child actions have been expanded """
    # TODO: exclude actions that are below a threshold according to the ppo policy
    def is_fully_expanded(self):
        valid_actions = self.mdp.get_actions(self.state, len(self.trajectory['actions']))  - self.children.keys()
        if len(valid_actions) == 0:
            return True
        else:
            return False
        
    """ Select a node that is not fully expanded """
    def select(self):
        if not self.is_fully_expanded() or self.mdp.is_terminal(self.trajectory['actions'], self.num_step):
            return self
        else:
            actions = list(self.children.keys())
            action = self.bandit.select(self.trajectory['actions'], actions, self.qfunction)
            outcome = self.get_outcome_child(action)
            recsel = outcome.select()
            return recsel

    """ Expand a node if it is not a terminal node """
    def expand(self):
        if not self.mdp.is_terminal(self.trajectory['actions'], self.num_step):
            # Randomly select an unexpanded action to expand
            actions = self.mdp.get_actions(self.state, len(self.trajectory['actions'])) - self.children.keys()

            if settings.branching_heuristic:
                # with heuristic
                max_qc = -np.inf
                action = -1
                for a in actions:
                    traj = deepcopy(self.trajectory)
                    (traj, reward, done, n_step) = self.mdp.execute(traj, a, self.num_step)
                    qc = self.mdp.get_reward(traj)
                    if qc > max_qc:
                        max_qc = qc
                        action = a
            else:
                # without heuristic
                action = random.choice(list(actions))

            self.children[action] = []
            outcome = self.get_outcome_child(action)

            return outcome
        return self, 0, 0
    
    """ Backpropogate the reward back to the parent node """
    def back_propagate(self, reward, child):
        action = child.action

        Node.visits[tuple(self.trajectory['actions'])] = Node.visits[tuple(self.trajectory['actions'])] + 1
        Node.visits[(tuple(self.trajectory['actions']), action)] = Node.visits[(tuple(self.trajectory['actions']), action)] + 1

        q_value = self.qfunction.get_q_value(self.trajectory['actions'], action)
        delta = (1 / (Node.visits[(tuple(self.trajectory['actions']), action)])) * (
            reward - q_value
        )
        self.qfunction.update(self.trajectory['actions'], action, delta)

        if self.parent != None:
            self.parent.back_propagate(reward, self)


    """ Simulate the outcome of an action, and return the child node """

    def get_outcome_child(self, action):
        # Choose one outcome based on transition probabilities
        traj = deepcopy(self.trajectory)
        (traj, reward, done, n_step) = self.mdp.execute(traj, action, self.num_step)

        # Find the corresponding state and return if this already exists
        for (child, _) in self.children[action]:
            if traj['actions'] == child.trajectory['actions']:
                return child

        # This outcome has not occured from this state-action pair previously
        next_state = traj['states'][-1].clone().detach()
        new_child = Node(
            self.mdp, self, next_state, traj, self.qfunction, self.bandit, n_step, action=action
        )

        # Find the probability of this outcome (only possible for model-based) for visualising tree
        self.children[action] += [(new_child, 1.0)]
        return new_child
        
    
class MCTS:
    def __init__(self, mdp, qfunction, bandit, ppo, num_step):
        self.mdp = mdp
        self.qfunction = qfunction
        self.bandit = bandit
        self.ppo = ppo
        self.num_step = num_step

    """
    Execute the MCTS algorithm from the initial state given, with timeout in seconds
    """
    def mcts(self, root_node=None):
        if root_node is None:
            root_node = self.create_root_node()
        else:
            root_node.mdp = self.mdp

        for _ in range(settings.num_iterations):
            # Find a state node to expand
            selected_node = root_node.select()
            if not self.mdp.is_terminal(selected_node.trajectory['actions'], selected_node.num_step):

                child = selected_node.expand()
                ttt = child.trajectory['actions']
                rewards = []
                for sims in range(settings.num_simulations):
                    reward = self.simulate(child)
                    rewards.append(reward)
                reward = np.mean(rewards)
                selected_node.back_propagate(reward, child)

        return root_node
    
    """ Choose a random action. Heustics can be used here to improve simulations. """
    def choose(self, state, num_steps):
        if settings.policy_simulation:
            # simulation according to the policy
            action_distribution = self.ppo.action_distribution_torch(state)
            # remove all actions below the threshold
            for i in range(action_distribution.shape[1]):
                if action_distribution[0][i] < settings.action_threshold:
                    action_distribution[0][i] = 0

            if num_steps > 0:
                action_distribution = torch.cat((action_distribution, torch.tensor([settings.likelihood_terminal]).view(1,1)), 1)
            m = Categorical(action_distribution)
            action = m.sample()
            action = action.item()

        else:
            # random simulation
            # make an even probability distribution over all actions, except for the terminal action which has a likelihood of settings.likelihood_terminal
            action_distribution = np.ones(len([0,1,2,3,4,5,6,7,8]))
            action_distribution = action_distribution * (1-settings.likelihood_terminal) / len([0,1,2,3,4,5,6,7,8])
            action_distribution = np.append(action_distribution, settings.likelihood_terminal)
            action_distribution = np.array(action_distribution)
            action_distribution = action_distribution / np.sum(action_distribution)
            action = np.random.choice([0,1,2,3,4,5,6,7,8,9], 1, p=action_distribution)[0]

        return action
    
    """ Simulate until a terminal state """
    # TODO: change how the reward is calculated to be based on quality criteria
    # TODO: change how the mdp is being simualted to be in the actual environment
    def simulate(self, node):
        trajectory = deepcopy(node.trajectory)
        state = deepcopy(node.state)
        num_step = copy(node.num_step)
        cumulative_reward = 0.0
        depth = 0
        done = False
        if node.action==9 or num_step >= MAX_STEPS -2:
            cumulative_reward = self.mdp.get_reward(trajectory)
        while not self.mdp.is_terminal(trajectory['actions'], num_step) and not done:
            # Choose an action to execute
            action = self.choose(state, len(trajectory['actions']))

            if len(trajectory['states'])-1+self.mdp.starting_step >= 73:
                a=0
            # Execute the action
            (trajectory, reward, done, num_step) = self.mdp.execute(trajectory, action, num_step)

            # Discount the reward
            cumulative_reward += pow(settings.discount_factor, depth) * reward
            depth += 1

            state = trajectory['states'][-1].clone().detach()
        return cumulative_reward
    
    def create_root_node(self):
        first_state = self.mdp.get_initial_state()
        first_reward = self.mdp.discriminator.g(first_state)[0][0].item()
        return Node(
            self.mdp, None, first_state, {'states': [first_state], 'actions': [], 'rewards': [first_reward]}, self.qfunction, self.bandit, self.num_step
        )

class UpperConfidenceBounds():
    def __init__(self):
        self.total = 0
        # number of times each action has been chosen
        self.times_selected = {}

    def select(self, action_sequence, actions, qfunction):

        # First execute each action one time
        for action in actions:
            if action not in self.times_selected.keys():
                self.times_selected[action] = 1
                self.total += 1
                return action

        max_actions = []
        max_value = float("-inf")
        # remove the item 9 from the actions if it exists. This avoids resampling the action 9. 9 does not have to be resampled, because there is not tree of actions below it
        if 9 in actions:
            actions.remove(9)
        for action in actions:
            value = qfunction.get_q_value(action_sequence, action) + math.sqrt(
                (2 * math.log(self.total)) / self.times_selected[action]
            )
            if value > max_value:
                max_actions = [action]
                max_value = value
            elif value == max_value:
                max_actions += [action]

        # if there are multiple actions with the highest value
        # choose one randomly
        result = random.choice(max_actions)
        self.times_selected[result] = self.times_selected[result] + 1
        self.total += 1
        return result
    
    """ Reset a multi-armed bandit to its initial configuration """
    def reset(self):
        self.__init__()
