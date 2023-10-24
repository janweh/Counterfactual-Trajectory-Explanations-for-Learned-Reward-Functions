import sys
from pathlib import Path
adjacent_folder = Path(__file__).parent.parent
sys.path.append(str(adjacent_folder))
import torch
import os
import pickle
import time
import numpy as np
import argparse
from evaluate_mimic import evaluate_mimic
import sys
import random
from helpers.folder_util_functions import iterate_through_folder, save_results, read, write
from copy import deepcopy
from utils_evaluation import *

class hyperparameters:
    learning_rate = 1e-1
    regularisation = 1e-2
    l1_lambda = 1e-1
    epochs_non_contrastive = 10000
    epochs_contrastive = 2000
    number_of_seeds = 10

class LM_params:
    learning_rate = 0.1
    regularisation = 0.01
    epochs = 1893

class NN_params:
    # learning_rate = 0.01
    # regularisation = 0.01
    # num_layers = 4
    # hidden_layer_sizes = [8,4]

    ## for step
    # learning_rate = 0.03
    # regularisation = 0.001
    # num_layers = 5
    # hidden_layer_sizes = [60, 60, 30]
    # epochs = 4512
    
    ## for MCTS
    learning_rate = 0.03
    regularisation = 0.001
    num_layers = 5
    hidden_layer_sizes = [92, 92, 46]
    epochs = 15569

class config:    
    features = ['citizens_saved', 'unsaved_citizens', 'distance_to_citizen', 'standing_on_extinguisher', 'length', 'could_have_saved', 'final_number_of_unsaved_citizens', 'moved_towards_closest_citizen', 'bias']
    model_type = 'linear' # model_type = 'NN' or 'linear' or 'stepwise'
    data_folds = 5
    results_path = "results" # Foldername to save res  ults to
    print_plot = False
    print_examples = False
    print_weights = False
    save_results = False
    print_worst_examples = False
    print_best_examples = False
    save_model = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# randomises the order of trajectories, while keeping the pairs of original and counterfactual trajectories together
# also makes them into tensors
def shuffle_together(org_trajs, cf_trajs):
    org_cf_trajs = list(zip(org_trajs, cf_trajs))
    random.shuffle(org_cf_trajs)
    org_trajs, cf_trajs = zip(*org_cf_trajs)
    org_trajs = torch.tensor(org_trajs, dtype=torch.float)
    cf_trajs = torch.tensor(cf_trajs, dtype=torch.float)
    return org_trajs, cf_trajs

def train_test_split(org_trajs, cf_trajs, num_features, train_ratio=0.8):
    # randomise the order of the trajectories
    org_trajs = np.random.permutation(org_trajs)
    cf_trajs = np.random.permutation(cf_trajs)
    org_trajs = torch.tensor(org_trajs, dtype=torch.float)
    cf_trajs = torch.tensor(cf_trajs, dtype=torch.float)
    n_train = int(train_ratio * len(org_trajs))

    train = torch.cat((org_trajs[:n_train,:num_features], cf_trajs[:n_train,:num_features]), dim=0)
    train_labels = torch.cat((org_trajs[:n_train,-1], cf_trajs[:n_train,-1]), dim=0)
    test = torch.cat((org_trajs[n_train:,:num_features], cf_trajs[n_train:,:num_features]), dim=0)
    test_labels = torch.cat((org_trajs[n_train:,-1], cf_trajs[n_train:,-1]), dim=0)

    return train, train_labels, test, test_labels

def train_test_split_contrastive(org_trajs, cf_trajs, num_features, train_ratio=0.8):
    n_train = int(train_ratio * len(org_trajs))

    org_trajs, cf_trajs = shuffle_together(org_trajs, cf_trajs)

    train = org_trajs[:n_train,:num_features] - cf_trajs[:n_train,:num_features]
    train_labels = org_trajs[:n_train,-1] - cf_trajs[:n_train,-1]
    test = org_trajs[n_train:,:num_features] - cf_trajs[n_train:,:num_features]
    test_labels = org_trajs[n_train:,-1] - cf_trajs[n_train:,-1]

    return train, train_labels, test, test_labels

def train_test_split_contrastive_sidebyside(org_trajs, cf_trajs, num_features, n_train):
    # train = org_trajs[:n_train,:num_features] - cf_trajs[:n_train,:num_features]
    # combine the orginal and counterfactual trajectory features into one feature vector
    train = torch.stack((org_trajs[:n_train,:num_features], cf_trajs[:n_train,:num_features]), dim=1).view(n_train, num_features*2)
    train_labels = org_trajs[:n_train,-1] - cf_trajs[:n_train,-1]
    test = torch.stack((org_trajs[n_train:,:num_features], cf_trajs[n_train:,:num_features]), dim=1).view(len(org_trajs) - n_train, num_features*2)
    # test = org_trajs[n_train:,:num_features] - cf_trajs[n_train:,:num_features]
    test_labels = org_trajs[n_train:,-1] - cf_trajs[n_train:,-1]

    return train, train_labels, test, test_labels 

def train_test_split_single_sidebyside(org_trajs, cf_trajs, num_features, n_train):
    # combine the orginal and counterfactual trajectory features into one feature vector
    train = torch.stack((org_trajs[:n_train,:num_features], cf_trajs[:n_train,:num_features]), dim=1).view(n_train, num_features*2)
    # put the labels in pairs to a shape (800,2)
    train_labels = torch.stack((org_trajs[:n_train,-1], cf_trajs[:n_train,-1]), dim=1)
    test = torch.stack((org_trajs[n_train:,:num_features], cf_trajs[n_train:,:num_features]), dim=1).view(len(org_trajs) - n_train, num_features*2)
    test_labels = torch.stack((org_trajs[n_train:,-1], cf_trajs[n_train:,-1]), dim=1)

    return train, train_labels, test, test_labels 

def train_validation_test_split_contrastive(org_trajs, cf_trajs, num_features, train_ratio=0.6, validation_ratio=0.2):
    n_train = int(train_ratio * len(org_trajs))
    n_validation = int(validation_ratio * len(org_trajs))

    org_trajs, cf_trajs = shuffle_together(org_trajs, cf_trajs)

    train = org_trajs[:n_train,:num_features] - cf_trajs[:n_train,:num_features]
    train_labels = org_trajs[:n_train,-1] - cf_trajs[:n_train,-1]
    validation = org_trajs[n_train:n_train+n_validation,:num_features] - cf_trajs[n_train:n_train+n_validation,:num_features]
    validation_labels = org_trajs[n_train:n_train+n_validation,-1] - cf_trajs[n_train:n_train+n_validation,-1]
    test = org_trajs[n_train+n_validation:,:num_features] - cf_trajs[n_train+n_validation:,:num_features]
    test_labels = org_trajs[n_train+n_validation:,-1] - cf_trajs[n_train+n_validation:,-1]

    return train, train_labels, validation, validation_labels, test, test_labels

def train_model_LM(train_set, train_labels, test_set_con, test_labels_con, train_set_sin=torch.tensor([]), train_labels_sin=torch.tensor([]), test_set_sin=None, test_labels_sin=None, num_features=46, epochs = hyperparameters.epochs_contrastive, learning_rate=hyperparameters.learning_rate, regularisation = hyperparameters.regularisation, num_layers = None, hidden_layer_sizes = None, base_path=None, l2=None, stop_epoch = 0, task_weights=None):
    model_sin = torch.nn.Linear(num_features, 2)
    model_con = torch.nn.Linear(num_features, 1)

    train_set = train_set.to(device)
    train_labels = train_labels.to(device)
    test_set_con = test_set_con.to(device)
    test_labels_con = test_labels_con.to(device)
    if train_set_sin.shape[0] != 0:
        train_set_sin = train_set_sin.to(device)
        train_labels_sin = train_labels_sin.to(device)
        test_set_sin = test_set_sin.to(device)
        test_labels_sin = test_labels_sin.to(device)

    loss_fn = torch.nn.MSELoss(reduction='mean')
    optimizer_sin = torch.optim.Adam(model_sin.parameters(), lr=learning_rate, weight_decay=regularisation)
    optimizer_con = torch.optim.Adam(model_con.parameters(), lr=learning_rate, weight_decay=regularisation)

    train_losses, train_losses_sin, weighted_train_losses, test_losses_con, test_losses_sin, stop_train_losses, stop_train_losses_sin, stop_test_losses_con, stop_test_losses_sin = [], [], [], [], [], [], [], [], []

    for t in range(epochs):
        # if we want to stop the training at a certain epoch, save the model and the losses, but continue training until the end
        if t == stop_epoch-1:
            stop_model_sin = deepcopy(model_sin)
            stop_model_con = deepcopy(model_con)
            # stop_base_model.add_module('linear' + str(num_layers-2))
            stop_train_losses = deepcopy(train_losses)
            stop_train_losses_sin = deepcopy(train_losses_sin)
            stop_test_losses_con = deepcopy(test_losses_con)
            stop_test_losses_sin = deepcopy(test_losses_sin)

        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model_con(train_set).squeeze()
        y_pred_sin = model_sin(train_set_sin).squeeze()

        # Compute and print loss.
        loss_con = loss_fn(y_pred, train_labels)
        loss_sin = loss_fn(y_pred_sin, train_labels_sin)
        if train_labels_sin.shape[0] == 0: loss_sin = torch.tensor(0.0)
        if train_labels.shape[0] == 0: loss_con = torch.tensor(0.0)

        train_losses.append(loss_con.item())
        train_losses_sin.append(loss_sin.item())

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable weights of the model)
        optimizer_sin.zero_grad()
        optimizer_con.zero_grad()
        # Backward pass: compute gradient of the loss with respect to model parameters
        loss_con.backward()
        loss_sin.backward()
        # Calling the step function on an Optimizer makes an update to its parameters
        optimizer_sin.step()
        optimizer_con.step()

        # record the loss at this epoch for the test set
        with torch.no_grad():
            model_sin.eval()
            model_con.eval()
            y_pred_test = model_con(test_set_con).squeeze()
            y_pred_test_sin = model_sin(test_set_sin).squeeze()
            loss = loss_fn(y_pred_test, test_labels_con)
            loss_sin = loss_fn(y_pred_test_sin, test_labels_sin)
            if test_labels_sin.shape[0] == 0: loss_sin = torch.tensor(0.0)
            if test_labels_con.shape[0] == 0: loss = torch.tensor(0.0)
            test_losses_con.append(loss.item())
            test_losses_sin.append(loss_sin.item())
            model_sin.train()
            model_con.train()
    
    if stop_epoch != 0:
        stop_model_sin.eval()
        stop_model_con.eval()
        if test_set_sin != None:
            return stop_model_con, stop_model_sin, train_losses, train_losses_sin, test_losses_con, test_losses_sin, stop_train_losses, stop_train_losses_sin, stop_test_losses_con, stop_test_losses_sin
        else:
            return stop_model_con, None, train_losses, train_losses_sin, test_losses_con, stop_train_losses, stop_train_losses_sin, stop_test_losses_con
    model_sin_full = model_sin
    model_con_full = model_con
    if test_set_sin != None:
        return model_con_full, model_sin_full, train_losses, train_losses_sin, test_losses_con, test_losses_sin, None, None, None
    else:
        return model_con_full, model_sin_full, train_losses, train_losses_sin, test_losses_con
    


def train_model(train_set, train_labels, test_set_con, test_labels_con, train_set_sin=torch.tensor([]), train_labels_sin=torch.tensor([]), test_set_sin=None, test_labels_sin=None, num_features=46, epochs = hyperparameters.epochs_contrastive, learning_rate=hyperparameters.learning_rate, regularisation = hyperparameters.regularisation, num_layers = None, hidden_layer_sizes = None, base_path=None, l2=None, stop_epoch = 0, task_weights=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: ', device)

    # Initialise the model (either NN or LM)
    if config.model_type=='NN':
        # make a list of the layer dimensions: num_features, hidden_layer_sizes, 1

        # Initialise the neural network with the number of layers and the hidden_sizes
        layers = [num_features] + hidden_layer_sizes + [1]
        base_model = torch.nn.Sequential()
        base_model.add_module('linear' + str(0), torch.nn.Linear(layers[0], layers[1]))
        for i in range(1, num_layers-2):
            base_model.add_module('relu' + str(i-1), torch.nn.ReLU())
            base_model.add_module('linear' + str(i), torch.nn.Linear(layers[i], layers[i+1]))
        base_model.add_module('relu' + str(num_layers-3), torch.nn.ReLU())
        model_sin = torch.nn.Linear(layers[-2], 2)
        model_con = torch.nn.Linear(layers[-2], 1)
        base_model = base_model.to(device)
        model_sin = model_sin.to(device)
        model_con = model_con.to(device)

    elif config.model_type=='linear':
        model = torch.nn.Linear(num_features, 1)
        model = model.to(device)

    train_set = train_set.to(device)
    train_labels = train_labels.to(device)
    test_set_con = test_set_con.to(device)
    test_labels_con = test_labels_con.to(device)
    if train_set_sin.shape[0] != 0:
        train_set_sin = train_set_sin.to(device)
        train_labels_sin = train_labels_sin.to(device)
        test_set_sin = test_set_sin.to(device)
        test_labels_sin = test_labels_sin.to(device)

    loss_fn = torch.nn.MSELoss(reduction='mean')
    optimizer_base = torch.optim.Adam(base_model.parameters(), lr=learning_rate, weight_decay=regularisation)
    optimizer_sin = torch.optim.Adam(model_sin.parameters(), lr=learning_rate, weight_decay=regularisation)
    optimizer_con = torch.optim.Adam(model_con.parameters(), lr=learning_rate, weight_decay=regularisation)

    train_losses, train_losses_sin, weighted_train_losses, test_losses_con, test_losses_sin, stop_train_losses, stop_train_losses_sin, stop_test_losses_con, stop_test_losses_sin = [], [], [], [], [], [], [], [], []

    for t in range(epochs):
        # if we want to stop the training at a certain epoch, save the model and the losses, but continue training until the end
        if t == stop_epoch-1:
            stop_model_sin = torch.nn.Sequential(deepcopy(base_model), deepcopy(model_sin))
            stop_model_con = torch.nn.Sequential(deepcopy(base_model), deepcopy(model_con))
            # stop_base_model.add_module('linear' + str(num_layers-2))
            stop_train_losses = deepcopy(train_losses)
            stop_train_losses_sin = deepcopy(train_losses_sin)
            stop_test_losses_con = deepcopy(test_losses_con)
            stop_test_losses_sin = deepcopy(test_losses_sin)

        # Forward pass: compute predicted y by passing x to the model.
        base_pred = base_model(train_set)
        y_pred = model_con(base_pred).squeeze()
        base_pred_sin = base_model(train_set_sin)
        y_pred_sin = model_sin(base_pred_sin).squeeze()

        # Compute and print loss.
        loss_con = loss_fn(y_pred, train_labels)
        loss_sin = loss_fn(y_pred_sin, train_labels_sin)
        if train_labels_sin.shape[0] == 0: loss_sin = torch.tensor(0.0)
        if train_labels.shape[0] == 0: loss_con = torch.tensor(0.0)

        train_losses.append(loss_con.item()/task_weights[0])
        train_losses_sin.append(loss_sin.item()/task_weights[1])
        weighted_loss = loss_con/task_weights[0] + loss_sin/task_weights[1]
        weighted_train_losses.append(weighted_loss.item())

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable weights of the model)
        optimizer_sin.zero_grad()
        optimizer_con.zero_grad()
        optimizer_base.zero_grad()
        # Backward pass: compute gradient of the loss with respect to model parameters
        weighted_loss.backward()
        # Calling the step function on an Optimizer makes an update to its parameters
        optimizer_sin.step()
        optimizer_con.step()
        optimizer_base.step()

        # record the loss at this epoch for the test set
        with torch.no_grad():
            base_model.eval()
            model_sin.eval()
            model_con.eval()
            base_pred = base_model(test_set_con)
            y_pred_test = model_con(base_pred).squeeze()
            base_pred_sin = base_model(test_set_sin)
            y_pred_test_sin = model_sin(base_pred_sin).squeeze()
            loss = loss_fn(y_pred_test, test_labels_con)
            loss_sin = loss_fn(y_pred_test_sin, test_labels_sin)
            if test_labels_sin.shape[0] == 0: loss_sin = torch.tensor(0.0)
            if test_labels_con.shape[0] == 0: loss = torch.tensor(0.0)
            test_losses_con.append(loss.item())
            test_losses_sin.append(loss_sin.item())
            base_model.train()
            model_sin.train()
            model_con.train()
    
    if stop_epoch != 0:
        stop_model_sin.eval()
        stop_model_con.eval()
        if test_set_sin != None:
            return stop_model_con, stop_model_sin, train_losses, train_losses_sin, test_losses_con, test_losses_sin, stop_train_losses, stop_train_losses_sin, stop_test_losses_con, stop_test_losses_sin
        else:
            return stop_model_con, None, train_losses, train_losses_sin, test_losses_con, stop_train_losses, stop_train_losses_sin, stop_test_losses_con
    model_sin_full = torch.nn.Sequential(base_model, model_sin)
    model_con_full = torch.nn.Sequential(base_model, model_con)
    if test_set_sin != None:
        return model_con_full, model_sin_full, train_losses, train_losses_sin, test_losses_con, test_losses_sin, None, None, None
    else:
        return model_con_full, model_sin_full, train_losses, train_losses_sin, test_losses_con

def learning_repeats(path_org, path_cf, base_path, contrastive=True, baseline=0, n_train=800, task_weights=None):

    test_lossess_con, train_lossess, all_train_losses, test_mean_errors_con, test_rmses_con, test_r2s_con, train_mean_errors, train_rmses, pearson_correlations_con, spearman_correlations_con, weights, all_test_losses_con, epochss, learning_rates, regularisations = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    test_lossess_sin, test_mean_errors_sin, test_rmses_sin, test_r2s_sin, pearson_correlations_sin, spearman_correlations_sin, all_test_losses_sin = [], [], [], [], [], [], []
    test_loss_oods_con, test_mean_error_oods_con, test_rmse_oods_con, r2_oods_con, pearson_correlation_oods_con, spearman_correlation_oods_con, pred_label_pairs_oods_con = [], [], [], [], [], [], []
    test_loss_oods_sin, test_mean_error_oods_sin, test_rmse_oods_sin, r2_oods_sin, pearson_correlation_oods_sin, spearman_correlation_oods_sin, pred_label_pairs_oods_sin = [], [], [], [], [], [], []

    # load ood test data
    # go up one folder from path
    path = os.path.dirname(base_path)

    num_layers = None
    hidden_sizes = None

    # load the data
    if os.path.isfile(os.path.join(base_path, 'data_split_con.pkl')):
        with open(os.path.join(base_path, 'data_split_con.pkl'), 'rb') as f:
            train_con, train_labels_con, test_con, test_labels_con = pickle.load(f)
        with open(os.path.join(base_path, 'data_split_sin.pkl'), 'rb') as f:
            train_sin, train_labels_sin, test_sin, test_labels_sin = pickle.load(f)
    else:
        with open(path_org, 'rb') as f:
            org_features = pickle.load(f)
        # org_features = pickle.load(open("datasets\\weights\\step\\weight0\\org_features_new.pkl", 'rb'))
        with open(path_cf, 'rb') as f:
            cf_features = pickle.load(f)

        org_features, cf_features = shuffle_together(org_features, cf_features)
        train_con, train_labels_con, test_con, test_labels_con = train_test_split_contrastive_sidebyside(org_features, cf_features, num_features, n_train=n_train)
        train_sin, train_labels_sin, test_sin, test_labels_sin = train_test_split_single_sidebyside(org_features, cf_features, num_features, n_train=n_train)

        # save the split of the data
        with open(os.path.join(base_path, 'data_split_con.pkl'), 'wb') as f:
            pickle.dump([train_con, train_labels_con, test_con, test_labels_con], f)
        with open(os.path.join(base_path, 'data_split_sinn.pkl'), 'wb') as f:
            pickle.dump([train_sin, train_labels_sin, test_sin, test_labels_sin], f)

    num_features = 46
    results_path = os.path.join(base_path, config.results_path + config.model_type, str(n_train))
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    with open(os.path.join(path, 'baseline', 'data_split_con.pkl'), 'rb') as f:
        _, _, test_set_ood_con, test_labels_ood_con = pickle.load(f)
    with open(os.path.join(path, 'baseline', 'data_split_sin.pkl'), 'rb') as f:
        _, _, test_set_ood_sin, test_labels_ood_sin = pickle.load(f)

    # epochs, learning_rate, regularisation, num_layers, hidden_sizes = 221, 0.3, 0.1, 4, [12,6]
    if config.model_type == 'NN':
        learning_rate, regularisation, num_layers, hidden_sizes, epochs = NN_params.learning_rate, NN_params.regularisation, NN_params.num_layers, NN_params.hidden_layer_sizes, NN_params.epochs
        # learning_rate, regularisation, num_layers, hidden_sizes = hyper_param_optimization_architecture(train_con, train_labels_con, train_sin, train_labels_sin, task_weights=task_weights)
        print(learning_rate, regularisation, num_layers, hidden_sizes)
    else:
        learning_rate, regularisation, epochs = hyperparameters.learning_rate, hyperparameters.regularisation, hyperparameters.epochs_contrastive

    # epochs = tune_epochs(train_con, train_labels_con, train_sin, train_labels_sin, learning_rate, regularisation, num_layers, hidden_sizes, task_weights=task_weights)
    

    for repeat in range(hyperparameters.number_of_seeds):
        start_time = time.time()
        # train the model (works for both, the linear and NN model)
        if config.model_type == 'NN':
            model_con, model_sin, full_train_losses, full_train_losses_sin, full_test_losses_con, full_test_losses_sin, train_losses, train_losses_sin, test_losses_con, test_losses_sin = train_model(train_con, train_labels_con, test_con, test_labels_con, train_sin, train_labels_sin, test_sin, test_labels_sin, num_features*2, epochs=hyperparameters.epochs_contrastive, stop_epoch=NN_params.epochs, learning_rate=learning_rate, regularisation=regularisation, num_layers=num_layers, hidden_layer_sizes=hidden_sizes, task_weights=task_weights)
        else:
            model_con, model_sin, full_train_losses, full_train_losses_sin, full_test_losses_con, full_test_losses_sin, train_losses, train_losses_sin, test_losses_con, test_losses_sin = train_model_LM(train_con, train_labels_con, test_con, test_labels_con, train_sin, train_labels_sin, test_sin, test_labels_sin, num_features*2, epochs=hyperparameters.epochs_contrastive, stop_epoch=LM_params.epochs, learning_rate=learning_rate, regularisation=regularisation, task_weights=task_weights)

        # here we test on the left out test set
        test_loss_con, test_mean_error_con, test_rmse_con, r2_con, pearson_correlation_con, spearman_correlation_con, pred_label_pairs_con = evaluate_mimic(model_con, test_con, test_labels_con, worst=config.print_worst_examples, best=config.print_best_examples, features=config.features)
        test_loss_sin, test_mean_error_sin, test_rmse_sin, r2_sin, pearson_correlation_sin, spearman_correlation_sin, pred_label_pairs_sin = evaluate_mimic(model_sin, test_sin, test_labels_sin, worst=config.print_worst_examples, best=config.print_best_examples, features=config.features)
        # here we test on a seperate test set from a different distribution
        test_loss_ood_con, test_mean_error_ood_con, test_rmse_ood_con, r2_ood_con, pearson_correlation_ood_con, spearman_correlation_ood_con, pred_label_pairs_ood_con = evaluate_mimic(model_con, test_set_ood_con, test_labels_ood_con, worst=config.print_worst_examples, best=config.print_best_examples, features=config.features)
        test_loss_ood_sin, test_mean_error_ood_sin, test_rmse_ood_sin, r2_ood_sin, pearson_correlation_ood_sin, spearman_correlation_ood_sin, pred_label_pairs_ood_sin = evaluate_mimic(model_sin, test_set_ood_sin, test_labels_ood_sin, worst=config.print_worst_examples, best=config.print_best_examples, features=config.features)

        test_lossess_con.append(test_loss_con)
        test_lossess_sin.append(test_loss_sin)
        test_mean_errors_con.append(test_mean_error_con)
        test_mean_errors_sin.append(test_mean_error_sin)
        test_rmses_con.append(test_rmse_con)
        test_rmses_sin.append(test_rmse_sin)
        test_r2s_con.append(r2_con)
        test_r2s_sin.append(r2_sin)
        pearson_correlations_con.append(pearson_correlation_con)
        pearson_correlations_sin.append(pearson_correlation_sin)
        spearman_correlations_con.append(spearman_correlation_con)
        spearman_correlations_sin.append(spearman_correlation_sin)
        train_lossess.append(train_losses[-1])
        all_train_losses.append(full_train_losses)
        all_test_losses_con.append(full_test_losses_con)
        all_test_losses_sin.append(full_test_losses_sin)

        test_loss_oods_con.append(test_loss_ood_con)
        test_loss_oods_sin.append(test_loss_ood_sin)
        test_mean_error_oods_con.append(test_mean_error_ood_con)
        test_mean_error_oods_sin.append(test_mean_error_ood_sin)
        test_rmse_oods_con.append(test_rmse_ood_con)
        test_rmse_oods_sin.append(test_rmse_ood_sin)
        r2_oods_con.append(r2_ood_con)
        r2_oods_sin.append(r2_ood_sin)
        pearson_correlation_oods_con.append(pearson_correlation_ood_con)
        pearson_correlation_oods_sin.append(pearson_correlation_ood_sin)
        spearman_correlation_oods_con.append(spearman_correlation_ood_con)
        spearman_correlation_oods_sin.append(spearman_correlation_ood_sin)
        pred_label_pairs_oods_con.append(pred_label_pairs_ood_con)
        pred_label_pairs_oods_sin.append(pred_label_pairs_ood_sin)
        print(test_mean_error_con, test_mean_error_ood_con)
        print(test_mean_error_sin, test_mean_error_ood_sin)

        print('repeat', repeat, 'time', time.time() - start_time)

        if config.save_model:
            path = os.path.join(results_path, 'saved_models')
            if not os.path.exists(path):
                os.makedirs(path)
            torch.save(model_sin.state_dict(), os.path.join(path, 'model_sin' + str(repeat) + '.pt'))
            torch.save(model_con.state_dict(), os.path.join(path, 'model_con' + str(repeat) + '.pt'))


    # print('final train_loss', np.mean(train_losses))
    print('test_loss', np.mean(test_lossess_con), np.mean(test_lossess_sin))
    print('test mean error', np.mean(test_mean_errors_con), np.mean(test_mean_errors_sin))
    print('test rmse', np.mean(test_rmses_con), np.mean(test_rmses_sin))
    print('test r2', np.mean(test_r2s_con), np.mean(test_r2s_sin))
    print('pearson correlation', np.mean(pearson_correlations_con), np.mean(pearson_correlations_sin))
    print('spearman correlation', np.mean(spearman_correlations_con), np.mean(spearman_correlations_sin))
    average_reward_con = torch.mean(train_labels_con)
    print('average reward contrastive', average_reward_con)
    average_reward_sin = torch.mean(train_labels_sin)
    print('average reward single', average_reward_sin)
    average_prediction = torch.mean(model_con(train_sin).squeeze())
    print('average prediction contrastive', average_prediction)
    average_prediction_sin = torch.mean(model_sin(train_sin).squeeze())
    print('average prediction single', average_prediction_sin)
    all_train_losses = np.mean(all_train_losses, axis=0)
    all_test_losses_con = np.mean(all_test_losses_con, axis=0)
    all_test_losses_sin = np.mean(all_test_losses_sin, axis=0)
    # show_loss_plot(all_train_losses, all_test_losses_con, show=config.print_plot, save_path=results_path, epochs=epochs, third_graph=all_test_losses_sin)
    # print('predicition label pairs', pred_label_pairs)

    print('test_loss_ood', np.mean(test_loss_oods_con), np.mean(test_loss_oods_sin))
    print('test mean error ood', np.mean(test_mean_error_oods_con), np.mean(test_mean_error_oods_sin))
    print('test rmse ood', np.mean(test_rmse_oods_con), np.mean(test_rmse_oods_sin))
    print('test r2 ood', np.mean(r2_oods_con), np.mean(r2_oods_sin))
    print('pearson correlation ood', np.mean(pearson_correlation_oods_con), np.mean(pearson_correlation_oods_sin))
    print('spearman correlation ood', np.mean(spearman_correlation_oods_con), np.mean(spearman_correlation_oods_sin))

    if config.save_results:
        to_save = {'train_losses': train_lossess, 'test_losses': test_lossess_con, 'test_mean_errors': test_mean_errors_con, 
                   'train_mean_errors': train_mean_errors, 'train_rmses': train_rmses, 'test_rmses': test_rmses_con, 'pearson_correlations': pearson_correlations_con, 'spearman_correlations': spearman_correlations_con, 
                   'weights': weights, 'all_train_losses': all_train_losses, 'all_test_losses': all_test_losses_con, 'average_reward': average_reward_con, 'average_prediction': average_prediction, 'r2s': test_r2s_con}
        save_results(to_save, results_path, baseline, type='results', data_mixture=(0,0), con=True)

        to_save = {'train_losses': train_lossess, 'test_losses': test_lossess_sin, 'test_mean_errors': test_mean_errors_sin, 
                   'train_mean_errors': train_mean_errors, 'train_rmses': train_rmses, 'test_rmses': test_rmses_sin, 'pearson_correlations': pearson_correlations_sin, 'spearman_correlations': spearman_correlations_sin, 
                   'weights': weights, 'all_train_losses': all_train_losses, 'all_test_losses': all_test_losses_sin, 'average_reward': average_reward_con, 'average_prediction': average_prediction, 'r2s': test_r2s_sin}
        save_results(to_save, results_path, baseline, type='results', data_mixture=(0,0), con=False)


        save_results(to_save, results_path, baseline, type='results', data_mixture=(0,0), con=True)

        hyper_params = {'epochs': epochs, 'learning_rate': learning_rate, 'l2_regularisation': regularisation}
        save_results(hyper_params, results_path, baseline, type='hyper_params', data_mixture=(0,0))

        to_save = {'test_losses': test_loss_oods_con, 'test_mean_errors': test_mean_error_oods_con, 'test_rmses': test_rmse_oods_con, 'pearson_correlations': pearson_correlation_oods_con, 'spearman_correlations': spearman_correlation_oods_con, 'r2s': r2_oods_con}
        save_results(to_save, results_path, baseline, type='results_ood', data_mixture=(0,0), con=True)

        to_save = {'test_losses': test_loss_oods_sin, 'test_mean_errors': test_mean_error_oods_sin, 'test_rmses': test_rmse_oods_sin, 'pearson_correlations': pearson_correlation_oods_sin, 'spearman_correlations': spearman_correlation_oods_sin, 'r2s': r2_oods_sin}
        save_results(to_save, results_path, baseline, type='results_ood', data_mixture=(0,0), con=False)

# split the data into k folds to run cross validation on
def split_for_cross_validation(train_set, train_labels, train_set_sin=[], train_labels_sin=[], k=5):
    # split data into k folds
    train_set_folds = []
    train_labels_folds = []
    train_set_folds_sin = []
    train_labels_folds_sin = []
    fold_size = int(len(train_set) / k)
    fold_size_sin = int(len(train_set_sin) / k)
    for i in range(k):
        train_set_folds.append(train_set[i*fold_size:(i+1)*fold_size])
        train_labels_folds.append(train_labels[i*fold_size:(i+1)*fold_size])
        train_set_folds_sin.append(train_set_sin[i*fold_size_sin:(i+1)*fold_size_sin])
        train_labels_folds_sin.append(train_labels_sin[i*fold_size_sin:(i+1)*fold_size_sin])
    return train_set_folds, train_labels_folds, train_set_folds_sin, train_labels_folds_sin

# from the k-folds of the training data, return the training and validation sets for the kth fold
def cross_validate(train_set_folds, train_labels_folds, train_set_folds_sin, train_labels_folds_sin, k):
    train_set_f_sin, train_labels_f_sin, validation_set_f_sin, validation_labels_f_sin = [], [], [], []
    train_set_f = torch.cat(train_set_folds[:k] + train_set_folds[k+1:])
    train_labels_f = torch.cat(train_labels_folds[:k] + train_labels_folds[k+1:])
    validation_set_f = train_set_folds[k]
    validation_labels_f = train_labels_folds[k]
    if train_set_folds_sin != []:
        train_set_f_sin = torch.cat(train_set_folds_sin[:k] + train_set_folds_sin[k+1:])
        train_labels_f_sin = torch.cat(train_labels_folds_sin[:k] + train_labels_folds_sin[k+1:])
        validation_set_f_sin = train_set_folds_sin[k]
        validation_labels_f_sin = train_labels_folds_sin[k]
    return train_set_f, train_labels_f, validation_set_f, validation_labels_f, torch.tensor(train_set_f_sin), train_labels_f_sin, torch.tensor(validation_set_f_sin), validation_labels_f_sin

def tune_epochs(train_con, train_labels_con, train_sin, train_labels_sin, lr, l2, num_layers=None, hidden_layer_sizes=None, task_weights=None):
    data_folds = config.data_folds
    train_set_folds_con, train_labels_folds_con, train_set_folds_sin, train_labels_folds_sin = split_for_cross_validation(train_con, train_labels_con, train_sin, train_labels_sin, k=data_folds)
    test_lossess = []
    num_features = 92

    for k in range(data_folds):
        train_set_f_con, train_labels_f_con, validation_set_f_con, validation_labels_f_con, train_set_f_sin, train_labels_f_sin, validation_set_f_sin, validation_labels_f_sin = cross_validate(train_set_folds_con, train_labels_folds_con, train_set_folds_sin, train_labels_folds_sin, k)
        if config.model_type == 'NN':
            model_sin, model_con, train_losses, train_losses_sin, test_losses_con, test_losses_sin , _, _, _ = train_model(train_set_f_con, train_labels_f_con, validation_set_f_con, validation_labels_f_con, train_set_f_sin, train_labels_f_sin, validation_set_f_sin, validation_labels_f_sin, num_features, epochs = hyperparameters.epochs_contrastive, learning_rate=lr, regularisation=l2, num_layers=num_layers, hidden_layer_sizes=hidden_layer_sizes, task_weights=task_weights)
        else:
            model_sin, model_con, train_losses, train_losses_sin, test_losses_con, test_losses_sin , _, _, _ = train_model_LM(train_set_f_con, train_labels_f_con, validation_set_f_con, validation_labels_f_con, train_set_f_sin, train_labels_f_sin, validation_set_f_sin, validation_labels_f_sin, num_features, epochs = hyperparameters.epochs_contrastive, learning_rate=lr, regularisation=l2, task_weights=task_weights)
        test_lossess.append([test_losses_sin[i]/task_weights[1] + test_losses_con[i]/task_weights[0] for i in range(len(test_losses_sin))])
    
    avg_test_losses = np.mean(test_lossess, axis=0)
    # get minimum value and index
    min_test_loss = np.amin(avg_test_losses)
    min_index = np.argmin(avg_test_losses)
    print(min_test_loss, min_index)
    return min_index

def hyper_param_optimization_architecture(train_set, train_labels, train_set_sin, train_labels_sin, task_weights=None):
    # we use 5-fold cross validation to find the best hyper parameters
    data_folds = config.data_folds
    # train_set_folds, train_labels_folds, train_set_folds_sin, train_labels_folds_sin = split_for_cross_validation(train_set, train_labels, k=data_folds)
    train_set_folds, train_labels_folds, train_set_folds_sin, train_labels_folds_sin = split_for_cross_validation(train_set, train_labels, train_set_sin, train_labels_sin, k=data_folds)

    num_features = len(train_set[0])
    best_loss = 100000000
    best_epoch = 0
    best_lr = 0
    best_l2 = 0
    best_num_layers = 0
    best_hidden_layer_sizes = []

    # different NN architectures with (number of layers, hidden layer sizes)
    if config.model_type == 'NN':
        # architectures = [(2, [[]]), (3, [[16], [8]]), (4, [[8,4], [8,8], [12,6], [12,12], [16,8], [16,16], [20,20]])]
        architectures = [(4, [[20,10], [30,15], [40,20], [60,30]]), (5, [[20,20,10], [30,30,10], [40,40,20], [50,50,50], [60,60,30], [92, 92, 46]])]
        architectures = [(5, [[92, 92, 46]])]
    else:
        architectures = [(None, [None])]
    # learning_rates = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]
    # learning_rates = [0.01]
    learning_rates = [0.01, 0.03, 0.1, 0.3]
    l2_lambdas = [0.01, 0.03, 0.1, 0.3]

    # loop over network architectures (not relevant for the LM)
    for num_layers, hidden_layer_sizes in architectures:
        for hidden_layer_size in hidden_layer_sizes:
            start_time = time.time()
            print(hidden_layer_size)
            # loop over hyper parameters
            for lrs in learning_rates:
                print('learning rate', lrs)
                for l2 in l2_lambdas:
                    # iterate over the k folds for cross validation
                    test_lossess = []
                    for k in range(data_folds):
                        train_set_f, train_labels_f, validation_set_f, validation_labels_f, train_set_f_sin, train_labels_f_sin, validation_set_f_sin, validation_labels_f_sin = cross_validate(train_set_folds, train_labels_folds, train_set_folds_sin, train_labels_folds_sin, k)
                        if config.model_type == 'NN':
                            model_sin, model_con, train_losses, train_losses_sin, test_losses_con, test_losses_sin , _, _, _ = train_model(train_set_f, train_labels_f, validation_set_f, validation_labels_f, train_set_f_sin, train_labels_f_sin, validation_set_f_sin, validation_labels_f_sin, num_features, epochs = 20000, learning_rate=lrs, regularisation=l2, num_layers=num_layers, hidden_layer_sizes=hidden_layer_size, task_weights=task_weights)
                        else:
                            model_sin, model_con, train_losses, train_losses_sin, test_losses_con, test_losses_sin , _, _, _ = train_model_LM(train_set_f, train_labels_f, validation_set_f, validation_labels_f, train_set_f_sin, train_labels_f_sin, validation_set_f_sin, validation_labels_f_sin, num_features, epochs = 2000, learning_rate=lrs, regularisation=l2, task_weights=task_weights)
                        test_lossess.append([test_losses_sin[i]/task_weights[1] + test_losses_con[i]/task_weights[0] for i in range(len(test_losses_sin))])

                    # show_loss_plot(train_losses, test_losses, show=False, lr=lrs, l2=l2)
                    avg_test_losses = np.mean(test_lossess, axis=0)
                    # get minimum value and index
                    min_test_loss = np.amin(avg_test_losses)
                    min_index = np.argmin(avg_test_losses)
                    print(min_test_loss, min_index)

                    if min_test_loss < best_loss:
                        best_loss = min_test_loss
                        best_lr = lrs
                        best_l2 = l2
                        best_epoch = min_index
                        best_num_layers = num_layers
                        best_hidden_layer_sizes = hidden_layer_size
            # plt.legend()
            # plt.show()
            print('time', time.time() - start_time)
    if config.model_type == 'NN':
        print(best_loss, best_epoch, best_lr, best_l2, best_hidden_layer_sizes, best_num_layers)
    else:
        print(best_loss, best_epoch, best_lr, best_l2)
    return best_epoch, best_lr, best_l2, best_num_layers, best_hidden_layer_sizes

def hyper_param_optimization(train_set, train_labels, task_weights=None):
    # we use 5-fold cross validation to find the best hyper parameters
    data_folds = config.data_folds
    train_set_folds, train_labels_folds = split_for_cross_validation(train_set, train_labels, k=data_folds)

    num_features = len(train_set[0])
    best_loss = 100000000
    best_epoch = 0
    best_lr = 0
    best_l2 = 0

    learning_rates = [0.001, 0.01, 0.1]
    # learning_rates = [0.01]
    l2_lambdas = [0.001, 0.01, 0.1]

    # return 32, 0.01, 0.01, 4, [8,4]

    # loop over hyper parameters
    for lrs in learning_rates:
        print('learning rate', lrs)
        for l2 in l2_lambdas:
            # iterate over the k folds for cross validation
            test_lossess = []
            for k in range(data_folds):
                train_set_f, train_labels_f, validation_set_f, validation_labels_f = cross_validate(train_set_folds, train_labels_folds, k)
                if config.model_type == 'NN':
                    model, train_losses, test_losses = train_model(train_set_f, train_labels_f, validation_set_f, validation_labels_f, num_features, epochs = hyperparameters.epochs_contrastive, learning_rate=lrs, regularisation=l2, num_layers=NN_params.num_layers, hidden_layer_sizes=NN_params.hidden_layer_sizes, task_weights=task_weights)
                else:
                    model, train_losses, test_losses = train_model_LM(train_set_f, train_labels_f, validation_set_f, validation_labels_f, num_features, epochs = hyperparameters.epochs_contrastive, learning_rate=lrs, regularisation=l2, task_weights=task_weights)
                test_lossess.append(test_losses)

            # show_loss_plot(train_losses, test_losses, show=False, lr=lrs, l2=l2)
            avg_test_losses = np.mean(test_lossess, axis=0)
            # get minimum value and index
            min_test_loss = np.amin(avg_test_losses)
            min_index = np.argmin(avg_test_losses)
            print(min_test_loss, min_index)

            if min_test_loss < best_loss:
                best_loss = min_test_loss
                best_lr = lrs
                best_l2 = l2
                best_epoch = min_index
            # plt.legend()
            # plt.show()
    if config.model_type == 'NN':
        print(best_loss, best_epoch, best_lr, best_l2)
    else:
        print(best_loss, best_epoch, best_lr, best_l2)
    return best_epoch, best_lr, best_l2

def read_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=int, default='0')
    parser.add_argument('--num_seeds', type=int, default='10')
    parser.add_argument('--method', type=str, default='mcts')
    parser.add_argument('--weight_con', type=float, default=None)
    parser.add_argument('--weight_sin', type=float, default=None)
    parser.add_argument('--model_type', type=str, default='NN')

    args = parser.parse_args()
    folder_path = os.path.join('datasets', 'ablations_norm', 'best_linear', args.method)   
        
    config.model_type = args.model_type
    hyperparameters.number_of_seeds = args.num_seeds
    return folder_path, (args.weight_con, args.weight_sin)

if __name__ == '__main__':
    folder_path, task_weights = read_arguments()

    print(folder_path, config.model_type)

    path_org_cte = os.path.join(folder_path,'org_features_new.pkl')
    path_cf_cte = os.path.join(folder_path,'cf_features_new.pkl')

    if task_weights == (None,None):
        task_weights = (1,3)

    # training_numbers = [5,10,20,50, 100, 200, 500, 800]
    # for n_train in training_numbers:
        # print(n_train)
        # learning_repeats(path_org_cte, path_cf_cte, folder_path, contrastive=True, baseline=0, n_train=n_train)

    learning_repeats(path_org_cte, path_cf_cte, folder_path, baseline=0, n_train=800, task_weights=task_weights)