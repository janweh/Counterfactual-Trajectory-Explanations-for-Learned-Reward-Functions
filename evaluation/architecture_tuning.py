from train_mimic_sidebyside_multi import *
import random
import pickle
import numpy
import torch

# Load data from all 3 types

# with open('datasets\\1000mcts\\1000\\results_sidebysideNN_multi\800\data_split_con.pkl', 'rb') as f:
#     train_set_mcts_con, train_labels_mcts_con, test_set_con, test_labels_con = pickle.load(f)
# train_set_mcts_con = train_set_mcts_con[:267]
# with open('datasets\\1000mcts\\1000\\results_sidebysideNN_multi\800\data_split_sin.pkl', 'rb') as f:
#     train_set_mcts_sin, train_labels_mcts_sin, test_set_sin, test_labels_sin = pickle.load(f)
# train_set_mcts_sin = train_set_mcts_sin[:267]

# with open('datasets\\1000step\\1000\\results_sidebysideNN_multi\800\data_split_con.pkl', 'rb') as f:
#     train_set_step_con, train_labels_step_con, test_set_con, test_labels_con = pickle.load(f)
# train_set_step_con = train_set_step_con[:267]
# with open('datasets\\1000step\\1000\\results_sidebysideNN_multi\800\data_split_sin.pkl', 'rb') as f:
#     train_set_step_sin, train_labels_step_sin, test_set_sin, test_labels_sin = pickle.load(f)
# train_set_step_sin = train_set_step_sin[:267]

# with open('datasets\\1000random\\1000\\results_sidebysideNN_multi\800\data_split_con.pkl', 'rb') as f:
#     train_set_random_con, train_labels_random_con, test_set_con, test_labels_con = pickle.load(f)
# train_set_random_con = train_set_random_con[:266]
# with open('datasets\\1000random\\1000\\results_sidebysideNN_multi\800\data_split_sin.pkl', 'rb') as f:
#     train_set_random_sin, train_labels_random_sin, test_set_sin, test_labels_sin = pickle.load(f)
# train_set_random_sin = train_set_random_sin[:266]

# # append training data together
# train_set_con = torch.cat((train_set_mcts_con, train_set_step_con, train_set_random_con))
# train_labels_con = torch.cat((train_labels_mcts_con, train_labels_step_con, train_labels_random_con))
# train_set_sin = torch.cat((train_set_mcts_sin, train_set_step_sin, train_set_random_sin))
# train_labels_sin = torch.cat((train_labels_mcts_sin, train_labels_step_sin, train_labels_random_sin))
# # shuffle
# train_set_labels_con = list(zip(train_set_con, train_labels_con))
# train_set_labels_sin = list(zip(train_set_sin, train_labels_sin))
# random.shuffle(train_set_labels_con)
# random.shuffle(train_set_labels_sin)
# train_set_con, train_labels_con = zip(*train_set_labels_con)
# train_set_sin, train_labels_sin = zip(*train_set_labels_sin)
# train_set_con = torch.stack(train_set_con)
# train_labels_con = torch.stack(train_labels_con)
# train_set_sin = torch.stack(train_set_sin)
# train_labels_sin = torch.stack(train_labels_sin)

# # do hyperparameter search
# epochs, learning_rate, regularisation, num_layers, hidden_sizes = hyper_param_optimization_architecture(train_set_con, train_labels_con, train_set_sin, train_labels_sin, (1,3))
# # epochs, learning_rate, regularisation = hyper_param_optimization(train_set, train_labels)

# # print(num_layers, hidden_sizes)
# print(epochs, learning_rate, regularisation, num_layers, hidden_sizes)

# # output is: num_layers = 4, hidden_sizes = [16, 8]

n_train = 800

train_set_con = []
train_labels_con = []
train_set_sin = []
train_labels_sin = []
test_set_con = []
test_labels_con = []
test_set_sin = []
test_labels_sin = []

# split into training and test data
# for folder in os.listdir(os.path.join('datasets', 'ablations_norm', 'best')):
#     if folder == 'baseline':
#         continue
#     # load data
#     org_features = pickle.load(open(os.path.join('datasets', 'ablations_norm', 'best', folder, 'org_features_new.pkl'), 'rb'))
#     cf_features = pickle.load(open(os.path.join('datasets', 'ablations_norm', 'best', folder, 'cf_features_new.pkl'), 'rb'))
#     num_features = len(org_features[0])-1

#     org_features, cf_features = shuffle_together(org_features, cf_features)
#     train_con, train_label_con, test_con, test_label_con = train_test_split_contrastive_sidebyside(org_features, cf_features, num_features, n_train=n_train)
#     train_sin, train_label_sin, test_sin, test_label_sin = train_test_split_single_sidebyside(org_features, cf_features, num_features, n_train=n_train)

#     # save train & test sets
#     with open(os.path.join('datasets', 'ablations_norm', 'best', folder, 'data_split_con.pkl'), 'wb') as f:
#         pickle.dump([train_con, train_label_con, test_con, test_label_con], f)
#     with open(os.path.join('datasets', 'ablations_norm', 'best', folder, 'data_split_sin.pkl'), 'wb') as f:
#         pickle.dump([train_sin, train_label_sin, test_sin, test_label_sin], f)

#     train_set_con.append(train_con[:267])
#     train_labels_con.append(train_label_con[:267])
#     train_set_sin.append(train_sin[:267])
#     train_labels_sin.append(train_label_sin[:267])
#     test_set_con.append(test_con)
#     test_labels_con.append(test_label_con)
#     test_set_sin.append(test_sin)
#     test_labels_sin.append(test_label_sin)

with open(os.path.join('datasets', 'ablations_norm', 'best_linear', 'mcts', 'data_split_con.pkl'), 'rb') as f:
    train_con_mcts, train_labels_con_mcts, test_con_mcts, test_labels_con_mcts = pickle.load(f)
with open(os.path.join('datasets', 'ablations_norm', 'best_linear', 'mcts', 'data_split_sin.pkl'), 'rb') as f:
    train_sin_mcts, train_labels_sin_mcts, test_sin_mcts, test_labels_sin_mcts = pickle.load(f)

with open(os.path.join('datasets', 'ablations_norm', 'best_linear', 'dac', 'data_split_con.pkl'), 'rb') as f:
    train_con_dac, train_labels_con_dac, test_con_dac, test_labels_con_dac = pickle.load(f)
with open(os.path.join('datasets', 'ablations_norm', 'best_linear', 'dac', 'data_split_sin.pkl'), 'rb') as f:
    train_sin_dac, train_labels_sin_dac, test_sin_dac, test_labels_sin_dac = pickle.load(f)

with open(os.path.join('datasets', 'ablations_norm', 'best_linear', 'random', 'data_split_con.pkl'), 'rb') as f:
    train_con_random, train_labels_con_random, test_con_random, test_labels_con_random = pickle.load(f)
with open(os.path.join('datasets', 'ablations_norm', 'best_linear', 'random', 'data_split_sin.pkl'), 'rb') as f:
    train_sin_random, train_labels_sin_random, test_sin_random, test_labels_sin_random = pickle.load(f)

train_set_con.append(train_con_mcts[:267])
train_labels_con.append(train_labels_con_mcts[:267])
train_set_sin.append(train_sin_mcts[:267])
train_labels_sin.append(train_labels_sin_mcts[:267])

train_set_con.append(train_con_dac[:267])
train_labels_con.append(train_labels_con_dac[:267])
train_set_sin.append(train_sin_dac[:267])
train_labels_sin.append(train_labels_sin_dac[:267])

train_set_con.append(train_con_random[:266])
train_labels_con.append(train_labels_con_random[:266])
train_set_sin.append(train_sin_random[:266])
train_labels_sin.append(train_labels_sin_random[:266])

# append training data together
train_set_con = torch.cat(train_set_con, dim=0)
train_labels_con = torch.cat(train_labels_con, dim=0)
train_set_sin = torch.cat(train_set_sin, dim=0)
train_labels_sin = torch.cat(train_labels_sin, dim=0)
# shuffle
train_set_labels_con = list(zip(train_set_con, train_labels_con))
train_set_labels_sin = list(zip(train_set_sin, train_labels_sin))
random.shuffle(train_set_labels_con)
random.shuffle(train_set_labels_sin)
train_set_con, train_labels_con = zip(*train_set_labels_con)
train_set_sin, train_labels_sin = zip(*train_set_labels_sin)
train_set_con = torch.stack(train_set_con)
train_labels_con = torch.stack(train_labels_con)
train_set_sin = torch.stack(train_set_sin)
train_labels_sin = torch.stack(train_labels_sin)

test_set_con = torch.cat((test_con_mcts, test_con_dac, test_con_random), dim=0)
test_labels_con = torch.cat((test_labels_con_mcts, test_labels_con_dac, test_labels_con_random), dim=0)
test_set_sin = torch.cat((test_sin_mcts, test_sin_dac, test_sin_random), dim=0)
test_labels_sin = torch.cat((test_labels_sin_mcts, test_labels_sin_dac, test_labels_sin_random), dim=0)

epochs, learning_rate, regularisation, num_layers, hidden_sizes = hyper_param_optimization_architecture(train_set_con, train_labels_con, train_set_sin, train_labels_sin, (1,3))

print(epochs, learning_rate, regularisation, num_layers, hidden_sizes)

# # check if folder exists
if not os.path.exists(os.path.join('datasets', 'ablations_norm', 'best_linear', 'baseline2')):
    os.makedirs(os.path.join('datasets', 'ablations_norm', 'best_linear', 'baseline2'))
with open(os.path.join('datasets', 'ablations_norm', 'best_linear', 'baseline2', 'data_split_con.pkl'), 'wb') as f:
    pickle.dump([train_set_con, train_labels_con, test_set_con, test_labels_con], f)
with open(os.path.join('datasets', 'ablations_norm', 'best_linear', 'baseline2', 'data_split_sin.pkl'), 'wb') as f:
    pickle.dump([train_set_sin, train_labels_sin, test_set_sin, test_labels_sin], f)