import sys
import os
from pathlib import Path
sys.path.append(os.path.join(os.getcwd(), '..', '..'))
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
from tabulate import tabulate

# pearson correlation
from scipy.stats import pearsonr, spearmanr

with open('datasets/ablations_norm/best/mcts/data_split_con.pkl', 'rb') as f:
    train_con_mcts, train_labels_con_mcts, test_con_mcts, test_labels_con_mcts = pickle.load(f)
with open('datasets/ablations_norm/best/mcts/data_split_sin.pkl', 'rb') as f:   
    train_sin_mcts, train_labels_sin_mcts, test_sin_mcts, test_labels_sin_mcts = pickle.load(f)

with open('datasets/ablations_norm/best/dac/data_split_con.pkl', 'rb') as f:
    train_con_dac, train_labels_con_dac, test_con_dac, test_labels_con_dac = pickle.load(f)
with open('datasets/ablations_norm/best/dac/data_split_sin.pkl', 'rb') as f:
    train_sin_dac, train_labels_sin_dac, test_sin_dac, test_labels_sin_dac = pickle.load(f)

with open('datasets/ablations_norm/best/random/data_split_con.pkl', 'rb') as f:
    train_con_random, train_labels_con_random, test_con_random, test_labels_con_random = pickle.load(f)
with open('datasets/ablations_norm/best/random/data_split_sin.pkl', 'rb') as f:
    train_sin_random, train_labels_sin_random, test_sin_random, test_labels_sin_random = pickle.load(f)


train_set_con = []
train_labels_con = []
train_set_sin = []
train_labels_sin = []
test_set_con = torch.cat([test_con_mcts, test_con_dac, test_con_random], dim=0)
test_labels_con = torch.cat([test_labels_con_mcts, test_labels_con_dac, test_labels_con_random], dim=0)
test_set_sin = torch.cat([test_sin_mcts, test_sin_dac, test_sin_random], dim=0)
test_labels_sin = torch.cat([test_labels_sin_mcts, test_labels_sin_dac, test_labels_sin_random], dim=0)


with open(os.path.join('datasets', 'ablations_norm', 'best', 'baseline2', 'data_split_con.pkl'), 'wb') as f:
    pickle.dump([train_set_con, train_labels_con, test_set_con, test_labels_con], f)
with open(os.path.join('datasets', 'ablations_norm', 'best', 'baseline2', 'data_split_sin.pkl'), 'wb') as f:
    pickle.dump([train_set_sin, train_labels_sin, test_set_sin, test_labels_sin], f)