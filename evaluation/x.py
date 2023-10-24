import pickle as pkl
import os
import sys
from pathlib import Path
adjacent_folder = Path(__file__).parent.parent
sys.path.append(str(adjacent_folder))
import torch
import numpy as np
import shutil


def stitch_mcts():
    with open('datasets\\1000mcts\\1000\cf_trajectories_tmp2.pkl', 'rb') as f:
        cf_trajs = pkl.load(f)
    with open('datasets\\1000mcts\\1000\org_trajectories_tmp2.pkl', 'rb') as f:
        org_trajs = pkl.load(f)
    with open('datasets\\1000mcts\\1000\cf_trajectories_tmp3.pkl', 'rb') as f:
        cf_trajs2 = pkl.load(f)
    with open('datasets\\1000mcts\\1000\org_trajectories_tmp3.pkl', 'rb') as f:
        org_trajs2 = pkl.load(f)
    print(len(cf_trajs), len(cf_trajs2))

    cf_trajs = cf_trajs + cf_trajs2[:70]
    org_trajs = org_trajs + org_trajs2[:70]
    print(len(cf_trajs), len(org_trajs))
    with open('datasets\\1000mcts\\1000\cf_trajectories2.pkl', 'wb') as f:
        pkl.dump(cf_trajs, f)
    with open('datasets\\1000mcts\\1000\org_trajectories2.pkl', 'wb') as f:
        pkl.dump(org_trajs, f)


def combine_data():
    with open('datasets\\1000random\\1000\\results_sidebysideNN_multi\800\data_split_con.pkl', 'rb') as f:
        train_set1_con, train_labels1_con, test_set1_con, test_labels1_con = pkl.load(f)
    with open('datasets\\1000random\\1000\\results_sidebysideNN_multi\800\data_split_sin.pkl', 'rb') as f:
        train_set1_sin, train_labels1_sin, test_set1_sin, test_labels1_sin = pkl.load(f)
    with open('datasets\\1000mcts\\1000\\results_sidebysideNN_multi\800\data_split_con.pkl', 'rb') as f:
        train_set2_con, train_labels2_con, test_set2_con, test_labels2_con = pkl.load(f)
    with open('datasets\\1000mcts\\1000\\results_sidebysideNN_multi\800\data_split_sin.pkl', 'rb') as f:
        train_set2_sin, train_labels2_sin, test_set2_sin, test_labels2_sin = pkl.load(f)
    with open('datasets\\1000step\\1000\\results_sidebysideNN_multi\800\data_split_con.pkl', 'rb') as f:
        train_set3_con, train_labels3_con, test_set3_con, test_labels3_con = pkl.load(f)
    with open('datasets\\1000step\\1000\\results_sidebysideNN_multi\800\data_split_sin.pkl', 'rb') as f:
        train_set3_sin, train_labels3_sin, test_set3_sin, test_labels3_sin = pkl.load(f)
    # with open('datasets\\1000_ablations\only_one\only_sparsity\\100\\results_sidebysideLM\80\data_split.pkl', 'rb') as f:
        # train_set6, train_labels6, test_set6, test_labels6 = pkl.load(f)

    # append training data together
    test_set_con = torch.tensor(np.concatenate((test_set1_con, test_set2_con, test_set3_con), axis=0))
    test_labels_con = torch.tensor(np.concatenate((test_labels1_con, test_labels2_con, test_labels3_con), axis=0))
    test_set_sin = torch.tensor(np.concatenate((test_set1_sin, test_set2_sin, test_set3_sin), axis=0))
    test_labels_sin = torch.tensor(np.concatenate((test_labels1_sin, test_labels2_sin, test_labels3_sin), axis=0))
    with open('datasets\\1000baseline\\1000combined\combined_test_set.pkl', 'wb') as f:
        pkl.dump([test_set_con, test_labels_con, test_set_sin, test_labels_sin], f)
        

def format_folders():
    for folder in os.listdir(os.path.join('datasets', 'weights_norm', 'step copy')):
        try:
            # delete folder statistics and all the files in it
            shutil.rmtree(os.path.join('datasets', 'weights_norm', 'step copy', folder, 'statistics'))
            # delete unneeded files
            os.remove(os.path.join('datasets', 'weights_norm', 'step copy', folder, 'cf_features_new.pkl'))
            os.remove(os.path.join('datasets', 'weights_norm', 'step copy', folder, 'org_features_new.pkl'))
            os.remove(os.path.join('datasets', 'weights_norm', 'step copy', folder, 'cf_trajectories.pkl'))
            os.remove(os.path.join('datasets', 'weights_norm', 'step copy', folder, 'org_trajectories.pkl'))
            # create folder 'results'
            os.mkdir(os.path.join('datasets', 'weights_norm', 'step copy', folder, 'results'))

        except:
            continue

format_folders()