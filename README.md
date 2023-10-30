# Counterfactual Trajectory Explanations for Learned Reward Functions

## Description
This work aims to explain reward functions that were learned through Adversarial Inverse Reinforcement Learning by generating Counterfactual Explanations. These Counterfactual Trajectory Explanations (CTEs) 
consist of one partial original and counterfactual trajectory and their respective average reward. By training a proxy-human model on the CTEs and measuring whether it can predict rewards similar to the 
learned reward function.

## Dependencies
* wandb
* tqdm
* pytorch \>= 1.7.0
* numpy \>= 1.20.0
* scipy \>= 1.1.0
* pycolab == 1.2


## Learning a reward function:
1. Train a policy $\pi*$:   
  ``python rl_airl\ppo_train.py``  
  Provided expert policy: `saved_models\ppo_v2_[1, 10].pt`  
2. Generate expert demonstrations $D$:
  ``python rl_airl\generate_demos.py``  
  Provided demonstrations: `demonstrations\original_trajectories_new_maxsteps75_airl_1000_new.pkl`  
3. Train AIRL to get the learned reward function $R_\theta$ and policy $\pi_\theta$:  
  ``python rl_airl\airl_train.py``  
  Provided discriminator (used as reward function): `saved_models\discriminator_v2_[1,10]_new.pt`   




## Generating Counterfactual Trajectory Explanations
1. Generate full original trajectories $\tau_{org}$:  
``python cte\generate_original_trajectories.py``  
Provided original trajectories: Provided generator (used as policy trained on reward function): demonstrations\original_trajectories_new_maxsteps75_airl_1000_new.pkl  
2. Generate CTEs:  
``python cte\counterfactual_trajectories.py``  
by specifying --method {'mcto', 'dac'} you can choose the method used. You can also adjust hyperparamters here. The default are the highest performing hyperparameters that were used in the project.  
You can specify which weight for quality criteria should be used:  

* --special_weights 'scrambled' you will get 1000 unifromly sampled set of weights  
* --weights {1,...,29} you get one set of weights for all created CTEs  

For randomly generated CTEs use: python cte\generation_methods\random_cf_generator.py  

## Evaluating informativeness of CTEs  
1. Extracting Features:  
``python evaluation\extract_features.py``  
You will have to place the path to folder folder where the CTEs are saved in the variable folder_path  

2. Generate combined dataset  
``python evaluation\architecture_tuning.py``  
In this step you can also perform hyperparameter tuning for all sets that go into the combined test set. You will have to replace the folder paths with the folders of CTEs (where the extracted features are saved) for which you want one combined test set  

3. Train the Proxy-human model  
``python evaluation\train_proxy_human_model.py``  
By default this will use a Neural Network as a model, but you can specify --model_type 'linear' for a linear model  
--num_seeds determines how many random initialisations of the model will be trained  

It will output multiple statistics. The key metric used in the project is: 'spearman correlation combined' and 'spearman correlation'   
