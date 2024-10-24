
import argparse

import os
import sys

import numpy as np
import pickle 

# import trainer
# -----------------------------
sys.path.append('/home/david/Desktop/projects/thesis/downstream_classification')
os.chdir('/home/david/Desktop/projects/thesis/downstream_classification')
from executors.train_on_local_machine_v2 import trainer

# get global argumnents
# -----------------------------
parser = argparse.ArgumentParser(description="Run trainer with specified number of channels to turn off.")
parser.add_argument('--num_channels_to_turn_off', type=int, required=True, help='Number of channels to turn off')

args = parser.parse_args()



# run script
# -----------------------------
best_rocauc_and_pr_auc = {
    'channels_to_turn_off':[],
    'seed':[],
    'validation-roc-auc':[],
    'validation-pr-auc':[],
    'test-roc-auc':[],
    'test-pr-auc':[],
}

seed = 123
np.random.seed(123)
seeds = np.random.randint(0,1000, 30)

for seed_ in seeds:
    config = {   
        # general
        'seed': seed_,
        # 'metadata_file_path': './data/combined_data/metadata_only_existant_readings_09042023.csv',
        # 'data_folder_path': './data/individual-signals-registered/',
        'metadata_file_path': './data/combined_data/ptb_signal_level_metadata_with_label.csv',
        'data_folder_path': './data/ptb-ecg-processed-divided-into-450/',


        # training
        'batch_size': 128,
        'n_epochs': 10,
        'weight_decay': 0.3,
        'lr': 0.0005,
        'eval_metric':'rocauc',
        'patience':3,

        # architecture
        'targets': ['AF'],
        'dropout': 0.5,
        'clip': 1,
        'scale':1,
        'depth': 1,
        
        # misc
        'model_saving_path': f'./models/AF-V10-different-seeds-ptb-{args.num_channels_to_turn_off}_channels_turned_off',
        'verbosity':True,

        # test
        'check_on_test':True,
        
        # external data
        'internal_data':False
    }

    print('-'*25)
    print(f'Turned off channels: {args.num_channels_to_turn_off}')
    print('-'*25)

    # set the number of channels to turn off
    config['channels_to_turn_off'] = args.num_channels_to_turn_off
    config['model_saving_path']= f'./models/AF-V10-different-seeds-ptb-{args.num_channels_to_turn_off}_channels_off'

    # run
    best_rocauc_and_pr_auc_seed = trainer(**config)

    # store results
    best_rocauc_and_pr_auc['channels_to_turn_off'].append(args.num_channels_to_turn_off)
    best_rocauc_and_pr_auc['seed'].append(config['seed'])
    best_rocauc_and_pr_auc['validation-roc-auc'].append(best_rocauc_and_pr_auc_seed['validation-roc-auc'])
    best_rocauc_and_pr_auc['validation-pr-auc'].append(best_rocauc_and_pr_auc_seed['validation-auc-pr'])
    best_rocauc_and_pr_auc['test-roc-auc'].append(best_rocauc_and_pr_auc_seed['test-roc-auc'])
    best_rocauc_and_pr_auc['test-pr-auc'].append(best_rocauc_and_pr_auc_seed['test-auc-pr'])

    with open(config['model_saving_path']+'/best_rocauc_and_pr_auc.pkl', 'wb') as f:
        pickle.dump(best_rocauc_and_pr_auc, f)