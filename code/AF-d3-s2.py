import os
import sys

# import trainer
sys.path.append('/home/david/Desktop/projects/thesis/downstream_classification')
os.chdir('/home/david/Desktop/projects/thesis/downstream_classification')
from executors.train_on_local_machine_v2 import trainer

config = {
    
    # general
    'seed': 123,
    # 'metadata_file_path': './data/combined_data/metadata_balanced_by_death.csv',
    'metadata_file_path': './data/combined_data/metadata_only_existant_readings_09042023.csv',
    'data_folder_path': './data/individual-signals/',
    
    # training
    'batch_size': 128,
    'n_epochs': 20,
    'weight_decay': 0.7,
    'lr': 0.0005,
    'eval_metric':'rocauc',
    'patience':7,

    # architecture
    'targets': ['AF'],
    'dropout': 0.1,
    'clip': 1,                        
    'scale':2,
    'depth': 3,
    
    # misc
    'model_saving_path': f'./models/AF-d3-s2_V1',
    'verbosity':True
}

# run
trainer(**config)
