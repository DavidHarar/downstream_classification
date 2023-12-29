import os
import sys

# import trainer
sys.path.append('/home/david/Desktop/projects/thesis/downstream_classification')
os.chdir('/home/david/Desktop/projects/thesis/downstream_classification')
from executors.train_on_local_machine_v2 import trainer

# config = {
    
#     # general
#     'seed': 123,
#     # 'metadata_file_path': './data/combined_data/metadata_balanced_by_death.csv',
#     'metadata_file_path': './data/combined_data/metadata_only_existant_readings_09042023.csv',
#     'data_folder_path': './data/individual-signals/',
    
#     # training
#     'batch_size': 128,
#     'n_epochs': 10,
#     'weight_decay': 0.7,
#     'lr': 0.0005,
#     'eval_metric':'rocauc',
#     'patience':3,

#     # architecture
#     'targets': ['AF'],
#     'dropout': 0.5,
#     'clip': 1,
#     'scale':1,
#     'depth': 5,
    
#     # misc
#     'model_saving_path': f'./models/AF-V3',
#     'verbosity':True
# }

# # run
# trainer(**config)

# V4
# -----------------
config = {
    
    # general
    'seed': 123,
    # 'metadata_file_path': './data/combined_data/metadata_balanced_by_death.csv',
    'metadata_file_path': './data/combined_data/metadata_only_existant_readings_09042023.csv',
    'data_folder_path': './data/individual-signals/',
    
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
    'depth': 5,
    
    # misc
    'model_saving_path': f'./models/AF-V4',
    'verbosity':True
}

# run
trainer(**config)

# V5
# -------------
config = {
    
    # general
    'seed': 123,
    # 'metadata_file_path': './data/combined_data/metadata_balanced_by_death.csv',
    'metadata_file_path': './data/combined_data/metadata_only_existant_readings_09042023.csv',
    'data_folder_path': './data/individual-signals/',
    
    # training
    'batch_size': 128,
    'n_epochs': 10,
    'weight_decay': 0.1,
    'lr': 0.0005,
    'eval_metric':'rocauc',
    'patience':3,

    # architecture
    'targets': ['AF'],
    'dropout': 0.3,
    'clip': 1,
    'scale':1,
    'depth': 5,
    
    # misc
    'model_saving_path': f'./models/AF-V5',
    'verbosity':True
}

# run
trainer(**config)

# V6
# -------------
config = {
    
    # general
    'seed': 123,
    # 'metadata_file_path': './data/combined_data/metadata_balanced_by_death.csv',
    'metadata_file_path': './data/combined_data/metadata_only_existant_readings_09042023.csv',
    'data_folder_path': './data/individual-signals/',
    
    # training
    'batch_size': 128,
    'n_epochs': 10,
    'weight_decay': 0.7,
    'lr': 0.001,
    'eval_metric':'rocauc',
    'patience':3,

    # architecture
    'targets': ['AF'],
    'dropout': 0.35,
    'clip': 1,
    'scale':1,
    'depth': 5,
    
    # misc
    'model_saving_path': f'./models/AF-V6',
    'verbosity':True
}

# run
trainer(**config)


# V7
# -------------
config = {
    
    # general
    'seed': 123,
    # 'metadata_file_path': './data/combined_data/metadata_balanced_by_death.csv',
    'metadata_file_path': './data/combined_data/metadata_only_existant_readings_09042023.csv',
    'data_folder_path': './data/individual-signals/',
    
    # training
    'batch_size': 128,
    'n_epochs': 10,
    'weight_decay': 0.7,
    'lr': 0.001,
    'eval_metric':'rocauc',
    'patience':3,

    # architecture
    'targets': ['AF'],
    'dropout': 0.5,
    'clip': 1,
    'scale':1,
    'depth': 7,
    
    # misc
    'model_saving_path': f'./models/AF-V7',
    'verbosity':True
}

# run
trainer(**config)


# V8
# -------------
config = {
    
    # general
    'seed': 123,
    # 'metadata_file_path': './data/combined_data/metadata_balanced_by_death.csv',
    'metadata_file_path': './data/combined_data/metadata_only_existant_readings_09042023.csv',
    'data_folder_path': './data/individual-signals/',
    
    # training
    'batch_size': 128,
    'n_epochs': 10,
    'weight_decay': 0.9,
    'lr': 0.0005,
    'eval_metric':'rocauc',
    'patience':3,

    # architecture
    'targets': ['AF'],
    'dropout': 0.35,
    'clip': 1,
    'scale':1,
    'depth': 7,
    
    # misc
    'model_saving_path': f'./models/AF-V8',
    'verbosity':True
}

# run
trainer(**config)
