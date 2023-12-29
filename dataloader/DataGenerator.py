
# imports
from typing import List

import os

import numpy as np
import pandas as pd
import random



class DataGenerator():
    def __init__(self, 
                 data_folder_path:str,
                 metadata_file_path:str,
                 targets:List,
                 sample:str,
                 batch_size:int,
                 seed:int,
                 shuffle:bool = True):

        # adjust target if a sting was entered
        if isinstance(targets, str):
            targets = [targets]
        
        # Load metadata file and keep only relevant targets
        metadata_file = pd.read_csv(metadata_file_path,index_col=0)
        metadata_file_sample = metadata_file[metadata_file['sample'] == sample].reset_index(drop=True)
        metadata_file_sample = metadata_file_sample[['reading']+targets]

        # store in self
        self.metadata_file_sample   = metadata_file_sample
        self.data_folder_path       = data_folder_path
        self.indices                = np.arange(len(self.metadata_file_sample))
        self.list_IDs               = self.metadata_file_sample['reading'].values
        self.Y                      = self.metadata_file_sample[targets].values
        self.shuffle                = shuffle
        self.batch_size             = batch_size

        random.seed(seed)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.indices)
            
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))


    def __getitem__(self, index):
        'Generate one batch of data using __data_generation'
        # Generate indexes of the batch
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_batch = [self.list_IDs[k] for k in indices]

        # get Ys
        Y_batch = np.array([self.Y[k] for k in indices])

        # Generate data (load signals) # TBA
        X_batch = self.__data_generation(list_IDs_batch)
        X_batch = np.nan_to_num(X_batch)
        
        return X_batch, Y_batch, list_IDs_batch

    def __data_generation(self, list_IDs_batch):
        return np.array([np.load(self.data_folder_path + str(signal_name) +'.npy') for signal_name in list_IDs_batch])



# data_generator = DataGenerator(
#     data_folder_path='../data/individual-signals/',
#     metadata_file_path='../data/combined_data/metadata_only_existant_readings_09042023.csv',
#     targets=['AF','one_year_until_death'],
#     sample='train',
#     batch_size=4,
#     shuffle=True
# )
# X,y,sig = data_generator.__getitem__(1)
