
# %%

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


# PTB compatible data generatore, which supports scrutiny of the data
class DataGenerator_ptb():
    def __init__(self, 
                 data_folder_path:str,
                 metadata_file_path:str,
                 targets:List,
                 sample:str,
                 batch_size:int,
                 seed:int,
                 channels_to_turn_off:int=0,
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
        self.channels_to_turn_off  = channels_to_turn_off

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
        
        # turn off channels
        X_batch = DataGenerator_ptb.__turn_off_channels(X_batch, self.channels_to_turn_off)

        return X_batch, Y_batch, list_IDs_batch

    def __data_generation(self, list_IDs_batch):
        return np.array([np.load(self.data_folder_path + str(signal_name) +'.npy') for signal_name in list_IDs_batch])

    def __turn_off_channels(ecg_batch: np.ndarray, channels_to_turn_off: int) -> np.ndarray:
        """
        Turns off a random set of channels in the ECG data by setting them to zero.
        Parameters:
            ecg_batch (np.ndarray): ECG data of shape (num_people, num_channels, num_timestamps).
            channels_to_turn_off (int): Number of channels to turn off (set to zero).
        
        Returns:
            np.ndarray: ECG data with some channels turned off.
        """
        num_recordings, num_channels, num_timestamps = ecg_batch.shape
        
        if channels_to_turn_off < 0 or channels_to_turn_off > num_channels:
            raise ValueError(f"channels_to_turn_off must be between 0 and {num_channels}, inclusive.")
        
        ecg_batch_ = ecg_batch.copy()
        
        for i in range(num_recordings):
            # Randomly select channels to turn off
            channels = np.random.choice(num_channels, channels_to_turn_off, replace=False)
            # Set the selected channels to zero
            ecg_batch_[i, channels, :] = 0
        
        return ecg_batch_


# %%

# Test data generator
# ---------------------------
# data_generator = DataGenerator(
#     data_folder_path='../data/individual-signals/',
#     metadata_file_path='../data/combined_data/metadata_only_existant_readings_09042023.csv',
#     targets=['AF','one_year_until_death'],
#     sample='train',
#     batch_size=4,
#     shuffle=True
# )
# X,y,sig = data_generator.__getitem__(1)


# Test PTB data generator
# ---------------------------


# def plot_ecg_signals(ecg_batch: np.ndarray, person_index: int = 0):
#     """
#     Plots the ECG signals for a specific person using a 6x2 subplot layout.
    
#     Parameters:
#         ecg_batch (np.ndarray): ECG data of shape (num_people, num_channels, num_timestamps).
#         person_index (int): Index of the person to plot the ECG signals for (default is 0).
#     """
#     import matplotlib.pyplot as plt

#     num_people, num_channels, num_timestamps = ecg_batch.shape
    
#     if person_index < 0 or person_index >= num_people:
#         raise ValueError(f"person_index must be between 0 and {num_people - 1}, inclusive.")
    
#     fig, axes = plt.subplots(6, 2, figsize=(15, 12))
#     fig.suptitle(f'ECG Signals for Person {person_index}', fontsize=16)
    
#     # Plot each channel's signal in a separate subplot
#     for i in range(num_channels):
#         row, col = divmod(i, 2)
#         ax = axes[row, col]
#         ax.plot(ecg_batch[person_index, i, :], label=f'Channel {i+1}')
#         ax.set_title(f'Channel {i+1}')
#         ax.set_xlabel('Timestamps')
#         ax.set_ylabel('Amplitude')
#         ax.legend(loc='upper right')

#     # Adjust layout
#     plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leaves space for the main title
#     plt.show()



# data_generator = DataGenerator_ptb(
#     data_folder_path='../data/ptb-ecg-processed-divided-into-450/',
#     metadata_file_path='../data/combined_data/ptb_signal_level_metadata_with_label.csv',
#     targets=['AF'],
#     sample='train',
#     batch_size=4,
#     shuffle=True,
#     channels_to_turn_off=4,
#     seed=1
# )
# X,y,sig = data_generator.__getitem__(1)
# plot_ecg_signals(X)
