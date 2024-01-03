import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import List
import json
import random
import os

class MIT_BIH_DataGen():
    def __init__(self, data_path, readings, batch_size, num_steps = 10000, window_size = 450):
        
        assert (batch_size % 10) == 0, 'batch_size has to be a multiplication of 10'

        # store to self
        self.readings = readings
        self.batch_size = int(batch_size/10)
        self.window_size = window_size
        self.data_path = data_path
        self.num_steps = num_steps
        
        # get numpy files
        self.npy_files = [x for x in os.listdir(data_path) if x.endswith('.npy')]

        # get relevant files
        self.N_files = []
        self.AF_files = []
        # self.test_N_files = []
        # self.test_AF_files = []

        for reading_number in self.readings:
            intervals_for_reading_all = [x for x in self.npy_files if x.split('_')[0] == reading_number]

            intervals_for_reading_N = [x for x in intervals_for_reading_all if x.endswith('_N.npy')]
            intervals_for_reading_AF = [x for x in intervals_for_reading_all if x.endswith('_AFIB.npy')]
            
            self.N_files += intervals_for_reading_N
            self.AF_files += intervals_for_reading_AF
        
        # shuffle lists
        random.shuffle(self.N_files)
        random.shuffle(self.AF_files)
        
        print('train_N_files[:5]:',self.N_files[:5])
        print('train_AF_files[:5]:',self.AF_files[:5])
        print('train_N_files Len:',len(self.N_files))
        print('train_AF_files Len:',len(self.AF_files))

        self.N_files = np.array(self.N_files)
        self.AF_files = np.array(self.AF_files)

    def __getitem__(self, index):
        """
        ASSUMPTIONS:
        - balanced batch each time
        - sample form five files each time, later versions can change that
        """
        N_files_ = self.N_files.copy()
        AF_files_ = self.AF_files.copy()
        batch_files_N  = np.random.choice(N_files_, size=5, replace=True)
        batch_files_AF = np.random.choice(AF_files_, size=5, replace=True)

        signals_in_batch = []
        targets_in_batch = []
        for filename in batch_files_N:
            try:
                interval = np.load(self.data_path+f'/{filename}')
                interval_length = len(interval)
                
                # draw window starts
                indices_to_draw_from = np.arange(interval_length-self.window_size-1)
                starts_of_signals = np.random.choice(indices_to_draw_from, size=self.batch_size, replace=False)
                
                # save signals
                signals_in_batch += [interval[start:(start+self.window_size)] for start in starts_of_signals]
                targets_in_batch += [0 for x in range(self.batch_size)]
            except:
                continue

        for filename in batch_files_AF:
            try:
                interval = np.load(self.data_path+f'/{filename}')
                interval_length = len(interval)
                
                # draw window starts
                indices_to_draw_from = np.arange(interval_length-self.window_size-1)
                starts_of_signals = np.random.choice(indices_to_draw_from, size=self.batch_size, replace=False)
                
                # save signals
                signals_in_batch += [interval[start:(start+self.window_size)] for start in starts_of_signals]
                targets_in_batch += [1 for _x in range(self.batch_size)]
            except:
                continue
        # combind signals with targets
        signals_and_targets_in_batch = [(signal, target) for (signal, target) in zip(signals_in_batch,targets_in_batch)]
        
        # shuffle
        random.shuffle(signals_and_targets_in_batch)

        # convert to np
        X = np.array([x[0] for x in signals_and_targets_in_batch])
        y = np.array([x[1] for x in signals_and_targets_in_batch])
        
        # transpose X
        X = np.transpose(X,(0,2,1))

        return X,y

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.num_steps
