import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt

from tqdm import tqdm
import logging

logging.basicConfig(filename=f"/home/david/Desktop/projects/thesis/downstream_classification/notebooks/6.1.saving-registered-data.log", 
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M')
logging.info("Saving data")
    

# os.getcwd()
os.chdir('/home/david/Desktop/projects/thesis/')
sys.path.append('/home/david/Desktop/projects/thesis/')

from downstream_classification.utils.registration import register_all_peaks_to_minimal_peak, extract_start

individual_signals_path = './downstream_classification/data/individual-signals/'
individual_registered_signals_path = './downstream_classification/data/individual-signals-registered/'

individual_signals = os.listdir(individual_signals_path)
signals = [x.split('.')[0] for x in individual_signals]

N = len(signals)
t = 0
for jj in tqdm(range(N)):
    signal = np.load(individual_signals_path+f'/{signals[jj]}.npy')
    signal_filled = np.nan_to_num(signal, nan = 0)

    registered_signal = register_all_peaks_to_minimal_peak(signal_filled)

    # if want to test
    # ----
    # plot_12_channels_vertically(signal_filled, f'{jj}-Original')
    # plot_12_channels_vertically(registered_signal, f'{jj}-Registered')
    
    # signal = np.load(f'{individual_registered_signals_path}/{signals[jj]}.npy')
    # plot_12_channels_vertically(registered_signal, f'{jj}-Registered loaded')


    np.save(f'{individual_registered_signals_path}/{signals[jj]}.npy', registered_signal)

    if ((t%1000)==0):
        logging.info(f'Finished {t} signals (out of {N} total)')
    t += 1
