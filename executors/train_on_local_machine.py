import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import random
import time


from dataloader.DataLoader import DataGenerator
from modeling.Transformer import TransformerEncoderDownstream
from modeling.Inception import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange
from tqdm.auto import tqdm
from utils.metrics import *

def trainer(seed, 
            metadata_file_path,
            data_folder_path, 
            fillna,
            # architecture
            targets,
            input_dimension,  
            hidden_dimmension, attention_heads, 
            encoder_number_of_layers,       # Only encoder 
            positional_encodings, dropout, clip,
            # training
            batch_size, 
            n_epochs,
            saving_path,
            # model type
            model_type = 'Transformer',
            minimal_number_of_leads=None,
            leads = ['LI', 'LII', 'LIII', 'aVF', 'aVL', 'aVR','V1','V2','V3','V4','V5','V6'],
            eval_metric = 'loss',
            weight_decay = 0,
            lr = 0.001,
            verbosity = False,
            patience=np.inf,
            loss_function_weight = None,
            continue_training=False,
         ):
    
    output_dimension = len(targets)
    
    # create a mapping for all possible leads
    leads_and_their_indices = {x:i for i,x in enumerate(['LI', 'LII', 'LIII', 'aVF', 'aVL', 'aVR','V1','V2','V3','V4','V5','V6'])}
    relevant_leads_indices = np.array([leads_and_their_indices[x] for x in leads])


    # Fix randomness
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device("mps")
    print('\n')
    print(f'training using device: {device}')
    print('\n')

    
    train_generator = DataGenerator(
        metadata_file_path= metadata_file_path,                 # path to metadata file
        data_folder_path = data_folder_path,                    # path to individual signals
        sample='train',                                         # sample we want to create a generator to. Either train, validation or test
        targets=targets,                                        # list of targets we want train on
        batch_size=batch_size,                                  # batch size
        shuffle=True,                                            # Whether to shuffle the list of IDs at the end of each epoch.
        fillna = fillna,
        minimal_number_of_leads = minimal_number_of_leads,
        leads = relevant_leads_indices
                    )


    validation_generator = DataGenerator(
        metadata_file_path= metadata_file_path,                 # path to metadata file
        data_folder_path = data_folder_path,                    # path to individual signals
        sample='validation',                                    # sample we want to create a generator to. Either train, validation or test
        targets=targets,                                        # list of targets we want train on
        batch_size=batch_size,                                  # batch size
        shuffle=True,                                            # Whether to shuffle the list of IDs at the end of each epoch.
        fillna = fillna,
        minimal_number_of_leads=minimal_number_of_leads,
        leads = relevant_leads_indices
                    )

    test_generator = DataGenerator(
        metadata_file_path= metadata_file_path,                 # path to metadata file
        data_folder_path = data_folder_path,                    # path to individual signals
        sample='test',                                          # sample we want to create a generator to. Either train, validation or test
        targets=targets,                                        # list of targets we want train on
        batch_size=batch_size,                                  # batch size
        shuffle=True,                                            # Whether to shuffle the list of IDs at the end of each epoch.
        fillna = fillna,
        minimal_number_of_leads=minimal_number_of_leads,
        leads = relevant_leads_indices
                                )
    
    
    # create a model
    if model_type == 'Transformer':
        model = TransformerEncoderDownstream(
            input_dimension, output_dimension,                      # output_dim is defined by the length of targets 
            hidden_dimmension, attention_heads, 
            encoder_number_of_layers, positional_encodings, dropout
            ).to(device)
        
    if model_type == 'Inception':
        model = DownstreamInception(dropout, num_inputs=len(leads)).to(device)
        if continue_training:
            model.load_state_dict(torch.load(f'{saving_path}.pt'))
            print('model params have been loaded')

    # optimizer = optim.Adam(model.parameters())
    optimizer = optim.AdamW(model.parameters(), weight_decay=weight_decay, lr=lr)

    if len(targets)>1:
        criterion = nn.CrossEntropyLoss()
    else:
        if loss_function_weight:
            criterion = nn.BCELoss(loss_function_weight)
        else: 
            criterion = nn.BCELoss()

    if verbosity:
        print(f'The model has {count_parameters(model):,} trainable parameters')
        print(model)


    # Training
    # ------------------
    best_valid_loss = float('inf')
    best_aucper = 0
    losses = {'train':[],
              'validation':[]}
    # for patience
    epochs_without_update = 0

    for epoch in range(n_epochs):
        
        # take starting time
        start_time = time.time()
        
        # train
        # number of leads (dictates training scheme):
        train_loss, y_train, y_train_pred = train_epoch(model, 
                           train_generator, 
                           optimizer, 
                           criterion, 
                           clip, 
                           device,
                           model_type = model_type
                           )
        

        # evaluate
        valid_loss, y_val, y_val_pred = evaluate_epoch(model, 
                              validation_generator, 
                              criterion, 
                              device,
                              output_dimension,
                              model_type = model_type
                              )
        
        # store losses
        losses['train'].append(train_loss)
        losses['validation'].append(valid_loss)

        # Plot distributions
        y_train_prediction = pd.DataFrame({'y_train': y_train,
                                           'y_train_pred':y_train_pred})
        
        y_valication_prediction = pd.DataFrame({'y_val': y_val,
                                               'y_val_pred':y_val_pred})

        fig, axs = plt.subplots(1, 2, figsize = (10,3))
        sns.histplot(data = y_train_prediction, x = 'y_train_pred', hue = 'y_train', common_norm=False, stat='probability', ax=axs[0])
        sns.histplot(data = y_valication_prediction, x = 'y_val_pred', hue = 'y_val', common_norm=False, stat='probability', ax=axs[1])
        axs[0].set_title('Scores Distribution on the Training Set')
        axs[1].set_title('Scores Distribution on the Validation Set')
        plt.show()



        with open(f'{saving_path}_loss.pkl', 'wb') as f:
            pickle.dump(losses, f)
        

        # take ending time
        end_time = time.time()
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        aucpr  = PRAUC(y_val, y_val_pred)
        aucroc = ROCAUC(y_val, y_val_pred)
        tprforbudget = get_tpr_for_fpr_budget(y_val, y_val_pred, fpr_budget = 0.6)
        

        # keep best model
        if eval_metric == 'loss':
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if saving_path:
                    print(f'New best validation loss was found, current best valid loss is {np.round(best_valid_loss,4)}')
                    torch.save(model.state_dict(), f'{saving_path}.pt')
                
                # if we update then patience restarts
                epochs_without_update = 0
            else:
                epochs_without_update+=1
                

        if eval_metric == 'aucpr':
            if aucpr>best_aucper:
                best_aucper = aucpr
                rocauc_given_best_aucpr = aucroc
                tprforbudget_given_best_aucpr = tprforbudget
                
                if saving_path:
                    print(f'New best aucpr was found, current best aucpr is {np.round(best_aucper,4)}')
                    torch.save(model.state_dict(), f'{saving_path}.pt')
                
                # if we update then patience restarts
                epochs_without_update = 0
            else:
                epochs_without_update+=1
        
        if epochs_without_update>patience:
            break

        # print summary
        print('-'*45)
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f}')
        print(f'\t ROC-AUC: {aucroc:.3f}')
        print(f'\t PR-AUC: {aucpr:.3f}')
        print(f'\t TPR for FPR=0.6 Budget: {tprforbudget:.3f}')
        print(f'\t Best Val. Loss: {best_valid_loss:.3f}')
        print('-'*45)
    
        # plt.plot(losses['train'])    
        # plt.plot(losses['validation'])    
        # plt.show()
    
    return best_aucper, rocauc_given_best_aucpr, tprforbudget_given_best_aucpr



def train_epoch(
        model, 
        iterator, 
        optimizer, 
        criterion, 
        clip, 
        device, 
        model_type,
          ):
    
    # set model on training state and init epoch loss    
    model.train()
    epoch_loss = 0
    denominator = 0
    ys = []
    outputs = []

    # get number of iterations for the progress bar. 
    it = iter(iterator)
    T = len(iterator)
    # set progress bar
    t = trange(T, desc='Within epoch loss (training)', leave=True)

    for i in t:
        # get data
        X, y = next(it)
        y=np.squeeze(y,-1)

        if model_type == 'Transformer':
            # fix shapes ( -> [length, batch_size, channels])
            X = np.float32(np.transpose(X, axes=(2,0,1)))
        else:
            X = np.float32(X)
        # don't run if there are NaNs
        if np.isnan(X).sum()>0:
            print('skipping because of NaNs')
            continue
        y = np.float32(y)


        X = torch.from_numpy(X)
        y = torch.from_numpy(y)

        X = X.to(device)
        y = y.to(device)
        
        # step
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

        j = np.round(epoch_loss/(i+1),5)
        t.set_description(f"Within epoch loss (training) {j}")
        t.refresh() # to show immediately the update

        # keep y and output
        ys += y.to('cpu').numpy().reshape(-1).tolist()
        outputs += output.detach().to('cpu').numpy().reshape(-1).tolist()
            

        denominator+=1

    return epoch_loss / denominator, np.array(ys), np.array(outputs)

def evaluate_epoch(model, 
                   iterator, 
                   criterion,
                   device,
                 # training paradigm is defined by output_dimension
                 output_dimension,
                 model_type):
    
    # set model on training state and init epoch loss    
    model.eval()
    epoch_loss = 0
    denominator = 0

    # get number of iterations for the progress bar. n_iters can be set to bypass it
    it = iter(iterator)
    T = len(iterator)
    # set progress bar
    t = trange(T, desc='Within epoch loss (validation)', leave=True)
    
    
    ys = []
    outputs = []

    with torch.no_grad():
        for i in t:

            # get data
            X, y = next(it)
            y=np.squeeze(y,-1)

            if model_type == 'Transformer':
                # fix shapes ( -> [length, batch_size, channels])
                X = np.float32(np.transpose(X, axes=(2,0,1)))
            else:
                X = np.float32(X)
            # don't run if there are NaNs
            if np.isnan(X).sum()>0:
                print('skipping because of NaNs')
                continue
            y = np.float32(y)


            X = torch.from_numpy(X)
            y = torch.from_numpy(y)

            X = X.to(device)
            y = y.to(device)

            output = model(X)
            loss = criterion(output, y)
            epoch_loss += loss.item()

            j = np.round(epoch_loss/(i+1),5)
            t.set_description(f"Within epoch loss (validation) {j}")
            t.refresh() # to show immediately the update

            # update values
            denominator+=1


            # keep y and output
            ys += y.to('cpu').numpy().reshape(-1).tolist()
            outputs += output.to('cpu').numpy().reshape(-1).tolist()
            
    return epoch_loss / denominator, np.array(ys), np.array(outputs)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)