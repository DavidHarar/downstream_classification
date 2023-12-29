import pickle

import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import random
import time


from dataloader.DataGenerator import DataGenerator
from modeling.Transformer import TransformerEncoderDownstream
from modeling.Inception import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange
from tqdm.auto import tqdm

from utils.metrics import *
from utils.plots import *

import logging


def trainer(seed, 
            metadata_file_path,
            data_folder_path, 
            # architecture
            targets,
            dropout, 
            clip,
            scale,
            depth,
            # training
            batch_size, 
            n_epochs,
            model_saving_path,
            # model type
            eval_metric = 'loss',
            weight_decay = 0,
            lr = 0.001,
            verbosity = False,
            patience=np.inf,
            loss_function_weight = None,
            continue_training=False,
         ):
    """
    params:
    - continue_training: Should the training continue from an existing model. 
        TODO: Needs to be fixed. Currently uses `model_saving_path`, but the new definition of `model_saving_path` is 
        a foder rather than the actual path of the model.
    - weight_decay: L2 panelty
    - loss_function_weight: Weight to give to the positive class. In case of multilabel, use vector of the same length 
        of the number of classes. 
            Example: loss_function_weight = torch.tensor([33.0]).to('mps')
    """
    # TODO: make it more elegant
    target_str = targets[0] if isinstance(targets,list) else targets

    # Init experiment directory
    if not os.path.exists(model_saving_path):
        os.makedirs(model_saving_path)

    # init log
    logging.basicConfig(filename=f"./{model_saving_path}/log.log", level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M')
    logging.info("Fit the preprocessing pipeline")
    

    # TODO: add into the model
    output_dimension = len(targets)
    
    # Fix randomness
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Training using device: {device}')

    logging.info(f'Creating generators')
    train_generator = DataGenerator(
        data_folder_path=data_folder_path,
        metadata_file_path=metadata_file_path,
        targets=targets,
        sample='train',
        seed=seed,
        batch_size=batch_size,
        shuffle = True
    )

    validation_generator = DataGenerator(
        data_folder_path=data_folder_path,
        metadata_file_path=metadata_file_path,
        targets=targets,
        sample='validation',
        seed=seed,
        batch_size=batch_size,
        shuffle = True
    )

    # create a model
    if depth == 1:
        model = DownstreamInception(dropout, scale, num_inputs=12).to(device)
    else:
        model = DownstreamInceptionResnet(dropout, scale, depth, num_inputs=12).to(device)
    
    logging.info(f'The model has {count_parameters(model):,} trainable parameters')
    logging.info('* Model:')
    logging.info('* -----------')
    logging.info(model)
    logging.info('* -----------')

    if continue_training:
        model.load_state_dict(torch.load(f'{model_saving_path}.pt'))
        print('model params have been loaded')

    optimizer = optim.AdamW(model.parameters(), weight_decay=weight_decay, lr=lr)

    if len(targets)>1:
        criterion = nn.CrossEntropyLoss()
    else:
        if loss_function_weight:
            criterion = nn.BCELoss(loss_function_weight)
        else: 
            criterion = nn.BCELoss()

    # Training
    # ------------------
    # initiate values
    best_valid_loss = float('inf')
    best_aucpr = 0
    best_rocauc = 0
    
    rocauc_given_best_aucpr = 0
    epochs_without_update = 0   
    best_recall_for_precision = 0

    losses = {'train':[],
              'validation':[]}
    
    logging.info(f'Evaluating model based on: {eval_metric}')
    
    logging.info(f'Training..\n')
    

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
                           device
                           )
        

        # evaluate
        valid_loss, y_val, y_val_pred = evaluate_epoch(model, 
                              validation_generator, 
                              criterion, 
                              device                              
                              )
        
        # store losses
        losses['train'].append(train_loss)
        losses['validation'].append(valid_loss)
        
        # Plot distributions
        y_train_prediction = pd.DataFrame({'y_train': y_train,
                                           'y_train_pred':y_train_pred})
        
        y_valication_prediction = pd.DataFrame({'y_val': y_val,
                                               'y_val_pred':y_val_pred})
        


        # save loss
        if model_saving_path:
            with open(f'{model_saving_path}/loss.pkl', 'wb') as f:
                pickle.dump(losses, f)
        
        # take ending time
        end_time = time.time()
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        aucpr  = PRAUC(y_val, y_val_pred)
        rocauc = ROCAUC(y_val, y_val_pred)
        recall_for_precision, threshold = MaxRecall_for_MinPrecision(y_val, y_val_pred, min_precision=0.4)


        # patience
        if eval_metric == 'loss':
            if valid_loss < best_valid_loss:
                epochs_without_update = 0
            else:
                epochs_without_update+=1
        if eval_metric == 'aucpr':
            if aucpr > best_aucpr:
                epochs_without_update = 0
            else:
                epochs_without_update+=1
        if eval_metric == 'rocauc':
            if rocauc > best_rocauc:
                epochs_without_update = 0
            else:
                epochs_without_update+=1
        if eval_metric == 'recall_for_precision':
            if recall_for_precision > best_recall_for_precision:
                epochs_without_update = 0
            else:
                epochs_without_update+=1

        # break if patience condition takes place
        if epochs_without_update>patience:
            break
        
        # terminal plots saving
        if eval_metric == 'loss':
            if valid_loss < best_valid_loss:
                if model_saving_path is not None:
                    fig, axs = plt.subplots(1, 2, figsize = (10,3))
                    sns.histplot(data = y_train_prediction, x = 'y_train_pred', hue = 'y_train', common_norm=False, stat='probability', ax=axs[0])
                    sns.histplot(data = y_valication_prediction, x = 'y_val_pred', hue = 'y_val', common_norm=False, stat='probability', ax=axs[1])
                    axs[0].set_title('Scores Distribution on the Training Set')
                    axs[1].set_title('Scores Distribution on the Validation Set')
                    axs[0].axvline(threshold, c='r')
                    axs[1].axvline(threshold, c='r')
                    fig.savefig(f'{model_saving_path}/epoch_{epoch}_val_loss.jpg')
                    plt.cla()
                    # plt.show()

        if eval_metric == 'aucpr':
            if aucpr > best_aucpr:
                if model_saving_path is not None:
                    fig, axs = plt.subplots(1, 2, figsize = (10,3))
                    sns.histplot(data = y_train_prediction, x = 'y_train_pred', hue = 'y_train', common_norm=False, stat='probability', ax=axs[0])
                    sns.histplot(data = y_valication_prediction, x = 'y_val_pred', hue = 'y_val', common_norm=False, stat='probability', ax=axs[1])
                    axs[0].set_title('Scores Distribution on the Training Set')
                    axs[1].set_title('Scores Distribution on the Validation Set')
                    axs[0].axvline(threshold, c='r')
                    axs[1].axvline(threshold, c='r')
                    fig.savefig(f'{model_saving_path}/epoch_{epoch}_val_aucpr.jpg')
                    plt.cla()
                    # plt.show()

        if eval_metric == 'rocauc':
            if rocauc > best_rocauc:
                if model_saving_path is not None:
                    fig, axs = plt.subplots(1, 2, figsize = (10,3))
                    sns.histplot(data = y_train_prediction, x = 'y_train_pred', hue = 'y_train', common_norm=False, stat='probability', ax=axs[0])
                    sns.histplot(data = y_valication_prediction, x = 'y_val_pred', hue = 'y_val', common_norm=False, stat='probability', ax=axs[1])
                    axs[0].set_title('Scores Distribution on the Training Set')
                    axs[1].set_title('Scores Distribution on the Validation Set')
                    axs[0].axvline(threshold, c='r')
                    axs[1].axvline(threshold, c='r')
                    fig.savefig(f'{model_saving_path}/epoch_{epoch}_val_rocauc.jpg')
                    plt.cla()
                    # plt.show()

        if eval_metric == 'recall_for_precision':
            if recall_for_precision > best_recall_for_precision:
                if model_saving_path is not None:
                    fig, axs = plt.subplots(1, 2, figsize = (10,3))
                    sns.histplot(data = y_train_prediction, x = 'y_train_pred', hue = 'y_train', common_norm=False, stat='probability', ax=axs[0])
                    sns.histplot(data = y_valication_prediction, x = 'y_val_pred', hue = 'y_val', common_norm=False, stat='probability', ax=axs[1])
                    axs[0].set_title('Scores Distribution on the Training Set')
                    axs[1].set_title('Scores Distribution on the Validation Set')
                    axs[0].axvline(threshold, c='r')
                    axs[1].axvline(threshold, c='r')
                    fig.savefig(f'{model_saving_path}/epoch_{epoch}_val_recall_for_precision.jpg')
                    plt.cla()


        # update best values
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            # save if val_loss is criterion for saving
            if eval_metric == 'loss':
                if model_saving_path:
                    torch.save(model.state_dict(), f'{model_saving_path}/model_val_loss.pt')

        if aucpr > best_aucpr:
            best_aucpr = aucpr
            # save if val_loss is criterion for saving
            if eval_metric == 'aucpr':
                if model_saving_path:
                    torch.save(model.state_dict(), f'{model_saving_path}/model_val_aucpr.pt')

        if rocauc > best_rocauc:
            best_rocauc = rocauc
            # save if val_loss is criterion for saving
            if eval_metric == 'rocauc':
                if model_saving_path:
                    torch.save(model.state_dict(), f'{model_saving_path}/model_val_rocauc.pt')

        if recall_for_precision > best_recall_for_precision:
            best_recall_for_precision = recall_for_precision
            # save if val_loss is criterion for saving
            if eval_metric == 'recall_for_precision':
                if model_saving_path:
                    torch.save(model.state_dict(), f'{model_saving_path}/model_val_recall_for_precision.pt')

        if eval_metric == 'recall_for_precision':
            if recall_for_precision == best_recall_for_precision:
                best_value = best_recall_for_precision
                update_about_it = True
        if eval_metric == 'aucpr':
            if aucpr == best_aucpr:
                best_value = best_aucpr
                update_about_it = True
        if eval_metric == 'rocauc':
            if rocauc == best_rocauc:
                best_value = best_rocauc
                update_about_it = True
        if eval_metric == 'loss':
            if valid_loss == best_valid_loss:
                best_value = best_valid_loss
                update_about_it = True
        
        # Summarize epoch results
        logging.info('-'*45)
        logging.info(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        if update_about_it:
            logging.info(f'\t New best val_rocauc loss was found, current best value is {np.round(best_value,5)}')
            update_about_it = False
        logging.info(f'\t Train Loss: {train_loss:.3f}')
        logging.info(f'\t Val. Loss: {valid_loss:.3f}')
        logging.info(f'\t ROC-AUC: {rocauc:.3f}')
        logging.info(f'\t PR-AUC: {aucpr:.3f}')
        logging.info(f'\t Recall for 0.4 precision: {recall_for_precision:.3f}')
        logging.info(f'\t Best Val. Loss: {best_valid_loss:.3f}')
        logging.info(f'\t Best ROC-AUC: {best_rocauc:.3f}')
        logging.info(f'\t Best PR-AUC: {best_aucpr:.3f}')
        logging.info(f'\t Best Recall for 0.4 precision: {best_recall_for_precision:.3f}')
        logging.info('-'*45)
    
    # get best model
    if depth == 1:
        best_model = DownstreamInception(dropout, scale, num_inputs=12).to(device)
    else:
        best_model = DownstreamInceptionResnet(dropout, scale, depth, num_inputs=12).to(device)
    if eval_metric == 'recall_for_precision':
        best_model.load_state_dict(torch.load(f'{model_saving_path}/model_val_recall_for_precision.pt'))
    if eval_metric == 'aucpr':
        best_model.load_state_dict(torch.load(f'{model_saving_path}/model_val_aucpr.pt'))
    if eval_metric == 'rocauc':
        best_model.load_state_dict(torch.load(f'{model_saving_path}/model_val_rocauc.pt'))
    if eval_metric == 'loss':
        best_model.load_state_dict(torch.load(f'{model_saving_path}/model_val_loss.pt'))

    # save additional plots
    validation_data = pd.read_csv(metadata_file_path,index_col=0)
    validation_data = validation_data[validation_data['sample'] == 'validation'].reset_index(drop=True)
    
    # best_model = best_model.to('cpu')

    predictions, nonmissing_leads = predict(
        device = device,
        readings= validation_data['reading'],
        model = best_model,
        data = validation_data,
        targets=targets,
        fillna=0,
        leads = ['LI', 'LII', 'LIII', 'aVF', 'aVL', 'aVR','V1','V2','V3','V4','V5','V6'],
        data_path=data_folder_path,
    )
    validation_data['y_pred'] = predictions
    post_reg_analysis(
        data = validation_data,
        y_true_column=target_str,
        y_pred_column='y_pred',
        saving_path=model_saving_path
    )

    logging.shutdown()

    return best_aucpr, rocauc_given_best_aucpr



def train_epoch(
        model, 
        iterator, 
        optimizer, 
        criterion, 
        clip, 
        device 
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
        X, y, _ = next(it)
        y=np.squeeze(y,-1)

        X = np.float32(X)
        # don't run if there are NaNs
        if np.isnan(X).sum()>0:
            print('skipping because of NaNs')
            continue
        y = np.float32(y)
        y = np.expand_dims(y,-1)

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
                   device):
    
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
            X, y, _ = next(it)
            y=np.squeeze(y,-1)

            X = np.float32(X)
            
            # don't run if there are NaNs
            if np.isnan(X).sum()>0:
                print('skipping because of NaNs')
                continue
            y = np.float32(y)
            y = np.expand_dims(y,-1)

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