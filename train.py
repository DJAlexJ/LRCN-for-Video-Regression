import numpy as np
import pandas as pd
import random
import os
import torch
from torchvision import datasets
#from torch.utils.tensorboard import SummaryWriter
from model import LRCN
from loading_data import load_data
from config import TRAINING_PATH, PATH_MODEL_CHECKPOINT

def init_model(device='cpu'):
    model = LRCN()
    loss = torch.nn.MSELoss()
    mae = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=9e-4)
    model.to(device) 
    return model, (loss, mae), optimizer

def create_test(dir_names, device='cpu', verbose=False, batch_size=100):
    X_test, y_test, dir_names = load_data(dir_names, verbose=True, batch_size=batch_size)
    X_test = X_test.to(device).float()
    y_test = y_test.to(device).float()
    
    return X_test, y_test, dir_names

def train(dir_names, X_test, y_test, n_epoch=5, batch_size=10, device='cpu', use_checkpoint=False, use_tensorb=False, verbose=False):
    
#     if use_tensorb:
#         tb = SummaryWriter()
        
    model, Loss, optimizer = init_model()
    loss, mae = Loss[0], Loss[1]
    
    dir_names = list(filter(lambda x: os.path.isdir(f"{TRAINING_PATH}/{x}"), dir_names)) #Filtering waste files
    random.shuffle(dir_names)

    train_mse_history = []
    test_mse_history = []
    test_mae_history = []

    learning_dir_names = dir_names.copy()
    #Training model
    for epoch in range(n_epoch):
        dir_names = learning_dir_names.copy()
        train_loss = 0
        for i in range(0, len(learning_dir_names), batch_size):
            optimizer.zero_grad()
            X_batch, y_batch, dir_names = load_data(dir_names, verbose, batch_size=batch_size)  

            X_batch = X_batch.to(device).float()
            y_batch = y_batch.to(device).float()

            preds = model.forward(X_batch).view(y_batch.size()[0])
            loss_value = loss(preds, y_batch)
            loss_value.backward()
            
            train_loss += loss_value.data.cpu()
            optimizer.step()
            
        train_mse_history.append(train_loss)

        with torch.no_grad():
            test_preds = model.forward(X_test).view(y_test.size()[0])
            test_mse_history.append(loss(test_preds, y_test).data.cpu())
            test_mae_history.append(mae(test_preds, y_test).data.cpu())

        #Saving checkpoint
        if use_checkpoint:
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'mse': test_mse_history[-1],
                    'mae': test_mae_history[-1],
                    }, PATH_MODEL_CHECKPOINT)

#         if use_tensorb:
#             tb.add_scalar('MSE', test_rmse_history[-1], epoch)
#             tb.add_scalar('MAE', test_mae_history[-1], epoch)

        print(f"{epoch+1}: MSE = {test_mse_history[-1]}, MAE = {test_mae_history[-1]}")
       
     #if use_tensorb:
     #    tb.close()
        
    return [train_mse_history, test_mse_history]

