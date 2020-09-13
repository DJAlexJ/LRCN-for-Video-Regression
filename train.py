#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import random
import os
import torch
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from model import LRCN
from loading_data import load_data

def init_model(device='cpu'):
    model = LRCN()
    loss = torch.nn.MSELoss()
    mae = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=9e-4)
    model.to(device) 
    return model, (loss, mae), optimizer

def create_test(dir_names, device='cpu', verbose=True, batch_size=100):
    X_test, y_test, dir_names = load_data(dir_names, verbose=True, batch_size=100)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    
    return X_test, y_test

def train():

    tb = SummaryWriter()
    model, Loss, optimizer = init_model()
    loss, mae = Loss[0], Loss[1]
    dir_names = os.listdir(f'{TRAINING_PATH}')
    random.shuffle(dir_names)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    X_test, y_test = create_test(dir_names, device, verbose, batch_size)

    batch_size = 10
    test_rmse_history = []
    test_mae_history = []
    PATH = "/content/checkpoint.tar"
    checkpoint = True
    tensorboard = True

    learning_dir_names = dir_names.copy()
    #Training model
    for epoch in range(7):
        dir_names = learning_dir_names.copy()
        for i in range(0, len(learning_dir_names), batch_size):
            optimizer.zero_grad()
            X_batch, y_batch, dir_names = load_data(dir_names, verbose=True, batch_size=batch_size)  

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            preds = model.forward(X_batch).view(y_batch.size()[0])
            loss_value = torch.sqrt(loss(preds, y_batch)) #Converting to RMSE
            loss_value.backward()

            optimizer.step()

        with torch.no_grad():
            test_preds = model.forward(X_test).view(y_test.size()[0])
            test_rmse_history.append(loss(test_preds, y_test).data.cpu())
            test_mae_history.append(mae(test_preds, y_test).data.cpu())

        #Saving checkpoint
        if checkpoint:
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'rmse': test_rmse_history[-1],
                    'mae': test_mae_history[-1],
                    }, PATH)

        if tensorboard:
            tb.add_scalar('RMSE', test_rmse_history[-1], epoch)
            tb.add_scalar('MAE', test_mae_history[-1], epoch)
        print(f"{epoch+1}: RMSE = {test_rmse_history[-1]}")
        print(f"{epoch+1}: MAE = {test_mae_history[-1]}")

    tb.close()
    return test_history

