import numpy as np
import pandas as pd
import random
import os
import torch
from torchvision import datasets
import argparse
#from torch.utils.tensorboard import SummaryWriter
from model import LRCN
from loading_data import load_data
from config import TRAINING_PATH, PATH_MODEL_CHECKPOINT


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-i", "--input_path", type=Path, help="Path with training trailers", required=True)
    arg("-out", "--output_path", type=Path, help="Path to output file with movie scores", required=True)
    arg("-l", "--loss", type=str, default='mse', help="Loss function")
    arg("-a", "--activation", type=str, default='relu, help="Activation function")
    arg("-ne", "--n_epoch", type=int, default=7, help="Number of epochs")
    arg("-bs", "--batch_size", type=int, default=10, help="Batch size")
    arg("-lr", "--learning_rate", type=int, default=3e-4, help="Learning rate")
    arg("-ns", "--n_subclips", type=int, default=3, "Number of sublcips extracted from each video")
    arg("-prep", "--preprocessing", default=True, help="Whether to perform preprocessing")
    arg("-v", "--verbose", default=False, help="Verbosity")
    arg("-d", "--device", default='cpu', type=str, help="Device for computations")
    arg("-vis", "--visualize", action="store_true", help="Visualize results")
  

def train():
    args = get_args()
    
    hparams = {
            "input_path": args.input_path,
            "output_path": args.output_path,
            "loss": args.loss,
            "activation": args.activation,
            "n_epoch": args.n_epoch,
            "batch_size": args.batch_size,
            "n_subclips": args.n_subclips,
            "preprocessing": args.preprocessing,
            "verbose": args.verbose,
            "device": args.device,
            "visualize": args.visualize,
        }
    
    if hparams['preprocessing'] == True:
        prep.movies_preprocess(os.listdir(hparams['input_path'], train=True, n_subclips=hparams['n_subclips'], verbose=hparams['verbose'])
                               
    model = LRCN(activation=hparams['activation'])
    model.to(hparams['device'])
                               
    dir_names = os.listdir(TRAINING_PATH)
    dir_names = list(filter(lambda x: os.path.isdir(f"{TRAINING_PATH}/{x}"), dir_names)) #Filtering waste files

    X_test, y_test, dir_names = ld.load_data(dir_names, train=True, verbose=hparams['verbose'], batch_size=1)
    model.train()
                               
                               

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
                               

if __name__ == "__main__":
    train()

