import numpy as np
import pandas as pd
import random
import os
import torch
from torchvision import datasets
import argparse
import preprocessing as prep
import loading_data as ld
#from torch.utils.tensorboard import SummaryWriter
from model import LRCN
from loading_data import load_data
from config import TRAINING_PATH


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-i", "--input_path", type=str, help="Path with training trailers", required=True)
    arg("-l", "--loss", type=str, default='mse', help="Loss function")
    arg("-a", "--activation", type=str, default='relu', help="Activation function")
    arg("-ne", "--n_epoch", type=int, default=7, help="Number of epochs")
    arg("-bs", "--batch_size", type=int, default=10, help="Batch size")
    arg("-tbs", "--test_batch_size", type=int, default=1, help="Batch size for test data")
    arg("-lr", "--learning_rate", type=float, default=3e-4, help="Learning rate")
    arg("-ns", "--n_subclips", type=int, default=3, help="Number of sublcips extracted from each video")
    arg("-sw", "--save_weights", default=True, help="Whether to save model weights")
    arg("-prep", "--preprocessing", default=True, help="Whether to perform preprocessing")
    arg("-v", "--verbose", default=False, help="Verbosity")
    arg("-d", "--device", default='cpu', type=str, help="Device for computations")
    arg("-vis", "--visualize", default=False, action="store_true", help="Visualize results")
    return parser.parse_args()

def train():
    args = get_args()
    
    if args.preprocessing == True:
        prep.movies_preprocess(os.listdir(args.input_path), train=True, n_subclips=args.n_subclips, verbose=args.verbose)
                               
    model = LRCN(activation=args.activation)
    model.to(args.device)
                               
    dir_names = os.listdir(TRAINING_PATH)
    print(dir_names)

    X_test, y_test, dir_names = ld.load_data(dir_names, train=True, verbose=args.verbose, batch_size=args.test_batch_size)
    
    model.fit(dir_names, X_test, y_test, lr=args.learning_rate,                     loss_name=args.loss, n_epoch=args.n_epoch,
                batch_size=args.batch_size, device=args.device,
                verbose=args.verbose, saving_results=args.save_weights)
                               
                               


if __name__ == "__main__":
    train()

