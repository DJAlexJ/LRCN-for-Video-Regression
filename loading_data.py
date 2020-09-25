import os
import pandas as pd
import torch
from torchvision import datasets, transforms
from config import MARKUP_PATH, TRAINING_PATH

markup = pd.read_excel(MARKUP_PATH)

def load_data(dir_names, verbose=False, batch_size=6, sequence_size=16):
    """
    dir_names: list of directory names containing video frames
    verbose: add verbosity to loading process
    batch_size: loading n sequences per function call
    sequence_size: size of each frame sequence
    
    return:
    images: torch.Tensor
    labels: torch.Tensor
    dir_names: list
    """
    transform = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])
    
    bs = batch_size
    lst_images = []
    labels = []
    for subdir in dir_names:
        if not os.path.isdir(f'{TRAINING_PATH}/{subdir}'):
            continue
        if bs == 0:
            break
        if verbose:
            print(f"processing {subdir}")
        dataset = datasets.ImageFolder(f"{TRAINING_PATH}/{subdir}", transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=sequence_size, shuffle=False)   #all sequences inside folder are approximately 16 frames
        name = name = '_'.join(subdir.split('_')[:-1])   #getting name to find its score in markup.csv
        try:
            labels.append(markup[markup.Title == name].Score.values[0])  
            images, _ = next(iter(dataloader))
            if images.size()[0] < sequence_size:
                images = torch.cat((images, torch.zeros((sequence_size-images.size()[0], 3, 224, 224)))) #Appending zero tensors to make all sequences equal
                
            lst_images.append(images)
            bs -= 1
        except:
            print("File is not found in the markup")
    
    dir_names = dir_names[batch_size:] #delete loaded subdir names to remove these frames from futher sequences
    images = torch.stack(lst_images)
    labels = torch.tensor(labels)
    return images, labels, dir_names

