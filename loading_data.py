import os
import pandas as pd
import torch
from torchvision import datasets, transforms
from config import MARKUP_PATH, TRAINING_PATH

markup = pd.read_excel(MARKUP_PATH)

def load_data(dir_names, verbose=False, batch_size=6):
    """
    dir_names: list of directory names containing video frames
    verbose: add verbosity to loading process
    batch_size: loading n sequences per function call
    
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
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)   #all sequences inside folder are approximately 16 frames
        name = name = '_'.join(subdir.split('_')[:-1])   #getting name to find its score in markup.csv
        try:
            labels.append(markup[markup.Title == name].Score.values[0])  
            images, _ = next(iter(dataloader))
            if images.size()[0] < 16:
                images = torch.cat((images, torch.zeros((16-images.size()[0], 3, 224, 224)))) #Appending zero tensors to make all sequences equal
                
            lst_images.append(images)
            dir_names.remove(subdir)  #delete subdir name to remove these frames from futher sequences
            bs -= 1
        except:
            print("File is not found in the markup")
            
    images = torch.stack(lst_images)
    labels = torch.tensor(labels)
    return images, labels, dir_names

