#!/usr/bin/env python
# coding: utf-8

# In[11]:


import preprocessing as prep
import torch
from loading_data import load_data
from model import CNN, LRCN
from config import PREDICTION_PATH, TEST_TRAILER_NAME, PATH_MODEL_CHECKPOINT


# In[33]:


def get_prediction(movie_name, verbose=True):
    prep.preprocess(movie_name, train=False, n_subclips=1, verbose=verbose)
    model = torch.load(PATH_MODEL_CHECKPOINT, map_location=torch.device('cpu'))
    model.to('cpu')
    model.eval()
    name = movie_name.rsplit('.', 1)[0]+'_0'
    images = load_data([name], train=False, verbose=verbose, batch_size=1)
    prediction = model.forward(images).detach().unsqueeze(-1).item()
    return prediction

if __name__ == '__main__':
    get_predict(TEST_TRAILER_NAME)


# In[ ]:




