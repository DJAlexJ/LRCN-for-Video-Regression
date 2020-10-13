import pytest
import torch
import loading_data as ld
import preprocessing as prep
import shutil
import os
from config import TRAINING_PATH, PREDICTION_PATH

@pytest.fixture(scope='module')
def train_prep():
    yield prep.preprocess('86.Малыш на драйве.mp4', train=True, verbose=True)
    shutil.rmtree(TRAINING_PATH)
    
@pytest.fixture(scope='module')
def test_prep():
    yield prep.preprocess('85.Дикая.mp4', train=False, verbose=True)
    shutil.rmtree(PREDICTION_PATH)
    
    
@pytest.mark.parametrize('path, training', 
                        [(TRAINING_PATH, True), 
                         (PREDICTION_PATH, False),
                        ])
def test_loading(train_prep, test_prep, path, training):
    if training == True:
        dir_names = os.listdir(path)
        dir_names = list(filter(lambda x: os.path.isdir(f"{path}/{x}"), dir_names))
        len_dir_names = len(dir_names)
        images, labels, dir_names = ld.load_data(dir_names, train=True, batch_size=2, sequence_size=15)
        assert len(dir_names) == len_dir_names-2
        assert images.size() == (2, 15, 3, 224, 224)
        assert len(labels) == 2
    else:
        dir_names = os.listdir(path)
        dir_names = list(filter(lambda x: os.path.isdir(f"{path}/{x}"), dir_names))
        images, names = ld.load_data(dir_names, train=False, batch_size=3, sequence_size=10)
        assert images.size() == (3, 10, 3, 224, 224)
        assert len(names) == 3

