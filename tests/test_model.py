import pytest
import torch
import os
import shutil
from config import TRAINING_PATH, PREDICTION_PATH, MODEL_WEIGHTS
import model as m
import preprocessing as prep
import loading_data as ld


@pytest.fixture(scope='module')
def train_prep():
    yield prep.preprocess('86.Малыш на драйве.mp4', train=True, verbose=True)
    shutil.rmtree(TRAINING_PATH)
    
@pytest.fixture(scope='function')
def test_prep():
    yield prep.preprocess('85.Дикая.mp4', train=False, verbose=True)
    shutil.rmtree(PREDICTION_PATH)
    

def test_forward_model():
    cnn = m.CNN(embedding_size=6)
    lrcn = m.LRCN(embedding_size=4)
    x = torch.rand(3, 3, 224, 224)
    assert cnn.forward(x).size() == (3, 6)
    x = torch.rand(2, 17, 3, 224, 224)
    assert lrcn.forward(x).size() == (2, 1)
    
def test_train_predict_single(train_prep, test_prep):
    lrcn = m.LRCN()
    dir_names = os.listdir(TRAINING_PATH)
    dir_names = list(filter(lambda x: os.path.isdir(f"{TRAINING_PATH}/{x}"), dir_names)) #Filtering waste files
    X_test, y_test, dir_names = ld.load_data(dir_names, train=True, verbose=True, batch_size=1)
    logs = lrcn.fit(dir_names, X_test, y_test, n_epoch=2, verbose=True, saving_results=True)
    assert len(logs[0]) == 2
    assert len(logs[1]) == 2
    
    y = lrcn.predict_single('85.Дикая.mp4', verbose=True)
    assert isinstance(y, float)
    
def test_predict():
    lrcn = m.LRCN()
    lrcn.load_state_dict(torch.load(MODEL_WEIGHTS))
    y = lrcn.predict([], preprocess=True, saving_results=True, input_path='./', output_path='./results.txt')
    
    assert os.path.exists('./results.txt')
    assert y.size() == (3, 1)
    os.remove(MODEL_WEIGHTS)
    shutil.rmtree(PREDICTION_PATH)

