import pytest
import shutil
import os
import lrcnreg.scripts.preprocessing as prep
from lrcnreg.config import TEST_TRAILER, TRAINING_PATH, PREDICTION_PATH


def test_format_ok():
    assert prep.format_ok('mp4') == True
    assert prep.format_ok('wrn') == False

@pytest.mark.parametrize('path, training', 
                        [(TRAINING_PATH, True), 
                         (PREDICTION_PATH, False),
                        ])
def test_preprocess(path, training):
    prep.preprocess(TEST_TRAILER, train=training, n_subclips=2)
    name = os.listdir(path)[0]
    digit = name[-1]
    assert len(os.listdir(path)) == 2
    assert 13 <= len(os.listdir(f'{path}/{name}/{digit}')) <= 18
    shutil.rmtree(path)

    
def test_movies_preprocess():
    prep.movies_preprocess(['85.Дикая.mp4', '86.Малыш на драйве.mp4'], train=True, n_subclips=1)
    assert len(os.listdir(TRAINING_PATH)) == 2
    shutil.rmtree(TRAINING_PATH)
    
    prep.movies_preprocess(os.listdir('./'), train=True, n_subclips=1)
    assert len(os.listdir(TRAINING_PATH)) == 3
    shutil.rmtree(TRAINING_PATH)

