# LRCN (Long-term Reccurent Convolutional Networks) approach for video regression (idea from https://arxiv.org/pdf/1411.4389.pdf).

To begin working with LRCN library, perform the following steps:

  1. `!git clone https://github.com/DJAlexJ/LRCN-for-Video-Regression.git`
  2. `cd LRCN-for-Video-Regression && pip install -e .`

Before training you have to create folder with movie trailers and prepare markup for them. Paths to the trailers, markup and model wieghts should be specified in config.py

### Training model
from lrcnreg/lrcnreg folder:
`python train.py` (`python train.py -h` to see additional arguments)


### Getting predictions#
from lrcnreg/lrcnreg/scripts:
`python predict.py --input_path='Path to the trailers' --output_path='File with predictions (e.g. ./res.txt)' --weights='Path to model weights'`
