import argparse
import torch
import os
from lrcnreg.scripts import preprocessing as prep
from lrcnreg.scripts.loading_data import load_data
from lrcnreg.scripts.model import CNN, LRCN
from lrcnreg.config import PREDICTION_PATH


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-i", "--input_path", type=str, help="Path with movie trailers", required=True)
    arg("-out", "--output_path", type=str, help="output file (with path) with movie scores", required=True)
    arg("-w", "--weights", type=str, help="Path to weights", required=True)
    arg("-v", "--verbose", default=False, help="Verbosity")
    arg("-d", "--device", default='cpu', type=str, help="Device for computations")
    arg("-vis", "--visualize", action="store_true", help="Visualize results")
    
    return parser.parse_args()

def predict():
    args = get_args()
    
    model = LRCN()
    model.load_state_dict(torch.load(args.weights, map_location=torch.device(args.device)))
    model.eval()
    
    prep.movies_preprocess(os.listdir(args.input_path), train=False, n_subclips=1, verbose=args.verbose)
    dir_names = os.listdir(PREDICTION_PATH)
    dir_names = list(filter(lambda x: os.path.isdir(f"{PREDICTION_PATH}/{x}"), dir_names))
    
    predictions = model.predict(dir_names, verbose=args.verbose, saving_results=True,
                                output_path=args.output_path)
    
    return predictions

if __name__ == '__main__':
    predict()






