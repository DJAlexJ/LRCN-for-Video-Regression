import argparse
import preprocessing as prep
import torch
from loading_data import load_data
from model import CNN, LRCN
from config import PREDICTION_PATH, TEST_TRAILER_NAME, PATH_MODEL_CHECKPOINT


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-i", "--input_path", type=Path, help="Path with movie trailers", required=True)
    arg("-out", "--output_path", type=Path, help="Path to output file with movie scores", required=True)
    arg("-w", "--weights", type=Path, help="Path to weights", required=True)
    arg("-v", "--verbose", default=False, help="Verbosity")
    arg("-d", "--device", default='cpu', type=str, help="Device for computations")
    arg("-vis", "--visualize", action="store_true", help="Visualize results")
    
    return parser.parse_args()

def predict():
    args = get_args()
    
    hparams = {
            "input_path": args.input_path,
            "output_path": args.output_path,
            "weights": args.weights,
            "verbose": args.verbose,
            "device": args.device,
            "visualize": args.visualize,
        }
    
    model = LRCN()
    model.load_state_dict(torch.load(hparams['weights'], map_location=torch.device(hparams['device'])))
    model.eval()
    
    prep.movies_preporcess(os.listdir(hparams['input_path']), train=False, n_subclips=1, verbose=hparams['verbose'])
    dir_names = os.listdir(PREDICTION_PATH)
    dir_names = list(filter(lambda x: os.path.isdir(f"{path}/{x}"), dir_names))
    
    predictions = model.predict(dir_names, verbose=hparams['verbose'], saving_results=True, 
                                output_path=hparams['output_path'])
    
    return predictions

if __name__ == '__main__':
    predict()






