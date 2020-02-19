"""
script to parse all app args
"""
import argparse
from pathlib import Path


class Config:
    """
    set configuration arguments as class attributes
    """
    def __init__(self, **kwargs):
        for k, val in kwargs.items():
            setattr(self, k, val)



def get_config(**kwargs):
    """
    get configuration arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train')

    # fMnist model args
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--validation_split', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--train_images_path', type=str, default='data/train-images-idx3-ubyte')
    parser.add_argument('--valid_images_path', type=str, default='data/t10k-images-idx3-ubyte')
    parser.add_argument('--train_labels_path', type=str, default='data/train-labels-idx1-ubyte')
    parser.add_argument('--valid_labels_path', type=str, default='data/t10k-labels-idx1-ubyte')

    # log
    parser.add_argument('-l', '--log_dir', type=str, default=Path('logs'))
    parser.add_argument('--detail_flag', type=bool, default=True)

    # model
    parser.add_argument('--model_save_dir', type=str, default=Path('checkpoints'))
    parser.add_argument('-m', '--model_path', type=str)

    args = parser.parse_args()
    # namespace -> dictionary
    args = vars(args)
    args.update(kwargs)

    return Config(**args)
