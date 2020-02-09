import argparse
from pathlib import Path

"""
set configuration arguments as class attributes
"""


class Config(object):
  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      setattr(self, k, v)


"""
get configuration arguments
"""


def get_config(**kwargs):
  parser = argparse.ArgumentParser()

  parser.add_argument('--mode', type=str, default='train')

  # fMnist model args
  parser.add_argument('--random_seed', type=int, default=42)
  parser.add_argument('--validation_split', type=float, default=0.2)
  parser.add_argument('--epochs', type=int, default=100)

  #   # features
  parser.add_argument('--images_path', type=str,
                      default='data/train-images-idx3-ubyte')
  parser.add_argument('--labels_path', type=str,
                      default='data/train-labels-idx1-ubyte')

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
