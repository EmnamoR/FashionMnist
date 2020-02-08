import sys
from collections import OrderedDict

import torch
import torch.nn as nn

from configs.config import get_config
from dataLoader import dataLoader


class trainer(object):
  def __init__(self):
    self.config = get_config()
    self.dataloader = dataLoader(self.config)
    self.params = OrderedDict(
      lr=[.001],
      batch_size=[64, 512],
      shuffle=[False]
    )
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
      print("Using CUDA, benchmarking implementations", file=sys.stderr)
      torch.backends.cudnn.benchmark = True
    self.criterion = nn.CrossEntropyLoss()