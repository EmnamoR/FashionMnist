import sys
from collections import OrderedDict
import torch
import torch.nn as nn

from configs.config import get_config
from dataLoader import dataLoader

from models import CNNModel


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

  def run(self):
    model = CNNModel().to(self.device)
    dataset, train_sampler, valid_sampler = self.dataloader.getLoaders(True)
    validation_loader = torch.utils.data.DataLoader(dataset,
                                                      batch_size=64,
                                                      sampler=valid_sampler)
    train_loader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=64,
                                                 sampler=train_sampler)

    # self.optimizer = optim.Adam(self.model.parameters(), lr=run.lr)
    self.optimizer = torch.optim.SGD(model.parameters(), lr=0.001,
                                  momentum=0.9, nesterov=True)



