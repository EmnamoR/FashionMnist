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
    # scheduler = StepLR(self.optimizer, step_size=2, gamma=0.1)

    train_losses, test_losses = [], []

    for epoch in range(self.config.epochs):
        train_loss = 0
        test_loss = 0
        accuracy = 0
        for images, labels in train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            op = model(images)
            loss = self.criterion(op, labels)
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
        else:
            with torch.no_grad():
                model.eval()
                for images, labels in validation_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    log_ps = model(images)
                    prob = torch.exp(log_ps)
                    top_probs, top_classes = prob.topk(1, dim=1)
                    equals = labels == top_classes.view(labels.shape)
                    accuracy += equals.type(torch.FloatTensor).mean()
                    test_loss += self.criterion(log_ps, labels)
            model.train()
        train_loss_e = train_loss / len(train_loader)
        test_loss_e = test_loss / len(validation_loader)
        accuracy_e = accuracy / len(validation_loader)
        print("Epoch: {}/{}.. ".format(epoch + 1, self.config.epochs),
              "Training Loss: {:.3f}.. ".format(train_loss_e),
              "Test Loss: {:.3f}.. ".format(test_loss_e),
              "Test Accuracy: {:.3f}".format(accuracy_e))
        train_losses.append(train_loss_e)
        test_losses.append(test_loss_e)


if __name__ == '__main__':
  t = trainer()
  t.run()




