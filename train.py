import os
import sys
from collections import OrderedDict

import torch
import torch.nn as nn

from configs.config import get_config
from dataLoader import dataLoader
from models import CNNModel
from utils.earlyStopping import EarlyStopping
from utils.hyperTune import RunBuilder
from utils.logger import logger
from utils.tflogs import logWriter


class trainer(object):
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.config = get_config()
        self.dataloader = dataLoader(self.config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print("Using CUDA, benchmarking implementations", file=sys.stderr)
            torch.backends.cudnn.benchmark = True
        self.criterion = nn.CrossEntropyLoss()
        self.params = OrderedDict(
            lr=[.001],
            batch_size=[64, 512],
            shuffle=[False]
        )
        self.logger = logger().get_logger(logger_name='Advertima logs')
        self.tf_logs = logWriter()

    def __init_training_params(self, run):
        model = CNNModel().to(self.device)
        dataset, train_sampler, valid_sampler = self.dataloader.getLoaders(run.shuffle)
        validation_loader = torch.utils.data.DataLoader(dataset,
                                                        batch_size=run.batch_size,
                                                        sampler=valid_sampler)
        train_loader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=run.batch_size,
                                                   sampler=train_sampler)

        # self.optimizer = optim.Adam(self.model.parameters(), lr=run.lr)
        optimizer = torch.optim.SGD(model.parameters(), lr=run.lr,
                                    momentum=0.9, nesterov=True)
        return model, train_loader, validation_loader, optimizer

    def __evaluate_model(self, model, validation_loader, accuracy, test_loss):
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
        return accuracy, test_loss

    def __train_model(self, model, train_loader, optimizer, train_loss):
        for images, labels in train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            optimizer.zero_grad()
            op = model(images)
            loss = self.criterion(op, labels)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        return train_loss

    def run(self):

        for run in RunBuilder.get_runs(self.params):
            self.tf_logs.begin_run(run)
            model, train_loader, validation_loader, optimizer = self.__init_training_params(run)
            train_losses, test_losses = [], []
            epoch_count = 0
            early_stopping = EarlyStopping(patience=10, verbose=self.verbose)
            for epoch in range(self.config.epochs):
                train_loss = 0
                test_loss = 0
                accuracy = 0

                train_loss = self.__train_model(model, train_loader, optimizer, train_loss)
                accuracy, test_loss = self.__evaluate_model(model, validation_loader, accuracy, test_loss)
                model.train()
                train_loss_e = train_loss / len(train_loader)
                test_loss_e = test_loss / len(validation_loader)
                accuracy_e = accuracy / len(validation_loader)
                if self.verbose:
                    self.logger.info(
                        'Epoch: {}/{} ==> Training Loss: {:.3f} | Test Loss: {:.3f} | Test Accuracy: {:.3f}'.format(
                            epoch + 1, self.config.epochs, train_loss_e, test_loss_e, accuracy_e))

                train_losses.append(train_loss_e)
                test_losses.append(test_loss_e)
                self.tf_logs.add_to_board(train_loss_e, test_loss_e, accuracy_e, epoch_count)
                # early_stopping needs the validation loss to check if it has decresed,
                # and if it has, it will make a checkpoint of the current model
                early_stopping(test_loss_e, model, os.path.join(self.config.model_save_dir,
                                                                str(run.lr) + '_' + str(run.batch_size) + '_' + str(
                                                                    run.shuffle) + '_checkpoint.pt'))

                epoch_count += 1
                if early_stopping.early_stop:
                    break


if __name__ == '__main__':
    t = trainer(True)
    t.run()
