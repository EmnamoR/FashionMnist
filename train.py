"""
Script to train Network ideas using Fashion Mnist dataset
"""
import os
import time
from collections import OrderedDict

import torch
import torch.nn as nn

from configs.config import get_config
from dataloader import DataLoader
from models import CNNModel5, mini_vgg
from utils.early_Stopping import EarlyStopping
from utils.hyperTune import RunBuilder
from utils.logger import Logger
from utils.tflogs import TfLogWriter
from utils.helpers import reset_model_weights

class Trainer:
    """
    Trainer class to train model using diffirent networks and parameters
    """

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.config = get_config()
        self.dataloader = DataLoader(self.config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = Logger().get_logger(logger_name='Advertima logs')

        if torch.cuda.is_available():
            self.logger.info('Using CUDA, benchmarking implementations')
            torch.backends.cudnn.benchmark = True
        else:
            self.logger.warning('Training using CPU may take longer time')
        self.criterion = nn.CrossEntropyLoss()
        self.params = OrderedDict(
            models=[CNNModel5(), mini_vgg()],
            lr=[.001],  # [.001, .01]
            batch_size=[64, 128],
            shuffle=[False]  # [false True]
        )
        self.tf_logs = TfLogWriter()

    def __init_training_params(self, run):
        model = run.models.to(self.device)
        dataset, train_sampler, valid_sampler = self.dataloader.get_loaders(run.shuffle)
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
                _, top_classes = prob.topk(1, dim=1)

                equals = labels == top_classes.view(labels.shape)
                accuracy += equals.type(torch.FloatTensor).mean()
                test_loss += self.criterion(log_ps, labels)
        return accuracy, test_loss

    def __train_model(self, model, train_loader, optimizer, train_loss):
        for images, labels in train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            optimizer.zero_grad()
            out = model(images)
            loss = self.criterion(out, labels)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        return train_loss

    def _train_epoch(self, model, optimizer, train_loader, validation_loader):
        train_loss = 0
        test_loss = 0
        accuracy = 0
        train_loss = self.__train_model(model, train_loader, optimizer, train_loss)
        accuracy, test_loss = self.__evaluate_model(model,
                                                    validation_loader, accuracy, test_loss)
        model.train()
        train_loss_e = train_loss / len(train_loader)
        test_loss_e = test_loss / len(validation_loader)
        accuracy_e = accuracy / len(validation_loader)

        return train_loss_e, test_loss_e, accuracy_e

    def run(self):
        """
        method to run trainer  with different networks and parameters
        save best model for each run (uses early stopping)
        """
        for run in RunBuilder.get_runs(self.params):
            self.tf_logs.begin_run(run)
            model, train_loader, validation_loader, optimizer = self.__init_training_params(run)
            train_losses, test_losses = [], []
            epoch_count = 0
            early_stopping = EarlyStopping(patience=10, verbose=self.verbose)
            epoch_start_time = time.time()
            for epoch in range(self.config.epochs):
                train_loss_e, test_loss_e \
                    , accuracy_e = self._train_epoch(model,
                                                     optimizer, train_loader, validation_loader)
                train_losses.append(train_loss_e)
                test_losses.append(test_loss_e)
                if self.verbose:
                    self.logger.info(
                        'Epoch: {}/{} ==> Training Loss: {:.3f} | '
                        'Test Loss: {:.3f} | Test Accuracy: {:.3f}'.format(
                            epoch + 1, self.config.epochs, train_loss_e, test_loss_e, accuracy_e))
                self.tf_logs.add_to_board(train_loss_e, test_loss_e, accuracy_e, epoch_count)
                # early_stopping needs the validation loss to check if it has decresed,
                # and if it has, it will make a checkpoint of the current model
                early_stopping(test_loss_e, model,
                               os.path.join(self.config.model_save_dir,
                                            str(run.models.__class__.__name__) + '_' + str(run.lr) + '_' + str(
                                                run.batch_size) + '_' + str(run.shuffle) + '_checkpoint.pt'),
                               self.logger)
                if early_stopping.early_stop:
                    break
                epoch_count += 1
            reset_model_weights(model)
            epoch_duration = time.time() - epoch_start_time
            self.logger.info('Networking training for {} '.format(epoch_duration))

if __name__ == '__main__':
    T = Trainer(True)
    T.run()
