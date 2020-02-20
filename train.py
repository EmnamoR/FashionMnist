"""
Script to train Network ideas using Fashion Mnist dataset
"""
import os
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler

from configs.config import get_config
from dataloader import DataLoader
from models import CNNModel5, mini_vgg
from models import EmbeddingNet, TripletNet
from triplet.losses import TripletLoss
from utils.early_Stopping import EarlyStopping
from utils.helpers import reset_model_weights, check_triplet
from utils.hyperTune import RunBuilder
from utils.logger import Logger
from utils.tflogs import TfLogWriter


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
        self.margin = 1.
        self.embedding_net = EmbeddingNet()

        if torch.cuda.is_available():
            self.logger.info('Using CUDA, benchmarking implementations')
            torch.backends.cudnn.benchmark = True
        else:
            self.logger.warning('Training using CPU may take longer time')
        self.params = OrderedDict(
            models=[TripletNet(self.embedding_net), mini_vgg(), CNNModel5()],
            lr=[.001],  # [.001, .01]
            batch_size=[64, 128]
        )
        self.tf_logs = TfLogWriter()

    def __init_training_params(self, run, model, triplet=False):
        train_dataset, valid_dataset = self.dataloader.get_loaders(triplet)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=run.batch_size)
        validation_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=run.batch_size)
        optimizer = torch.optim.SGD(model.parameters(), lr=run.lr,
                                    momentum=0.9, nesterov=True)
        return train_loader, validation_loader, optimizer

    def fit(self, run, train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, device, log_interval,
            metrics=[], triplet=False, start_epoch=0):
        epoch_start_time = time.time()
        early_stopping = EarlyStopping(patience=10, verbose=self.verbose)
        for epoch in range(0, start_epoch):
            scheduler.step()
        for epoch in range(start_epoch, n_epochs):
            scheduler.step()

            # Train stage
            train_loss, metrics = self.train_epoch(train_loader, model, loss_fn, optimizer, device, log_interval,
                                                   metrics,
                                                   triplet)

            message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            val_loss, metrics = self.test_epoch(val_loader, model, loss_fn, device, metrics, triplet)
            val_loss /= len(val_loader)

            message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, val_loss)
            self.logger.info(message)
            self.tf_logs.add_to_board(train_loss, val_loss, epoch)
            early_stopping(val_loss, model,
                           os.path.join(self.config.model_save_dir,
                                        str(run.models.__class__.__name__) + '_' + str(run.lr) + '_' + str(
                                            run.batch_size) + '_' + '_checkpoint.pt'),
                           self.logger)
            if early_stopping.early_stop:
                break
        model.apply(reset_model_weights)

        epoch_duration = time.time() - epoch_start_time
        self.logger.info('Networking training for {} '.format(epoch_duration))

    def train_epoch(self, train_loader, model, loss_fn, optimizer, device, log_interval, metrics, triplet):
        for metric in metrics:
            metric.reset()
        model.train()
        losses = []
        total_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            target = target if len(target) > 0 else None
            optimizer.zero_grad()
            if target is not None:
                target = target.to(device)
            if triplet:
                if not type(data) in (tuple, list):
                    data = (data,)
                data = tuple(d.to(device) for d in data)
                outputs = model(*data)
                if type(outputs) not in (tuple, list):
                    outputs = (outputs,)
                loss_inputs = outputs
                if target is not None:
                    target = (target,)
                    loss_inputs += target

                loss_outputs = loss_fn(*loss_inputs)
                loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            else:
                data = data.to(device)
                outputs = model(data)
                loss = loss_fn(outputs, target)
            losses.append(loss.item())
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            for metric in metrics:
                metric(outputs, target, loss_outputs)

            if batch_idx % log_interval == 0:
                if triplet:
                    message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        batch_idx * len(data[0]), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), np.mean(losses))
                else:
                    message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        batch_idx * data.size(0), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), np.mean(losses))

                self.logger.info(message)
                losses = []

        total_loss /= (batch_idx + 1)

        return total_loss, metrics

    def test_epoch(self, val_loader, model, loss_fn, device, metrics, triplet):
        with torch.no_grad():
            for metric in metrics:
                metric.reset()
            model.eval()
            val_loss = 0
            for batch_idx, (data, target) in enumerate(val_loader):
                target = target if len(target) > 0 else None
                if target is not None:
                    target = target.to(device)
                if triplet:
                    if not type(data) in (tuple, list):
                        data = (data,)
                    data = tuple(d.to(device) for d in data)
                    outputs = model(*data)
                    if type(outputs) not in (tuple, list):
                        outputs = (outputs,)
                    loss_inputs = outputs
                    if target is not None:
                        target = (target,)
                        loss_inputs += target
                    loss_outputs = loss_fn(*loss_inputs)
                    loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
                else:
                    data = data.to(device)
                    outputs = model(data)
                    loss = loss_fn(outputs, target)
                val_loss += loss.item()
        return val_loss, metrics

    def run(self):
        for run in RunBuilder.get_runs(self.params):
            self.tf_logs.begin_run(run)
            model = run.models.to(self.device)
            if check_triplet(model):
                triplet = True
                loss_fn = TripletLoss(self.margin)
            else:
                triplet = False
                loss_fn = nn.CrossEntropyLoss()
            train_loader, validation_loader, optimizer = self.__init_training_params(run, model, triplet=triplet)
            scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
            log_interval = 450
            self.fit(run, train_loader, validation_loader, model, loss_fn, optimizer, scheduler, self.config.epochs,
                     self.device,
                     log_interval, triplet=triplet)


if __name__ == '__main__':
    T = Trainer(True)
    T.run()
