import logging
logger = logging.getLogger(__name__)
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 16
from tracker import Tracker

class Trainer:
    def __init__(self, context):
        self.context = context
        self.tracker = Tracker(context=context)

    @torch.no_grad()
    def compute_last_layer_weight(self, Z, y_t):
        
        n = Z.shape[0]
        I_h = torch.eye(Z.shape[1]).to(self.context["device"])
        a_hat = torch.linalg.inv(Z.t() @ Z + self.context["reg_lamba"]*n*I_h) @ Z.t() @ y_t
        return a_hat

    @torch.no_grad()
    def eval(self, student, probe_loader, test_loader, epoch):
        loss_fn = torch.nn.MSELoss()
        epoch_loss = 0
        for batch_idx, (X, y_t) in enumerate(probe_loader):
            # ensure only one-batch of data for metrics on full-dataset
            assert batch_idx == 0
        _ = student(X)
        # capture hidden-layer features
        Z = student.affine_features[0]
        Z /= np.sqrt(self.context["d"])
        Z = student.activation_fn(Z)
        a_hat = self.compute_last_layer_weight(Z=Z, y_t=y_t)
        pred = Z @ a_hat
        epoch_loss = loss_fn(pred, y_t)
        self.tracker.store_training_loss(loss=epoch_loss, epoch=epoch)
        logger.info("Training (probe) loss: {}".format(epoch_loss))

        # use a_hat to compute loss
        epoch_loss = 0
        for batch_idx, (X, y_t) in enumerate(test_loader):
            X, y_t = X.to(self.context["device"]) , y_t.to(self.context["device"])
            student.zero_grad()
            pred = student(X)
            Z = student.affine_features[0]
            Z /= np.sqrt(self.context["d"])
            Z = student.activation_fn(Z)
            pred = Z @ a_hat
            loss = loss_fn(pred, y_t)
            epoch_loss += loss.detach().cpu().numpy() * X.shape[0]
        epoch_loss /= self.context["n_test"]
        self.tracker.store_val_loss(loss=epoch_loss, epoch=epoch)
        logger.info("Val loss: {}".format(epoch_loss))

    def get_optimizer(self, student):
        if self.context["optimizer"] == "sgd":
            optimizer = torch.optim.SGD(
                params=student.parameters(),
                lr=self.context["lr"],
                momentum=self.context["momentum"],
                weight_decay=self.context["weight_decay"],
            )
        elif self.context["optimizer"] == "adam":
            optimizer = torch.optim.Adam(
                params=student.parameters(),
                lr=self.context["lr"],
                weight_decay=self.context["weight_decay"],
            )
        return optimizer

    def run(self, student, train_loader, test_loader, probe_loader):
        optimizer = self.get_optimizer(student=student)
        self.tracker.probe_weights(student=student, epoch=0)
        self.tracker.probe_features(student=student, probe_loader=probe_loader, epoch=0)
        self.eval(student=student, probe_loader=probe_loader, test_loader=test_loader, epoch=0)
        loss_fn = torch.nn.MSELoss()
        for epoch in tqdm(range(1, self.context["num_epochs"]+1)):
            epoch_loss = 0
            for batch_idx, (X, y_t) in enumerate(train_loader):
                X, y_t = X.to(self.context["device"]) , y_t.to(self.context["device"])
                student.zero_grad()
                pred = student(X)
                loss = loss_fn(pred, y_t)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().cpu().numpy() * X.shape[0]
            epoch_loss /= self.context["n"]
            # we perform ridge regression on the probe loader
            # to calculate the train loss.
            # logger.info("Epoch: {} training loss: {}".format(epoch, epoch_loss))
            if epoch%self.context["probe_freq"] == 0:
                # self.tracker.store_training_loss(loss=epoch_loss, epoch=epoch)
                self.tracker.probe_weights(student=student, epoch=epoch)
                self.tracker.probe_features(student=student, probe_loader=probe_loader, epoch=epoch)
                self.eval(student=student, probe_loader=probe_loader, test_loader=test_loader, epoch=epoch)
        self.tracker.plot_training_loss()
        self.tracker.plot_val_loss()
        self.tracker.plot_initial_final_weight_vals()
        self.tracker.plot_initial_final_weight_esd()
        self.tracker.plot_initial_final_activation_vals()
        self.tracker.plot_initial_final_activation_esd()
        self.tracker.plot_epoch_KTA()
        return student
