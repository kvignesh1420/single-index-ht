import logging
logger = logging.getLogger(__name__)
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import math
plt.rcParams['axes.labelsize'] = 16
from tracker_minibatch import Tracker
import os

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
            X, y_t = X.to(self.context["device"]) , y_t.to(self.context["device"])
            # ensure only one-batch of data for metrics on full-dataset
            assert batch_idx == 0
        _ = student(X)
        # capture hidden-layer features
        Z = student.affine_features[0]
        Z /= np.sqrt(self.context["d"])
        Z = student.activation_fn(Z)
        a_hat = self.compute_last_layer_weight(Z=Z, y_t=y_t)
        pred = Z @ a_hat
        epoch_loss = loss_fn(pred, y_t).item()
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

    def run(self, teacher, student, train_loader, test_loader, probe_loader):
        optimizer = self.get_optimizer(student=student)
        self.tracker.probe_weights(teacher=teacher, student=student, epoch=0)
        self.tracker.probe_features(student=student, probe_loader=probe_loader, epoch=0)
        #self.tracker.probe_iteration_weights(teacher=teacher, student=student, iteration=0)
        #self.tracker.probe_iteration_features(student=student, probe_loader=probe_loader, iteration=0)

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
                self.tracker.probe_iteration_gradient_weights(teacher=teacher, student=student, iteration=batch_idx+(epoch-1)*(math.ceil(self.context["n"]/self.context["batch_size"])))

                optimizer.step()

                #iter=batch_idx+(epoch-1)*(math.ceil(self.context["n"]/self.context["batch_size"]))
                
                epoch_loss += loss.detach().cpu().numpy() * X.shape[0]
                #self.tracker.probe_iteration_weights(teacher=teacher, student=student, iteration=batch_idx+(epoch-1)*(math.ceil(self.context["n"]/self.context["batch_size"])))
                #self.tracker.probe_iteration_features(student=student, probe_loader=probe_loader, iteration=batch_idx+(epoch-1)*(math.ceil(self.context["n"]/self.context["batch_size"])))

            epoch_loss /= self.context["n"]
            # we perform ridge regression on the probe loader
            # to calculate the train loss.
            # logger.info("Epoch: {} training loss: {}".format(epoch, epoch_loss))
            if epoch%self.context["probe_freq"] == 0:
                # self.tracker.store_training_loss(loss=epoch_loss, epoch=epoch)
                self.tracker.probe_weights(teacher=teacher, student=student, epoch=epoch)
                self.tracker.probe_features(student=student, probe_loader=probe_loader, epoch=epoch)
                self.eval(student=student, probe_loader=probe_loader, test_loader=test_loader, epoch=epoch)
        self.tracker.plot_losses()
        self.tracker.plot_initial_final_weight_vals()
        self.tracker.plot_initial_final_weight_esd()
        self.tracker.plot_initial_final_activation_vals()
        self.tracker.plot_initial_final_activation_esd()
        self.tracker.plot_epoch_KTA()
        self.tracker.plot_epoch_W_beta_alignment()
        self.tracker.plot_epoch_weight_vals()
        self.tracker.plot_epoch_weight_esd()
        self.tracker.plot_epoch_activation_vals()
        self.tracker.plot_epoch_activation_esd()
        self.tracker.plot_initial_final_gradient_weight_vals()
        self.tracker.plot_initial_final_gradient_weight_esd()
        #self.tracker.plot_iteration_weight_vals()
        #self.tracker.plot_iteration_weight_esd()
        #self.tracker.plot_iteration_activation_vals()
        #self.tracker.plot_iteration_activation_esd()
        np.save(os.path.join(self.context['acc_loss_metric_dir'], f'training_loss.npy'), self.tracker.training_loss)
        np.save(os.path.join(self.context['acc_loss_metric_dir'], f'val_loss.npy'), self.tracker.val_loss)
        np.save(os.path.join(self.context['acc_loss_metric_dir'], f'epoch_weight_esd.npy'), self.tracker.epoch_weight_esd)
        np.save(os.path.join(self.context['acc_loss_metric_dir'], f'epoch_weight_vals.npy'), self.tracker.epoch_weight_vals)
        np.save(os.path.join(self.context['acc_loss_metric_dir'], f'epoch_activation_esd.npy'), self.tracker.epoch_activation_esd)
        np.save(os.path.join(self.context['acc_loss_metric_dir'], f'epoch_activation_vals.npy'), self.tracker.epoch_activation_vals)
        np.save(os.path.join(self.context['acc_loss_metric_dir'], f'epoch_KTA.npy'), self.tracker.epoch_KTA)
        np.save(os.path.join(self.context['acc_loss_metric_dir'], f'epoch_W_beta_alignment.npy'), self.tracker.epoch_W_beta_alignment)
        np.save(os.path.join(self.context['acc_loss_metric_dir'], f'epoch_spectral_norm.npy'), self.tracker.epoch_spectral_norm)
        np.save(os.path.join(self.context['acc_loss_metric_dir'], f'epoch_frobenuis_norm.npy'), self.tracker.epoch_frobenuis_norm)
        np.save(os.path.join(self.context['acc_loss_metric_dir'], f'epoch_stable_rank.npy'), self.tracker.epoch_stable_rank)
        np.save(os.path.join(self.context['acc_loss_metric_dir'], f'iteration_gradient_spectral_norm.npy'), self.tracker.iteration_gradient_spectral_norm)
        np.save(os.path.join(self.context['acc_loss_metric_dir'], f'iteration_gradient_frobenuis_norm.npy'), self.tracker.iteration_gradient_frobenuis_norm)
        np.save(os.path.join(self.context['acc_loss_metric_dir'], f'iteration_gradient_stable_rank.npy'), self.tracker.iteration_gradient_stable_rank)
        np.save(os.path.join(self.context['acc_loss_metric_dir'], f'iteration_gradient_weight_esd.npy'), self.tracker.iteration_gradient_weight_esd)
        np.save(os.path.join(self.context['acc_loss_metric_dir'], f'iteration_gradient_weight_vals.npy'), self.tracker.iteration_gradient_weight_vals)


        #np.save(os.path.join(self.context['acc_loss_metric_dir'], f'iteration_weight_esd.npy'), self.tracker.iteration_weight_esd)
        #np.save(os.path.join(self.context['acc_loss_metric_dir'], f'iteration_weight_vals.npy'), self.tracker.iteration_weight_vals)
        #np.save(os.path.join(self.context['acc_loss_metric_dir'], f'iteration_activation_esd.npy'), self.tracker.iteration_activation_esd)
        #np.save(os.path.join(self.context['acc_loss_metric_dir'], f'iteration_activation_vals.npy'), self.tracker.iteration_activation_vals)
        #np.save(os.path.join(self.context['acc_loss_metric_dir'], f'iteration_KTA.npy'), self.tracker.iteration_KTA)
        #np.save(os.path.join(self.context['acc_loss_metric_dir'], f'iteration_W_beta_alignment.npy'), self.tracker.iteration_W_beta_alignment)
        return student
