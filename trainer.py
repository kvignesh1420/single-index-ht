from collections import defaultdict
import logging
logger = logging.getLogger(__name__)
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'axes.labelsize': 20,
    'legend.fontsize': 14
})
from tracker import Tracker



LR_SCHEDULER_FACTORY = {
    "StepLR": torch.optim.lr_scheduler.StepLR
}


class Trainer:
    def __init__(self, context):
        self.context = context
        self.tracker = Tracker(context=context)

    @torch.no_grad()
    def eval(self, student, probe_loader, test_loader, step):
        if self.context["fix_last_layer"]:
            self.eval_with_regression(student=student, probe_loader=probe_loader, test_loader=test_loader, step=step)
        else:
            self.eval_without_regression(student=student, test_loader=test_loader, step=step)

    @torch.no_grad()
    def compute_last_layer_weight(self, Z, y_t):
        n = Z.shape[0]
        h = Z.shape[1]
        I_h = torch.eye(Z.shape[1]).to(self.context["device"])
        a_hat = torch.linalg.inv(Z.t() @ Z + (self.context["reg_lambda"]/h)*n*I_h) @ Z.t() @ y_t
        return a_hat

    @torch.no_grad()
    def eval_with_regression(self, student, probe_loader, test_loader, step):
        loss_fn = torch.nn.MSELoss()
        step_loss = 0
        for batch_idx, (X, y_t) in enumerate(probe_loader):
            # ensure only one-batch of data for metrics on full-dataset
            assert batch_idx == 0
        _ = student(X)
        # capture hidden-layer features
        Z = student.affine_features[0]
        Z /= np.sqrt(self.context["d"])
        Z = student.activation_fn(Z)
        Z /= np.sqrt(self.context["h"])
        a_hat = self.compute_last_layer_weight(Z=Z, y_t=y_t)
        pred = Z @ a_hat
        step_loss = loss_fn(pred, y_t)
        self.tracker.store_training_loss(loss=step_loss, step=step)
        logger.info("Training (probe) loss: {}".format(step_loss))

        # use a_hat to compute loss
        step_loss = 0
        for batch_idx, (X, y_t) in enumerate(test_loader):
            X, y_t = X.to(self.context["device"]) , y_t.to(self.context["device"])
            student.zero_grad()
            pred = student(X)
            Z = student.affine_features[0]
            Z /= np.sqrt(self.context["d"])
            Z = student.activation_fn(Z)
            Z /= np.sqrt(self.context["h"])
            pred = Z @ a_hat
            loss = loss_fn(pred, y_t)
            step_loss += loss.detach().cpu().numpy() * X.shape[0]
        step_loss /= self.context["n_test"]
        self.tracker.store_val_loss(loss=step_loss, step=step)
        logger.info("Val loss: {}".format(step_loss))

    @torch.no_grad()
    def eval_without_regression(self, student, test_loader, step):
        loss_fn = torch.nn.MSELoss()
        step_loss = 0
        for (X, y_t) in test_loader:
            X, y_t = X.to(self.context["device"]) , y_t.to(self.context["device"])
            student.zero_grad()
            pred = student(X)
            loss = loss_fn(pred, y_t)
            step_loss += loss.detach().cpu().numpy() * X.shape[0]
        step_loss /= self.context["n_test"]
        self.tracker.store_val_loss(loss=step_loss, step=step)
        logger.info("Val loss: {}".format(step_loss))


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
        self.tracker.plot_init_W_pc_and_beta_alignment(student=student, teacher=teacher)
        if self.context["probe_weights"]:
            self.tracker.probe_weights(teacher=teacher, student=student, step=0)
        if self.context["probe_features"]:
            self.tracker.probe_features(student=student, probe_loader=probe_loader, step=0)
        if self.context["fix_last_layer"]:
            self.eval(student=student, probe_loader=probe_loader, test_loader=test_loader, step=0)
        loss_fn = torch.nn.MSELoss()
        num_batches = int(np.ceil(self.context["n"]/self.context["batch_size"]))

        # handle lr scheduling
        if "lr_scheduler_cls" in self.context:
            lr_scheduler_cls = LR_SCHEDULER_FACTORY[self.context["lr_scheduler_cls"]]
            lr_scheduler = lr_scheduler_cls(optimizer=optimizer, **self.context["lr_scheduler_kwargs"])

        for epoch in range(1, self.context["num_epochs"]+1):
            epoch_loss = 0
            for batch_idx, (X, y_t) in enumerate(train_loader):
                step = (epoch-1)*num_batches + batch_idx + 1
                X, y_t = X.to(self.context["device"]) , y_t.to(self.context["device"])
                if self.context["enable_weight_normalization"]:
                    student.hidden_layer.weight.data /= torch.norm(student.hidden_layer.weight.data, p='fro')
                    student.hidden_layer.weight.data *= torch.sqrt(torch.tensor(self.context["h"]*self.context["d"]))
                student.zero_grad()
                pred = student(X)
                loss = loss_fn(pred, y_t)
                loss.backward(retain_graph=True)
                optimizer.step()
                epoch_loss += loss.detach().cpu().numpy() * X.shape[0]
                if step%self.context["probe_freq_steps"] == 0:
                    if self.context["probe_weights"]:
                        self.tracker.probe_weights(teacher=teacher, student=student, step=step)
                    if self.context["probe_features"]:
                        self.tracker.probe_features(student=student, probe_loader=probe_loader, step=step)
                    # The train loss is computed by performing ridge regression on the
                    # probe loader. Use the below loss if the last layer is not fixed.
                    if not self.context["fix_last_layer"]:
                        self.tracker.store_training_loss(loss=loss, step=step)
                    self.eval(student=student, probe_loader=probe_loader, test_loader=test_loader, step=step)
                # handle lr scheduling
                if "lr_scheduler_cls" in self.context:
                    lr_scheduler.step()
            epoch_loss /= self.context["n"]
            # logger.info("Epoch: {} training loss: {}".format(epoch, epoch_loss))
        self.plot_results()
        return student

    def plot_results(self):
        # self.tracker.plot_losses()
        if self.context["probe_weights"]:
            pass
            # self.tracker.plot_step_weight_stable_rank()
            # self.tracker.plot_initial_final_weight_vals()
            # self.tracker.plot_initial_final_weight_esd()
            # self.tracker.plot_initial_final_weight_nolog_esd()
            # self.tracker.plot_all_steps_W_M_alignment()
            # self.tracker.plot_step_W_beta_alignment()
        if self.context["probe_features"]:
            pass
            # self.tracker.plot_step_activation_stable_rank()
            # self.tracker.plot_step_activation_effective_ranks()
            # self.tracker.plot_initial_final_activation_vals()
            # self.tracker.plot_initial_final_activation_esd()
            # self.tracker.plot_step_KTA()

