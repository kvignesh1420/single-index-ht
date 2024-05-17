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

        for epoch in tqdm(range(1, self.context["num_epochs"]+1)):
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
                loss.backward()
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
        self.tracker.plot_losses()
        if self.context["probe_weights"]:
            # self.tracker.plot_step_weight_stable_rank()
            # self.tracker.plot_initial_final_weight_vals()
            # self.tracker.plot_initial_final_weight_esd()
            # self.tracker.plot_initial_final_weight_nolog_esd()
            # self.tracker.plot_all_steps_W_M_alignment()
            self.tracker.plot_step_W_beta_alignment()
        if self.context["probe_features"]:
            # self.tracker.plot_step_activation_stable_rank()
            # self.tracker.plot_step_activation_effective_ranks()
            # self.tracker.plot_initial_final_activation_vals()
            # self.tracker.plot_initial_final_activation_esd()
            self.tracker.plot_step_KTA()


class OneStepBulkTrainer:
    def __init__(self, contexts, varying_params):
        self.contexts = contexts
        self.varying_params = varying_params
        # support only optimizer, lr as of now
        assert self.varying_params == ["optimizer", "lr"]
        self.trainers = []
        for context in contexts:
            trainer = Trainer(context=context)
            self.trainers.append(trainer)

    def run(self, teacher, students, train_loader, test_loader, probe_loader):
        for trainer, student in zip(self.trainers, students):
            trainer.run(
                teacher=teacher,
                student=student,
                train_loader=train_loader,
                test_loader=test_loader,
                probe_loader=probe_loader,
            )
        self.plot_results()
        return students

    def plot_results(self):
        self.plot_losses()
        self.plot_step_KTA()
        self.plot_step_W_beta_alignment()
        self.plot_step_weight_esd_pl_alpha()

    @torch.no_grad()
    def plot_losses(self):
        steps = list(self.trainers[0].tracker.val_loss.keys())
        final_step = steps[-1]
        # temporary assertion for one-step experiments.
        assert len(steps) == 2

        final_training_losses = defaultdict(list)
        final_val_losses = defaultdict(list)
        lrs = defaultdict(list)

        for trainer, context in zip(self.trainers, self.contexts):
            lrs[context["optimizer"]].append(context["lr"])
            training_loss = trainer.tracker.training_loss[final_step]
            final_training_losses[context["optimizer"]].append(training_loss)
            val_loss = trainer.tracker.val_loss[final_step]
            final_val_losses[context["optimizer"]].append(val_loss)

        assert lrs["sgd"] == lrs["adam"]
        lrs = np.log10(lrs["sgd"])
        plt.plot(lrs, final_training_losses["sgd"], marker='o', label="GD:train")
        plt.plot(lrs, final_training_losses["adam"], marker='o', label="FB-Adam:train")
        plt.plot(lrs, final_val_losses["sgd"], marker='x', label="GD:test", linestyle='dashed')
        plt.plot(lrs, final_val_losses["adam"], marker='x', label="FB-Adam:test", linestyle='dashed')

        plt.xlabel("$\log_{10}(\eta)$")
        plt.ylabel("loss")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("{}bulk_losses.jpg".format(context["bulk_vis_dir"]))
        plt.clf()

    @torch.no_grad()
    def plot_step_KTA(self):
        steps = list(self.trainers[0].tracker.step_KTA.keys())
        final_step = steps[-1]
        # temporary assertion for one-step experiments.
        assert len(steps) == 2

        layer_idx = 0
        KTAs = defaultdict(list)
        lrs = defaultdict(list)
        for trainer, context in zip(self.trainers, self.contexts):
            lrs[context["optimizer"]].append(context["lr"])
            KTA = trainer.tracker.step_KTA[final_step][layer_idx]
            KTAs[context["optimizer"]].append(KTA)

        assert lrs["sgd"] == lrs["adam"]
        lrs = np.log10(lrs["sgd"])

        plt.plot(lrs, KTAs["sgd"], marker='o', label="GD")
        plt.plot(lrs, KTAs["adam"], marker='o', label="FB-Adam")

        plt.xlabel("$\log_{10}(\eta)$")
        plt.ylabel("KTA")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        name="{}bulk_KTA.jpg".format(context["bulk_vis_dir"])
        plt.savefig(name)
        plt.clf()

    @torch.no_grad()
    def plot_step_W_beta_alignment(self):
        steps = list(self.trainers[0].tracker.step_W_beta_alignment.keys())
        final_step = steps[-1]
        # temporary assertion for one-step experiments.
        assert len(steps) == 2
        layer_idx = 0
        lrs = defaultdict(list)
        sims = defaultdict(list)
        for trainer, context in zip(self.trainers, self.contexts):
            lrs[context["optimizer"]].append(context["lr"])
            sim = trainer.tracker.step_W_beta_alignment[final_step][layer_idx]
            sims[context["optimizer"]].append(sim)

        assert lrs["sgd"] == lrs["adam"]
        lrs = np.log10(lrs["sgd"])

        plt.plot(lrs, sims["sgd"], marker='o', label="GD")
        plt.plot(lrs, sims["adam"], marker='o', label="FB-Adam")

        plt.xlabel("$\log_{10}(\eta)$")
        plt.ylabel("$sim(W, \\beta^*)$")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        name="{}bulk_W_beta_alignment.jpg".format(context["bulk_vis_dir"])
        plt.savefig(name)
        plt.clf()
    
    @torch.no_grad()
    def plot_step_weight_esd_pl_alpha(self):
        steps = list(self.trainers[0].tracker.step_weight_esd_pl_alpha.keys())
        final_step = steps[-1]
        # temporary assertion for one-step experiments.
        assert len(steps) == 2
        layer_idx = 0
        lrs = defaultdict(list)
        pl_alphas = defaultdict(list)
        for trainer, context in zip(self.trainers, self.contexts):
            lrs[context["optimizer"]].append(context["lr"])
            pl_alpha = trainer.tracker.step_weight_esd_pl_alpha[final_step][layer_idx]
            pl_alphas[context["optimizer"]].append(pl_alpha)

        assert lrs["sgd"] == lrs["adam"]
        lrs = np.log10(lrs["sgd"])

        plt.plot(lrs, pl_alphas["sgd"], marker='o', label="GD")
        plt.plot(lrs, pl_alphas["adam"], marker='o', label="FB-Adam")

        plt.xlabel("$\log_{10}(\eta)$")
        plt.ylabel("PL_Alpha_Hill")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        name="{}bulk_W_esd_pl_alpha.jpg".format(context["bulk_vis_dir"])
        plt.savefig(name)
        plt.clf()


class MultiStepLossBulkTrainer:
    def __init__(self, contexts, varying_params):
        self.contexts = contexts
        self.varying_params = varying_params
        assert len(self.varying_params) == 1
        self.trainers = []
        for context in contexts:
            trainer = Trainer(context=context)
            self.trainers.append(trainer)

    def run(self, teachers, students, train_loaders, test_loaders, probe_loaders):
        for trainer, teacher, student, train_loader, test_loader, probe_loader in zip(self.trainers, teachers, students, train_loaders, test_loaders, probe_loaders):
            trainer.run(
                teacher=teacher,
                student=student,
                train_loader=train_loader,
                test_loader=test_loader,
                probe_loader=probe_loader,
            )
        self.plot_results()
        return students

    def plot_results(self):
        self.plot_losses()

    @torch.no_grad()
    def plot_losses(self):
        varying_param = self.varying_params[0]
        for trainer, context in zip(self.trainers, self.contexts):
            steps = list(trainer.tracker.val_loss.keys())
            training_losses = list(trainer.tracker.training_loss.values())
            val_losses = list(trainer.tracker.val_loss.values())
            if varying_param == "reg_lambda":
                label_name = "$\lambda$"
            elif varying_param == "gamma":
                label_name = "$\gamma$"
            elif varying_param == "label_noise_std":
                label_name = "$\\rho_{e}$"
            else:
                label_name = varying_param
            if varying_param == "gamma":
                label_value = context["lr_scheduler_kwargs"]["gamma"]
            else:
                label_value = context[varying_param]

            plt.plot(steps, training_losses, marker='o', label="{}={}".format(label_name, label_value ))
            plt.plot(steps, val_losses, marker='x', label="{}={}".format(label_name, label_value), linestyle='dashed')

        plt.xlabel("step")
        plt.ylabel("loss")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("{}bulk_losses.jpg".format(context["bulk_vis_dir"]))
        plt.clf()
