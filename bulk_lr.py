import os
import sys
from collections import defaultdict
from copy import deepcopy
import logging
logger = logging.getLogger(__name__)
from trainer import Trainer
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'axes.labelsize': 20,
    'legend.fontsize': 14
})

from data import prepare_dataloaders
from models import get_student_model
from models import get_teacher_model
from utils import setup_runtime_context


def setup_logging(context):
    logging.basicConfig(
        filename=context["results_file"],
        filemode='a',
        format='%(asctime)s, %(name)s %(levelname)s %(message)s',
        level=logging.INFO
    )
    logger.addHandler(logging.StreamHandler(sys.stdout))


class BulkPlotter:
    def __init__(self, trainers, contexts):
        self.trainers = trainers
        self.contexts = contexts
        self.varying_params = ["optimizer", "lr"]

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

        final_training_losses = defaultdict(lambda : defaultdict(list))
        final_val_losses = defaultdict(lambda: defaultdict(list))
        lrs = defaultdict(lambda: defaultdict(int))

        for trainer, context in zip(self.trainers, self.contexts):
            lr = context["lr"]
            lrs[context["optimizer"]][lr] += 1
            training_loss = trainer.tracker.training_loss[final_step]
            final_training_losses[context["optimizer"]][lr].append(training_loss)
            val_loss = trainer.tracker.val_loss[final_step]
            final_val_losses[context["optimizer"]][lr].append(val_loss)

        assert lrs["sgd"] == lrs["adam"]
        np_lrs = np.sort(list(lrs["sgd"].keys()))

        plot_metadata = [
            # train
            (final_training_losses, "sgd", "o", "GD:train", "solid", "tab:blue"),
            (final_training_losses, "adam", "o", "FB-Adam:train", "solid", "tab:orange"),
            # test
            (final_val_losses, "sgd", "x", "GD:test", "dashed", "tab:green"),
            (final_val_losses, "adam", "x", "FB-Adam:test", "dashed", "tab:red")
        ]

        for loss_list, optimizer_val, marker, label, linestyle, color in plot_metadata:
            means = np.array([ np.mean(loss_list[optimizer_val][np_lr]) for np_lr in np_lrs])
            stds = np.array([ np.std(loss_list[optimizer_val][np_lr]) for np_lr in np_lrs])
            plt.plot(np.log10(np_lrs), means, marker=marker, label=label, linestyle=linestyle, color=color)
            plt.fill_between(np.log10(np_lrs), means - stds, means + stds, alpha=0.2)

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
        KTAs = defaultdict(lambda: defaultdict(list))
        lrs = defaultdict(lambda: defaultdict(int))
        for trainer, context in zip(self.trainers, self.contexts):
            lr = context["lr"]
            lrs[context["optimizer"]][lr] += 1
            KTA = trainer.tracker.step_KTA[final_step][layer_idx]
            KTAs[context["optimizer"]][lr].append(KTA)

        assert lrs["sgd"] == lrs["adam"]
        np_lrs = np.sort(list(lrs["sgd"].keys()))

        plot_metadata = [
            # train
            (KTAs, "sgd", "o", "GD", "solid", "tab:blue"),
            (KTAs, "adam", "o", "FB-Adam", "solid", "tab:orange")
        ]

        for KTA_list, optimizer_val, marker, label, linestyle, color in plot_metadata:
            means = np.array([ np.mean(KTA_list[optimizer_val][np_lr]) for np_lr in np_lrs])
            stds = np.array([ np.std(KTA_list[optimizer_val][np_lr]) for np_lr in np_lrs])
            plt.plot(np.log10(np_lrs), means, marker=marker, label=label, linestyle=linestyle, color=color)
            plt.fill_between(np.log10(np_lrs), means - stds, means + stds, alpha=0.2)

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

        sims = defaultdict(lambda: defaultdict(list))
        lrs = defaultdict(lambda: defaultdict(int))
        for trainer, context in zip(self.trainers, self.contexts):
            lr = context["lr"]
            lrs[context["optimizer"]][lr] += 1
            sim = trainer.tracker.step_W_beta_alignment[final_step][layer_idx]
            sims[context["optimizer"]][lr].append(sim)

        assert lrs["sgd"] == lrs["adam"]
        np_lrs = np.sort(list(lrs["sgd"].keys()))

        plot_metadata = [
            # train
            (sims, "sgd", "o", "GD", "solid", "tab:blue"),
            (sims, "adam", "o", "FB-Adam", "solid", "tab:orange")
        ]

        for sim_list, optimizer_val, marker, label, linestyle, color in plot_metadata:
            means = np.array([ np.mean(sim_list[optimizer_val][np_lr]) for np_lr in np_lrs])
            stds = np.array([ np.std(sim_list[optimizer_val][np_lr]) for np_lr in np_lrs])
            plt.plot(np.log10(np_lrs), means, marker=marker, label=label, linestyle=linestyle, color=color)
            plt.fill_between(np.log10(np_lrs), means - stds, means + stds, alpha=0.2)


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

        pl_alphas = defaultdict(lambda: defaultdict(list))
        lrs = defaultdict(lambda: defaultdict(int))

        for trainer, context in zip(self.trainers, self.contexts):
            lr = context["lr"]
            lrs[context["optimizer"]][lr] += 1
            pl_alpha = trainer.tracker.step_weight_esd_pl_alpha[final_step][layer_idx]
            pl_alphas[context["optimizer"]][lr].append(pl_alpha)

        assert lrs["sgd"] == lrs["adam"]
        np_lrs = np.sort(list(lrs["sgd"].keys()))

        plot_metadata = [
            # train
            (pl_alphas, "sgd", "o", "GD", "solid", "tab:blue"),
            (pl_alphas, "adam", "o", "FB-Adam", "solid", "tab:orange")
        ]

        for alphas_list, optimizer_val, marker, label, linestyle, color in plot_metadata:
            means = np.array([ np.mean(alphas_list[optimizer_val][np_lr]) for np_lr in np_lrs])
            stds = np.array([ np.std(alphas_list[optimizer_val][np_lr]) for np_lr in np_lrs])
            plt.plot(np.log10(np_lrs), means, marker=marker, label=label, linestyle=linestyle, color=color)
            plt.fill_between(np.log10(np_lrs), means - stds, means + stds, alpha=0.2)

        plt.xlabel("$\log_{10}(\eta)$")
        plt.ylabel("PL_Alpha_Hill")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        name="{}bulk_W_esd_pl_alpha.jpg".format(context["bulk_vis_dir"])
        plt.savefig(name)
        plt.clf()



if __name__ == "__main__":
    exp_context = {
        "L": 2,
        "n": 8000,
        "batch_size": 8000,
        "n_test": 200,
        "batch_size_test": 200,
        "h": 1500,
        "d": 1000,
        "label_noise_std": 0.3,
        "num_epochs": 10,
        "optimizer": ["sgd", "adam"],
        "momentum": 0,
        "weight_decay": 0,
        "lr": [0.001, 0.01, 0.1, 1, 10, 100, 1000, 2000, 3000],
        "reg_lambda": 0.01,
        "enable_weight_normalization": False,
        # NOTE: The probing now occurs based on number of steps.
        # set appropriate values based on n, batch_size and num_epochs.
        "probe_freq_steps": 10,
        "probe_weights": True,
        "probe_features": True,
        "fix_last_layer": True,
        "enable_ww": False, # setting `enable_ww` to True will open plots that need to be closed manually.
        "repeat": 1, # repeat counter for plotting means and std of results.
    }
    base_context = setup_runtime_context(context=exp_context)
    setup_logging(context=base_context)
    logger.info("*"*100)
    logger.info("context: \n{}".format(base_context))

    trainers = []
    contexts = []
    varying_params = ["optimizer", "lr"]
    optimizers = base_context["optimizer"]
    lrs = base_context["lr"]
    # handle bulk experiment vis
    base_context["bulk_vis_dir"] = base_context["vis_dir"]
   
    total_iterations = base_context["repeat"] * len(optimizers) * len(lrs)

    with tqdm(total=total_iterations) as pbar:
        for repeat_count in range(base_context["repeat"]):
            # reset caches
            base_teacher = get_teacher_model(context=base_context, use_cache=False, refresh_cache=True)
            _ = get_student_model(context=base_context, use_cache=False, refresh_cache=True)
            dataloaders = prepare_dataloaders(context=base_context, teacher=base_teacher, use_cache=False)

            for optimizer in optimizers:
                for idx, lr in enumerate(lrs):
                    context = deepcopy(base_context)
                    context["optimizer"] = optimizer
                    context["lr"] = lr
                    context["vis_dir"] = context["bulk_vis_dir"] + "lr{}/".format(lr)
                    if not os.path.exists(context["vis_dir"]):
                        os.makedirs(context["vis_dir"])

                    teacher = get_teacher_model(context=base_context, use_cache=True)
                    student = get_student_model(context=base_context, use_cache=True)
                    # fix last layer during training
                    if context["fix_last_layer"]:
                        student.final_layer.requires_grad_(requires_grad=False)

                    trainer = Trainer(context=context)
                    trained_student = trainer.run(
                        teacher=teacher,
                        student=student,
                        train_loader=dataloaders["train_loader"],
                        test_loader=dataloaders["test_loader"],
                        probe_loader=dataloaders["probe_loader"],
                    )
                    trainers.append(trainer)
                    contexts.append(context)
                    
                    pbar.update(1)

    plotter = BulkPlotter(trainers=trainers, contexts=contexts)
    plotter.plot_results()
