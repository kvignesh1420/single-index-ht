from collections import defaultdict
import logging

logger = logging.getLogger(__name__)
import torch
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "axes.labelsize": 20,
        "legend.fontsize": 14,
    }
)


class BulkLRPlotter:
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

        final_training_losses = defaultdict(lambda: defaultdict(list))
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
            (
                final_training_losses,
                "adam",
                "o",
                "FB-Adam:train",
                "solid",
                "tab:orange",
            ),
            # test
            (final_val_losses, "sgd", "x", "GD:test", "dashed", "tab:green"),
            (final_val_losses, "adam", "x", "FB-Adam:test", "dashed", "tab:red"),
        ]

        for loss_list, optimizer_val, marker, label, linestyle, color in plot_metadata:
            means = np.array(
                [np.mean(loss_list[optimizer_val][np_lr]) for np_lr in np_lrs]
            )
            stds = np.array(
                [np.std(loss_list[optimizer_val][np_lr]) for np_lr in np_lrs]
            )
            plt.plot(
                np.log10(np_lrs),
                means,
                marker=marker,
                label=label,
                linestyle=linestyle,
                color=color,
            )
            plt.fill_between(
                np.log10(np_lrs), means - stds, means + stds, color=color, alpha=0.2
            )

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
            (KTAs, "adam", "o", "FB-Adam", "solid", "tab:orange"),
        ]

        for KTA_list, optimizer_val, marker, label, linestyle, color in plot_metadata:
            means = np.array(
                [np.mean(KTA_list[optimizer_val][np_lr]) for np_lr in np_lrs]
            )
            stds = np.array(
                [np.std(KTA_list[optimizer_val][np_lr]) for np_lr in np_lrs]
            )
            plt.plot(
                np.log10(np_lrs),
                means,
                marker=marker,
                label=label,
                linestyle=linestyle,
                color=color,
            )
            plt.fill_between(np.log10(np_lrs), means - stds, means + stds, alpha=0.2)

        plt.xlabel("$\log_{10}(\eta)$")
        plt.ylabel("KTA")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        name = "{}bulk_KTA.jpg".format(context["bulk_vis_dir"])
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
            (sims, "adam", "o", "FB-Adam", "solid", "tab:orange"),
        ]

        for sim_list, optimizer_val, marker, label, linestyle, color in plot_metadata:
            means = np.array(
                [np.mean(sim_list[optimizer_val][np_lr]) for np_lr in np_lrs]
            )
            stds = np.array(
                [np.std(sim_list[optimizer_val][np_lr]) for np_lr in np_lrs]
            )
            plt.plot(
                np.log10(np_lrs),
                means,
                marker=marker,
                label=label,
                linestyle=linestyle,
                color=color,
            )
            plt.fill_between(np.log10(np_lrs), means - stds, means + stds, alpha=0.2)

        plt.xlabel("$\log_{10}(\eta)$")
        plt.ylabel("$sim(W, \\beta^*)$")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        name = "{}bulk_W_beta_alignment.jpg".format(context["bulk_vis_dir"])
        plt.savefig(name)
        plt.clf()

    @torch.no_grad()
    def plot_step_weight_esd_pl_alpha(self):
        steps = list(self.trainers[0].tracker.step_weight_esd_pl_alpha_hill.keys())
        final_step = steps[-1]
        # temporary assertion for one-step experiments.
        assert len(steps) == 2
        layer_idx = 0

        pl_alphas_hill = defaultdict(lambda: defaultdict(list))
        pl_alphas_mle_ks = defaultdict(lambda: defaultdict(list))
        lrs = defaultdict(lambda: defaultdict(int))

        for trainer, context in zip(self.trainers, self.contexts):
            lr = context["lr"]
            lrs[context["optimizer"]][lr] += 1
            pl_alpha_hill = trainer.tracker.step_weight_esd_pl_alpha_hill[final_step][
                layer_idx
            ]
            pl_alphas_hill[context["optimizer"]][lr].append(pl_alpha_hill)

            pl_alpha_mle_ks = trainer.tracker.step_weight_esd_pl_alpha_mle_ks[
                final_step
            ][layer_idx]
            pl_alphas_mle_ks[context["optimizer"]][lr].append(pl_alpha_mle_ks)

        assert lrs["sgd"] == lrs["adam"]
        np_lrs = np.sort(list(lrs["sgd"].keys()))

        plot_metadata = [
            (pl_alphas_hill, "sgd", "o", "GD", "solid", "tab:blue"),
            (pl_alphas_hill, "adam", "o", "FB-Adam", "solid", "tab:orange"),
        ]

        for (
            alphas_list,
            optimizer_val,
            marker,
            label,
            linestyle,
            color,
        ) in plot_metadata:
            means = np.array(
                [np.mean(alphas_list[optimizer_val][np_lr]) for np_lr in np_lrs]
            )
            stds = np.array(
                [np.std(alphas_list[optimizer_val][np_lr]) for np_lr in np_lrs]
            )
            plt.plot(
                np.log10(np_lrs),
                means,
                marker=marker,
                label=label,
                linestyle=linestyle,
                color=color,
            )
            plt.fill_between(np.log10(np_lrs), means - stds, means + stds, alpha=0.2)

        plt.xlabel("$\log_{10}(\eta)$")
        plt.ylabel("PL_Alpha_Hill")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        name = "{}bulk_W_esd_pl_alpha_hill.jpg".format(context["bulk_vis_dir"])
        plt.savefig(name)
        plt.clf()

        plot_metadata = [
            (pl_alphas_mle_ks, "sgd", "o", "GD", "solid", "tab:blue"),
            (pl_alphas_mle_ks, "adam", "o", "FB-Adam", "solid", "tab:orange"),
        ]

        for (
            alphas_list,
            optimizer_val,
            marker,
            label,
            linestyle,
            color,
        ) in plot_metadata:
            means = np.array(
                [np.mean(alphas_list[optimizer_val][np_lr]) for np_lr in np_lrs]
            )
            stds = np.array(
                [np.std(alphas_list[optimizer_val][np_lr]) for np_lr in np_lrs]
            )
            plt.plot(
                np.log10(np_lrs),
                means,
                marker=marker,
                label=label,
                linestyle=linestyle,
                color=color,
            )
            plt.fill_between(np.log10(np_lrs), means - stds, means + stds, alpha=0.2)

        plt.xlabel("$\log_{10}(\eta)$")
        plt.ylabel("PL_Alpha_KS")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        name = "{}bulk_W_esd_pl_alpha_mle_ks.jpg".format(context["bulk_vis_dir"])
        plt.savefig(name)
        plt.clf()


class BulkLossPlotter:
    def __init__(self, trainers, varying_param):
        self.trainers = trainers
        self.contexts = [trainer.context for trainer in self.trainers]
        self.varying_param = varying_param

    def plot_results(self):
        self.plot_losses()

    def get_label_prefix(self):
        if self.varying_param == "reg_lambda":
            label_prefix = "$\lambda$"
        elif self.varying_param == "gamma":
            label_prefix = "$\gamma$"
        elif self.varying_param == "label_noise_std":
            label_prefix = "$\\rho_{e}$"
        else:
            label_prefix = self.varying_param
        return label_prefix

    @torch.no_grad()
    def plot_losses(self):
        steps = list(self.trainers[0].tracker.val_loss.keys())

        training_losses = defaultdict(list)
        val_losses = defaultdict(list)
        param_values = defaultdict(int)

        for trainer, context in zip(self.trainers, self.contexts):
            if self.varying_param == "gamma":
                param_val = context["lr_scheduler_kwargs"]["gamma"]
            else:
                param_val = context[self.varying_param]
            param_values[param_val] += 1

            training_loss_arr = np.array(list(trainer.tracker.training_loss.values()))
            training_losses[param_val].append(training_loss_arr)
            val_loss_arr = np.array(list(trainer.tracker.val_loss.values()))
            val_losses[param_val].append(val_loss_arr)

        for param_val in param_values.keys():
            label_prefix = self.get_label_prefix()
            label_name = "{}={}".format(label_prefix, param_val)

            plot_metadata = [
                (training_losses, "o", "solid"),
                (val_losses, "x", "dashed"),
            ]

            for losses, marker, linestyle in plot_metadata:
                arr = losses[param_val]
                arr = np.array(arr)
                means = np.mean(arr, axis=0)
                stds = np.std(arr, axis=0)

                plt.plot(
                    steps, means, marker=marker, label=label_name, linestyle=linestyle
                )
                plt.fill_between(steps, means - stds, means + stds, alpha=0.2)

        plt.xlabel("step")
        plt.ylabel("loss")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("{}bulk_losses.jpg".format(context["bulk_vis_dir"]))
        plt.clf()
