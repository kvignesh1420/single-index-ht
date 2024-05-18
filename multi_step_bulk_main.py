import os
import sys
from collections import defaultdict
from copy import deepcopy
import logging
logger = logging.getLogger(__name__)
from trainer import Trainer
import numpy as np
import torch
from tqdm import tqdm
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


class BulkLossPlotter:
    def __init__(self, trainers, contexts, varying_params):
        self.trainers = trainers
        self.contexts = contexts
        self.varying_params = varying_params
        assert len(self.varying_params) == 1

    def plot_results(self):
        self.plot_losses()
    
    def get_label_prefix(self, varying_param):
        if varying_param == "reg_lambda":
            label_prefix = "$\lambda$"
        elif varying_param == "gamma":
            label_prefix = "$\gamma$"
        elif varying_param == "label_noise_std":
            label_prefix = "$\\rho_{e}$"
        else:
            label_prefix = varying_param
        return label_prefix

    @torch.no_grad()
    def plot_losses(self):
        varying_param = self.varying_params[0]
        steps = list(self.trainers[0].tracker.val_loss.keys())
        
        training_losses = defaultdict(list)
        val_losses  = defaultdict(list)
        param_values = defaultdict(int)

        for trainer, context in zip(self.trainers, self.contexts):
            if varying_param == "gamma":
                param_val = context["lr_scheduler_kwargs"]["gamma"]
            else:
                param_val = context[varying_param]
            param_values[param_val] += 1
            
            training_loss_arr = np.array(list(trainer.tracker.training_loss.values()))
            training_losses[param_val].append(training_loss_arr)
            val_loss_arr = np.array(list(trainer.tracker.val_loss.values()))
            val_losses[param_val].append(val_loss_arr)

            # plt.plot(steps, training_losses, marker='o', label="{}={}".format(label_name, label_value ))
            # plt.plot(steps, val_losses, marker='x', label="{}={}".format(label_name, label_value), linestyle='dashed')
        
        for param_val in param_values.keys():
            label_prefix = self.get_label_prefix(varying_param=varying_param)
            label_name = "{}={}".format(label_prefix, param_val)

            plot_metadata = [
                (training_losses, "o", "solid"),
                (val_losses, "x", "dashed")
            ]

            for losses, marker, linestyle in plot_metadata:
                arr = losses[param_val]
                arr = np.array(arr)
                means = np.mean(arr, axis=0)
                stds = np.std(arr, axis=0)

                plt.plot(steps, means, marker=marker, label=label_name, linestyle=linestyle)
                plt.fill_between(steps, means - stds, means + stds, alpha=0.2)


        plt.xlabel("step")
        plt.ylabel("loss")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("{}bulk_losses.jpg".format(context["bulk_vis_dir"]))
        plt.clf()


if __name__ == "__main__":
    exp_context = {
        "L": 2,
        "n": [500, 2000, 4000, 8000],
        "batch_size": 8000,
        "n_test": 200,
        "batch_size_test": 200,
        "h": 1500,
        "d": 1000,
        "label_noise_std": 0.3,
        "num_epochs": 10,
        "optimizer": "adam",
        "momentum": 0,
        "weight_decay": 0,
        "lr": 1,
        # "lr_scheduler_cls": "StepLR",
        # "lr_scheduler_kwargs": {
        #     "step_size": 1,
        #     "gamma": 0.4
        # },
        "reg_lambda": 1e-2,
        "enable_weight_normalization": False,
        # NOTE: The probing now occurs based on number of steps.
        # set appropriate values based on n, batch_size and num_epochs.
        "probe_freq_steps": 1,
        "probe_weights": False,
        "probe_features": False,
        "fix_last_layer": True,
        "enable_ww": False, # setting `enable_ww` to True will open plots that need to be closed manually.
        "repeat": 5, # repeat counter for plotting means and std of results.
    }
    base_context = setup_runtime_context(context=exp_context)
    setup_logging(context=base_context)
    logger.info("*"*100)
    logger.info("context: \n{}".format(base_context))

    # handle bulk experiment vis
    base_context["bulk_vis_dir"] = base_context["vis_dir"]

    contexts = []
    trainers = []
    varying_params = ["reg_lambda"]
    n_list = base_context["n"]

    total_iterations = base_context["repeat"] * len(n_list)

    with tqdm(total=total_iterations) as pbar:
        for repeat_count in range(base_context["repeat"]):
            base_teacher = get_teacher_model(context=base_context, use_cache=False, refresh_cache=True)
            _ = get_student_model(context=base_context, use_cache=False, refresh_cache=True)
            dataloaders = prepare_dataloaders(context=base_context, teacher=base_teacher, use_cache=False)

            for idx, n in enumerate(n_list):
                context = deepcopy(base_context)
                context["n"] = n

                del context["device"]
                context = setup_runtime_context(context=context)
                
                context["vis_dir"] = context["bulk_vis_dir"] + "reg_lambda{}/".format(reg_lambda)

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

    plotter = BulkLossPlotter(trainers=trainers, contexts=contexts, varying_params=varying_params)
    plotter.plot_results()
