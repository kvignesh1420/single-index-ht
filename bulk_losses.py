import os
import sys
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)
from tqdm import tqdm
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "axes.labelsize": 20,
        "legend.fontsize": 14,
    }
)

from src.trainer import Trainer
from src.plotter import BulkLossPlotter
from src.data import prepare_dataloaders
from src.models import get_student_model
from src.models import get_teacher_model
from src.utils import setup_runtime_context
from src.utils import parse_config


def setup_logging(context):
    logging.basicConfig(
        filename=context["results_file"],
        filemode="a",
        format="%(asctime)s, %(name)s %(levelname)s %(message)s",
        level=logging.INFO,
    )
    logger.addHandler(logging.StreamHandler(sys.stdout))


def prepare_contexts(base_context, param_vals, varying_param):
    contexts = {}
    for param_val in param_vals:
        context = deepcopy(base_context)
        context[varying_param] = param_val
        if varying_param == "n":
            context["batch_size"] = param_val
        del context["device"]
        context = setup_runtime_context(context=context)
        context["vis_dir"] = context["bulk_vis_dir"] + "{}{}/".format(
            varying_param, param_val
        )
        if not os.path.exists(context["vis_dir"]):
            os.makedirs(context["vis_dir"])
        contexts[param_val] = context
    return contexts


def train_with_diff_dataloaders(base_context, contexts, total_iterations, param_vals):
    """
    When `varying_param` is `n` or `label_noise_std`, then for each `param_val`,
    we create a fresh dataset `repeat` number of times to run the trainer.
    """
    trainers = []
    with tqdm(total=total_iterations) as pbar:
        for param_val in param_vals:
            context = contexts[param_val]
            teacher = get_teacher_model(
                context=context, use_cache=False, refresh_cache=True
            )
            common_student = get_student_model(
                context=context, use_cache=False, refresh_cache=True
            )
            for _ in range(base_context["repeat"]):
                student = deepcopy(common_student)
                student._assign_hooks()
                # fix last layer during training
                if context["fix_last_layer"]:
                    student.final_layer.requires_grad_(requires_grad=False)
                dataloaders = prepare_dataloaders(
                    context=context, teacher=teacher, use_cache=False
                )
                trainer = Trainer(context=context)
                trainer.run(
                    teacher=teacher,
                    student=student,
                    train_loader=dataloaders["train_loader"],
                    test_loader=dataloaders["test_loader"],
                    probe_loader=dataloaders["probe_loader"],
                )
                trainers.append(trainer)
                pbar.update(1)
    return trainers


def train_with_same_dataloaders(base_context, contexts, total_iterations, param_vals):
    """
    When `varying_param` is `reg_lambda` or `gamma`, then for each `param_val`,
    we use a common dataset `repeat` number of times to run the trainer. Thus,
    ensuring a fair comparison between `param_val`s.
    """
    trainers = []
    with tqdm(total=total_iterations) as pbar:
        for _ in range(base_context["repeat"]):
            common_teacher = get_teacher_model(
                context=context, use_cache=False, refresh_cache=True
            )
            # same dataloaders for all param_vals
            common_dataloaders = prepare_dataloaders(
                context=base_context, teacher=common_teacher, use_cache=False
            )
            common_student = get_student_model(
                context=context, use_cache=False, refresh_cache=True
            )
            for param_val in param_vals:
                context = contexts[param_val]
                # same student for varying param_val
                student = deepcopy(common_student)
                # fix last layer during training
                if context["fix_last_layer"]:
                    student.final_layer.requires_grad_(requires_grad=False)

                trainer = Trainer(context=context)
                trainer.run(
                    teacher=common_teacher,
                    student=student,
                    train_loader=common_dataloaders["train_loader"],
                    test_loader=common_dataloaders["test_loader"],
                    probe_loader=common_dataloaders["probe_loader"],
                )
                trainers.append(trainer)
                pbar.update(1)
    return trainers


def main():
    exp_context = parse_config()
    base_context = setup_runtime_context(context=exp_context)
    setup_logging(context=base_context)
    logger.info("*" * 100)
    logger.info("context: \n{}".format(base_context))

    # handle bulk experiment vis
    base_context["bulk_vis_dir"] = base_context["vis_dir"]
    varying_param = "label_noise_std"
    param_vals = base_context[varying_param]
    total_iterations = base_context["repeat"] * len(param_vals)

    # prepare lists
    contexts = prepare_contexts(
        base_context=base_context, param_vals=param_vals, varying_param=varying_param
    )

    if varying_param in ["n", "label_noise_std"]:
        train_fn = train_with_diff_dataloaders
    elif varying_param in ["reg_lambda", "gamma"]:
        train_fn = train_with_same_dataloaders

    trainers = train_fn(
        base_context=base_context,
        contexts=contexts,
        total_iterations=total_iterations,
        param_vals=param_vals,
    )
    plotter = BulkLossPlotter(trainers=trainers, varying_param=varying_param)
    plotter.plot_results()


if __name__ == "__main__":
    main()
