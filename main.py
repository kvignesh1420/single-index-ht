"""
Launcher script for experiments
"""

import sys
import logging

logger = logging.getLogger(__name__)
from trainer import Trainer
from data import prepare_dataloaders
from models import get_student_model
from models import get_teacher_model
from utils import setup_runtime_context


def setup_logging(context):
    logging.basicConfig(
        filename=context["results_file"],
        filemode="a",
        format="%(asctime)s, %(name)s %(levelname)s %(message)s",
        level=logging.INFO,
    )
    logger.addHandler(logging.StreamHandler(sys.stdout))


if __name__ == "__main__":
    exp_context = {
        "L": 2,
        "n": 2000,
        "batch_size": 2000,
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
        "reg_lambda": 1e-2,
        "enable_weight_normalization": False,
        # NOTE: The probing now occurs based on number of steps.
        # set appropriate values based on n, batch_size and num_epochs.
        "probe_freq_steps": 10,
        "probe_weights": True,
        # set "plot_overlaps": True for plotting the overlaps between singular vectors.
        # Note: make sure that "probe_freq_steps": 1 when this is set.
        "plot_overlaps": False,
        "probe_features": True,
        "fix_last_layer": True,
        "enable_ww": False,  # setting `enable_ww` to True will open plots that need to be closed manually.
    }
    context = setup_runtime_context(context=exp_context)
    setup_logging(context=context)
    logger.info("*" * 100)
    logger.info("context: \n{}".format(context))

    teacher = get_teacher_model(context=context)
    student = get_student_model(context=context)
    # fix last layer during training
    if context["fix_last_layer"]:
        student.final_layer.requires_grad_(requires_grad=False)

    logger.info("Teacher: {}".format(teacher))
    logger.info("Student: {}".format(student))

    dataloaders = prepare_dataloaders(context=context, teacher=teacher)
    student_trainer = Trainer(context=context)
    trained_student = student_trainer.run(
        teacher=teacher,
        student=student,
        train_loader=dataloaders["train_loader"],
        test_loader=dataloaders["test_loader"],
        probe_loader=dataloaders["probe_loader"],
    )
