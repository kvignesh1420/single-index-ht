import os
import sys
from copy import deepcopy
import logging
logger = logging.getLogger(__name__)
from trainer import OneStepBulkTrainer
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
        "num_epochs": 1,
        "optimizer": ["sgd", "adam"],
        "momentum": 0,
        "weight_decay": 0,
        "lr": [0.001, 0.01, 0.1, 1, 10, 100, 1000, 2000, 3000],
        "reg_lambda": 0.01,
        "enable_weight_normalization": False,
        # NOTE: The probing now occurs based on number of steps.
        # set appropriate values based on n, batch_size and num_epochs.
        "probe_freq_steps": 1,
        "probe_weights": True,
        "probe_features": True,
        "fix_last_layer": True,
        "enable_ww": False # setting `enable_ww` to True will open plots that need to be closed manually.
    }
    base_context = setup_runtime_context(context=exp_context)
    setup_logging(context=base_context)
    logger.info("*"*100)
    logger.info("context: \n{}".format(base_context))

    teacher = get_teacher_model(context=base_context)

    students = []
    contexts = []
    varying_params = ["optimizer", "lr"]
    optimizers = base_context["optimizer"]
    lrs = base_context["lr"]
    # handle bulk experiment vis
    base_context["bulk_vis_dir"] = base_context["vis_dir"]

    for optimizer in optimizers:
        for idx, lr in enumerate(lrs):
            context = deepcopy(base_context)
            context["optimizer"] = optimizer
            context["lr"] = lr
            context["vis_dir"] = context["bulk_vis_dir"] + "lr{}/".format(lr)
            if not os.path.exists(context["vis_dir"]):
                logger.info("Vis folder does not exist. Creating {}".format(context["vis_dir"]))
                os.makedirs(context["vis_dir"])
            else:
                logger.info("Vis folder {} already exists!".format(context["vis_dir"]))
            logger.info("student: {} context: \n{}".format(idx, context))

            student = get_student_model(context=context)
            # fix last layer during training
            if context["fix_last_layer"]:
                student.final_layer.requires_grad_(requires_grad=False)
            students.append(student)
            contexts.append(context)
            logger.info("Student: {}".format(student))
        logger.info("Teacher: {}".format(teacher))

    dataloaders = prepare_dataloaders(context=base_context, teacher=teacher)
    bulk_student_trainer = OneStepBulkTrainer(contexts=contexts, varying_params=varying_params)
    trained_students = bulk_student_trainer.run(
        teacher=teacher,
        students=students,
        train_loader=dataloaders["train_loader"],
        test_loader=dataloaders["test_loader"],
        probe_loader=dataloaders["probe_loader"]
    )
