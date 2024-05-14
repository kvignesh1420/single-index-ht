import os
import sys
from copy import deepcopy
import logging
logger = logging.getLogger(__name__)
from trainer import MultiStepLossBulkTrainer
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
        "num_epochs": 10,
        "optimizer": "adam",
        "momentum": 0,
        "weight_decay": 0,
        "lr": 1,
        "lr_scheduler_cls": "StepLR",
        "lr_scheduler_kwargs": {
            "step_size": 1,
            "gamma": [0.2, 0.4, 0.6, 0.8]
        },
        "reg_lambda": 1e-2,
        "enable_weight_normalization": False,
        # NOTE: The probing now occurs based on number of steps.
        # set appropriate values based on n, batch_size and num_epochs.
        "probe_freq_steps": 1,
        "probe_weights": False,
        "probe_features": False,
        "fix_last_layer": True,
        "enable_ww": False # setting `enable_ww` to True will open plots that need to be closed manually.
    }
    base_context = setup_runtime_context(context=exp_context)
    setup_logging(context=base_context)
    logger.info("*"*100)
    logger.info("context: \n{}".format(base_context))

    students = []
    contexts = []
    varying_params = ["gamma"]
    gammas = base_context["lr_scheduler_kwargs"]["gamma"]

    # handle bulk experiment vis
    base_context["bulk_vis_dir"] = base_context["vis_dir"]

    teachers = []
    dataloaders_list = []
    for idx, gamma in enumerate(gammas):
        context = deepcopy(base_context)
        context["lr_scheduler_kwargs"]["gamma"] = gamma
        del context["device"]
        context = setup_runtime_context(context=context)
        context["vis_dir"] = context["bulk_vis_dir"] + "gamma{}/".format(gamma)
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

        teacher = get_teacher_model(context=context)
        teachers.append(teacher)

        dataloaders = prepare_dataloaders(context=context, teacher=teacher)
        dataloaders_list.append(dataloaders)

    # logger.info("Teacher: {}".format(teacher))


    bulk_student_trainer = MultiStepLossBulkTrainer(contexts=contexts, varying_params=varying_params)
    trained_students = bulk_student_trainer.run(
        teachers=teachers,
        students=students,
        train_loaders=[dataloader["train_loader"] for dataloader in dataloaders_list],
        test_loaders=[dataloader["test_loader"] for dataloader in dataloaders_list],
        probe_loaders=[dataloader["probe_loader"] for dataloader in dataloaders_list],
    )
