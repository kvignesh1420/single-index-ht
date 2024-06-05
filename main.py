"""
Launcher script for experiments
"""

import sys
import logging

logger = logging.getLogger(__name__)
from src.trainer import Trainer
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


if __name__ == "__main__":
    exp_context = parse_config()
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
