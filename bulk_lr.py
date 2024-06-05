import os
import sys
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)
from tqdm import tqdm

from src.trainer import Trainer
from src.plotter import BulkLRPlotter
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
    base_context = setup_runtime_context(context=exp_context)
    setup_logging(context=base_context)
    logger.info("*" * 100)
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
            base_teacher = get_teacher_model(
                context=base_context, use_cache=False, refresh_cache=True
            )
            _ = get_student_model(
                context=base_context, use_cache=False, refresh_cache=True
            )
            dataloaders = prepare_dataloaders(
                context=base_context, teacher=base_teacher, use_cache=False
            )

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

    plotter = BulkLRPlotter(trainers=trainers, contexts=contexts)
    plotter.plot_results()
