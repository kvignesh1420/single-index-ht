import os
import json
import hashlib
import sys
from copy import deepcopy
import logging
logger = logging.getLogger(__name__)
import torch
from torch.utils.data import DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal
from data import TeacherDataset
from trainer import BulkTrainer
from models import Teacher
from models import Student2Layer

def prepare_config_hash(context):
    _string_context = json.dumps(context, sort_keys=True).encode("utf-8")
    parsed_context_hash = hashlib.md5(_string_context).hexdigest()
    return parsed_context_hash

def setup_runtime_context(context):
    # create a unique hash for the model
    config_uuid = prepare_config_hash(context=context)
    context["config_uuid"] = config_uuid
    context["device"] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    context["out_dir"] = "out/"
    bulk_vis_dir = context["out_dir"] + context["config_uuid"] + "/plots/"
    results_dir = context["out_dir"] + context["config_uuid"] + "/results/"
    if not os.path.exists(bulk_vis_dir):
        logger.info("Vis folder does not exist. Creating {}".format(bulk_vis_dir))
        os.makedirs(bulk_vis_dir)
    else:
        logger.info("Vis folder {} already exists!".format(bulk_vis_dir))
    if not os.path.exists(results_dir):
        logger.info("Resuls folder does not exist. Creating {}".format(results_dir))
        os.makedirs(results_dir)
    else:
        logger.info("Resuls folder {} already exists!".format(results_dir))
    context["bulk_vis_dir"] = bulk_vis_dir
    context["results_file"] = results_dir + "run.txt"
    return context

def prepare_train_input(context):
    n = context["n"]
    d = context["d"]
    dist = MultivariateNormal(torch.zeros(d), torch.eye(d))
    X = dist.sample(sample_shape=[n])
    return X

def prepare_test_input(context):
    n = context["n_test"]
    d = context["d"]
    dist = MultivariateNormal(torch.zeros(d), torch.eye(d))
    X = dist.sample(sample_shape=[n])
    return X


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
        "tau": 0.2,
        "num_epochs": 1,
        "optimizer": "adam",
        "momentum": 0,
        "weight_decay": 0,
        "lr": [0.1, 1, 10, 2000],
        "reg_lamba": 0.01,
        "probe_freq": 1
    }
    base_context = setup_runtime_context(context=exp_context)
    if base_context["L"] != 2:
        sys.exit("Only L=2 is supported.")

    logging.basicConfig(
        filename=base_context["results_file"],
        filemode='a',
        format='%(asctime)s, %(name)s %(levelname)s %(message)s',
        level=logging.INFO
    )
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.info("*"*100)
    logger.info("base context: \n{}".format(base_context))
    teacher = Teacher(context=base_context).to(base_context["device"])

    X_train = prepare_train_input(context=base_context)
    y_t_train = teacher(X=X_train)
    train_dataset = TeacherDataset(X=X_train, y_t=y_t_train)
    train_kwargs = {"batch_size": base_context["batch_size"], "shuffle": True}
    train_loader = DataLoader(train_dataset, **train_kwargs)

    # additional training data for computing feature metrics
    X_probe = prepare_train_input(context=base_context)
    y_t_probe = teacher(X=X_probe)
    probe_dataset = TeacherDataset(X=X_probe, y_t=y_t_probe)
    probe_kwargs = {"batch_size": base_context["n"], "shuffle": True}
    probe_loader = DataLoader(probe_dataset, **probe_kwargs)

    X_test = prepare_test_input(context=base_context)
    y_t_test = teacher(X=X_test)
    test_dataset = TeacherDataset(X=X_test, y_t=y_t_test)
    test_kwargs = {"batch_size": base_context["batch_size_test"], "shuffle": True}
    test_loader = DataLoader(test_dataset, **test_kwargs)

    students = []
    contexts = []
    varying_params = ["lr"]
    lrs = base_context["lr"]
    # varying learning rates for students
    for idx, lr in enumerate(lrs):
        context = deepcopy(base_context)
        context["lr"] = lr
        context["vis_dir"] = context["bulk_vis_dir"] + "lr{}/".format(lr)
        if not os.path.exists(context["vis_dir"]):
            logger.info("Vis folder does not exist. Creating {}".format(context["vis_dir"]))
            os.makedirs(context["vis_dir"])
        else:
            logger.info("Vis folder {} already exists!".format(context["vis_dir"]))
        logger.info("student: {} context: \n{}".format(idx, context))
        student = Student2Layer(context=context).to(context["device"])
        # fix last layer during training
        student.final_layer.requires_grad_(requires_grad=False)
        students.append(student)
        contexts.append(context)
        logger.info("Student: {}".format(student))
    logger.info("Teacher: {}".format(teacher))

    bulk_student_trainer = BulkTrainer(contexts=contexts, varying_params=varying_params)
    trained_students = bulk_student_trainer.run(
        teacher=teacher,
        students=students,
        train_loader=train_loader,
        test_loader=test_loader,
        probe_loader=probe_loader,
    )
