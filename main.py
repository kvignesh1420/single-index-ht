import os
import json
import hashlib
import sys
import logging
logger = logging.getLogger(__name__)
import torch
from torch.utils.data import DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal
from data import TeacherDataset
from trainer import Trainer
from models import Teacher
from models import Student2Layer
from models import Student3Layer

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
    vis_dir = context["out_dir"] + context["config_uuid"] + "/plots/"
    results_dir = context["out_dir"] + context["config_uuid"] + "/results/"
    results_file = results_dir + "run.txt"
    if not os.path.exists(vis_dir):
        logger.info("Vis folder does not exist. Creating {}".format(vis_dir))
        os.makedirs(vis_dir)
    else:
        logger.info("Vis folder {} already exists!".format(vis_dir))
    if not os.path.exists(results_dir):
        logger.info("Resuls folder does not exist. Creating {}".format(results_dir))
        os.makedirs(results_dir)
    else:
        logger.info("Resuls folder {} already exists!".format(results_dir))
    context["vis_dir"] = vis_dir
    context["results_file"] = results_file
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

def compute_alignment(teacher, student):
    W = student.layers[0].weight.data.clone()
    beta = teacher.beta.squeeze().to(context["device"])
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    # logger.info(U.shape, Vh.shape, beta.shape)
    sim = torch.dot(Vh[0], beta)
    sim /= torch.norm(Vh[0], p=2)
    return sim

if __name__ == "__main__":
    exp_context = {
        "L": 2,
        "n": 2000,
        "n_test": 200,
        "batch_size": 2000,
        "h": 1500,
        "d": 1000,
        "label_noise_std": 0.3,
        "tau": 0.2,
        "num_epochs": 1,
        "optimizer": "adam",
        "momentum": 0,
        "weight_decay": 0,
        "lr": 2000,
        "probe_freq": 1
    }
    context = setup_runtime_context(context=exp_context)
    logging.basicConfig(
        filename=context["results_file"],
        filemode='a',
        format='%(asctime)s, %(name)s %(levelname)s %(message)s',
        level=logging.INFO
    )
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.info("*"*100)
    logger.info("context: \n{}".format(context))
    teacher = Teacher(context=context).to(context["device"])
    if context["L"] == 2:
        student = Student2Layer(context=context).to(context["device"])
    elif context["L"] == 3:
        student = Student3Layer(context=context).to(context["device"])
    else:
        sys.exit("L should be either 2 or 3.")

    X_train = prepare_train_input(context=context)
    y_t_train = teacher(X=X_train)
    train_dataset = TeacherDataset(X=X_train, y_t=y_t_train)
    train_kwargs = {"batch_size": context["batch_size"], "shuffle": True}
    train_loader = DataLoader(train_dataset, **train_kwargs)

    X_test = prepare_test_input(context=context)
    y_t_test = teacher(X=X_test)
    test_dataset = TeacherDataset(X=X_test, y_t=y_t_test)
    test_kwargs = {"batch_size": context["batch_size"], "shuffle": True}
    test_loader = DataLoader(test_dataset, **test_kwargs)
    student_trainer = Trainer(context=context)

    logger.info("Teacher: {}".format(teacher))
    logger.info("Student: {}".format(student))

    sim = compute_alignment(teacher=teacher, student=student)
    logger.info("initial alignment (cosine of angle) between "\
                 "link function and first PC of W: {}".format(sim))

    trained_student = student_trainer.run(
        student=student,
        train_loader=train_loader,
        test_loader=test_loader
    )

    sim = compute_alignment(teacher=teacher, student=trained_student)
    logger.info("final alignment (cosine of angle) between" \
                 "link function and first PC of W: {}".format(sim))
