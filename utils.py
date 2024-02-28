"""Utils for handling context and training"""


import os
import json
import hashlib
import torch
import logging
logger = logging.getLogger(__name__)


def prepare_data_hash(context):
    relevant_fields = ["n", "n_test", "d"]
    data_context = { k: v for k, v in context.items() if k in relevant_fields  }
    _string_data_context = json.dumps(data_context, sort_keys=True).encode("utf-8")
    parsed_data_context_hash = hashlib.md5(_string_data_context).hexdigest()
    return parsed_data_context_hash

def prepare_teacher_model_hash(context):
    relevant_fields = ["d", "label_noise_std", "tau"]
    model_context = { k: v for k, v in context.items() if k in relevant_fields  }
    _string_model_context = json.dumps(model_context, sort_keys=True).encode("utf-8")
    parsed_model_context_hash = hashlib.md5(_string_model_context).hexdigest()
    return parsed_model_context_hash

def prepare_student_model_hash(context):
    relevant_fields = ["L", "d", "h"]
    model_context = { k: v for k, v in context.items() if k in relevant_fields  }
    _string_model_context = json.dumps(model_context, sort_keys=True).encode("utf-8")
    parsed_model_context_hash = hashlib.md5(_string_model_context).hexdigest()
    return parsed_model_context_hash

def prepare_config_hash(context):
    _string_context = json.dumps(context, sort_keys=True).encode("utf-8")
    parsed_context_hash = hashlib.md5(_string_context).hexdigest()
    return parsed_context_hash

def setup_runtime_context(context):
    # create a unique hash for the model
    config_uuid = prepare_config_hash(context=context)
    context["config_uuid"] = config_uuid
    context["device"] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # data
    data_config_uuid = prepare_data_hash(context=context)
    context["data_dir"] = "data/{}".format(data_config_uuid)
    os.makedirs(context["data_dir"], exist_ok=True)
    # teacher model
    teacher_config_uuid = prepare_teacher_model_hash(context=context)
    context["teacher_model_dir"] = "teacher_models/{}".format(teacher_config_uuid)
    os.makedirs(context["teacher_model_dir"], exist_ok=True)
    # student model
    student_config_uuid = prepare_student_model_hash(context=context)
    context["student_model_dir"] = "student_models/{}".format(student_config_uuid)
    os.makedirs(context["student_model_dir"], exist_ok=True)
    # outputs
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
