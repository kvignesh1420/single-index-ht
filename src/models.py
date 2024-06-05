"""
Teacher and student models
"""

import os
import sys
import numpy as np
import torch


class Teacher(torch.nn.Module):
    def __init__(self, context):
        super().__init__()
        self.context = context
        self.init_weights()
        self.activation_fn = torch.nn.Softplus()

    def init_weights(self):
        d = self.context["d"]
        beta = torch.randn(size=[d])
        beta = beta / torch.norm(beta, p=2)
        beta = beta.unsqueeze(1)
        self.beta = torch.nn.Parameter(beta)

    def forward(self, X):
        Z = X @ self.beta
        out = self.activation_fn(Z)
        ## apply input norm square
        # norm_scale = self.context["tau"]/self.context["d"]
        # out += norm_scale * torch.norm(X, p=2, dim=1).unsqueeze(1)
        ## apply noise
        n = X.shape[0]
        label_noise_std = self.context["label_noise_std"]
        out += torch.randn(size=[n]).unsqueeze(1) * label_noise_std
        return out


class Student2Layer(torch.nn.Module):
    def __init__(self, context):
        super().__init__()
        self.context = context
        self.activation_fn = torch.nn.Tanh()
        self.init_weights()
        self._assign_hooks()

    def init_weights(self):
        self.hidden_layer = torch.nn.Linear(
            in_features=self.context["d"], out_features=self.context["h"], bias=False
        )
        torch.nn.init.normal_(self.hidden_layer.weight)
        self.final_layer = torch.nn.Linear(
            in_features=self.context["h"], out_features=1, bias=False
        )
        torch.nn.init.normal_(self.final_layer.weight)
        self.layers = [self.hidden_layer, self.final_layer]

    @torch.no_grad()
    def _probe_affine_features(self, idx):
        def hook(model, inp, out):
            self.affine_features[idx] = out.detach()

        return hook

    @torch.no_grad()
    def _assign_hooks(self):
        self.affine_features = {}
        self.hidden_layer.register_forward_hook(self._probe_affine_features(idx=0))

    def forward(self, X):
        Z = self.hidden_layer(X)
        Z /= np.sqrt(self.context["d"])
        Z = self.activation_fn(Z)
        out = self.final_layer(Z)
        out /= np.sqrt(self.context["h"])
        return out


def get_teacher_model(context, use_cache=True, refresh_cache=False):
    teacher = Teacher(context=context).to(context["device"])
    teacher_model_path = os.path.join(context["teacher_model_dir"], "model.pth")
    if use_cache:
        if os.path.exists(teacher_model_path):
            print(
                "Loading the init state of teacher model from {}".format(
                    teacher_model_path
                )
            )
            teacher.load_state_dict(torch.load(teacher_model_path))
        else:
            print(
                "Saving the init state of teacher model to {}".format(
                    teacher_model_path
                )
            )
            torch.save(teacher.state_dict(), teacher_model_path)
    elif refresh_cache:
        print(
            "Refreshing the cache of the init state of teacher model to {}".format(
                teacher_model_path
            )
        )
        torch.save(teacher.state_dict(), teacher_model_path)
    return teacher


def get_student_model(context, use_cache=True, refresh_cache=False):
    if context["L"] == 2:
        student = Student2Layer(context=context).to(context["device"])
        student_model_path = os.path.join(context["student_model_dir"], "model.pth")
        if use_cache:
            if os.path.exists(student_model_path):
                print(
                    "Loading the init state of student model from {}".format(
                        student_model_path
                    )
                )
                student.load_state_dict(torch.load(student_model_path))
            else:
                print(
                    "Saving the init state of student model to {}".format(
                        student_model_path
                    )
                )
                torch.save(student.state_dict(), student_model_path)
        elif refresh_cache:
            print(
                "Refreshing the cache of the init state of student model to {}".format(
                    student_model_path
                )
            )
            torch.save(student.state_dict(), student_model_path)
        return student
    else:
        sys.exit("Only L=2 is supported for the student model.")
