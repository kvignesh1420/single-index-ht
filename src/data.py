"""
Data preparation module
"""

import os
import logging

logger = logging.getLogger(__name__)
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal


class TeacherDataset(Dataset):
    def __init__(self, context, teacher, type, use_cache=True) -> None:
        super().__init__()
        self.context = context
        self.teacher = teacher
        self.type = type
        self.use_cache = use_cache
        self.prepare_data()

    def _prepare_fresh_data(self, n, d):
        dist = MultivariateNormal(torch.zeros(d), torch.eye(d))
        self.X = dist.sample(sample_shape=[n])
        self.y = self.teacher(X=self.X.to(self.context["device"]))
        self.save_state()

    def prepare_data(self):
        if self.type == "test":
            n = self.context["n_test"]
        else:
            n = self.context["n"]
        d = self.context["d"]
        # try loading saved data
        if self.use_cache:
            load_success = self.load_state()
            if not load_success:
                self._prepare_fresh_data(n=n, d=d)
        else:
            self._prepare_fresh_data(n=n, d=d)

    def load_state(self):
        print("loading {} X, y from {}".format(self.type, self.context["data_dir"]))
        names = ["X_{}.pt".format(self.type), "y_{}.pt".format(self.type)]
        for name in names:
            filepath = os.path.join(self.context["data_dir"], name)
            if not os.path.exists(filepath):
                error = "Attempting to load {} , which doesn't exist. Data will be regenerated.".format(
                    filepath
                )
                logger.warning(error)
                return False

        self.X = torch.load(
            os.path.join(self.context["data_dir"], "X_{}.pt".format(self.type))
        )
        self.y = torch.load(
            os.path.join(self.context["data_dir"], "y_{}.pt".format(self.type))
        )
        print("Load sucessful.")
        return True

    def save_state(self):
        print("saving {} X, y to {}".format(self.type, self.context["data_dir"]))
        torch.save(
            self.X, os.path.join(self.context["data_dir"], "X_{}.pt".format(self.type))
        )
        torch.save(
            self.y, os.path.join(self.context["data_dir"], "y_{}.pt".format(self.type))
        )

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]


def prepare_dataloaders(context, teacher, use_cache=True):
    train_dataset = TeacherDataset(
        context=context, teacher=teacher, type="train", use_cache=use_cache
    )
    train_kwargs = {"batch_size": context["batch_size"], "shuffle": True}
    train_loader = DataLoader(train_dataset, **train_kwargs)

    # additional training data for computing feature metrics
    if context["fix_last_layer"]:
        probe_dataset = TeacherDataset(
            context=context, teacher=teacher, type="probe", use_cache=use_cache
        )
        probe_kwargs = {"batch_size": context["n"], "shuffle": True}
        probe_loader = DataLoader(probe_dataset, **probe_kwargs)
    else:
        # probe loader will not be used in this case for computing loss.
        # But we stay with the current dictionary output in case we need it.
        probe_loader = train_loader

    test_dataset = TeacherDataset(
        context=context, teacher=teacher, type="test", use_cache=use_cache
    )
    test_kwargs = {"batch_size": context["batch_size_test"], "shuffle": True}
    test_loader = DataLoader(test_dataset, **test_kwargs)

    return {
        "train_loader": train_loader,
        "probe_loader": probe_loader,
        "test_loader": test_loader,
    }
