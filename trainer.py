import logging
logger = logging.getLogger(__name__)
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 16
from tracker import Tracker

class Trainer:
    def __init__(self, context):
        self.context = context
        self.tracker = Tracker(context=context)

    @torch.no_grad()
    def eval(self, student, test_loader):
        epoch_loss = 0
        # note that the reduction operation has been changed to "sum"
        loss_fn = torch.nn.MSELoss(reduction="sum")
        for batch_idx, (X, y_t) in enumerate(test_loader):
            X, y_t = X.to(self.context["device"]) , y_t.to(self.context["device"])
            student.zero_grad()
            pred = student(X)
            loss = loss_fn(pred, y_t)
            epoch_loss += loss.detach().cpu().numpy()
        epoch_loss /= self.context["n_test"]
        self.tracker.store_val_loss(loss=epoch_loss)
        logger.info("Val loss: {}".format(epoch_loss))

    def run(self, student, train_loader, test_loader):
        if self.context["optimizer"] == "sgd":
            optimizer = torch.optim.SGD(
                params=student.parameters(),
                lr=self.context["lr"],
                momentum=self.context["momentum"],
                weight_decay=self.context["weight_decay"],
            )
        elif self.context["optimizer"] == "adam":
            optimizer = torch.optim.Adam(
                params=student.parameters(),
                lr=self.context["lr"],
                weight_decay=self.context["weight_decay"],
            )
        loss_fn = torch.nn.MSELoss()
        self.tracker.probe_weights(student=student, epoch=0)
        for epoch in tqdm(range(1, self.context["num_epochs"]+1)):
            epoch_loss = 0
            for batch_idx, (X, y_t) in enumerate(train_loader):
                X, y_t = X.to(self.context["device"]) , y_t.to(self.context["device"])
                student.zero_grad()
                pred = student(X)
                loss = loss_fn(pred, y_t)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().cpu().numpy() * X.shape[0]
            if epoch%self.context["probe_freq"] == 0:
                epoch_loss /= self.context["n"]
                self.tracker.store_training_loss(loss=epoch_loss)
                self.tracker.probe_weights(student=student, epoch=epoch)
                self.eval(student=student, test_loader=test_loader)
        self.tracker.plot_training_loss()
        self.tracker.plot_val_loss()
        self.tracker.plot_initial_final_esd()
        return student
