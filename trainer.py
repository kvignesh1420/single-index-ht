from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from tracker import Tracker

class Trainer:
    def __init__(self, context):
        self.context = context
        self.tracker = Tracker(context=context)

    @torch.no_grad()
    def eval(self, student, test_loader):
        loss_val = 0
        loss_fn = torch.nn.MSELoss(reduction="sum")
        for batch_idx, (X, y_t) in enumerate(test_loader):
            X, y_t = X.to(self.context["device"]) , y_t.to(self.context["device"])
            student.zero_grad()
            pred = student(X)
            loss = loss_fn(pred, y_t)
            loss_val += loss.detach().cpu().numpy()
        loss_val /= self.context["n_test"]
        print("Val loss: {}".format(loss_val))

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
        loss_values = []
        self.tracker.probe_weights(student=student, epoch=0)
        for epoch in tqdm(range(1, self.context["num_epochs"]+1)):
            for batch_idx, (X, y_t) in enumerate(train_loader):
                X, y_t = X.to(self.context["device"]) , y_t.to(self.context["device"])
                student.zero_grad()
                pred = student(X)
                loss = loss_fn(pred, y_t)
                loss.backward()
                optimizer.step()
            if epoch%self.context["probe_freq"] == 0:
                loss_values.append(loss.detach().cpu().numpy())
                self.tracker.probe_weights(student=student, epoch=epoch)
                self.eval(student=student, test_loader=test_loader)
        plt.plot(loss_values)
        plt.grid(True)
        plt.savefig("{}loss.jpg".format(self.context["vis_dir"]))
        plt.clf()
        self.tracker.plot_initial_final_esd()
        return student

