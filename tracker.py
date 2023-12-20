from collections import OrderedDict
import torch
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 16

class Tracker:
    def __init__(self, context):
        self.context = context
        self.epoch_weight_esd = OrderedDict()
        self.training_loss = []
        self.val_loss = []

    @torch.no_grad()
    def store_training_loss(self, loss):
        self.training_loss.append(loss)

    @torch.no_grad()
    def plot_training_loss(self):
        plt.plot(self.training_loss)
        plt.grid(True)
        plt.savefig("{}training_loss.jpg".format(self.context["vis_dir"]))
        plt.clf()

    @torch.no_grad()
    def store_val_loss(self, loss):
        self.val_loss.append(loss)

    @torch.no_grad()
    def plot_val_loss(self):
        plt.plot(self.val_loss)
        plt.grid(True)
        plt.savefig("{}val_loss.jpg".format(self.context["vis_dir"]))
        plt.clf()

    @torch.no_grad()
    def probe_weights(self, student, epoch):
        for idx, layer in enumerate(student.layers):
            W = layer.weight.data.clone()
            S_W = torch.linalg.svdvals(W)
            name="{}W{}_epoch{}".format(self.context["vis_dir"], idx, epoch)
            self.plot_svd(S_W = S_W, name=name)
            WtW = W.t() @ W
            S_WtW = torch.linalg.svdvals(WtW)
            self.plot_esd(S_WtW = S_WtW, name=name)
            if epoch not in self.epoch_weight_esd:
                self.epoch_weight_esd[epoch] = OrderedDict()
            self.epoch_weight_esd[epoch][idx] = S_WtW

    @torch.no_grad()
    def plot_svd(self, S_W, name):
       plt.bar(x=list(range(S_W.shape[0])), height=S_W.cpu().numpy())
       plt.xlabel("$i$")
       plt.ylabel("$\lambda_i$")
       plt.savefig("{}_sv.jpg".format(name))
       plt.clf()

    @torch.no_grad()
    def plot_esd(self, S_WtW, name):
       vals = np.log10(S_WtW.cpu().numpy())
       plt.hist(vals, bins=100, log=True, density=True)
       plt.xlabel("$\log_{10}(\lambda_i)$")
       plt.ylabel(" ESD $\log_{10}$ scale")
       plt.savefig("{}_esd.jpg".format(name))
       plt.clf()

    @torch.no_grad()
    def plot_initial_final_esd(self):
        epochs = list(self.epoch_weight_esd.keys())
        initial_epoch = epochs[0]
        final_epoch = epochs[-1]
        for idx in self.epoch_weight_esd[0]:
            initial_S_WtW = self.epoch_weight_esd[initial_epoch][idx]
            final_S_WtW = self.epoch_weight_esd[final_epoch][idx]
            initial_vals = np.log10(initial_S_WtW.cpu().numpy())
            final_vals = np.log10(final_S_WtW.cpu().numpy())
            plt.hist(initial_vals, bins=100, log=True, density=True, color="red", alpha=0.5, edgecolor='red', label="initial")
            plt.hist(final_vals, bins=100, log=True, density=True, color="blue", alpha=0.5, edgecolor='blue', label="epoch{}".format(final_epoch))
            plt.xlabel("$\log_{10}(\lambda_i)$")
            plt.ylabel("$\log_{10}(ESD)$")
            plt.legend()
            name="{}W{}".format(self.context["vis_dir"], idx)
            plt.savefig("{}_initial_final_esd.jpg".format(name))
            plt.clf()

