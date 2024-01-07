import logging
logger = logging.getLogger(__name__)
from collections import OrderedDict
import torch
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 16

class Tracker:
    def __init__(self, context):
        self.context = context
        self.epoch_weight_esd = OrderedDict()
        self.epoch_weight_vals = OrderedDict()
        self.epoch_activation_esd = OrderedDict()
        self.epoch_activation_vals = OrderedDict()
        self.epoch_KTA = OrderedDict()
        self.training_loss = OrderedDict()
        self.val_loss = OrderedDict()

    @torch.no_grad()
    def store_training_loss(self, loss, epoch):
        self.training_loss[epoch] = loss

    @torch.no_grad()
    def plot_training_loss(self):
        epochs = list(self.training_loss.keys())
        losses = list(self.training_loss.values())
        plt.plot(epochs, losses)
        plt.grid(True)
        plt.savefig("{}training_loss.jpg".format(self.context["vis_dir"]))
        plt.clf()

    @torch.no_grad()
    def store_val_loss(self, loss, epoch):
        self.val_loss[epoch] = loss

    @torch.no_grad()
    def plot_val_loss(self):
        epochs = list(self.val_loss.keys())
        losses = list(self.val_loss.values())
        plt.plot(epochs, losses)
        plt.grid(True)
        plt.savefig("{}val_loss.jpg".format(self.context["vis_dir"]))
        plt.clf()

    @torch.no_grad()
    def probe_weights(self, student, epoch):
        for idx, layer in enumerate(student.layers):
            W = layer.weight.data.clone()
            name="{}W{}_epoch{}".format(self.context["vis_dir"], idx, epoch)
            self.plot_tensor(M=W, name=name)
            if epoch not in self.epoch_weight_vals:
                self.epoch_weight_vals[epoch] = OrderedDict()
            self.epoch_weight_vals[epoch][idx] = W
            S_W = torch.linalg.svdvals(W)
            self.plot_svd(S_M=S_W, name=name)
            WtW = W.t() @ W
            S_WtW = torch.linalg.svdvals(WtW)
            self.plot_esd(S_MtM=S_WtW, name=name)
            if epoch not in self.epoch_weight_esd:
                self.epoch_weight_esd[epoch] = OrderedDict()
            self.epoch_weight_esd[epoch][idx] = S_WtW

    @torch.no_grad()
    def plot_tensor(self, M, name):
        M = torch.flatten(M)
        plt.hist(M, bins=100, density=True)
        plt.xlabel("val")
        plt.ylabel("density(val)")
        plt.savefig("{}_vals.jpg".format(name))
        plt.clf()

    @torch.no_grad()
    def plot_initial_final_weight_vals(self):
        epochs = list(self.epoch_weight_vals.keys())
        initial_epoch = epochs[0]
        final_epoch = epochs[-1]
        for idx in self.epoch_weight_vals[0]:
            initial_W = self.epoch_weight_vals[initial_epoch][idx]
            final_W = self.epoch_weight_vals[final_epoch][idx]
            initial_vals = torch.flatten(initial_W).cpu().numpy()
            final_vals = torch.flatten(final_W).cpu().numpy()
            plt.hist(initial_vals, bins=100, density=True, color="orange", alpha=0.5, edgecolor='red', label="initial")
            plt.hist(final_vals, bins=100, density=True, color="violet", alpha=0.5, edgecolor='blue', label="epoch{}".format(final_epoch))
            plt.xlabel("vals")
            plt.ylabel("density(vals)")
            plt.legend()
            name="{}W{}".format(self.context["vis_dir"], idx)
            plt.savefig("{}_initial_final_vals.jpg".format(name))
            plt.clf()

    @torch.no_grad()
    def plot_svd(self, S_M, name):
       plt.bar(x=list(range(S_M.shape[0])), height=S_M.cpu().numpy())
       plt.xlabel("$i$")
       plt.ylabel("$\lambda_i$")
       plt.savefig("{}_sv.jpg".format(name))
       plt.clf()

    @torch.no_grad()
    def plot_esd(self, S_MtM, name):
       vals = np.log10(S_MtM.cpu().numpy())
       plt.hist(vals, bins=100, log=True, density=True)
       plt.xlabel("$\log_{10}(\lambda_i)$")
       plt.ylabel(" ESD $\log_{10}$ scale")
       plt.savefig("{}_esd.jpg".format(name))
       plt.clf()

    @torch.no_grad()
    def plot_initial_final_weight_esd(self):
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

    @torch.no_grad()
    def probe_features(self, student, probe_loader, epoch):
        for batch_idx, (X, y_t) in enumerate(probe_loader):
            # ensure only one-batch of data for metrics on full-dataset
            assert batch_idx == 0
        _ = student(X)
        # capture affine features of layers
        # and compute the activation features
        for layer_idx, Z in student.affine_features.items():
            Z /= np.sqrt(self.context["d"]) if layer_idx==0 else np.sqrt(self.context["h"])
            Z = student.activation_fn(Z)
            logger.info("Shape of Z: {} layer: {}".format(Z.shape, layer_idx))
            name="{}Z{}_epoch{}".format(self.context["vis_dir"], layer_idx, epoch)
            # vals
            self.plot_tensor(M=Z, name=name)
            if epoch not in self.epoch_activation_vals:
                self.epoch_activation_vals[epoch] = OrderedDict()
            self.epoch_activation_vals[epoch][layer_idx] = Z
            # ESD
            S_Z = torch.linalg.svdvals(Z)
            self.plot_svd(S_M=S_Z, name=name)
            ZtZ = Z.t() @ Z
            S_ZtZ = torch.linalg.svdvals(ZtZ)
            self.plot_esd(S_MtM=S_ZtZ, name=name)
            if epoch not in self.epoch_activation_esd:
                self.epoch_activation_esd[epoch] = OrderedDict()
            self.epoch_activation_esd[epoch][layer_idx] = S_ZtZ
            # KTA
            KTA = self.compute_KTA(Z=Z, y_t=y_t)
            if epoch not in self.epoch_KTA:
                self.epoch_KTA[epoch] = OrderedDict()
            self.epoch_KTA[epoch][layer_idx] = KTA
            logger.info("KTA for epoch:{} layer: {} = {}".format(epoch, layer_idx, KTA))


    @torch.no_grad()
    def compute_KTA(self, Z, y_t):
        # Kernel target alignment
        # Z has shape: (n, h)
        K = Z @ Z.t()
        K_y = y_t @ y_t.t()
        logger.info("K.shape {} K_y.shape:{}".format(K.shape, K_y.shape))
        KTA = torch.sum(K * K_y)/( torch.norm(K, p="fro") * torch.norm(K_y, p="fro") )
        return KTA

    @torch.no_grad()
    def plot_epoch_KTA(self):
        epochs = list(self.epoch_KTA.keys())
        layer_idxs = list(self.epoch_KTA[epochs[0]].keys())
        for layer_idx in layer_idxs:
            vals = [self.epoch_KTA[e][layer_idx] for e in epochs]
            plt.plot(epochs, vals, label="layer:{}".format(layer_idx))
        plt.xlabel("epochs")
        plt.ylabel("KTA")
        plt.legend()
        name="{}KTA.jpg".format(self.context["vis_dir"])
        plt.savefig(name)
        plt.clf()

    @torch.no_grad()
    def plot_initial_final_activation_vals(self):
        epochs = list(self.epoch_activation_vals.keys())
        initial_epoch = epochs[0]
        final_epoch = epochs[-1]
        for idx in self.epoch_activation_vals[0]:
            initial_Z = self.epoch_activation_vals[initial_epoch][idx]
            final_Z = self.epoch_activation_vals[final_epoch][idx]
            initial_vals = torch.flatten(initial_Z).cpu().numpy()
            final_vals = torch.flatten(final_Z).cpu().numpy()
            plt.hist(initial_vals, bins=100, density=True, color="orange", alpha=0.5, edgecolor='red', label="initial")
            plt.hist(final_vals, bins=100, density=True, color="violet", alpha=0.5, edgecolor='blue', label="epoch{}".format(final_epoch))
            plt.xlabel("vals")
            plt.ylabel("density(vals)")
            plt.legend()
            name="{}Z{}".format(self.context["vis_dir"], idx)
            plt.savefig("{}_initial_final_vals.jpg".format(name))
            plt.clf()

    @torch.no_grad()
    def plot_initial_final_activation_esd(self):
        epochs = list(self.epoch_activation_esd.keys())
        initial_epoch = epochs[0]
        final_epoch = epochs[-1]
        for idx in self.epoch_activation_esd[0]:
            initial_S_ZtZ = self.epoch_activation_esd[initial_epoch][idx]
            final_S_ZtZ = self.epoch_activation_esd[final_epoch][idx]
            initial_vals = np.log10(initial_S_ZtZ.cpu().numpy())
            final_vals = np.log10(final_S_ZtZ.cpu().numpy())
            plt.hist(initial_vals, bins=100, log=True, density=True, color="red", alpha=0.5, edgecolor='red', label="initial")
            plt.hist(final_vals, bins=100, log=True, density=True, color="blue", alpha=0.5, edgecolor='blue', label="epoch{}".format(final_epoch))
            plt.xlabel("$\log_{10}(\lambda_i)$")
            plt.ylabel("$\log_{10}(ESD)$")
            plt.legend()
            name="{}Z{}".format(self.context["vis_dir"], idx)
            plt.savefig("{}_initial_final_esd.jpg".format(name))
            plt.clf()
