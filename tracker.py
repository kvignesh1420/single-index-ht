import torch
import numpy as np
import matplotlib.pyplot as plt

class Tracker:
    def __init__(self, context):
        self.context = context

    @torch.no_grad()
    def probe_weights(self, student, epoch):
        for idx, layer in enumerate(student.layers):
            self.plot_svd(
                W = layer.weight.data,
                name="{}W{}_epoch{}".format(self.context["vis_dir"], idx, epoch)
            )
            self.plot_esd(
                W = layer.weight.data,
                name="{}W{}_epoch{}".format(self.context["vis_dir"], idx, epoch)
            )

    @torch.no_grad()
    def plot_svd(self, W, name):
       S = torch.linalg.svdvals(W)
       plt.bar(x=list(range(S.shape[0])), height=S.cpu().numpy())
       plt.xlabel("$i$")
       plt.ylabel("$\lambda_i$")
       plt.grid(True)
       plt.savefig("{}_sv.jpg".format(name))
       plt.clf()

    @torch.no_grad()
    def plot_esd(self, W, name):
       M = W.t() @ W
       S = torch.linalg.svdvals(M)
       vals = S.cpu().numpy()
       plt.hist(vals, bins=100, density=True)
       plt.xlabel("$\lambda_i$")
       plt.ylabel("$ESD$")
       plt.grid(True)
       plt.savefig("{}_esd.jpg".format(name))
       plt.clf()

