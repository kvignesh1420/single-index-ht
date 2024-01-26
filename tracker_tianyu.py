import logging
logger = logging.getLogger(__name__)
from collections import OrderedDict
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
plt.rcParams['axes.labelsize'] = 16
import seaborn as sns
#cmap = sns.color_palette("dark:salmon_r", as_cmap=True)
cmap = get_cmap('viridis')

class Tracker:
    def __init__(self, context):
        self.context = context
        self.epoch_weight_esd = OrderedDict()
        self.epoch_weight_vals = OrderedDict()
        self.epoch_activation_esd = OrderedDict()
        self.epoch_activation_vals = OrderedDict()
        self.epoch_KTA = OrderedDict()
        self.epoch_W_beta_alignment = OrderedDict()
        self.iteration_weight_esd = OrderedDict()
        self.iteration_weight_vals = OrderedDict()
        self.iteration_activation_esd = OrderedDict()
        self.iteration_activation_vals = OrderedDict()
        self.iteration_KTA = OrderedDict()
        self.iteration_W_beta_alignment = OrderedDict()
        self.training_loss = OrderedDict()
        self.epoch_spectral_norm=OrderedDict()
        self.epoch_frobenuis_norm=OrderedDict()
        self.epoch_stable_rank=OrderedDict()

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
    def plot_losses(self):
        # combined plots for easy viz
        epochs = list(self.val_loss.keys())
        training_losses = list(self.training_loss.values())
        val_losses = list(self.val_loss.values())
        #for i in range(len(training_losses)):
            #training_losses[i] = training_losses[i].cpu().detach().numpy()
        #for i in range(len( val_losses)):   
           # val_losses[i] = val_losses[i].cpu().detach().numpy()

        #training_losses=np.array(training_losses )
        #val_losses=np.array(val_losses)
        #if self.context['device'].type =='cuda':
             #training_losses= training_losses.cpu().numpy()
             #val_losses= val_losses.cpu().numpy()
        plt.plot(epochs, training_losses, label="train")
        plt.plot(epochs, val_losses, label="test", linestyle='dashed')
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.grid(True)
        plt.legend()
        plt.savefig("{}losses.jpg".format(self.context["vis_dir"]))
        plt.clf()

    @torch.no_grad()
    def probe_weights(self, teacher, student, epoch):
        # W and beta alignment(applicable only for first hidden layer)
        W_beta_alignment = self.compute_W_beta_alignment(teacher=teacher, student=student)
        if epoch not in self.epoch_W_beta_alignment:
            self.epoch_W_beta_alignment[epoch] = OrderedDict()
        self.epoch_W_beta_alignment[epoch][0] = W_beta_alignment
        logger.info("W_beta_alignment for epoch:{} layer: 0 = {}".format(epoch, W_beta_alignment))

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
            eigs, _ = torch.sort(S_WtW, descending=False)
            spectral_norm = eigs[-1].item()
            fnorm = torch.sum(eigs).item()
            stable_rank=fnorm/spectral_norm 
            nz_S_WtW = S_WtW[S_WtW > 0.00001]#set EVALS_THRESH=0.00001 
            N = len(nz_S_WtW)
            # somethines N may equal 0, if that happens, we don't filter eigs
            if N == 0:
                #print(f"{name} No non-zero eigs, use original total eigs")
                nz_S_WtW= S_WtW
                N = len(nz_S_WtW)
            #name_2="{}W{}_epoch{}\n sn{}_fn{} \n sr{}".format(self.context["vis_dir"], idx, epoch,spectral_norm,fnorm,stable_rank)
            self.plot_weight_esd(S_MtM=nz_S_WtW, name=name,spectral_norm=spectral_norm,fnorm=fnorm,stable_rank=stable_rank)
            if epoch not in self.epoch_weight_esd:
                self.epoch_weight_esd[epoch] = OrderedDict()
            self.epoch_weight_esd[epoch][idx] = nz_S_WtW
            if epoch not in self.epoch_spectral_norm:
                self.epoch_spectral_norm[epoch] = OrderedDict()
            self.epoch_spectral_norm[epoch][idx] = spectral_norm
            if epoch not in self.epoch_frobenuis_norm:
                self.epoch_frobenuis_norm[epoch] = OrderedDict()
            self.epoch_frobenuis_norm[epoch][idx] =fnorm
            if epoch not in self.epoch_stable_rank:
                self.epoch_stable_rank[epoch] = OrderedDict()
            self.epoch_stable_rank[epoch][idx] = stable_rank
    
    @torch.no_grad()
    def probe_iteration_weights(self, teacher, student, iteration):
        # W and beta alignment(applicable only for first hidden layer)
        W_beta_alignment = self.compute_W_beta_alignment(teacher=teacher, student=student)
        if iteration not in self.iteration_W_beta_alignment:
            self.iteration_W_beta_alignment[iteration] = OrderedDict()
        self.iteration_W_beta_alignment[iteration][0] = W_beta_alignment
        logger.info("W_beta_alignment for iteration:{} layer: 0 = {}".format(iteration, W_beta_alignment))
        
        for idx, layer in enumerate(student.layers):
            W = layer.weight.data.clone()
            name="{}W{}_iteration{}".format(self.context["vis_dir"], idx, iteration)
            self.plot_tensor(M=W, name=name)
            if iteration not in self.iteration_weight_vals:
                self.iteration_weight_vals[iteration] = OrderedDict()
            self.iteration_weight_vals[iteration][idx] = W
            S_W = torch.linalg.svdvals(W)
            self.plot_svd(S_M=S_W, name=name)
            WtW = W.t() @ W
            S_WtW = torch.linalg.svdvals(WtW)
            
            nz_S_WtW = S_WtW[S_WtW > 0.00001]#set EVALS_THRESH=0.00001 
            N = len(nz_S_WtW)
            # somethines N may equal 0, if that happens, we don't filter eigs
            if N == 0:
                #print(f"{name} No non-zero eigs, use original total eigs")
                nz_S_WtW= S_WtW
                N = len(nz_S_WtW)
            self.plot_esd(S_MtM=nz_S_WtW , name=name)

            if iteration not in self.iteration_weight_esd:
                self.iteration_weight_esd[iteration] = OrderedDict()
            self.iteration_weight_esd[iteration][idx] = nz_S_WtW

    @torch.no_grad()
    def plot_tensor(self, M, name):
        M = torch.flatten(M)
        if self.context['device'].type =='cuda':
            M=M.cpu()
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
    def plot_weight_esd(self, S_MtM, name,spectral_norm,fnorm,stable_rank):
       vals = np.log10(S_MtM.cpu().numpy())
       plt.hist(vals, bins=100, log=True, density=True)
       plt.xlabel("$\log_{10}(\lambda_i)$")
       plt.ylabel(" ESD $\log_{10}$ scale")
       plt.title(f'sn{spectral_norm}_fn{fnorm} \n sr{stable_rank}')
       plt.savefig("{}_esd.jpg".format(name))
       plt.clf()
    
    def plot_esd(self, S_MtM, name):
       vals = np.log10(S_MtM.cpu().numpy())
       plt.hist(vals, bins=100, log=True, density=True)
       plt.xlabel("$\log_{10}(\lambda_i)$")
       plt.ylabel(" ESD $\log_{10}$ scale")
       #plt.title(f'sn{spectral_norm} \n fn{fnorm} \n sr{stable_rank}')
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
            X, y_t = X.to(self.context["device"]) , y_t.to(self.context["device"])
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
            
            nz_S_ZtZ = S_ZtZ[S_ZtZ > 0.00001]#set EVALS_THRESH=0.00001 
            N = len(nz_S_ZtZ)
            # somethines N may equal 0, if that happens, we don't filter eigs
            if N == 0:
                #print(f"{name} No non-zero eigs, use original total eigs")
                nz_S_ZtZ= S_ZtZ
                N = len(nz_S_ZtZ)
            self.plot_esd(S_MtM=nz_S_ZtZ, name=name)
            if epoch not in self.epoch_activation_esd:
                self.epoch_activation_esd[epoch] = OrderedDict()
            self.epoch_activation_esd[epoch][layer_idx] = nz_S_ZtZ
            # KTA
            KTA = self.compute_KTA(Z=Z, y_t=y_t)
            if epoch not in self.epoch_KTA:
                self.epoch_KTA[epoch] = OrderedDict()
            self.epoch_KTA[epoch][layer_idx] = KTA
            logger.info("KTA for epoch:{} layer: {} = {}".format(epoch, layer_idx, KTA))
    
    @torch.no_grad()
    def probe_iteration_features(self, student, probe_loader, iteration):
        for batch_idx, (X, y_t) in enumerate(probe_loader):
            X, y_t = X.to(self.context["device"]) , y_t.to(self.context["device"])
            # ensure only one-batch of data for metrics on full-dataset
            assert batch_idx == 0

        _ = student(X)
        # capture affine features of layers
        # and compute the activation features
        for layer_idx, Z in student.affine_features.items():
            Z /= np.sqrt(self.context["d"]) if layer_idx==0 else np.sqrt(self.context["h"])
            Z = student.activation_fn(Z)
            logger.info("Shape of Z: {} layer: {}".format(Z.shape, layer_idx))
            name="{}Z{}_iteration{}".format(self.context["vis_dir"], layer_idx, iteration)
            # vals
            self.plot_tensor(M=Z, name=name)
            if iteration not in self.iteration_activation_vals:
                self.iteration_activation_vals[iteration] = OrderedDict()
            self.iteration_activation_vals[iteration][layer_idx] = Z
            # ESD
            S_Z = torch.linalg.svdvals(Z)
            self.plot_svd(S_M=S_Z, name=name)
            ZtZ = Z.t() @ Z
            S_ZtZ = torch.linalg.svdvals(ZtZ)
            
            nz_S_ZtZ = S_ZtZ[S_ZtZ > 0.00001]#set EVALS_THRESH=0.00001 
            N = len(nz_S_ZtZ)
            # somethines N may equal 0, if that happens, we don't filter eigs
            if N == 0:
                #print(f"{name} No non-zero eigs, use original total eigs")
                nz_S_ZtZ= S_ZtZ
                N = len(nz_S_ZtZ)
            self.plot_esd(S_MtM=nz_S_ZtZ, name=name)
            if iteration not in self.iteration_activation_esd:
                self.iteration_activation_esd[iteration] = OrderedDict()
            self.iteration_activation_esd[iteration][layer_idx] = nz_S_ZtZ
            # KTA
            KTA = self.compute_KTA(Z=Z, y_t=y_t)
            if iteration not in self.iteration_KTA:
                self.iteration_KTA[iteration] = OrderedDict()
            self.iteration_KTA[iteration][layer_idx] = KTA
            logger.info("KTA for iteration:{} layer: {} = {}".format(iteration, layer_idx, KTA))

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
            vals = [self.epoch_KTA[e][layer_idx].item() for e in epochs]
            plt.plot(epochs, vals, label="layer:{}".format(layer_idx))
        plt.xlabel("epochs")
        plt.ylabel("KTA")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        name="{}KTA.jpg".format(self.context["vis_dir"])
        plt.savefig(name)
        plt.clf()

    @torch.no_grad()
    def compute_W_beta_alignment(self, teacher, student):
        W = student.layers[0].weight.data.clone()
        beta = teacher.beta.squeeze().to(self.context["device"])
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        # logger.info(U.shape, Vh.shape, beta.shape)
        sim = torch.dot(Vh[0], beta)
        sim /= torch.norm(Vh[0], p=2)
        return abs(sim)

    @torch.no_grad()
    def plot_epoch_W_beta_alignment(self):
        epochs = list(self.epoch_W_beta_alignment.keys())
        layer_idxs = list(self.epoch_W_beta_alignment[epochs[0]].keys())
        for layer_idx in layer_idxs:
            vals = [self.epoch_W_beta_alignment[e][layer_idx].item() for e in epochs]
            plt.plot(epochs, vals, label="layer:{}".format(layer_idx))
        plt.xlabel("epochs")
        plt.ylabel("$sim(W, \\beta^*)$")
        plt.legend()
        plt.grid(True)
        name="{}W_beta_alignment.jpg".format(self.context["vis_dir"])
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

    @torch.no_grad()
    def plot_epoch_weight_vals(self):
            epochs = list(self.epoch_weight_vals.keys())
            for idx in self.epoch_weight_vals[0]:# layer-wise
                for epoch in epochs:
                    W = self.epoch_weight_vals[epoch][idx]
                    #inal_W = self.epoch_weight_vals[final_epoch][idx]
                    vals = torch.flatten(W).cpu().numpy()
                    #final_vals = torch.flatten(final_W).cpu().numpy()
                    plt.hist(vals, bins=100, density=True, color=cmap(epoch/len(epochs)), alpha=0.5, edgecolor=cmap(epoch/len(epochs)), label="epoch{}".format(epoch))
                    #plt.hist(final_vals, bins=100, density=True, color="violet", alpha=0.5, edgecolor='blue', label="epoch{}".format(final_epoch))
                    plt.xlabel("vals")
                    plt.ylabel("density(vals)")
                    plt.legend()
                name="{}W{}".format(self.context["vis_dir"], idx)
                plt.savefig("{}_epoch_weight_vals.jpg".format(name))
                plt.clf()
    
    @torch.no_grad()
    def plot_epoch_weight_esd(self):
        epochs = list(self.epoch_weight_esd.keys())
        
        for idx in self.epoch_weight_esd[0]:
            for epoch in epochs:
                S_WtW = self.epoch_weight_esd[epoch][idx]
                #final_S_WtW = self.epoch_weight_esd[final_epoch][idx]
                vals = np.log10(S_WtW.cpu().numpy())
                #final_vals = np.log10(final_S_WtW.cpu().numpy())
                plt.hist(vals, bins=100, log=True, density=True, color=cmap(epoch/len(epochs)), alpha=0.5, edgecolor=cmap(epoch/len(epochs)), label="epoch{}".format(epoch))
                #plt.hist(final_vals, bins=100, log=True, density=True, color="blue", alpha=0.5, edgecolor='blue', label="epoch{}".format(final_epoch))
                plt.xlabel("$\log_{10}(\lambda_i)$")
                plt.ylabel("$\log_{10}(ESD)$")
                plt.legend()
            name="{}W{}".format(self.context["vis_dir"], idx)
            plt.savefig("{}_epoch_weight_esd.jpg".format(name))
            plt.clf()
    
    @torch.no_grad()
    def plot_epoch_activation_vals(self):
        epochs = list(self.epoch_activation_vals.keys())
        
        for idx in self.epoch_activation_vals[0]:
            for epoch in epochs:
                Z = self.epoch_activation_vals[epoch][idx]
                #final_Z = self.epoch_activation_vals[final_epoch][idx]
                vals = torch.flatten(Z).cpu().numpy()
                #final_vals = torch.flatten(final_Z).cpu().numpy()
                plt.hist(vals, bins=100, density=True, color=cmap(epoch/len(epochs)), alpha=0.5, edgecolor=cmap(epoch/len(epochs)), label="epoch{}".format(epoch))
                #plt.hist(final_vals, bins=100, density=True, color="violet", alpha=0.5, edgecolor='blue', label="epoch{}".format(final_epoch))
                plt.xlabel("vals")
                plt.ylabel("density(vals)")
                plt.legend()
            name="{}Z{}".format(self.context["vis_dir"], idx)
            plt.savefig("{}_epoch_activation_vals.jpg".format(name))
            plt.clf()

    @torch.no_grad()
    def plot_epoch_activation_esd(self):
        epochs = list(self.epoch_activation_esd.keys())
        
        for idx in self.epoch_activation_esd[0]:
            for epoch in epochs:
                S_ZtZ = self.epoch_activation_esd[epoch][idx]
                #final_S_ZtZ = self.epoch_activation_esd[final_epoch][idx]
                initial_vals = np.log10(S_ZtZ.cpu().numpy())
                #final_vals = np.log10(final_S_ZtZ.cpu().numpy())
                plt.hist(initial_vals, bins=100, log=True, density=True, color=cmap(epoch/len(epochs)), alpha=0.5, edgecolor=cmap(epoch/len(epochs)), label="epoch{}".format(epoch))
                #plt.hist(final_vals, bins=100, log=True, density=True, color="blue", alpha=0.5, edgecolor='blue', label="epoch{}".format(final_epoch))
                plt.xlabel("$\log_{10}(\lambda_i)$")
                plt.ylabel("$\log_{10}(ESD)$")
                plt.legend()
            name="{}Z{}".format(self.context["vis_dir"], idx)
            plt.savefig("{}_epoch_activation_esd.jpg".format(name))
            plt.clf()
    
    @torch.no_grad()
    def plot_iteration_weight_vals(self):
            iterations = list(self.iteration_weight_vals.keys())
            for idx in self.iteration_weight_vals[0]:# layer-wise
                for iteration in iterations:
                    W = self.iteration_weight_vals[iteration][idx]
                    #inal_W = self.iteration_weight_vals[final_iteration][idx]
                    vals = torch.flatten(W).cpu().numpy()
                    #final_vals = torch.flatten(final_W).cpu().numpy()
                    plt.hist(vals, bins=100, density=True, color=cmap(iteration/len(iterations)), alpha=0.5, edgecolor=cmap(iteration/len(iterations)), label="iteration{}".format(iteration))
                    #plt.hist(final_vals, bins=100, density=True, color="violet", alpha=0.5, edgecolor='blue', label="iteration{}".format(final_iteration))
                    plt.xlabel("vals")
                    plt.ylabel("density(vals)")
                    plt.legend()
                name="{}W{}".format(self.context["vis_dir"], idx)
                plt.savefig("{}_iteration_weight_vals.jpg".format(name))
                plt.clf()
    
    @torch.no_grad()
    def plot_iteration_weight_esd(self):
        iterations = list(self.iteration_weight_esd.keys())
        
        for idx in self.iteration_weight_esd[0]:
            for iteration in iterations:
                S_WtW = self.iteration_weight_esd[iteration][idx]
                #final_S_WtW = self.iteration_weight_esd[final_iteration][idx]
                vals = np.log10(S_WtW.cpu().numpy())
                #final_vals = np.log10(final_S_WtW.cpu().numpy())
                plt.hist(vals, bins=100, log=True, density=True, color=cmap(iteration/len(iterations)), alpha=0.5, edgecolor=cmap(iteration/len(iterations)), label="iteration{}".format(iteration))
                #plt.hist(final_vals, bins=100, log=True, density=True, color="blue", alpha=0.5, edgecolor='blue', label="iteration{}".format(final_iteration))
                plt.xlabel("$\log_{10}(\lambda_i)$")
                plt.ylabel("$\log_{10}(ESD)$")
                plt.legend()
            name="{}W{}".format(self.context["vis_dir"], idx)
            plt.savefig("{}_iteration_weight_esd.jpg".format(name))
            plt.clf()
    
    @torch.no_grad()
    def plot_iteration_activation_vals(self):
        iterations = list(self.iteration_activation_vals.keys())
        
        for idx in self.iteration_activation_vals[0]:
            for iteration in iterations:
                Z = self.iteration_activation_vals[iteration][idx]

                vals = torch.flatten(Z).cpu().numpy()
                #final_vals = torch.flatten(final_Z).cpu().numpy()
                plt.hist(vals, bins=100, density=True, color=cmap(iteration/len(iterations)), alpha=0.5, edgecolor=cmap(iteration/len(iterations)), label="iteration{}".format(iteration))
                #plt.hist(final_vals, bins=100, density=True, color="violet", alpha=0.5, edgecolor='blue', label="iteration{}".format(final_iteration))
                plt.xlabel("vals")
                plt.ylabel("density(vals)")
                plt.legend()
            name="{}Z{}".format(self.context["vis_dir"], idx)
            plt.savefig("{}_iteration_activation_vals.jpg".format(name))
            plt.clf()

    @torch.no_grad()
    def plot_iteration_activation_esd(self):
        iterations = list(self.iteration_activation_esd.keys())
        
        for idx in self.iteration_activation_esd[0]:
            for iteration in iterations:
                S_ZtZ = self.iteration_activation_esd[iteration][idx]
                initial_vals = np.log10(S_ZtZ.cpu().numpy())
                plt.hist(initial_vals, bins=100, log=True, density=True, color=cmap(iteration/len(iterations)), alpha=0.5, edgecolor=cmap(iteration/len(iterations)), label="iteration{}".format(iteration))
                plt.xlabel("$\log_{10}(\lambda_i)$")
                plt.ylabel("$\log_{10}(ESD)$")
                plt.legend()
            name="{}Z{}".format(self.context["vis_dir"], idx)
            plt.savefig("{}_iteration_activation_esd.jpg".format(name))
            plt.clf()

