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
        self.step_weight_stable_rank = OrderedDict()
        self.step_weight_esd = OrderedDict()
        self.step_weight_vals = OrderedDict()
        self.step_activation_esd = OrderedDict()
        self.step_activation_vals = OrderedDict()
        self.step_activation_stable_rank = OrderedDict()
        self.step_activation_effective_ranks = OrderedDict()
        self.step_KTA = OrderedDict()
        self.step_W_beta_alignment = OrderedDict()
        self.step_weighted_U_pc_align_stddev = OrderedDict()
        self.step_weighted_Vh_pc_align_stddev = OrderedDict()
        self.step_U_pc_align_stddev = OrderedDict()
        self.step_Vh_pc_align_stddev = OrderedDict()
        self.training_loss = OrderedDict()
        self.val_loss = OrderedDict()

    @torch.no_grad()
    def store_training_loss(self, loss, step):
        self.training_loss[step] = loss

    @torch.no_grad()
    def plot_training_loss(self):
        steps = list(self.training_loss.keys())
        losses = list(self.training_loss.values())
        plt.plot(steps, losses)
        plt.grid(True)
        plt.savefig("{}training_loss.jpg".format(self.context["vis_dir"]))
        plt.clf()

    @torch.no_grad()
    def store_val_loss(self, loss, step):
        self.val_loss[step] = loss

    @torch.no_grad()
    def plot_val_loss(self):
        steps = list(self.val_loss.keys())
        losses = list(self.val_loss.values())
        plt.plot(steps, losses)
        plt.grid(True)
        plt.savefig("{}val_loss.jpg".format(self.context["vis_dir"]))
        plt.clf()

    @torch.no_grad()
    def plot_losses(self):
        # combined plots for easy viz
        steps = list(self.val_loss.keys())
        training_losses = list(self.training_loss.values())
        val_losses = list(self.val_loss.values())
        plt.plot(steps, training_losses, label="train")
        plt.plot(steps, val_losses, label="test", linestyle='dashed')
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.grid(True)
        plt.legend()
        plt.savefig("{}losses.jpg".format(self.context["vis_dir"]))
        plt.clf()

    @torch.no_grad()
    def probe_weights(self, teacher, student, step):
        # W and beta alignment(applicable only for first hidden layer)
        W_beta_alignment = self.compute_W_beta_alignment(teacher=teacher, student=student)
        if step not in self.step_W_beta_alignment:
            self.step_W_beta_alignment[step] = OrderedDict()
        self.step_W_beta_alignment[step][0] = W_beta_alignment
        logger.info("W_beta_alignment for step:{} layer: 0 = {}".format(step, W_beta_alignment))

        for idx, layer in enumerate(student.layers):
            W = layer.weight.data.clone()
            name="{}W{}_step{}".format(self.context["vis_dir"], idx, step)
            self.plot_tensor(M=W, name=name)
            if step not in self.step_weight_vals:
                self.step_weight_vals[step] = OrderedDict()
            self.step_weight_vals[step][idx] = W

            S_W = torch.linalg.svdvals(W)
            self.plot_svd(S_M=S_W, name=name)
            WtW = W.t() @ W
            S_WtW = torch.linalg.svdvals(WtW)
            self.plot_esd(S_MtM=S_WtW, name=name)
            if step not in self.step_weight_esd:
                self.step_weight_esd[step] = OrderedDict()
            self.step_weight_esd[step][idx] = S_WtW

            W_stable_rank = self.get_stable_rank(M=W)
            if step not in self.step_weight_stable_rank:
                self.step_weight_stable_rank[step] = OrderedDict()
            self.step_weight_stable_rank[step][idx] = W_stable_rank

    @torch.no_grad()
    def get_stable_rank(self, M):
        norm_frob = torch.linalg.norm(M, ord="fro")
        norm_2 = torch.linalg.norm(M, ord=2)
        stable_rank = (norm_frob/norm_2)**2
        return stable_rank

    @torch.no_grad()
    def get_effective_ranks(self, M):
        S = torch.linalg.svdvals(M)
        total_svs = S.shape[0]
        # effective rank variant 1
        er_r = []
        for k in range(total_svs-1):
            if S[k+1] <= 0:
                val = 0
            else:
                temp_sum = 0
                for i in range(k+1, total_svs-1):
                    temp_sum += S[i]
                val = temp_sum/S[k+1]
            er_r.append(val)
        # effective rank variant 2
        er_R = []
        for k in range(total_svs-1):
            temp_sum1 = 0
            temp_sum2 = 0
            for i in range(k+1, total_svs-1):
                temp_sum1 += S[i]
                temp_sum2 += (S[i])**2
            if temp_sum2 > 0:
                val = temp_sum1**2/temp_sum2
            else: val = 0
            er_R.append(val)
        return er_r, er_R

    @torch.no_grad()
    def plot_step_weight_stable_rank(self):
        steps = list(self.step_weight_stable_rank.keys())
        layer_idxs = list(self.step_weight_stable_rank[steps[0]].keys())
        for layer_idx in layer_idxs:
            vals = [self.step_weight_stable_rank[e][layer_idx] for e in steps]
            plt.plot(steps, vals, label="layer:{}".format(layer_idx))
            plt.xlabel("steps")
            plt.ylabel("stable rank")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            name="{}W{}_stable_rank.jpg".format(self.context["vis_dir"], layer_idx)
            plt.savefig(name)
            plt.clf()

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
        steps = list(self.step_weight_vals.keys())
        initial_step = steps[0]
        final_step = steps[-1]
        for idx in self.step_weight_vals[0]:
            initial_W = self.step_weight_vals[initial_step][idx]
            final_W = self.step_weight_vals[final_step][idx]
            initial_vals = torch.flatten(initial_W).cpu().numpy()
            final_vals = torch.flatten(final_W).cpu().numpy()
            plt.hist(initial_vals, bins=100, density=True, color="orange", alpha=0.5, edgecolor='red', label="initial")
            plt.hist(final_vals, bins=100, density=True, color="violet", alpha=0.5, edgecolor='blue', label="step{}".format(final_step))
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
        steps = list(self.step_weight_esd.keys())
        initial_step = steps[0]
        final_step = steps[-1]
        for idx in self.step_weight_esd[0]:
            initial_S_WtW = self.step_weight_esd[initial_step][idx]
            final_S_WtW = self.step_weight_esd[final_step][idx]
            initial_vals = np.log10(initial_S_WtW.cpu().numpy())
            final_vals = np.log10(final_S_WtW.cpu().numpy())
            plt.hist(initial_vals, bins=100, log=True, density=True, color="red", alpha=0.5, edgecolor='red', label="initial")
            plt.hist(final_vals, bins=100, log=True, density=True, color="blue", alpha=0.5, edgecolor='blue', label="step{}".format(final_step))
            plt.xlabel("$\log_{10}(\lambda_i)$")
            plt.ylabel("$\log_{10}(ESD)$")
            plt.legend()
            name="{}W{}".format(self.context["vis_dir"], idx)
            plt.savefig("{}_initial_final_esd.jpg".format(name))
            plt.clf()

    @torch.no_grad()
    def probe_features(self, student, probe_loader, step):
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
            name="{}Z{}_step{}".format(self.context["vis_dir"], layer_idx, step)
            # vals
            self.plot_tensor(M=Z, name=name)
            if step not in self.step_activation_vals:
                self.step_activation_vals[step] = OrderedDict()
            self.step_activation_vals[step][layer_idx] = Z
            # ESD
            S_Z = torch.linalg.svdvals(Z)
            self.plot_svd(S_M=S_Z, name=name)
            ZtZ = Z.t() @ Z
            S_ZtZ = torch.linalg.svdvals(ZtZ)
            self.plot_esd(S_MtM=S_ZtZ, name=name)
            if step not in self.step_activation_esd:
                self.step_activation_esd[step] = OrderedDict()
            self.step_activation_esd[step][layer_idx] = S_ZtZ
            # KTA
            KTA = self.compute_KTA(Z=Z, y_t=y_t)
            if step not in self.step_KTA:
                self.step_KTA[step] = OrderedDict()
            self.step_KTA[step][layer_idx] = KTA
            logger.info("KTA for step:{} layer: {} = {}".format(step, layer_idx, KTA))
            # stable rank
            Z_stable_rank = self.get_stable_rank(M=Z)
            if step not in self.step_activation_stable_rank:
                self.step_activation_stable_rank[step] = OrderedDict()
            self.step_activation_stable_rank[step][layer_idx] = Z_stable_rank
            # effective ranks
            K = Z @ Z.t()
            Z_effective_ranks = self.get_effective_ranks(M=K)
            if step not in self.step_activation_effective_ranks:
                self.step_activation_effective_ranks[step] = OrderedDict()
            self.step_activation_effective_ranks[step][layer_idx] = {
                "variant1": Z_effective_ranks[0],
                "variant2": Z_effective_ranks[1]
            }

    @torch.no_grad()
    def plot_step_activation_stable_rank(self):
        steps = list(self.step_activation_stable_rank.keys())
        layer_idxs = list(self.step_activation_stable_rank[steps[0]].keys())
        for layer_idx in layer_idxs:
            vals = [self.step_activation_stable_rank[e][layer_idx] for e in steps]
            plt.plot(steps, vals, label="layer:{}".format(layer_idx))
            plt.xlabel("steps")
            plt.ylabel("stable rank")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            name="{}Z{}_stable_rank.jpg".format(self.context["vis_dir"], layer_idx)
            plt.savefig(name)
            plt.clf()

    @torch.no_grad()
    def plot_step_activation_effective_ranks(self):
        steps = list(self.step_activation_effective_ranks.keys())
        layer_idxs = list(self.step_activation_effective_ranks[steps[0]].keys())
        for layer_idx in layer_idxs:
            for step in steps:
                variant1_vals = self.step_activation_effective_ranks[step][layer_idx]["variant1"]
                variant2_vals = self.step_activation_effective_ranks[step][layer_idx]["variant2"]
                plt.plot(variant1_vals, label="layer:{},step:{},r".format(layer_idx, step))
                plt.plot(variant2_vals, label="layer:{},step:{},R".format(layer_idx, step))
            plt.xlabel("singular value idx")
            plt.ylabel("effective ranks")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            name="{}Z{}_effective_ranks.jpg".format(self.context["vis_dir"], layer_idx)
            plt.savefig(name)
            plt.clf()

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
    def plot_step_KTA(self):
        steps = list(self.step_KTA.keys())
        layer_idxs = list(self.step_KTA[steps[0]].keys())
        for layer_idx in layer_idxs:
            vals = [self.step_KTA[e][layer_idx] for e in steps]
            plt.plot(steps, vals, label="layer:{}".format(layer_idx))
        plt.xlabel("steps")
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
    def plot_step_W_beta_alignment(self):
        steps = list(self.step_W_beta_alignment.keys())
        layer_idxs = list(self.step_W_beta_alignment[steps[0]].keys())
        for layer_idx in layer_idxs:
            vals = [self.step_W_beta_alignment[e][layer_idx] for e in steps]
            plt.plot(steps, vals, label="layer:{}".format(layer_idx))
        plt.xlabel("steps")
        plt.ylabel("$sim(W, \\beta^*)$")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        name="{}W_beta_alignment.jpg".format(self.context["vis_dir"])
        plt.savefig(name)
        plt.clf()

    @torch.no_grad()
    def plot_initial_final_activation_vals(self):
        steps = list(self.step_activation_vals.keys())
        initial_step = steps[0]
        final_step = steps[-1]
        for idx in self.step_activation_vals[0]:
            initial_Z = self.step_activation_vals[initial_step][idx]
            final_Z = self.step_activation_vals[final_step][idx]
            initial_vals = torch.flatten(initial_Z).cpu().numpy()
            final_vals = torch.flatten(final_Z).cpu().numpy()
            plt.hist(initial_vals, bins=100, density=True, color="orange", alpha=0.5, edgecolor='red', label="initial")
            plt.hist(final_vals, bins=100, density=True, color="violet", alpha=0.5, edgecolor='blue', label="step{}".format(final_step))
            plt.xlabel("vals")
            plt.ylabel("density(vals)")
            plt.legend()
            name="{}Z{}".format(self.context["vis_dir"], idx)
            plt.savefig("{}_initial_final_vals.jpg".format(name))
            plt.clf()

    @torch.no_grad()
    def plot_initial_final_activation_esd(self):
        steps = list(self.step_activation_esd.keys())
        initial_step = steps[0]
        final_step = steps[-1]
        for idx in self.step_activation_esd[0]:
            initial_S_ZtZ = self.step_activation_esd[initial_step][idx]
            final_S_ZtZ = self.step_activation_esd[final_step][idx]
            initial_vals = np.log10(initial_S_ZtZ.cpu().numpy())
            final_vals = np.log10(final_S_ZtZ.cpu().numpy())
            plt.hist(initial_vals, bins=100, log=True, density=True, color="red", alpha=0.5, edgecolor='red', label="initial")
            plt.hist(final_vals, bins=100, log=True, density=True, color="blue", alpha=0.5, edgecolor='blue', label="step{}".format(final_step))
            plt.xlabel("$\log_{10}(\lambda_i)$")
            plt.ylabel("$\log_{10}(ESD)$")
            plt.legend()
            name="{}Z{}".format(self.context["vis_dir"], idx)
            plt.savefig("{}_initial_final_esd.jpg".format(name))
            plt.clf()

    @torch.no_grad()
    def plot_init_W_pc_and_beta_alignment(self, student, teacher):
        W = student.layers[0].weight.data.clone().to(self.context["device"])
        beta = teacher.beta.squeeze().to(self.context["device"])
        U_W, S_W, Vh_W = torch.linalg.svd(W, full_matrices=False)
        Vh_sim = torch.abs( Vh_W @ beta )

        plt.plot(Vh_sim)
        Vh_sim_max_index = np.argmax(Vh_sim)
        Vh_sim_max_value = Vh_sim[Vh_sim_max_index]
        plt.axvline(x=Vh_sim_max_index, color='r', linestyle='--', label=f'Max Value ({Vh_sim_max_value:.2f}), i={Vh_sim_max_index}', linewidth=2)
        plt.xlabel('i')
        plt.ylabel('alignment')
        plt.legend()
        plt.tight_layout()
        name="{}init_W_pc_beta_sim.jpg".format(self.context["vis_dir"])
        plt.savefig(name)
        plt.clf()

    @torch.no_grad()
    def plot_W_and_grad_alignment(self, X, y_t, student, step):
        W = student.layers[0].weight.data.clone()
        a = student.layers[1].weight.data.clone()
        # W shape: h x d
        # a shape: 1 x h
        # X shape: batch_size x d
        # y_t shape: batch_size x 1
        print("step: {} W shape: {} a shape: {} X shape: {} y_t shape:{}".format(
            step, W.shape, a.shape, X.shape, y_t.shape))
        # approx_G = (X.t() @ y_t @ a).t()
        G = student.layers[0].weight.grad.clone()
        signG = torch.sign(G)
        # if self.context["optimizer"] == "adam":
        print("shape of G: {}".format(G.shape))
        U_G, S_G, Vh_G = torch.linalg.svd(G, full_matrices=False)
        U_signG, S_signG, Vh_signG = torch.linalg.svd(signG, full_matrices=False)
        U_W, S_W, Vh_W = torch.linalg.svd(W, full_matrices=False)
        S_signG_logvals = torch.log10(S_signG).detach().cpu().numpy()
        S_G_logvals = torch.log10(S_G).detach().cpu().numpy()
        S_W_logvals = torch.log10(S_W).detach().cpu().numpy()
        plt.hist(S_G_logvals, bins=100, log=True, density=True, color="red", alpha=0.5, edgecolor='red', label="G")
        plt.hist(S_signG_logvals, bins=100, log=True, density=True, color="green", alpha=0.5, edgecolor='green', label="signG")
        plt.hist(S_W_logvals, bins=100, log=True, density=True, color="violet", alpha=0.3, edgecolor='violet', label="W")
        plt.xlabel("$\log_{10}(\lambda_i)$")
        plt.ylabel("$\log_{10}(ESD)$")
        plt.legend()
        plt.tight_layout()
        name="{}W_G_step{}".format(self.context["vis_dir"], step)
        plt.savefig("{}_esd.jpg".format(name))
        plt.clf()

        plt.plot(S_G[:10])
        plt.xlabel("i")
        plt.ylabel("$\lambda_i$")
        plt.grid()
        plt.tight_layout()
        name="{}G{}".format(self.context["vis_dir"], step)
        plt.savefig("{}_sv.jpg".format(name))
        plt.clf()

        name="{}W_G_pc_sim_step{}.jpg".format(self.context["vis_dir"], step)
        self.plot_pc_sims(U_W=U_W, U_G=U_G, Vh_W=Vh_W, Vh_G=Vh_G, name=name)

        name="{}W_signG_pc_sim_step{}.jpg".format(self.context["vis_dir"], step)
        self.plot_pc_sims(U_W=U_W, U_G=U_signG, Vh_W=Vh_W, Vh_G=Vh_signG, name=name)


    @torch.no_grad()
    def plot_pc_sims(self, U_W, U_G, Vh_W, Vh_G, name):
        U_sim = torch.abs( (U_W.t() @ U_G) )
        Vh_sim = torch.abs( (Vh_W @ Vh_G.t()) )

        U_sim = U_sim[:, 0]
        Vh_sim = Vh_sim[:, 0]

        fig, axs = plt.subplots(1, 2)
        axs[0].plot(U_sim)
        U_sim_max_index = np.argmax(U_sim)
        U_sim_max_value = U_sim[U_sim_max_index]
        axs[0].axvline(x=U_sim_max_index, color='r', linestyle='--', label=f'Max Value ({U_sim_max_value:.2f}), i={U_sim_max_index}', linewidth=2)
        axs[0].set_xlabel('i')
        axs[0].set_ylabel('alignment')
        axs[0].set_title('Right PC')
        axs[0].legend()

        axs[1].plot(Vh_sim)
        Vh_sim_max_index = np.argmax(Vh_sim)
        Vh_sim_max_value = Vh_sim[Vh_sim_max_index]
        axs[1].axvline(x=Vh_sim_max_index, color='r', linestyle='--', label=f'Max Value ({Vh_sim_max_value:.2f}), i={Vh_sim_max_index}', linewidth=2)
        axs[1].set_xlabel('i')
        axs[1].set_ylabel('alignment')
        axs[1].set_title('Left PC')
        axs[1].legend()

        plt.tight_layout()
        plt.savefig(name)
        plt.clf()
