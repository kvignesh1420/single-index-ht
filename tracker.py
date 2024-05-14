import logging
logger = logging.getLogger(__name__)
from collections import OrderedDict
from tqdm import tqdm
import os
import torch
import numpy as np
import weightwatcher as ww
import matplotlib.pyplot as plt
plt.rcParams.update({
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'axes.labelsize': 20,
    'legend.fontsize': 14
})

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
        plt.plot(steps, losses, marker='o')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("{}training_loss.jpg".format(self.context["vis_dir"]))
        plt.clf()

    @torch.no_grad()
    def store_val_loss(self, loss, step):
        self.val_loss[step] = loss

    @torch.no_grad()
    def plot_val_loss(self):
        steps = list(self.val_loss.keys())
        losses = list(self.val_loss.values())
        plt.plot(steps, losses, marker='o')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("{}val_loss.jpg".format(self.context["vis_dir"]))
        plt.clf()

    @torch.no_grad()
    def plot_losses(self):
        # combined plots for easy viz
        steps = list(self.val_loss.keys())
        training_losses = list(self.training_loss.values())
        val_losses = list(self.val_loss.values())
        plt.plot(steps, training_losses, marker='o', label="train")
        plt.plot(steps, val_losses, marker='x', label="test", linestyle='dashed')
        plt.xlabel("steps")
        plt.ylabel("loss")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
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

        if self.context["enable_ww"]:
            self.get_ww_summary(student=student, step=step)

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
    def get_ww_summary(self, student, step):
        # create ww compatible filepath for saving results
        for idx, layer in enumerate(student.layers):
            watcher = ww.WeightWatcher(model=layer)
            ww_vis_dir = os.path.join(self.context["vis_dir"], "ww_step_{}_layer_{}".format(step, idx))
            os.makedirs(ww_vis_dir, exist_ok=True)
            details = watcher.analyze(plot=True, savefig=ww_vis_dir)
            summary = watcher.get_summary(details)
            logger.info("step: {} layer: {} ww_summary: {}".format(step, idx, summary))


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
            plt.plot(steps, vals, marker='o', label="layer:{}".format(layer_idx))
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
        plt.tight_layout()
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
            plt.tight_layout()
            name="{}W{}".format(self.context["vis_dir"], idx)
            plt.savefig("{}_initial_final_vals.jpg".format(name))
            plt.clf()

    @torch.no_grad()
    def plot_svd(self, S_M, name):
       plt.bar(x=list(range(S_M.shape[0])), height=S_M.cpu().numpy())
       plt.xlabel("$i$")
       plt.ylabel("$\lambda_i$")
       plt.tight_layout()
       plt.savefig("{}_sv.jpg".format(name))
       plt.clf()

    @torch.no_grad()
    def plot_esd(self, S_MtM, name):
       vals = np.log10(S_MtM.cpu().numpy())
       plt.hist(vals, bins=100, log=True, density=True, color="blue", alpha=0.5, edgecolor='blue')
       plt.xlabel("$\log_{10}(\lambda_i)$")
       plt.ylabel("$\log_{10}(ESD)$")
       plt.tight_layout()
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
            plt.tight_layout()
            name="{}W{}".format(self.context["vis_dir"], idx)
            plt.savefig("{}_initial_final_esd.jpg".format(name))
            plt.clf()

    @torch.no_grad()
    def plot_initial_final_weight_nolog_esd(self):
        steps = list(self.step_weight_esd.keys())
        initial_step = steps[0]
        final_step = steps[-1]
        for idx in self.step_weight_esd[0]:
            initial_S_WtW = self.step_weight_esd[initial_step][idx]
            final_S_WtW = self.step_weight_esd[final_step][idx]
            initial_vals = initial_S_WtW.cpu().numpy()
            final_vals = final_S_WtW.cpu().numpy()
            plt.hist(initial_vals, bins=100, density=True, color="red", alpha=0.5, edgecolor='red', label="initial")
            plt.hist(final_vals, bins=100, density=True, color="blue", alpha=0.5, edgecolor='blue', label="step{}".format(final_step))
            plt.xlabel("$\lambda_i$")
            plt.ylabel("$ESD$")
            plt.legend()
            plt.tight_layout()
            name="{}W{}".format(self.context["vis_dir"], idx)
            plt.savefig("{}_initial_final_nolog_esd.jpg".format(name))
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
            if layer_idx < self.context["L"] - 1:
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
            # KTA
            KTA = self.compute_KTA(Z=Z, y_t=y_t)
            if step not in self.step_KTA:
                self.step_KTA[step] = OrderedDict()
            self.step_KTA[step][layer_idx] = KTA
            logger.info("KTA for step:{} layer: {} = {}".format(step, layer_idx, KTA))

    @torch.no_grad()
    def plot_step_activation_stable_rank(self):
        steps = list(self.step_activation_stable_rank.keys())
        layer_idxs = list(self.step_activation_stable_rank[steps[0]].keys())
        for layer_idx in layer_idxs:
            vals = [self.step_activation_stable_rank[e][layer_idx] for e in steps]
            plt.plot(steps, vals, marker='o', label="layer:{}".format(layer_idx))
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
                plt.plot(variant1_vals, marker='o', label="layer:{},step:{},r".format(layer_idx, step))
                plt.plot(variant2_vals, marker='x', label="layer:{},step:{},R".format(layer_idx, step))
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
            plt.plot(steps, vals, marker='o', label="layer:{}".format(layer_idx))
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
            plt.plot(steps, vals, marker='o', label="layer:{}".format(layer_idx))
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
            plt.tight_layout()
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
            plt.tight_layout()
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
    def compute_resolvent(self, z, Q):
        dim0, dim1 = Q.shape
        assert dim0 == dim1
        return torch.linalg.pinv(z*torch.eye(dim0) - Q)

    @torch.no_grad()
    def compute_stieltjes_transform(self, z, Q):
        resolvent = self.compute_resolvent(z=z, Q=Q)
        dim0 = Q.shape[0]
        return torch.trace(resolvent)/dim0

    @torch.no_grad()
    def compute_block_stieltjes_transform(self, z1, z2, W1):
        val1 = z2 * self.compute_stieltjes_transform(z=z1*z2, Q=(W1 @ W1.t()))
        val2 = z1 * self.compute_stieltjes_transform(z=z1*z2, Q=(W1.t() @ W1))
        return [val1, val2]

    @torch.no_grad()
    def compute_nu_A_B(self, s_hat, s, W1, bst):
        h, d = W1.shape
        c = h/d
        sigma = 1
        nu = s_hat - c * (sigma**2) * d * bst[0]
        A = nu * ( nu - (sigma**2)*d*(1 - c)/s_hat ) - s**2
        B = 2 * c * (sigma**2)*d * ( nu - (sigma**2)*d*(1 - c)/(2*s_hat) )
        logger.info("nu: {} A: {} B: {}".format(nu, A, B))
        return nu, A, B

    @torch.no_grad()
    def compute_expected_W1_M_alignment_left(self, s_hat, s, W1, bst):
        h, d = W1.shape
        c = h/d
        sigma = 1
        nu, A, B = self.compute_nu_A_B(s_hat=s_hat, s=s, W1=W1, bst=bst)
        num = nu*B - c * (sigma**2)*d*A
        den = A**2
        return num/den

    @torch.no_grad()
    def compute_expected_W1_M_alignment_right(self, s_hat, s, W1, bst):
        h, d = W1.shape
        c = h/d
        sigma = 1
        nu, A, B = self.compute_nu_A_B(s_hat=s_hat, s=s, W1=W1, bst=bst)
        num = ( nu - (sigma**2)*d*(1-c)/s_hat )*B - c*(sigma**2)*d*A
        den = A**2
        return num/den

    @torch.no_grad()
    def plot_first_step_W1_M_alignment(self):
        steps = list(self.step_weight_vals.keys())
        initial_step = steps[0]
        final_step = steps[1]
        for idx in self.step_weight_vals[0]:
            initial_W = self.step_weight_vals[initial_step][idx]
            final_W = self.step_weight_vals[final_step][idx]

            U_initW, S_initW, Vh_initW = torch.linalg.svd(initial_W, full_matrices=False)
            U_finW, S_finW, Vh_finW = torch.linalg.svd(final_W, full_matrices=False)

            M = final_W - initial_W
            U_M, S_M, Vh_M = torch.linalg.svd(M, full_matrices=False)
            name="{}M_layer{}".format(self.context["vis_dir"], idx)
            self.plot_svd(S_M=S_M, name=name)

            s_hat = S_finW[0]
            logger.info("s_hat: {}".format(s_hat))
            logger.info("computing W1 M pc alignment")
            left_alignments = []
            right_alignments = []
            bst = self.compute_block_stieltjes_transform(z1=s_hat, z2=s_hat, W1=final_W)
            for i in tqdm(range(S_M.shape[0])):
                s = S_M[i]
                # logger.info("s: {}".format(s))
                left_alignment = self.compute_expected_W1_M_alignment_left(s_hat=s_hat, s=s, W1=final_W, bst=bst)
                left_alignments.append(left_alignment)
                right_alignment = self.compute_expected_W1_M_alignment_right(s_hat=s_hat, s=s, W1=final_W, bst=bst)
                right_alignments.append(right_alignment)
            # logger.info("expected_W1_M_alignment_left: {} expected_W1_M_alignment_right: {}".format(
            #     left_alignment, right_alignment
            # ))
            left_alignments = np.array(left_alignments)
            left_alignments /= np.sum(left_alignments)
            left_alignments = np.log10(left_alignments)

            right_alignments = np.array(right_alignments)
            right_alignments /= np.sum(right_alignments)
            right_alignments = np.log10(right_alignments)

            logger.info("initial_W shape: {} final_W: {} update shape: {}".format(
                initial_W.shape, final_W.shape, M.shape))

            # since we consider the update matrix to contain the signal:
            # we measure the singular vector alignments of M and final_W
            name="{}W{}_final_M_pc_sim.jpg".format(self.context["vis_dir"], idx)
            self.plot_pc_sims(U_W=U_finW, U_G=U_M, Vh_W=Vh_finW, Vh_G=Vh_M,
                              la=left_alignments, ra=right_alignments, name=name)

    @torch.no_grad()
    def plot_pc_sims(self, U_W, U_G, Vh_W, Vh_G, la, ra, name):
        # TODO: improve plots
        U_sim = torch.square( (U_W.t() @ U_G) )
        Vh_sim = torch.square( (Vh_W @ Vh_G.t()) )

        U_sim = U_sim[0, :]
        U_sim /= torch.sum(U_sim)
        U_sim = torch.log10(U_sim)
        Vh_sim = Vh_sim[0, :]
        Vh_sim /= torch.sum(Vh_sim)
        Vh_sim = torch.log10(Vh_sim)

        fig, axs = plt.subplots(1, 2, figsize=(16, 6))
        axs[0].plot(U_sim, label="exp")
        axs[0].plot(la, label="theory")
        U_sim_max_index = np.argmax(U_sim)
        U_sim_max_value = U_sim[U_sim_max_index]
        la_max_index = np.argmax(la)
        la_max_value = la[la_max_index]
        axs[0].axvline(x=U_sim_max_index, color='r', linestyle='--', label=f'exp max ({U_sim_max_value:.2f}), i={U_sim_max_index}', linewidth=1)
        axs[0].axvline(x=la_max_index, color='g', linestyle='--', label=f'theory max ({la_max_value:.2f}), i={la_max_index}', linewidth=1)
        axs[0].set_xlabel('i')
        axs[0].set_ylabel('alignment')
        # axs[0].set_title('Left PC')
        axs[0].legend()

        axs[1].plot(Vh_sim, label="exp")
        axs[1].plot(ra, label="theory")
        Vh_sim_max_index = np.argmax(Vh_sim)
        Vh_sim_max_value = Vh_sim[Vh_sim_max_index]
        ra_max_index = np.argmax(ra)
        ra_max_value = ra[ra_max_index]
        axs[1].axvline(x=Vh_sim_max_index, color='r', linestyle='--', label=f'exp max ({Vh_sim_max_value:.2f}), i={Vh_sim_max_index}', linewidth=1)
        axs[1].axvline(x=ra_max_index, color='g', linestyle='--', label=f'theory max ({ra_max_value:.2f}), i={ra_max_index}', linewidth=1)
        axs[1].set_xlabel('i')
        axs[1].set_ylabel('alignment')
        # axs[1].set_title('Right PC')
        axs[1].legend()

        plt.tight_layout()
        plt.savefig(name)
        plt.clf()

    @torch.no_grad()
    def plot_all_steps_W_M_alignment(self):
        steps = list(self.step_weight_vals.keys())
        for step_idx in range(len(steps)-1):
            initial_step = steps[step_idx]
            final_step = steps[step_idx+1]
            for layer_idx in self.step_weight_vals[0]:
                # skip second layer as of now.
                if layer_idx == 1: continue
                initial_W = self.step_weight_vals[initial_step][layer_idx]
                final_W = self.step_weight_vals[final_step][layer_idx]

                U_initW, S_initW, Vh_initW = torch.linalg.svd(initial_W, full_matrices=False)
                U_finW, S_finW, Vh_finW = torch.linalg.svd(final_W, full_matrices=False)

                M = final_W - initial_W
                U_M, S_M, Vh_M = torch.linalg.svd(M, full_matrices=False)
                name="{}M_layer{}_step{}".format(self.context["vis_dir"], layer_idx, step_idx+1)
                self.plot_svd(S_M=S_M, name=name)

                # compute sim between initW and M
                name="{}W{}_step{}_M_step{}".format(self.context["vis_dir"], layer_idx, step_idx, step_idx)
                self.plot_3d_sv_inner_product_squares(U_W=U_initW, U_M=U_M, Vh_W=Vh_initW, Vh_M=Vh_M, name=name)

                # compute sim between finW and M
                name="{}W{}_step{}_M_step{}".format(self.context["vis_dir"], layer_idx, step_idx+1, step_idx)
                self.plot_3d_sv_inner_product_squares(U_W=U_finW, U_M=U_M, Vh_W=Vh_finW, Vh_M=Vh_M, name=name)

    @torch.no_grad()
    def plot_3d_sv_inner_product_squares(self, U_W, U_M, Vh_W, Vh_M, name):
        U_sim = torch.square( (U_W.t() @ U_M) ).detach().cpu().numpy()
        Vh_sim = torch.square( (Vh_W @ Vh_M.t()) ).detach().cpu().numpy()

        x_left, y_left = np.meshgrid(range(U_sim.shape[0]), range(U_sim.shape[1]))
        x_left = x_left.flatten()
        y_left = y_left.flatten()
        z_left = U_sim.flatten()

        x_right, y_right = np.meshgrid(range(Vh_sim.shape[0]), range(Vh_sim.shape[1]))
        x_right = x_right.flatten()
        y_right = y_right.flatten()
        z_right = Vh_sim.flatten()

        # Plot 3D figure for inner products of left singular vectors
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x_left, y_left, z_left, c=z_left, cmap='viridis_r')
        # ax1.set_title('Inner Products of Left Singular Vectors')
        ax.set_xlabel('U_W', labelpad=15)
        ax.set_ylabel('U_M', labelpad=25)
        ax.set_zlabel('overlap', labelpad=25)
        ax.zaxis.set_tick_params(pad=15)
        plt.tight_layout()
        plt.savefig("{}_left.jpg".format(name))
        plt.clf()
        plt.close()

        # Plot 3D figure for inner products of right singular vectors
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x_right, y_right, z_right, c=z_right, cmap='viridis_r')
        # ax2.set_title('Inner Products of Right Singular Vectors')
        ax.set_xlabel('V_W', labelpad=15)
        ax.set_ylabel('V_M', labelpad=25)
        ax.set_zlabel('overlap', labelpad=25)
        ax.zaxis.set_tick_params(pad=15)
        # ax2.view_init(30, 60)
        plt.tight_layout()
        plt.savefig("{}_right.jpg".format(name))
        plt.clf()
        plt.close()