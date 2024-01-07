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
        beta = beta/torch.norm(beta, p=2)
        self.beta = beta.unsqueeze(1)

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
        self.activation_fn = torch.tanh
        self.init_weights()
        self._assign_hooks()

    def init_weights(self):
        self.hidden_layer = torch.nn.Linear(
            in_features=self.context["d"],
            out_features=self.context["h"],
            bias=False
        )
        torch.nn.init.normal_(self.hidden_layer.weight)
        self.final_layer = torch.nn.Linear(
            in_features=self.context["h"],
            out_features=1,
            bias=False
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
        self.hidden_layer.register_forward_hook(
            self._probe_affine_features(idx=0)
        )

    def forward(self, X):
        Z = self.hidden_layer(X)
        Z /= np.sqrt(self.context["d"])
        Z = self.activation_fn(Z)
        out = self.final_layer(Z)
        out /= np.sqrt(self.context["h"])
        return out

class Student3Layer(torch.nn.Module):
    def __init__(self, context):
        super().__init__()
        self.context = context
        self.init_weights()
        self.activation_fn = torch.tanh

    def init_weights(self):
        self.hidden_layer1 = torch.nn.Linear(
            in_features=self.context["d"],
            out_features=self.context["h"],
            bias=False
        )
        torch.nn.init.normal_(self.hidden_layer1.weight)
        self.hidden_layer2 = torch.nn.Linear(
            in_features=self.context["h"],
            out_features=self.context["h"],
            bias=False
        )
        torch.nn.init.normal_(self.hidden_layer2.weight)
        self.final_layer = torch.nn.Linear(
            in_features=self.context["h"],
            out_features=1,
            bias=False
        )
        torch.nn.init.normal_(self.final_layer.weight)
        self.layers = [self.hidden_layer1, self.hidden_layer2, self.final_layer]

    def forward(self, X):
        Z = self.hidden_layer1(X)
        Z /= np.sqrt(self.context["d"])
        Z = self.activation_fn(Z)
        Z = self.hidden_layer2(Z)
        Z /= np.sqrt(self.context["h"])
        Z = self.activation_fn(Z)
        out = self.final_layer(Z)
        out /= np.sqrt(self.context["h"])
        return out
