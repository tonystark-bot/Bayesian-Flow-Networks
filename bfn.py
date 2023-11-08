"""
Implementation from https://github.com/Algomancer/Bayesian-Flow-Networks/blob/main/model.py
I just added type annotations, made it agnostic to the NN used, and cleaned it up a bit.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from jaxtyping import Float, Int
from typing import Optional
from gpt import GPT


def get_basic_net(D: int, vocab_size: int, hidden_dim: int):
    return nn.Sequential(
        # TODO: not sure if input size of D * vocab_size + 1 is correct, shouldn't it be D * vocab_size?
        # edit: okay the +1 is so that we can add in the timestep value
        nn.Linear(D * vocab_size + 1, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, D * vocab_size),
    )

def get_gpt_net(D: int, vocab_size: int, hidden_dim: int, num_layers: int, num_heads: Optional[int] = None, dropout: float = 0.0) -> GPT:
    config = GPT.get_default_config()
    config.model_type = None
    config.n_layer = num_layers
    if num_heads is not None:
        config.n_head = num_heads
    else:
        assert hidden_dim % 64 == 0, "hidden_dim must be divisible by 64"
        config.n_head = hidden_dim // 64
    config.n_embd = hidden_dim
    config.vocab_size = vocab_size
    config.block_size = D
    config.embd_pdrop = dropout
    config.resid_pdrop = dropout
    config.attn_pdrop = dropout
    print(config)
    return GPT(config)


class BayesianFlowNetwork(nn.Module):
    """
    Bayesian Flow Network (BFN) model from https://arxiv.org/pdf/2308.07037.pdf

    Args:
        net (torch.nn.Module): Neural network to use for the BFN. This should take in a tensor of shape (B, D * K + 1) and output a tensor of shape (B, D * K).
        D (int): Number of dimensions of the input data (aka sequence length for all you gpt people).
        vocab_size (int): Size of the vocabulary / number of classes.
        beta (float, optional): Beta parameter. This controls the accuracy schedule, see eq 183 in the paper. Defaults to 3.0.
    """

    def __init__(self, net: torch.nn.Module, D: int, vocab_size: int, beta: float = 3.0):
        super(BayesianFlowNetwork, self).__init__()
        self.beta = beta  # TODO: what does this do
        self.D = D
        self.vocab_size = vocab_size
        self.layer = net

    def forward(
        self, theta: Float[Tensor, "B D K"], t: Float[Tensor, "B"]
    ) -> Float[Tensor, "B D K"]:
        """
        Forward pass of the Bayesian Flow Network.
        """
        theta = (theta * 2) - 1  # scaled in [-1, 1]
        # for simplenet:
        # theta = theta.view(theta.shape[0], -1)  # (B, D * K)
        # input_ = torch.cat((theta, t.unsqueeze(-1)), dim=-1)
        # for gpt:
        t = t.unsqueeze(-1).unsqueeze(-1).repeat(1, theta.shape[1], 1)  # (B, 1, K)
        input_ = torch.cat((theta, t), dim=-1) # add t to hidden state
        output: Tensor = self.layer(input_)
        output = output.view(output.shape[0], self.D, -1)
        return output

    def discrete_output_distribution(
        self, theta: Float[Tensor, "B D K"], t: Float[Tensor, "B"]
    ) -> Float[Tensor, "B D K"]:
        """
        Computes the discrete output distribution.
        """
        output = self.forward(theta, t)
        p0 = torch.nn.functional.softmax(output, dim=-1)
        return p0

    def process(self, x: Int[Tensor, "B D"]):
        """
        The forward pass during training.
        """
        # Step 1: Sample t from U(0, 1)
        t = torch.rand((x.size(0),), device=x.device, dtype=torch.float32)

        # Step 2: Sample y from N(beta * (K * one_hot(X))
        beta = self.beta * (t**2)  # (B,)
        one_hot_x = F.one_hot(x, num_classes=self.vocab_size).float()  # (B, D, K)
        mean = beta[:, None, None] * (self.vocab_size * one_hot_x - 1)
        std = (beta * self.vocab_size)[:, None, None].sqrt()
        eps = torch.randn_like(mean)
        y = mean + std * eps

        # Step 3: Compute the Theta
        theta = F.softmax(y, dim=-1)

        # Step 4: Calculate the output distribution
        p_0 = self.discrete_output_distribution(theta, t)  # (B, D, K)

        e_x = one_hot_x
        e_hat = p_0  # (B, D, K)
        L_infinity: Tensor = (
            self.vocab_size * self.beta * t[:, None, None] * ((e_x - e_hat) ** 2)
        )
        return L_infinity.mean()

    @torch.inference_mode()
    def sample(self, batch_size: int = 128, nb_steps: int = 10, device="cpu"):
        self.eval()

        # get prior
        theta = (
            torch.ones((batch_size, self.D, self.vocab_size), device=device)
            / self.vocab_size
        )

        for i in range(1, nb_steps + 1):
            t = (i - 1) / nb_steps
            t = t * torch.ones((theta.shape[0]), device=theta.device, dtype=theta.dtype)

            k_probs = self.discrete_output_distribution(theta, t)  # (B, D, K)
            k = torch.distributions.Categorical(probs=k_probs).sample()  # (B, D)
            alpha = self.beta * (2 * i - 1) / (nb_steps**2)

            e_k = F.one_hot(k, num_classes=self.vocab_size).float()  # (B, D, K)
            mean = alpha * (self.vocab_size * e_k - 1)
            var = alpha * self.vocab_size
            std = torch.full_like(mean, fill_value=var).sqrt()
            eps = torch.randn_like(e_k)

            y = mean + std * eps  # (B, D, K)

            theta_prime = torch.exp(y) * theta
            theta = theta_prime / theta_prime.sum(-1, keepdim=True)

        k_probs_final = self.discrete_output_distribution(theta, torch.ones_like(t))
        k_final = torch.distributions.Categorical(probs=k_probs_final).sample()

        return k_final

def get_param_groups(model: torch.nn.Module, weight_decay: float):
    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, )
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
            # random note: because named_modules and named_parameters are recursive
            # we will see the same tensors p many many times. but doing it this way
            # allows us to know which parent module any tensor p belongs to...
            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                % (str(param_dict.keys() - union_params), )

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    return optim_groups