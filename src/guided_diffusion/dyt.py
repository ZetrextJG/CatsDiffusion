"""
My implementation of DynamicTanh, a custom activation function from the paper:
Zhu, J., Chen, X., He, K., LeCun, Y., & Liu, Z. (2025). 
Transformers without normalization. arXiv preprint arXiv:2503.10622.
"""
import torch as th
import torch.nn as nn


class DyTNorm(nn.Module):

    def __init__(self, channels, init_alpha = 0.5):
        super().__init__()
        self.channels = channels
        self.alpha = nn.Parameter(th.ones(channels) * init_alpha)
        self.gamma = nn.Parameter(th.ones(channels))
        self.beta = nn.Parameter(th.zeros(channels))

    def forward(self, x):
        # Input of shape [B, C, ...]
        view_shape = [1] * len(x.shape)
        view_shape[1] = self.channels

        alpha = self.alpha.view(view_shape)
        beta = self.beta.view(view_shape)
        gamma = self.gamma.view(view_shape)

        x = th.tanh(alpha * x)
        return gamma * x + beta