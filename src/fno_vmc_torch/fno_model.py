import torch
import torch.nn as nn
from neuralop.models import FNO2d


class FNOAnsatz(nn.Module):
    """FNO-based variational ansatz."""

    def __init__(self, dim: int, modes1: int, modes2: int = None, width: int = 32):
        super().__init__()
        self.dim = dim
        self.in_channels = 1
        if dim == 1:
            modes2 = 1
        else:
            assert modes2 is not None, "modes2 must be set for 2D"
        # FNO2d(in_channels, modes1, modes2, width)
        self.fno = FNO2d(self.in_channels, modes1, modes2, width)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(width, 1)

    def forward(self, x):
        # x: [batch, N] with values in {-1, +1}
        batch = x.shape[0]
        if self.dim == 1:
            L = x.shape[1]
            u = x.view(batch, L, 1, 1).float()
        else:
            L = int(x.shape[1] ** 0.5)
            u = x.view(batch, L, L, 1).float()
        # to [batch, in_channels, H, W]
        u = u.permute(0, 3, 1, 2)
        out = self.fno(u)
        out = self.pool(out).view(batch, -1)
        log_psi = self.head(out).squeeze(-1)
        return log_psi