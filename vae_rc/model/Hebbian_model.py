import torch.nn as nn
from NEF.vae_rc.method.functional import *
import torch.nn.functional as F
from NEF.vae_rc.utils import lorenz_system, display_hybrid_images


class HebbianLearning(nn.Module):
    def __init__(self, config, input_dim, out_size):
        super().__init__()
        self.config = config

        self.tau = 0.5
        self.learning_rate_fn = lambda t: 1.0 / (t + 5)

        self.feedforward_weights = nn.Parameter(torch.empty(input_dim, out_size))
        stdv = 1.0 / (input_dim ** 0.5)
        nn.init.uniform_(self.feedforward_weights, -stdv, stdv)

        self.lateral_weights = nn.Parameter(0.5*torch.eye(out_size))
        self.register_buffer('time_step', torch.tensor(0.0))
        self.to(config.device)

    def forward(self, data):
        data = data.reshape(data.size(0), -1)
        lateral_weights_inv = torch.inverse(self.lateral_weights)

        output = (data @ self.feedforward_weights) @ lateral_weights_inv.t()
        self.current_output = output.detach()

        return output

    def local_update(self, data):
        data_batch = data.reshape(data.size(0), -1)
        w_xy = self.feedforward_weights.data.t().clone()
        w_yy = self.lateral_weights.data.clone()

        for sample in data_batch:
            projection = w_xy @ sample
            y = torch.linalg.solve(w_yy, projection)

            # 1) Update w_xy
            step_w = self.learning_rate_fn(self.time_step.item())
            yx_outer = y.unsqueeze(1) * sample.unsqueeze(0)
            w_xy.mul_(1 - 2 * step_w).add_(2 * step_w * yx_outer)

            # 2) Update w_yy
            step_m = step_w / self.tau
            yy_outer = y.unsqueeze(1) * y.unsqueeze(0)
            w_yy.mul_(1 - step_m).add_(step_m * yy_outer)

            self.time_step += 1

        with torch.no_grad():
            self.feedforward_weights.data.copy_(w_xy.t())
            self.lateral_weights.data.copy_(w_yy)

        self.feedforward_weights.grad = None
        self.lateral_weights.grad = None

        recon_error = ((data - self.current_output @ self.feedforward_weights.t()) ** 2).sum()
        return recon_error
