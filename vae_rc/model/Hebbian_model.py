import torch.nn as nn
from NEF.vae_rc.method.functional import *
import torch.nn.functional as F
from NEF.vae_rc.utils import lorenz_system, display_hybrid_images


class HebbianLearning(nn.Module):
    # def __init__(self, config, input_dim, out_size):
    #     super(HebbianLearning, self).__init__()
    #     self.config = config
    #     self.training = True
    #     self.lr = 6e-4
    #     self.gamma = 5e-4
    #
    #     self.feedforward_weights = nn.Parameter(torch.empty(input_dim, out_size), requires_grad=True)
    #     stdv = 1 / (input_dim ** 0.5)
    #     nn.init.uniform_(self.feedforward_weights, -stdv, stdv)
    #
    #     self.lateral_weights = nn.Parameter(torch.zeros(out_size, out_size), requires_grad=True)
    #     with torch.no_grad():
    #         self.lateral_weights.fill_diagonal_(0)
    #
    #     self.register_buffer('DY', torch.full((out_size,), 1e-3))
    #     self.to(config.device)
    #
    # def forward(self, data):
    #     self.total_cost = 0
    #
    #     data = data - data.mean(dim=0, keepdim=True)
    #     y = data @ self.feedforward_weights
    #     eta_act = 0.9
    #
    #     for _ in range(self.config.max_iter):
    #         y_new = (1 - eta_act) * y + eta_act * (data @ self.feedforward_weights - y @ self.lateral_weights)
    #         if (y_new - y).abs().max() < 1e-5:
    #             break
    #         y = y_new
    #
    #     self.current_output = y.detach()
    #     if self.training:
    #         self.compute_forward_update(data)
    #         self.compute_lateral_update()
    #     return y
    #
    # def compute_forward_update(self, data):
    #     torch.set_grad_enabled(False)
    #     data = data.reshape(data.shape[0], -1)
    #     r = self.current_output
    #     batch_y2 = (r ** 2).sum(dim=0) / data.shape[0]
    #
    #     eta = 0.8
    #     self.DY += batch_y2
    #     # print(self.DY)
    #     num = data.t() @ r
    #     decay = self.feedforward_weights * batch_y2.unsqueeze(0)
    #     self.delta_w_hebbian = (num - decay) / self.DY.unsqueeze(0)
    #     recon = ((data - r @ self.feedforward_weights.t()) ** 2).sum()
    #     self.total_cost = recon
    #
    # def compute_lateral_update(self):
    #     r = self.current_output
    #     y_outer = r.t() @ r
    #     decay = ((r ** 2).sum(dim=0) / r.shape[0]).unsqueeze(1) * self.lateral_weights
    #     self.delta_lateral = ((1 + self.gamma) * y_outer - decay) / self.DY.unsqueeze(1)
    #     self.delta_lateral.fill_diagonal_(0)
    #
    # def local_update(self):
    #     with torch.no_grad():
    #         self.feedforward_weights += self.lr * self.delta_w_hebbian
    #         self.lateral_weights += self.gamma * self.delta_lateral
    #
    #     return self.total_cost

    def __init__(self, config, input_dim, out_size):
        super().__init__()
        self.config = config

        self.tau = 0.5
        self.learning_rate_fn = lambda t: 1.0 / (t + 5)

        self.feedforward_weights = nn.Parameter(torch.empty(input_dim, out_size))
        stdv = 1.0 / (input_dim ** 0.5)
        nn.init.uniform_(self.feedforward_weights, -stdv, stdv)

        self.lateral_weights = nn.Parameter(torch.eye(out_size))
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
