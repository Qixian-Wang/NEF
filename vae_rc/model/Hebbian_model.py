import torch
import torch.nn as nn
from NEF.vae_rc.method.functional import *
import torch.nn.functional as F
from NEF.vae_rc.utils import lorenz_system, display_hybrid_images


class HebbianLearning(nn.Module):
    # delta_w = alpha * r * (x - reconst)
    def __init__(self, config, in_channels, out_size, kernel_size):
        super(HebbianLearning, self).__init__()
        self.config = config
        self.training = True
        self.kernel_size = kernel_size
        self.in_channels = in_channels

        # Set output function, similarity function and learning rule
        self.lrn_sim = kernel_mult2d
        self.lrn_act = identity
        self.out_sim = kernel_mult2d
        self.out_act = identity
        self.teacher_signal = None  # Teacher signal for supervised training

        # Alpha is the constant which determines the tradeoff between global and local updates
        self.alpha = config.alpha

        # Init weights
        out_channels = out_size
        kernel_size = [self.kernel_size, self.kernel_size]
        stdv = 1 / (self.in_channels * kernel_size[0] * kernel_size[1]) ** 0.5
        self.weight = nn.Parameter(torch.empty(out_channels, self.in_channels, kernel_size[0], kernel_size[1]),
                                   requires_grad=True)
        nn.init.uniform_(self.weight, -stdv, stdv)

        # Buffer where the weight update is stored
        self.register_buffer('delta_w', torch.zeros_like(self.weight))
        self.to(config.device)

    def forward(self, data, label, epoch):
        self.total_recon_loss = 0
        self.epoch = epoch
        output = self.out_act(self.out_sim(data, self.weight))
        if self.training and self.alpha != 0: self.compute_update(data, label)
        return output

    def compute_update(self, x, label):
        # Store previous gradient computation flag and disable gradient computation before computing update
        torch.set_grad_enabled(True)

        # Prepare the inputs
        s = self.lrn_sim(x, self.weight)
        y = self.lrn_act(s)

        s = s.permute(0, 2, 3, 1).contiguous().view(-1, self.weight.size(0))
        y = y.permute(0, 2, 3, 1).contiguous().view(-1, self.weight.size(0))
        x_unf = unfold_map2d(x, self.weight.size(2), self.weight.size(3))
        x_unf = x_unf.permute(0, 2, 3, 1, 4).contiguous().view(s.size(0), 1, -1)

        if self.config.ff_activate:
            self.ff_loss, logits = self.calc_ff_loss(y, label)
            self.ff_grad = torch.autograd.grad(self.ff_loss, self.weight, retain_graph=True)[0]

        torch.set_grad_enabled(False)

        if self.config.ff_activate:
            label_mask = label.view(-1, 1)
        else:
            label_mask = torch.ones((x.shape[0], 1)).to(self.config.device)

        r = y * label_mask
        r_abs = r.abs()

        # Compute delta_w (serialized version for computation of delta_w using less memory)
        w = self.weight.view(1, self.weight.size(0), -1)
        delta_w_avg = torch.zeros_like(self.weight.view(self.weight.size(0), -1))
        x_bar = None

        for i in range((self.weight.size(0) // self.config.HEBB_UPD_GRP) +
                       (1 if self.weight.size(0) % self.config.HEBB_UPD_GRP != 0 else 0)):
            start = i * self.config.HEBB_UPD_GRP
            end = min((i + 1) * self.config.HEBB_UPD_GRP, self.weight.size(0))
            w_i = w[:, start:end, :]
            r_i = r.unsqueeze(2)[:, start:end, :]
            r_abs_i = r_abs.unsqueeze(2)[:, start:end, :]
            x_bar = torch.cumsum(r_i * w_i, dim=1) + (x_bar[:, -1, :].unsqueeze(1) if x_bar is not None else 0.)
            x_bar = x_bar if x_bar is not None else 0.

            delta_w_i = r_i * (x_unf - x_bar)
            r_sum = r_abs_i.sum(0)
            r_sum = r_sum + (r_sum == 0).float()
            delta_w_avg[start:end, :] = (delta_w_i * r_abs_i).sum(0) / r_sum

            # Compute recon loss
            per_sample_err = (x_unf - x_bar).pow(2).sum(dim=(1, 2))  # [B]
            mask1d = label_mask.view(x_unf.size(0)).float()  # [B]
            recon_error = (per_sample_err * mask1d).sum()

            self.total_recon_loss += recon_error
            # if self.epoch == 3:
            #     display_hybrid_images(x_bar[i, :, :].reshape(28, 28).detach().cpu().numpy(), x_unf[i, :, :].reshape(28, 28).detach().cpu().numpy(), x_unf[0, :, :].detach().cpu().numpy())

        # Apply delta
        self.delta_w_hebbian = delta_w_avg.view_as(self.weight)

    def local_update(self):
        hebbian_grad = self.alpha * (-self.delta_w_hebbian)
        ff_grad = self.ff_grad if self.config.ff_activate else torch.zeros_like(hebbian_grad)

        a = hebbian_grad.sum().item()
        b = ff_grad.sum().item()
        self.weight.grad = hebbian_grad + 1 * ff_grad
        return self.total_recon_loss

    def calc_ff_loss(self, z, labels):
        HW = z.size(0) // labels.size(0)
        z2 = z.view(labels.size(0), HW, z.size(1))
        logits = z2.pow(2).sum(dim=2)
        logits = logits - z.size(1)
        labels_map = labels.view(labels.size(0), 1).expand(labels.size(0), HW).float()
        ff_loss = F.binary_cross_entropy_with_logits(logits, labels_map)
        return ff_loss, logits
