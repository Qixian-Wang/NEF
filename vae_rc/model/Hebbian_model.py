import torch
import torch.nn as nn
from NEF.vae_rc.method.functional import *

class HebbianLearning(nn.Module):
    # delta_w = alpha * r * (x - reconst)
    def __init__(self, config, in_channels, out_size, kernel_size):
        super(HebbianLearning, self).__init__()
        self.config = config
        self.training = True
        self.kernel_size = kernel_size
        self.in_channels = in_channels

        # Init weights
        self.out_size = torch.tensor([out_size])
        out_channels = self.out_size.prod().item()
        kernel_size = [self.kernel_size, self.kernel_size]
        stdv = 1 / (self.in_channels * kernel_size[0] * kernel_size[1]) ** 0.5
        self.weight = nn.Parameter(torch.empty(out_channels, self.in_channels, kernel_size[0], kernel_size[1]),
                                   requires_grad=True)
        nn.init.uniform_(self.weight, -stdv, stdv)
        if config.weight_init == 'norm': self.weight = self.weight / self.weight.view(self.weight.size(0), -1).norm(
            dim=1, p=2).view(-1, 1, 1, 1)  # normalize weights

        # Enable/disable features as random abstention, competitive learning, lateral feedback, type of reconstruction
        self.reconstruction = config.reconstruction
        self.reduction = config.reduction
        self.tau = config.tau

        # Set output function, similarity function and learning rule
        self.lrn_sim = kernel_mult2d
        self.lrn_act = identity
        self.out_sim = kernel_mult2d
        self.out_act = identity
        self.teacher_signal = None  # Teacher signal for supervised training
        self.weight_upd_rule = config.weight_upd_rule

        # Alpha is the constant which determines the trade off between global and local updates
        self.alpha = config.alpha

        # Buffer where the weight update is stored
        self.register_buffer('delta_w', torch.zeros_like(self.weight))
        self.to(config.device)

    def forward(self, data, label):
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

        winner_mask = torch.ones_like(y, device=y.device)
        lfb_out = winner_mask

        # Compute step modulation coefficient
        r = lfb_out  # RULE_BASE
        if self.weight_upd_rule == 'hebb':
            r = r * y

        r_abs = r.abs()
        r_sign = r.sign()

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
            r_sign_i = r_sign.unsqueeze(2)[:, start:end, :]
            if self.reconstruction == 'qnt': x_bar = w_i
            if self.reconstruction == 'qnt_sgn': x_bar = r_sign_i * w_i
            if self.reconstruction == 'lin_cmb': x_bar = torch.cumsum(r_i * w_i, dim=1) + (
                x_bar[:, -1, :].unsqueeze(1) if x_bar is not None else 0.)
            x_bar = x_bar if x_bar is not None else 0.
            delta_w_i = r_i * (x_unf - x_bar)
            # Since we use batches of inputs, we need to aggregate the different update steps of each kernel in a unique
            # update. We do this by taking the weighted average of the steps, the weights being the r coefficients that
            # determine the length of each step
            if self.reduction == 'w_avg':
                r_sum = r_abs_i.sum(0)
                r_sum = r_sum + (r_sum == 0).float()  # Prevent divisions by zero
                delta_w_avg[start:end, :] = (delta_w_i * r_abs_i).sum(0) / r_sum
            else:
                delta_w_avg[start:end, :] = delta_w_i.mean(dim=0)  # RED_AVG

        # Apply delta
        self.delta_w_hebbian = delta_w_avg.view_as(self.weight)

    def local_update(self):
        hebbian_grad = self.alpha * (-self.delta_w_hebbian)
        ff_grad = self.ff_grad if self.config.ff_activate else torch.zeros_like(hebbian_grad)

        self.weight.grad = hebbian_grad #+ ff_grad
        return hebbian_grad, self.ff_loss

    def calc_ff_loss(self, z, labels):
        HW = z.size(0) // labels.size(0)
        z2 = z.view(labels.size(0), HW, z.size(1))
        logits = z2.pow(2).sum(dim=2)
        logits = logits - z.size(1)
        labels_map = labels.view(labels.size(0), 1).expand(labels.size(0), HW).float()
        ff_loss = F.binary_cross_entropy_with_logits(logits, labels_map)
        return ff_loss, logits
