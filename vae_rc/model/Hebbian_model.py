import torch.nn as nn
from NEF.vae_rc.method.functional import *

class HebbianLearning(nn.Module):
    # delta_w = alpha * r * (x - reconst)
    def __init__(self, config):
        super(HebbianLearning, self).__init__()
        self.config = config
        self.training = True
        self.kernel_size = config.kernel_size
        self.in_channels = config.in_channels

        # Init weights
        self.out_size = torch.tensor([config.num_hidden])
        out_channels = self.out_size.prod().item()
        kernel_size = [self.kernel_size, self.kernel_size]
        stdv = 1 / (self.in_channels * kernel_size[0] * kernel_size[1]) ** 0.5
        self.weight = nn.Parameter(torch.empty(out_channels, self.in_channels, kernel_size[0], kernel_size[1]),
                                   requires_grad=True)
        nn.init.uniform_(self.weight, -stdv, stdv)
        if config.weight_init == 'norm': self.weight = self.weight / self.weight.view(self.weight.size(0), -1).norm(
            dim=1, p=2).view(-1, 1, 1, 1)  # normalize weights

        # Enable/disable features as random abstention, competitive learning, lateral feedback, type of reconstruction
        self.competitive = config.competitive
        self.reconstruction = config.reconstruction
        self.reduction = config.reduction
        self.random_abstention = self.competitive and config.random_abstention
        self.lfb_on = self.competitive and isinstance(config.lfb_value, str)
        self.lfb_value = config.lfb_value
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

        # Set parameters related to the lateral feedback feature
        if self.lfb_on:
            # Prepare the variables to generate the kernel that will be used to apply lateral feedback
            map_radius = (self.out_size - 1) // 2
            sigma_lfb = map_radius.max().item()
            x = torch.abs(torch.arange(0, self.out_size[0].item()) - map_radius[0])
            for i in range(1, self.out_size.size(0)):
                x_new = torch.abs(torch.arange(0, self.out_size[i].item()) - map_radius[i])
                for j in range(i): x_new = x_new.unsqueeze(j)
                x = torch.max(x.unsqueeze(-1),
                              x_new)  # max gives L_infinity distance, sum would give L_1 distance, root_p(sum x^p) for L_p
            # Store the kernel that will be used to apply lateral feedback in a registered buffer
            if self.lfb_value == self.LFB_EXP or self.lfb_value == self.LFB_DoE:
                self.register_buffer('lfb_kernel', torch.exp(-x.float() / sigma_lfb))
            else:
                self.register_buffer('lfb_kernel', torch.exp(-x.pow(2).float() / (2 * (sigma_lfb ** 2))))
            # Padding that will pad the inputs before applying the lfb kernel
            pad_pre = map_radius.unsqueeze(1)
            pad_post = (self.out_size - 1 - map_radius).unsqueeze(1)
            self.pad = tuple(torch.cat((pad_pre, pad_post), dim=1).flip(0).view(-1))
            # LFB kernel shrinking parameter
            self.gamma = torch.exp(torch.log(torch.tensor(sigma_lfb).float()) / self.tau).item()
            if self.lfb_value == self.LFB_GAUSS or self.lfb_value == self.LFB_DoG: self.gamma = self.gamma ** 2
        else:
            self.register_buffer('lfb_kernel', None)

        # Init variables for statistics collection
        if self.random_abstention:
            self.register_buffer('victories_count', torch.zeros(out_channels))
        else:
            self.register_buffer('victories_count', None)

        self.to(config.device)

    def forward(self, x):
        y = self.out_act(self.out_sim(x, self.weight))
        if self.training and self.alpha != 0: self.compute_update(x)
        return y

    def compute_update(self, x):
        # Store previous gradient computation flag and disable gradient computation before computing update
        prev_grad_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(False)

        # Prepare the inputs
        s = self.lrn_sim(x, self.weight)
        # Compute y and y'
        torch.set_grad_enabled(True)
        y = self.lrn_act(s)

        s = s.permute(0, 2, 3, 1).contiguous().view(-1, self.weight.size(0))
        y = y.permute(0, 2, 3, 1).contiguous().view(-1, self.weight.size(0))
        x_unf = unfold_map2d(x, self.weight.size(2), self.weight.size(3))
        x_unf = x_unf.permute(0, 2, 3, 1, 4).contiguous().view(s.size(0), 1, -1)

        # Random abstention
        if self.random_abstention:
            abst_prob = self.victories_count / (self.victories_count.max() + y.size(0) / y.size(1)).clamp(1)
            scores = y * (torch.rand_like(abst_prob, device=y.device) >= abst_prob).float().unsqueeze(0)
        else:
            scores = y

        # Competition. The returned winner_mask is a bitmap telling where a neuron won and where one lost.
        if self.competitive:
            winner_mask = (scores == scores.max(1, keepdim=True)[0]).float()
            if self.random_abstention:  # Update statistics if using random abstension
                winner_mask_sum = winner_mask.sum(0)  # Number of inputs over which a neuron won
                self.victories_count += winner_mask_sum
                self.victories_count -= self.victories_count.min().item()
        else:
            winner_mask = torch.ones_like(y, device=y.device)

        # Lateral feedback
        if self.lfb_on:
            lfb_kernel = self.lfb_kernel
            if self.lfb_value == self.LFB_DoG or self.lfb_value == self.LFB_DoE: lfb_kernel = 2 * lfb_kernel - lfb_kernel.pow(
                0.5)  # Difference of Gaussians/Exponentials (mexican hat shaped function)
            lfb_in = F.pad(winner_mask.view(-1, *self.out_size), self.pad)
            if self.out_size.size(0) == 1:
                lfb_out = torch.conv1d(lfb_in.unsqueeze(1), lfb_kernel.unsqueeze(0).unsqueeze(1))
            elif self.out_size.size(0) == 2:
                lfb_out = torch.conv2d(lfb_in.unsqueeze(1), lfb_kernel.unsqueeze(0).unsqueeze(1))
            else:
                lfb_out = torch.conv3d(lfb_in.unsqueeze(1), lfb_kernel.unsqueeze(0).unsqueeze(1))
            lfb_out = lfb_out.clamp(-1, 1).view_as(y)
        else:
            lfb_out = winner_mask
            if self.competitive:
                lfb_out[lfb_out == 0] = self.lfb_value

        # Compute step modulation coefficient
        r = lfb_out  # RULE_BASE
        if self.weight_upd_rule == 'hebb': r = r * y
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
        self.delta_w = delta_w_avg.view_as(self.weight)

        # LFB kernel shrinking schedule
        if self.lfb_on: self.lfb_kernel = self.lfb_kernel.pow(self.gamma)

        # Restore gradient computation
        torch.set_grad_enabled(prev_grad_enabled)

    def local_update(self):
        if self.alpha != 0:
            self.weight.grad = self.alpha * (-self.delta_w)