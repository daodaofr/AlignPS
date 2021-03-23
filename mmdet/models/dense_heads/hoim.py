from __future__ import absolute_import

import torch
import torch.nn.functional as F
from torch import nn, autograd

from .oim_utils import tensor_gather


class OIM2CQ(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets,
                lut, cq, cqb, cq_omega, cqb_omega,
                omega_decay, momentum):
        outputs_labeled = inputs.mm(lut.t())
        outputs_unlabeled = inputs.mm(cq.t())
        outputs_background = inputs.mm(cqb.t())

        omega_x_unlabeled = outputs_labeled.max(dim=1)[0] /  \
            (outputs_unlabeled.max(dim=1)[0] + 1e-12)
        omega_x_background = torch.cat([outputs_labeled, outputs_unlabeled], dim=1).max(dim=1)[0] / \
            (outputs_background.max(dim=1)[0] + 1e-12)

        ctx.save_for_backward(inputs, targets, omega_x_unlabeled,
                              omega_x_background,
                              lut, cq, cqb, cq_omega, cqb_omega,
                              omega_decay, momentum)
        return torch.cat([outputs_background, outputs_labeled, outputs_unlabeled], dim=1)

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets, omega_x_unlabeled, \
            omega_x_background, \
            lut, cq, cqb, cq_omega, cqb_omega,\
            omega_decay, momentum = ctx.saved_tensors

        inputs, targets, omega_x_unlabeled, omega_x_background = \
            tensor_gather(
                [inputs, targets, omega_x_unlabeled, omega_x_background])

        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(
                torch.cat([cqb, lut, cq], dim=0))
            if grad_inputs.dtype == torch.float16:
                grad_inputs = grad_inputs.to(torch.float32)

        for x, y, omega, omega_b in zip(inputs, targets, omega_x_unlabeled, omega_x_background):
            if y == -2:
                if omega_b < cqb_omega.min():
                    continue
                else:
                    header_cqb = cqb_omega.argmin().item()
                    cqb[header_cqb] = x
                    cqb_omega[header_cqb] = omega_b

            elif y < len(lut) and y >= 0:
                lut[y] = momentum * \
                    lut[y] + (1. - momentum) * x
                lut[y] /= max(lut[y].norm(), 1e-12)

            else:
                if omega < cq_omega.min():
                    continue
                else:
                    header_cq = cq_omega.argmin().item()
                    cq[header_cq] = x
                    cq_omega[header_cq] = omega

        cq_omega *= omega_decay
        cqb_omega *= omega_decay

        return grad_inputs, None, None, None, None, None, None, None, None


def oim2cq(inputs, targets, lut, cq, cqb, cq_omega, cqb_omega, omega_decay=0.99, momentum=0.5):
    return OIM2CQ.apply(inputs, targets,
                        lut, cq, cqb, cq_omega, cqb_omega,
                        torch.tensor(omega_decay), torch.tensor(momentum))


class HOIMLoss(nn.Module):

    def __init__(self, num_features, num_pids, num_cq_size, num_bg_size,
                 oim_momentum, oim_scalar,
                 omega_decay=0.99,
                 dynamic_lambda=True,
                 alpha_d=None, alpha_r=None, gamma_d=None, gamma_r=None):
        super(HOIMLoss, self).__init__()
        self.num_features = num_features
        self.num_pids = num_pids
        self.num_unlabeled = num_cq_size
        self.num_background = num_bg_size
        self.momentum = oim_momentum
        self.omega_decay = omega_decay
        self.oim_scalar = oim_scalar
        self.dynamic_lambda = dynamic_lambda

        # Focal Loss, alpha = 0.25, gamma = 2 works best according to the
        # paper.
        self.alpha_d = alpha_d if alpha_d is not None else 0.25
        self.alpha_r = alpha_r if alpha_r is not None else 0.25
        self.gamma_d = gamma_d if gamma_d is not None else 2.0
        self.gamma_r = gamma_r if gamma_r is not None else 2.0

        self.register_buffer('lut', torch.zeros(
            self.num_pids, self.num_features))

        self.register_buffer('cq', torch.zeros(
            self.num_unlabeled, self.num_features))
        self.register_buffer('cq_omega', torch.zeros(
            self.num_unlabeled))

        self.register_buffer('cqb', torch.zeros(
            self.num_background, self.num_features))
        self.register_buffer('cqb_omega', torch.zeros(
            self.num_background))

    def build_det_loss(self, all_probs, roi_label=None):
        label = roi_label + 2

        label = label.squeeze().clamp(0, 1)  # background label = 0

        cls_score = torch.cat([
            all_probs[:, :self.num_background].sum(dim=1, keepdim=True),
            all_probs[:, self.num_background:].sum(dim=1, keepdim=True),
        ], dim=1)


        focal_score = self.alpha_d * \
            (1 - cls_score)**self.gamma_d * cls_score.log()
        loss_det = F.nll_loss(focal_score, label)


        return cls_score, loss_det

    def forward(self, inputs, roi_label=None):
        label = roi_label

        projected = oim2cq(inputs, label, self.lut, self.cq, self.cqb, self.cq_omega,
                           self.cqb_omega, self.omega_decay, momentum=self.momentum)
        projected *= self.oim_scalar


        inds = (label >= -1)
        label = label[inds]
        projected_non_bg = projected[inds.unsqueeze(1).expand_as(
            projected)].view(-1, (self.num_background + self.num_pids + self.num_unlabeled))
        projected_non_bg = projected_non_bg[:, self.num_background:]

        p_i = F.softmax(projected_non_bg, dim=1)
        focal_p_i = self.alpha_r * (1 - p_i)**self.gamma_r * p_i.log()

        loss_oim = F.nll_loss(focal_p_i, label, reduction='none',
                                ignore_index=-1)

        # size = batch_size * ()
        all_probs = F.softmax(projected, dim=1)
        # loss_detection
        cls_score, loss_det = self.build_det_loss(all_probs, roi_label)


        if self.dynamic_lambda:
            # Dynamic lambda
            cls_score_non_bg = cls_score[inds, 1].clone()  # p(\Lambda) only
            cls_score_non_bg = cls_score_non_bg.detach()
            cls_score_non_bg.requires_grad = False
            loss_oim *= cls_score_non_bg.pow(2)

        loss_oim = loss_oim.mean()
        return cls_score, loss_det, loss_oim

#        return cls_score, loss_det, None  # loss_oim