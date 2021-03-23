from __future__ import absolute_import

import torch
import torch.nn.functional as F
from torch import nn, autograd

from .oim_utils import tensor_gather
from .triplet_loss import TripletLossFilter

class OIMNew(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets,
                lut, cq, header, momentum):
        ctx.save_for_backward(inputs, targets,
                              lut, cq, header, momentum)
        outputs_labeled = inputs.mm(lut.t())
        outputs_unlabeled = inputs.mm(cq.t())

        pos_feats = lut.clone().detach()
        pos_idx = targets > 0
        pos_pids = targets[pos_idx]
        pos_feats = pos_feats[pos_pids]
        pos_feats.require_grad = False
        pos_pids.require_grad = False

        return torch.cat([outputs_labeled, outputs_unlabeled], dim=1), pos_feats, pos_pids

    @staticmethod
    def backward(ctx, grad_outputs, grad_feat, grad_pids):
        inputs, targets, \
            lut, cq, header, momentum = ctx.saved_tensors

        inputs, targets = \
            tensor_gather((inputs, targets))

        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(
                torch.cat([lut, cq], dim=0))
            if grad_inputs.dtype == torch.float16:
                grad_inputs = grad_inputs.to(torch.float32)

        for x, y in zip(inputs, targets):
            if y < len(lut):
                lut[y] = momentum * \
                    lut[y] + (1. - momentum) * x
                #lut[y] /= lut[y].norm()
            else:
                cq[header][:64] = x[:64]
                header = (header + 1) % cq.size(0)
        return grad_inputs, None, None, None, None, None


def oimnew(inputs, targets, lut, cq, header, momentum=0.5):
    return OIMNew.apply(inputs, targets,
                     lut, cq,
                     torch.tensor(header), torch.tensor(momentum))

class OIMLossNewFocal(nn.Module):
    """docstring for OIMLoss"""

    def __init__(self, num_features, num_pids, num_cq_size,
                 oim_momentum, oim_scalar):
        super(OIMLossNewFocal, self).__init__()
        self.num_features = num_features
        self.num_pids = num_pids
        self.num_unlabeled = num_cq_size
        self.momentum = oim_momentum
        self.oim_scalar = oim_scalar

        self.register_buffer('lut', torch.zeros(
            self.num_pids, self.num_features))
        self.register_buffer('cq',  torch.zeros(
            self.num_unlabeled, self.num_features))

        self.header_cq = 0
        self.tri_loss = TripletLossFilter()

    def forward(self, inputs, roi_label):
        # merge into one batch, background label = 0
        #targets = torch.cat(roi_label)
        targets = roi_label
        label = roi_label
        #label = targets  - 1  # background label = -1

        inds = (label >= -1)
        label = label[inds]
        inputs = inputs[inds.unsqueeze(1).expand_as(
            inputs)].view(-1, self.num_features)

        projected, labeled_matching_reid, labeled_matching_ids = oimnew(inputs, label, self.lut, self.cq,
                        self.header_cq, momentum=self.momentum)
        projected *= self.oim_scalar

        self.header_cq = ((self.header_cq +
                           (label == -1).long().sum().item()) %
                          self.num_unlabeled)
        #loss_oim = F.cross_entropy(projected, label,
         #                          ignore_index=-1)
        p_i = F.softmax(projected, dim=1)
        #focal_p_i = 0.25 * (1 - p_i)**2 * p_i.log()
        focal_p_i = (1 - p_i)**2 * p_i.log()

        loss_oim = F.nll_loss(focal_p_i, label, reduction='none',
                                  ignore_index=-1)

        pos_reid = torch.cat((inputs, labeled_matching_reid), dim=0)
        pid_labels = torch.cat((label, labeled_matching_ids), dim=0)
        #loss_tri = 1e-8*self.tri_loss(pos_reid, pid_labels)
        loss_tri = self.tri_loss(pos_reid, pid_labels)

        return loss_oim, loss_tri



class OIMSMRNew(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets,
                lut, cq, cq_omega,
                omega_decay, momentum):
        outputs_labeled = inputs.mm(lut.t())
        outputs_unlabeled = inputs.mm(cq.t())

        omega_x_unlabeled = outputs_labeled.max(dim=1)[0] /  \
            outputs_unlabeled.max(dim=1)[0]

        ctx.save_for_backward(inputs, targets, omega_x_unlabeled,
                              lut, cq, cq_omega,
                              omega_decay, momentum)

        pos_feats = lut.clone().detach()
        pos_idx = targets > 0
        pos_pids = targets[pos_idx]
        pos_feats = pos_feats[pos_pids]
        pos_feats.require_grad = False
        pos_pids.require_grad = False

        return torch.cat([outputs_labeled, outputs_unlabeled], dim=1), pos_feats, pos_pids

    @staticmethod
    def backward(ctx, grad_outputs, grad_feat, grad_pids):
        inputs, targets, omega_x_unlabeled, \
            lut, cq, cq_omega, \
            omega_decay, momentum = ctx.saved_tensors

        inputs, targets, omega_x_unlabeled = \
            tensor_gather((inputs, targets, omega_x_unlabeled))

        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(
                torch.cat([lut, cq], dim=0))

        for x, y, omega in zip(inputs, targets, omega_x_unlabeled):
            if y < len(lut):
                lut[y] = momentum * \
                    lut[y] + (1. - momentum) * x
                #lut[y] /= lut[y].norm()
            else:
                if omega < cq_omega.min():
                    continue
                else:
                    header_cq = cq_omega.argmin().item()
                    cq[header_cq][:64] = x[:64]
                    cq_omega[header_cq] = omega

        cq_omega *= omega_decay

        return grad_inputs, None, None, None, None, None, None


def oimsmrnew(inputs, targets, lut, cq, cq_omega, omega_decay=0.99, momentum=0.5):
    return OIMSMRNew.apply(inputs, targets,
                        lut, cq, cq_omega,
                        torch.tensor(omega_decay), torch.tensor(momentum))


class OIMLossSMRNewFocal(nn.Module):

    def __init__(self, num_features, num_pids, num_cq_size,
                 oim_momentum, oim_scalar,
                 omega_decay=0.99):
        super(OIMLossSMRNewFocal, self).__init__()
        self.num_features = num_features
        self.num_pids = num_pids
        self.num_unlabeled = num_cq_size
        self.momentum = oim_momentum
        self.omega_decay = omega_decay
        self.oim_scalar = oim_scalar
        self.tri_loss = TripletLossFilter()

        self.register_buffer('lut', torch.zeros(
            self.num_pids, self.num_features))
        self.register_buffer('cq',  torch.zeros(
            self.num_unlabeled, self.num_features))
        self.register_buffer('cq_omega', torch.zeros(
            self.num_unlabeled))

    def forward(self, inputs, roi_label):
        # merge into one batch, background label = 0
        #targets = torch.cat(roi_label)
        #label = targets - 1  # background label = -1
        label = roi_label

        inds = (label >= -1)
        label = label[inds]
        inputs = inputs[inds.unsqueeze(1).expand_as(
            inputs)].view(-1, self.num_features)

        projected, labeled_matching_reid, labeled_matching_ids = oimsmrnew(inputs, label, self.lut, self.cq,
                           self.cq_omega, self.omega_decay, momentum=self.momentum)
        projected *= self.oim_scalar
        #focal loss
        p_i = F.softmax(projected, dim=1)
        #focal_p_i = 0.25 * (1 - p_i)**2 * p_i.log()
        focal_p_i = (1 - p_i)**2 * p_i.log()

        loss_oim = F.nll_loss(focal_p_i, label, reduction='none',
                                  ignore_index=-1)

        pos_reid = torch.cat((inputs, labeled_matching_reid), dim=0)
        pid_labels = torch.cat((label, labeled_matching_ids), dim=0)
        #loss_tri = 1e-8*self.tri_loss(pos_reid, pid_labels)
        loss_tri = self.tri_loss(pos_reid, pid_labels)

        #loss_oim = F.cross_entropy(projected, label,
        #                           ignore_index=-1)
        return loss_oim, loss_tri
