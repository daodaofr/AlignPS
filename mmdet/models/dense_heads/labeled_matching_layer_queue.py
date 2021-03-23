import torch
import torch.nn as nn
from torch.autograd import Function


class LabeledMatching(Function):
    @staticmethod
    def forward(ctx, features, pid_labels, lookup_table, momentum=0.5):
        # The lookup_table can't be saved with ctx.save_for_backward(), as we would
        # modify the variable which has the same memory address in backward()
        ctx.save_for_backward(features, pid_labels)
        ctx.lookup_table = lookup_table
        ctx.momentum = momentum

        scores = features.mm(lookup_table.t())
        #print(features, lookup_table, scores)
        pos_feats = lookup_table.clone().detach()
        pos_idx = pid_labels > 0
        pos_pids = pid_labels[pos_idx]
        pos_feats = pos_feats[pos_pids]
        #pos_feats.require_grad = False
        
        return scores, pos_feats, pos_pids

    @staticmethod
    def backward(ctx, grad_output, grad_feat, grad_pids):
        features, pid_labels = ctx.saved_tensors
        lookup_table = ctx.lookup_table
        momentum = ctx.momentum

        grad_feats = None
        if ctx.needs_input_grad[0]:
            grad_feats = grad_output.mm(lookup_table)

        # Update lookup table, but not by standard backpropagation with gradients
        for indx, label in enumerate(pid_labels):
            if label >= 0:
                lookup_table[label] = (
                    momentum * lookup_table[label] + (1 - momentum) * features[indx]
                )
                #lookup_table[label] /= lookup_table[label].norm()

        return grad_feats, None, None, None


class LabeledMatchingLayerQueue(nn.Module):
    """
    Labeled matching of OIM loss function.
    """

    def __init__(self, num_persons=5532, feat_len=256):
        """
        Args:
            num_persons (int): Number of labeled persons.
            feat_len (int): Length of the feature extracted by the network.
        """
        super(LabeledMatchingLayerQueue, self).__init__()
        self.register_buffer("lookup_table", torch.zeros(num_persons, feat_len))

    def forward(self, features, pid_labels):
        """
        Args:
            features (Tensor[N, feat_len]): Features of the proposals.
            pid_labels (Tensor[N]): Ground-truth person IDs of the proposals.

        Returns:
            scores (Tensor[N, num_persons]): Labeled matching scores, namely the similarities
                                             between proposals and labeled persons.
        """
        scores, pos_feats, pos_pids = LabeledMatching.apply(features, pid_labels, self.lookup_table)
        return scores, pos_feats, pos_pids