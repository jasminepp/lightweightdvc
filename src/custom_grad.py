import torch
from torch.autograd import Function


class STE(Function):
    """
    Straight Through Estimator.
    """

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class GradDecay(Function):
    """
    Gradient decay, which set the gradient of masked weights as ``beta``.
    ``beta`` decays in the range (0, 1) as the training progresses.
    """

    @staticmethod
    def forward(ctx, x, mask, beta):
        ctx.save_for_backward(mask, beta)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        mask, beta = ctx.saved_tensors
        grad = torch.where(mask == 0, beta, grad_output)
        return grad, None, None
