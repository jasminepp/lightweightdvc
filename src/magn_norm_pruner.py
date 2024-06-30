import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import List
from .custom_grad import *
import torch.nn.init as init
from torchmetrics import CosineSimilarity
from src.layers import SubpelConv


cosineSim = CosineSimilarity(reduction="mean")
eps = 1e-7


class ConvPruner(nn.Module):
    """
    A customised magnitude-based channel-wise pruner.
    Args:
        - layer (nn.Module): The convolutional layer to be pruned.
        - normType (str): The type of norm to be used for magnitude comparison ("L1" or "L2").
        - pruningMode (str): The pruning mode, inclduing:
            - STE: approximate the gradient via Straigh Through Estimation
            - weightDecay: decay the masked weights with a gradually annealing ``beta``
            - gradientDecay: anneal the STE's gradient (direct copy) with ``beta`` to better approximate masking
        - temperature: used to control the steepness of Sigmod(·) for approximating the sparsity
    """

    def __init__(self, layer, normType="L2", pruningMode="STE", temperature=0.4, enableDistill=False):
        super().__init__()
        assert pruningMode in [
            "STE",
            "weightDecay",
            "gradientDecay",
        ], f"Unsupported pruning mode {pruningMode}!"

        self.layer = layer
        self._normType = normType
        self._pruningMode = pruningMode
        self._temperature = temperature
        self._enableDistill = False

        self.GRADIENT_FUNCTIONS = {
            "STE": {"fn": STE.apply, "kwargs": {}},
            "gradientDecay": {
                "fn": GradDecay.apply,
                "kwargs": {"beta": None, "mask": None},
            },
        }
        self.grad_fn = self.GRADIENT_FUNCTIONS[pruningMode]["fn"]

        # sparsification constraints
        self.penalty = 0.0
        self.sparsity = 0.0
        self.update_mask_pre = True

        # params for convolution layer
        self.stride = layer.stride
        self.padding = layer.padding
        if isinstance(layer, nn.ConvTranspose2d):
            self.output_padding = layer.output_padding
        self.dilation = layer.dilation
        self.groups = layer.groups

        # keep the original weight and bias in CPU unless required for computing the distillation loss
        self.weight_prune = nn.Parameter(layer.weight.data.clone())
        if layer.bias is not None:
            self.bias_prune = nn.Parameter(layer.bias.data.clone())
        else:
            self.bias_prune = None

        # register the buffer and the forward pre-hooks
        self.register_buffer("mask", None)
        self.register_buffer(
            "beta", torch.tensor(1.0, requires_grad=False)
        )  # an annealing factor for progressive training

        # initialize the value of threshold
        norms_val = self.calc_norms()
        max_norm = norms_val.max().item()
        init_threshold = 0.1 * max_norm
        self.threshold = nn.Parameter(torch.tensor(init_threshold), requires_grad=True)

    def calc_norms(self):
        if self._normType == "L1":
            norms_val = self.weight_prune.abs().sum(dim=(1, 2, 3))
        elif self._normType == "L2":
            norms_val = self.weight_prune.pow(2).sum(dim=(1, 2, 3)).sqrt()
        return norms_val

    def get_mask_and_distance(self):
        distance = self.compute_mask()
        self.compute_sparsity(distance)

    def compute_mask(self):
        # update or register the norms of the current weight
        norms_val = self.calc_norms()
        distance = norms_val - self.threshold

        mask = torch.where(distance <= 0, 0.0, 1.0)
        self.mask = mask.view(-1, 1, 1, 1)
        return distance

    def set_beta(self, new_beta):
        self.beta = torch.tensor(new_beta)

    def sigmoid(self, x, temperature):
        return 1 / (1 + torch.exp(-x / temperature))

    def compute_sparsity(self, distance):
        """
        Calculates the sparsity of the current layer as
        the ratio between the (soft) masked weights and the original weights.
        Note that, sparsity cannot be enforced with hard mask as
        the gradient flow from sparsity loss and threshold & weight is cut-off
        in this case, we induce sparsity with a sigmoid-based proxy function
        """
        self.sparsity = 1.0 - self.sigmoid(distance, self._temperature).mean()

    def forward(self, x):
        if self.update_mask_pre:
            self.get_mask_and_distance()
        pruned_weight = self.weight_prune * self.mask

        if self._pruningMode == "gradientDecay":
            self.GRADIENT_FUNCTIONS[self._pruningMode]["kwargs"]["beta"] = self.beta
            self.GRADIENT_FUNCTIONS[self._pruningMode]["kwargs"]["mask"] = self.mask
        kwargs = self.GRADIENT_FUNCTIONS[self._pruningMode]["kwargs"]

        pruned_weight = self.GRADIENT_FUNCTIONS[self._pruningMode]["fn"](
            pruned_weight, **kwargs
        )

        if isinstance(self.layer, nn.Conv2d):
            pruned_bias = self.bias_prune * self.mask.squeeze()
            pruned_bias = self.GRADIENT_FUNCTIONS[self._pruningMode]["fn"](
                pruned_bias, **kwargs
            )
            conv_out = F.conv2d(
                x,
                pruned_weight,
                pruned_bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        elif isinstance(self.layer, nn.ConvTranspose2d):
            conv_out = F.conv_transpose2d(
                x,
                pruned_weight,
                self.bias_prune,
                self.stride,
                self.padding,
                self.output_padding,
                self.groups,
                self.dilation,
            )
        else:
            raise ValueError(f"The type of the layer <{type(self.layer)}> is WRONG!")

        # ********** if distillation is enabled for the layer **********
        if self._enableDistill:
            with torch.no_grad():
                orig_out = self.layer(x)
            norm_conv_out = conv_out / (
                torch.linalg.norm(conv_out, dim=1, ord=2, keepdim=True) + eps
            )
            norm_orig_out = orig_out / (
                torch.linalg.norm(orig_out, dim=1, ord=2, keepdim=True) + eps
            )
            self.distill_loss = torch.mean(F.mse_loss(norm_conv_out, norm_orig_out))

        return conv_out


class SubpelConvPruner(ConvPruner):
    def __init__(self, block, r=2, normType="L2", pruningMode="STE"):
        super(SubpelConvPruner, self).__init__(
            layer=list(block.children())[0], normType=normType, pruningMode=pruningMode
        )
        self.r = r
        self.block = block

        assert (
            self.weight_prune.shape[0] % (r**2) == 0
        ), "Mismatched channels for pixel-shuffle."
        self.shuffle_layer = list(block.children())[1]
        assert isinstance(
            self.shuffle_layer, nn.PixelShuffle
        ), "Expected the 2nd layer to be `nn.PixelShuffle`"

        # initialize the value of threshold
        norms_val = self.calc_norms()
        reshaped_norms = norms_val.view(-1, self.r**2)
        agg_norms = reshaped_norms.mean(dim=1)
        max_norm = agg_norms.max().item()
        init_threshold = 0.1 * max_norm
        self.threshold = nn.Parameter(torch.tensor(init_threshold), requires_grad=True)

    def compute_mask(self):
        """update the soft mask based on current threshold."""
        norms_val = self.calc_norms()
        reshaped_norms = norms_val.view(-1, self.r**2)
        agg_norms = reshaped_norms.mean(dim=1)

        # calculation of the current mask
        distance = agg_norms - self.threshold

        mask = torch.where(distance <= 0, 0.0, 1.0)
        self.mask = mask.repeat_interleave(self.r**2).view(-1, 1, 1, 1)  # (C, 1, 1, 1)
        return distance

    def forward(self, x):
        # Structured decay in convolutional weights
        if self.update_mask_pre:
            self.get_mask_and_distance()
        pruned_weight = self.weight_prune * self.mask
        pruned_bias = self.bias_prune * self.mask.squeeze()

        if self._pruningMode == "gradientDecay":
            self.GRADIENT_FUNCTIONS[self._pruningMode]["kwargs"]["beta"] = self.beta
            self.GRADIENT_FUNCTIONS[self._pruningMode]["kwargs"]["mask"] = self.mask
        kwargs = self.GRADIENT_FUNCTIONS[self._pruningMode]["kwargs"]

        # Apply the gradient function
        pruned_weight = self.GRADIENT_FUNCTIONS[self._pruningMode]["fn"](
            pruned_weight, **kwargs
        )
        pruned_bias = self.GRADIENT_FUNCTIONS[self._pruningMode]["fn"](
            pruned_bias, **kwargs
        )

        conv_out = F.conv2d(
            x,
            pruned_weight,
            pruned_bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        shuff_out = self.shuffle_layer(conv_out)

        # ********** if distillation is enabled for the layer ********** #
        if self._enableDistill:
            with torch.no_grad():
                orig_out = self.block(x)
            norm_shuff_out = shuff_out / (
                torch.linalg.norm(shuff_out, dim=1, ord=2, keepdim=True) + eps
            )
            norm_orig_out = orig_out / (
                torch.linalg.norm(orig_out, dim=1, ord=2, keepdim=True) + eps
            )
            self.distill_loss = torch.mean(F.mse_loss(norm_shuff_out, norm_orig_out))

        return shuff_out


class ResBlockPruner(ConvPruner):
    def __init__(self, block, normType="L2", pruningMode="STE"):
        # Extract the first convolutional layer for initialization in the parent class
        all_layers = self.get_all_layers(block)
        layer = next(layer for layer in all_layers if isinstance(layer, nn.Conv2d))
        super(ResBlockPruner, self).__init__(
            layer=layer, normType=normType, pruningMode=pruningMode
        )

        # wrap each convolution layer in the ResBlock with a ConvPruner
        self.block = block
        self.conv_pruner_1 = ConvPruner(self.block.conv1)
        self.conv_pruner_2 = ConvPruner(self.block.conv2)
        # self.conv_pruner_2.update_mask_pre = False

    def get_all_layers(self, block):
        """Recursively extract all layers from nested children of the block."""
        layers = []
        for child in block.children():
            if isinstance(child, (nn.Sequential, nn.ModuleList, nn.ModuleDict)):
                layers.extend(self.get_all_layers(child))
            else:
                layers.append(child)
        return layers

    def forward(self, x):
        identity = x
        self.conv_pruner_2.get_mask_and_distance()
        x = x * self.conv_pruner_2.mask.permute(1, 0, 2, 3).contiguous().expand(
            x.size(0), -1, -1, -1
        )

        out = self.block.first_layer(x)
        out = self.conv_pruner_1(out)
        out = self.block.relu(out)
        out = self.conv_pruner_2(out)
        out = self.block.last_layer(out)
        # identity = identity* self.conv_pruner_2.mask.clone().permute(1, 0, 2, 3).contiguous().expand(identity.size(0), -1, -1, -1)

        out += identity

        if self._enableDistill:
            with torch.no_grad():
                orig_out = self.block(x)
            norm_out = out / torch.linalg.norm(out, dim=1, ord=2, keepdim=True)
            norm_orig_out = orig_out / torch.linalg.norm(
                orig_out, dim=1, ord=2, keepdim=True
            )
            self.distill_loss = torch.mean(F.mse_loss(norm_out, norm_orig_out))
        return out


class ResidualBlockPrunerUpsample(ConvPruner):
    def __init__(self, block, normType="L2", pruningMode="STE"):
        # Extract the first convolutional layer for initialization in the parent class
        all_layers = self.get_all_layers(block)
        layer = next(layer for layer in all_layers if isinstance(layer, nn.Conv2d))
        super(ResidualBlockPrunerUpsample, self).__init__(
            layer=layer, normType=normType, pruningMode=pruningMode
        )

        # wrap each convolution layer in the ResBlock with a ConvPruner
        self.block = block
        self.subpel_conv_pruner = SubpelConvPruner(self.block.subpel_conv)
        self.upsample_pruner = SubpelConvPruner(self.block.upsample)
        self.upsample_pruner.update_mask_pre = False

    def get_all_layers(self, block):
        """Recursively extract all layers from nested children of the block."""
        layers = []
        for child in block.children():
            if isinstance(child, (nn.Sequential, nn.ModuleList, nn.ModuleDict)):
                layers.extend(self.get_all_layers(child))
            else:
                layers.append(child)
        return layers

    def forward(self, x):
        identity = x
        out = self.subpel_conv_pruner(x)
        out = self.block.leaky_relu(out)
        out = self.block.conv(out)
        out = self.block.leaky_relu(out)
        self.upsample_pruner.mask = self.subpel_conv_pruner.mask.clone()
        identity = self.upsample_pruner(x)
        out += identity

        if self._enableDistill:
            with torch.no_grad():
                orig_out = self.block(x)
            norm_out = out / (torch.linalg.norm(out, dim=1, ord=2, keepdim=True) + eps)
            norm_orig_out = orig_out / (
                torch.linalg.norm(orig_out, dim=1, ord=2, keepdim=True) + eps
            )
            self.distill_loss = torch.mean(F.mse_loss(norm_out, norm_orig_out))

        return out


class ResidualBlockPruner(ConvPruner):
    def __init__(self, block, normType="L2", pruningMode="STE"):
        # Extract the first convolutional layer for initialization in the parent class
        all_layers = self.get_all_layers(block)
        layer = next(layer for layer in all_layers if isinstance(layer, nn.Conv2d))
        super(ResidualBlockPruner, self).__init__(
            layer=layer, normType=normType, pruningMode=pruningMode
        )

        # wrap each convolution layer in the ResBlock with a ConvPruner
        self.block = block
        self.conv_pruner_1 = ConvPruner(self.block.conv1)
        self.conv_pruner_2 = ConvPruner(self.block.conv2)
        self.conv_pruner_2.update_mask_pre = False

    def get_all_layers(self, block):
        """Recursively extract all layers from nested children of the block."""
        layers = []
        for child in block.children():
            if isinstance(child, (nn.Sequential, nn.ModuleList, nn.ModuleDict)):
                layers.extend(self.get_all_layers(child))
            else:
                layers.append(child)
        return layers

    def forward(self, x):
        identity = x
        self.conv_pruner_2.get_mask_and_distance()
        x = x * self.conv_pruner_2.mask.permute(1, 0, 2, 3).contiguous().expand(
            x.size(0), -1, -1, -1
        )
        out = self.conv_pruner_1(x)
        out = self.block.leaky_relu(out)
        out = self.conv_pruner_2(out)
        out = self.block.leaky_relu(out)
        # identity = identity* self.conv_pruner_2.mask.clone().permute(1, 0, 2, 3).contiguous().expand(identity.size(0), -1, -1, -1)

        out += identity

        if self._enableDistill:
            with torch.no_grad():
                orig_out = self.block(x)
            norm_out = out / (torch.linalg.norm(out, dim=1, ord=2, keepdim=True) + eps)
            norm_orig_out = orig_out / (
                torch.linalg.norm(orig_out, dim=1, ord=2, keepdim=True) + eps
            )
            self.distill_loss = torch.mean(F.mse_loss(norm_out, norm_orig_out))

        return out
