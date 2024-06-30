"""
Utils for automatically wrapping the submodules of a model for structured pruning.
"""

import torch
import torch.nn as nn
from src.layers import ResidualBlockUpsample, SubpelConv, ResidualBlock, ResBlock

from src.magn_norm_pruner import (
    ConvPruner as MagnNormConvPruner,
    SubpelConvPruner as MagnNormSubpelConvPruner,
    ResBlockPruner as MagnNormResBlockPruner,
    ResidualBlockPruner as MagnNormResidualBlockPruner,
    ResidualBlockPrunerUpsample as MagnNormResBlockUpsamplePruner,
)


# ******************* HELPERs for wrapping *******************
def wrap_modules(
    module: nn.Module,
    excluded_modules=[],
    pruner_type="magnitude",
    normType="L2",
    pruned_modules=None,
    parent_name="",
    debug=False,
):
    """Recursively wrap the sub-modules to insert the pruning & quantisation mask."""
    if pruned_modules is None:
        pruned_modules = []

    if pruner_type == "magnitude":
        conv_pruner = MagnNormConvPruner
        subpel_pruner = MagnNormSubpelConvPruner
        resblock_pruner = MagnNormResBlockPruner
        residualblock_pruner = MagnNormResidualBlockPruner
        residualblockupsample_pruner = MagnNormResBlockUpsamplePruner
    else:
        raise ValueError("Invalid pruner_type")

    for name, child in module.named_children():
        full_name = (
            parent_name + "." + name if parent_name else name
        )  # construct the full path name
        if name in excluded_modules:
            if debug:
                print(f"Skip wrapping for module:: {name}")
            continue

        if isinstance(child, (ResidualBlock)):
            # wrap the residual blocks
            if pruner_type == "magnitude":
                wrapper = residualblock_pruner(child, normType=normType)
            else:
                wrapper = residualblock_pruner(child)
            setattr(module, name, wrapper)
            pruned_modules.append((full_name, wrapper))

        elif isinstance(child, (ResidualBlockUpsample)):
            # wrap the residualUpsample blocks
            if pruner_type == "magnitude":
                wrapper = residualblockupsample_pruner(child, normType=normType)
            else:
                wrapper = residualblockupsample_pruner(child)
            setattr(module, name, wrapper)
            pruned_modules.append((full_name, wrapper))

        elif isinstance(child, (ResBlock)):
            # wrap the Resblocks
            if pruner_type == "magnitude":
                wrapper = resblock_pruner(child, normType=normType)
            else:
                wrapper = resblock_pruner(child)
            setattr(module, name, wrapper)
            pruned_modules.append((full_name, wrapper))
        elif isinstance(child, SubpelConv):
            if pruner_type == "magnitude":
                wrapper = subpel_pruner(child, normType=normType)
            else:
                wrapper = subpel_pruner(child)
            setattr(module, name, wrapper)
            pruned_modules.append((full_name, wrapper))
        elif isinstance(child, (nn.Conv2d, nn.ConvTranspose2d)):
            if pruner_type == "magnitude":
                wrapper = conv_pruner(child, normType=normType)
            else:
                wrapper = conv_pruner(child)
            setattr(module, name, wrapper)
            pruned_modules.append((full_name, wrapper))
        else:
            # recursively process this child's children
            wrap_modules(
                child,
                excluded_modules,
                pruner_type,
                normType,
                pruned_modules,
                full_name,
            )
    return pruned_modules


def wrap_subpelconv(module: nn.Module):
    """Modify in-place the `subpel_conv3x3` layers."""
    for name, child in module.named_children():
        if any(
            isinstance(superchild, nn.PixelShuffle)
            for superchild in list(child.children())
        ):
            c_in = child[0].in_channels
            c_out = (
                child[0].out_channels // 4
            )  # PixelShuffule for r=2 reduces channels for factor of 4
            r = 2

            new_layer = SubpelConv(c_in, c_out, r)
            new_layer.conv.weight.data = child[0].weight.data
            if new_layer.conv.bias is not None:
                new_layer.conv.bias.data = child[0].bias.data
            else:
                new_layer.conv.bias = None

            # replace the old layer with the new one
            setattr(module, name, new_layer)
