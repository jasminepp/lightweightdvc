## (PCS2024) Accelerating Learnt Video Codecs with Gradient Decay and Layer-wise Distillation [arXiV](https://arxiv.org/abs/2312.02605)

## contributions

We propose two optimization strategies that enable better rate-distortion performance retainment from structured pruning, whilst noticably reducing the decoding latency & complexity.

1. We propose a novel **gradient decay** gradient approximator as a drop-in replacement for the commonly used STE proxy, where the gradient is progressively annealed from 1.0 to 0.0 (for pruned weights) to enable free exploration in the parameter space in early stage of training and avoid forward-backward mismatching when the pruning pattern is fixed.

2. We propose to adopt layer-wise distillation as auxiliary loss to support pruning with the multi-stage training strategy (*e.g.* motion estimation $\Rightarrow$ residual coding $\Rightarrow$ full model optimisation) by distilling the knowledge from the pre-trained, dense teacher model to the student model.

Please refer to the paper for more details.

## usage
We provied an implementation of the gradient decay approximator with customised autograd function, and a collection of pruners for different building blocks (convolution layer, deconvolution layer, residual blocks, etc.). The pruner automatically wraps the corresponding module of a potentially complicated network and is able to prune in both forward and backward pass.

## dependencies
```
numpy>=1.20.0
scipy
matplotlib
torch>=1.7.0
pytorch-msssim==0.2.0
tensorboard
torchvision
tqdm
bd-metric
ptflops
```

## performance
<figure>
  <img src="https://github.com/ge1-gao/lightweight-video-compression/assets/119049371/cc6828ac-ed96-4b9a-8b33-2fd429816fe9" alt="performance versus receiver compute" width="700" height="400">
  <figcaption>Figure 1: Performance versus receiver compute</figcaption>
</figure>

<figure>
  <img src="https://github.com/ge1-gao/lightweight-video-compression/assets/119049371/acd5b3c2-dbb4-413a-ba94-4dc41956494f" alt="RD Plot UVG">
  <figcaption>Figure 2: Rate-Distortion Plot for UVG Dataset</figcaption>
</figure>

<figure>
  <img src="https://github.com/ge1-gao/lightweight-video-compression/assets/119049371/a4a17774-50bf-4c6a-bc76-c7701dd0d3b7" alt="RD Plot MCL">
  <figcaption>Figure 3: Rate-Distortion Plot for MCL-JCV Dataset</figcaption>
</figure>

## next steps
Currently, the actual acceleration is achieved via "**hand-crafted**" architectural refactoring after the masks have been determined, which is highly time-consuming and not extendable to more sophisticated architectures. Further, the structured filter pruning is too coarsed-grained, leading to non-trivial performance loss despite better optimization strategies. Hence, we are actively working on the followings:

- [ ] build dependency graph (e.g. [Torch-Pruning](https://github.com/VainF/Torch-Pruning)) to support automated structured pruning of common neural video compression and implicit video compression models
- [ ] further support N:M sparsity and quantization to enable finer-grained complexity reduction

Stay tuned!

# citation
Please consider citing our work if you find that it is useful. Much appreciated!

```
@inproceedings{peng2024accelerating,
  title       = {Accelerating learnt video codecs with gradient decay and layer-wise distillation},
  author      = {Peng, Tianhao and Gao, Ge and Sun, Heming and Zhang, Fan and Bull, David},
  booktitle   = {2024 Picture Coding Symposium (PCS)},
  pages       = {1--5},
  year        = {2024},
  organization= {IEEE}
}
```
