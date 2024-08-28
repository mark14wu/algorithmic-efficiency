"""U-Net Model.

Adapted from fastMRI:
https://github.com/facebookresearch/fastMRI/blob/main/fastmri/models/unet.py
"""

from functools import partial
from typing import Optional

import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F

from algorithmic_efficiency import init_utils

import os
# DEBUG = os.getenv('WUHAO_DEBUG')
DEBUG = False
# OPT_CHANNEL_LAST = os.getenv('OPT_CHANNEL_LAST')
OPT_CHANNEL_LAST = False

class UNet(nn.Module):
  r"""U-Net model from
    `"U-net: Convolutional networks
    for biomedical image segmentation"
    <hhttps://arxiv.org/pdf/1505.04597.pdf>`_.
    """

  def __init__(self,
               in_chans: int = 1,
               out_chans: int = 1,
               num_channels: int = 32,
               num_pool_layers: int = 4,
               dropout_rate: Optional[float] = 0.0,
               use_tanh: bool = False,
               use_layer_norm: bool = False) -> None:
    super().__init__()

    self.in_chans = in_chans
    self.out_chans = out_chans
    self.num_channels = num_channels
    self.num_pool_layers = num_pool_layers
    if dropout_rate is None:
      dropout_rate = 0.0
    self.down_sample_layers = nn.ModuleList([
        ConvBlock(in_chans,
                  num_channels,
                  dropout_rate,
                  use_tanh,
                  use_layer_norm)
    ])
    ch = num_channels
    for _ in range(num_pool_layers - 1):
      self.down_sample_layers.append(
          ConvBlock(ch, ch * 2, dropout_rate, use_tanh, use_layer_norm))
      ch *= 2
    self.conv = ConvBlock(ch, ch * 2, dropout_rate, use_tanh, use_layer_norm)

    self.up_conv = nn.ModuleList()
    self.up_transpose_conv = nn.ModuleList()

    for _ in range(num_pool_layers - 1):
      self.up_transpose_conv.append(
          TransposeConvBlock(ch * 2, ch, use_tanh, use_layer_norm))
      self.up_conv.append(
          ConvBlock(ch * 2, ch, dropout_rate, use_tanh, use_layer_norm))
      ch //= 2

    self.up_transpose_conv.append(
        TransposeConvBlock(ch * 2, ch, use_tanh, use_layer_norm))
    self.up_conv.append(
        nn.Sequential(
            ConvBlock(ch * 2, ch, dropout_rate, use_tanh, use_layer_norm),
            nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
        ))

    for m in self.modules():
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        init_utils.pytorch_default_init(m)

  def forward(self, x: Tensor) -> Tensor:
    stack = []
    output = x

    if DEBUG:
      print('==========0==========')
      print('stride:', output.stride())
      print('=====================')

    # apply down-sampling layers
    for layer in self.down_sample_layers:
      output = layer(output)

      # output = output.to(memory_format=torch.channels_last)

      if DEBUG:
        print('========1.1=======')
        print('stride:', output.stride())
        print('==================')
      stack.append(output)
      output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)
      if DEBUG:
        print('========1.2=======')
        print('stride:', output.stride())
        print('==================')

    output = self.conv(output)

    # apply up-sampling layers
    for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
      downsample_layer = stack.pop()
      output = transpose_conv(output)
      if DEBUG:
        print('========2=========')
        print('stride:', output.stride())
        print('==================')

      # reflect pad on the right/botton if needed to handle
      # odd input dimensions
      padding = [0, 0, 0, 0]
      if output.shape[-1] != downsample_layer.shape[-1]:
        padding[1] = 1  # padding right
      if output.shape[-2] != downsample_layer.shape[-2]:
        padding[3] = 1  # padding bottom
      if torch.sum(torch.tensor(padding)) != 0:
        output = F.pad(output, padding, "reflect")
        if DEBUG:
          print('========3=========')
          print('stride:', output.stride())
          print('===================')

      output = torch.cat([output, downsample_layer], dim=1)
      if DEBUG:
        print('==========4==========')
        print('stride:', output.stride())
        print('=====================')
      output = conv(output)
      if DEBUG:
        print('==========5==========')
        print('stride:', output.stride())
        print('=====================')

    return output


def _is_contiguous(tensor: torch.Tensor) -> bool:
  if torch.jit.is_scripting():
      return tensor.is_contiguous()
  else:
      return tensor.is_contiguous(memory_format=torch.contiguous_format)

# class LayerNorm2d(nn.LayerNorm):
#   r""" LayerNorm for channels_first tensors with 2d spatial dimensions (ie N, C, H, W).
#   """

#   def __init__(self, normalized_shape, eps=1e-6):
#       super().__init__(normalized_shape, eps=eps)

#   def forward(self, x) -> torch.Tensor:
#       if _is_contiguous(x):
#           # still faster than going to alternate implementation
#           # call contiguous at the end, because otherwise the rest of the model is computed in channels-last
#           return F.layer_norm(
#               x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2).contiguous()
#       elif x.is_contiguous(memory_format=torch.channels_last):
#           x = x.permute(0,2,3,1)
#           # trick nvfuser into picking up layer norm, even though it's a single op
#           # it's a slight pessimization (~.2%) if nvfuser is not enabled
#           x = F.layer_norm(
#               x, self.normalized_shape, self.weight, self.bias, self.eps) * 1.
#           return x.permute(0, 3, 1, 2)
#       else:
#           s, u = torch.var_mean(x, dim=1, unbiased=False, keepdim=True)
#           x = (x - u) * torch.rsqrt(s + self.eps)
#           x = x * self.weight[:, None, None] + self.bias[:, None, None]
#           return x

class InstanceNorm2d(nn.InstanceNorm2d):
  r""" InstanceNorm for channels_first tensors with 2d spatial dimensions (ie N, C, H, W).
  """

  def __init__(self, normalized_shape, eps=1e-6):
    super().__init__(normalized_shape, eps=eps)

  def forward(self, x) -> torch.Tensor:
      if _is_contiguous(x):
          # still faster than going to alternate implementation
          # call contiguous at the end, because otherwise the rest of the model is computed in channels-last
          return F.instance_norm(
              x.permute(0, 2, 3, 1), self.running_mean, self.running_var, self.weight, self.bias, self.training, self.momentum, self.eps).permute(0, 3, 1, 2).contiguous()
      elif x.is_contiguous(memory_format=torch.channels_last):
          x = x.permute(0,2,3,1)
          # trick nvfuser into picking up instance norm, even though it's a single op
          # it's a slight pessimization (~.2%) if nvfuser is not enabled
          x = F.instance_norm(
              x, self.running_mean, self.running_var, self.weight, self.bias, self.training, self.momentum, self.eps) * 1.
          return x.permute(0, 3, 1, 2)
      else:
          return super().forward(x)

class ConvBlock(nn.Module):
  # A Convolutional Block that consists of two convolution layers each
  # followed by instance normalization, LeakyReLU activation and dropout_rate.

  def __init__(self,
               in_chans: int,
               out_chans: int,
               dropout_rate: float,
               use_tanh: bool,
               use_layer_norm: bool) -> None:
    super().__init__()

    if use_layer_norm:
      if OPT_CHANNEL_LAST:
        raise NotImplementedError
      else:
        norm_layer = partial(nn.GroupNorm, 1, eps=1e-6)
    else:
      # norm_layer = nn.InstanceNorm2d
      if OPT_CHANNEL_LAST:
        norm_layer = partial(InstanceNorm2d, eps=1e-6)
      else:
        norm_layer = nn.InstanceNorm2d
    if use_tanh:
      activation_fn = nn.Tanh()
    else:
      activation_fn = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    self.conv_layers = nn.Sequential(
        nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
        norm_layer(out_chans),
        activation_fn,
        nn.Dropout2d(dropout_rate),
        nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
        norm_layer(out_chans),
        activation_fn,
        nn.Dropout2d(dropout_rate),
    )

  def forward(self, x: Tensor) -> Tensor:
    return self.conv_layers(x)


class TransposeConvBlock(nn.Module):
  # A Transpose Convolutional Block that consists of one convolution transpose
  # layers followed by instance normalization and LeakyReLU activation.

  def __init__(
      self,
      in_chans: int,
      out_chans: int,
      use_tanh: bool,
      use_layer_norm: bool,
  ):
    super().__init__()
    if use_tanh:
      activation_fn = nn.Tanh()
    else:
      activation_fn = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    if OPT_CHANNEL_LAST:
      norm_layer = partial(InstanceNorm2d, eps=1e-6)
    else:
      norm_layer = nn.InstanceNorm2d

    self.layers = nn.Sequential(
        nn.ConvTranspose2d(
            in_chans, out_chans, kernel_size=2, stride=2, bias=False),
        norm_layer(out_chans),
        activation_fn,
    )

  def forward(self, x: Tensor) -> Tensor:
    return self.layers(x)
