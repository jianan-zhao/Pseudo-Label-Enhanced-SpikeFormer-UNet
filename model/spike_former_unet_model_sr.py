import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Callable
from spikingjelly.activation_based import neuron, functional, surrogate, layer, base
from timm.layers import trunc_normal_, DropPath

class GeneralParametricLIFNode(neuron.BaseNode):
    """
    General Parametric LIF Neuron Model with learnable time constant and threshold.
    """
    def __init__(self,
                 init_tau: float = 2.0,
                 init_threshold: float = 1.0,
                 learnable_tau: bool = True,
                 learnable_threshold: bool = True,
                 decay_input: bool = True,
                 v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False,
                 step_mode='s',
                 backend='cupy',
                 store_v_seq: bool = False,
                 visualise: bool = False):

        super().__init__(v_threshold=0., 
                         v_reset=v_reset,
                         surrogate_function=surrogate_function,
                         detach_reset=detach_reset,
                         step_mode=step_mode,
                         backend=backend,
                         store_v_seq=store_v_seq)

        self.decay_input = decay_input
        self.visualise = visualise

        # τ = 1 / sigmoid(w)
        init_w = -math.log(init_tau - 1.)
        w_tensor = torch.tensor(init_w, dtype=torch.float)
        if learnable_tau:
            self.w = nn.Parameter(w_tensor)
        else:
            self.register_buffer('w', w_tensor)

        # Learnable or fixed threshold
        threshold_tensor = torch.tensor(init_threshold, dtype=torch.float)
        if learnable_threshold:
            self.v_threshold = nn.Parameter(threshold_tensor)
        else:
            self.register_buffer('v_threshold', threshold_tensor)
            
        # -----------------------
        # buffer save spike and v at each time step for visualization
        # -----------------------
        if self.visualise:
            self.spike_record = None  # [T, N]
            self.v_record = None      # [T, N]

    @property
    def supported_backends(self):
        """Returns the supported backends based on the step mode."""
        return ('torch',) if self.step_mode == 's' else ('torch', 'cupy')

    def extra_repr(self):
        """Returns a string representation of the neuron with its parameters."""
        with torch.no_grad():
            tau = 1. / self.w.sigmoid()
        return super().extra_repr() + \
               f', tau={tau.item():.4f}, learnable_tau={isinstance(self.w, nn.Parameter)}, ' \
               f'threshold={self.v_threshold.item():.4f}, learnable_threshold={isinstance(self.v_threshold, nn.Parameter)}'

    def neuronal_charge(self, x: torch.Tensor):
        """Updates the membrane potential based on the input current and decay."""
        tau_inv = self.w.sigmoid()

        if self.decay_input:
            if self.v_reset is None or self.v_reset == 0.:
                self.v = self.v + (x - self.v) * tau_inv
            else:
                self.v = self.v + (x - (self.v - self.v_reset)) * tau_inv
        else:
            if self.v_reset is None or self.v_reset == 0.:
                self.v = self.v * (1. - tau_inv) + x
            else:
                self.v = self.v - (self.v - self.v_reset) * tau_inv + x

    def neuronal_fire(self):
        """Computes the spike output based on the membrane potential and threshold."""
        if self.visualise:
            spike = self.surrogate_function(self.v - self.v_threshold)

            # Save spike and v at each time step for visualization
            spike_cpu = spike.detach().cpu()
            v_cpu = self.v.detach().cpu()
            if self.spike_record is None:
                self.spike_record = spike_cpu.unsqueeze(0)  # [1, N]
                self.v_record = v_cpu.unsqueeze(0)          # [1, N]
            else:
                self.spike_record = torch.cat([self.spike_record, spike_cpu.unsqueeze(0)], dim=0)
                self.v_record = torch.cat([self.v_record, v_cpu.unsqueeze(0)], dim=0)

            return spike
        else:
            return self.surrogate_function(self.v - self.v_threshold)
    
    
class InstanceNorm(nn.InstanceNorm3d, base.StepModule):
    """
    Spiking InstanceNorm3d layer supporting single-step and multi-step modes.
    """
    def __init__(
            self,
            num_features: int,
            eps: float = 1e-5,
            momentum: float = 0.1,
            affine: bool = True,
            track_running_stats: bool = False,
            step_mode: str = 's'
    ):
        """
        * :ref:`API in English <InstanceNorm-en>`

        .. _InstanceNorm-en:

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Refer to :class:`torch.nn.InstanceNorm3d` for other parameters' API
        """
        super().__init__(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
        self.step_mode = step_mode

    def extra_repr(self):
        """Returns a string representation of the InstanceNorm layer with its step mode."""
        return super().extra_repr() + f', step_mode={self.step_mode}'

    def forward(self, x: Tensor):
        """Forward pass for the InstanceNorm layer."""
        if self.step_mode == 's':
            return super().forward(x)
        elif self.step_mode == 'm':
            return functional.seq_to_ann_forward(x, super().forward)
        else:
            raise ValueError(f"Unknown step_mode: {self.step_mode}")
        
        

class NormAndPad3DLayer(nn.Module):
    def __init__(self, pad_voxels, num_features, num_groups=8, step_mode='m', **norm_kwargs):
        """
        Support BatchNorm3d and GroupNorm 3D normalization + Padding combination module.

        :param pad_voxels: int, number of voxels to pad in six directions
        :param num_features: number of channels
        :param norm_type: 'batch' or 'group'
        :param num_groups: number of groups for GroupNorm (effective when norm_type='group')
        :param norm_kwargs: parameters for BatchNorm3d (e.g., eps, momentum)
        """
        super().__init__()
        self.pad_voxels = pad_voxels
        self.step_mode = step_mode
        self.norm = layer.GroupNorm(num_groups=num_groups, num_channels=num_features, step_mode=step_mode)


    def _compute_pad_value(self):
        if self.norm.affine:
            pad_value = self.norm.bias.detach()
        else:
            pad_value = torch.zeros(self.norm.num_channels, device=self.norm.weight.device)

        return pad_value.view(1, -1, 1, 1, 1)  # [1, C, 1, 1, 1]

    def _pad_tensor(self, x, pad_value):
        pad = [self.pad_voxels] * 6  # [W_l, W_r, H_t, H_b, D_f, D_b]
        x = F.pad(x, pad)  # First pad with 0
        # Replace with pad_value
        x[:, :, :self.pad_voxels, :, :] = pad_value
        x[:, :, -self.pad_voxels:, :, :] = pad_value
        x[:, :, :, :self.pad_voxels, :] = pad_value
        x[:, :, :, -self.pad_voxels:, :] = pad_value
        x[:, :, :, :, :self.pad_voxels] = pad_value
        x[:, :, :, :, -self.pad_voxels:] = pad_value
        return x
    
    def _pad_tensor_batch(self, x, pad_value):
        # Input x shape: [T, N, C, D, H, W]
        T, N, C, D, H, W = x.shape

        # Merge time and batch dimensions for easier padding
        x = x.view(T * N, C, D, H, W)

        pad = [self.pad_voxels] * 6  # pad format: left-right, top-bottom, front-back

        x = F.pad(x, pad)  # First pad with 0

        # Replace with pad_value
        x[:, :, :self.pad_voxels, :, :] = pad_value  # D axis front
        x[:, :, -self.pad_voxels:, :, :] = pad_value  # D axis back
        x[:, :, :, :self.pad_voxels, :] = pad_value  # H axis top
        x[:, :, :, -self.pad_voxels:, :] = pad_value  # H axis bottom
        x[:, :, :, :, :self.pad_voxels] = pad_value  # W axis left
        x[:, :, :, :, -self.pad_voxels:] = pad_value  # W axis right

        # Restore to original 6D shape, including padded size
        x = x.view(T, N, C, D + 2 * self.pad_voxels, H + 2 * self.pad_voxels, W + 2 * self.pad_voxels)
        return x

    def forward(self, x):
        if self.step_mode == 's':
            x = self.norm(x)  # shape: [N, C, D, H, W]
            if self.pad_voxels > 0:
                pad_value = self._compute_pad_value()
                x = self._pad_tensor(x, pad_value)
            return x

        elif self.step_mode == 'm':
            if x.dim() != 6:
                raise ValueError(f"Expected input shape [T, N, C, D, H, W], but got {x.shape}")
            x = self.norm(x)
            if self.pad_voxels > 0:
                pad_value = self._compute_pad_value()
                x = self._pad_tensor_batch(x, pad_value)
            return x

    @property
    def weight(self):
        return self.norm.weight

    @property
    def bias(self):
        return self.norm.bias

    @property
    def eps(self):
        return self.norm.eps if hasattr(self.norm, 'eps') else 1e-5



class RepConv3D(nn.Module):
    def __init__(self, in_channel, out_channel, pad_voxels=1, num_groups=8, bias=False, step_mode='m'):
        super().__init__()

        # 1x1 projection conv
        self.proj_conv = layer.Conv3d(in_channels=in_channel, out_channels=in_channel,
                                      kernel_size=1, stride=1, padding=0, bias=bias, step_mode=step_mode)

        # Norm + Padding
        self.norm_pad = NormAndPad3DLayer(pad_voxels=pad_voxels, num_features=in_channel, step_mode=step_mode)

        # Depthwise 3x3 conv
        self.dw_conv3x3 = layer.Conv3d(in_channels=in_channel, out_channels=in_channel,
                                       kernel_size=3, stride=1, padding=0, bias=bias,
                                       groups=in_channel, step_mode=step_mode)

        # Pointwise 1x1 conv
        self.pw_conv1x1 = layer.Conv3d(in_channels=in_channel, out_channels=out_channel,
                                       kernel_size=1, stride=1, padding=0, bias=bias,
                                       step_mode=step_mode)

        # Output Norm
        self.out_norm = layer.GroupNorm(num_groups=num_groups, num_channels=out_channel, step_mode=step_mode)

    def forward(self, x):  
        x = self.proj_conv(x)          # 1×1 conv
        x = self.norm_pad(x)             # Norm + padding
        x = self.dw_conv3x3(x)         # depthwise 3×3 conv
        x = self.pw_conv1x1(x)         # pointwise 1×1 conv
        x = self.out_norm(x)             # output Norm
        return x


class SepConv3D(nn.Module):
    """
    Spiking 3D version of inverted separable convolution from MobileNetV2.
    Input: [T, B, C, D, H, W]
    """

    def __init__(
        self,
        dim,
        expansion_ratio=2,
        kernel_size=7,
        padding=3,
        tau=2.0,
        step_mode='m',
        num_groups=8,
        bias=False):
        super().__init__()
        med_channels = int(expansion_ratio * dim)

        # spike layer 1
        self.lif1 = GeneralParametricLIFNode(
            init_tau=tau,
            init_threshold=1.0,
            learnable_tau=True,
            learnable_threshold=True,
            decay_input=True,
            detach_reset=True,                
            v_reset=0.0,
            surrogate_function=surrogate.ATan(), # surrogate.ATan()
            step_mode=step_mode,
            backend='cupy'
        )

        # pointwise conv 1
        self.pwconv1 = layer.Conv3d(dim, med_channels, kernel_size=1, stride=1,
                                    bias=bias, step_mode=step_mode)
        
        # norm layer 1
        self.norm1 = layer.GroupNorm(num_groups=num_groups, num_channels=med_channels, step_mode=step_mode)

        # spike layer 2        
        self.lif2 = GeneralParametricLIFNode(
            init_tau=tau,
            init_threshold=1.0,
            learnable_tau=True,
            learnable_threshold=True,
            decay_input=True,
            detach_reset=True,                
            v_reset=0.0,
            surrogate_function=surrogate.ATan(), # surrogate.ATan()
            step_mode=step_mode,
            backend='cupy'
        )

        # depthwise conv
        self.dwconv = layer.Conv3d(med_channels, med_channels, kernel_size=kernel_size,
                                   padding=padding, groups=med_channels,
                                   bias=bias, step_mode=step_mode)

        # pointwise conv 2
        self.pwconv2 = layer.Conv3d(med_channels, dim, kernel_size=1, stride=1,
                                    bias=bias, step_mode=step_mode)
        
        # norm layer 2
        self.norm2 = layer.GroupNorm(num_groups=num_groups, num_channels=dim, step_mode=step_mode)


    def forward(self, x):
        # x: [T, B, C, D, H, W]
        x = self.lif1(x)
        x = self.pwconv1(x)
        x = self.norm1(x)
        x = self.lif2(x)
        x = self.dwconv(x)
        x = self.pwconv2(x)
        x = self.norm2(x)
        return x


class MS_SpikeConvBlock3D(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio=4.0,
        tau=2.0,
        num_groups=8,
        step_mode='m'):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)

        self.sep_conv = SepConv3D(dim=dim, step_mode=step_mode)

        self.lif1 = GeneralParametricLIFNode(
            init_tau=tau,
            init_threshold=1.0,
            learnable_tau=True,
            learnable_threshold=True,
            decay_input=True,
            detach_reset=True,                
            v_reset=0.0,
            surrogate_function=surrogate.ATan(), # surrogate.ATan()
            step_mode=step_mode,
            backend='cupy'
        )
        
        self.conv1 = layer.Conv3d(in_channels=dim, out_channels=hidden_dim,
                                  kernel_size=3, padding=1, bias=False,
                                  step_mode=step_mode)
        
        self.norm1 = layer.GroupNorm(num_groups=num_groups, num_channels=hidden_dim, step_mode=step_mode)

        # Spike + Conv + Norm block2
        self.lif2 = GeneralParametricLIFNode(
            init_tau=tau,
            init_threshold=1.0,
            learnable_tau=True,
            learnable_threshold=True,
            decay_input=True,
            detach_reset=True,                
            v_reset=0.0,
            surrogate_function=surrogate.ATan(), # surrogate.ATan()
            step_mode=step_mode,
            backend='cupy'
        )
        
        self.conv2 = layer.Conv3d(in_channels=hidden_dim, out_channels=dim,
                                  kernel_size=3, padding=1, bias=False,
                                  step_mode=step_mode)
        
        self.norm2 = layer.GroupNorm(num_groups=num_groups, num_channels=dim, step_mode=step_mode)

    def forward(self, x):
        # x: [T, B, C, D, H, W]

        # Branch 1: Lightweight convolution block + residual
        x = self.sep_conv(x) + x
        x_feat = x

        # Branch 2: MLP-like spike conv
        x = self.lif1(x)
        x = self.conv1(x)
        x = self.norm1(x)

        x = self.lif2(x)
        x = self.conv2(x)
        x = self.norm2(x)

        # Final residual
        x = x_feat + x

        return x


class MS_SpikeMLP3D(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        tau=2.0,
        num_groups=8,
        step_mode='m'):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # Using 1x1x1 convolution to implement "fully connected" operation
        self.fc1_conv = layer.Conv3d(in_features, hidden_features, kernel_size=1,
                                    stride=1, padding=0, bias=False, step_mode=step_mode)
        
        self.fc1_norm = layer.GroupNorm(num_groups=num_groups, num_channels=hidden_features, step_mode=step_mode)

        self.fc1_lif = GeneralParametricLIFNode(
            init_tau=tau,
            init_threshold=1.0,
            learnable_tau=True,
            learnable_threshold=True,
            decay_input=True,
            detach_reset=True,                
            v_reset=0.0,
            surrogate_function=surrogate.ATan(), # surrogate.ATan()
            step_mode=step_mode,
            backend='cupy'
        )

        self.fc2_conv = layer.Conv3d(hidden_features, out_features, kernel_size=1,
                                    stride=1, padding=0, bias=False, step_mode=step_mode)

        self.fc2_norm = layer.GroupNorm(num_groups=num_groups, num_channels=out_features, step_mode=step_mode)

        self.fc2_lif = GeneralParametricLIFNode(
            init_tau=tau,
            init_threshold=1.0,
            learnable_tau=True,
            learnable_threshold=True,
            decay_input=True,
            detach_reset=True,                
            v_reset=0.0,
            surrogate_function=surrogate.ATan(), # surrogate.ATan()
            step_mode=step_mode,
            backend='cupy'
        )

    def forward(self, x):
        # x: [T, B, C, D, H, W]
        x = self.fc1_lif(x)
        x = self.fc1_conv(x)
        x = self.fc1_norm(x)

        x = self.fc2_lif(x)
        x = self.fc2_conv(x)
        x = self.fc2_norm(x)
        return x


class MS_SpikeAttention_RepConv3D_qkv_id(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=2,
        tau=2.0,
        num_groups=8,
        step_mode='m'):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125 if qk_scale is None else qk_scale

        self.sr_ratio = sr_ratio

        if sr_ratio > 1:
            self.sr_lif_k = GeneralParametricLIFNode(
                init_tau=tau,
                init_threshold=1.0,
                learnable_tau=True,
                learnable_threshold=True,
                decay_input=True,
                detach_reset=True,                
                v_reset=0.0,
                surrogate_function=surrogate.ATan(), # surrogate.ATan()
                step_mode=step_mode,
                backend='cupy'
            )
            self.sr_k = layer.Conv3d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, groups=dim,
                                     bias=qkv_bias, step_mode=step_mode)
            self.sr_norm_k = layer.GroupNorm(num_groups=num_groups, num_channels=dim, step_mode=step_mode)
                
            self.sr_lif_v = GeneralParametricLIFNode(
                init_tau=tau,
                init_threshold=1.0,
                learnable_tau=True,
                learnable_threshold=True,
                decay_input=True,
                detach_reset=True,                
                v_reset=0.0,
                surrogate_function=surrogate.ATan(), # surrogate.ATan()
                step_mode=step_mode,
                backend='cupy'
            )
            self.sr_v = layer.Conv3d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, groups=dim,
                                     bias=qkv_bias, step_mode=step_mode)
            self.sr_norm_v = layer.GroupNorm(num_groups=num_groups, num_channels=dim, step_mode=step_mode)             
            
        self.head_lif = GeneralParametricLIFNode(
            init_tau=tau,
            init_threshold=1.0,
            learnable_tau=True,
            learnable_threshold=True,
            decay_input=True,
            detach_reset=True,                
            v_reset=0.0,
            surrogate_function=surrogate.ATan(), # surrogate.ATan()
            step_mode=step_mode,
            backend='cupy'
        )
        
        q_norm = layer.GroupNorm(num_groups=num_groups, num_channels=dim, step_mode=step_mode)
        k_norm = layer.GroupNorm(num_groups=num_groups, num_channels=dim, step_mode=step_mode)
        v_norm = layer.GroupNorm(num_groups=num_groups, num_channels=dim, step_mode=step_mode)
        proj_norm = layer.GroupNorm(num_groups=num_groups, num_channels=dim, step_mode=step_mode)

        self.q_conv = nn.Sequential(
            RepConv3D(dim, dim, bias=qkv_bias, step_mode=step_mode),
            q_norm)
        self.k_conv = nn.Sequential(
            RepConv3D(dim, dim, bias=qkv_bias, step_mode=step_mode),
            k_norm)
        self.v_conv = nn.Sequential(
            RepConv3D(dim, dim, bias=qkv_bias, step_mode=step_mode),
            v_norm)

        self.q_lif = GeneralParametricLIFNode(
            init_tau=tau,
            init_threshold=1.0,
            learnable_tau=True,
            learnable_threshold=True,
            decay_input=True,
            detach_reset=True,                
            v_reset=0.0,
            surrogate_function=surrogate.ATan(), # surrogate.ATan()
            step_mode=step_mode,
            backend='cupy'
        )
        self.k_lif = GeneralParametricLIFNode(
            init_tau=tau,
            init_threshold=1.0,
            learnable_tau=True,
            learnable_threshold=True,
            decay_input=True,
            detach_reset=True,                
            v_reset=0.0,
            surrogate_function=surrogate.ATan(), # surrogate.ATan()
            step_mode=step_mode,
            backend='cupy'
        )
        self.v_lif = GeneralParametricLIFNode(
            init_tau=tau,
            init_threshold=1.0,
            learnable_tau=True,
            learnable_threshold=True,
            decay_input=True,
            detach_reset=True,                
            v_reset=0.0,
            surrogate_function=surrogate.ATan(), # surrogate.ATan()
            step_mode=step_mode,
            backend='cupy'
        )
        self.attn_lif = GeneralParametricLIFNode(
            init_tau=tau,
            init_threshold=0.5,
            learnable_tau=True,
            learnable_threshold=True,
            decay_input=True,
            detach_reset=True,                
            v_reset=0.0,
            surrogate_function=surrogate.ATan(), # surrogate.ATan()
            step_mode=step_mode,
            backend='cupy'
        )
            
        self.alpha = nn.Parameter(torch.tensor(0.3))

        self.proj_conv = nn.Sequential(
            RepConv3D(dim, dim, bias=qkv_bias, step_mode=step_mode),
            proj_norm)

    def forward(self, x):
        # x: [T, B, C, D, H, W]
        T, B, C, D, H, W = x.shape
        N = D * H * W

        x = self.head_lif(x)
        q = self.q_conv(x)
        k = self.k_conv(x)
        v = self.v_conv(x)
        
        # Spatial Reduction (only K,V)
        if self.sr_ratio > 1:
            k_ = self.sr_norm_k(self.sr_k(self.sr_lif_k(k)))
            v_ = self.sr_norm_v(self.sr_v(self.sr_lif_v(v)))

            D_, H_, W_ = k_.shape[3:]
            N_ = D_ * H_ * W_
        else:
            k_ = k
            v_ = v
            N_ = N
        
        # reshape
        q = self.q_lif(q).flatten(3)
        q = q.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads)
        q = q.permute(0, 1, 3, 2, 4).contiguous()  # [T,B,h,N,d]

        k_ = self.k_lif(k_).flatten(3)
        k_ = k_.transpose(-1, -2).reshape(T, B, N_, self.num_heads, C // self.num_heads)
        k_ = k_.permute(0, 1, 3, 2, 4).contiguous()

        v_ = self.v_lif(v_).flatten(3)
        v_ = v_.transpose(-1, -2).reshape(T, B, N_, self.num_heads, C // self.num_heads)
        v_ = v_.permute(0, 1, 3, 2, 4).contiguous()

        # Attention calculation: k^T @ v
        attn_kv = torch.matmul(k_.transpose(-2, -1), v_)  # [T, B, heads, head_dim, head_dim]
        x = torch.matmul(q, attn_kv) * self.scale # [T, B, heads, N, head_dim]

        # Restore spatial dimensions
        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x).reshape(T, B, C, D, H, W)

        x = self.proj_conv(x)

        return x


class MS_SpikeTransformerBlock3D(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        sr_ratio=1,
        tau=2.0,
        num_groups=8,
        step_mode='m'):
        super().__init__()

        self.attn = MS_SpikeAttention_RepConv3D_qkv_id(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
            tau=tau,
            num_groups=num_groups,
            step_mode=step_mode)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MS_SpikeMLP3D(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim,
            tau=tau,
            num_groups=num_groups,
            step_mode=step_mode
        )

    def forward(self, x):
        # x: [T, B, C, D, H, W]
        # x = x + self.drop_path(self.attn(x))
        # x = x + self.drop_path(self.mlp(x))
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x



class TimeDistributed(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):  # [T, B, ...]
        T, B = x.shape[:2]
        x = x.view(T * B, *x.shape[2:])
        x = self.module(x)
        x = x.view(T, B, *x.shape[1:])
        return x

class MS_SpikeDownSampling3D(nn.Module):
    """
    Spiking 3D downsampling block.
    """
    def __init__(
        self,
        in_channels=4,
        embed_dims=96,
        kernel_size=3,
        stride=2,
        padding=1,
        first_layer=True,
        tau=2.0,
        num_groups=8,
        step_mode='m'):
        super().__init__()

        self.encode_conv = layer.Conv3d(
            in_channels,
            embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
            step_mode=step_mode
        )

        self.encode_norm = layer.GroupNorm(num_groups=num_groups, num_channels=embed_dims, step_mode=step_mode)

        # self.relu = TimeDistributed(nn.ReLU())
        self.use_lif = not first_layer
        if self.use_lif:
            self.encode_lif = GeneralParametricLIFNode(
                init_tau=tau,
                init_threshold=1.0,
                learnable_tau=True,
                learnable_threshold=True,
                decay_input=True,
                detach_reset=True,                
                v_reset=0.0,
                surrogate_function=surrogate.ATan(), # surrogate.ATan()
                step_mode=step_mode,
                backend='cupy'
            )            
            
    def forward(self, x):
        # x: [T, B, C, D, H, W]
        if self.use_lif:
            x = self.encode_lif(x)
        x = self.encode_conv(x)
        x = self.encode_norm(x)
        return x
    
    
class MS_SpikeUpSampling3D(nn.Module):
    """
    Spiking 3D upsampling block.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=2,
        padding=1,
        output_padding=1,
        last_layer=False,
        tau=2.0,
        num_groups=8,
        step_mode='m'
    ):
        super().__init__()

        self.decode_conv = layer.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=True,
            step_mode=step_mode
        )

        self.decode_norm = layer.GroupNorm(num_groups=num_groups, num_channels=out_channels, step_mode=step_mode)

        self.use_lif = not last_layer
        if self.use_lif:
            self.decode_lif = GeneralParametricLIFNode(
                init_tau=tau,
                init_threshold=1.0,
                learnable_tau=True,
                learnable_threshold=True,
                decay_input=True,
                detach_reset=True,                
                v_reset=0.0,
                surrogate_function=surrogate.ATan(), # surrogate.ATan()
                step_mode=step_mode,
                backend='cupy'
            )     

    def forward(self, x):
        # x: [T, B, C, D, H, W]
        if self.use_lif:
            x = self.decode_lif(x)
        x = self.decode_conv(x)
        x = self.decode_norm(x)
        return x    
    
    
class AddConverge3D(base.MemoryModule):
    """
    Spiking 3D addition convergence block.
    """
    def __init__(self, channels, num_groups=8, tau=2.0, step_mode='m'):
        super().__init__()

        self.lif1 = GeneralParametricLIFNode(
            init_tau=tau,
            init_threshold=1.0,
            learnable_tau=True,
            learnable_threshold=True,
            decay_input=True,
            detach_reset=True,                
            v_reset=0.0,
            surrogate_function=surrogate.ATan(), # surrogate.ATan()
            step_mode=step_mode,
            backend='cupy'
        )
        self.lif2 = GeneralParametricLIFNode(
            init_tau=tau,
            init_threshold=1.0,
            learnable_tau=True,
            learnable_threshold=True,
            decay_input=True,
            detach_reset=True,                
            v_reset=0.0,
            surrogate_function=surrogate.ATan(), # surrogate.ATan()
            step_mode=step_mode,
            backend='cupy'
        )      

        self.norm = layer.GroupNorm(num_groups=num_groups, num_channels=channels, step_mode=step_mode)

    def forward(self, x1, x2):
        x1 = self.lif1(x1)
        x2 = self.lif2(x2)
        x = x1 + x2  # skip connection by addition
        x = self.norm(x)
        return x 
 
    
class Spike_Former_Unet3D(nn.Module):
    """Spiking 3D U-shaped Transformer network."""
    def __init__(
        self,
        in_channels=4,
        num_classes=3,
        embed_dim=24,
        num_heads=[1, 2, 4, 8],
        mlp_ratios=[4, 4, 4, 4],
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        depths=[8, 8, 8, 8],
        layers=[2, 2, 6, 2],
        sr_ratio=2,
        T=2,
        num_groups=8,
        step_mode='m'):
        super().__init__()
        self.T = T

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Encode-Stage 1
        self.downsample1_a = MS_SpikeDownSampling3D(
            in_channels=in_channels,
            embed_dims=embed_dim,
            kernel_size=7,
            stride=2,
            padding=3,
            first_layer=True,
            num_groups=num_groups,
            step_mode=step_mode)
        
        self.encode_block1_a = nn.ModuleList([
            MS_SpikeConvBlock3D(dim=embed_dim, mlp_ratio=mlp_ratios[0], num_groups=num_groups, step_mode=step_mode)])

        self.downsample1_b = MS_SpikeDownSampling3D(
            in_channels=embed_dim,
            embed_dims=embed_dim * 2,
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
            num_groups=num_groups,
            step_mode=step_mode)
        
        self.encode_block1_b = nn.ModuleList([
            MS_SpikeConvBlock3D(dim=embed_dim * 2, mlp_ratio=mlp_ratios[0], num_groups=num_groups, step_mode=step_mode)])

        # Encode-Stage 2
        self.downsample2 = MS_SpikeDownSampling3D(
            in_channels=embed_dim * 2,
            embed_dims=embed_dim * 4,
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
            num_groups=num_groups,
            step_mode=step_mode)
                
        self.encode_block2_a = nn.ModuleList([
            MS_SpikeConvBlock3D(dim=embed_dim * 4, mlp_ratio=mlp_ratios[1], num_groups=num_groups,step_mode=step_mode)])
       
        self.encode_block2_b = nn.ModuleList([
            MS_SpikeConvBlock3D(dim=embed_dim * 4, mlp_ratio=mlp_ratios[1], num_groups=num_groups,step_mode=step_mode)])

        # Encode-Stage 3
        self.downsample3 = MS_SpikeDownSampling3D(
            in_channels=embed_dim * 4,
            embed_dims=embed_dim * 8,
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
            num_groups=num_groups,
            step_mode=step_mode)
        
        self.encode_block3 = nn.ModuleList([
            MS_SpikeTransformerBlock3D(
                dim=embed_dim * 8,
                num_heads=num_heads[2],
                mlp_ratio=mlp_ratios[2],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                sr_ratio=sr_ratio,
                num_groups=num_groups,
                step_mode=step_mode
            ) for i in range(layers[2])])

        # feature-Stage
        self.feature_downsample = MS_SpikeDownSampling3D(
            in_channels=embed_dim * 8,
            embed_dims=embed_dim * 16,
            kernel_size=3,
            stride=1,
            padding=1,
            first_layer=False,
            num_groups=num_groups,
            step_mode=step_mode)
        
        self.feature_block = nn.ModuleList([
            MS_SpikeTransformerBlock3D(
                dim=embed_dim * 16,
                num_heads=num_heads[3],
                mlp_ratio=mlp_ratios[3],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                sr_ratio=sr_ratio,
                num_groups=num_groups,
                step_mode=step_mode
            ) for i in range(layers[3])
        ])
        
        # Decode-Stage 3
        self.upsample3 = MS_SpikeUpSampling3D(
            in_channels=embed_dim * 16,
            out_channels=embed_dim * 8,
            kernel_size=3,
            stride=1,
            padding=1,
            output_padding=0,
            last_layer=False,
            num_groups=num_groups,
            step_mode=step_mode)
        
        self.decode_block3 = nn.ModuleList([
            MS_SpikeTransformerBlock3D(
                dim=embed_dim * 8,
                num_heads=num_heads[2],
                mlp_ratio=mlp_ratios[2],
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                sr_ratio=sr_ratio,
                num_groups=num_groups,
                step_mode=step_mode
            ) for i in range(layers[2])])

        self.converge3 = AddConverge3D(
            channels=embed_dim * 8, num_groups=num_groups,step_mode=step_mode)

        # Decode-Stage 2
        self.upsample2 = MS_SpikeUpSampling3D(
            in_channels=embed_dim * 8,
            out_channels=embed_dim * 4,
            kernel_size=3,
            stride=2,
            padding=1,
            last_layer=False,
            num_groups=num_groups, 
            step_mode=step_mode)
                
        self.decode_block2_a = nn.ModuleList([
            MS_SpikeConvBlock3D(dim=embed_dim * 4, mlp_ratio=mlp_ratios[1], num_groups=num_groups, step_mode=step_mode)])
       
        self.decode_block2_b = nn.ModuleList([
            MS_SpikeConvBlock3D(dim=embed_dim * 4, mlp_ratio=mlp_ratios[1], num_groups=num_groups, step_mode=step_mode)])
        
        self.converge2 = AddConverge3D(
            channels=embed_dim * 4, num_groups=num_groups, step_mode=step_mode)

                   
        # Decode-Stage 1
        self.upsample1_b = MS_SpikeUpSampling3D(
            in_channels=embed_dim * 4,
            out_channels= embed_dim * 2,
            kernel_size=3,
            stride=2,
            padding=1,
            last_layer=False,
            num_groups=num_groups, 
            step_mode=step_mode)
        
        self.decode_block1_b = nn.ModuleList([
            MS_SpikeConvBlock3D(dim=embed_dim * 2, mlp_ratio=mlp_ratios[0], num_groups=num_groups, step_mode=step_mode)])
        
        self.upsample1_a = MS_SpikeUpSampling3D(
            in_channels=embed_dim * 2,
            out_channels=embed_dim,
            kernel_size=7,
            stride=2,
            padding=3,
            last_layer=False,
            num_groups=num_groups, 
            step_mode=step_mode)
        
        self.decode_block1_a = nn.ModuleList([
            MS_SpikeConvBlock3D(dim=embed_dim, mlp_ratio=mlp_ratios[0], num_groups=num_groups, step_mode=step_mode)])

        self.converge1 = AddConverge3D(
            channels=embed_dim * 2, num_groups=num_groups, step_mode=step_mode)

        
        self.final_upsample = MS_SpikeUpSampling3D(
            in_channels=embed_dim,
            out_channels=embed_dim // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            last_layer=True,
            num_groups=4, 
            step_mode=step_mode)
        
        self.ds_conv3 = layer.Conv3d(embed_dim * 8, num_classes, kernel_size=1, step_mode='m')
        self.ds_conv2 = layer.Conv3d(embed_dim * 4, num_classes, kernel_size=1, step_mode='m')
        self.ds_conv1 = layer.Conv3d(embed_dim, num_classes, kernel_size=1, step_mode='m') 
        
        self.readout = layer.Conv3d(embed_dim // 2, num_classes, kernel_size=1, step_mode=step_mode)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.InstanceNorm3d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder_decoder(self, x):         # input shape: [T, B, 4, 64, 64, 64]
        # Encode-stage 1
        e1 = self.downsample1_a(x)          # Downsample1_a output shape: [T, B, 48, 32, 32, 32]
        for blk in self.encode_block1_a:
            e1 = blk(e1)                     # shape: [T, B, 48, 32, 32, 32]
        
        e1 = self.downsample1_b(e1)          # Downsample1_b output shape: [T, B, 96, 16, 16, 16]
        for blk in self.encode_block1_b:
            e1 = blk(e1)
        skip1 = e1                 # Skip2 shape: [T, B, 96, 16, 16, 16]

        # Encode-stage 2
        e2 = self.downsample2(e1)            # Downsample2 output shape: [T, B, 192, 8, 8, 8]
        for blk in self.encode_block2_a:
            e2 = blk(e2)
        for blk in self.encode_block2_b:
            e2 = blk(e2)
        skip2 = e2                  # Skip3 shape: [T, B, 192, 8, 8, 8]

        # Encode-stage 3
        e3 = self.downsample3(e2)            # Downsample3 output shape: [T, B, 384, 4, 4, 4]
        for blk in self.encode_block3:
            e3 = blk(e3)
        skip3 = e3                  # Skip4 shape: [T, B, 384, 4, 4, 4]

        # Encode-stage 4
        e4 = self.feature_downsample(e3)     # Downsample4 output shape: [T, B, 480, 4, 4, 4]
        for blk in self.feature_block:
            e4 = blk(e4)                     # After Encode-Stage 4: [T, B, 480, 4, 4, 4]
        
        # Decode-Stage 3
        d3 = self.upsample3(e4)              # Upsample3 output shape: [T, B, 384, 4, 4, 4]
        d3 = self.converge3(d3, skip3)       # converge3 output shape: [T, B, 384, 4, 4, 4]
        for blk in self.decode_block3:
            d3 = blk(d3)                     # After Decode-Stage3: [T, B, 384, 4, 4, 4]
        ds3 = d3
        
        # Decode-Stage 2
        d2 = self.upsample2(d3)              # Upsample2 output shape: [T, B, 192, 8, 8, 8]
        
        d2 = self.converge2(d2, skip2)       # Converge2 output shape: [T, B, 192, 8, 8, 8]
        for blk in self.decode_block2_a:
            d2 = blk(d2)
        for blk in self.decode_block2_b:
            d2 = blk(d2)                     # After Decode-Stage2: [T, B, 192, 8, 8, 8]
        ds2 = d2

        # Decode-Stage 1
        d1 = self.upsample1_b(d2)            # Upsample1_b output shape: [T, B, 96, 16, 16, 16]
        d1 = self.converge1(d1, skip1)       # Converge1 output shape: [T, B, 96, 16, 16, 16]
        for blk in self.decode_block1_b:
            d1 = blk(d1)

        d1 = self.upsample1_a(d1)            # Upsample1_a output shape: [T, B, 48, 32, 32, 32]
        for blk in self.decode_block1_a:
            d1 = blk(d1)
            
        ds1 = d1
            
        out =self.final_upsample(d1)          # Final Upsample output shape: [T, B, 24, 64, 64, 64]

        return out, ds1, ds2, ds3

    def forward(self, x):
        functional.reset_net(self)
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1, 1)        
        out, ds1, ds2, ds3 = self.forward_encoder_decoder(x)  # [T, B, C, D, H, W]
        T, B, _, D_out, H_out, W_out = out.shape

        # deep supervision conv + mean(0)
        ds1 = self.ds_conv1(ds1).mean(0)  # [B, num_classes, d, h, w]
        ds2 = self.ds_conv2(ds2).mean(0)
        ds3 = self.ds_conv3(ds3).mean(0)

        # Upsample to final output size
        ds1 = F.interpolate(ds1, size=(D_out, H_out, W_out), mode='trilinear', align_corners=False)
        ds2 = F.interpolate(ds2, size=(D_out, H_out, W_out), mode='trilinear', align_corners=False)
        ds3 = F.interpolate(ds3, size=(D_out, H_out, W_out), mode='trilinear', align_corners=False)

        # Final output
        output = self.readout(out).mean(0)# [B, num_classes, D, H, W]

        return [output, ds1, ds2, ds3]



def spike_former_unet3D_1111_16(in_channels=4, num_classes=3, T=4, step_mode='m', **kwargs):
    model = Spike_Former_Unet3D(
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dim=16,
        num_heads=[2, 4, 4, 8],
        mlp_ratios=[2, 2, 2, 2],
        qkv_bias=False,
        depths=[2, 2, 2, 2],
        layers=[1, 1, 1, 1],
        sr_ratio=2,
        T=T,
        step_mode=step_mode,
        **kwargs
    )
    return model



def spike_former_unet3D_2222_24(in_channels=4, num_classes=3, T=2, step_mode='m', **kwargs):
    model = Spike_Former_Unet3D(
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dim= 24,
        num_heads= [8, 8, 8, 8],
        mlp_ratios=[2, 2, 2, 2],
        qkv_bias=False,
        depths=[2, 2, 2, 2],
        layers=[2, 2, 2, 2],
        sr_ratio=2,
        T=T,
        num_groups=8,
        step_mode=step_mode,
        **kwargs
    )
    return model


def spike_former_unet3D_2222_32(in_channels=4, num_classes=3, T=2, step_mode='m', **kwargs):
    model = Spike_Former_Unet3D(
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dim= 32,
        num_heads= [8, 8, 8, 8],
        mlp_ratios=[2, 2, 2, 2],
        qkv_bias=False,
        depths=[2, 2, 2, 2],
        layers=[2, 2, 2, 2],
        sr_ratio=2,
        T=T,
        num_groups=8,
        step_mode=step_mode,
        **kwargs
    )
    return model



def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    return total_params, trainable_params

