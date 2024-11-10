
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange



from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


class LinearTokenMixer(nn.Module):
    def __init__(self, dim=512, attn_bias=False, proj_drop=0.,is_spectral=False):
        super().__init__()
        self.qkv = nn.Conv2d(dim, 3 * dim, 1, stride=1, padding=0, bias=attn_bias)
        self.oper_q = nn.Sequential(
            ChannelAttn(dim) if is_spectral else SpatialAttn(dim),
        )
        self.oper_k = nn.Sequential(
            ChannelAttn(dim) if is_spectral else SpatialAttn(dim),
        )
        self.dwc = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.proj = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=1)
        q = self.oper_q(q)
        k = self.oper_k(k)
        out = self.proj(self.dwc(q + k) * v)
        out = self.proj_drop(out)
        return out


class LinearBlock(nn.Module):
    def __init__(self, dim, MLP_ratio=4., attn_bias=False, is_spectral=False):
        super().__init__()
        self.local_perception = LocalMLP(dim)
        self.attn = LinearTokenMixer(dim, attn_bias=attn_bias,is_spectral=is_spectral)
        MLP_hidden_dim = int(dim * MLP_ratio)
        self.MLP = MLP(in_features=dim, hidden_features=MLP_hidden_dim)

    def forward(self, x):
        x = x + self.local_perception(x)
        x = x + self.attn(x)
        x = x + self.MLP(x)
        return x
        
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SpatialAttn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, 1, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)

class ChannelAttn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)

class LocalMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 1, 1, 0),
        )

    def forward(self, x):
        return self.net(x)


class LTFF(nn.Module):
    def __init__(self,hsi_bands,msi_bands=3, depth=4,
                 hidden_dim=64,scale=4,add_fft=True,add_qshift=True):
        super(LTFF, self).__init__()
        self.hsi_bands = hsi_bands
        self.msi_bands = msi_bands
        self.hidden_dim = hidden_dim
        self.scale = scale
        self.add_fft = add_fft
        self.add_qshift = add_qshift
        
        self.in_conv = nn.Conv2d(hsi_bands + msi_bands,hidden_dim,3,1,1)
        
        self.global_filters = nn.ModuleList()
        self.encoders = nn.ModuleList()
        for idx in range(depth):
            self.global_filters.append(FrequencyFilter(dim=hidden_dim, h=64, w=64))
            self.encoders.append(LinearBlock(hidden_dim,is_spectral=(idx % 2 == 0)))
            
        self.refine_conv = nn.Sequential(
            nn.Conv2d(hidden_dim * depth,hsi_bands,3,1,1),
            nn.SiLU(),
            nn.Conv2d(hsi_bands,hsi_bands,3,1,1)
        )
        
    def forward(self, hsi,msi):
        up_hsi = F.interpolate(hsi,scale_factor=self.scale,mode='bicubic',align_corners=True)
        b,C,H,W = up_hsi.size()
        x = self.in_conv(torch.cat([up_hsi,msi],dim=1))
        res = []
        for idx in range(len(self.encoders)):
            x = self.encoders[idx](x)
            x = self.global_filters[idx](x)
            res.append(x)
        x = torch.cat(res,dim=1)
        x = self.refine_conv(x) + up_hsi
        return x


class FrequencyFilter(nn.Module):
    def __init__(self, dim=64, h=64, w=64):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(1, dim, h, w,2,dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h


            
    def forward(self, x):
        b,c,h,w= x.size()
        x = torch.fft.fft2(x)
        x = x * torch.view_as_complex(self.complex_weight)
        x = torch.fft.ifft2(x)
        x = x.real
        return x



