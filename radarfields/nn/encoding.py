import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """位置编码模块

    Maps v to positional encoding representation phi(v)

    Arguments:
        i_dim (int): v的输入维度
        N_freqs (int): #采样频率 (default: 10)
        类似nerf 将位置编码编到高频上去, 来获得更好的高频信息

    adapted from: https://github.com/yliess86/NeRF/blob/main/nerf/core/features.py
    and from: https://github.com/bebeal/mipnerf-pytorch/blob/main/model.py
    """
    
    def __init__(self, i_dim, a=0, b=4):
        super().__init__()
        self.i_dim = i_dim
        self.a = a
        self.b = b
        self.N_freqs = b - a

        #输出的维度为 输入维度 + 2*N_freqs*输入维度（sin和cos）
        self.o_dim = self.i_dim + (2 * self.N_freqs) * self.i_dim

        freq_bands = 2 ** torch.arange(a, b)
        self.register_buffer("freq_bands", freq_bands)
        #参数不进行更新，但是进行保存
        
    def fourier_features(self, v):
        """Map v to positional encoding representation phi(v)

        Arguments:
            v (Tensor): input features (B, IFeatures)

        Returns:
            phi(v) (Tensor): fourier features (B, IFeatures + (2 * N_freqs) * IFeatures)
        """
        pe = [v]
        for freq in self.freq_bands:
            fv = freq * v
            #在原本的位置的基础上，增加升高频率的位置
            pe += [torch.sin(fv), torch.cos(fv)]
        return torch.cat(pe, dim=-1)
    
    def forward(self, x):
        """将x进行位置编码
    
        Arguments:
            x (Tensor): 输入 (B, IFeatures)

        Returns:
            phi(x) (Tensor): fourier features (B, IFeatures + (2 * (b-a)) * IFeatures)
        """
        return self.fourier_features(x)