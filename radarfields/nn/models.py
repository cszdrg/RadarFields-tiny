import torch
import torch.nn as nn
import numpy as np
from radarfields.nn.encoding import PositionalEncoding

def mask(encoding, mask_coef):
    if mask_coef is None: return encoding
    
    mask_coef = 0.4 + 0.6*mask_coef #限定为 0.4 - 1.0
    
    mask = torch.zeros_like(encoding[0:1])
    mask_ceil = int(np.ceil(mask_coef * encoding.shape[1]))
    mask[:, :, mask_ceil] = 1.0
    return encoding * mask

def linear(in_params, out_params, relu=True, leaky=False, slope=0.01, bn=False):

    layer = [nn.Linear(in_features=in_params,out_features=out_params)]

    if bn: layer.append(nn.BatchNorm1d(out_params))

    if relu:
        if leaky: layer.append(nn.LeakyReLU(negative_slope=slope))
        else: layer.append(nn.ReLU())

    return nn.Sequential(*layer)

# 多层感知机 输入： 输入维度 层数 中间隐藏层数 最终输出层数 bn
class MLP(nn.Module):
    def __init__(self, in_dim, num_layers, hidden_dim, out_dim, bn=False):
        super().__init__()

        self.model = nn.Sequential(linear(in_dim, hidden_dim, bn), 
                                   *[linear(hidden_dim, hidden_dim, bn) 
                                     for _ in range(num_layers-2)],
                                     linear(hidden_dim, out_dim, relu=False))
    
    def forward(self, x):
        return self.model(x)

class RadarField(nn.Module):
    def __init__(self, 
                 in_dim=3,
                 xyz_encoding="HashGrid",
                 num_layers=6,
                 hidden_dim=64,
                 xyz_feat_dim=20,
                 alpha_dim=1,
                 alpha_activation="sigmoid",
                 sigmoid_tightness=8.0,
                 rd_dim=1,
                 softplus_rd=True,
                 angle_dim=3,
                 angle_in_layer=5,
                 angle_encoding="SphericalHarmonics",
                 num_bands_xyz=10,
                 resolution=2048,
                 n_levels=16,
                 bound=1,
                 bn=False,
                 ):
        super().__init__()
        
        self.alpha_activation = alpha_activation
        self.softplus_rd = softplus_rd
        self.hashgrid = xyz_encoding == "HashGrid"
        self.bn = bn

        # Activations
        self.sigmoid = nn.Sigmoid()
        self.sigmoid_tightness = sigmoid_tightness
        self.softplus = nn.Softplus()       
        
        # 位置编码
        self.encode_xyz = PositionalEncoding(in_dim, b=num_bands_xyz)
        self.encode_angle = PositionalEncoding(angle_dim)
        
        self.xyz_net = MLP(self.encode_xyz.o_dim,
                       angle_in_layer - 1,
                       hidden_dim,
                       xyz_feat_dim,
                       bn=bn)
        self.alpha_net = MLP(xyz_feat_dim,
                             num_layers - angle_in_layer + 1,
                             hidden_dim,
                             alpha_dim)
        self.rd_net = MLP(self.encode_angle.o_dim + xyz_feat_dim,
                           num_layers - angle_in_layer + 1,
                           hidden_dim,
                           rd_dim) 

    def forward(self, xyz, angle, sin_epoch = None):
        out = {}
        
        I = xyz.shape[-1]
        original_shape = tuple(list(xyz.shape)[:-1] + [-1])

        xyz = xyz.reshape((-1, I))
        angle = angle.reshape((-1, I))
        
        #位置编码
        xyz_encoded = self.encode_xyz(xyz)
        angle_encoded = self.encode_angle(angle)
        
        if self.hashgrid: xyz_encoded = mask(xyz_encoded , sin_epoch)
        
        xyz_features = self.xyz_net(xyz_encoded)
        alpha = self.alpha_net(xyz_features) #预测占用信息
        rd = self.rd_net(torch.cat((angle_encoded, xyz_features), dim = 1)) #预测反射信息
        
        #激活层
        if self.alpha_activation == 'sofrplus':
            alpha = self.softplus(alpha)
        elif self.alpha_activation == 'sigmoid':
            alpha = self.sigmoid(alpha * self.sigmoid_tightness)
        if self.softplus_rd:
            rd = self.softplus(rd)
            
        out["alpha"] = alpha.reshape(original_shape)
        out["rd"] = rd.reshape(original_shape)
        out["xyz_encoded"] = xyz_encoded.reshape(original_shape)
        return out
        
    def get_params(self, lr):
        params = [
            {"params": self.encoding_xyz_parameters(), "lr": lr},
            {"params": self.encoding_angle_parameters(), "lr": lr},
            {"params": self.xyz_net.parameters(), "lr": lr},
            {"params": self.alpha_net.parameters(), "lr": lr},
            {"params": self.rd_net.parameters(), "lr": lr}
        ]
        return params
