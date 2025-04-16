import csv
from dataclasses import fields

import numpy as np
import torch
from torchvision import transforms
from PIL import Image

#读取雷达数据
def read_fft_image(fft_path, numpy=False):
    '''## Read in FFT PNG as np.ndarray or torch.Tensor'''
    fft_img = Image.open(fft_path)

    # Remove metadata
    #fft_img [1152,1152]-> np.ndarray[1152,1152] -> torch.Tensorp[1,1152,1152]
    fft_img = np.asarray(fft_img)
    to_tensor = transforms.ToTensor()
    fft_img = to_tensor(fft_img.copy())
    
    return torch.squeeze(fft_img, dim=0)

# def read_LUT(LUT_path, square_gain=True):
#     """## Read-in, sort, linearlize, and square antenna gain profile as LUT"""
#     with open(LUT_path, 'r') as f:
#                 reader = csv.reader(f)
#                 lut = list(reader)
#                 lut = sort_LUT(np.array(lut, dtype=float))
#                 lut = linearize_LUT(lut)
#                 if square_gain: lut[:,1] = np.power(lut[:,1], 2)
#                 assert(lut.shape[1] == 2)
#                 return lut
# def sort_LUT(samples):
#     '''## Sort antena gain profile LUT by angular offset value (first column)'''
#     return samples[np.argsort(samples[:,0], axis=None), :]