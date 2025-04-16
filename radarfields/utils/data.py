import csv
from dataclasses import fields

import numpy as np
import torch
from torchvision import transforms
from PIL import Image

def read_fft_image(fft_path):
    '''## Read in FFT PNG as np.ndarray or torch.Tensor'''
    fft_img = Image.open(fft_path)

    # Remove metadata
    #fft_img [1152,1152]-> np.ndarray[1152,1152] -> torch.Tensorp[1,1152,1152]
    fft_img = np.asarray(fft_img)
    to_tensor = transforms.ToTensor()
    fft_img = to_tensor(fft_img.copy())
    return torch.squeeze(fft_img, dim=0).T

def read_LUT(LUT_path, square_gain=True):
    #获取雷达的天线方向增益图
    """## Read-in, sort, linearlize, and square antenna gain profile as LUT"""
    with open(LUT_path, 'r') as f:
                reader = csv.reader(f)
                lut = list(reader)
                lut = sort_LUT(np.array(lut, dtype=float))
                lut = linearize_LUT(lut)
                if square_gain: lut[:,1] = np.power(lut[:,1], 2)
                assert(lut.shape[1] == 2)
                return lut

def sort_LUT(samples):
    '''## Sort antena gain profile LUT by angular offset value (first column)'''
    return samples[np.argsort(samples[:,0], axis=None), :]

def linearize_LUT(samples):
    '''## linearize antenna gain values in LUT (second column) from dBic'''
    y_pts = samples[:,1]
    linearized_samples = samples.copy()
    linearized_samples[:, 1] = np.power(10, y_pts/10.0)
    return linearized_samples