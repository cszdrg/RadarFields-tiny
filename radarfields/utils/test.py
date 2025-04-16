import numpy as np
import data
from PIL import Image
from torchvision import transforms
import torch


fft_img = Image.open("/root/RadarFields-tiny/radarFields/utils/000001.png")
fft_img = np.asarray(fft_img)
fft_img = fft_img[11:, :]
print(fft_img.shape)

to_tensor = transforms.ToTensor()
fft_img = to_tensor(fft_img.copy())
fft_img = torch.squeeze(fft_img, dim=0).T

print(fft_img.shape)