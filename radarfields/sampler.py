import torch

# from utils.train import get_rotation_matrix

def get_azimuths(B, N, num_azimuths, device, all=False):
    '''
    ## Sample N different azimuth angles in [0,num_azimuths) for\
    each of the B radar frames in a batch
    # 对b采样图片, 采样n个不同的角度
    # N:采样的个数 num_azimuths:400(全部的射线
    '''
    if all: # Sample all azimuth angles of the radar 400
        azim_samples = torch.arange(0,num_azimuths, device=device)[None].expand((B,num_azimuths))
    
    else: azim_samples = torch.sort(torch.randint(0, num_azimuths, (B, N), device=device))[0]
    return azim_samples # [B, N]

def get_range_samples(B, N, num_samples, bounds, device, all=False):
    '''
    ## Randomly samples num_samples different ranges for each of the\
    B*N rays. Samples are integer values in [bounds[0],bounds[1]],\
    inclusive, which correspond to range bin indices in the radar FFT data.

    :param B (int): batch size
    :param N (int): number of azimuth rays per batch
    :param num_samples (int): number of range bins to sample
    :param bounds (tuple): [lower, upper] bounds on sampled indices
    :param all (bool): if True, return all indices within bounds
    :return samples (torch.Tensor): [B, N, R] range samples; R=num_samples
    采样num_samples个不同的距离
    '''
    min, max = bounds[0], bounds[1]
    assert(min > 0)
    num_bins = max-min+1
    
    if all: return torch.arange(min,max+1,device=device)[None,None].expand((B, N, num_bins))

    assert(num_samples <= num_bins)

    samples = torch.rand(B, num_bins, device=device).argsort(dim=-1)[:, :num_samples] + min
    samples, _ = torch.sort(samples, dim=-1)
    return samples[:, None, :].expand((B, N, num_samples))