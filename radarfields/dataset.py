
from dataclasses import dataclass
from pathlib import Path
import json

import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from radarfields.sampler import get_azimuths, get_range_samples
from utils.data import read_fft_image, read_LUT
from utils.pose import range_to_world

@dataclass
class RadarDataset:
    device: str
    split: str

    # 数据存储地址
    data_path: str
    preprocess_file: str
    radar_dir: str = "radar"
    preprocess_dir: str = "preprocess_results"

    # 采样规格设置
    #雷达一圈的采样个数
    num_rays_radar: int = 200
    #角度上的采样个数
    num_fov_samples: int = 10
    #采样的最近距离
    min_range_bin: int = 1 # NOTE: inclusive
    #采样的最远距离
    max_range_bin: int = 1200 # NOTE: also inclusive
    #距离方位上的采样点数
    num_range_samples: int = 1199
    #按照天线方向图进行合并
    integrate_rays: bool = True
    #batchsize
    bs: int = 10

    # 可选设置
    sample_all_ranges: bool = False
    train_thresholded: bool = True
    reg_occ: bool = True
    additive: bool = False
    preload: bool = False
    square_gain: bool = True

    # 雷达内参
    #雷达的距离单元数量
    num_range_bins: int = 7536
    #雷达一个距离的长度
    bin_size_radar: float = 0.044
    #雷达转一圈的测量数量
    num_azimuths_radar: int = 400
    #雷达波束水平和垂直的角度
    opening_h: float = 1.8
    opening_v: float = 40.0

    def __post_init__(self):
        self.training = self.split == "train"
        self.intrinsics_radar = (self.opening_h, self.opening_v)
        self.range_bounds = (self.min_range_bin, self.max_range_bin)

        # 加载json文件
        #../../preprocess_results/preprocess_radar.json
        self.project_root = Path(__file__).parent.parent
        preprocess_path = self.project_root / self.preprocess_dir
        with open(preprocess_path / self.preprocess_file.strip("\""),) as f:
            print(f"Loading in data from: {self.preprocess_file}")
            preprocess = json.load(f)
        self.preprocess = preprocess

        # Sampler
        #josn的数据1:train_indeces/test_indeces 索引（一个FFT/位姿）
        self.indices = preprocess[self.split + '_indices']
        self.sampler = SubsetRandomSampler(self.indices) #进行随机采样

        # json数据2:偏移和缩放
        self.offsets = torch.tensor(preprocess["offsets"], device=self.device)
        self.scalers = torch.tensor(preprocess["scalers"], device=self.device)

        # json数据3:位姿
        self.poses_radar = []
        for f in tqdm.tqdm(preprocess["radar2worlds"], desc=f"Loading radar poses"):
            pose_radar = np.array(f, dtype=np.float32) # [4, 4]
            self.poses_radar.append(pose_radar)

        # json数据4:FFT数据 横轴为角度 纵轴为距离 （方图）
        fft_path = self.project_root / self.data_path / self.radar_dir
        fft_frames = preprocess["timestamps_radar"]
        self.timestamps = fft_frames
        self.fft_frames = []
        #原有代码可以选择thresholded_fft数据进行训练
        #FFT数据的路径：../../data_path/radar/帧名
        for fft_frame in tqdm.tqdm(fft_frames, desc=f"Loading FFT data"):
            raw_radar_fft = read_fft_image(fft_path / fft_frame)
            self.fft_frames.append(raw_radar_fft)

        # 占用信息
        #../../preprocess_results/occupancy_component/帧名
        if self.reg_occ:
            self.occ_frames = []
            occ_path = self.project_root / 'preprocess_results' / 'occupancy_component' / str(self.preprocess_file).split('.')[0]
            for fft_frame in tqdm.tqdm(fft_frames, desc=f"Loading occupancy components"):
                timestamp = str(fft_frame).split('.')[0] + '.npy'
                occupancy_component = torch.tensor(np.load(occ_path / timestamp), dtype=torch.float32)
                self.occ_frames.append(occupancy_component)
        
        # 如果不是训练 对水平所有射线进行采样 否则采样200个 ｜ 对所有距离方位上进行采样
        if not self.training: # Sample all azimuth-range bins during testing
            self.num_rays_radar = self.num_azimuths_radar
            self.num_range_samples = self.max_range_bin-self.min_range_bin+1
            self.sample_all_ranges = True
        if (self.max_range_bin-self.min_range_bin+1) ==self.num_range_samples: self.sample_all_ranges = True
        
        # Radiation pattern look-up table (LUT) for integrating beam samples
        # 读取雷达天线的方向图：
        elevation_LUT_path = (self.project_root / self.data_path).parent / "elevation.csv" #俯角方向图
        azimuth_LUT_path = (self.project_root / self.data_path).parent / "azimuth.csv"     #仰角方向图
        print(f"loading elevation radiation pattern from {elevation_LUT_path}")
        self.elevation_LUT_linear = read_LUT(elevation_LUT_path)
        print(f"loading azimuth radiation pattern from {azimuth_LUT_path}")
        self.azimuth_LUT_linear = read_LUT(azimuth_LUT_path)

        # 位姿、FFT数据、占用信息 转换为torch.Tensor
        self.poses_radar = torch.from_numpy(np.stack(self.poses_radar, axis=0)) # [B, 4, 4]
        self.fft_frames = torch.from_numpy(np.stack(self.fft_frames, axis=0)).float() # [B, H, W]
        if self.reg_occ: self.occ_frames = torch.stack(self.occ_frames, dim=0) # [B, H, W]

        # If enabled, preload all data onto GPU memory
        if self.preload:
            self.poses_radar = self.poses_radar.to(self.device)
            self.fft_frames = self.fft_frames.to(self.device)
            if self.reg_occ: self.occ_frames = self.occ_frames.to(self.device)

    def collate(self, index):
        '''
        ## Custom collate_fn to collate a batch of raw FFT data.
        ### Also samples range-azimuth bins for each FFT frame in the batch.
        '''
        B = len(index) # index is a list of length [B] 多少张FFT图片
        N = self.num_rays_radar    #360度一圈 采样多少个射线
        S = self.num_fov_samples   #每个射线 在角度上采样多少
        R = self.num_range_samples #在距离上 包含多少范围

        results = {}

        # 雷达位姿：radar -> world
        poses_radar = self.poses_radar[index].to(self.device)  # [B, 4, 4]

        # Sample azimuth angles at which to query model (in terms of idx from 0 -> 400)
        # (sample all azimuths during test)
        azimuth_samples = get_azimuths(B, N, self.num_azimuths_radar, self.device, all=not self.training)
        #(B,N)

        # Sample range bins at which to query model, and convert to radial distances in meters
        range_samples_idx = get_range_samples(B, self.num_rays_radar,
                                          self.num_range_samples,
                                          self.range_bounds,
                                          device=self.device,
                                          all=self.sample_all_ranges) # [B, N, R]
        # B个位姿 N个射线 R个距离
        range_samples = range_to_world(range_samples_idx, self.bin_size_radar) # [B, N, R]
        # 每条射线都细分为s条
        range_samples_expanded = range_samples.repeat_interleave(S, dim=1) # [B, N*S, R]

        # Crop radar FFT frames & occupancy components to provided bin ranges
        # NOTE: bins are 1-indexed, tensors are 0-indexed
        fft = self.fft_frames[index].to(self.device)
        
        fft = fft[:,:,self.min_range_bin-1:self.max_range_bin] # [B, H, W]
        if self.reg_occ:
            occ = self.occ_frames[index].to(self.device)
            occ = occ[:,:,self.min_range_bin-1:self.max_range_bin] # [B, H, W]

        results.update({
            "intrinsics": self.intrinsics_radar, # (fov_h, fov_v)
            "bs": B,
            "num_rays_radar": N,
            "num_fov_samples": S,
            "num_range_samples": R,
            "indices": index, # batch FFT frame indices
            "poses": poses_radar, # [B, 4, 4]
            "azimuths": azimuth_samples, # [B, N]
            "ranges": range_samples_expanded, # [B, N*S, R]
            "ranges_original": range_samples, # [B, N, R]
            "offsets": self.offsets,
            "scalers": self.scalers,
            "timestamps": [self.timestamps[i] for i in index], # batch FFT frame timestamps
            "azim_LUT": self.azimuth_LUT_linear,
            "elev_LUT": self.elevation_LUT_linear,
            "fft": fft # [B, H, R]
            })
        if self.reg_occ: results["occ"] = occ # [B, H, R]

        if not self.training: return results # We test on all FFT bins

        # 如果是训练 在fft图像中，只取出训练的射线
        num_bins = self.max_range_bin-self.min_range_bin+1
        results["fft"] = torch.gather(fft, 1, azimuth_samples[...,None].expand((B,N,num_bins)))
        if self.reg_occ: results["occ"] = torch.gather(occ, 1, azimuth_samples[...,None].expand((B,N,num_bins)))

        #如果采样所有的距离 直接返回
        if self.sample_all_ranges: return results

        # 否则只返回采样距离的射线
        results["fft"] = torch.gather(results["fft"], 2, range_samples_idx-self.min_range_bin)
        if self.reg_occ: results["occ"] = torch.gather(results["occ"], 2, range_samples_idx-self.min_range_bin)
        return results
    
    def dataloader(self, batch_size):
        size = len(self.poses_radar)
        loader = DataLoader(
            list(range(size)),
            batch_size=batch_size,
            collate_fn=self.collate,
            num_workers=0,
            sampler=self.sampler,
            pin_memory=False
        )
        loader._data = self
        loader.num_poses = size
        return loader
