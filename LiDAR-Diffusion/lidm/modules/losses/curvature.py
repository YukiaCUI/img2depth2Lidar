from functools import partial

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class Curvature(nn.Module):
    def __init__(self,dataset_config=dict()):
        super().__init__()
        
        fov = dataset_config.fov
        self.fov_up = fov[0] / 180.0 * np.pi  # field of view up in rad
        self.fov_down = fov[1] / 180.0 * np.pi  # field of view down in rad
        self.fov_range = abs(self.fov_down) + abs(self.fov_up)  # get field of view total in rad
        self.depth_scale = dataset_config.depth_scale
        self.depth_min, self.depth_max = dataset_config.depth_range
        self.log_scale = dataset_config.log_scale
        self.size = dataset_config['size']
        self.register_conversion()   

        # 计算曲率的卷积核
        self.conv_kernel = torch.tensor([1, 1, 1, 1, 1, -10, 1, 1, 1, 1, 1], dtype=torch.float32).view(1, 1, 1, -1).to('cuda')
    
    def register_conversion(self):
        scan_x, scan_y = np.meshgrid(np.arange(self.size[1]), np.arange(self.size[0]))
        scan_x = scan_x.astype(np.float64) / self.size[1]
        scan_y = scan_y.astype(np.float64) / self.size[0]

        yaw = (np.pi * (scan_x * 2 - 1))
        pitch = ((1.0 - scan_y) * self.fov_range - abs(self.fov_down))

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('cos_yaw', torch.cos(to_torch(yaw)))
        self.register_buffer('sin_yaw', torch.sin(to_torch(yaw)))
        self.register_buffer('cos_pitch', torch.cos(to_torch(pitch)))
        self.register_buffer('sin_pitch', torch.sin(to_torch(pitch)))

    def batch_range2xyz(self, imgs):
        batch_depth = (imgs * 0.5 + 0.5) * self.depth_scale
        if self.log_scale:
            batch_depth = torch.exp2(batch_depth) - 1
        batch_depth = batch_depth.clamp(self.depth_min, self.depth_max)

        batch_x = self.cos_yaw * self.cos_pitch * batch_depth
        batch_y = -self.sin_yaw * self.cos_pitch * batch_depth
        batch_z = self.sin_pitch * batch_depth
        batch_xyz = torch.cat([batch_x, batch_y, batch_z], dim=1)

        return batch_xyz
    
    def curvature(self, input_xyz):
        diff_x = F.conv2d(input_xyz[:, 0:1, :, :], self.conv_kernel, padding=(0, 5), groups=1)
        diff_y = F.conv2d(input_xyz[:, 1:2, :, :], self.conv_kernel, padding=(0, 5), groups=1)
        diff_z = F.conv2d(input_xyz[:, 2:3, :, :], self.conv_kernel, padding=(0, 5), groups=1)

        # 所有点的曲率
        cloud_curvature = diff_x ** 2 + diff_y ** 2 + diff_z ** 2

        # 筛选曲率
        B, _, H, W = input_xyz.shape
        cloud_neighbor_picked = torch.zeros((B, H, W), dtype=torch.bool, device=input_xyz.device)

        depth = torch.sqrt((input_xyz ** 2).sum(dim=1))

        diff = torch.sqrt(((input_xyz[:, :, :, 1:] - input_xyz[:, :, :, :-1]) ** 2).sum(dim=1))
        diff = torch.cat([diff, torch.zeros((B, H, 1), device=input_xyz.device)], dim=-1)  # 对齐维度

        mask1 = diff > 0.1
        depth_ratio = depth[:, :, 1:] / depth[:, :, :-1]
        diff_ratio = torch.sqrt(((input_xyz[:, :, :, 1:] - input_xyz[:, :, :, :-1] * depth_ratio.unsqueeze(1)) ** 2).sum(dim=1))
        diff_ratio = torch.cat([torch.ones((B, H, 1), device=input_xyz.device), diff_ratio], dim=-1)  # 对齐维度
        mask2 = diff_ratio < 0.1
        cloud_neighbor_picked[:, :, 5:-6] |= (mask1[:, :, 5:-6] & mask2[:, :, 5:-6])

        # 处理离群点
        diff2 = torch.sqrt(((input_xyz[:, :, :, :-1] - input_xyz[:, :, :, 1:]) ** 2).sum(dim=1))
        diff2 = torch.cat([diff2, torch.zeros((B, H, 1), device=input_xyz.device)], dim=-1)

        dis = (input_xyz ** 2).sum(dim=1)
        mask_outlier = (diff > 0.0002 * dis) & (diff2 > 0.0002 * dis)

        cloud_neighbor_picked |= mask_outlier
        cloud_curvature[:, 0, :, :][cloud_neighbor_picked] = float('inf')

        return cloud_curvature


    def forward(self, input):
        input = input / 2. + .5
        input_xyz = self.batch_range2xyz(input)
        curve = self.curvature(input_xyz)

        return curve
