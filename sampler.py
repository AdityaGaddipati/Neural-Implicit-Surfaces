import math
from typing import List

import torch
from ray_utils import RayBundle
from pytorch3d.renderer.cameras import CamerasBase


# Sampler which implements stratified (uniform) point sampling along rays
class StratifiedRaysampler(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.n_pts_per_ray = cfg.n_pts_per_ray
        self.min_depth = cfg.min_depth
        self.max_depth = cfg.max_depth

    def forward(
        self,
        ray_bundle,
    ):
        # TODO (1.4): Compute z values for self.n_pts_per_ray points uniformly sampled between [near, far]
        # z_vals, _ = torch.FloatTensor(self.n_pts_per_ray,1).uniform_(self.min_depth, self.max_depth).sorted()
        z_vals = torch.linspace(self.min_depth, self.max_depth, self.n_pts_per_ray)
        z_vals = z_vals.reshape(-1,1)

        # TODO (1.4): Sample points from z values
        
        num_rays = ray_bundle.origins.shape[0]
        sample_lengths = (torch.ones(num_rays,self.n_pts_per_ray,1)*z_vals).to(torch.cuda.current_device())
        sample_points = ray_bundle.origins.unsqueeze(1) + ray_bundle.directions.unsqueeze(1)*sample_lengths

        # print(ray_bundle.origins.shape)
        # print(ray_bundle.directions.shape)
        # print(z_vals.shape)
        # print(sample_points.shape)
        # print(sample_lengths.shape)

        # Return
        return ray_bundle._replace(
            sample_points=sample_points,
            # sample_lengths=z_vals * torch.ones_like(sample_points[..., :1]),
            sample_lengths=sample_lengths,

        )


sampler_dict = {
    'stratified': StratifiedRaysampler
}