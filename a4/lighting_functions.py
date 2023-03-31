import numpy as np
import torch
from ray_utils import RayBundle

def phong(
    normals,
    view_dirs, 
    light_dir,
    params,
    colors
):
    # TODO: Implement a simplified version Phong shading
    # Inputs:
    #   normals: (N x d, 3) tensor of surface normals
    #   view_dirs: (N x d, 3) tensor of view directions
    #   light_dir: (3,) tensor of light direction
    #   params: dict of Phong parameters
    #   colors: (N x d, 3) tensor of colors
    # Outputs:
    #   illumination: (N x d, 3) tensor of shaded colors
    #
    # Note: You can use torch.clamp to clamp the dot products to [0, 1]
    # Assume the ambient light (i_a) is of unit intensity 
    # While the general Phong model allows rerendering with multiple lights, 
    # here we only implement a single directional light source of unit intensity

    ka = params['ka']
    kd = params['kd']
    ks = params['ks']
    alpha = params['n']

    Rm = 2*torch.mm(normals, (light_dir).reshape(3,1)) * normals - light_dir
    scale = kd*torch.mm(normals, (light_dir).reshape(3,1)) + ks* torch.pow((Rm*view_dirs).sum(dim=1).reshape(-1,1),alpha)
    illumination = ka + scale*colors

    return illumination

relighting_dict = {
    'phong': phong
}
