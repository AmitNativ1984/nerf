import os
import wget
import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
import torch

if not os.path.exists('data/tiny_nerf_data.npz'):
    # download tiny_nerf_data.npz
    wget.download('http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz', 'data/tiny_nerf_data.npz')    

def get_rays(height: int, width: int, focal_length: int, cam2world: torch.Tensor):
    """
    Get rays origin and direction in the world coordinate system, using pinhole camera model.
    Args:
        height (int): image height
        width (int): image width
        focal_length (int): focal length
        cam2world (torch.Tensor): camera to world transformation matrix
    Returns:
        rays_origins (torch.Tensor): rays origins in the world coordinate system
        rays_directions (torch.Tensor): rays directions in the world coordinate system
    """

    # Apply pinhole camera model to gather ray origins and directions:
    x = torch.arange(width, dtype=torch.float32).to(cam2world)
    y = torch.arange(height, dtype=torch.float32).to(cam2world)
    i, j = torch.meshgrid(x,y, indexing='ij')

    # Get ray directions
    dirs = torch.stack([(j - width * 0.5) / focal_length, 
                        -(i - height * 0.5) / focal_length, 
                        -torch.ones_like(i)
                        ], dim=-1)
    
    # transform ray directions from camera space to world space
    rays_directions = torch.sum(dirs[..., None, :] * cam2world[:3, :3], dim=-1)

    # ray origins are the camera origins
    rays_origins = cam2world[:3, -1].expand(rays_directions.shape)

    return rays_origins, rays_directions

def get_minibatches(inputs: torch.Tensor, chunksize: int=1024*8):
    """
        Split the input ray bundle (given as the input tensor) to minibatches of size chunksize
    """

    minibatches = []
    for i in range(0, inputs.shape[0], chunksize):
        minibatches.append(inputs[i:i+chunksize])

    return minibatches




