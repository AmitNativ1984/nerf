import torch
import torch.nn as nn

def compute_query_points_from_rays(
        ray_origins: torch.Tensor,
        ray_directions: torch.Tensor,
        near_threshold: float,
        far_threshold: float,
        num_samples: int,
        randomize: bool = True
        ) -> (torch.Tensor, torch.Tensor):
    """
        Compute query 3D points along each ray. The near and far thresholds indicate the bounds
        within which 3D points are to be sampled.
        There are two ways to perform this sampling: 
        1. uniformly sample along each ray
        2. Split the rays into evenly spaced intervals and sample randomly within these segments
    """

    depth_values = torch.linspace(near_threshold, far_threshold, num_samples).to(ray_origins)
    if randomize:
        bin_width = (far_threshold - near_threshold) / num_samples
        # random values to add to each bin:
        noise = torch.rand_like(depth_values) * bin_width
        depth_values += noise

    # Compute the query points: 
    # These are points along the ray with length specified by depth_values
    query_points = ray_origins[..., None, :] + ray_directions[..., None, :] * depth_values[..., None]

    # query_points [W, H, num_samples, 3]
    # check depth_values [num_samples]
    return query_points, depth_values

def render_volume_density(
        raw_radiance_field: torch.Tensor,
        ray_origins: torch.Tensor,
        depth_values: torch.Tensor
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    """
        Render the radiance field of each ray, given the raw radiance field, 
        ray origins and depth values

        Args:
            raw_radiance_field: (W, H, num_samples, 4): the radiance field at query location (XYZ) along each ray.
                The last dimension contains the emitted RGB value and volume density (denoted :math:`\sigma` in the paper)
            ray_origins: (W, H, 3): the origin of each ray
            depth_values: (num_samples): the depth values along each ray

        Returns:
            rgb: (W, H, 3): the rendered RGB values along each ray
            depth_map: (W, H): the rendered depth map
            acc_map: (W, H): the accumulated transmittance along each ray
    """

    sigma = nn.functional.relu(raw_radiance_field[..., -1]) # volume density - adding relu to assure positivity
    rgb = torch.sigmoid(raw_radiance_field[..., :3]) # RGB values (between 0 and 1)

    # computer the distance between each consecutive pair of points along the ray
    dist = depth_values[..., 1:] - depth_values[..., :-1]
    dist = torch.cat([dist, 1e10 * torch.ones_like(dist[..., :1])], dim=-1) # add a large value at the end to account for infinite distance at the end of the ray

    # Predict density of each sample along each ray. Higher values imply
    # higher likelihood of being absorbed at this post. [n_rays, n_samples]
    exp_term = torch.exp(-sigma * dist) # [n_rays, n_samples]
       
    # compute the accumulated transmittance along each ray
    transmittance = cumprod_exclusive(exp_term + 1e-10) # [n_rays, n_samples]
       
    # compute the weights for each ray sample
    alpha = 1. - exp_term # transmittance
    weights = alpha * transmittance # [n_rays, n_samples]

    rgb = weights[..., None] * rgb # [H, W, n_samples, 3]
    rgb = torch.sum(rgb, dim=-2) # sum along the samples: [H, W, 3]
    weighted_depth_map = weights * depth_values #[H, W, n_samples]
    weighted_depth_map = torch.sum(weighted_depth_map, dim=-1) # sum along the samples: [H, W]
    # accumulate the weights along the ray. Is this the accumulative transmittance?
    acc_map = torch.sum(weights, dim=-1) # [n_rays]

    return rgb, weighted_depth_map, acc_map


def cumprod_exclusive(tensor: torch.Tensor) -> torch.Tensor:
    """
        Compute the cumulative product of a tensor excluding the current element
    """

    # compute the cumulative product:
    # [a, b, c, d] -> [a, a*b, a*b*c, a*b*c*d]
    cumprod = tensor.cumprod(dim=-1) 
    
    # roll the tensor along the last dimension by `1` element:
    # [a, a*b, a*b*c, a*b*c*d] -> [a*b*c*d, a, a*b, a*b*c]
    cumprod = torch.roll(cumprod, 1, dims=-1)

    # set the first element to 1, as this is what tf.cumprod(..., exclusive=True) does
    cumprod[..., 0] = 1.

    return cumprod