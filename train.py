import numpy as np
import torch
import matplotlib.pyplot as plt

from data.dataloader import get_rays, get_minibatches
from nerf_utils.nerf_optimizers import compute_query_points_from_rays, render_volume_density

from model import TinyNerfModel, FullNerfModel, PositionalEncoding

from tqdm import tqdm

def run_one_iter_of_nerf(height, width, focal_length, pose_cam2world,
                         near_threshold, far_threshold, num_samples_per_ray,
                         positional_encoding,
                         nerf_model):
    """
        Run one iteration of NeRF optimization
    """

    ############################################
    # GET RAYS                                 #
    ############################################
    with torch.no_grad():
        ray_origins, ray_directions = get_rays(height, width, focal_length, pose_cam2world)
    
    
    ############################################
    # COMPUTE QUERY POINTS                     #
    # (points that are sampled along each ray) #
    ############################################
    query_points, depth_values = compute_query_points_from_rays(ray_origins,
                                                                ray_directions,
                                                                near_threshold,
                                                                far_threshold,
                                                                num_samples_per_ray,
                                                                randomize=True)

    # reshape query points to (H*W*num_samples, 3)
    query_points_flattened = query_points.reshape(-1, 3)
    
    ############################################
    # SPLIT RAYS TO MINIBATCHES                #
    ############################################
    query_points_batches = get_minibatches(query_points_flattened, chunksize=1024*8)

    #####################################################
    # RUN NERF MODEL ON ALL MINIBATCHES, CONCAT RESULTS #
    #####################################################
    predicted_radiance_field = []
    for batch in query_points_batches:
        # Apply positional encoding to the input points
        encoded_points = positional_encoding(batch)
        # Run the tiny nerf model
        raw_radiance_field = nerf_model(encoded_points)
        # Concatenate the results
        predicted_radiance_field.append(raw_radiance_field)
    
    # Concatenate the minibatch results
    predicted_radiance_field = torch.cat(predicted_radiance_field, dim=0)

    # reshape the predicted radiance field to (H, W, num_samples_per_ray, 4)
    predicted_radiance_field = predicted_radiance_field.reshape(height, width, num_samples_per_ray, 4)

    # render the perdicted radiance field through the scene, towards the camera
    rgb_predicted, weighted_depth_map, acc_predicted = render_volume_density(predicted_radiance_field,
                                                                              ray_origins,
                                                                              depth_values)
    
    return rgb_predicted, weighted_depth_map, acc_predicted

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    data = np.load('data/tiny_nerf_data.npz')
    images = data['images']
    poses = data['poses']
    focal = data['focal']

    print(f'Images.shape: {images.shape}')
    print(f'Poses shape: {poses.shape}')
    print(f'Focal length: {focal}')

    n_training = 100
    testimg_idx = 101
    testimg, testpose = images[testimg_idx], poses[testimg_idx]
    plt.figure('test image')
    plt.imshow(testimg), 
    print('Pose')
    print(testpose)


    #############################################
    # Calculate the camera origin and direction #
    #############################################
    # The poses matrices store the transformation from camera space to world space

    cam_dirs = np.stack([pose[:3, :3] @ np.array([0,0,-1]) for pose in poses])
    cam_origins = np.stack([pose[:3, -1] for pose in poses])

    ax = plt.figure('camera origin and direction').add_subplot(projection='3d')
    ax.quiver(cam_origins[:,0], cam_origins[:,1], cam_origins[:,2], cam_dirs[:,0], cam_dirs[:,1], cam_dirs[:,2], length=0.5, normalize=True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('z')
    
    
    #gather data as tensors
    images = torch.from_numpy(data['images'][:n_training]).to(device)
    poses = torch.from_numpy(data['poses']).to(device)
    focal_length = torch.from_numpy(data['focal']).to(device)
    testimg = torch.from_numpy(data['images'][testimg_idx]).to(device)
    testpose = torch.from_numpy(data['poses'][testimg_idx]).to(device)

    # get rays
    with torch.no_grad():
        ray_origins, ray_directions = get_rays(height=images.shape[1], width=images.shape[2], focal_length=focal_length, cam2world=poses[testimg_idx])
    
    height=images.shape[1]
    width=images.shape[2]
    print('Ray Origin')
    print(ray_origins.shape)
    print(ray_origins[height // 2, width // 2, :])
    print('')

    print('Ray Direction')
    print(ray_directions.shape)
    print(ray_directions[height // 2, width // 2, :])
    print('')
    
    
    #################################
    # TRAIN NERF ON ALL RAY BUNDLES #
    #################################
    lr = 5e-3
    iterations = 15000   # num of iterations to run training
    num_encoding_functions = 10
    depth_samples_per_ray = 128
    near_threshold = 2.0
    far_threshold = 6.0
    
    display_every = 100

    #########
    # MODEL #
    #########
    nerf_model = TinyNerfModel(filter_size=128, num_encoding_functions=num_encoding_functions).to(device)
    # nerf_model = FullNerfModel(filter_size=256, num_encoding_functions=num_encoding_functions).to(device)
    positional_encoding = PositionalEncoding(n_freq=num_encoding_functions).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(nerf_model.parameters(), lr=lr)

    # Seed for repeatability
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    for i in tqdm(range(iterations)):
        # Randomly pick an image and pose
        target_idx = np.random.randint(0, n_training)
        target_img = images[target_idx].to(device)
        target_pose = poses[target_idx].to(device)

        # Run one iteration of NeRF optimization
        rgb_predicted, weighted_predicted, acc_predicted = run_one_iter_of_nerf(height=images.shape[1],
                                                                                width=images.shape[2],
                                                                                focal_length=focal_length,
                                                                                pose_cam2world=target_pose,
                                                                                near_threshold=near_threshold,
                                                                                far_threshold=far_threshold,
                                                                                num_samples_per_ray=depth_samples_per_ray,
                                                                                positional_encoding=positional_encoding,
                                                                                nerf_model=nerf_model)
                                                                                
        # Compute the loss:
        # The loss is the mse between rgb_predicted and target_img
        loss = torch.nn.functional.mse_loss(rgb_predicted, target_img)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
                                                                                
    # Display training progress on test image:

    rgb_predicted, weighted_predicted_depth, acc_predicted = run_one_iter_of_nerf(height=images.shape[1],
                                                                        width=images.shape[2],
                                                                        focal_length=focal_length,
                                                                        pose_cam2world=testpose,
                                                                        near_threshold=near_threshold,
                                                                        far_threshold=far_threshold,
                                                                        num_samples_per_ray=depth_samples_per_ray,
                                                                        positional_encoding=positional_encoding,
                                                                        nerf_model=nerf_model)
    loss = torch.nn.functional.mse_loss(rgb_predicted, testimg)
    plt.figure('rendered image')
    plt.imshow(rgb_predicted.detach().cpu().numpy())

    plt.figure('predicted depth')
    plt.imshow(weighted_predicted_depth.detach().cpu().numpy())

    plt.figure('acc_predicted')
    plt.imshow(acc_predicted.detach().cpu().numpy())
    
    plt.show()