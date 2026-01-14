"""
Bundle Adjustment: Refine Camera Poses and 3D Points
Jointly optimizes all camera poses and 3D points to minimize reprojection error
"""

import numpy as np
import cv2
import json
import os
from scipy.optimize import least_squares
import argparse


def project_point(P, point_3d):
    """
    Project a 3D point to 2D using camera projection matrix
    
    Args:
        P: Camera projection matrix (3x4)
        point_3d: 3D point (3,)
    
    Returns:
        point_2d: 2D projection (2,)
    """
    point_homogeneous = np.append(point_3d, 1.0)
    projected = P @ point_homogeneous
    if projected[2] != 0:
        projected_2d = projected[:2] / projected[2]
    else:
        projected_2d = projected[:2]
    return projected_2d


def compute_reprojection_error(params, K, observations, structure, n_cameras):
    """
    Compute reprojection errors for bundle adjustment
    
    Args:
        params: Flattened parameters [camera_params..., 3D_points...]
        K: Camera intrinsics (3x3)
        observations: List of (camera_idx, point_3d_idx, observed_2d)
        structure: Pre-computed structure for fast indexing
        n_cameras: Number of cameras
    
    Returns:
        errors: Reprojection errors (flattened)
    """
    n_points = len(structure['points_3d'])
    
    # Extract camera parameters
    camera_params = params[:n_cameras * 6].reshape(n_cameras, 6)
    
    # Extract 3D points
    points_3d = params[n_cameras * 6:].reshape(n_points, 3)
    
    errors = []
    for cam_idx, pt_idx, obs_2d in observations:
        # Get camera pose
        rvec = camera_params[cam_idx, :3]
        tvec = camera_params[cam_idx, 3:]
        
        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        
        # Build projection matrix
        P = K @ np.hstack([R, tvec.reshape(3, 1)])
        
        # Project 3D point
        pt_3d = points_3d[pt_idx]
        projected = project_point(P, pt_3d)
        
        # Compute error
        error = projected - obs_2d
        errors.extend(error.tolist())
    
    return np.array(errors)


def bundle_adjust(reconstruction_data):
    """
    Perform bundle adjustment on reconstruction
    
    Args:
        reconstruction_data: Dictionary with 'K', 'camera_poses', 'points_3d', 'observations'
    
    Returns:
        refined_data: Dictionary with refined camera poses and 3D points
    """
    K = np.array(reconstruction_data['K'])
    camera_poses = reconstruction_data['camera_poses']
    points_3d = np.array(reconstruction_data['points_3d'])
    observations = reconstruction_data['observations']
    
    n_cameras = len(camera_poses)
    n_points = len(points_3d)
    
    print(f"Bundle Adjustment:")
    print(f"  Cameras: {n_cameras}")
    print(f"  3D Points: {n_points}")
    print(f"  Observations: {len(observations)}")
    
    # Convert camera poses to parameter format (rotation vector + translation)
    camera_params = []
    camera_indices = sorted([int(k) for k in camera_poses.keys()])
    idx_to_cam = {idx: i for i, idx in enumerate(camera_indices)}
    
    for cam_idx in camera_indices:
        R = np.array(camera_poses[str(cam_idx)]['R'])
        t = np.array(camera_poses[str(cam_idx)]['t']).reshape(3)
        rvec, _ = cv2.Rodrigues(R)
        rvec = rvec.ravel()
        camera_params.append(np.concatenate([rvec, t]))
    
    camera_params = np.array(camera_params)
    
    # Prepare observations (map camera indices to sequential indices)
    obs_list = []
    for obs in observations:
        cam_orig_idx = obs['camera_idx']
        cam_seq_idx = idx_to_cam[cam_orig_idx]
        pt_idx = obs['point_3d_idx']
        obs_2d = np.array(obs['observation'])
        obs_list.append((cam_seq_idx, pt_idx, obs_2d))
    
    # If too many observations, sample them for faster optimization
    max_obs = 2000  # Limit observations to speed up
    if len(obs_list) > max_obs:
        print(f"  Sampling {max_obs} observations from {len(obs_list)} for faster optimization")
        import random
        random.seed(42)
        obs_list = random.sample(obs_list, max_obs)
    
    # Initial parameters
    initial_params = np.concatenate([
        camera_params.flatten(),
        points_3d.flatten()
    ])
    
    # Structure for fast lookup
    structure = {'points_3d': points_3d}
    
    print("\nRunning bundle adjustment...")
    
    # Run optimization using Trust Region Reflective (faster than LM)
    result = least_squares(
        compute_reprojection_error,
        initial_params,
        args=(K, obs_list, structure, n_cameras),
        method='trf',  # Trust Region Reflective - faster than LM
        verbose=2,
        max_nfev=30,   # Reduced iterations for faster convergence
        ftol=1e-3,     # Relaxed tolerance
        xtol=1e-6,
        gtol=1e-6
    )
    
    print(f"\nOptimization completed:")
    print(f"  Final cost: {result.cost:.4f}")
    print(f"  Status: {result.status}")
    print(f"  Iterations: {result.nfev}")
    
    # Extract refined parameters
    refined_params = result.x
    refined_camera_params = refined_params[:n_cameras * 6].reshape(n_cameras, 6)
    refined_points_3d = refined_params[n_cameras * 6:].reshape(n_points, 3)
    
    # Convert back to camera poses
    refined_camera_poses = {}
    for i, cam_idx in enumerate(camera_indices):
        rvec = refined_camera_params[i, :3]
        tvec = refined_camera_params[i, 3:]
        R, _ = cv2.Rodrigues(rvec)
        refined_camera_poses[str(cam_idx)] = {
            'R': R.tolist(),
            't': tvec.tolist()
        }
    
    return {
        'camera_poses': refined_camera_poses,
        'points_3d': refined_points_3d.tolist(),
        'K': K.tolist(),
        'initial_cost': np.sum(compute_reprojection_error(initial_params, K, obs_list, structure, n_cameras)**2),
        'final_cost': result.cost
    }


def load_reconstruction(input_dir):
    """Load reconstruction data from files"""
    # Load cameras
    cameras_file = os.path.join(input_dir, "cameras.json")
    with open(cameras_file, 'r') as f:
        camera_data = json.load(f)
    
    # Load points
    points_file = os.path.join(input_dir, "points_3d.txt")
    points_3d = []
    colors = []
    with open(points_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                points_3d.append([float(parts[0]), float(parts[1]), float(parts[2])])
                colors.append([int(parts[3]), int(parts[4]), int(parts[5])])
    
    # Load correspondences
    correspondences_file = os.path.join(input_dir, "correspondences.json")
    observations = []
    if os.path.exists(correspondences_file):
        with open(correspondences_file, 'r') as f:
            correspondences = json.load(f)
            for corr in correspondences:
                observations.append({
                    'camera_idx': corr['image_idx'],
                    'point_3d_idx': corr['point_3d_idx'],
                    'observation': corr['observation']
                })
    
    return {
        'K': np.array(camera_data['K']),
        'camera_poses': camera_data['camera_poses'],
        'points_3d': points_3d,
        'colors': colors,
        'observations': observations,
        'image_names': camera_data.get('image_names', [])
    }


def save_refined_reconstruction(refined_data, colors, output_dir, image_names):
    """Save refined reconstruction results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save refined point cloud
    points_file = os.path.join(output_dir, "points_3d_refined.txt")
    with open(points_file, 'w') as f:
        for pt, color in zip(refined_data['points_3d'], colors):
            f.write(f"{pt[0]} {pt[1]} {pt[2]} {color[0]} {color[1]} {color[2]}\n")
    
    # Save refined camera poses
    cameras_file = os.path.join(output_dir, "cameras_refined.json")
    camera_data = {
        'K': refined_data['K'],
        'camera_poses': refined_data['camera_poses'],
        'image_names': image_names,
        'optimization': {
            'initial_cost': refined_data['initial_cost'],
            'final_cost': refined_data['final_cost'],
            'improvement': refined_data['initial_cost'] - refined_data['final_cost']
        }
    }
    with open(cameras_file, 'w') as f:
        json.dump(camera_data, f, indent=2)
    
    print(f"\nRefined reconstruction saved to: {output_dir}/")
    print(f"  - Point cloud: {points_file}")
    print(f"  - Camera poses: {cameras_file}")
    print(f"  - Cost improvement: {refined_data['initial_cost']:.2f} -> {refined_data['final_cost']:.2f}")


def main():
    parser = argparse.ArgumentParser(description='Bundle Adjustment for SfM')
    parser.add_argument(
        '--input',
        type=str,
        default='output_multiview',
        help='Input directory with reconstruction results'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output_bundle_adjusted',
        help='Output directory for refined reconstruction'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input directory not found: {args.input}")
        return
    
    print("=" * 60)
    print("Bundle Adjustment")
    print("=" * 60)
    
    # Load reconstruction
    print(f"\nLoading reconstruction from: {args.input}")
    reconstruction = load_reconstruction(args.input)
    
    if len(reconstruction['observations']) == 0:
        print("Error: No observations found. Run multi-view reconstruction first.")
        return
    
    # Run bundle adjustment
    refined_data = bundle_adjust(reconstruction)
    
    # Save results
    save_refined_reconstruction(
        refined_data,
        reconstruction['colors'],
        args.output,
        reconstruction['image_names']
    )
    
    print("\n" + "=" * 60)
    print("Bundle Adjustment Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
