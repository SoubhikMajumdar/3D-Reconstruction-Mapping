"""
Multi-View Stereo (MVS): Dense 3D Reconstruction
Creates dense point clouds from sparse SfM reconstruction using stereo matching
"""

import cv2
import numpy as np
import json
import os
import argparse
from collections import defaultdict


def load_reconstruction_data(input_dir):
    """Load bundle-adjusted reconstruction data"""
    # Load camera poses
    cameras_file = os.path.join(input_dir, "cameras_refined.json")
    with open(cameras_file, 'r') as f:
        camera_data = json.load(f)
    
    K = np.array(camera_data['K'])
    camera_poses = camera_data['camera_poses']
    image_names = camera_data.get('image_names', [])
    
    # Load images directory
    images_dir = os.path.join(os.path.dirname(input_dir), "..", "dataset", "south-building", "images")
    if not os.path.exists(images_dir):
        # Try alternative path
        images_dir = "dataset/south-building/images"
    
    return {
        'K': K,
        'camera_poses': camera_poses,
        'image_names': image_names,
        'images_dir': images_dir
    }


def load_image(images_dir, image_name):
    """Load image from directory"""
    img_path = os.path.join(images_dir, image_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Could not load image: {img_path}")
    return img


def create_stereo_matcher():
    """Create stereo matcher for depth estimation"""
    # Use Semi-Global Block Matching (SGBM) for better quality
    num_disparities = 16 * 10  # Must be divisible by 16
    block_size = 11
    
    stereo_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * 3 * block_size ** 2,
        P2=32 * 3 * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    
    return stereo_matcher


def compute_depth_map(img1, img2, K, R1, t1, R2, t2, rectified=True):
    """
    Compute depth map from two images using stereo matching
    
    Args:
        img1, img2: Input images (BGR)
        K: Camera intrinsics
        R1, t1: Camera 1 pose
        R2, t2: Camera 2 pose
        rectified: Whether to perform stereo rectification
    
    Returns:
        depth_map: Depth map (same size as images)
        disparity_map: Disparity map
    """
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
    
    if rectified:
        # Stereo rectification
        R = R2 @ R1.T
        t = t2.reshape(3, 1) - R @ t1.reshape(3, 1)
        
        R1_rect, R2_rect, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            K, None, K, None, gray1.shape[::-1], R, t, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.9
        )
        
        # Rectify images
        map1x, map1y = cv2.initUndistortRectifyMap(K, None, R1_rect, P1, gray1.shape[::-1], cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(K, None, R2_rect, P2, gray2.shape[::-1], cv2.CV_32FC1)
        
        gray1_rect = cv2.remap(gray1, map1x, map1y, cv2.INTER_LINEAR)
        gray2_rect = cv2.remap(gray2, map2x, map2y, cv2.INTER_LINEAR)
    else:
        gray1_rect = gray1
        gray2_rect = gray2
        Q = None
    
    # Create stereo matcher
    stereo_matcher = create_stereo_matcher()
    
    # Compute disparity
    disparity = stereo_matcher.compute(gray1_rect, gray2_rect).astype(np.float32) / 16.0
    
    # Filter invalid disparities
    filtered_disp = np.where(disparity > 0, disparity, 0.0)
    
    # Convert disparity to depth
    if Q is not None and rectified:
        # Use Q matrix from rectification
        points_3d = cv2.reprojectImageTo3D(filtered_disp, Q)
        depth_map = points_3d[:, :, 2]
    else:
        # Simple depth from disparity (for unrectified)
        focal_length = K[0, 0]
        baseline = np.linalg.norm(t2 - t1) if rectified else 1.0
        depth_map = np.zeros_like(filtered_disp)
        valid = filtered_disp > 0
        depth_map[valid] = focal_length * baseline / (filtered_disp[valid] + 1e-6)
        depth_map[~valid] = 0
    
    return depth_map, filtered_disp


def depth_map_to_point_cloud(depth_map, img, K, R, t, mask=None):
    """
    Convert depth map to 3D point cloud
    
    Args:
        depth_map: Depth map (HxW)
        img: Color image (HxWx3)
        K: Camera intrinsics (3x3)
        R: Rotation matrix (3x3)
        t: Translation vector (3x1)
    
    Returns:
        points_3d: 3D points (Nx3)
        colors: RGB colors (Nx3)
    """
    h, w = depth_map.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # Create pixel coordinates
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    # Convert to camera coordinates
    z = depth_map
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # Stack coordinates
    points_cam = np.stack([x, y, z], axis=-1)
    
    # Filter invalid points
    if mask is not None:
        valid = (z > 0) & (z < 1000.0) & mask
    else:
        valid = (z > 0) & (z < 1000.0)
    
    points_cam = points_cam[valid]
    
    # Transform to world coordinates
    points_world = (R @ points_cam.T).T + t.reshape(1, 3)
    
    # Extract colors
    if len(img.shape) == 3:
        colors = img[valid]
        # Convert BGR to RGB
        colors = colors[:, [2, 1, 0]]
    else:
        colors = np.zeros((len(points_world), 3))
    
    return points_world, colors


def mvs_dense_reconstruction(input_dir, images_dir, output_dir, num_image_pairs=3):
    """
    Perform MVS dense reconstruction
    
    Args:
        input_dir: Directory with bundle-adjusted reconstruction
        images_dir: Directory with input images
        output_dir: Output directory for dense reconstruction
        num_image_pairs: Number of image pairs to process
    """
    print("=" * 60)
    print("Multi-View Stereo (MVS): Dense Reconstruction")
    print("=" * 60)
    
    # Load reconstruction data
    print("\nLoading reconstruction data...")
    data = load_reconstruction_data(input_dir)
    K = data['K']
    camera_poses = data['camera_poses']
    image_names = data['image_names']
    
    # Get camera indices
    camera_indices = sorted([int(k) for k in camera_poses.keys()])
    print(f"Cameras found: {len(camera_indices)}")
    print(f"Camera indices: {camera_indices}")
    
    # Limit number of pairs
    num_image_pairs = min(num_image_pairs, len(camera_indices) - 1)
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_points = []
    all_colors = []
    
    # Process image pairs
    print(f"\nProcessing {num_image_pairs} image pairs for dense reconstruction...")
    
    for i in range(num_image_pairs):
        idx1 = camera_indices[i]
        idx2 = camera_indices[i + 1]
        
        print(f"\nProcessing pair {i+1}/{num_image_pairs}: Images {idx1} and {idx2}")
        
        # Load images
        img1_name = image_names[idx1] if idx1 < len(image_names) else f"P1180{141+idx1}.JPG"
        img2_name = image_names[idx2] if idx2 < len(image_names) else f"P1180{141+idx2}.JPG"
        
        img1 = load_image(images_dir, img1_name)
        img2 = load_image(images_dir, img2_name)
        
        if img1 is None or img2 is None:
            print(f"  Skipping pair {idx1}-{idx2}: Could not load images")
            continue
        
        # Get camera poses
        pose1 = camera_poses[str(idx1)]
        pose2 = camera_poses[str(idx2)]
        
        R1 = np.array(pose1['R'])
        t1 = np.array(pose1['t']).reshape(3, 1) if isinstance(pose1['t'][0], (int, float)) else np.array(pose1['t'])
        R2 = np.array(pose2['R'])
        t2 = np.array(pose2['t']).reshape(3, 1) if isinstance(pose2['t'][0], (int, float)) else np.array(pose2['t'])
        
        if len(t1.shape) == 1:
            t1 = t1.reshape(3, 1)
        if len(t2.shape) == 1:
            t2 = t2.reshape(3, 1)
        
        print(f"  Computing depth map...")
        
        try:
            # Compute depth map (use simple method without rectification for speed)
            depth_map, disparity = compute_depth_map(img1, img2, K, R1, t1, R2, t2, rectified=False)
            
            # Subsample depth map for efficiency (optional)
            scale = 4  # Downsample by 4x
            if scale > 1:
                h, w = depth_map.shape
                depth_map_small = cv2.resize(depth_map, (w // scale, h // scale), interpolation=cv2.INTER_NEAREST)
                img1_small = cv2.resize(img1, (w // scale, h // scale))
            else:
                depth_map_small = depth_map
                img1_small = img1
            
            # Convert to point cloud
            print(f"  Converting depth map to point cloud...")
            points_3d, colors = depth_map_to_point_cloud(depth_map_small, img1_small, K, R1, t1)
            
            print(f"  Generated {len(points_3d)} points from pair {idx1}-{idx2}")
            
            all_points.append(points_3d)
            all_colors.append(colors)
            
        except Exception as e:
            print(f"  Error processing pair {idx1}-{idx2}: {e}")
            continue
    
    # Combine all point clouds
    if len(all_points) > 0:
        print(f"\nCombining {len(all_points)} point clouds...")
        combined_points = np.vstack(all_points)
        combined_colors = np.vstack(all_colors)
        
        print(f"Total dense points: {len(combined_points)}")
        
        # Save dense point cloud
        output_file = os.path.join(output_dir, "dense_points_3d.txt")
        with open(output_file, 'w') as f:
            for pt, color in zip(combined_points, combined_colors):
                f.write(f"{pt[0]} {pt[1]} {pt[2]} {int(color[0])} {int(color[1])} {int(color[2])}\n")
        
        print(f"Dense point cloud saved to: {output_file}")
        print(f"Points: {len(combined_points)}")
        print("=" * 60)
        print("MVS Dense Reconstruction Complete!")
        print("=" * 60)
        
        return combined_points, combined_colors
    else:
        print("Error: No points generated")
        return None, None


def main():
    parser = argparse.ArgumentParser(description='MVS Dense 3D Reconstruction')
    parser.add_argument(
        '--input',
        type=str,
        default='output_bundle_adjusted',
        help='Input directory with bundle-adjusted reconstruction'
    )
    parser.add_argument(
        '--images',
        type=str,
        default='dataset/south-building/images',
        help='Directory with input images'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output_mvs_dense',
        help='Output directory for dense reconstruction'
    )
    parser.add_argument(
        '--num-pairs',
        type=int,
        default=3,
        help='Number of image pairs to process (default: 3)'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input directory not found: {args.input}")
        return
    
    if not os.path.exists(args.images):
        print(f"Error: Images directory not found: {args.images}")
        return
    
    mvs_dense_reconstruction(args.input, args.images, args.output, args.num_pairs)


if __name__ == "__main__":
    main()
