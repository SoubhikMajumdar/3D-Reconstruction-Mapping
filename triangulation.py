"""
Triangulation: Recover 3D Points from Two Views
Given camera poses from two-view geometry, triangulate 3D points from matched features
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os
import json

# Import functions from two_view_geometry
from two_view_geometry import (
    detect_and_describe_features,
    match_features,
    estimate_fundamental_matrix,
    estimate_essential_matrix,
    estimate_camera_intrinsics,
    recover_pose
)


def triangulate_points(P1, P2, pts1, pts2):
    """
    Triangulate 3D points from two camera views
    
    Args:
        P1, P2: Camera projection matrices (3x4)
        pts1, pts2: Corresponding 2D points (Nx2 arrays)
    
    Returns:
        points_3d: 3D points (Nx3)
    """
    # Ensure points are float32 and properly shaped
    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)
    
    # Triangulate points (cv2.triangulatePoints expects points as 2xN arrays)
    pts_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    
    # Convert from homogeneous to 3D coordinates
    pts_3d = (pts_4d[:3] / pts_4d[3]).T
    
    return pts_3d


def filter_points(pts_3d, pts1, pts2, P1, P2, min_depth=0.1, max_depth=1000.0):
    """
    Filter 3D points based on various criteria
    
    Args:
        pts_3d: 3D points (Nx3)
        pts1, pts2: 2D points (Nx2)
        P1, P2: Camera projection matrices
        min_depth, max_depth: Depth range limits
    
    Returns:
        valid_mask: Boolean mask for valid points
    """
    valid = np.ones(len(pts_3d), dtype=bool)
    
    # Filter 1: Points must be in front of both cameras (positive Z)
    # For camera 1 (at origin), points should have z > 0
    valid = valid & (pts_3d[:, 2] > min_depth)
    
    # Filter 2: Points should be within reasonable depth range
    valid = valid & (pts_3d[:, 2] < max_depth)
    
    # Filter 3: Check reprojection error (optional, more advanced)
    # For now, we'll skip this for simplicity
    
    return valid


def get_colors_from_images(img1, img2, pts1, pts2, valid_mask):
    """
    Extract colors for 3D points from images
    
    Args:
        img1, img2: Input images
        pts1, pts2: 2D points in images
        valid_mask: Mask for valid points
    
    Returns:
        colors: RGB colors for valid 3D points (Nx3)
    """
    colors = []
    pts1_valid = pts1[valid_mask]
    pts2_valid = pts2[valid_mask]
    
    for i, (pt1, pt2) in enumerate(zip(pts1_valid, pts2_valid)):
        x1, y1 = int(pt1[0]), int(pt1[1])
        x2, y2 = int(pt2[0]), int(pt2[1])
        
        # Get color from first image (or average of both)
        if 0 <= y1 < img1.shape[0] and 0 <= x1 < img1.shape[1]:
            color = img1[y1, x1]
            # Convert BGR to RGB
            colors.append([int(color[2]), int(color[1]), int(color[0])])
        elif 0 <= y2 < img2.shape[0] and 0 <= x2 < img2.shape[1]:
            color = img2[y2, x2]
            colors.append([int(color[2]), int(color[1]), int(color[0])])
        else:
            colors.append([128, 128, 128])  # Gray for invalid pixels
    
    return np.array(colors)


def visualize_point_cloud(points_3d, colors, title="3D Point Cloud"):
    """
    Visualize 3D point cloud using matplotlib
    
    Args:
        points_3d: 3D points (Nx3)
        colors: RGB colors (Nx3), normalized to [0, 1]
        title: Plot title
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Normalize colors to [0, 1] if needed
    if colors.max() > 1.0:
        colors_normalized = colors / 255.0
    else:
        colors_normalized = colors
    
    # Plot points
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
              c=colors_normalized, s=1, alpha=0.6)
    
    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'{title} ({len(points_3d)} points)')
    
    # Set equal aspect ratio
    max_range = np.array([
        points_3d[:, 0].max() - points_3d[:, 0].min(),
        points_3d[:, 1].max() - points_3d[:, 1].min(),
        points_3d[:, 2].max() - points_3d[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (points_3d[:, 0].max() + points_3d[:, 0].min()) * 0.5
    mid_y = (points_3d[:, 1].max() + points_3d[:, 1].min()) * 0.5
    mid_z = (points_3d[:, 2].max() + points_3d[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    return fig, ax


def save_point_cloud(points_3d, colors, filename="points_3d.txt"):
    """
    Save point cloud to text file (x y z r g b format)
    
    Args:
        points_3d: 3D points (Nx3)
        colors: RGB colors (Nx3)
        filename: Output filename
    """
    with open(filename, 'w') as f:
        for pt, color in zip(points_3d, colors):
            f.write(f"{pt[0]} {pt[1]} {pt[2]} {int(color[0])} {int(color[1])} {int(color[2])}\n")
    print(f"Point cloud saved to: {filename}")


def triangulate_two_views(img1_path, img2_path, detector_type='ORB', visualize=True, save=True):
    """
    Complete pipeline: Two-view geometry + Triangulation
    
    Args:
        img1_path, img2_path: Paths to input images
        detector_type: 'ORB' or 'SIFT'
        visualize: Whether to show 3D visualization
        save: Whether to save point cloud
    
    Returns:
        dict: Results containing 3D points, colors, camera poses, etc.
    """
    print("=" * 60)
    print("Triangulation: 3D Point Cloud Reconstruction")
    print("=" * 60)
    
    # Load images
    print(f"\nLoading images...")
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        raise ValueError(f"Could not load images: {img1_path}, {img2_path}")
    
    print(f"Image 1: {img1.shape}")
    print(f"Image 2: {img2.shape}")
    
    # Step 1: Detect and describe features
    print(f"\nStep 1: Detecting features using {detector_type}...")
    kp1, desc1 = detect_and_describe_features(img1, detector_type)
    kp2, desc2 = detect_and_describe_features(img2, detector_type)
    print(f"  Image 1: {len(kp1)} keypoints")
    print(f"  Image 2: {len(kp2)} keypoints")
    
    # Step 2: Match features
    print(f"\nStep 2: Matching features...")
    good_matches, _ = match_features(desc1, desc2, detector_type)
    print(f"  Found {len(good_matches)} good matches")
    
    if len(good_matches) < 8:
        raise ValueError(f"Not enough matches ({len(good_matches)}) for reconstruction")
    
    # Extract matched points
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
    
    # Step 3: Estimate camera intrinsics
    K = estimate_camera_intrinsics(img1.shape)
    
    # Step 4: Estimate Essential matrix
    print(f"\nStep 3: Estimating Essential matrix...")
    E, E_mask = estimate_essential_matrix(pts1, pts2, K, method='RANSAC')
    inliers_E = np.sum(E_mask)
    print(f"  Essential matrix inliers: {inliers_E}/{len(good_matches)}")
    
    # Step 5: Recover pose
    print(f"\nStep 4: Recovering camera pose...")
    pts1_inlier = pts1[E_mask.ravel() == 1]
    pts2_inlier = pts2[E_mask.ravel() == 1]
    R, t, pose_mask = recover_pose(E, pts1_inlier, pts2_inlier, K)
    
    print(f"  Rotation matrix R:")
    print(f"  {R}")
    print(f"  Translation vector t: {t.ravel()}")
    print(f"  Pose mask shape: {pose_mask.shape}, sum: {np.sum(pose_mask)}")
    
    # Filter points that satisfy pose constraints
    # recoverPose returns mask with same length as input points
    if len(pose_mask.shape) > 1:
        pose_mask = pose_mask.ravel()
    
    pts1_final = pts1_inlier[pose_mask == 1]
    pts2_final = pts2_inlier[pose_mask == 1]
    print(f"  Points satisfying pose constraints: {len(pts1_final)}")
    
    # If no points pass, use all inliers (might be a mask format issue)
    if len(pts1_final) == 0:
        print(f"  Warning: No points passed pose filter, using all E inliers")
        pts1_final = pts1_inlier
        pts2_final = pts2_inlier
    
    # Step 6: Build camera projection matrices
    print(f"\nStep 5: Building camera projection matrices...")
    # Camera 1: Identity (at origin)
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    # Camera 2: Rotated and translated
    P2 = K @ np.hstack([R, t])
    
    print(f"  Camera 1 projection matrix P1:")
    print(f"  {P1}")
    print(f"  Camera 2 projection matrix P2:")
    print(f"  {P2}")
    
    # Step 7: Triangulate 3D points
    print(f"\nStep 6: Triangulating 3D points...")
    points_3d = triangulate_points(P1, P2, pts1_final, pts2_final)
    print(f"  Triangulated {len(points_3d)} 3D points")
    
    # Step 8: Filter points
    print(f"\nStep 7: Filtering 3D points...")
    valid_mask = filter_points(points_3d, pts1_final, pts2_final, P1, P2)
    points_3d_filtered = points_3d[valid_mask]
    pts1_filtered = pts1_final[valid_mask]
    pts2_filtered = pts2_final[valid_mask]
    
    print(f"  Valid points after filtering: {len(points_3d_filtered)}/{len(points_3d)}")
    print(f"  Point depth range: [{points_3d_filtered[:, 2].min():.2f}, {points_3d_filtered[:, 2].max():.2f}]")
    
    # Step 9: Get colors
    print(f"\nStep 8: Extracting colors...")
    colors = get_colors_from_images(img1, img2, pts1_filtered, pts2_filtered, valid_mask)
    print(f"  Colors extracted for {len(colors)} points")
    
    # Step 10: Visualize
    if visualize:
        print(f"\nStep 9: Visualizing 3D point cloud...")
        fig, ax = visualize_point_cloud(points_3d_filtered, colors)
        plt.savefig('triangulation_result.png', dpi=150, bbox_inches='tight')
        print(f"  Visualization saved to: triangulation_result.png")
        plt.show()
    
    # Step 11: Save point cloud
    if save:
        print(f"\nStep 10: Saving point cloud...")
        save_point_cloud(points_3d_filtered, colors, "triangulated_points.txt")
    
    # Return results
    results = {
        'points_3d': points_3d_filtered,
        'colors': colors,
        'R': R,
        't': t,
        'K': K,
        'P1': P1,
        'P2': P2,
        'num_points': len(points_3d_filtered),
        'image1_path': img1_path,
        'image2_path': img2_path
    }
    
    print("\n" + "=" * 60)
    print("Triangulation Complete!")
    print(f"Reconstructed {len(points_3d_filtered)} 3D points")
    print("=" * 60)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Triangulation: Recover 3D points from two camera views'
    )
    parser.add_argument(
        '--img1',
        type=str,
        required=False,
        help='Path to first image'
    )
    parser.add_argument(
        '--img2',
        type=str,
        required=False,
        help='Path to second image'
    )
    parser.add_argument(
        '--detector',
        type=str,
        default='ORB',
        choices=['ORB', 'SIFT'],
        help='Feature detector type (default: ORB)'
    )
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Disable visualization'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Disable saving point cloud'
    )
    
    args = parser.parse_args()
    
    # Default images if not provided
    if not args.img1 or not args.img2:
        img1_path = "dataset/south-building/images/P1180141.JPG"
        img2_path = "dataset/south-building/images/P1180142.JPG"
        
        if os.path.exists(img1_path) and os.path.exists(img2_path):
            print("Using default images from dataset.")
        else:
            print("Error: Please provide --img1 and --img2 arguments")
            return
    else:
        img1_path = args.img1
        img2_path = args.img2
    
    # Check if images exist
    if not os.path.exists(img1_path):
        print(f"Error: Image 1 not found: {img1_path}")
        return
    
    if not os.path.exists(img2_path):
        print(f"Error: Image 2 not found: {img2_path}")
        return
    
    # Run triangulation
    try:
        results = triangulate_two_views(
            img1_path,
            img2_path,
            detector_type=args.detector,
            visualize=not args.no_viz,
            save=not args.no_save
        )
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
