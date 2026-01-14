"""
Visualize Bundle-Adjusted 3D Reconstruction
Shows the refined point cloud and optionally compares with original
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import argparse
import os


def load_point_cloud(filename):
    """Load point cloud from file"""
    points = []
    colors = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                colors.append([int(parts[3]), int(parts[4]), int(parts[5])])
    return np.array(points), np.array(colors)


def load_camera_poses(cameras_file):
    """Load camera poses from JSON file"""
    with open(cameras_file, 'r') as f:
        data = json.load(f)
    return data['camera_poses'], data.get('image_names', [])


def visualize_point_cloud(points_3d, colors, camera_poses=None, title="3D Point Cloud", output_file=None):
    """Visualize 3D point cloud with optional camera positions"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Normalize colors to [0, 1]
    colors_norm = colors / 255.0 if colors.max() > 1.0 else colors
    
    # Plot points
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
              c=colors_norm, s=1, alpha=0.6)
    
    # Plot camera positions if provided
    if camera_poses is not None:
        camera_positions = []
        for cam_idx, pose_data in camera_poses.items():
            if isinstance(pose_data, dict):
                t = np.array(pose_data['t'])
                if len(t.shape) > 1:
                    t = t.reshape(3)
                camera_positions.append(t)
        
        if camera_positions:
            camera_positions = np.array(camera_positions)
            ax.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2],
                      c='red', s=100, marker='^', label='Cameras', alpha=0.8)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    if camera_poses is not None:
        ax.legend()
    
    # Set equal aspect
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
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_file}")
    
    plt.show()


def compare_reconstructions(original_dir, refined_dir, output_file=None):
    """Compare original and refined reconstructions"""
    # Load original
    orig_points, orig_colors = load_point_cloud(os.path.join(original_dir, "points_3d.txt"))
    orig_cameras_file = os.path.join(original_dir, "cameras.json")
    orig_camera_poses, _ = load_camera_poses(orig_cameras_file) if os.path.exists(orig_cameras_file) else (None, None)
    
    # Load refined
    refined_points, refined_colors = load_point_cloud(os.path.join(refined_dir, "points_3d_refined.txt"))
    refined_cameras_file = os.path.join(refined_dir, "cameras_refined.json")
    refined_camera_poses, _ = load_camera_poses(refined_cameras_file) if os.path.exists(refined_cameras_file) else (None, None)
    
    # Load optimization stats
    opt_stats = None
    if os.path.exists(refined_cameras_file):
        with open(refined_cameras_file, 'r') as f:
            data = json.load(f)
            opt_stats = data.get('optimization', None)
    
    fig = plt.figure(figsize=(16, 8))
    
    # Original reconstruction
    ax1 = fig.add_subplot(121, projection='3d')
    colors_norm = orig_colors / 255.0 if orig_colors.max() > 1.0 else orig_colors
    ax1.scatter(orig_points[:, 0], orig_points[:, 1], orig_points[:, 2],
               c=colors_norm, s=1, alpha=0.6)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'Original Reconstruction\n({len(orig_points)} points)')
    
    # Refined reconstruction
    ax2 = fig.add_subplot(122, projection='3d')
    colors_norm = refined_colors / 255.0 if refined_colors.max() > 1.0 else refined_colors
    ax2.scatter(refined_points[:, 0], refined_points[:, 1], refined_points[:, 2],
               c=colors_norm, s=1, alpha=0.6)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    title = f'Bundle-Adjusted Reconstruction\n({len(refined_points)} points)'
    if opt_stats:
        improvement = opt_stats.get('improvement', 0)
        title += f'\nCost reduction: {improvement:.0f}'
    ax2.set_title(title)
    
    # Set equal aspect for both
    for ax in [ax1, ax2]:
        max_range = np.array([
            refined_points[:, 0].max() - refined_points[:, 0].min(),
            refined_points[:, 1].max() - refined_points[:, 1].min(),
            refined_points[:, 2].max() - refined_points[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (refined_points[:, 0].max() + refined_points[:, 0].min()) * 0.5
        mid_y = (refined_points[:, 1].max() + refined_points[:, 1].min()) * 0.5
        mid_z = (refined_points[:, 2].max() + refined_points[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Comparison visualization saved to: {output_file}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize Bundle-Adjusted 3D Reconstruction')
    parser.add_argument(
        '--refined-dir',
        type=str,
        default='output_bundle_adjusted',
        help='Directory with refined reconstruction results'
    )
    parser.add_argument(
        '--original-dir',
        type=str,
        default=None,
        help='Directory with original reconstruction for comparison (optional)'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare original vs refined reconstructions'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file for visualization (optional)'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.refined_dir):
        print(f"Error: Refined directory not found: {args.refined_dir}")
        return
    
    refined_points_file = os.path.join(args.refined_dir, "points_3d_refined.txt")
    if not os.path.exists(refined_points_file):
        print(f"Error: Point cloud file not found: {refined_points_file}")
        return
    
    if args.compare and args.original_dir:
        # Compare mode
        compare_reconstructions(args.original_dir, args.refined_dir, args.output)
    else:
        # Single visualization
        points, colors = load_point_cloud(refined_points_file)
        
        cameras_file = os.path.join(args.refined_dir, "cameras_refined.json")
        camera_poses = None
        if os.path.exists(cameras_file):
            camera_poses, _ = load_camera_poses(cameras_file)
        
        title = f"Bundle-Adjusted 3D Reconstruction ({len(points)} points)"
        
        # Load optimization stats if available
        if os.path.exists(cameras_file):
            with open(cameras_file, 'r') as f:
                data = json.load(f)
                opt_stats = data.get('optimization', None)
                if opt_stats:
                    final_cost = opt_stats.get('final_cost', 0)
                    title += f'\nFinal cost: {final_cost:.0f}'
        
        output_file = args.output or "bundle_adjusted_visualization.png"
        visualize_point_cloud(points, colors, camera_poses, title, output_file)


if __name__ == "__main__":
    main()
