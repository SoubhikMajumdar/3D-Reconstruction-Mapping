"""
Visualize Dense MVS Point Cloud
Handles large point clouds with optional subsampling
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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


def subsample_points(points, colors, max_points=50000):
    """Subsample point cloud for faster visualization"""
    if len(points) <= max_points:
        return points, colors
    
    print(f"Subsampling {len(points):,} points to {max_points:,} for visualization...")
    indices = np.random.choice(len(points), max_points, replace=False)
    return points[indices], colors[indices]


def visualize_dense_point_cloud(points_3d, colors, title="Dense MVS Point Cloud", output_file=None, subsample=True):
    """Visualize dense 3D point cloud"""
    # Subsample if too many points
    if subsample and len(points_3d) > 50000:
        points_3d, colors = subsample_points(points_3d, colors, max_points=50000)
    
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Normalize colors to [0, 1]
    colors_norm = colors / 255.0 if colors.max() > 1.0 else colors
    
    # Plot points with smaller size for dense clouds
    point_size = 0.5 if len(points_3d) > 10000 else 1.0
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
              c=colors_norm, s=point_size, alpha=0.6)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'{title}\n({len(points_3d):,} points displayed)')
    
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


def compare_sparse_dense(sparse_file, dense_file, output_file=None):
    """Compare sparse SfM vs dense MVS point clouds"""
    # Load sparse
    print("Loading sparse point cloud...")
    sparse_points, sparse_colors = load_point_cloud(sparse_file)
    
    # Load dense
    print("Loading dense point cloud...")
    dense_points, dense_colors = load_point_cloud(dense_file)
    
    # Subsample dense for comparison
    dense_points_display, dense_colors_display = subsample_points(dense_points, dense_colors, max_points=50000)
    
    fig = plt.figure(figsize=(20, 10))
    
    # Sparse reconstruction
    ax1 = fig.add_subplot(121, projection='3d')
    sparse_colors_norm = sparse_colors / 255.0 if sparse_colors.max() > 1.0 else sparse_colors
    ax1.scatter(sparse_points[:, 0], sparse_points[:, 1], sparse_points[:, 2],
               c=sparse_colors_norm, s=2, alpha=0.8)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'Sparse SfM Reconstruction\n({len(sparse_points):,} points)')
    
    # Dense reconstruction
    ax2 = fig.add_subplot(122, projection='3d')
    dense_colors_norm = dense_colors_display / 255.0 if dense_colors_display.max() > 1.0 else dense_colors_display
    ax2.scatter(dense_points_display[:, 0], dense_points_display[:, 1], dense_points_display[:, 2],
               c=dense_colors_norm, s=0.5, alpha=0.6)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title(f'Dense MVS Reconstruction\n({len(dense_points):,} points, {len(dense_points_display):,} displayed)')
    
    # Set equal aspect for both (use dense range)
    for ax in [ax1, ax2]:
        max_range = np.array([
            dense_points[:, 0].max() - dense_points[:, 0].min(),
            dense_points[:, 1].max() - dense_points[:, 1].min(),
            dense_points[:, 2].max() - dense_points[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (dense_points[:, 0].max() + dense_points[:, 0].min()) * 0.5
        mid_y = (dense_points[:, 1].max() + dense_points[:, 1].min()) * 0.5
        mid_z = (dense_points[:, 2].max() + dense_points[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Comparison visualization saved to: {output_file}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize Dense MVS Point Cloud')
    parser.add_argument(
        '--dense-file',
        type=str,
        default='output_mvs_dense/dense_points_3d.txt',
        help='Dense point cloud file'
    )
    parser.add_argument(
        '--sparse-file',
        type=str,
        default=None,
        help='Sparse point cloud file for comparison (optional)'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare sparse vs dense reconstruction'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file for visualization'
    )
    parser.add_argument(
        '--no-subsample',
        action='store_true',
        help='Disable subsampling (may be very slow)'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dense_file):
        print(f"Error: Dense point cloud file not found: {args.dense_file}")
        return
    
    if args.compare and args.sparse_file:
        if not os.path.exists(args.sparse_file):
            print(f"Error: Sparse point cloud file not found: {args.sparse_file}")
            return
        compare_sparse_dense(args.sparse_file, args.dense_file, args.output)
    else:
        # Single visualization
        print("Loading dense point cloud...")
        points, colors = load_point_cloud(args.dense_file)
        print(f"Loaded {len(points):,} points")
        
        output_file = args.output or "dense_mvs_visualization.png"
        visualize_dense_point_cloud(
            points, colors, 
            title="Dense MVS Point Cloud",
            output_file=output_file,
            subsample=not args.no_subsample
        )


if __name__ == "__main__":
    main()
