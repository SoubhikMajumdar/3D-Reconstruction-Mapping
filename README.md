# Structure-from-Motion and Multi-View Stereo 3D Reconstruction Pipeline

A complete Python implementation of Structure-from-Motion (SfM) and Multi-View Stereo (MVS) for 3D reconstruction from multiple images. This pipeline implements incremental SfM with bundle adjustment and dense reconstruction using stereo matching.

## Overview

This project implements a complete 3D reconstruction pipeline that processes multiple images of a scene to create dense 3D point clouds. The pipeline consists of several stages:

1. **Two-View Geometry**: Feature detection, matching, and relative pose estimation between image pairs
2. **Triangulation**: 3D point reconstruction from two camera views
3. **Multi-View Reconstruction**: Incremental SfM that adds images sequentially using PnP
4. **Bundle Adjustment**: Joint optimization of camera poses and 3D points to minimize reprojection error
5. **Dense MVS**: Multi-view stereo reconstruction using depth maps from stereo matching

## Features

- Feature detection and matching (ORB/SIFT)
- Essential matrix estimation with RANSAC
- Camera pose recovery and triangulation
- Incremental multi-view reconstruction with PnP
- Bundle adjustment using scipy.optimize
- Dense reconstruction using Semi-Global Block Matching (SGBM)
- Visualization tools for sparse and dense point clouds
- Point cloud export and comparison utilities

## Dataset

This project uses the South Building dataset. The dataset should be placed in the `dataset/south-building/images/` directory. The dataset contains 128 images of a building facade captured from different viewpoints.

Dataset structure:
```
dataset/
  south-building/
    images/
      P1180141.JPG
      P1180142.JPG
      ...
      P1180347.JPG
```

The images are sequential captures of the same building scene from different camera positions, suitable for Structure-from-Motion and Multi-View Stereo reconstruction.

Note: You can use any dataset with multiple overlapping images. Place your images in a directory and specify the path using the `--dataset` or `--images` arguments.

## Requirements

- Python 3.8+
- OpenCV (with contrib modules)
- NumPy
- SciPy
- Matplotlib

See `requirements.txt` for specific versions.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/SoubhikMajumdar/3D-Reconstruction-Mapping.git
cd 3D-Reconstruction-Mapping
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── two_view_geometry.py          # Two-view geometry and pose estimation
├── triangulation.py              # 3D point triangulation from two views
├── multiview_reconstruction.py   # Incremental multi-view SfM reconstruction
├── bundle_adjustment.py          # Bundle adjustment optimization
├── mvs_dense_reconstruction.py   # Dense MVS reconstruction
├── visualize_bundle_adjusted.py  # Visualization for bundle-adjusted results
├── visualize_dense.py            # Visualization for dense point clouds
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── README_TWO_VIEW.md           # Detailed documentation for two-view geometry
```

## Usage

### Two-View Geometry

Estimate relative camera motion between two images:

```bash
python two_view_geometry.py --img1 dataset/south-building/images/P1180141.JPG --img2 dataset/south-building/images/P1180142.JPG
```

Options:
- `--img1`, `--img2`: Paths to input images
- `--detector`: Feature detector type ('ORB' or 'SIFT', default: 'ORB')
- `--no-viz`: Disable visualization

### Triangulation

Reconstruct 3D points from two camera views:

```bash
python triangulation.py --img1 dataset/south-building/images/P1180141.JPG --img2 dataset/south-building/images/P1180142.JPG
```

Options:
- `--img1`, `--img2`: Paths to input images
- `--detector`: Feature detector type ('ORB' or 'SIFT', default: 'ORB')
- `--no-viz`: Disable visualization
- `--no-save`: Disable saving point cloud

### Multi-View Reconstruction

Incremental SfM reconstruction with multiple images:

```bash
python multiview_reconstruction.py --dataset dataset/south-building/images --num-images 10
```

Options:
- `--dataset`: Path to images directory
- `--num-images`: Number of images to use (default: 5)
- `--detector`: Feature detector type ('ORB' or 'SIFT', default: 'ORB')
- `--no-viz`: Disable visualization

Output:
- `output_multiview/points_3d.txt`: Reconstructed 3D points
- `output_multiview/cameras.json`: Camera poses and intrinsics
- `output_multiview/correspondences.json`: 2D-3D correspondences

### Bundle Adjustment

Refine camera poses and 3D points using bundle adjustment:

```bash
python bundle_adjustment.py --input output_multiview --output output_bundle_adjusted
```

Options:
- `--input`: Input directory with reconstruction results
- `--output`: Output directory for refined reconstruction

Output:
- `output_bundle_adjusted/points_3d_refined.txt`: Refined 3D points
- `output_bundle_adjusted/cameras_refined.json`: Refined camera poses with optimization statistics

### Dense MVS Reconstruction

Create dense point clouds using multi-view stereo:

```bash
python mvs_dense_reconstruction.py --input output_bundle_adjusted --images dataset/south-building/images --num-pairs 3
```

Options:
- `--input`: Directory with bundle-adjusted reconstruction
- `--images`: Directory with input images
- `--output`: Output directory (default: output_mvs_dense)
- `--num-pairs`: Number of image pairs to process (default: 3)

Output:
- `output_mvs_dense/dense_points_3d.txt`: Dense point cloud

### Visualization

Visualize bundle-adjusted reconstruction:

```bash
python visualize_bundle_adjusted.py --refined-dir output_bundle_adjusted --compare --original-dir output_multiview
```

Visualize dense MVS point cloud:

```bash
python visualize_dense.py --dense-file output_mvs_dense/dense_points_3d.txt
```

Compare sparse vs dense reconstruction:

```bash
python visualize_dense.py --dense-file output_mvs_dense/dense_points_3d.txt --sparse-file output_bundle_adjusted/points_3d_refined.txt --compare
```

## Complete Pipeline Example

Run the full pipeline from two-view to dense reconstruction:

```bash
# 1. Multi-view reconstruction (10 images)
python multiview_reconstruction.py --dataset dataset/south-building/images --num-images 10 --no-viz

# 2. Bundle adjustment
python bundle_adjustment.py --input output_multiview --output output_bundle_adjusted

# 3. Dense MVS reconstruction
python mvs_dense_reconstruction.py --input output_bundle_adjusted --images dataset/south-building/images --num-pairs 3

# 4. Visualize results
python visualize_bundle_adjusted.py --refined-dir output_bundle_adjusted --output bundle_adjusted_result.png
python visualize_dense.py --dense-file output_mvs_dense/dense_points_3d.txt --output dense_mvs_result.png
python visualize_dense.py --dense-file output_mvs_dense/dense_points_3d.txt --sparse-file output_bundle_adjusted/points_3d_refined.txt --compare --output sparse_vs_dense_comparison.png
```

## Output Format

### Point Cloud Format

Point clouds are saved as text files with format:
```
x y z r g b
```

Each line represents a 3D point with:
- `x, y, z`: 3D coordinates
- `r, g, b`: RGB color values (0-255)

### Camera Poses Format

Camera poses are saved as JSON files with structure:
```json
{
  "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "camera_poses": {
    "0": {
      "R": [[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]],
      "t": [tx, ty, tz]
    }
  },
  "image_names": ["image1.jpg", "image2.jpg", ...]
}
```

## Performance Notes

- **Sparse Reconstruction**: Typically produces hundreds to thousands of 3D points from feature matches
- **Dense Reconstruction**: Can produce hundreds of thousands of points using stereo matching
- **Bundle Adjustment**: Optimization time depends on number of points and cameras (typically 30-60 iterations)
- **Visualization**: Dense point clouds are automatically subsampled for faster display (50K points by default)

## Limitations

- Camera intrinsics are estimated from image dimensions (not calibrated)
- Scale ambiguity: translation is known only up to scale
- Limited to scenes with sufficient texture for feature matching
- Dense reconstruction requires good stereo pairs with sufficient overlap
- Bundle adjustment uses simplified optimization (production systems use sparse solvers)

## Theory

### Two-View Geometry

The pipeline estimates relative camera motion using:
- Fundamental matrix (F): Relates corresponding points in two images
- Essential matrix (E): Relates camera poses (E = K2^T * F * K1)
- Pose recovery: Decomposes E to get rotation R and translation t

### Triangulation

3D points are recovered from 2D correspondences using:
- Linear triangulation: Solves the system P * X = x for each point
- Point filtering: Removes points behind cameras or at invalid depths

### Multi-View Reconstruction

Incremental SfM:
- Starts with two-view initialization
- Adds images using PnP (Perspective-n-Point) with existing 3D points
- Triangulates new points from each new image

### Bundle Adjustment

Joint optimization of:
- Camera poses (rotation + translation for each camera)
- 3D points (X, Y, Z for each point)
- Minimizes reprojection error: sum ||project(P_i, X_j) - x_ij||^2

### Dense MVS

Multi-view stereo:
- Stereo matching between image pairs using SGBM
- Depth map computation from disparity
- Projection of depth maps to 3D space using camera poses
- Fusion of multiple depth maps into dense point cloud

## References

- Hartley, R. and Zisserman, A. "Multiple View Geometry in Computer Vision" (2004)
- OpenCV Documentation: Camera Calibration and 3D Reconstruction
- Lowe, D. "Distinctive Image Features from Scale-Invariant Keypoints" (2004)
- Schonberger, J. L. and Frahm, J. M. "Structure-from-Motion Revisited" (2016)

## License

This project is provided as-is for educational and research purposes.

## Author

Soubhik Majumdar
