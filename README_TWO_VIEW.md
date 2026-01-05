# Two-View Geometry: Relative Camera Motion Estimation

A Python implementation for estimating relative camera motion between two images using feature matching and epipolar geometry.

## Overview

This script implements the classic two-view geometry pipeline:
1. **Feature Detection**: Detects and describes features using ORB or SIFT
2. **Feature Matching**: Matches features between images using BFMatcher with ratio test
3. **Fundamental Matrix**: Estimates the fundamental matrix using RANSAC
4. **Essential Matrix**: Computes the essential matrix from camera intrinsics
5. **Pose Recovery**: Recovers relative camera rotation (R) and translation (t)
6. **Visualization**: Shows feature matches and epipolar lines

## Theory

### Two-View Geometry Pipeline

1. **Feature Detection & Matching**
   - Detects keypoints in both images using ORB or SIFT
   - Matches features using brute-force matcher with Lowe's ratio test
   - Filters out ambiguous matches

2. **Fundamental Matrix (F)**
   - Relates corresponding points in two images: `x2^T * F * x1 = 0`
   - Estimated using RANSAC for robustness to outliers
   - 7-point or 8-point algorithm

3. **Essential Matrix (E)**
   - Relates camera poses: `E = [t]_× * R`
   - Computed from fundamental matrix: `E = K2^T * F * K1`
   - Where K1, K2 are camera intrinsic matrices

4. **Pose Recovery**
   - Decomposes essential matrix to get rotation R and translation t
   - Uses `cv2.recoverPose()` which handles the 4 possible solutions
   - Selects the solution with most points in front of cameras

5. **Epipolar Lines**
   - Epipolar line in image 2: `l2 = F * x1`
   - Epipolar line in image 1: `l1 = F^T * x2`
   - Used to validate the fundamental matrix estimation

## Requirements

```bash
pip install opencv-contrib-python numpy matplotlib
```

Or use the existing requirements:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
# Use default images (first two from dataset)
python two_view_geometry.py

# Specify custom images
python two_view_geometry.py --img1 image1.jpg --img2 image2.jpg

# Use SIFT detector instead of ORB
python two_view_geometry.py --img1 img1.jpg --img2 img2.jpg --detector SIFT

# Disable visualization (faster, just prints results)
python two_view_geometry.py --img1 img1.jpg --img2 img2.jpg --no-viz
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--img1` | Path to first image | Required (or uses dataset default) |
| `--img2` | Path to second image | Required (or uses dataset default) |
| `--detector` | Feature detector: `ORB` or `SIFT` | `ORB` |
| `--no-viz` | Disable visualization | False (shows plots) |

### Example

```bash
# Using images from south-building dataset
python two_view_geometry.py \
    --img1 dataset/south-building/images/P1180141.JPG \
    --img2 dataset/south-building/images/P1180142.JPG \
    --detector ORB
```

## Output

### Console Output

The script prints:
- Number of keypoints detected in each image
- Number of matches after ratio test
- Estimated camera intrinsic matrix (K)
- Fundamental matrix (F)
- Essential matrix (E)
- Rotation matrix (R)
- Translation vector (t)
- Number of inliers for each estimation

### Visualization

If visualization is enabled (`--no-viz` not used), the script creates:

1. **Feature Matches Plot**: Shows matched keypoints between images
2. **Epipolar Lines (Image 1)**: Shows epipolar lines and corresponding points
3. **Epipolar Lines (Image 2)**: Shows epipolar lines and corresponding points
4. **Statistics Panel**: Summary of results

The visualization is saved as `two_view_geometry_result.png`.

### Example Output

```
============================================================
Two-View Geometry: Relative Camera Motion Estimation
============================================================

Loading images...
Image 1: (2304, 3072, 3)
Image 2: (2304, 3072, 3)

Step 1: Detecting features using ORB...
  Image 1: 5000 keypoints
  Image 2: 5000 keypoints

Step 2: Matching features...
  Found 349 good matches (after ratio test)

Step 3: Estimated camera intrinsics:
  K = [[3072. 0. 1536.]
       [0. 3072. 1152.]
       [0. 0. 1.]]

Step 4: Estimating Fundamental matrix (RANSAC)...
  Inliers: 206/349

Step 5: Estimating Essential matrix (RANSAC)...
  Inliers: 263/349

Step 6: Recovering camera pose...
  Rotation matrix (R): [[0.999 0.029 0.019]
                        [-0.029 1.000 -0.004]
                        [-0.019 0.003 1.000]]
  Translation vector (t): [-0.898 -0.126 -0.420]
```

## Algorithm Details

### Feature Detection

- **ORB**: Fast, binary descriptor, good for real-time applications
- **SIFT**: More accurate, slower, better for high-quality reconstruction

### Feature Matching

Uses **Lowe's ratio test**:
- For each feature in image 1, find 2 best matches in image 2
- Accept match if: `distance(best_match) < ratio_threshold * distance(second_best)`
- Default ratio: 0.75 (lower = stricter)

### Camera Intrinsics Estimation

The script estimates camera intrinsics from image dimensions:
- Focal length: `f ≈ image_width`
- Principal point: `(cx, cy) = (width/2, height/2)`

**Note**: For accurate results, use known camera calibration. The estimated intrinsics work for demonstration but may not be precise.

### RANSAC Parameters

- **Fundamental Matrix**: `ransacReprojThreshold=1.0`, `confidence=0.99`
- **Essential Matrix**: `threshold=1.0`, `prob=0.999`

## Understanding the Results

### Rotation Matrix (R)
- 3×3 orthogonal matrix describing rotation between camera poses
- Relative rotation from camera 1 to camera 2
- Should be close to identity for small camera movements

### Translation Vector (t)
- 3×1 vector (up to scale) describing translation direction
- Scale is ambiguous from images alone (scale can't be determined)
- Normalized: `||t|| ≈ 1`

### Epipolar Lines Validation
- Epipolar lines should pass through corresponding points
- Visual inspection helps verify correct fundamental matrix estimation
- Parallel lines indicate pure rotation or degenerate configuration

## Limitations

1. **Scale Ambiguity**: Translation is known only up to scale
2. **Camera Intrinsics**: Estimated from image size (not calibrated)
3. **Two Views Only**: Full SfM requires multiple views and bundle adjustment
4. **No Bundle Adjustment**: This is a minimal two-view solution

## Extending to Full SfM

To extend this to full Structure-from-Motion:
1. Use multiple images (not just 2)
2. Implement incremental reconstruction (add images one by one)
3. Perform bundle adjustment to refine all camera poses and 3D points
4. Consider using COLMAP for production-quality results

## Troubleshooting

### Not Enough Matches
- Images may not have enough overlap
- Try different detector (SIFT if using ORB, or vice versa)
- Adjust ratio threshold (lower = stricter, higher = more matches)

### Poor Epipolar Lines
- Fundamental matrix estimation may have failed
- Check if images have sufficient overlap
- Verify images are from the same scene

### Singular/Invalid Matrices
- May indicate degenerate configuration (pure rotation, planar scene)
- Ensure images have parallax (camera translation)

## References

- Hartley & Zisserman, "Multiple View Geometry in Computer Vision"
- OpenCV Documentation: [Camera Calibration and 3D Reconstruction](https://docs.opencv.org/master/d9/d0c/group__calib3d.html)
- Lowe, "Distinctive Image Features from Scale-Invariant Keypoints"

## See Also

- `sfm_mvs.py`: Full SfM-MVS pipeline (simplified)
- `run_colmap.py`: COLMAP integration script
- `visualize_simple.py`: Point cloud visualization

