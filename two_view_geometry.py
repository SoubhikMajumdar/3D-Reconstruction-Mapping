"""
Two-View Geometry: Estimate Relative Camera Motion
Given two images, estimate the relative camera pose (R, t) and visualize matches + epipolar lines
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


def detect_and_describe_features(img, detector_type='ORB'):
    """
    Detect and describe features in an image
    
    Args:
        img: Input image (grayscale or BGR)
        detector_type: 'ORB' or 'SIFT'
    
    Returns:
        keypoints: List of keypoints
        descriptors: Feature descriptors
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    if detector_type == 'ORB':
        detector = cv2.ORB_create(nfeatures=5000)
    elif detector_type == 'SIFT':
        detector = cv2.SIFT_create(nfeatures=5000)
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")
    
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    return keypoints, descriptors


def match_features(desc1, desc2, detector_type='ORB', ratio_threshold=0.75):
    """
    Match features between two images using BFMatcher and ratio test
    
    Args:
        desc1, desc2: Descriptors from two images
        detector_type: 'ORB' or 'SIFT' (determines distance metric)
        ratio_threshold: Ratio test threshold (Lowe's ratio test)
    
    Returns:
        good_matches: List of good matches after ratio test
        all_matches: All matches for visualization
    """
    if detector_type == 'ORB':
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:  # SIFT
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    
    # Use knnMatch for ratio test
    matches = matcher.knnMatch(desc1, desc2, k=2)
    
    # Apply ratio test (Lowe's ratio test)
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)
    
    return good_matches, matches


def estimate_fundamental_matrix(pts1, pts2, method='RANSAC'):
    """
    Estimate Fundamental matrix using RANSAC
    
    Args:
        pts1, pts2: Corresponding points in two images (Nx2 arrays)
        method: 'RANSAC' or '8POINT'
    
    Returns:
        F: Fundamental matrix (3x3)
        mask: Inlier mask
    """
    if method == 'RANSAC':
        F, mask = cv2.findFundamentalMat(
            pts1, pts2,
            method=cv2.FM_RANSAC,
            ransacReprojThreshold=1.0,
            confidence=0.99,
            maxIters=2000
        )
    else:  # 8POINT
        F, mask = cv2.findFundamentalMat(
            pts1, pts2,
            method=cv2.FM_8POINT
        )
    
    return F, mask


def estimate_essential_matrix(pts1, pts2, K, method='RANSAC'):
    """
    Estimate Essential matrix from corresponding points and camera matrix
    
    Args:
        pts1, pts2: Corresponding points in two images (Nx2 arrays)
        K: Camera intrinsic matrix (3x3)
        method: 'RANSAC' or '8POINT'
    
    Returns:
        E: Essential matrix (3x3)
        mask: Inlier mask
    """
    if method == 'RANSAC':
        E, mask = cv2.findEssentialMat(
            pts1, pts2, K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )
    else:  # 8POINT
        E, mask = cv2.findEssentialMat(
            pts1, pts2, K,
            method=cv2.RANSAC  # Essential matrix typically uses RANSAC
        )
    
    return E, mask


def recover_pose(E, pts1, pts2, K):
    """
    Recover relative camera pose from Essential matrix
    
    Args:
        E: Essential matrix (3x3)
        pts1, pts2: Corresponding points (Nx2 arrays)
        K: Camera intrinsic matrix (3x3)
    
    Returns:
        R: Rotation matrix (3x3)
        t: Translation vector (3x1)
        mask: Points that satisfy cheirality constraint
    """
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
    return R, t, mask


def estimate_camera_intrinsics(img_shape):
    """
    Estimate camera intrinsics from image dimensions
    Simple approximation: focal length ≈ image width
    
    Args:
        img_shape: (height, width) or (height, width, channels)
    
    Returns:
        K: Camera intrinsic matrix (3x3)
    """
    h, w = img_shape[:2]
    focal_length = w  # Approximate focal length
    cx, cy = w / 2.0, h / 2.0
    
    K = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    return K


def draw_epipolar_lines(img1, img2, pts1, pts2, F, title1='Image 1', title2='Image 2'):
    """
    Draw epipolar lines on two images
    
    Args:
        img1, img2: Input images
        pts1, pts2: Corresponding points
        F: Fundamental matrix
        title1, title2: Titles for the images
    """
    # Convert to grayscale if needed
    if len(img1.shape) == 3:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        img1_gray = img1.copy()
    
    if len(img2.shape) == 3:
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        img2_gray = img2.copy()
    
    # Convert to color for drawing
    img1_color = cv2.cvtColor(img1_gray, cv2.COLOR_GRAY2BGR)
    img2_color = cv2.cvtColor(img2_gray, cv2.COLOR_GRAY2BGR)
    
    # Find epipolar lines in second image corresponding to points in first image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    
    # Find epipolar lines in first image corresponding to points in second image
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    
    # Draw lines and points on images
    h1, w1 = img1_color.shape[:2]
    h2, w2 = img2_color.shape[:2]
    
    # Draw epipolar lines on image 2
    for line, pt in zip(lines2, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -line[2] / line[1]])
        x1, y1 = map(int, [w2, -(line[2] + line[0] * w2) / line[1]])
        cv2.line(img2_color, (x0, y0), (x1, y1), color, 1)
        cv2.circle(img2_color, tuple(pt.astype(int)), 5, color, -1)
    
    # Draw epipolar lines on image 1
    for line, pt in zip(lines1, pts1):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -line[2] / line[1]])
        x1, y1 = map(int, [w1, -(line[2] + line[0] * w1) / line[1]])
        cv2.line(img1_color, (x0, y0), (x1, y1), color, 1)
        cv2.circle(img1_color, tuple(pt.astype(int)), 5, color, -1)
    
    return img1_color, img2_color


def visualize_matches(img1, kp1, img2, kp2, matches, title='Feature Matches'):
    """
    Visualize feature matches between two images
    
    Args:
        img1, img2: Input images
        kp1, kp2: Keypoints
        matches: List of matches
        title: Plot title
    """
    img_matches = cv2.drawMatches(
        img1, kp1, img2, kp2, matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0)
    )
    
    return img_matches


def two_view_geometry(img1_path, img2_path, detector_type='ORB', visualize=True):
    """
    Complete two-view geometry pipeline
    
    Args:
        img1_path, img2_path: Paths to input images
        detector_type: 'ORB' or 'SIFT'
        visualize: Whether to show visualizations
    
    Returns:
        dict: Results containing R, t, F, E, matches, etc.
    """
    print("=" * 60)
    print("Two-View Geometry: Relative Camera Motion Estimation")
    print("=" * 60)
    
    # Load images
    print(f"\nLoading images...")
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        raise ValueError(f"Could not load images. Check paths: {img1_path}, {img2_path}")
    
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
    good_matches, all_matches = match_features(desc1, desc2, detector_type)
    print(f"  Found {len(good_matches)} good matches (after ratio test)")
    
    if len(good_matches) < 8:
        raise ValueError(f"Not enough matches ({len(good_matches)}) for robust estimation. Need at least 8.")
    
    # Extract matched points
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
    
    # Step 3: Estimate camera intrinsics
    K = estimate_camera_intrinsics(img1.shape)
    print(f"\nStep 3: Estimated camera intrinsics:")
    print(f"  K = \n{K}")
    
    # Step 4: Estimate Fundamental matrix
    print(f"\nStep 4: Estimating Fundamental matrix (RANSAC)...")
    F, F_mask = estimate_fundamental_matrix(pts1, pts2, method='RANSAC')
    inliers_F = np.sum(F_mask)
    print(f"  Fundamental matrix (F):")
    print(f"  {F}")
    print(f"  Inliers: {inliers_F}/{len(good_matches)}")
    
    # Step 5: Estimate Essential matrix
    print(f"\nStep 5: Estimating Essential matrix (RANSAC)...")
    E, E_mask = estimate_essential_matrix(pts1, pts2, K, method='RANSAC')
    inliers_E = np.sum(E_mask)
    print(f"  Essential matrix (E):")
    print(f"  {E}")
    print(f"  Inliers: {inliers_E}/{len(good_matches)}")
    
    # Step 6: Recover pose
    print(f"\nStep 6: Recovering camera pose...")
    pts1_inlier = pts1[E_mask.ravel() == 1]
    pts2_inlier = pts2[E_mask.ravel() == 1]
    R, t, pose_mask = recover_pose(E, pts1_inlier, pts2_inlier, K)
    
    print(f"  Rotation matrix (R):")
    print(f"  {R}")
    print(f"  Translation vector (t):")
    print(f"  {t.ravel()}")
    print(f"  Points in front of cameras: {np.sum(pose_mask)}/{len(pts1_inlier)}")
    
    # Visualize results
    if visualize:
        print(f"\nStep 7: Visualizing results...")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))
        
        # Plot 1: Feature matches
        ax1 = plt.subplot(2, 2, 1)
        img_matches = visualize_matches(img1, kp1, img2, kp2, good_matches[:50])  # Show first 50 matches
        ax1.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
        ax1.set_title(f'Feature Matches ({len(good_matches)} matches)', fontsize=12)
        ax1.axis('off')
        
        # Plot 2: Epipolar lines (image 1)
        ax2 = plt.subplot(2, 2, 2)
        img1_epi, img2_epi = draw_epipolar_lines(img1, img2, pts1_inlier[:50], pts2_inlier[:50], F)
        ax2.imshow(cv2.cvtColor(img1_epi, cv2.COLOR_BGR2RGB))
        ax2.set_title('Epipolar Lines in Image 1', fontsize=12)
        ax2.axis('off')
        
        # Plot 3: Epipolar lines (image 2)
        ax3 = plt.subplot(2, 2, 3)
        ax3.imshow(cv2.cvtColor(img2_epi, cv2.COLOR_BGR2RGB))
        ax3.set_title('Epipolar Lines in Image 2', fontsize=12)
        ax3.axis('off')
        
        # Plot 4: Statistics
        ax4 = plt.subplot(2, 2, 4)
        ax4.axis('off')
        stats_text = f"""
        Two-View Geometry Results
        {'='*40}
        
        Detector: {detector_type}
        Keypoints:
          Image 1: {len(kp1)}
          Image 2: {len(kp2)}
        
        Matches:
          Total: {len(good_matches)}
          F inliers: {inliers_F}
          E inliers: {inliers_E}
        
        Camera Motion:
          Rotation (R): 3x3 matrix
          Translation (t): {t.ravel()}
        
        Translation scale: {np.linalg.norm(t):.4f}
        (Note: Scale is ambiguous from images alone)
        """
        ax4.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                verticalalignment='center', transform=ax4.transAxes)
        
        plt.tight_layout()
        plt.savefig('two_view_geometry_result.png', dpi=150, bbox_inches='tight')
        print(f"  Visualization saved to: two_view_geometry_result.png")
        plt.show()
    
    # Return results
    results = {
        'R': R,
        't': t,
        'F': F,
        'E': E,
        'K': K,
        'matches': good_matches,
        'keypoints1': kp1,
        'keypoints2': kp2,
        'points1': pts1,
        'points2': pts2,
        'F_inliers': inliers_F,
        'E_inliers': inliers_E
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Two-View Geometry: Estimate relative camera motion from two images'
    )
    parser.add_argument(
        '--img1',
        type=str,
        required=True,
        help='Path to first image'
    )
    parser.add_argument(
        '--img2',
        type=str,
        required=True,
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
    
    args = parser.parse_args()
    
    # Check if images exist
    if not os.path.exists(args.img1):
        print(f"Error: Image 1 not found: {args.img1}")
        return
    
    if not os.path.exists(args.img2):
        print(f"Error: Image 2 not found: {args.img2}")
        return
    
    # Run two-view geometry
    try:
        results = two_view_geometry(
            args.img1,
            args.img2,
            detector_type=args.detector,
            visualize=not args.no_viz
        )
        
        print("\n" + "=" * 60)
        print("Reconstruction complete!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # If run without arguments, use first two images from dataset as example
    import sys
    
    if len(sys.argv) == 1:
        # Default: use first two images from south-building dataset
        img1_path = "dataset/south-building/images/P1180141.JPG"
        img2_path = "dataset/south-building/images/P1180142.JPG"
        
        if os.path.exists(img1_path) and os.path.exists(img2_path):
            print("No arguments provided. Using default images from dataset.")
            print(f"Image 1: {img1_path}")
            print(f"Image 2: {img2_path}\n")
            
            two_view_geometry(img1_path, img2_path, detector_type='ORB', visualize=True)
        else:
            print("Usage:")
            print("  python two_view_geometry.py --img1 <path> --img2 <path>")
            print("\nExample:")
            print("  python two_view_geometry.py --img1 img1.jpg --img2 img2.jpg")
            print("\nOr use images from dataset:")
            print("  python two_view_geometry.py --img1 dataset/south-building/images/P1180141.JPG --img2 dataset/south-building/images/P1180142.JPG")
    else:
        main()



