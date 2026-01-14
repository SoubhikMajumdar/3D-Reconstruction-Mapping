"""
Multi-View Reconstruction: Incremental SfM
Builds 3D reconstruction by adding images incrementally
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os
from collections import defaultdict

# Import functions from previous modules
from two_view_geometry import (
    detect_and_describe_features,
    match_features,
    estimate_essential_matrix,
    estimate_camera_intrinsics,
    recover_pose
)


class MultiViewReconstruction:
    """Multi-view Structure from Motion reconstruction"""
    
    def __init__(self, images_dir, detector_type='ORB'):
        self.images_dir = images_dir
        self.detector_type = detector_type
        self.images = []
        self.image_names = []
        self.keypoints = []
        self.descriptors = []
        
        # 3D reconstruction data
        self.points_3d = []  # List of 3D points (Nx3)
        self.colors = []  # Colors for 3D points (Nx3)
        self.camera_poses = {}  # Camera poses: {image_idx: (R, t)}
        self.camera_matrices = {}  # Camera projection matrices: {image_idx: P}
        self.K = None  # Camera intrinsics
        
        # Point-to-image correspondences
        # point_to_images[i] = [(image_idx, keypoint_idx), ...] for 3D point i
        self.point_to_images = []
        
        # Track which 2D points correspond to which 3D points
        self.image_to_points = defaultdict(dict)  # {image_idx: {keypoint_idx: point_3d_idx}}
        
        # Initialize detector
        if detector_type == 'ORB':
            self.detector = cv2.ORB_create(nfeatures=5000)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:  # SIFT
            self.detector = cv2.SIFT_create(nfeatures=5000)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    
    def load_images(self):
        """Load all images from directory"""
        print("Loading images...")
        image_files = sorted([f for f in os.listdir(self.images_dir) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG'))])
        
        for img_file in image_files:
            img_path = os.path.join(self.images_dir, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                self.images.append(img)
                self.image_names.append(img_file)
        
        print(f"Loaded {len(self.images)} images")
        return len(self.images) > 0
    
    def detect_features_all(self):
        """Detect features in all images"""
        print("\nDetecting features in all images...")
        for i, img in enumerate(self.images):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            kp, desc = self.detector.detectAndCompute(gray, None)
            self.keypoints.append(kp)
            self.descriptors.append(desc)
            if (i + 1) % 10 == 0 or i == len(self.images) - 1:
                print(f"  Processed {i+1}/{len(self.images)} images: {len(kp)} keypoints")
    
    def initialize_two_views(self, idx1=0, idx2=1):
        """Initialize reconstruction with first two images"""
        print("\n" + "=" * 60)
        print("Step 1: Initializing with two views")
        print("=" * 60)
        
        # Estimate camera intrinsics
        self.K = estimate_camera_intrinsics(self.images[idx1].shape)
        print(f"\nCamera intrinsics K:\n{self.K}")
        
        # Extract matched points
        matches, _ = match_features(self.descriptors[idx1], self.descriptors[idx2], 
                                   self.detector_type)
        
        if len(matches) < 50:
            print(f"Warning: Only {len(matches)} matches between images {idx1} and {idx2}")
            return False
        
        pts1 = np.float32([self.keypoints[idx1][m.queryIdx].pt for m in matches])
        pts2 = np.float32([self.keypoints[idx2][m.trainIdx].pt for m in matches])
        
        # Estimate Essential matrix
        E, E_mask = estimate_essential_matrix(pts1, pts2, self.K, method='RANSAC')
        inliers = np.sum(E_mask)
        print(f"Essential matrix inliers: {inliers}/{len(matches)}")
        
        if inliers < 50:
            return False
        
        # Recover pose
        pts1_inlier = pts1[E_mask.ravel() == 1]
        pts2_inlier = pts2[E_mask.ravel() == 1]
        R, t, pose_mask = recover_pose(E, pts1_inlier, pts2_inlier, self.K)
        
        # Use all inliers for initial reconstruction
        pts1_final = pts1_inlier
        pts2_final = pts2_inlier
        
        # Set camera 1 at origin
        R1 = np.eye(3)
        t1 = np.zeros((3, 1))
        self.camera_poses[idx1] = (R1, t1)
        P1 = self.K @ np.hstack([R1, t1])
        self.camera_matrices[idx1] = P1
        
        # Set camera 2
        self.camera_poses[idx2] = (R, t)
        P2 = self.K @ np.hstack([R, t])
        self.camera_matrices[idx2] = P2
        
        # Triangulate initial points
        pts_4d = cv2.triangulatePoints(P1, P2, pts1_final.T, pts2_final.T)
        pts_3d = (pts_4d[:3] / pts_4d[3]).T
        
        # Filter points (must be in front of cameras)
        valid = (pts_3d[:, 2] > 0.1) & (pts_3d[:, 2] < 1000.0)
        pts_3d = pts_3d[valid]
        pts1_valid = pts1_final[valid]
        pts2_valid = pts2_final[valid]
        
        # Store 3D points
        self.points_3d = pts_3d.tolist()
        self.point_to_images = [[] for _ in range(len(pts_3d))]
        
        # Store correspondences
        for i, (pt1, pt2) in enumerate(zip(pts1_valid, pts2_valid)):
            kp1_idx = np.argmin([np.linalg.norm(np.array(pt1) - np.array(kp.pt)) 
                                 for kp in self.keypoints[idx1]])
            kp2_idx = np.argmin([np.linalg.norm(np.array(pt2) - np.array(kp.pt)) 
                                 for kp in self.keypoints[idx2]])
            
            self.point_to_images[i] = [(idx1, kp1_idx), (idx2, kp2_idx)]
            self.image_to_points[idx1][kp1_idx] = i
            self.image_to_points[idx2][kp2_idx] = i
        
        # Get colors
        for pt in pts1_valid:
            x, y = int(pt[0]), int(pt[1])
            if 0 <= y < self.images[idx1].shape[0] and 0 <= x < self.images[idx1].shape[1]:
                color = self.images[idx1][y, x]
                self.colors.append([int(color[2]), int(color[1]), int(color[0])])
        
        print(f"\nInitialized with {len(self.points_3d)} 3D points")
        print(f"Camera poses: {list(self.camera_poses.keys())}")
        
        return True
    
    def add_image(self, img_idx):
        """Add a new image to the reconstruction using PnP"""
        print(f"\nAdding image {img_idx} ({self.image_names[img_idx]})...")
        
        if img_idx in self.camera_poses:
            print(f"  Image {img_idx} already in reconstruction")
            return True
        
        # Find 2D-3D correspondences
        object_points = []
        image_points = []
        point_indices = []
        
        # For each existing 3D point, check if it's visible in the new image
        desc_new = self.descriptors[img_idx]
        kp_new = self.keypoints[img_idx]
        
        matches_found = 0
        for pt_3d_idx, (pt_3d, corr_list) in enumerate(zip(self.points_3d, self.point_to_images)):
            if len(corr_list) == 0:
                continue
            
            # Get descriptor from one of the images that see this point
            img_old_idx, kp_old_idx = corr_list[0]
            desc_old = self.descriptors[img_old_idx][kp_old_idx:kp_old_idx+1]
            
            if desc_old is None or len(desc_old) == 0:
                continue
            
            # Match with new image
            matches = self.matcher.knnMatch(desc_old, desc_new, k=2)
            
            if len(matches) > 0 and len(matches[0]) >= 2:
                m, n = matches[0][0], matches[0][1]
                if m.distance < 0.75 * n.distance:  # Ratio test
                    object_points.append(pt_3d)
                    image_points.append(kp_new[m.trainIdx].pt)
                    point_indices.append(pt_3d_idx)
                    matches_found += 1
        
        if len(object_points) < 6:  # Need at least 6 points for PnP
            print(f"  Not enough 2D-3D correspondences: {len(object_points)}")
            return False
        
        object_points = np.float32(object_points)
        image_points = np.float32(image_points)
        
        print(f"  Found {len(object_points)} 2D-3D correspondences")
        
        # Estimate camera pose using PnP
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points, image_points, self.K, None,
            iterationsCount=1000,
            reprojectionError=2.0,
            confidence=0.99
        )
        
        if not success or len(inliers) < 6:
            print(f"  PnP failed or insufficient inliers: {len(inliers) if success else 0}")
            return False
        
        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        t = tvec
        
        # Store camera pose
        self.camera_poses[img_idx] = (R, t)
        P = self.K @ np.hstack([R, t])
        self.camera_matrices[img_idx] = P
        
        # Update correspondences
        inlier_indices = inliers.ravel()
        for i in inlier_indices:
            pt_3d_idx = point_indices[i]
            kp_idx = np.argmin([np.linalg.norm(np.array(image_points[i]) - np.array(kp.pt)) 
                               for kp in kp_new])
            self.point_to_images[pt_3d_idx].append((img_idx, kp_idx))
            self.image_to_points[img_idx][kp_idx] = pt_3d_idx
        
        print(f"  Added image {img_idx}: {len(inliers)} inliers")
        
        # Triangulate new points from matches with other images
        self._triangulate_new_points(img_idx)
        
        return True
    
    def _triangulate_new_points(self, new_img_idx):
        """Triangulate new 3D points from the new image"""
        # Find matches with existing images
        for old_img_idx in self.camera_poses.keys():
            if old_img_idx == new_img_idx:
                continue
            
            matches, _ = match_features(self.descriptors[old_img_idx], 
                                       self.descriptors[new_img_idx], 
                                       self.detector_type)
            
            if len(matches) < 10:
                continue
            
            # Filter out matches that already correspond to 3D points
            new_matches = []
            for m in matches:
                old_kp_idx = m.queryIdx
                new_kp_idx = m.trainIdx
                
                # Check if these keypoints already correspond to 3D points
                if (old_kp_idx not in self.image_to_points[old_img_idx] and 
                    new_kp_idx not in self.image_to_points[new_img_idx]):
                    new_matches.append(m)
            
            if len(new_matches) < 10:
                continue
            
            # Triangulate new points
            pts_old = np.float32([self.keypoints[old_img_idx][m.queryIdx].pt for m in new_matches])
            pts_new = np.float32([self.keypoints[new_img_idx][m.trainIdx].pt for m in new_matches])
            
            P_old = self.camera_matrices[old_img_idx]
            P_new = self.camera_matrices[new_img_idx]
            
            pts_4d = cv2.triangulatePoints(P_old, P_new, pts_old.T, pts_new.T)
            pts_3d = (pts_4d[:3] / pts_4d[3]).T
            
            # Filter valid points
            valid = (pts_3d[:, 2] > 0.1) & (pts_3d[:, 2] < 1000.0)
            pts_3d_valid = pts_3d[valid]
            pts_old_valid = pts_old[valid]
            pts_new_valid = pts_new[valid]
            
            # Add new points
            start_idx = len(self.points_3d)
            valid_match_indices = np.where(valid)[0]  # Get indices of valid matches
            
            for local_idx, i in enumerate(valid_match_indices):
                pt_3d = pts_3d_valid[local_idx]
                self.points_3d.append(pt_3d.tolist())
                
                # Get the match for this valid point
                m = new_matches[i]
                
                # Find actual keypoint indices
                kp_old_idx = m.queryIdx
                kp_new_idx = m.trainIdx
                
                pt_idx = start_idx + local_idx
                self.point_to_images.append([(old_img_idx, kp_old_idx), (new_img_idx, kp_new_idx)])
                self.image_to_points[old_img_idx][kp_old_idx] = pt_idx
                self.image_to_points[new_img_idx][kp_new_idx] = pt_idx
                
                # Get color
                x, y = int(pts_old_valid[local_idx][0]), int(pts_old_valid[local_idx][1])
                if 0 <= y < self.images[old_img_idx].shape[0] and 0 <= x < self.images[old_img_idx].shape[1]:
                    color = self.images[old_img_idx][y, x]
                    self.colors.append([int(color[2]), int(color[1]), int(color[0])])
    
    def reconstruct(self, num_images=None):
        """Run multi-view reconstruction"""
        if not self.load_images():
            return False
        
        if len(self.images) < 2:
            print("Need at least 2 images")
            return False
        
        # Detect features
        self.detect_features_all()
        
        # Limit number of images if specified
        if num_images is None:
            num_images = min(10, len(self.images))  # Default to 10 images
        num_images = min(num_images, len(self.images))
        
        print(f"\nReconstructing with {num_images} images...")
        
        # Initialize with first two images
        if not self.initialize_two_views(0, 1):
            print("Failed to initialize reconstruction")
            return False
        
        # Add remaining images
        for i in range(2, num_images):
            self.add_image(i)
        
        print(f"\n" + "=" * 60)
        print("Multi-view reconstruction complete!")
        print(f"Total 3D points: {len(self.points_3d)}")
        print(f"Images used: {len(self.camera_poses)}")
        print("=" * 60)
        
        return True
    
    def visualize(self, output_file="multiview_result.png"):
        """Visualize 3D reconstruction"""
        if len(self.points_3d) == 0:
            print("No points to visualize")
            return
        
        points_3d = np.array(self.points_3d)
        colors = np.array(self.colors) / 255.0
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
                  c=colors, s=1, alpha=0.6)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Multi-View Reconstruction ({len(points_3d)} points, {len(self.camera_poses)} cameras)')
        
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
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {output_file}")
        plt.show()
    
    def save_results(self, output_dir="output_multiview"):
        """Save reconstruction results"""
        import json
        os.makedirs(output_dir, exist_ok=True)
        
        # Save point cloud
        points_file = os.path.join(output_dir, "points_3d.txt")
        with open(points_file, 'w') as f:
            for pt, color in zip(self.points_3d, self.colors):
                f.write(f"{pt[0]} {pt[1]} {pt[2]} {color[0]} {color[1]} {color[2]}\n")
        
        # Save camera poses and intrinsics
        camera_data = {
            'K': self.K.tolist(),
            'camera_poses': {},
            'image_names': self.image_names
        }
        for img_idx, (R, t) in self.camera_poses.items():
            camera_data['camera_poses'][str(img_idx)] = {
                'R': R.tolist(),
                't': t.tolist(),
                'image_name': self.image_names[img_idx]
            }
        
        cameras_file = os.path.join(output_dir, "cameras.json")
        with open(cameras_file, 'w') as f:
            json.dump(camera_data, f, indent=2)
        
        # Save correspondences for bundle adjustment
        correspondences = []
        for pt_3d_idx, corr_list in enumerate(self.point_to_images):
            for img_idx, kp_idx in corr_list:
                kp = self.keypoints[img_idx][kp_idx]
                correspondences.append({
                    'point_3d_idx': int(pt_3d_idx),
                    'image_idx': int(img_idx),
                    'keypoint_idx': int(kp_idx),
                    'observation': [float(kp.pt[0]), float(kp.pt[1])]
                })
        
        correspondences_file = os.path.join(output_dir, "correspondences.json")
        with open(correspondences_file, 'w') as f:
            json.dump(correspondences, f, indent=2)
        
        print(f"Results saved to: {output_dir}/")
        print(f"  - Point cloud: {points_file} ({len(self.points_3d)} points)")
        print(f"  - Camera poses: {cameras_file}")
        print(f"  - Correspondences: {correspondences_file} ({len(correspondences)} observations)")


def main():
    parser = argparse.ArgumentParser(description='Multi-View 3D Reconstruction')
    parser.add_argument(
        '--dataset',
        type=str,
        default='dataset/south-building/images',
        help='Path to images directory'
    )
    parser.add_argument(
        '--num-images',
        type=int,
        default=5,
        help='Number of images to use (default: 5)'
    )
    parser.add_argument(
        '--detector',
        type=str,
        default='ORB',
        choices=['ORB', 'SIFT'],
        help='Feature detector type'
    )
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Disable visualization'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset directory not found: {args.dataset}")
        return
    
    # Run reconstruction
    reconstruction = MultiViewReconstruction(args.dataset, detector_type=args.detector)
    
    if reconstruction.reconstruct(num_images=args.num_images):
        reconstruction.save_results()
        if not args.no_viz:
            reconstruction.visualize()
    else:
        print("Reconstruction failed")


if __name__ == "__main__":
    main()
