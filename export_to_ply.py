#!/usr/bin/env python3
"""
Export point cloud data to PLY files for visualization.

This script reads the output from demo.py and converts it to PLY format.
PLY files can be viewed in tools like MeshLab, CloudCompare, or Open3D.

Usage:
    python export_to_ply.py --output_dir OUTPUT_DIR --ply_file OUTPUT.ply [--merge]

Example:
    # Export each frame as separate PLY file
    python export_to_ply.py --output_dir demo_tmp --ply_file output

    # Merge all frames into a single PLY file
    python export_to_ply.py --output_dir demo_tmp --ply_file merged.ply --merge
"""

import os
import sys
import argparse
import numpy as np


def write_ply(filename, points, colors, confidence=None):
    """
    Write point cloud to PLY file.
    
    Args:
        filename: Output PLY file path
        points: numpy array of shape [N, 3] containing xyz coordinates
        colors: numpy array of shape [N, 3] containing RGB colors (0-1 range)
        confidence: numpy array of shape [N] containing confidence values (optional)
    """
    N = points.shape[0]
    
    # Ensure arrays are contiguous and correct type
    points = np.ascontiguousarray(points, dtype=np.float32)
    colors = np.ascontiguousarray(colors, dtype=np.float32)
    
    # Clamp colors to [0, 1] and convert to uint8
    colors_uint8 = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
    
    # Write PLY header
    with open(filename, 'wb') as f:
        # Write ASCII header
        f.write(b'ply\n')
        f.write(b'format binary_little_endian 1.0\n')
        f.write(f'element vertex {N}\n'.encode('ascii'))
        f.write(b'property float x\n')
        f.write(b'property float y\n')
        f.write(b'property float z\n')
        f.write(b'property uchar red\n')
        f.write(b'property uchar green\n')
        f.write(b'property uchar blue\n')
        
        if confidence is not None:
            f.write(b'property float confidence\n')
        
        f.write(b'end_header\n')
        
        # Write binary data
        if confidence is not None:
            confidence = np.ascontiguousarray(confidence, dtype=np.float32)
            # Interleave x, y, z, r, g, b, confidence
            for i in range(N):
                f.write(points[i].tobytes())
                f.write(colors_uint8[i].tobytes())
                f.write(confidence[i:i+1].tobytes())
        else:
            # Interleave x, y, z, r, g, b
            for i in range(N):
                f.write(points[i].tobytes())
                f.write(colors_uint8[i].tobytes())


def load_data_from_output_dir(output_dir):
    """
    Load point cloud data from output directory created by demo.py.
    
    Args:
        output_dir: Directory containing depth/, color/, conf/, camera/ subdirectories
    
    Returns:
        tuple: (points_list, colors_list, confidence_list, poses_list)
    """
    depth_dir = os.path.join(output_dir, "depth")
    color_dir = os.path.join(output_dir, "color")
    conf_dir = os.path.join(output_dir, "conf")
    camera_dir = os.path.join(output_dir, "camera")
    
    if not os.path.exists(depth_dir) or not os.path.exists(color_dir) or not os.path.exists(camera_dir):
        raise ValueError(f"Output directory {output_dir} does not contain required subdirectories")
    
    # Check if confidence directory exists
    has_confidence = os.path.exists(conf_dir)
    if not has_confidence:
        print(f"Warning: No confidence directory found in {output_dir}")
    
    # Get all frame files
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.npy')])
    color_files = sorted([f for f in os.listdir(color_dir) if f.endswith('.png')])
    camera_files = sorted([f for f in os.listdir(camera_dir) if f.endswith('.npz')])
    
    if has_confidence:
        conf_files = sorted([f for f in os.listdir(conf_dir) if f.endswith('.npy')])
        if len(depth_files) != len(conf_files):
            print(f"Warning: Mismatch in number of confidence files. Depth: {len(depth_files)}, Conf: {len(conf_files)}")
    
    if len(depth_files) != len(color_files) or len(depth_files) != len(camera_files):
        print(f"Warning: Mismatch in number of files. Depth: {len(depth_files)}, Color: {len(color_files)}, Camera: {len(camera_files)}")
    
    num_frames = min(len(depth_files), len(color_files), len(camera_files))
    
    points_list = []
    colors_list = []
    confidence_list = []
    poses_list = []
    
    print(f"Loading {num_frames} frames from {output_dir}...")
    
    for i in range(num_frames):
        # Load depth
        depth_path = os.path.join(depth_dir, depth_files[i])
        depth = np.load(depth_path)
        
        # Load color
        import cv2
        color_path = os.path.join(color_dir, color_files[i])
        color_bgr = cv2.imread(color_path)
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB) / 255.0
        
        # Load confidence if available
        if has_confidence and i < len(conf_files):
            conf_path = os.path.join(conf_dir, conf_files[i])
            confidence = np.load(conf_path)
        else:
            confidence = None
        
        # Load camera pose
        camera_path = os.path.join(camera_dir, camera_files[i])
        camera_data = np.load(camera_path)
        c2w = camera_data['pose']  # 4x4 matrix
        
        # Reconstruct 3D points from depth
        H, W = depth.shape
        intrinsics = camera_data['intrinsics']
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        # Generate pixel coordinates
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        
        # Convert to 3D points in camera frame
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth
        
        # Stack into point cloud (camera frame)
        points_cam = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        
        # Transform to world coordinates using c2w matrix
        points_homogeneous = np.hstack([points_cam, np.ones((points_cam.shape[0], 1))])
        points_world = (c2w @ points_homogeneous.T).T[:, :3]
        
        # Reshape colors to match points
        colors = color_rgb.reshape(-1, 3)
        
        # Reshape confidence to match points if available
        if confidence is not None:
            confidence_flat = confidence.reshape(-1)
        else:
            confidence_flat = None
        
        # Filter out invalid points (zero depth or NaN)
        valid_mask = (depth.reshape(-1) > 0) & ~np.isnan(points_world).any(axis=1)
        points_world = points_world[valid_mask]
        colors = colors[valid_mask]
        if confidence_flat is not None:
            confidence_flat = confidence_flat[valid_mask]
        
        points_list.append(points_world)
        colors_list.append(colors)
        confidence_list.append(confidence_flat)
        poses_list.append(c2w)
        
        if (i + 1) % 10 == 0:
            print(f"  Loaded {i + 1}/{num_frames} frames...")
    
    print(f"Successfully loaded {num_frames} frames")
    return points_list, colors_list, confidence_list, poses_list


def export_separate_ply_files(output_dir, ply_prefix):
    """
    Export each frame as a separate PLY file.
    
    Args:
        output_dir: Directory containing the output data from demo.py
        ply_prefix: Prefix for output PLY files (e.g., "frame" -> "frame_000000.ply", "frame_000001.ply", ...)
    """
    # Load data
    points_list, colors_list, confidence_list, poses_list = load_data_from_output_dir(output_dir)
    
    num_frames = len(points_list)
    has_confidence = confidence_list[0] is not None if confidence_list else False
    
    print(f"\nExporting {num_frames} frames as separate PLY files...")
    print(f"  Prefix: {ply_prefix}")
    print(f"  Confidence data: {'Yes' if has_confidence else 'No'}")
    
    # Create output directory if needed
    ply_dir = os.path.dirname(ply_prefix)
    if ply_dir and not os.path.exists(ply_dir):
        os.makedirs(ply_dir, exist_ok=True)
    
    for i in range(num_frames):
        ply_file = f"{ply_prefix}_{i:06d}.ply"
        write_ply(
            ply_file,
            points_list[i],
            colors_list[i],
            confidence=confidence_list[i] if has_confidence else None
        )
        
        if (i + 1) % 10 == 0:
            print(f"  Written {i + 1}/{num_frames} files...")
    
    print(f"\nSuccessfully exported {num_frames} PLY files")
    print(f"  Pattern: {ply_prefix}_*.ply")
    
    # Calculate total file size
    total_size = 0
    for i in range(num_frames):
        ply_file = f"{ply_prefix}_{i:06d}.ply"
        if os.path.exists(ply_file):
            total_size += os.path.getsize(ply_file)
    
    print(f"  Total size: {total_size / (1024*1024):.2f} MB")


def export_merged_ply_file(output_dir, ply_file, confidence_threshold=None):
    """
    Export all frames merged into a single PLY file.
    
    Args:
        output_dir: Directory containing the output data from demo.py
        ply_file: Output PLY file path
        confidence_threshold: Optional confidence threshold to filter points (e.g., 1.5)
    """
    # Load data
    points_list, colors_list, confidence_list, poses_list = load_data_from_output_dir(output_dir)
    
    num_frames = len(points_list)
    has_confidence = confidence_list[0] is not None if confidence_list else False
    
    print(f"\nMerging {num_frames} frames into a single PLY file...")
    print(f"  Output: {ply_file}")
    print(f"  Confidence data: {'Yes' if has_confidence else 'No'}")
    if confidence_threshold is not None and has_confidence:
        print(f"  Confidence threshold: {confidence_threshold}")
    
    # Merge all points
    all_points = []
    all_colors = []
    all_confidence = [] if has_confidence else None
    
    for i in range(num_frames):
        points = points_list[i]
        colors = colors_list[i]
        confidence = confidence_list[i] if has_confidence else None
        
        # Apply confidence threshold if specified
        if confidence_threshold is not None and confidence is not None:
            mask = confidence >= confidence_threshold
            points = points[mask]
            colors = colors[mask]
            if confidence is not None:
                confidence = confidence[mask]
        
        all_points.append(points)
        all_colors.append(colors)
        if has_confidence:
            all_confidence.append(confidence)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{num_frames} frames...")
    
    # Concatenate all data
    merged_points = np.vstack(all_points)
    merged_colors = np.vstack(all_colors)
    merged_confidence = np.concatenate(all_confidence) if has_confidence else None
    
    print(f"  Total points: {len(merged_points):,}")
    
    # Create output directory if needed
    ply_dir = os.path.dirname(ply_file)
    if ply_dir and not os.path.exists(ply_dir):
        os.makedirs(ply_dir, exist_ok=True)
    
    # Write merged PLY file
    write_ply(ply_file, merged_points, merged_colors, confidence=merged_confidence)
    
    print(f"\nSuccessfully exported merged PLY file: {ply_file}")
    print(f"  File size: {os.path.getsize(ply_file) / (1024*1024):.2f} MB")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Export TTT3R output to PLY files for visualization."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory containing output data from demo.py (should contain depth/, color/, camera/ subdirectories)",
    )
    parser.add_argument(
        "--ply_file",
        type=str,
        required=True,
        help="Output PLY file path or prefix. If --merge, this is the output file. Otherwise, this is the prefix (e.g., 'output' -> 'output_000000.ply')",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge all frames into a single PLY file instead of exporting separate files",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=None,
        help="Confidence threshold for filtering points (only used with --merge, e.g., 1.5)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    if not os.path.exists(args.output_dir):
        print(f"Error: Output directory does not exist: {args.output_dir}")
        sys.exit(1)
    
    try:
        if args.merge:
            export_merged_ply_file(args.output_dir, args.ply_file, args.confidence_threshold)
            print("\nYou can view the PLY file with:")
            print(f"  - MeshLab: meshlab {args.ply_file}")
            print(f"  - CloudCompare: cloudcompare {args.ply_file}")
            print(f"  - Open3D: python -c \"import open3d as o3d; o3d.visualization.draw_geometries([o3d.io.read_point_cloud('{args.ply_file}')])\"\n")
        else:
            # Remove .ply extension if present (we'll add it with frame numbers)
            ply_prefix = args.ply_file
            if ply_prefix.endswith('.ply'):
                ply_prefix = ply_prefix[:-4]
            
            export_separate_ply_files(args.output_dir, ply_prefix)
            print("\nYou can view the PLY files with:")
            print(f"  - MeshLab: meshlab {ply_prefix}_000000.ply")
            print(f"  - CloudCompare: cloudcompare {ply_prefix}_*.ply")
            print(f"  - Open3D: python -c \"import open3d as o3d; o3d.visualization.draw_geometries([o3d.io.read_point_cloud('{ply_prefix}_000000.ply')])\"\n")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
