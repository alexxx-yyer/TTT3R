#!/usr/bin/env python3
"""
Export point cloud data to ROS bag file for RViz visualization.

This script reads the output from demo.py and converts it to ROS bag format.
It publishes point clouds and camera poses that can be visualized in RViz.

Usage:
    python export_to_rosbag.py --output_dir OUTPUT_DIR --bag_file BAG_FILE [--frame_rate FRAME_RATE] [--topic_prefix TOPIC_PREFIX]

Example:
    python export_to_rosbag.py --output_dir demo_tmp --bag_file pointcloud.bag --frame_rate 30
"""

import os
import sys
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R

# Try to import ROS modules
try:
    import rospy
    import rosbag
    from sensor_msgs.msg import PointCloud2, PointField
    from geometry_msgs.msg import PoseStamped
    from std_msgs.msg import Header
    ROS_AVAILABLE = True
except ImportError as e:
    ROS_AVAILABLE = False
    print("Warning: ROS modules not available. Please install ROS or rosbag package.")
    print(f"Error: {e}")
    print("\nTo install ROS dependencies:")
    print("  For ROS Noetic: sudo apt-get install ros-noetic-rosbag")
    print("  Or install Python packages: pip install rospkg catkin_pkg")
    sys.exit(1)


def numpy_to_pointcloud2(points, colors, confidence=None, frame_id="map", timestamp=None):
    """
    Convert numpy arrays to ROS PointCloud2 message.
    
    Args:
        points: numpy array of shape [N, 3] containing xyz coordinates
        colors: numpy array of shape [N, 3] containing RGB colors (0-1 range)
        confidence: numpy array of shape [N] containing confidence values (optional)
        frame_id: ROS frame ID
        timestamp: rospy.Time object, if None uses current time
    
    Returns:
        sensor_msgs.PointCloud2 message
    """
    if timestamp is None:
        timestamp = rospy.Time.now()
    
    # Ensure points and colors are contiguous arrays
    points = np.ascontiguousarray(points, dtype=np.float32)
    colors = np.ascontiguousarray(colors, dtype=np.float32)
    
    # Clamp colors to [0, 1] and convert to uint8
    colors_uint8 = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
    
    # Pack RGB into UINT32 format: R << 16 | G << 8 | B
    # Note: RViz expects RGB in this format
    rgb_packed = (colors_uint8[:, 0].astype(np.uint32) << 16) | \
                 (colors_uint8[:, 1].astype(np.uint32) << 8) | \
                 colors_uint8[:, 2].astype(np.uint32)
    
    # Combine points, colors, and confidence
    N = points.shape[0]
    
    if confidence is not None:
        confidence = np.ascontiguousarray(confidence, dtype=np.float32)
        dtype_list = [
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('rgb', np.uint32),
            ('confidence', np.float32),
        ]
    else:
        dtype_list = [
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('rgb', np.uint32),
        ]
    
    # Create structured array
    cloud_array = np.empty(N, dtype=dtype_list)
    cloud_array['x'] = points[:, 0]
    cloud_array['y'] = points[:, 1]
    cloud_array['z'] = points[:, 2]
    cloud_array['rgb'] = rgb_packed
    if confidence is not None:
        cloud_array['confidence'] = confidence
    
    # Create PointCloud2 message
    msg = PointCloud2()
    msg.header = Header()
    msg.header.stamp = timestamp
    msg.header.frame_id = frame_id
    
    msg.height = 1
    msg.width = N
    
    if confidence is not None:
        msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('rgb', 12, PointField.UINT32, 1),
            PointField('confidence', 16, PointField.FLOAT32, 1),
        ]
        msg.point_step = 20  # 3*4 (float32) + 1*4 (uint32 rgb) + 1*4 (confidence float32) = 20 bytes
    else:
        msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('rgb', 12, PointField.UINT32, 1),
        ]
        msg.point_step = 16  # 3*4 (float32) + 1*4 (uint32 rgb) = 16 bytes
    
    msg.is_bigendian = False
    msg.row_step = msg.point_step * N
    msg.is_dense = True
    msg.data = cloud_array.tobytes()
    
    return msg


def matrix_to_pose_stamped(c2w_matrix, frame_id="map", timestamp=None):
    """
    Convert 4x4 camera-to-world matrix to ROS PoseStamped message.
    
    Args:
        c2w_matrix: 4x4 numpy array representing camera-to-world transformation
        frame_id: ROS frame ID
        timestamp: rospy.Time object, if None uses current time
    
    Returns:
        geometry_msgs.PoseStamped message
    """
    if timestamp is None:
        timestamp = rospy.Time.now()
    
    # Extract rotation and translation
    R_matrix = c2w_matrix[:3, :3]
    t = c2w_matrix[:3, 3]
    
    # Convert rotation matrix to quaternion
    rotation = R.from_matrix(R_matrix)
    quat = rotation.as_quat()  # [x, y, z, w]
    
    # Create PoseStamped message
    pose = PoseStamped()
    pose.header = Header()
    pose.header.stamp = timestamp
    pose.header.frame_id = frame_id
    
    pose.pose.position.x = t[0]
    pose.pose.position.y = t[1]
    pose.pose.position.z = t[2]
    
    pose.pose.orientation.x = quat[0]
    pose.pose.orientation.y = quat[1]
    pose.pose.orientation.z = quat[2]
    pose.pose.orientation.w = quat[3]
    
    return pose


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
        
        # Use the points directly from the saved data if available
        # Otherwise, reconstruct from depth
        # For now, we'll use the depth to reconstruct points
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


def export_to_rosbag(output_dir, bag_file, frame_rate=30.0, topic_prefix="/ttt3r"):
    """
    Export point cloud data to ROS bag file.
    
    Args:
        output_dir: Directory containing the output data from demo.py
        bag_file: Output ROS bag file path
        frame_rate: Frame rate for the bag file (Hz)
        topic_prefix: Prefix for ROS topics
    """
    # Load data
    points_list, colors_list, confidence_list, poses_list = load_data_from_output_dir(output_dir)
    
    num_frames = len(points_list)
    frame_duration = rospy.Duration(1.0 / frame_rate)
    
    has_confidence = confidence_list[0] is not None if confidence_list else False
    
    print(f"\nWriting ROS bag file: {bag_file}")
    print(f"  Frames: {num_frames}")
    print(f"  Frame rate: {frame_rate} Hz")
    print(f"  Duration: {num_frames / frame_rate:.2f} seconds")
    print(f"  Confidence data: {'Yes' if has_confidence else 'No'}")
    print(f"  Topics:")
    print(f"    - {topic_prefix}/pointcloud")
    print(f"    - {topic_prefix}/camera_pose")
    
    # Initialize ROS (required for creating messages)
    rospy.init_node('ttt3r_rosbag_exporter', anonymous=True)
    
    # Open bag file for writing
    bag = rosbag.Bag(bag_file, 'w')
    
    try:
        start_time = rospy.Time.now()
        
        for i in range(num_frames):
            timestamp = start_time + frame_duration * i
            
            # Write point cloud with confidence if available
            pc_msg = numpy_to_pointcloud2(
                points_list[i],
                colors_list[i],
                confidence=confidence_list[i],
                frame_id="map",
                timestamp=timestamp
            )
            bag.write(f"{topic_prefix}/pointcloud", pc_msg, timestamp)
            
            # Write camera pose
            pose_msg = matrix_to_pose_stamped(
                poses_list[i],
                frame_id="map",
                timestamp=timestamp
            )
            bag.write(f"{topic_prefix}/camera_pose", pose_msg, timestamp)
            
            if (i + 1) % 10 == 0:
                print(f"  Written {i + 1}/{num_frames} frames...")
        
        print(f"\nSuccessfully wrote {num_frames} frames to {bag_file}")
        print(f"Bag file size: {os.path.getsize(bag_file) / (1024*1024):.2f} MB")
        
    finally:
        bag.close()
        print("Bag file closed.")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Export TTT3R output to ROS bag file for RViz visualization."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory containing output data from demo.py (should contain depth/, color/, camera/ subdirectories)",
    )
    parser.add_argument(
        "--bag_file",
        type=str,
        required=True,
        help="Output ROS bag file path (e.g., pointcloud.bag)",
    )
    parser.add_argument(
        "--frame_rate",
        type=float,
        default=30.0,
        help="Frame rate for the bag file in Hz (default: 30.0)",
    )
    parser.add_argument(
        "--topic_prefix",
        type=str,
        default="/ttt3r",
        help="Prefix for ROS topics (default: /ttt3r)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    if not os.path.exists(args.output_dir):
        print(f"Error: Output directory does not exist: {args.output_dir}")
        sys.exit(1)
    
    # Create output directory for bag file if needed
    bag_dir = os.path.dirname(args.bag_file)
    if bag_dir and not os.path.exists(bag_dir):
        os.makedirs(bag_dir, exist_ok=True)
    
    try:
        export_to_rosbag(
            args.output_dir,
            args.bag_file,
            args.frame_rate,
            args.topic_prefix
        )
        print("\nExport completed successfully!")
        print(f"\nTo play the bag file in RViz:")
        print(f"  rosbag play {args.bag_file}")
        print(f"\nTopics in the bag:")
        print(f"  - {args.topic_prefix}/pointcloud (sensor_msgs/PointCloud2)")
        print(f"  - {args.topic_prefix}/camera_pose (geometry_msgs/PoseStamped)")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

