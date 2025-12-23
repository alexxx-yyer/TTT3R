#!/usr/bin/env python3
"""
3D Point Cloud Inference and Visualization Script

This script performs inference using the ARCroco3DStereo model and visualizes the
resulting 3D point clouds with the PointCloudViewer. Use the command-line arguments
to adjust parameters such as the model checkpoint path, image sequence directory,
image size, device, etc.

Usage:
    python demo.py [--model_path MODEL_PATH] [--seq_path SEQ_PATH] [--size IMG_SIZE]
                            [--device DEVICE] [--vis_threshold VIS_THRESHOLD] [--output_dir OUT_DIR]

Example:
    python demo.py --model_path src/cut3r_512_dpt_4_64.pth \
        --seq_path examples/001 --device cuda --size 512
"""

import os
import numpy as np
import torch
import time
import glob
import random
import cv2
import argparse
import tempfile
import shutil
from copy import deepcopy
from add_ckpt_path import add_path_to_dust3r
import imageio.v2 as iio
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.decomposition import PCA
import datetime
from tqdm import tqdm
from skimage.filters import threshold_otsu, threshold_multiotsu
from einops import rearrange

# Set random seed for reproducibility.
random.seed(42)

framerate = 30

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run 3D point cloud inference and visualization using ARCroco3DStereo."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="src/cut3r_512_dpt_4_64.pth",
        help="Path to the pretrained model checkpoint.",
    )
    parser.add_argument(
        "--seq_path",
        type=str,
        default="",
        help="Path to the directory containing the image sequence.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (e.g., 'cuda' or 'cpu').",
    )
    parser.add_argument(
        "--size",
        type=int,
        default="512",
        help="Shape that input images will be rescaled to; if using 224+linear model, choose 224 otherwise 512",
    )
    parser.add_argument(
        "--vis_threshold",
        type=float,
        default=1.5,
        help="Visualization threshold for the point cloud viewer. Ranging from 1 to INF",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./demo_tmp",
        help="value for tempfile.tempdir",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="port for the point cloud viewer",
    )
    parser.add_argument(
        "--model_update_type",
        type=str,
        default="cut3r",
        help="model update type: cut3r or ttt3r",
    )
    parser.add_argument(
        "--frame_interval",
        type=int,
        default=1,
        help="Frame interval for video processing (e.g., 1 means every frame, 2 means every other frame)",
    )
    parser.add_argument(
        "--reset_interval",
        type=int,
        default=1000000,
        help="Only used for demo, reset state for extremely long sequence, chunks are aligned via global camera poses",
    )
    parser.add_argument(
        "--downsample_factor",
        type=int,
        default=1,
        help="Downsample factor for the point cloud viewer",
    )
    parser.add_argument(
        "--lazy_load",
        action="store_true",
        help="Use lazy loading to reduce memory usage for long sequences",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=100,
        help="Save interval for streaming inference (default: 100 frames per chunk)",
    )
    parser.add_argument(
        "--save_ply",
        action="store_true",
        help="Save point clouds as PLY files (per-frame and combined)",
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=1.0,
        help="Confidence threshold for point cloud filtering (default: 1.0)",
    )
    parser.add_argument(
        "--skip_viewer",
        action="store_true",
        help="Skip launching the point cloud viewer (useful for batch processing)",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.02,
        help="Voxel size for downsampling combined point cloud (default: 0.02, set 0 to disable)",
    )
    return parser.parse_args()


def prepare_input(
    img_paths, img_mask, size, raymaps=None, raymap_mask=None, revisit=1, update=True, reset_interval=10000
):
    """
    Prepare input views for inference from a list of image paths.

    Args:
        img_paths (list): List of image file paths.
        img_mask (list of bool): Flags indicating valid images.
        size (int): Target image size.
        raymaps (list, optional): List of ray maps.
        raymap_mask (list, optional): Flags indicating valid ray maps.
        revisit (int): How many times to revisit each view.
        update (bool): Whether to update the state on revisits.

    Returns:
        list: A list of view dictionaries.
    """
    # Import image loader (delayed import needed after adding ckpt path).
    from src.dust3r.utils.image import load_images

    images = load_images(img_paths, size=size)
    views = []

    if raymaps is None and raymap_mask is None:
        # Only images are provided.
        for i in range(len(images)):
            view = {
                "img": images[i]["img"],
                "ray_map": torch.full(
                    (
                        images[i]["img"].shape[0],
                        6,
                        images[i]["img"].shape[-2],
                        images[i]["img"].shape[-1],
                    ),
                    torch.nan,
                ),
                "true_shape": torch.from_numpy(images[i]["true_shape"]),
                "idx": i,
                "instance": str(i),
                "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(
                    0
                ),
                "img_mask": torch.tensor(True).unsqueeze(0),
                "ray_mask": torch.tensor(False).unsqueeze(0),
                "update": torch.tensor(True).unsqueeze(0),
                "reset": torch.tensor((i+1) % reset_interval == 0).unsqueeze(0),
            }
            views.append(view)
            if (i+1) % reset_interval == 0:
                overlap_view = deepcopy(view)
                overlap_view["reset"] = torch.tensor(False).unsqueeze(0)
                views.append(overlap_view)
    else:
        # Combine images and raymaps.
        num_views = len(images) + len(raymaps)
        assert len(img_mask) == len(raymap_mask) == num_views
        assert sum(img_mask) == len(images) and sum(raymap_mask) == len(raymaps)

        j = 0
        k = 0
        for i in range(num_views):
            view = {
                "img": (
                    images[j]["img"]
                    if img_mask[i]
                    else torch.full_like(images[0]["img"], torch.nan)
                ),
                "ray_map": (
                    raymaps[k]
                    if raymap_mask[i]
                    else torch.full_like(raymaps[0], torch.nan)
                ),
                "true_shape": (
                    torch.from_numpy(images[j]["true_shape"])
                    if img_mask[i]
                    else torch.from_numpy(np.int32([raymaps[k].shape[1:-1][::-1]]))
                ),
                "idx": i,
                "instance": str(i),
                "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(
                    0
                ),
                "img_mask": torch.tensor(img_mask[i]).unsqueeze(0),
                "ray_mask": torch.tensor(raymap_mask[i]).unsqueeze(0),
                "update": torch.tensor(img_mask[i]).unsqueeze(0),
                "reset": torch.tensor((i+1) % reset_interval == 0).unsqueeze(0),
            }
            if img_mask[i]:
                j += 1
            if raymap_mask[i]:
                k += 1
            views.append(view)
            if (i+1) % reset_interval == 0:
                overlap_view = deepcopy(view)
                overlap_view["reset"] = torch.tensor(False).unsqueeze(0)
                views.append(overlap_view)
        assert j == len(images) and k == len(raymaps)

    if revisit > 1:
        new_views = []
        for r in range(revisit):
            for i, view in enumerate(views):
                new_view = deepcopy(view)
                new_view["idx"] = r * len(views) + i
                new_view["instance"] = str(r * len(views) + i)
                if r > 0 and not update:
                    new_view["update"] = torch.tensor(False).unsqueeze(0)
                new_views.append(new_view)
        return new_views

    return views


class LazyViewLoader:
    """
    Lazy view loader that loads images on demand.
    Compatible with forward_recurrent_lighter's iteration pattern.
    """
    def __init__(self, img_paths, size, reset_interval=10000):
        from src.dust3r.utils.image import LazyImageLoader
        self.img_loader = LazyImageLoader(img_paths, size)
        self.size = size
        self.reset_interval = reset_interval
        self._len = len(self.img_loader)

    def __len__(self):
        return self._len

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def _load_single_view(self, idx):
        """Load a single view at index."""
        img_data = self.img_loader[idx]
        view = {
            "img": img_data["img"],
            "ray_map": torch.full(
                (
                    img_data["img"].shape[0],
                    6,
                    img_data["img"].shape[-2],
                    img_data["img"].shape[-1],
                ),
                torch.nan,
            ),
            "true_shape": torch.from_numpy(img_data["true_shape"]),
            "idx": idx,
            "instance": str(idx),
            "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0),
            "img_mask": torch.tensor(True).unsqueeze(0),
            "ray_mask": torch.tensor(False).unsqueeze(0),
            "update": torch.tensor(True).unsqueeze(0),
            "reset": torch.tensor((idx + 1) % self.reset_interval == 0).unsqueeze(0),
        }
        return view

    def __getitem__(self, idx):
        """Load view(s) at index on demand. Supports both int and slice."""
        if isinstance(idx, slice):
            indices = range(*idx.indices(self._len))
            return [self._load_single_view(i) for i in indices]
        else:
            return self._load_single_view(idx)


def prepare_input_lazy(img_paths, size, reset_interval=10000):
    """
    Prepare a lazy view loader for inference.
    Images are loaded on demand, reducing memory usage for long sequences.

    Args:
        img_paths (list or str): List of image file paths or folder path.
        size (int): Target image size.
        reset_interval (int): Interval for state reset.

    Returns:
        LazyViewLoader: A lazy loader that yields views on demand.
    """
    return LazyViewLoader(img_paths, size, reset_interval)


def prepare_output(outputs, outdir, revisit=1, use_pose=True):
    """
    Process inference outputs to generate point clouds and camera parameters for visualization.

    Args:
        outputs (dict): Inference outputs.
        revisit (int): Number of revisits per view.
        use_pose (bool): Whether to transform points using camera pose.

    Returns:
        tuple: (points, colors, confidence, camera parameters dictionary)
    """
    from src.dust3r.utils.camera import pose_encoding_to_camera
    from src.dust3r.post_process import estimate_focal_knowing_depth
    from src.dust3r.utils.geometry import geotrf, matrix_cumprod
    import roma
    from viser_utils import convert_scene_output_to_glb


    # Only keep the outputs corresponding to one full pass.
    valid_length = len(outputs["pred"]) // revisit
    outputs["pred"] = outputs["pred"][-valid_length:]
    outputs["views"] = outputs["views"][-valid_length:]

    # delet overlaps: reset_mask=True outputs["pred"] and outputs["views"]
    reset_mask = torch.cat([view["reset"] for view in outputs["views"]], 0)
    shifted_reset_mask = torch.cat([torch.tensor(False).unsqueeze(0), reset_mask[:-1]], dim=0)

    outputs["pred"] = [
        pred for pred, mask in zip(outputs["pred"], shifted_reset_mask) if not mask]
    outputs["views"] = [
        view for view, mask in zip(outputs["views"], shifted_reset_mask) if not mask]
    reset_mask = reset_mask[~shifted_reset_mask]

    pts3ds_self_ls = [output["pts3d_in_self_view"].cpu() for output in outputs["pred"]]
    pts3ds_other = [output["pts3d_in_other_view"].cpu() for output in outputs["pred"]]
    conf_self = [output["conf_self"].cpu() for output in outputs["pred"]]
    conf_other = [output["conf"].cpu() for output in outputs["pred"]]
    pts3ds_self = torch.cat(pts3ds_self_ls, 0)

    # Recover camera poses.
    pr_poses = [
        pose_encoding_to_camera(pred["camera_pose"].clone()).cpu()
        for pred in outputs["pred"]
    ]

    if reset_mask.any():
        pr_poses = torch.cat(pr_poses, 0)
        identity = torch.eye(4, device=pr_poses.device)
        reset_poses = torch.where(reset_mask.unsqueeze(-1).unsqueeze(-1), pr_poses, identity)
        cumulative_bases = matrix_cumprod(reset_poses)
        shifted_bases = torch.cat([identity.unsqueeze(0), cumulative_bases[:-1]], dim=0)
        pr_poses = torch.einsum('bij,bjk->bik', shifted_bases, pr_poses)
        # Convert sequence_scale list
        pr_poses = list(pr_poses.unsqueeze(1).unbind(0))

    R_c2w = torch.cat([pr_pose[:, :3, :3] for pr_pose in pr_poses], 0)
    t_c2w = torch.cat([pr_pose[:, :3, 3] for pr_pose in pr_poses], 0)

    if use_pose:
        transformed_pts3ds_other = []
        for pose, pself in zip(pr_poses, pts3ds_self):
            transformed_pts3ds_other.append(geotrf(pose, pself.unsqueeze(0)))
        pts3ds_other = transformed_pts3ds_other
        conf_other = conf_self

    # Estimate focal length based on depth.
    B, H, W, _ = pts3ds_self.shape
    pp = torch.tensor([W // 2, H // 2], device=pts3ds_self.device).float().repeat(B, 1)
    focal = estimate_focal_knowing_depth(pts3ds_self, pp, focal_mode="weiszfeld")

    colors = [
        0.5 * (output["img"].permute(0, 2, 3, 1) + 1.0) for output in outputs["views"]
    ]

    cam_dict = {
        "focal": focal.cpu().numpy(),
        "pp": pp.cpu().numpy(),
        "R": R_c2w.cpu().numpy(),
        "t": t_c2w.cpu().numpy(),
    }

    pts3ds_self_tosave = pts3ds_self  # B, H, W, 3
    depths_tosave = pts3ds_self_tosave[..., 2]
    pts3ds_other_tosave = torch.cat(pts3ds_other)  # B, H, W, 3
    conf_self_tosave = torch.cat(conf_self)  # B, H, W
    conf_other_tosave = torch.cat(conf_other)  # B, H, W
    colors_tosave = torch.cat(
        [
            0.5 * (output["img"].permute(0, 2, 3, 1).cpu() + 1.0)
            for output in outputs["views"]
        ]
    )  # [B, H, W, 3]
    cam2world_tosave = torch.cat(pr_poses)  # B, 4, 4
    intrinsics_tosave = (
        torch.eye(3).unsqueeze(0).repeat(cam2world_tosave.shape[0], 1, 1)
    )  # B, 3, 3
    intrinsics_tosave[:, 0, 0] = focal.detach().cpu()
    intrinsics_tosave[:, 1, 1] = focal.detach().cpu()
    intrinsics_tosave[:, 0, 2] = pp[:, 0]
    intrinsics_tosave[:, 1, 2] = pp[:, 1]

    if os.path.exists(os.path.join(outdir, "depth")):
        shutil.rmtree(os.path.join(outdir, "depth"))
    if os.path.exists(os.path.join(outdir, "conf")):
        shutil.rmtree(os.path.join(outdir, "conf"))
    if os.path.exists(os.path.join(outdir, "color")):
        shutil.rmtree(os.path.join(outdir, "color"))
    if os.path.exists(os.path.join(outdir, "camera")):
        shutil.rmtree(os.path.join(outdir, "camera"))
    os.makedirs(os.path.join(outdir, "depth"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "conf"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "color"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "camera"), exist_ok=True)
    for f_id in range(len(pts3ds_self)):
        depth = depths_tosave[f_id].cpu().numpy()
        conf = conf_self_tosave[f_id].cpu().numpy()
        color = colors_tosave[f_id].cpu().numpy()
        c2w = cam2world_tosave[f_id].cpu().numpy()
        intrins = intrinsics_tosave[f_id].cpu().numpy()
        np.save(os.path.join(outdir, "depth", f"{f_id:06d}.npy"), depth)
        np.save(os.path.join(outdir, "conf", f"{f_id:06d}.npy"), conf)
        iio.imwrite(
            os.path.join(outdir, "color", f"{f_id:06d}.png"),
            (color * 255).astype(np.uint8),
        )
        np.savez(
            os.path.join(outdir, "camera", f"{f_id:06d}.npz"),
            pose=c2w,
            intrinsics=intrins,
        )

    # # convert_scene_output_to_glb(outdir, (colors_tosave * 255).to(torch.uint8), pts3ds_other_tosave, conf_other_tosave > 1, focal, cam2world_tosave, as_pointcloud=True)
    return pts3ds_other, colors, conf_other, cam_dict

def parse_seq_path(p, frame_interval=1):
    global framerate
    
    if os.path.isdir(p):
        all_img_paths = sorted(glob.glob(f"{p}/*"))
        img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        img_paths = [path for path in all_img_paths 
                    if os.path.splitext(path.lower())[1] in img_extensions]
        
        if not img_paths:
            raise ValueError(f"No image files found in directory {p}")
        
        if frame_interval > 1:
            img_paths = img_paths[::frame_interval]
            print(f" - Image sequence: Total images: {len(all_img_paths)}, "
                  f"Frame interval: {frame_interval}, Images to process: {len(img_paths)}")
        
        framerate = 30.0 / frame_interval
        
        tmpdirname = None
    else:
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file {p}")
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if video_fps == 0:
            cap.release()
            raise ValueError(f"Error: Video FPS is 0 for {p}")
        
        framerate = video_fps / frame_interval
        
        frame_indices = list(range(0, total_frames, frame_interval))
        print(
            f" - Video FPS: {video_fps}, Frame Interval: {frame_interval}, Total Frames to Read: {len(frame_indices)}, Processed Framerate: {framerate}"
        )
        img_paths = []
        tmpdirname = tempfile.mkdtemp()
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(tmpdirname, f"frame_{i}.jpg")
            cv2.imwrite(frame_path, frame)
            img_paths.append(frame_path)
        cap.release()
    return img_paths, tmpdirname


def write_ply_binary(filename, points, colors, confidence=None):
    """
    Write point cloud to PLY file in binary format.

    Args:
        filename: Output PLY file path
        points: numpy array of shape [N, 3] containing xyz coordinates
        colors: numpy array of shape [N, 3] containing RGB colors (0-1 range)
        confidence: numpy array of shape [N] containing confidence values (optional)
    """
    N = points.shape[0]
    if N == 0:
        print(f"Warning: No points to write to {filename}")
        return

    points = np.ascontiguousarray(points, dtype=np.float32)
    colors_uint8 = (np.clip(colors, 0, 1) * 255).astype(np.uint8)

    with open(filename, 'wb') as f:
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

        # Write in batches for efficiency
        batch_size = 100000
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            pts_batch = points[start:end]
            col_batch = colors_uint8[start:end]

            if confidence is not None:
                conf_batch = np.ascontiguousarray(confidence[start:end], dtype=np.float32)
                # Create structured array for efficient writing
                dtype = np.dtype([('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                  ('r', 'u1'), ('g', 'u1'), ('b', 'u1'), ('conf', 'f4')])
                data = np.empty(end - start, dtype=dtype)
                data['x'] = pts_batch[:, 0]
                data['y'] = pts_batch[:, 1]
                data['z'] = pts_batch[:, 2]
                data['r'] = col_batch[:, 0]
                data['g'] = col_batch[:, 1]
                data['b'] = col_batch[:, 2]
                data['conf'] = conf_batch
            else:
                dtype = np.dtype([('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                  ('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
                data = np.empty(end - start, dtype=dtype)
                data['x'] = pts_batch[:, 0]
                data['y'] = pts_batch[:, 1]
                data['z'] = pts_batch[:, 2]
                data['r'] = col_batch[:, 0]
                data['g'] = col_batch[:, 1]
                data['b'] = col_batch[:, 2]

            f.write(data.tobytes())


def voxel_downsample(points, colors, confs, voxel_size):
    """
    Voxel downsampling - keep one point per voxel (the one with highest confidence).

    Args:
        points: [N, 3] numpy array
        colors: [N, 3] numpy array
        confs: [N] numpy array
        voxel_size: Size of voxel grid

    Returns:
        Downsampled points, colors, confs
    """
    if voxel_size <= 0 or len(points) == 0:
        return points, colors, confs

    # Compute voxel indices
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)

    # Create unique voxel keys
    # Shift to positive indices
    min_indices = voxel_indices.min(axis=0)
    voxel_indices = voxel_indices - min_indices

    # Create a single integer key for each voxel
    max_idx = voxel_indices.max(axis=0) + 1
    voxel_keys = (voxel_indices[:, 0] * max_idx[1] * max_idx[2] +
                  voxel_indices[:, 1] * max_idx[2] +
                  voxel_indices[:, 2])

    # Find unique voxels and keep point with highest confidence
    unique_keys, inverse_indices = np.unique(voxel_keys, return_inverse=True)

    # For each unique voxel, find the point with highest confidence
    n_voxels = len(unique_keys)
    best_indices = np.zeros(n_voxels, dtype=np.int64)
    best_confs = np.full(n_voxels, -np.inf)

    for i, (key_idx, conf) in enumerate(zip(inverse_indices, confs)):
        if conf > best_confs[key_idx]:
            best_confs[key_idx] = conf
            best_indices[key_idx] = i

    return points[best_indices], colors[best_indices], confs[best_indices]


def save_pointclouds(pts3ds_list, colors_list, conf_list, output_dir, conf_threshold=1.0, voxel_size=0.02):
    """
    Save point clouds as PLY files (per-frame and combined).

    Args:
        pts3ds_list: List of point cloud tensors [B, H, W, 3]
        colors_list: List of color tensors [B, H, W, 3]
        conf_list: List of confidence tensors [B, H, W]
        output_dir: Output directory
        conf_threshold: Confidence threshold for filtering
        voxel_size: Voxel size for downsampling combined point cloud (0 to disable)

    Returns:
        combined_ply_path: Path to the combined PLY file
    """
    pcd_dir = os.path.join(output_dir, "pcd")
    os.makedirs(pcd_dir, exist_ok=True)

    num_frames = len(pts3ds_list)
    print(f"Saving {num_frames} point cloud frames to {pcd_dir}...")

    # Collect all points for combined PLY
    all_points = []
    all_colors = []
    all_confs = []

    for i in range(num_frames):
        # Get data for this frame
        pts = pts3ds_list[i].cpu().numpy() if hasattr(pts3ds_list[i], 'cpu') else pts3ds_list[i]
        cols = colors_list[i].cpu().numpy() if hasattr(colors_list[i], 'cpu') else colors_list[i]
        conf = conf_list[i].cpu().numpy() if hasattr(conf_list[i], 'cpu') else conf_list[i]

        # Reshape to [N, 3] and [N]
        pts_flat = pts.reshape(-1, 3)
        cols_flat = cols.reshape(-1, 3)
        conf_flat = conf.reshape(-1)

        # Filter by confidence
        valid_mask = (conf_flat >= conf_threshold) & ~np.isnan(pts_flat).any(axis=1)
        pts_valid = pts_flat[valid_mask]
        cols_valid = cols_flat[valid_mask]
        conf_valid = conf_flat[valid_mask]

        # Save per-frame PLY
        ply_path = os.path.join(pcd_dir, f"{i:06d}.ply")
        write_ply_binary(ply_path, pts_valid, cols_valid, conf_valid)

        # Collect for combined
        all_points.append(pts_valid)
        all_colors.append(cols_valid)
        all_confs.append(conf_valid)

        if (i + 1) % 100 == 0:
            print(f"  Saved {i + 1}/{num_frames} frames...")

    print(f"Saved {num_frames} per-frame PLY files to {pcd_dir}/")

    # Save combined PLY
    combined_ply_path = os.path.join(pcd_dir, "combined.ply")
    print(f"Saving combined point cloud to {combined_ply_path}...")

    combined_points = np.vstack(all_points) if all_points else np.empty((0, 3))
    combined_colors = np.vstack(all_colors) if all_colors else np.empty((0, 3))
    combined_confs = np.concatenate(all_confs) if all_confs else np.empty(0)

    print(f"  Total points before downsampling: {len(combined_points):,}")

    # Apply voxel downsampling to reduce file size
    if voxel_size > 0 and len(combined_points) > 0:
        print(f"  Applying voxel downsampling (voxel_size={voxel_size})...")
        combined_points, combined_colors, combined_confs = voxel_downsample(
            combined_points, combined_colors, combined_confs, voxel_size
        )
        print(f"  Points after downsampling: {len(combined_points):,}")

    write_ply_binary(combined_ply_path, combined_points, combined_colors, combined_confs)

    file_size = os.path.getsize(combined_ply_path) / (1024 * 1024)
    print(f"  Combined PLY size: {file_size:.2f} MB")

    return combined_ply_path


def visualize_loop_closure(current_frame_idx, matched_frame_idx, current_img, matched_img, 
                          confidence, output_dir, loop_count=0):
    """
    Visualize loop closure detection result.
    
    Args:
        current_frame_idx: Current frame index
        matched_frame_idx: Matched keyframe index
        current_img: Current frame image (numpy array or tensor)
        matched_img: Matched keyframe image (numpy array or tensor)
        confidence: Detection confidence score
        output_dir: Output directory
        loop_count: Loop closure count (for naming)
    """
    import cv2
    import numpy as np
    
    # Convert to numpy if needed
    if isinstance(current_img, torch.Tensor):
        current_img = current_img.cpu().numpy()
    if isinstance(matched_img, torch.Tensor):
        matched_img = matched_img.cpu().numpy()
    
    # Normalize to [0, 255] if needed
    if current_img.max() <= 1.0:
        current_img = (current_img * 255).astype(np.uint8)
    if matched_img.max() <= 1.0:
        matched_img = (matched_img * 255).astype(np.uint8)
    
    # Handle different image formats
    if len(current_img.shape) == 4:  # [B, C, H, W]
        current_img = current_img[0].transpose(1, 2, 0)
    if len(matched_img.shape) == 4:
        matched_img = matched_img[0].transpose(1, 2, 0)
    
    if len(current_img.shape) == 3 and current_img.shape[0] == 3:  # [C, H, W]
        current_img = current_img.transpose(1, 2, 0)
    if len(matched_img.shape) == 3 and matched_img.shape[0] == 3:
        matched_img = matched_img.transpose(1, 2, 0)
    
    # Convert RGB to BGR for OpenCV if needed
    if current_img.shape[2] == 3:
        current_img = cv2.cvtColor(current_img, cv2.COLOR_RGB2BGR)
    if matched_img.shape[2] == 3:
        matched_img = cv2.cvtColor(matched_img, cv2.COLOR_RGB2BGR)
    
    # Resize to same height if needed
    h1, w1 = current_img.shape[:2]
    h2, w2 = matched_img.shape[:2]
    if h1 != h2:
        scale = h1 / h2
        new_w2 = int(w2 * scale)
        matched_img = cv2.resize(matched_img, (new_w2, h1))
    
    # Create side-by-side comparison
    comparison = np.hstack([current_img, matched_img])
    
    # Add text annotations
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(comparison, f'Current Frame: {current_frame_idx}', 
                (10, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(comparison, f'Matched Frame: {matched_frame_idx}', 
                (w1 + 10, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(comparison, f'Confidence: {confidence:.3f}', 
                (10, h1 - 20), font, 0.8, (0, 255, 255), 2)
    cv2.putText(comparison, f'Frame Gap: {current_frame_idx - matched_frame_idx}', 
                (w1 + 10, h1 - 20), font, 0.8, (0, 255, 255), 2)
    
    # Save image
    os.makedirs(os.path.join(output_dir, "loop_closures"), exist_ok=True)
    save_path = os.path.join(output_dir, "loop_closures", 
                            f"loop_{loop_count:04d}_frame_{current_frame_idx}_match_{matched_frame_idx}.png")
    cv2.imwrite(save_path, comparison)
    
    return save_path


def visualize_trajectory(poses, loop_pairs, output_dir):
    """
    Visualize camera trajectory with loop closures.
    
    Args:
        poses: List of camera poses (4x4 matrices or [tx, ty, tz, qw, qx, qy, qz])
        loop_pairs: List of (current_idx, matched_idx, confidence) tuples
        output_dir: Output directory
    """
    import matplotlib.pyplot as plt
    from scipy.spatial.transform import Rotation as R
    
    # Extract positions
    positions = []
    for pose in poses:
        if isinstance(pose, np.ndarray) and pose.shape == (4, 4):
            pos = pose[:3, 3]
        elif isinstance(pose, torch.Tensor) and pose.shape == (4, 4):
            pos = pose[:3, 3].cpu().numpy()
        elif len(pose) == 7:  # [tx, ty, tz, qw, qx, qy, qz]
            pos = np.array(pose[:3])
        else:
            continue
        positions.append(pos)
    
    if len(positions) == 0:
        return
    
    positions = np.array(positions)
    
    # Create trajectory plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot trajectory
    ax.plot(positions[:, 0], positions[:, 2], 'b-', alpha=0.6, linewidth=2, label='Trajectory')
    ax.scatter(positions[0, 0], positions[0, 2], c='green', s=100, marker='o', 
               label='Start', zorder=5)
    ax.scatter(positions[-1, 0], positions[-1, 2], c='red', s=100, marker='s', 
               label='End', zorder=5)
    
    # Plot loop closures
    for current_idx, matched_idx, confidence in loop_pairs:
        if current_idx < len(positions) and matched_idx < len(positions):
            ax.plot([positions[current_idx, 0], positions[matched_idx, 0]],
                   [positions[current_idx, 2], positions[matched_idx, 2]],
                   'r--', alpha=0.5, linewidth=1.5)
            ax.scatter(positions[current_idx, 0], positions[current_idx, 2],
                      c='orange', s=50, marker='*', zorder=4)
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Z (m)', fontsize=12)
    ax.set_title('Camera Trajectory with Loop Closures', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Save plot
    save_path = os.path.join(output_dir, 'loop_closure_trajectory.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path


def run_inference(args):
    """
    Execute the full inference and visualization pipeline.

    Args:
        args: Parsed command-line arguments.
    """
    # Set up the computation device.
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available. Switching to CPU.")
        device = "cpu"

    # Add the checkpoint path (required for model imports in the dust3r package).
    add_path_to_dust3r(args.model_path)

    # Import model and inference functions after adding the ckpt path.
    from src.dust3r.inference import inference, inference_recurrent, inference_recurrent_lighter, inference_recurrent_streaming, load_streaming_results
    from src.dust3r.model import ARCroco3DStereo
    from viser_utils import PointCloudViewer

    # Prepare image file paths.
    img_paths, tmpdirname = parse_seq_path(args.seq_path, args.frame_interval)
    if not img_paths:
        print(f"No images found in {args.seq_path}. Please verify the path.")
        return

    print(f"Found {len(img_paths)} images in {args.seq_path}.")
    
    # Store img_paths for loop closure visualization
    original_img_paths = img_paths.copy() if isinstance(img_paths, list) else img_paths

    # Prepare input views (lazy or eager loading).
    if args.lazy_load:
        print("Preparing input views (lazy loading - memory efficient)...")
        views = prepare_input_lazy(
            img_paths=img_paths,
            size=args.size,
            reset_interval=args.reset_interval
        )
    else:
        print("Preparing input views (eager loading)...")
        img_mask = [True] * len(img_paths)
        views = prepare_input(
            img_paths=img_paths,
            img_mask=img_mask,
            size=args.size,
            revisit=1,
            update=True,
            reset_interval=args.reset_interval
        )
    # For eager loading, we can clean up temp dir now since images are in memory.
    # For lazy loading, we must wait until after inference to clean up.
    if tmpdirname is not None and not args.lazy_load:
        shutil.rmtree(tmpdirname)

    # Load and prepare the model.
    print(f"Loading model from {args.model_path}...")
    model = ARCroco3DStereo.from_pretrained(args.model_path).to(device)
    model.config.model_update_type = args.model_update_type

    model.eval()

    # Run inference.
    print("Running inference...")
    start_time = time.time()

    if args.lazy_load:
        # Use streaming inference for memory efficiency
        os.makedirs(args.output_dir, exist_ok=True)
        cache_dir = os.path.join(args.output_dir, "inference_cache")
        metadata, state_args = inference_recurrent_streaming(
            views, model, device, cache_dir, verbose=True, save_interval=args.save_interval
        )

        total_time = time.time() - start_time
        per_frame_time = total_time / len(views)
        FPS_num = 1 / per_frame_time
        print(
            f"Inference completed in {total_time:.2f} seconds (average {per_frame_time:.2f} s per frame), FPS: {FPS_num:.2f}."
        )

        # Clean up temp directory after inference
        if tmpdirname is not None:
            shutil.rmtree(tmpdirname)

        # Load results from cache and process
        print("Loading results from cache and preparing output...")
        outputs = load_streaming_results(metadata)
        pts3ds_other, colors, conf, cam_dict = prepare_output(
            outputs, args.output_dir, 1, True
        )
        
        # Note: Loop closure visualization will be handled after this block
        # since we need colors_to_vis which is created later

        # Clean up cache after processing
        print("Cleaning up inference cache...")
        shutil.rmtree(cache_dir)
    else:
        outputs, state_args = inference_recurrent_lighter(views, model, device)

        total_time = time.time() - start_time
        per_frame_time = total_time / len(views)
        FPS_num = 1 / per_frame_time
        print(
            f"Inference completed in {total_time:.2f} seconds (average {per_frame_time:.2f} s per frame), FPS: {FPS_num:.2f}."
        )

        # Process outputs for visualization.
        print("Preparing output for visualization...")
        pts3ds_other, colors, conf, cam_dict = prepare_output(
            outputs, args.output_dir, 1, True
        )

    # Convert tensors to numpy arrays for visualization.
    pts3ds_to_vis = [p.cpu().numpy() for p in pts3ds_other]
    colors_to_vis = [c.cpu().numpy() for c in colors]
    edge_colors = [None] * len(pts3ds_to_vis)

    # Visualize loop closures if detected
    if hasattr(model, 'loop_closures') and len(model.loop_closures) > 0:
        print(f"\nFound {len(model.loop_closures)} loop closure(s). Visualizing...")
        loop_count = 0
        loop_pairs = []
        
        for loop_info in model.loop_closures:
            current_idx = loop_info['current_idx']
            matched_idx = loop_info['matched_idx']
            confidence = loop_info['confidence']
            
            # Load current frame image
            current_img = None
            if current_idx < len(colors_to_vis):
                current_img = colors_to_vis[current_idx]
            else:
                # Try to load from saved color images
                color_path = os.path.join(args.output_dir, "color", f"{current_idx:06d}.png")
                if os.path.exists(color_path):
                    current_img = cv2.imread(color_path)
                    current_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2RGB) / 255.0
                else:
                    # Try to load from original image paths
                    if current_idx < len(original_img_paths):
                        from src.dust3r.utils.image import load_images
                        matched_img_data = load_images([original_img_paths[current_idx]], size=args.size)
                        if matched_img_data:
                            current_img = matched_img_data[0]["img"]
                            if isinstance(current_img, torch.Tensor):
                                current_img = current_img.permute(1, 2, 0).cpu().numpy()
                            current_img = 0.5 * (current_img + 1.0)  # Denormalize
            
            if current_img is None:
                print(f"Warning: Could not load image for frame {current_idx}")
                continue
            
            # Load matched frame image
            matched_img = None
            if matched_idx < len(colors_to_vis):
                matched_img = colors_to_vis[matched_idx]
            else:
                # Try to load from saved color images
                color_path = os.path.join(args.output_dir, "color", f"{matched_idx:06d}.png")
                if os.path.exists(color_path):
                    matched_img = cv2.imread(color_path)
                    matched_img = cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB) / 255.0
                else:
                    # Try to load from original image paths
                    if matched_idx < len(original_img_paths):
                        from src.dust3r.utils.image import load_images
                        matched_img_data = load_images([original_img_paths[matched_idx]], size=args.size)
                        if matched_img_data:
                            matched_img = matched_img_data[0]["img"]
                            if isinstance(matched_img, torch.Tensor):
                                matched_img = matched_img.permute(1, 2, 0).cpu().numpy()
                            matched_img = 0.5 * (matched_img + 1.0)  # Denormalize
            
            if matched_img is not None:
                visualize_loop_closure(
                    current_idx, matched_idx, current_img, matched_img,
                    confidence, args.output_dir, loop_count
                )
                loop_pairs.append((current_idx, matched_idx, confidence))
                loop_count += 1
                print(f"  Saved loop closure visualization: frame {current_idx} <-> frame {matched_idx} (confidence: {confidence:.3f})")
        
        # Visualize trajectory if we have camera poses
        if len(loop_pairs) > 0 and 'R' in cam_dict and 't' in cam_dict:
            try:
                # Reconstruct poses from R and t
                poses = []
                for i in range(len(cam_dict['R'])):
                    R = cam_dict['R'][i]
                    t = cam_dict['t'][i]
                    pose = np.eye(4)
                    pose[:3, :3] = R
                    pose[:3, 3] = t
                    poses.append(pose)
                
                visualize_trajectory(poses, loop_pairs, args.output_dir)
                print(f"  Saved trajectory visualization: {args.output_dir}/loop_closure_trajectory.png")
            except Exception as e:
                print(f"  Warning: Could not visualize trajectory: {e}")
    
    # Save point clouds as PLY files if requested
    if args.save_ply:
        print("\nSaving point clouds...")
        combined_ply = save_pointclouds(
            pts3ds_other, colors, conf, args.output_dir, args.conf_threshold, args.voxel_size
        )
        print(f"\nPoint clouds saved to {args.output_dir}/pcd/")
        print(f"  - Per-frame: {args.output_dir}/pcd/000000.ply, ...")
        print(f"  - Combined:  {combined_ply}")
        print("\nView with: meshlab " + combined_ply)

    # Create and run the point cloud viewer (unless skipped)
    if args.skip_viewer:
        print("\nSkipping point cloud viewer (--skip_viewer specified)")
        print("Processing complete!")
    else:
        print("Launching point cloud viewer...")
        viewer = PointCloudViewer(
            model,
            state_args,
            pts3ds_to_vis,
            colors_to_vis,
            conf,
            cam_dict,
            device=device,
            edge_color_list=edge_colors,
            show_camera=True,
            vis_threshold=args.vis_threshold,
            size = args.size,
            port = args.port,
            downsample_factor=args.downsample_factor
        )
        viewer.run()


def main():
    args = parse_args()
    if not args.seq_path:
        print(
            "No inputs found! Please use our gradio demo if you would like to iteractively upload inputs."
        )
        return
    else:
        run_inference(args)


if __name__ == "__main__":
    main()
