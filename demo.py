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
import re
import numpy as np
import torch
import time
import glob
import random
import cv2
import argparse
import tempfile
import shutil
import gc
import atexit
from copy import deepcopy


def natural_sort_key(path):
    """
    自然排序键函数，正确处理文件名中的数字。
    例如: img1, img2, img10, img100 而不是 img1, img10, img100, img2
    """
    basename = os.path.basename(path)
    # 将字符串分割为文本和数字部分
    parts = re.split(r'(\d+)', basename)
    # 将数字部分转换为整数以实现正确排序
    return [int(part) if part.isdigit() else part.lower() for part in parts]


from add_ckpt_path import add_path_to_dust3r
import imageio.v2 as iio
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.decomposition import PCA
import datetime
from tqdm import tqdm
from skimage.filters import threshold_otsu, threshold_multiotsu
from einops import rearrange


class OnlineDiskCache:
    """
    参考 VGGT-Long 的内存管理机制：
    - Online 推理阶段：逐帧将结果保存到磁盘
    - Global Alignment 阶段：从磁盘读取数据进行后处理
    - 任务结束时自动清理临时文件
    """
    def __init__(self, cache_dir=None, delete_temp_files=True):
        if cache_dir is None:
            self.cache_dir = tempfile.mkdtemp(prefix="ttt3r_cache_")
        else:
            self.cache_dir = os.path.join(cache_dir, "_tmp_results")
            os.makedirs(self.cache_dir, exist_ok=True)

        self.pred_dir = os.path.join(self.cache_dir, "pred")
        self.view_dir = os.path.join(self.cache_dir, "view")
        os.makedirs(self.pred_dir, exist_ok=True)
        os.makedirs(self.view_dir, exist_ok=True)

        self.delete_temp_files = delete_temp_files
        self.num_frames = 0

        # 注册退出时清理
        atexit.register(self.cleanup)
        print(f"[OnlineDiskCache] 临时缓存目录: {self.cache_dir}")

    def save_frame(self, frame_idx, pred, view):
        """保存单帧的预测结果和视图数据到磁盘"""
        # 保存 pred（转换为 CPU numpy）
        pred_data = {}
        for k, v in pred.items():
            if isinstance(v, torch.Tensor):
                pred_data[k] = v.cpu().numpy()
            else:
                pred_data[k] = v
        np.save(os.path.join(self.pred_dir, f"{frame_idx:06d}.npy"), pred_data)

        # 保存 view（转换为 CPU numpy）
        view_data = {}
        for k, v in view.items():
            if isinstance(v, torch.Tensor):
                view_data[k] = v.cpu().numpy()
            else:
                view_data[k] = v
        np.save(os.path.join(self.view_dir, f"{frame_idx:06d}.npy"), view_data)

        self.num_frames = max(self.num_frames, frame_idx + 1)

    def save_outputs(self, outputs):
        """保存完整的 outputs 到磁盘，然后释放内存"""
        print(f"[OnlineDiskCache] 正在将 {len(outputs['pred'])} 帧保存到磁盘...")
        for i in tqdm(range(len(outputs["pred"])), desc="保存帧数据到磁盘"):
            self.save_frame(i, outputs["pred"][i], outputs["views"][i])

        self.num_frames = len(outputs["pred"])

        # 释放原始数据
        del outputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[OnlineDiskCache] 已保存 {self.num_frames} 帧，内存已释放")

    def load_frame(self, frame_idx, device='cpu'):
        """从磁盘加载单帧数据"""
        pred_path = os.path.join(self.pred_dir, f"{frame_idx:06d}.npy")
        view_path = os.path.join(self.view_dir, f"{frame_idx:06d}.npy")

        pred_data = np.load(pred_path, allow_pickle=True).item()
        view_data = np.load(view_path, allow_pickle=True).item()

        # 转换回 tensor
        pred = {}
        for k, v in pred_data.items():
            if isinstance(v, np.ndarray):
                pred[k] = torch.from_numpy(v).to(device)
            else:
                pred[k] = v

        view = {}
        for k, v in view_data.items():
            if isinstance(v, np.ndarray):
                view[k] = torch.from_numpy(v).to(device)
            else:
                view[k] = v

        return pred, view

    def load_frame_pair(self, idx1, idx2, device='cpu'):
        """加载一对相邻帧用于 global alignment"""
        pred1, view1 = self.load_frame(idx1, device)
        pred2, view2 = self.load_frame(idx2, device)
        return pred1, view1, pred2, view2

    def get_num_frames(self):
        """获取已保存的帧数"""
        return self.num_frames

    def cleanup(self):
        """清理所有临时文件"""
        if self.delete_temp_files and os.path.exists(self.cache_dir):
            try:
                # 计算释放的空间
                total_size = 0
                for root, dirs, files in os.walk(self.cache_dir):
                    for f in files:
                        total_size += os.path.getsize(os.path.join(root, f))

                shutil.rmtree(self.cache_dir)
                print(f"[OnlineDiskCache] 已清理缓存，释放 {total_size / 1024 / 1024 / 1024:.2f} GiB 磁盘空间")
            except Exception as e:
                print(f"[OnlineDiskCache] 清理缓存失败: {e}")

    def clear_memory(self):
        """强制清理 Python 内存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# 全局缓存实例
_global_disk_cache = None

def get_disk_cache(cache_dir=None):
    """获取或创建全局磁盘缓存实例"""
    global _global_disk_cache
    if _global_disk_cache is None:
        _global_disk_cache = OnlineDiskCache(cache_dir)
    return _global_disk_cache

# Set random seed for reproducibility.
random.seed(42)

framerate = 30

def forward_backward_permutations(n, interval=1):
    """Generate forward and backward permutations for pairwise inference."""
    original = list(range(n))
    result = [original]
    for i in range(1, n):
        new_list = original[i::interval]
        result.append(new_list)
        new_list = original[: i + 1][::-interval]
        result.append(new_list)
    return result

def listify(elems):
    return [x for e in elems for x in e]

def collate_with_cat(whatever, lists=False):
    """Collate and concatenate tensors from a nested structure."""
    if isinstance(whatever, dict):
        return {k: collate_with_cat(vals, lists=lists) for k, vals in whatever.items()}

    elif isinstance(whatever, (tuple, list)):
        if len(whatever) == 0:
            return whatever
        elem = whatever[0]
        T = type(whatever)

        if elem is None:
            return None
        if isinstance(elem, (bool, float, int, str)):
            return whatever
        if isinstance(elem, tuple):
            return T(collate_with_cat(x, lists=lists) for x in zip(*whatever))
        if isinstance(elem, dict):
            return {
                k: collate_with_cat([e[k] for e in whatever], lists=lists) for k in elem
            }

        if isinstance(elem, torch.Tensor):
            return listify(whatever) if lists else torch.cat(whatever)
        if isinstance(elem, np.ndarray):
            return (
                listify(whatever)
                if lists
                else torch.cat([torch.from_numpy(x) for x in whatever])
            )

        # otherwise, we just chain lists
        return sum(whatever, T())

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
        "--use_keyframe_memory_bank",
        action="store_true",
        help="Enable Keyframe Memory Bank mechanism",
    )
    parser.add_argument(
        "--keyframe_memory_lambda_time",
        type=float,
        default=0.5,
        help="Weight for time similarity in hybrid similarity calculation",
    )
    parser.add_argument(
        "--keyframe_memory_top_k",
        type=int,
        default=64,
        help="Number of top-k keyframes to retrieve from Memory Bank",
    )
    parser.add_argument(
        "--keyframe_memory_max_size",
        type=int,
        default=None,
        help="Maximum size of Keyframe Memory Bank (None for unlimited)",
    )
    parser.add_argument(
        "--keyframe_memory_diversity_threshold",
        type=float,
        default=0.9,
        help="Diversity threshold for keyframe sampling (only add if similarity < threshold)",
    )
    parser.add_argument(
        "--use_global_alignment",
        action="store_true",
        help="Enable global alignment optimization after inference",
    )
    parser.add_argument(
        "--ga_niter",
        type=int,
        default=300,
        help="Number of iterations for global alignment optimization",
    )
    parser.add_argument(
        "--ga_lr",
        type=float,
        default=0.01,
        help="Learning rate for global alignment optimization",
    )
    # Loop closure arguments
    parser.add_argument(
        "--enable_loop_closure",
        action="store_true",
        help="Enable loop closure detection and correction",
    )
    parser.add_argument(
        "--loop_closure_threshold",
        type=float,
        default=0.99,
        help="Similarity threshold for loop closure detection",
    )
    parser.add_argument(
        "--loop_closure_min_frame_gap",
        type=int,
        default=30,
        help="Minimum frame gap for loop closure detection",
    )
    parser.add_argument(
        "--loop_closure_keyframe_interval",
        type=int,
        default=10,
        help="Interval for adding keyframes to loop closure database",
    )
    # Depth consistency arguments
    parser.add_argument(
        "--use_depth_consistency",
        action="store_true",
        help="Enable depth consistency constraint from similar keyframes",
    )
    parser.add_argument(
        "--depth_consistency_top_k",
        type=int,
        default=3,
        help="Number of similar keyframes for depth consistency",
    )
    parser.add_argument(
        "--depth_consistency_weight",
        type=float,
        default=0.5,
        help="Weight for depth consistency fusion (0-1)",
    )
    # SLAM-aware keyframe bank arguments
    parser.add_argument(
        "--use_slam_aware_keyframe_bank",
        action="store_true",
        help="Enable SLAM-aware keyframe bank with multi-signal decision and pose graph",
    )
    parser.add_argument(
        "--slam_baseline_threshold",
        type=float,
        default=0.1,
        help="Baseline threshold for keyframe selection (relative to scene scale)",
    )
    parser.add_argument(
        "--slam_rotation_threshold",
        type=float,
        default=15.0,
        help="Rotation threshold in degrees for keyframe selection",
    )
    parser.add_argument(
        "--slam_coverage_threshold",
        type=float,
        default=0.7,
        help="View coverage threshold (0-1) for keyframe selection",
    )
    parser.add_argument(
        "--slam_reproj_threshold",
        type=float,
        default=5.0,
        help="Reprojection error threshold in pixels for keyframe selection",
    )
    parser.add_argument(
        "--slam_alpha_feature",
        type=float,
        default=0.3,
        help="Weight for feature similarity in SLAM-aware retrieval",
    )
    parser.add_argument(
        "--slam_beta_overlap",
        type=float,
        default=0.5,
        help="Weight for view overlap in SLAM-aware retrieval",
    )
    parser.add_argument(
        "--slam_gamma_graph",
        type=float,
        default=0.2,
        help="Weight for graph distance penalty in SLAM-aware retrieval",
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


def _prepare_output_with_global_alignment(outdir, device, ga_niter, ga_lr, disk_cache):
    """
    从磁盘读取数据进行 global alignment（参考 VGGT-Long 的内存管理机制）。

    Args:
        outdir: 输出目录
        device: 计算设备
        ga_niter: global alignment 迭代次数
        ga_lr: global alignment 学习率
        disk_cache: 磁盘缓存实例

    Returns:
        tuple: (points, colors, confidence, camera parameters dictionary)
    """
    from cloud_opt.dust3r_opt import global_aligner, GlobalAlignerMode
    import imageio.v2 as iio

    print("Applying global alignment optimization (VGGT-Long style)...")

    num_frames = disk_cache.get_num_frames()
    print(f"[DiskCache] 从磁盘读取 {num_frames} 帧数据进行 global alignment...")

    # 逐帧从磁盘加载并构建 pairwise 数据
    view1_list = []
    view2_list = []
    pred1_list = []
    pred2_list = []

    for i in tqdm(range(num_frames - 1), desc="从磁盘加载帧对"):
        pred1, view1 = disk_cache.load_frame(i, device='cpu')
        pred2, view2 = disk_cache.load_frame(i + 1, device='cpu')

        view1_list.append(view1)
        view2_list.append(view2)
        pred1_list.append(pred1)
        pred2_list.append(pred2)

        # 每加载一批后清理内存
        if (i + 1) % 50 == 0:
            gc.collect()

    print(f"[DiskCache] 已加载 {len(view1_list)} 对帧")

    # Collate the data
    output_ga = {
        "view1": collate_with_cat(view1_list),
        "view2": collate_with_cat(view2_list),
        "pred1": collate_with_cat(pred1_list),
        "pred2": collate_with_cat(pred2_list),
    }

    # 释放临时列表
    del view1_list, view2_list, pred1_list, pred2_list
    disk_cache.clear_memory()
    print("[Memory] 数据合并完成")

    # Run global alignment
    with torch.enable_grad():
        mode = GlobalAlignerMode.PointCloudOptimizer
        scene = global_aligner(
            output_ga,
            device=device,
            mode=mode,
            verbose=True,
        )

        # 释放 output_ga
        del output_ga
        disk_cache.clear_memory()
        print("[Memory] 已释放 output_ga")

        _ = scene.compute_global_alignment(
            init="mst",
            niter=ga_niter,
            schedule="linear",
            lr=ga_lr,
        )

    scene.clean_pointcloud()

    # 清理磁盘缓存文件
    disk_cache.cleanup()
    print("[DiskCache] 临时缓存已清理")

    pts3d = scene.get_pts3d()
    depths = scene.get_depthmaps()
    poses = scene.get_im_poses()
    focals = scene.get_focals()
    pps = scene.get_principal_points()
    confs = scene.get_conf(mode="none")

    # Convert to the expected format
    pts3ds_other = [pts.detach().cpu().unsqueeze(0) for pts in pts3d]
    depths_ga = [d.detach().cpu().unsqueeze(0) for d in depths]
    colors_ga = [torch.from_numpy(img).unsqueeze(0) for img in scene.imgs]
    confs_ga = [conf.detach().cpu().unsqueeze(0) for conf in confs]

    cam_dict = {
        "focal": focals.detach().cpu().numpy(),
        "pp": pps.detach().cpu().numpy(),
        "R": poses.detach().cpu().numpy()[..., :3, :3],
        "t": poses.detach().cpu().numpy()[..., :3, 3],
    }

    # Save global alignment results
    depths_tosave = torch.cat(depths_ga)
    pts3ds_other_tosave = torch.cat(pts3ds_other)
    conf_self_tosave = torch.cat(confs_ga)
    colors_tosave = torch.cat(colors_ga)
    cam2world_tosave = poses.detach().cpu()
    intrinsics_tosave = torch.eye(3).unsqueeze(0).repeat(cam2world_tosave.shape[0], 1, 1)
    intrinsics_tosave[:, 0, 0] = focals[:, 0].detach().cpu()
    intrinsics_tosave[:, 1, 1] = focals[:, 0].detach().cpu()
    intrinsics_tosave[:, 0, 2] = pps[:, 0].detach().cpu()
    intrinsics_tosave[:, 1, 2] = pps[:, 1].detach().cpu()

    # Save files
    for subdir in ["depth", "conf", "color", "camera"]:
        path = os.path.join(outdir, subdir)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

    for f_id in range(len(depths_tosave)):
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

    return pts3ds_other, colors_ga, confs_ga, cam_dict


def prepare_output(outputs, outdir, revisit=1, use_pose=True, use_global_alignment=False, device="cuda", ga_niter=300, ga_lr=0.01, disk_cache=None):
    """
    Process inference outputs to generate point clouds and camera parameters for visualization.

    Args:
        outputs (dict): Inference outputs (可以为 None，如果 disk_cache 已提供).
        revisit (int): Number of revisits per view.
        use_pose (bool): Whether to transform points using camera pose.
        use_global_alignment (bool): Whether to apply global alignment optimization.
        device (str): Device for global alignment.
        ga_niter (int): Number of iterations for global alignment.
        ga_lr (float): Learning rate for global alignment.
        disk_cache (OnlineDiskCache): 磁盘缓存实例，用于从磁盘读取数据.

    Returns:
        tuple: (points, colors, confidence, camera parameters dictionary)
    """
    from src.dust3r.utils.camera import pose_encoding_to_camera
    from src.dust3r.post_process import estimate_focal_knowing_depth
    from src.dust3r.utils.geometry import geotrf, matrix_cumprod
    import roma
    from viser_utils import convert_scene_output_to_glb

    # 如果使用 global alignment 且有 disk_cache，直接跳到 global alignment
    # 数据将从磁盘读取，不需要 outputs
    if use_global_alignment and disk_cache is not None:
        return _prepare_output_with_global_alignment(
            outdir, device, ga_niter, ga_lr, disk_cache
        )

    # 常规处理流程（不使用 global alignment，或没有 disk_cache）
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

    # # convert_scene_output_to_glb(outdir, (colors_tosave * 255).to(torch.uint8), pts3ds_other_tosave, conf_other_tosave > 1, focal, cam2world_tosave, as_pointcloud=True)

    # Save files (without global alignment)
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
    
    return pts3ds_other, colors, conf_other, cam_dict

def parse_seq_path(p, frame_interval=1):
    global framerate
    
    if os.path.isdir(p):
        all_img_paths = sorted(glob.glob(f"{p}/*"), key=natural_sort_key)
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


def visualize_loop_closures(cam_dict, loop_closures, output_dir):
    """
    Visualize camera trajectory with loop closure connections.

    Args:
        cam_dict: Dictionary containing camera parameters (R, t, focal, pp)
        loop_closures: List of loop closure dictionaries
        output_dir: Directory to save the visualization
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Extract camera positions
    R = cam_dict['R']  # [N, 3, 3]
    t = cam_dict['t']  # [N, 3]

    # Camera positions in world coordinates
    # For c2w matrices, t is already the camera position
    positions = t  # [N, 3]

    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot camera trajectory
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
            'b-', linewidth=1, alpha=0.6, label='Camera Trajectory')

    # Mark start and end points
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2],
               c='green', s=100, marker='o', label='Start')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2],
               c='red', s=100, marker='s', label='End')

    # Plot loop closure connections
    for lc in loop_closures:
        current_idx = lc['current_idx']
        matched_idx = lc['matched_idx']
        confidence = lc['confidence']

        if current_idx < len(positions) and matched_idx < len(positions):
            p1 = positions[current_idx]
            p2 = positions[matched_idx]

            # Draw loop closure connection (red dashed line)
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                    'r--', linewidth=2, alpha=0.8)

            # Mark the loop closure points
            ax.scatter(p1[0], p1[1], p1[2], c='orange', s=60, marker='^')
            ax.scatter(p2[0], p2[1], p2[2], c='purple', s=60, marker='v')

    # Add legend for loop closures
    if len(loop_closures) > 0:
        ax.plot([], [], 'r--', linewidth=2, label=f'Loop Closures ({len(loop_closures)})')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Camera Trajectory with Loop Closures\n({len(loop_closures)} loops detected)')
    ax.legend()

    # Save figure
    output_path = os.path.join(output_dir, 'loop_closure_trajectory.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Loop closure trajectory visualization saved to: {output_path}")

    # Also create a 2D top-down view (X-Z plane)
    fig2, ax2 = plt.subplots(figsize=(10, 10))
    ax2.plot(positions[:, 0], positions[:, 2], 'b-', linewidth=1, alpha=0.6, label='Camera Trajectory')
    ax2.scatter(positions[0, 0], positions[0, 2], c='green', s=100, marker='o', label='Start')
    ax2.scatter(positions[-1, 0], positions[-1, 2], c='red', s=100, marker='s', label='End')

    for lc in loop_closures:
        current_idx = lc['current_idx']
        matched_idx = lc['matched_idx']

        if current_idx < len(positions) and matched_idx < len(positions):
            p1 = positions[current_idx]
            p2 = positions[matched_idx]
            ax2.plot([p1[0], p2[0]], [p1[2], p2[2]], 'r--', linewidth=2, alpha=0.8)
            ax2.scatter(p1[0], p1[2], c='orange', s=60, marker='^')
            ax2.scatter(p2[0], p2[2], c='purple', s=60, marker='v')

    if len(loop_closures) > 0:
        ax2.plot([], [], 'r--', linewidth=2, label=f'Loop Closures ({len(loop_closures)})')

    ax2.set_xlabel('X')
    ax2.set_ylabel('Z')
    ax2.set_title(f'Camera Trajectory (Top-Down View)\n({len(loop_closures)} loops detected)')
    ax2.legend()
    ax2.axis('equal')

    output_path_2d = os.path.join(output_dir, 'loop_closure_trajectory_2d.png')
    plt.savefig(output_path_2d, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Loop closure trajectory (2D) visualization saved to: {output_path_2d}")


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
    from src.dust3r.inference import inference, inference_recurrent, inference_recurrent_lighter
    from src.dust3r.model import ARCroco3DStereo
    from viser_utils import PointCloudViewer

    # Prepare image file paths.
    img_paths, tmpdirname = parse_seq_path(args.seq_path, args.frame_interval)
    if not img_paths:
        print(f"No images found in {args.seq_path}. Please verify the path.")
        return

    print(f"Found {len(img_paths)} images in {args.seq_path}.")
    img_mask = [True] * len(img_paths)

    # Prepare input views.
    print("Preparing input views...")
    views = prepare_input(
        img_paths=img_paths,
        img_mask=img_mask,
        size=args.size,
        revisit=1,
        update=True,
        reset_interval=args.reset_interval
    )
    if tmpdirname is not None:
        shutil.rmtree(tmpdirname)

    # Load and prepare the model.
    print(f"Loading model from {args.model_path}...")
    model = ARCroco3DStereo.from_pretrained(args.model_path).to(device)

    # Debug: print model config
    print(f"Model config: head_type={model.head_type}, pose_head_flag={model.pose_head_flag}")
    print(f"Has pose_retriever: {hasattr(model, 'pose_retriever') and model.pose_retriever is not None}")

    model.config.model_update_type = args.model_update_type
    
    # Set Keyframe Memory Bank parameters
    model.config.use_keyframe_memory_bank = args.use_keyframe_memory_bank
    model.config.keyframe_memory_lambda_time = args.keyframe_memory_lambda_time
    model.config.keyframe_memory_top_k = args.keyframe_memory_top_k
    model.config.keyframe_memory_max_size = args.keyframe_memory_max_size
    model.config.keyframe_memory_diversity_threshold = args.keyframe_memory_diversity_threshold

    # Initialize Keyframe Memory Bank if enabled (V2 with anchor mechanism)
    if args.use_keyframe_memory_bank:
        from src.dust3r.model import KeyframeMemoryBankV2, LocalMemory
        from functools import partial
        import torch.nn as nn

        # Note: global_feat is projected by proj_q (k_dim -> v_dim), so feat_dim = dec_embed_dim = 768
        model.keyframe_memory_bank = KeyframeMemoryBankV2(
            max_size=args.keyframe_memory_max_size or 64,
            device=device,
            diversity_threshold=args.keyframe_memory_diversity_threshold,
            min_interval=getattr(args, 'keyframe_memory_min_interval', 10),
            num_anchor_frames=1,  # 保护初始帧不被剪枝
            feat_dim=model.dec_embed_dim,      # 768, proj_global_feat 维度
            pose_feat_dim=model.dec_embed_dim  # 768, out_pose_feat 维度
        )

        # Enable pose_head_flag since keyframe memory bank requires it
        model.pose_head_flag = True

        # Initialize pose_token and pose_retriever if they don't exist
        if not hasattr(model, 'pose_retriever') or model.pose_retriever is None:
            print("Initializing pose_retriever for keyframe memory bank...")
            model.pose_token = nn.Parameter(
                torch.randn(1, 1, model.dec_embed_dim) * 0.02, requires_grad=False
            ).to(device)
            model.pose_retriever = LocalMemory(
                size=model.config.local_mem_size,
                k_dim=model.enc_embed_dim,
                v_dim=model.dec_embed_dim,
                num_heads=model.dec_num_heads,
                mlp_ratio=4,
                qkv_bias=True,
                attn_drop=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                rope=None,
            ).to(device)

        print(f"Keyframe Memory Bank enabled: lambda_time={args.keyframe_memory_lambda_time}, "
              f"top_k={args.keyframe_memory_top_k}, max_size={args.keyframe_memory_max_size}, "
              f"diversity_threshold={args.keyframe_memory_diversity_threshold}")

    # Initialize Loop Closure components if enabled
    if args.enable_loop_closure:
        from src.dust3r.model import LoopClosureKeyframeDB

        model.config.enable_loop_closure = True
        model.config.loop_closure_threshold = args.loop_closure_threshold
        model.config.loop_closure_min_frame_gap = args.loop_closure_min_frame_gap
        model.config.loop_closure_keyframe_interval = args.loop_closure_keyframe_interval

        # Enable pose_head_flag since loop closure requires it
        model.pose_head_flag = True

        # Initialize loop closure components if they don't exist
        if not hasattr(model, 'loop_closure_keyframe_db') or model.loop_closure_keyframe_db is None:
            print("Initializing loop closure components with memory recall...")
            model.loop_closure_keyframe_db = LoopClosureKeyframeDB(
                max_keyframes=100,
                feat_dim=model.enc_embed_dim,
                state_dim=model.dec_embed_dim,
                state_size=model.config.state_size,
                mem_size=model.config.local_mem_size,
                mem_dim=model.dec_embed_dim * 2,
            ).to(device)
            model.loop_closures = []

        print(f"Loop Closure with Memory Recall enabled: threshold={args.loop_closure_threshold}, "
              f"min_frame_gap={args.loop_closure_min_frame_gap}, "
              f"keyframe_interval={args.loop_closure_keyframe_interval}")

    # Configure depth consistency
    if args.use_depth_consistency:
        model.config.use_depth_consistency = True
        model.config.depth_consistency_top_k = args.depth_consistency_top_k
        model.config.depth_consistency_weight = args.depth_consistency_weight
        print(f"Depth Consistency enabled: top_k={args.depth_consistency_top_k}, "
              f"weight={args.depth_consistency_weight}")

    # Configure SLAM-aware Keyframe Bank
    if args.use_slam_aware_keyframe_bank:
        from src.dust3r.model import SLAMAwareKeyframeBank

        model.config.use_slam_aware_keyframe_bank = True
        model.config.slam_baseline_threshold = args.slam_baseline_threshold
        model.config.slam_rotation_threshold = args.slam_rotation_threshold
        model.config.slam_coverage_threshold = args.slam_coverage_threshold
        model.config.slam_reproj_threshold = args.slam_reproj_threshold
        model.config.slam_alpha_feature = args.slam_alpha_feature
        model.config.slam_beta_overlap = args.slam_beta_overlap
        model.config.slam_gamma_graph = args.slam_gamma_graph

        # Enable pose_head_flag since SLAM bank requires poses
        model.pose_head_flag = True

        # Initialize SLAM-aware keyframe bank
        if not hasattr(model, 'slam_keyframe_bank') or model.slam_keyframe_bank is None:
            print("Initializing SLAM-aware Keyframe Bank...")
            model.slam_keyframe_bank = SLAMAwareKeyframeBank(
                max_size=args.keyframe_memory_max_size or 100,
                device=device,
                baseline_threshold=args.slam_baseline_threshold,
                rotation_threshold=args.slam_rotation_threshold,
                coverage_threshold=args.slam_coverage_threshold,
                reproj_threshold=args.slam_reproj_threshold,
                alpha_feature=args.slam_alpha_feature,
                beta_overlap=args.slam_beta_overlap,
                gamma_graph=args.slam_gamma_graph,
            )

        print(f"SLAM-aware Keyframe Bank enabled: "
              f"baseline_th={args.slam_baseline_threshold}, "
              f"rotation_th={args.slam_rotation_threshold}, "
              f"alpha={args.slam_alpha_feature}, beta={args.slam_beta_overlap}, gamma={args.slam_gamma_graph}")

    model.eval()

    # Run inference.
    print("Running inference...")
    start_time = time.time()
    outputs, state_args = inference_recurrent_lighter(views, model, device)

    total_time = time.time() - start_time
    per_frame_time = total_time / len(views)
    FPS_num = 1 / per_frame_time
    print(
        f"Inference completed in {total_time:.2f} seconds (average {per_frame_time:.2f} s per frame), FPS: {FPS_num:.2f}."
    )

    # 应用回溯尺度校正（如果启用了深度一致性）
    if args.use_depth_consistency and hasattr(model, 'keyframe_memory_bank') and model.keyframe_memory_bank is not None:
        drift_detected, cumulative_scale = model.keyframe_memory_bank.detect_scale_drift(threshold=0.05)
        if drift_detected:
            print(f"\n[Retrospective] 检测到尺度漂移: cumulative={cumulative_scale:.4f}")
            outputs = model.keyframe_memory_bank.apply_retrospective_correction(outputs)
        else:
            print(f"[Retrospective] 尺度漂移在可接受范围内: cumulative={cumulative_scale:.4f}")

    # 如果使用 global alignment，先将 outputs 保存到磁盘（参考 VGGT-Long）
    disk_cache = None
    if args.use_global_alignment:
        print("[VGGT-Long Style] 将 online 推理结果保存到磁盘...")
        disk_cache = OnlineDiskCache(cache_dir=args.output_dir)
        disk_cache.save_outputs(outputs)
        # outputs 已在 save_outputs 中被释放
        outputs = None

    # Process outputs for visualization.
    print("Preparing output for visualization...")
    pts3ds_other, colors, conf, cam_dict = prepare_output(
        outputs, args.output_dir, 1, True,
        use_global_alignment=args.use_global_alignment,
        device=device,
        ga_niter=args.ga_niter,
        ga_lr=args.ga_lr,
        disk_cache=disk_cache
    )

    # Convert tensors to numpy arrays for visualization.
    pts3ds_to_vis = [p.cpu().numpy() for p in pts3ds_other]
    colors_to_vis = [c.cpu().numpy() for c in colors]
    edge_colors = [None] * len(pts3ds_to_vis)

    # Print loop closure summary if enabled
    if args.enable_loop_closure and hasattr(model, 'loop_closures') and len(model.loop_closures) > 0:
        print(f"\n{'='*60}")
        print(f"Loop Closure Summary: {len(model.loop_closures)} loop(s) detected")
        print(f"{'='*60}")
        for lc in model.loop_closures:
            print(f"  Frame {lc['current_idx']} -> Frame {lc['matched_idx']} (confidence: {lc['confidence']:.3f})")
        print(f"{'='*60}\n")

        # Save loop closure information to file
        loop_closure_file = os.path.join(args.output_dir, "loop_closures.txt")
        with open(loop_closure_file, 'w') as f:
            f.write(f"Loop Closure Summary: {len(model.loop_closures)} loop(s) detected\n")
            f.write("="*60 + "\n")
            for lc in model.loop_closures:
                f.write(f"Frame {lc['current_idx']} -> Frame {lc['matched_idx']} (confidence: {lc['confidence']:.3f})\n")
        print(f"Loop closure information saved to: {loop_closure_file}")

        # Visualize loop closures on trajectory
        visualize_loop_closures(cam_dict, model.loop_closures, args.output_dir)

    # Create and run the point cloud viewer.
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
