import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from copy import deepcopy
from functools import partial
from typing import Optional, Tuple, List, Any
from dataclasses import dataclass
from transformers import PretrainedConfig
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.file_utils import ModelOutput
import time
from dust3r.utils.misc import (
    fill_default_args,
    freeze_all_params,
    is_symmetrized,
    interleave,
    transpose_to_landscape,
)
from dust3r.heads import head_factory
from dust3r.utils.camera import PoseEncoder
from dust3r.patch_embed import get_patch_embed
import dust3r.utils.path_to_croco  # noqa: F401
from models.croco import CroCoNet, CrocoConfig  # noqa
from dust3r.blocks import (
    Block,
    DecoderBlock,
    Mlp,
    Attention,
    CrossAttention,
    DropPath,
)  # noqa

inf = float("inf")
from accelerate.logging import get_logger

from einops import rearrange
from dust3r.utils.device import to_cpu, to_gpu

printer = get_logger(__name__, log_level="DEBUG")


@dataclass
class ARCroco3DStereoOutput(ModelOutput):
    """
    Custom output class for ARCroco3DStereo.
    """

    ress: Optional[List[Any]] = None
    views: Optional[List[Any]] = None


def strip_module(state_dict):
    """
    Removes the 'module.' prefix from the keys of a state_dict.
    Args:
        state_dict (dict): The original state_dict with possible 'module.' prefixes.
    Returns:
        OrderedDict: A new state_dict with 'module.' prefixes removed.
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    return new_state_dict


def load_model(model_path, device, verbose=True):
    if verbose:
        print("... loading model from", model_path)
    ckpt = torch.load(model_path, map_location="cpu",weights_only=False)
    args = ckpt["args"].model.replace(
        "ManyAR_PatchEmbed", "PatchEmbedDust3R"
    )  # ManyAR only for aspect ratio not consistent
    if "landscape_only" not in args:
        args = args[:-2] + ", landscape_only=False))"
    else:
        args = args.replace(" ", "").replace(
            "landscape_only=True", "landscape_only=False"
        )
    assert "landscape_only=False" in args
    if verbose:
        print(f"instantiating : {args}")
    net = eval(args)
    s = net.load_state_dict(ckpt["model"], strict=False)
    if verbose:
        print(s)
    return net.to(device)


class ARCroco3DStereoConfig(PretrainedConfig):
    model_type = "arcroco_3d_stereo"

    def __init__(
        self,
        output_mode="pts3d",
        head_type="linear",  # or dpt
        depth_mode=("exp", -float("inf"), float("inf")),
        conf_mode=("exp", 1, float("inf")),
        pose_mode=("exp", -float("inf"), float("inf")),
        freeze="none",
        landscape_only=True,
        patch_embed_cls="PatchEmbedDust3R",
        ray_enc_depth=2,
        state_size=324,
        local_mem_size=256,
        state_pe="2d",
        state_dec_num_heads=16,
        depth_head=False,
        rgb_head=False,
        pose_conf_head=False,
        pose_head=False,
        model_update_type="cut3r",
        use_keyframe_memory_bank=False,
        keyframe_memory_lambda_time=0.1,
        keyframe_memory_top_k=5,
        keyframe_memory_max_size=None,
        keyframe_memory_diversity_threshold=0.9,  # 只添加相似度低于此阈值的帧（投影特征相似度很高）
        # Loop closure config
        enable_loop_closure=False,
        loop_closure_max_keyframes=100,
        loop_closure_threshold=0.85,
        loop_closure_min_frame_gap=30,
        loop_closure_keyframe_interval=10,
        **croco_kwargs,
    ):
        super().__init__()
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        self.pose_mode = pose_mode
        self.freeze = freeze
        self.landscape_only = landscape_only
        self.patch_embed_cls = patch_embed_cls
        self.ray_enc_depth = ray_enc_depth
        self.state_size = state_size
        self.state_pe = state_pe
        self.state_dec_num_heads = state_dec_num_heads
        self.local_mem_size = local_mem_size
        self.depth_head = depth_head
        self.rgb_head = rgb_head
        self.pose_conf_head = pose_conf_head
        self.pose_head = pose_head
        self.model_update_type = model_update_type
        self.use_keyframe_memory_bank = use_keyframe_memory_bank
        self.keyframe_memory_lambda_time = keyframe_memory_lambda_time
        self.keyframe_memory_top_k = keyframe_memory_top_k
        self.keyframe_memory_max_size = keyframe_memory_max_size
        self.keyframe_memory_diversity_threshold = keyframe_memory_diversity_threshold
        # Loop closure config
        self.enable_loop_closure = enable_loop_closure
        self.loop_closure_max_keyframes = loop_closure_max_keyframes
        self.loop_closure_threshold = loop_closure_threshold
        self.loop_closure_min_frame_gap = loop_closure_min_frame_gap
        self.loop_closure_keyframe_interval = loop_closure_keyframe_interval
        self.croco_kwargs = croco_kwargs


class LocalMemory(nn.Module):
    def __init__(
        self,
        size,
        k_dim,
        v_dim,
        num_heads,
        depth=2,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        norm_mem=True,
        rope=None,
    ) -> None:
        super().__init__()
        self.v_dim = v_dim
        self.proj_q = nn.Linear(k_dim, v_dim)
        self.masked_token = nn.Parameter(
            torch.randn(1, 1, v_dim) * 0.2, requires_grad=True
        ) # [1, 1, 768] pose mask token
        self.mem = nn.Parameter(
            torch.randn(1, size, 2 * v_dim) * 0.2, requires_grad=True
        ) # [1, 256, 1536] pose mem
        self.write_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    2 * v_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    attn_drop=attn_drop,
                    drop=drop,
                    drop_path=drop_path,
                    act_layer=act_layer,
                    norm_mem=norm_mem,
                    rope=rope,
                )
                for _ in range(depth)
            ]
        )
        self.read_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    2 * v_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    attn_drop=attn_drop,
                    drop=drop,
                    drop_path=drop_path,
                    act_layer=act_layer,
                    norm_mem=norm_mem,
                    rope=rope,
                )
                for _ in range(depth)
            ]
        )

    def update_mem(self, mem, feat_k, feat_v, return_attn=False):
        """
        mem_k: [B, size, C]
        mem_v: [B, size, C]
        feat_k: [B, 1, C] global_img_feat
        feat_v: [B, 1, C] out_pose_feat
        """
        feat_k = self.proj_q(feat_k)  # [B, 1, C]
        feat = torch.cat([feat_k, feat_v], dim=-1)

        attention_maps = []
        for blk in self.write_blocks:
            mem, _, self_attn, cross_attn = blk(mem, feat, None, None, return_attn=return_attn)
            attention_maps.append((self_attn, cross_attn))
        return mem

    def inquire(self, query, mem, return_attn=False):
        x = self.proj_q(query)  # [B, 1, C]
        x = torch.cat([x, self.masked_token.expand(x.shape[0], -1, -1)], dim=-1) # [1, 1, 768 global_img_feat_i + 768 masked_token(pose)]
        attention_maps = []
        for blk in self.read_blocks:
            x, _, self_attn, cross_attn = blk(x, mem, None, None, return_attn=return_attn)
            attention_maps.append((self_attn, cross_attn))
        return x[..., -self.v_dim :]

    def inquire_with_keyframes(self, query, mem, keyframe_features=None, return_attn=False):
        """
        Enhanced inquire method that can use keyframe features from Memory Bank.
        
        Args:
            query: [B, 1, C] query feature
            mem: [B, size, 2*C] memory
            keyframe_features: Optional [B, K, 2*C] keyframe features from Memory Bank
            return_attn: Whether to return attention maps
        """
        x = self.proj_q(query)  # [B, 1, C]
        x = torch.cat([x, self.masked_token.expand(x.shape[0], -1, -1)], dim=-1) # [B, 1, 2*C]
        
        # If keyframe features are provided, concatenate them with mem
        if keyframe_features is not None:
            # keyframe_features: [B, K, 2*C], mem: [B, size, 2*C]
            enhanced_mem = torch.cat([mem, keyframe_features], dim=1)  # [B, size+K, 2*C]
        else:
            enhanced_mem = mem
        
        attention_maps = []
        for blk in self.read_blocks:
            x, _, self_attn, cross_attn = blk(x, enhanced_mem, None, None, return_attn=return_attn)
            attention_maps.append((self_attn, cross_attn))
        return x[..., -self.v_dim :]


class KeyframeMemoryBank:
    """
    Keyframe Memory Bank for storing and retrieving keyframe features.

    Stores global image features and pose features with timestamps,
    and provides diversity-based retrieval using cosine similarity.

    Key design changes (inspired by InfiniteVGGT):
    - Diversity sampling: only add frames that are sufficiently different from existing keyframes
    - Diversity retrieval: retrieve low-similarity frames to provide diverse constraints
    """

    def __init__(self, max_size=None, device='cuda', diversity_threshold=0.9, min_interval=10):
        """
        Initialize the Keyframe Memory Bank.

        Args:
            max_size: Maximum number of keyframes to store (None for unlimited)
            device: Device to store tensors on
            diversity_threshold: Only add frames with max similarity below this threshold (default 0.9)
            min_interval: Minimum frame interval between keyframes (force add if exceeded)
        """
        self.max_size = max_size
        self.device = device
        self.diversity_threshold = diversity_threshold
        self.min_interval = min_interval
        self.features = []  # List of [B, 1, C] tensors
        self.pose_features = []  # List of [B, 1, C] tensors
        self.timestamps = []  # List of frame indices
        self.size = 0
    
    def add(self, frame_idx, global_feat, pose_feat, force_add=False):
        """
        Add a keyframe to the memory bank with diversity check.

        Only adds the frame if it's sufficiently different from existing keyframes
        (max cosine similarity < diversity_threshold), unless force_add=True.

        Args:
            frame_idx: Frame index (timestamp)
            global_feat: [B, 1, C] global image feature
            pose_feat: [B, 1, C] pose feature
            force_add: If True, skip diversity check (e.g., for first frame)

        Returns:
            bool: True if frame was added, False if rejected due to high similarity
        """
        # Move to device if needed
        global_feat = global_feat.to(self.device)
        pose_feat = pose_feat.to(self.device)

        # Diversity check: only add if sufficiently different from existing keyframes
        # OR if enough time has passed since last keyframe (min_interval)
        if not force_add and self.size > 0:
            max_sim = self._compute_max_similarity(global_feat)
            last_timestamp = self.timestamps[-1] if self.timestamps else 0
            time_since_last = frame_idx - last_timestamp

            # Force add if min_interval exceeded (temporal sampling)
            if time_since_last >= self.min_interval:
                if frame_idx % 50 == 0:
                    print(f"[KMB] Frame {frame_idx}: Force adding (interval={time_since_last} >= {self.min_interval})")
            elif max_sim >= self.diversity_threshold:
                # Too similar and not enough time passed, skip
                if frame_idx % 50 == 0:
                    print(f"[KMB Debug] Frame {frame_idx}: Skipped (max_sim={max_sim:.4f} >= {self.diversity_threshold})")
                return False
            else:
                if frame_idx % 50 == 0:
                    print(f"[KMB] Frame {frame_idx}: Adding (max_sim={max_sim:.4f} < {self.diversity_threshold})")

        self.features.append(global_feat.detach().clone())
        self.pose_features.append(pose_feat.detach().clone())
        self.timestamps.append(frame_idx)
        self.size += 1

        # If max_size is set and exceeded, remove most redundant frame (highest similarity to mean)
        if self.max_size is not None and self.size > self.max_size:
            self._evict_most_redundant()

        return True

    def _compute_max_similarity(self, query_feat):
        """
        Compute maximum cosine similarity between query and all existing keyframes.

        Args:
            query_feat: [B, 1, C] query feature

        Returns:
            float: Maximum similarity score
        """
        max_sim = -1.0
        for i in range(self.size):
            sim = self.compute_cosine_similarity(query_feat, self.features[i])
            sim_val = sim.mean().item()  # Average over batch
            if sim_val > max_sim:
                max_sim = sim_val
        return max_sim

    def _evict_most_redundant(self):
        """
        Evict the most redundant keyframe (highest similarity to mean feature).
        Inspired by InfiniteVGGT's eviction strategy.
        """
        if self.size <= 1:
            return

        # Compute mean feature
        all_features = torch.cat(self.features, dim=1)  # [B, size, C]
        mean_feat = all_features.mean(dim=1, keepdim=True)  # [B, 1, C]

        # Find keyframe with highest similarity to mean (most redundant)
        max_sim = -1.0
        max_idx = 0
        for i in range(self.size):
            sim = self.compute_cosine_similarity(self.features[i], mean_feat)
            sim_val = sim.mean().item()
            if sim_val > max_sim:
                max_sim = sim_val
                max_idx = i

        # Remove the most redundant keyframe
        self.features.pop(max_idx)
        self.pose_features.pop(max_idx)
        self.timestamps.pop(max_idx)
        self.size -= 1
    
    def compute_cosine_similarity(self, query_feat, bank_feat):
        """
        Compute cosine similarity between query feature and bank feature.
        
        Formula: sim_cos = sum_p (F_i_p · X_hat_p^T) / (||F_i_p|| · ||X_hat_p||)
        
        Args:
            query_feat: [B, 1, C] query feature
            bank_feat: [B, 1, C] bank feature
            
        Returns:
            [B, 1] cosine similarity scores
        """
        # Normalize features
        query_norm = F.normalize(query_feat, p=2, dim=-1)  # [B, 1, C]
        bank_norm = F.normalize(bank_feat, p=2, dim=-1)  # [B, 1, C]
        
        # Compute cosine similarity: dot product of normalized features
        # Sum over feature dimension p
        cosine_sim = (query_norm * bank_norm).sum(dim=-1, keepdim=True)  # [B, 1, 1]
        return cosine_sim.squeeze(-1)  # [B, 1]
    
    def compute_time_similarity(self, query_time, bank_time, device):
        """
        Compute time similarity using Gaussian decay.

        Formula: sim_time = exp(-(time(F_i) - time(X_hat))^2)

        Args:
            query_time: Scalar or tensor query timestamp
            bank_time: Scalar or tensor bank timestamp
            device: Device to create tensor on

        Returns:
            Time similarity score as tensor on specified device
        """
        time_diff = query_time - bank_time
        time_sim = torch.exp(torch.tensor(-(time_diff ** 2), dtype=torch.float32, device=device))
        return time_sim

    def compute_hybrid_similarity(self, query_feat, query_time, lambda_time):
        """
        Compute hybrid similarity for all keyframes in the bank.

        Formula: sim_hybrid = sim_cos + lambda_time * sim_time

        Args:
            query_feat: [B, 1, C] query feature
            query_time: Scalar query timestamp (frame index)
            lambda_time: Weight for time similarity

        Returns:
            List of [B, 1] hybrid similarity scores for each keyframe
        """
        if self.size == 0:
            return []

        device = query_feat.device
        similarities = []
        for i in range(self.size):
            # Compute cosine similarity
            cos_sim = self.compute_cosine_similarity(query_feat, self.features[i])  # [B, 1]

            # Compute time similarity on same device as query_feat
            time_sim = self.compute_time_similarity(query_time, self.timestamps[i], device)

            # Compute hybrid similarity
            hybrid_sim = cos_sim + lambda_time * time_sim
            similarities.append(hybrid_sim)
        
        return similarities
    
    def retrieve_top_k(self, query_feat, query_time, k, lambda_time):
        """
        Retrieve top-k most DIVERSE keyframes from the memory bank.

        Strategy: Select keyframes with LOWEST similarity to current frame,
        providing diverse constraints rather than redundant nearby frames.
        (Inspired by InfiniteVGGT's diversity-based selection)

        Args:
            query_feat: [B, 1, C] query feature
            query_time: Scalar query timestamp (frame index)
            k: Number of keyframes to retrieve
            lambda_time: Weight for time penalty (now penalizes nearby frames)

        Returns:
            keyframe_features: [B, K, 2*C] concatenated [global_feat, pose_feat] for top-k diverse keyframes
            indices: List of indices of selected keyframes
        """
        if self.size == 0:
            return None, []

        # Compute cosine similarities for all keyframes (without time component for diversity)
        device = query_feat.device
        similarities = []
        for i in range(self.size):
            cos_sim = self.compute_cosine_similarity(query_feat, self.features[i])  # [B, 1]
            similarities.append(cos_sim)

        if len(similarities) == 0:
            return None, []

        # Stack similarities: [size, B, 1]
        sim_stack = torch.stack(similarities, dim=0)  # [size, B, 1]
        sim_stack = sim_stack.squeeze(-1)  # [size, B]

        # DIVERSITY: Select LOWEST similarity frames (using -sim_stack for topk)
        # This provides diverse constraints instead of redundant nearby frames
        _, top_k_indices = torch.topk(-sim_stack[:, 0], k=min(k, self.size), dim=0)
        top_k_indices = top_k_indices.cpu().tolist()
        if not isinstance(top_k_indices, list):
            top_k_indices = [top_k_indices]

        # Retrieve selected keyframe features
        # Concatenate global_feat and pose_feat: [B, 1, C] + [B, 1, C] -> [B, 1, 2*C]
        keyframe_features_list = []
        for idx in top_k_indices:
            global_feat = self.features[idx]  # [B, 1, C]
            pose_feat = self.pose_features[idx]  # [B, 1, C]
            # Concatenate along feature dimension
            combined = torch.cat([global_feat, pose_feat], dim=-1)  # [B, 1, 2*C]
            keyframe_features_list.append(combined)

        if len(keyframe_features_list) > 0:
            # Stack: [K, B, 1, 2*C] -> [B, K, 1, 2*C] -> [B, K, 2*C]
            keyframe_features = torch.stack(keyframe_features_list, dim=0)  # [K, B, 1, 2*C]
            keyframe_features = keyframe_features.permute(1, 0, 2, 3)  # [B, K, 1, 2*C]
            keyframe_features = keyframe_features.squeeze(2)  # [B, K, 2*C]
        else:
            keyframe_features = None

        return keyframe_features, top_k_indices
    
    def clear(self):
        """Clear all stored keyframes."""
        self.features = []
        self.pose_features = []
        self.timestamps = []
        self.size = 0


class LoopClosureKeyframeDB(nn.Module):
    """
    Keyframe database for loop closure detection.
    Stores keyframe features, poses, and memory states for matching and recall.
    """
    def __init__(self, max_keyframes=100, feat_dim=1024, state_dim=768, state_size=324, mem_size=256, mem_dim=1536):
        super().__init__()
        self.max_keyframes = max_keyframes
        self.feat_dim = feat_dim
        self.state_dim = state_dim
        self.state_size = state_size
        self.mem_size = mem_size
        self.mem_dim = mem_dim
        # Runtime buffers (not learnable parameters)
        self.register_buffer('features', torch.zeros(1, max_keyframes, feat_dim))
        self.register_buffer('poses', torch.zeros(1, max_keyframes, 7))
        self.register_buffer('frame_ids', torch.zeros(1, max_keyframes, dtype=torch.long))
        self.register_buffer('count', torch.zeros(1, dtype=torch.long))
        # Memory state storage (for memory recall)
        self.state_feats = None  # Will be initialized in reset()
        self.mems = None         # Will be initialized in reset()

    def reset(self, batch_size, device=None):
        """Reset database for new sequence"""
        if device is None:
            device = self.features.device
        self.features = torch.zeros(batch_size, self.max_keyframes, self.feat_dim, device=device)
        self.poses = torch.zeros(batch_size, self.max_keyframes, 7, device=device)
        self.frame_ids = torch.zeros(batch_size, self.max_keyframes, dtype=torch.long, device=device)
        self.count = torch.zeros(batch_size, dtype=torch.long, device=device)
        # Initialize memory state storage
        self.state_feats = torch.zeros(batch_size, self.max_keyframes, self.state_size, self.state_dim, device=device)
        self.mems = torch.zeros(batch_size, self.max_keyframes, self.mem_size, self.mem_dim, device=device)

    def add_keyframe(self, feat, pose, frame_id, state_feat=None, mem=None, batch_mask=None):
        """
        Add keyframe to database with optional memory state
        Args:
            feat: [B, 1, feat_dim] global image feature
            pose: [B, 7] camera pose (tx, ty, tz, qw, qx, qy, qz)
            frame_id: int, current frame index
            state_feat: [B, state_size, state_dim] global state (optional)
            mem: [B, mem_size, mem_dim] local memory (optional)
            batch_mask: [B] optional mask for which batches to update
        """
        B = feat.shape[0]
        for b in range(B):
            if batch_mask is not None and not batch_mask[b]:
                continue
            idx = self.count[b] % self.max_keyframes
            self.features[b, idx] = feat[b, 0]
            self.poses[b, idx] = pose[b]
            self.frame_ids[b, idx] = frame_id
            # Save memory state if provided
            if state_feat is not None and self.state_feats is not None:
                self.state_feats[b, idx] = state_feat[b].detach()
            if mem is not None and self.mems is not None:
                self.mems[b, idx] = mem[b].detach()
            self.count[b] += 1

    def get_memory(self, batch_idx, keyframe_idx):
        """
        Retrieve memory state for a specific keyframe
        Args:
            batch_idx: batch index
            keyframe_idx: keyframe slot index (not frame_id)
        Returns:
            state_feat: [state_size, state_dim] or None
            mem: [mem_size, mem_dim] or None
        """
        if self.state_feats is None or self.mems is None:
            return None, None
        return self.state_feats[batch_idx, keyframe_idx], self.mems[batch_idx, keyframe_idx]

    def query(self, feat, top_k=5):
        """
        Query most similar keyframes
        Args:
            feat: [B, 1, feat_dim] query feature
            top_k: number of candidates to return
        Returns:
            top_sim: [B, top_k] similarity scores
            top_idx: [B, top_k] keyframe indices (slot indices, not frame_ids)
            top_poses: [B, top_k, 7] keyframe poses
            top_frame_ids: [B, top_k] keyframe frame indices
        """
        B = feat.shape[0]
        # Normalize features for cosine similarity
        feat_norm = F.normalize(feat[:, 0], dim=-1)  # [B, feat_dim]
        db_norm = F.normalize(self.features, dim=-1)  # [B, max_kf, feat_dim]

        # Compute similarities
        similarities = torch.bmm(feat_norm.unsqueeze(1), db_norm.transpose(1, 2)).squeeze(1)
        # [B, max_kf]

        # Mask unused slots
        valid_mask = torch.arange(self.max_keyframes, device=feat.device).unsqueeze(0) < self.count.unsqueeze(1)
        similarities = similarities.masked_fill(~valid_mask, -1e9)

        # Get top-k
        actual_k = min(top_k, self.max_keyframes)
        top_sim, top_idx = similarities.topk(actual_k, dim=-1)

        # Gather corresponding poses and frame ids
        top_poses = torch.gather(self.poses, 1, top_idx.unsqueeze(-1).expand(-1, -1, 7))
        top_frame_ids = torch.gather(self.frame_ids, 1, top_idx)

        return top_sim, top_idx, top_poses, top_frame_ids

    def query_with_memory(self, feat, current_frame_id, min_frame_gap=30, threshold=0.85):
        """
        Query for loop closure and return matching memory state
        Args:
            feat: [B, 1, feat_dim] query feature
            current_frame_id: current frame index
            min_frame_gap: minimum frame gap to consider as loop
            threshold: similarity threshold for loop detection
        Returns:
            loop_detected: [B] bool tensor
            best_state_feat: [B, state_size, state_dim] or None
            best_mem: [B, mem_size, mem_dim] or None
            confidence: [B] confidence scores
            matched_frame_id: [B] matched frame ids
        """
        B = feat.shape[0]
        device = feat.device

        # Query top-k keyframes
        top_sim, top_idx, top_poses, top_frame_ids = self.query(feat, top_k=5)

        # Check temporal gap
        frame_gap = current_frame_id - top_frame_ids
        valid_gap = frame_gap >= min_frame_gap

        # Select best valid match
        masked_sim = top_sim.masked_fill(~valid_gap, -1e9)
        best_sim, best_local_idx = masked_sim.max(dim=-1)  # [B]

        # Get the actual keyframe slot index
        batch_idx = torch.arange(B, device=device)
        best_slot_idx = top_idx[batch_idx, best_local_idx]  # [B]
        matched_frame_id = top_frame_ids[batch_idx, best_local_idx]  # [B]

        # Determine if loop is detected
        loop_detected = best_sim > threshold
        confidence = torch.clamp(best_sim, 0.0, 1.0)

        # Retrieve memory states for detected loops
        best_state_feat = None
        best_mem = None
        if loop_detected.any() and self.state_feats is not None:
            # Gather memory states for all batches (will mask later)
            best_state_feat = self.state_feats[batch_idx, best_slot_idx]  # [B, state_size, state_dim]
            best_mem = self.mems[batch_idx, best_slot_idx]  # [B, mem_size, mem_dim]

        return loop_detected, best_state_feat, best_mem, confidence, matched_frame_id


class LoopDetector(nn.Module):
    """
    Detect loop closure based on feature similarity (no learnable parameters).

    Uses cosine similarity between current frame features and keyframe database
    to detect when the camera returns to a previously visited location.
    """
    def __init__(self, feat_dim=1024, threshold=0.85, min_frame_gap=30):
        super().__init__()
        self.feat_dim = feat_dim
        self.threshold = threshold
        self.min_frame_gap = min_frame_gap

    def forward(self, current_feat, keyframe_db, current_frame_id):
        """
        Detect loop closure
        Args:
            current_feat: [B, 1, feat_dim] current frame feature
            keyframe_db: LoopClosureKeyframeDB instance
            current_frame_id: int, current frame index
        Returns:
            loop_detected: [B] bool tensor
            loop_frame_id: [B] matched frame id
            loop_pose: [B, 7] matched pose
            confidence: [B] confidence score
        """
        B = current_feat.shape[0]
        device = current_feat.device

        # Query keyframe database
        top_sim, top_idx, top_poses, top_frame_ids = keyframe_db.query(current_feat, top_k=5)

        # Check temporal gap (avoid matching recent frames)
        frame_gap = current_frame_id - top_frame_ids  # [B, 5]
        valid_gap = frame_gap >= self.min_frame_gap

        # Select best valid match
        masked_sim = top_sim.masked_fill(~valid_gap, -1e9)
        best_sim, best_idx = masked_sim.max(dim=-1)  # [B]

        # Get corresponding pose and frame id
        batch_idx = torch.arange(B, device=device)
        loop_pose = top_poses[batch_idx, best_idx]  # [B, 7]
        loop_frame_id = top_frame_ids[batch_idx, best_idx]  # [B]

        # Determine if loop is detected
        loop_detected = best_sim > self.threshold

        # Confidence is the similarity score (clamped)
        confidence = torch.clamp(best_sim, 0.0, 1.0)

        return loop_detected, loop_frame_id, loop_pose, confidence


class LoopCorrector(nn.Module):
    """
    Geometric loop closure corrector (no learnable parameters).

    When a loop is detected, this module applies a soft correction to the
    state and memory based on the pose discrepancy. The idea is that when
    we return to a previously visited location, we should partially reset
    our accumulated state to reduce drift.
    """
    def __init__(self, state_dim=768, mem_dim=1536, pose_dim=7,
                 state_correction_strength=0.3, mem_correction_strength=0.2):
        super().__init__()
        self.state_dim = state_dim
        self.mem_dim = mem_dim
        # Hyperparameters (not learned)
        self.state_correction_strength = state_correction_strength
        self.mem_correction_strength = mem_correction_strength

    def _compute_pose_error(self, current_pose, loop_pose):
        """
        Compute pose error magnitude between current and loop pose.

        Args:
            current_pose: [B, 7] (tx, ty, tz, qw, qx, qy, qz)
            loop_pose: [B, 7]

        Returns:
            error: [B] normalized pose error in [0, 1]
        """
        # Translation error (L2 distance)
        trans_current = current_pose[:, :3]
        trans_loop = loop_pose[:, :3]
        trans_error = torch.norm(trans_current - trans_loop, dim=-1)

        # Rotation error (quaternion distance)
        quat_current = F.normalize(current_pose[:, 3:], dim=-1)
        quat_loop = F.normalize(loop_pose[:, 3:], dim=-1)
        # Quaternion dot product gives cos(angle/2)
        quat_dot = torch.abs(torch.sum(quat_current * quat_loop, dim=-1))
        rot_error = 1.0 - quat_dot  # 0 when identical, 1 when opposite

        # Combine errors (normalize translation by typical scale)
        combined_error = trans_error / (trans_error.mean() + 1e-6) * 0.5 + rot_error * 0.5
        return torch.clamp(combined_error, 0.0, 1.0)

    def forward(self, state_feat, mem, current_pose, loop_pose, confidence):
        """
        Apply geometric loop closure correction.

        When a loop is detected with high confidence but the poses differ,
        we apply a soft correction that blends the current state towards
        a more neutral value, effectively reducing accumulated drift.

        Args:
            state_feat: [B, state_size, state_dim] global state
            mem: [B, mem_size, mem_dim] local memory
            current_pose: [B, 7] current estimated pose
            loop_pose: [B, 7] matched loop pose
            confidence: [B] detection confidence

        Returns:
            corrected_state: [B, state_size, state_dim]
            corrected_mem: [B, mem_size, mem_dim]
        """
        # Compute pose discrepancy
        pose_error = self._compute_pose_error(current_pose, loop_pose)

        # Correction strength: high confidence + high error = strong correction
        # The intuition: if we're confident we've returned to a place but our
        # pose estimate differs significantly, we have accumulated drift
        correction_factor = confidence * pose_error  # [B]
        correction_factor = correction_factor[:, None, None]  # [B, 1, 1]

        # Apply soft reset to state (blend towards mean)
        state_mean = state_feat.mean(dim=(1, 2), keepdim=True)
        state_correction = self.state_correction_strength * correction_factor
        corrected_state = state_feat * (1 - state_correction) + state_mean * state_correction

        # Apply soft reset to memory (blend towards mean)
        mem_mean = mem.mean(dim=(1, 2), keepdim=True)
        mem_correction = self.mem_correction_strength * correction_factor
        corrected_mem = mem * (1 - mem_correction) + mem_mean * mem_correction

        return corrected_state, corrected_mem


class ARCroco3DStereo(CroCoNet):
    config_class = ARCroco3DStereoConfig
    base_model_prefix = "arcroco3dstereo"
    supports_gradient_checkpointing = True

    def __init__(self, config: ARCroco3DStereoConfig):
        self.gradient_checkpointing = False
        self.fixed_input_length = True
        config.croco_kwargs = fill_default_args(
            config.croco_kwargs, CrocoConfig.__init__
        )
        self.patch_embed_cls = config.patch_embed_cls
        self.croco_args = config.croco_kwargs
        croco_cfg = CrocoConfig(**self.croco_args)
        super().__init__(croco_cfg)
        # CRITICAL FIX: PreTrainedModel.__init__ overwrites self.config with croco_cfg
        # We must restore the original ARCroco3DStereoConfig after super().__init__
        self.config = config
        self.enc_blocks_ray_map = nn.ModuleList(
            [
                Block(
                    self.enc_embed_dim,
                    16,
                    4,
                    qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    rope=self.rope,
                )
                for _ in range(config.ray_enc_depth)
            ]
        )
        self.enc_norm_ray_map = nn.LayerNorm(self.enc_embed_dim, eps=1e-6)
        self.dec_num_heads = self.croco_args["dec_num_heads"]
        self.pose_head_flag = config.pose_head
        if self.pose_head_flag:
            self.pose_token = nn.Parameter(
                torch.randn(1, 1, self.dec_embed_dim) * 0.02, requires_grad=True
            ) # [1, 1, 768]
            self.pose_retriever = LocalMemory(
                size=config.local_mem_size,
                k_dim=self.enc_embed_dim,
                v_dim=self.dec_embed_dim,
                num_heads=self.dec_num_heads,
                mlp_ratio=4,
                qkv_bias=True,
                attn_drop=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                rope=None,
            )
        self.register_tokens = nn.Embedding(config.state_size, self.enc_embed_dim) # init state tokens [768, 1024]
        self.state_size = config.state_size
        self.state_pe = config.state_pe
        self.masked_img_token = nn.Parameter(
            torch.randn(1, self.enc_embed_dim) * 0.02, requires_grad=True
        )
        self.masked_ray_map_token = nn.Parameter(
            torch.randn(1, self.enc_embed_dim) * 0.02, requires_grad=True
        )
        self._set_state_decoder(
            self.enc_embed_dim,
            self.dec_embed_dim,
            config.state_dec_num_heads,
            self.dec_depth,
            self.croco_args.get("mlp_ratio", None),
            self.croco_args.get("norm_layer", None),
            self.croco_args.get("norm_im2_in_dec", None),
        )
        self.set_downstream_head(
            config.output_mode,
            config.head_type,
            config.landscape_only,
            config.depth_mode,
            config.conf_mode,
            config.pose_mode,
            config.depth_head,
            config.rgb_head,
            config.pose_conf_head,
            config.pose_head,
            **self.croco_args,
        )
        self.set_freeze(config.freeze)
        
        # Initialize Keyframe Memory Bank if enabled
        if config.use_keyframe_memory_bank:
            self.keyframe_memory_bank = KeyframeMemoryBank(
                max_size=config.keyframe_memory_max_size,
                device='cuda',  # Will be set properly during inference
                diversity_threshold=config.keyframe_memory_diversity_threshold,
                min_interval=getattr(config, 'keyframe_memory_min_interval', 10)
            )
        else:
            self.keyframe_memory_bank = None

        # Initialize Loop Closure components if enabled
        if config.enable_loop_closure and self.pose_head_flag:
            self.loop_closure_keyframe_db = LoopClosureKeyframeDB(
                max_keyframes=config.loop_closure_max_keyframes,
                feat_dim=self.enc_embed_dim,
                state_dim=self.dec_embed_dim,
                state_size=config.state_size,
                mem_size=config.local_mem_size,
                mem_dim=self.dec_embed_dim * 2,  # mem uses 2*v_dim
            )
            self.loop_detector = LoopDetector(
                feat_dim=self.enc_embed_dim,
                threshold=config.loop_closure_threshold,
                min_frame_gap=config.loop_closure_min_frame_gap,
            )
            self.loop_corrector = LoopCorrector(
                state_dim=self.dec_embed_dim,
                mem_dim=self.dec_embed_dim * 2,  # mem is 2*v_dim
                pose_dim=7,
            )
            # Store loop closure information for visualization
            self.loop_closures = []
        else:
            self.loop_closure_keyframe_db = None
            self.loop_detector = None
            self.loop_corrector = None
            self.loop_closures = []

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            model = load_model(pretrained_model_name_or_path, device="cpu")
            return model
        else:
            try:
                model = super(ARCroco3DStereo, cls).from_pretrained(
                    pretrained_model_name_or_path, **kw
                )
            except TypeError as e:
                raise Exception(
                    f"tried to load {pretrained_model_name_or_path} from huggingface, but failed"
                )
            return model

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = get_patch_embed(
            self.patch_embed_cls, img_size, patch_size, enc_embed_dim, in_chans=3
        )
        self.patch_embed_ray_map = get_patch_embed(
            self.patch_embed_cls, img_size, patch_size, enc_embed_dim, in_chans=6
        )

    def _set_decoder(
        self,
        enc_embed_dim,
        dec_embed_dim,
        dec_num_heads,
        dec_depth,
        mlp_ratio,
        norm_layer,
        norm_im2_in_dec,
    ):
        self.dec_depth = dec_depth
        self.dec_embed_dim = dec_embed_dim
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        self.dec_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    dec_embed_dim,
                    dec_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    norm_mem=norm_im2_in_dec,
                    rope=self.rope,
                )
                for i in range(dec_depth)
            ]
        )
        self.dec_norm = norm_layer(dec_embed_dim)

    def _set_state_decoder(
        self,
        enc_embed_dim,
        dec_embed_dim,
        dec_num_heads,
        dec_depth,
        mlp_ratio,
        norm_layer,
        norm_im2_in_dec,
    ):
        self.dec_depth_state = dec_depth
        self.dec_embed_dim_state = dec_embed_dim
        self.decoder_embed_state = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        self.dec_blocks_state = nn.ModuleList(
            [
                DecoderBlock(
                    dec_embed_dim,
                    dec_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    norm_mem=norm_im2_in_dec,
                    rope=self.rope,
                )
                for i in range(dec_depth)
            ]
        )
        self.dec_norm_state = norm_layer(dec_embed_dim)

    def load_state_dict(self, ckpt, **kw):
        if all(k.startswith("module") for k in ckpt):
            ckpt = strip_module(ckpt)
        new_ckpt = dict(ckpt)
        if not any(k.startswith("dec_blocks_state") for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith("dec_blocks"):
                    new_ckpt[key.replace("dec_blocks", "dec_blocks_state")] = value
        try:
            return super().load_state_dict(new_ckpt, **kw)
        except:
            try:
                new_new_ckpt = {
                    k: v
                    for k, v in new_ckpt.items()
                    if not k.startswith("dec_blocks")
                    and not k.startswith("dec_norm")
                    and not k.startswith("decoder_embed")
                }
                return super().load_state_dict(new_new_ckpt, **kw)
            except:
                new_new_ckpt = {}
                for key in new_ckpt:
                    if key in self.state_dict():
                        if new_ckpt[key].size() == self.state_dict()[key].size():
                            new_new_ckpt[key] = new_ckpt[key]
                        else:
                            printer.info(
                                f"Skipping '{key}': size mismatch (ckpt: {new_ckpt[key].size()}, model: {self.state_dict()[key].size()})"
                            )
                    else:
                        printer.info(f"Skipping '{key}': not found in model")
                return super().load_state_dict(new_new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze
        to_be_frozen = {
            "none": [],
            "mask": [self.mask_token] if hasattr(self, "mask_token") else [],
            "encoder": [
                self.patch_embed,
                self.patch_embed_ray_map,
                self.masked_img_token,
                self.masked_ray_map_token,
                self.enc_blocks,
                self.enc_blocks_ray_map,
                self.enc_norm,
                self.enc_norm_ray_map,
            ],
            "encoder_and_head": [
                self.patch_embed,
                self.patch_embed_ray_map,
                self.masked_img_token,
                self.masked_ray_map_token,
                self.enc_blocks,
                self.enc_blocks_ray_map,
                self.enc_norm,
                self.enc_norm_ray_map,
                self.downstream_head,
            ],
            "encoder_and_decoder": [
                self.patch_embed,
                self.patch_embed_ray_map,
                self.masked_img_token,
                self.masked_ray_map_token,
                self.enc_blocks,
                self.enc_blocks_ray_map,
                self.enc_norm,
                self.enc_norm_ray_map,
                self.dec_blocks,
                self.dec_blocks_state,
                self.pose_retriever,
                self.pose_token,
                self.register_tokens,
                self.decoder_embed_state,
                self.decoder_embed,
                self.dec_norm,
                self.dec_norm_state,
            ],
            "decoder": [
                self.dec_blocks,
                self.dec_blocks_state,
                self.pose_retriever,
                self.pose_token,
            ],
        }
        freeze_all_params(to_be_frozen[freeze])

    def _set_prediction_head(self, *args, **kwargs):
        """No prediction head"""
        return

    def set_downstream_head(
        self,
        output_mode,
        head_type,
        landscape_only,
        depth_mode,
        conf_mode,
        pose_mode,
        depth_head,
        rgb_head,
        pose_conf_head,
        pose_head,
        patch_size,
        img_size,
        **kw,
    ):
        assert (
            img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0
        ), f"{img_size=} must be multiple of {patch_size=}"
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        self.pose_mode = pose_mode
        self.downstream_head = head_factory(
            head_type,
            output_mode,
            self,
            has_conf=bool(conf_mode),
            has_depth=bool(depth_head),
            has_rgb=bool(rgb_head),
            has_pose_conf=bool(pose_conf_head),
            has_pose=bool(pose_head),
        )
        self.head = transpose_to_landscape(
            self.downstream_head, activate=landscape_only
        )

    def _encode_image(self, image, true_shape):
        x, pos = self.patch_embed(image, true_shape=true_shape)
        assert self.enc_pos_embed is None
        for blk in self.enc_blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(blk, x, pos, use_reentrant=False)
            else:
                x = blk(x, pos)
        x = self.enc_norm(x)
        return [x], pos, None

    def _encode_ray_map(self, ray_map, true_shape):
        x, pos = self.patch_embed_ray_map(ray_map, true_shape=true_shape)
        assert self.enc_pos_embed is None
        for blk in self.enc_blocks_ray_map:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(blk, x, pos, use_reentrant=False)
            else:
                x = blk(x, pos)
        x = self.enc_norm_ray_map(x)
        return [x], pos, None

    def _encode_state(self, image_tokens, image_pos):
        batch_size = image_tokens.shape[0]
        state_feat = self.register_tokens(
            torch.arange(self.state_size, device=image_pos.device)
        ) # [768, 1024]
        if self.state_pe == "1d":
            state_pos = (
                torch.tensor(
                    [[i, i] for i in range(self.state_size)],
                    dtype=image_pos.dtype,
                    device=image_pos.device,
                )[None]
                .expand(batch_size, -1, -1)
                .contiguous()
            )  # .long()
        elif self.state_pe == "2d":
            width = int(self.state_size**0.5)
            width = width + 1 if width % 2 == 1 else width
            state_pos = (
                torch.tensor(
                    [[i // width, i % width] for i in range(self.state_size)],
                    dtype=image_pos.dtype,
                    device=image_pos.device,
                )[None]
                .expand(batch_size, -1, -1)
                .contiguous()
            )
        elif self.state_pe == "none":
            state_pos = None
        state_feat = state_feat[None].expand(batch_size, -1, -1)
        return state_feat, state_pos, None

    def _encode_views(self, views, img_mask=None, ray_mask=None):
        device = views[0]["img"].device
        batch_size = views[0]["img"].shape[0]
        given = True
        if img_mask is None and ray_mask is None:
            given = False
        if not given:
            img_mask = torch.stack(
                [view["img_mask"] for view in views], dim=0
            )  # Shape: (num_views, batch_size)
            ray_mask = torch.stack(
                [view["ray_mask"] for view in views], dim=0
            )  # Shape: (num_views, batch_size)
        imgs = torch.stack(
            [view["img"] for view in views], dim=0
        )  # Shape: (num_views, batch_size, C, H, W)
        ray_maps = torch.stack(
            [view["ray_map"] for view in views], dim=0
        )  # Shape: (num_views, batch_size, H, W, C)
        shapes = []
        for view in views:
            if "true_shape" in view:
                shapes.append(view["true_shape"])
            else:
                shape = torch.tensor(view["img"].shape[-2:], device=device)
                shapes.append(shape.unsqueeze(0).repeat(batch_size, 1))
        shapes = torch.stack(shapes, dim=0).to(
            imgs.device
        )  # Shape: (num_views, batch_size, 2)
        imgs = imgs.view(
            -1, *imgs.shape[2:]
        )  # Shape: (num_views * batch_size, C, H, W)
        ray_maps = ray_maps.view(
            -1, *ray_maps.shape[2:]
        )  # Shape: (num_views * batch_size, H, W, C)
        shapes = shapes.view(-1, 2)  # Shape: (num_views * batch_size, 2)
        img_masks_flat = img_mask.view(-1)  # Shape: (num_views * batch_size)
        ray_masks_flat = ray_mask.view(-1)
        selected_imgs = imgs[img_masks_flat]
        selected_shapes = shapes[img_masks_flat]
        if selected_imgs.size(0) > 0:
            img_out, img_pos, _ = self._encode_image(selected_imgs, selected_shapes)
        else:
            raise NotImplementedError
        full_out = [
            torch.zeros(
                len(views) * batch_size, *img_out[0].shape[1:], device=img_out[0].device
            )
            for _ in range(len(img_out))
        ]
        full_pos = torch.zeros(
            len(views) * batch_size,
            *img_pos.shape[1:],
            device=img_pos.device,
            dtype=img_pos.dtype,
        )
        for i in range(len(img_out)):
            full_out[i][img_masks_flat] += img_out[i]
            full_out[i][~img_masks_flat] += self.masked_img_token
        full_pos[img_masks_flat] += img_pos
        ray_maps = ray_maps.permute(0, 3, 1, 2)  # Change shape to (N, C, H, W)
        selected_ray_maps = ray_maps[ray_masks_flat]
        selected_shapes_ray = shapes[ray_masks_flat]
        if selected_ray_maps.size(0) > 0:
            ray_out, ray_pos, _ = self._encode_ray_map(
                selected_ray_maps, selected_shapes_ray
            )
            assert len(ray_out) == len(full_out), f"{len(ray_out)}, {len(full_out)}"
            for i in range(len(ray_out)):
                full_out[i][ray_masks_flat] += ray_out[i]
                full_out[i][~ray_masks_flat] += self.masked_ray_map_token
            full_pos[ray_masks_flat] += (
                ray_pos * (~img_masks_flat[ray_masks_flat][:, None, None]).long()
            )
        else:
            raymaps = torch.zeros(
                1, 6, imgs[0].shape[-2], imgs[0].shape[-1], device=img_out[0].device
            )
            ray_mask_flat = torch.zeros_like(img_masks_flat)
            ray_mask_flat[:1] = True
            ray_out, ray_pos, _ = self._encode_ray_map(raymaps, shapes[ray_mask_flat])
            for i in range(len(ray_out)):
                full_out[i][ray_mask_flat] += ray_out[i] * 0.0
                full_out[i][~ray_mask_flat] += self.masked_ray_map_token * 0.0
        return (
            shapes.chunk(len(views), dim=0),
            [out.chunk(len(views), dim=0) for out in full_out],
            full_pos.chunk(len(views), dim=0),
        )

    def _decoder(self, f_state, pos_state, f_img, pos_img, f_pose, pos_pose, return_attn):
        final_output = [(f_state, f_img)]  # before projection
        assert f_state.shape[-1] == self.dec_embed_dim
        f_img = self.decoder_embed(f_img) # Linear: [1, 576, 1024] -> [1, 576, 768]
        if self.pose_head_flag:
            assert f_pose is not None and pos_pose is not None
            f_img = torch.cat([f_pose, f_img], dim=1) # [1, 1 + 576, 768]
            pos_img = torch.cat([pos_pose, pos_img], dim=1) # [1, 1 + 576, 2]
        final_output.append((f_state, f_img))
        attention_maps = []
        for blk_state, blk_img in zip(self.dec_blocks_state, self.dec_blocks):
            if (
                self.gradient_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                f_state, _, self_attn_state, cross_attn_state = checkpoint(
                    blk_state,
                    *final_output[-1][::+1],
                    pos_state,
                    pos_img,
                    return_attn,
                    use_reentrant=not self.fixed_input_length,
                )
                f_img, _, self_attn_img, cross_attn_img = checkpoint(
                    blk_img,
                    *final_output[-1][::-1],
                    pos_img,
                    pos_state,
                    return_attn,
                    use_reentrant=not self.fixed_input_length,
                )
            else:
                f_state, _, self_attn_state, cross_attn_state = blk_state(*final_output[-1][::+1], pos_state, pos_img, return_attn=return_attn)
                f_img, _, self_attn_img, cross_attn_img = blk_img(*final_output[-1][::-1], pos_img, pos_state, return_attn=return_attn)
            final_output.append((f_state, f_img))
            attention_maps.append((self_attn_state, cross_attn_state, self_attn_img, cross_attn_img))
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = (
            self.dec_norm_state(final_output[-1][0]),
            self.dec_norm(final_output[-1][1]),
        )
        return zip(*final_output), zip(*attention_maps)

    def _downstream_head(self, decout, img_shape, **kwargs):
        B, S, D = decout[-1].shape
        head = getattr(self, f"head")
        return head(decout, img_shape, **kwargs)

    def _init_state(self, image_tokens, image_pos):
        """
        Current Version: input the first frame img feature and pose to initialize the state feature and pose
        # [1, 768, 768] [1, 768, 2]
        """
        state_feat, state_pos, _ = self._encode_state(image_tokens, image_pos)
        state_feat = self.decoder_embed_state(state_feat) # Linear: [1, 768, 1024] -> [1, 768, 768]
        return state_feat, state_pos

    def _recurrent_rollout(
        self,
        state_feat,
        state_pos,
        current_feat,
        current_pos,
        pose_feat,
        pose_pos,
        init_state_feat,
        img_mask=None,
        reset_mask=None,
        update=None,
        return_attn=False,
    ):
        (new_state_feat, dec), (self_attn_state, cross_attn_state, self_attn_img, cross_attn_img) = self._decoder(
            state_feat, state_pos, current_feat, current_pos, pose_feat, pose_pos, return_attn
        )
        new_state_feat = new_state_feat[-1]
        return new_state_feat, dec, self_attn_state, cross_attn_state, self_attn_img, cross_attn_img

    def _get_img_level_feat(self, feat):
        return torch.mean(feat, dim=1, keepdim=True)

    # tbptt training encoder: Truncated Backpropagation Through Time
    def _forward_encoder(self, views):
        shape, feat_ls, pos = self._encode_views(views)
        feat = feat_ls[-1]
        state_feat, state_pos = self._init_state(feat[0], pos[0])
        mem = self.pose_retriever.mem.expand(feat[0].shape[0], -1, -1)
        init_state_feat = state_feat.clone()
        init_mem = mem.clone()
        return (feat, pos, shape), (
            init_state_feat,
            init_mem,
            state_feat,
            state_pos,
            mem,
        )

    # tbptt training decoder step: Truncated Backpropagation Through Time
    def _forward_decoder_step(
        self,
        views,
        i,
        feat_i,
        pos_i,
        shape_i,
        init_state_feat,
        init_mem,
        state_feat,
        state_pos,
        mem,
    ):
        if self.pose_head_flag:
            global_img_feat_i = self._get_img_level_feat(feat_i)
            if i == 0:
                pose_feat_i = self.pose_token.expand(feat_i.shape[0], -1, -1)
            else:
                pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem)
            pose_pos_i = -torch.ones(
                feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
            )
        else:
            pose_feat_i = None
            pose_pos_i = None
        new_state_feat, dec, self_attn_state, cross_attn_state, self_attn_img, cross_attn_img = self._recurrent_rollout(
            state_feat,
            state_pos,
            feat_i,
            pos_i,
            pose_feat_i,
            pose_pos_i,
            init_state_feat,
            img_mask=views[i]["img_mask"],
            reset_mask=views[i]["reset"],
            update=views[i].get("update", None),
            return_attn=False,
        )
        out_pose_feat_i = dec[-1][:, 0:1]
        new_mem = self.pose_retriever.update_mem(
            mem, global_img_feat_i, out_pose_feat_i
        )
        head_input = [
            dec[0].float(),
            dec[self.dec_depth * 2 // 4][:, 1:].float(),
            dec[self.dec_depth * 3 // 4][:, 1:].float(),
            dec[self.dec_depth].float(),
        ]
        res = self._downstream_head(head_input, shape_i, pos=pos_i)
        img_mask = views[i]["img_mask"]
        update = views[i].get("update", None)
        if update is not None:
            update_mask = img_mask & update  # if don't update, then whatever img_mask
        else:
            update_mask = img_mask
        update_mask = update_mask[:, None, None].float()
        state_feat = new_state_feat * update_mask + state_feat * (
            1 - update_mask
        )  # update global state
        mem = new_mem * update_mask + mem * (1 - update_mask)  # then update local state
        reset_mask = views[i]["reset"]
        if reset_mask is not None:
            reset_mask = reset_mask[:, None, None].float()
            state_feat = init_state_feat * reset_mask + state_feat * (1 - reset_mask)
            mem = init_mem * reset_mask + mem * (1 - reset_mask)
        return res, (state_feat, mem)

    # training and testing
    def _forward_impl(self, views, ret_state=False):
        # [B, C, H, W] -> [B, H/16*W/16, 1024]
        shape, feat_ls, pos = self._encode_views(views) # [15, 3, 288, 512] -> feat [15, 576, 1024], pos [15, 576, 2]
        feat = feat_ls[-1]
        state_feat, state_pos = self._init_state(feat[0], pos[0]) # init state feat [1, 768, 768], state_pos [1, 768, 2]
        mem = self.pose_retriever.mem.expand(feat[0].shape[0], -1, -1) # [1, 256, 1536] init pose mem
        init_state_feat = state_feat.clone()
        init_mem = mem.clone()
        all_state_args = [(state_feat, state_pos, init_state_feat, mem, init_mem)]
        ress = []
        for i in range(len(views)):
            feat_i = feat[i]
            pos_i = pos[i]
            if self.pose_head_flag:
                global_img_feat_i = self._get_img_level_feat(feat_i) # avg pool: [1, 576, 1024] -> [1, 1, 1024]
                if i == 0:
                    pose_feat_i = self.pose_token.expand(feat_i.shape[0], -1, -1) # [1, 1, 768] init pose token
                else:
                    pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem) 
                    # [1, 1, 768] use [global_img_feat_i, masked_token(pose)] as query, cross-attend mem, get pose_feat_i
                pose_pos_i = -torch.ones(
                    feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
                ) # [1, 1, 2]
            else:
                pose_feat_i = None
                pose_pos_i = None
            new_state_feat, dec, self_attn_state, cross_attn_state, self_attn_img, cross_attn_img = self._recurrent_rollout(
                state_feat, # [1, 768, 768]
                state_pos, # [1, 768, 2]
                feat_i, # [1, 576, 1024]
                pos_i, # [1, 576, 2]
                pose_feat_i, # [1, 1, 768] coarse pose token from pose_retriever
                pose_pos_i, # [1, 1, 2]
                init_state_feat,
                img_mask=views[i]["img_mask"],
                reset_mask=views[i]["reset"],
                update=views[i].get("update", None),
                return_attn=True,
            ) # [1, 768, 768]
            out_pose_feat_i = dec[-1][:, 0:1] # [1, 1, 768] refined pose token from dust3r
            new_mem = self.pose_retriever.update_mem(
                mem, global_img_feat_i, out_pose_feat_i
            ) # [1, 256, 1536] use mem as query, cross-attend [global_img_feat_i, out_pose_feat_i], get new_mem
            assert len(dec) == self.dec_depth + 1
            head_input = [
                dec[0].float(), # [1, 576, 1024]
                dec[self.dec_depth * 2 // 4][:, 1:].float(), # [1, 576, 768]
                dec[self.dec_depth * 3 // 4][:, 1:].float(), # [1, 576, 768]
                dec[self.dec_depth].float(), # [1, 1 + 576, 768]
            ]
            res = self._downstream_head(head_input, shape[i], pos=pos_i)
            ress.append(res)
            img_mask = views[i]["img_mask"]
            update = views[i].get("update", None)
            if update is not None:
                update_mask = (
                    img_mask & update
                )  # if don't update, then whatever img_mask
            else:
                update_mask = img_mask
            update_mask = update_mask[:, None, None].float()

            # update with learning rate
            if i  == 0:
                update_mask1 = update_mask
            else:
                if self.config.model_update_type == "cut3r":
                    update_mask1 = update_mask
                elif self.config.model_update_type == "ttt3r":
                    cross_attn_state = rearrange(torch.cat(cross_attn_state, dim=0), 'l h nstate nimg -> 1 nstate nimg (l h)') # [12, 16, 768, 1 + 576] -> [1, 768, 1 + 576, 12*16]
                    state_query_img_key = cross_attn_state.mean(dim=(-1, -2))
                    update_mask1 = update_mask * torch.sigmoid(state_query_img_key)[..., None] * 1.0
                else:
                    raise ValueError(f"Invalid model type: {self.config.model_update_type}")

            update_mask2 = update_mask
            state_feat = new_state_feat * update_mask1 + state_feat * (
                1 - update_mask1
            )  # update global state
            mem = new_mem * update_mask2 + mem * (
                1 - update_mask2
            )  # then update local state
            reset_mask = views[i]["reset"]
            if reset_mask is not None:
                reset_mask = reset_mask[:, None, None].float()
                state_feat = init_state_feat * reset_mask + state_feat * (
                    1 - reset_mask
                )
                mem = init_mem * reset_mask + mem * (1 - reset_mask)
            all_state_args.append(
                (state_feat, state_pos, init_state_feat, mem, init_mem)
            )
        if ret_state:
            return ress, views, all_state_args
        return ress, views

    def forward(self, views, ret_state=False):
        if ret_state:
            ress, views, state_args = self._forward_impl(views, ret_state=ret_state)
            return ARCroco3DStereoOutput(ress=ress, views=views), state_args
        else:
            ress, views = self._forward_impl(views, ret_state=ret_state)
            return ARCroco3DStereoOutput(ress=ress, views=views)

    # testing: generate rgb xyz condition on raymap
    def inference_step(
        self, view, state_feat, state_pos, init_state_feat, mem, init_mem
    ):
        batch_size = view["img"].shape[0]
        raymaps = []
        shapes = []
        for j in range(batch_size):
            assert view["ray_mask"][j]
            raymap = view["ray_map"][[j]].permute(0, 3, 1, 2)
            raymaps.append(raymap)
            shapes.append(
                view.get(
                    "true_shape",
                    torch.tensor(view["ray_map"].shape[-2:])[None].repeat(
                        view["ray_map"].shape[0], 1
                    ),
                )[[j]]
            )

        raymaps = torch.cat(raymaps, dim=0)
        shape = torch.cat(shapes, dim=0).to(raymaps.device)
        feat_ls, pos, _ = self._encode_ray_map(raymaps, shapes) # [1, 6, 384, 512] -> feat [1, 768, 1024], pos [1, 768, 2]

        feat_i = feat_ls[-1]
        pos_i = pos
        if self.pose_head_flag:
            global_img_feat_i = self._get_img_level_feat(feat_i)
            pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem)
            pose_pos_i = -torch.ones(
                feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
            )
        else:
            pose_feat_i = None
            pose_pos_i = None
        new_state_feat, dec, self_attn_state, cross_attn_state, self_attn_img, cross_attn_img = self._recurrent_rollout(
            state_feat,
            state_pos,
            feat_i,
            pos_i,
            pose_feat_i,
            pose_pos_i,
            init_state_feat,
            img_mask=view["img_mask"],
            reset_mask=view["reset"],
            update=view.get("update", None),
            return_attn=False,
        )

        out_pose_feat_i = dec[-1][:, 0:1]
        new_mem = self.pose_retriever.update_mem(
            mem, global_img_feat_i, out_pose_feat_i
        )
        assert len(dec) == self.dec_depth + 1
        head_input = [
            dec[0].float(),
            dec[self.dec_depth * 2 // 4][:, 1:].float(),
            dec[self.dec_depth * 3 // 4][:, 1:].float(),
            dec[self.dec_depth].float(),
        ]
        res = self._downstream_head(head_input, shape, pos=pos_i)
        return res, view

    # recurrent testing
    def forward_recurrent(self, views, device, ret_state=False):
        ress = []
        all_state_args = []
        for i, view in enumerate(views):
            device = view["img"].device
            batch_size = view["img"].shape[0]
            img_mask = view["img_mask"].reshape(
                -1, batch_size
            )  # Shape: (1, batch_size)
            ray_mask = view["ray_mask"].reshape(
                -1, batch_size
            )  # Shape: (1, batch_size)
            imgs = view["img"].unsqueeze(0)  # Shape: (1, batch_size, C, H, W)
            ray_maps = view["ray_map"].unsqueeze(
                0
            )  # Shape: (num_views, batch_size, H, W, C)
            shapes = (
                view["true_shape"].unsqueeze(0)
                if "true_shape" in view
                else torch.tensor(view["img"].shape[-2:], device=device)
                .unsqueeze(0)
                .repeat(batch_size, 1)
                .unsqueeze(0)
            )  # Shape: (num_views, batch_size, 2)
            imgs = imgs.view(
                -1, *imgs.shape[2:]
            )  # Shape: (num_views * batch_size, C, H, W)
            ray_maps = ray_maps.view(
                -1, *ray_maps.shape[2:]
            )  # Shape: (num_views * batch_size, H, W, C)
            shapes = shapes.view(-1, 2).to(
                imgs.device
            )  # Shape: (num_views * batch_size, 2)
            img_masks_flat = img_mask.view(-1)  # Shape: (num_views * batch_size)
            ray_masks_flat = ray_mask.view(-1)
            selected_imgs = imgs[img_masks_flat]
            selected_shapes = shapes[img_masks_flat]
            if selected_imgs.size(0) > 0:
                img_out, img_pos, _ = self._encode_image(selected_imgs, selected_shapes)
            else:
                img_out, img_pos = None, None
            ray_maps = ray_maps.permute(0, 3, 1, 2)  # Change shape to (N, C, H, W)
            selected_ray_maps = ray_maps[ray_masks_flat]
            selected_shapes_ray = shapes[ray_masks_flat]
            if selected_ray_maps.size(0) > 0:
                ray_out, ray_pos, _ = self._encode_ray_map(
                    selected_ray_maps, selected_shapes_ray
                )
            else:
                ray_out, ray_pos = None, None

            shape = shapes
            if img_out is not None and ray_out is None:
                feat_i = img_out[-1]
                pos_i = img_pos
            elif img_out is None and ray_out is not None:
                feat_i = ray_out[-1]
                pos_i = ray_pos
            elif img_out is not None and ray_out is not None:
                feat_i = img_out[-1] + ray_out[-1]
                pos_i = img_pos
            else:
                raise NotImplementedError

            if i == 0:
                state_feat, state_pos = self._init_state(feat_i, pos_i)
                mem = self.pose_retriever.mem.expand(feat_i.shape[0], -1, -1)
                init_state_feat = state_feat.clone()
                init_mem = mem.clone()
                all_state_args.append(
                    (state_feat, state_pos, init_state_feat, mem, init_mem)
                )

            if self.pose_head_flag:
                global_img_feat_i = self._get_img_level_feat(feat_i)
                if i == 0:
                    pose_feat_i = self.pose_token.expand(feat_i.shape[0], -1, -1)
                else:
                    pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem)
                pose_pos_i = -torch.ones(
                    feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
                )
            else:
                pose_feat_i = None
                pose_pos_i = None
            new_state_feat, dec, self_attn_state, cross_attn_state, self_attn_img, cross_attn_img = self._recurrent_rollout(
                state_feat,
                state_pos,
                feat_i,
                pos_i,
                pose_feat_i,
                pose_pos_i,
                init_state_feat,
                img_mask=view["img_mask"],
                reset_mask=view["reset"],
                update=view.get("update", None),
                return_attn=False,
            )
            out_pose_feat_i = dec[-1][:, 0:1]
            new_mem = self.pose_retriever.update_mem(
                mem, global_img_feat_i, out_pose_feat_i
            )
            assert len(dec) == self.dec_depth + 1
            head_input = [
                dec[0].float(),
                dec[self.dec_depth * 2 // 4][:, 1:].float(),
                dec[self.dec_depth * 3 // 4][:, 1:].float(),
                dec[self.dec_depth].float(),
            ]
            res = self._downstream_head(head_input, shape, pos=pos_i)
            ress.append(res)
            img_mask = view["img_mask"]
            update = view.get("update", None)
            if update is not None:
                update_mask = (
                    img_mask & update
                )  # if don't update, then whatever img_mask
            else:
                update_mask = img_mask
            update_mask = update_mask[:, None, None].float()
            state_feat = new_state_feat * update_mask + state_feat * (
                1 - update_mask
            )  # update global state
            mem = new_mem * update_mask + mem * (
                1 - update_mask
            )  # then update local state
            reset_mask = view["reset"]
            if reset_mask is not None:
                reset_mask = reset_mask[:, None, None].float()
                state_feat = init_state_feat * reset_mask + state_feat * (
                    1 - reset_mask
                )
                mem = init_mem * reset_mask + mem * (1 - reset_mask)
            all_state_args.append(
                (state_feat, state_pos, init_state_feat, mem, init_mem)
            )
        if ret_state:
            return ress, views, all_state_args
        return ress, views

    def forward_recurrent_lighter(self, views, device='cuda', ret_state=False):
        ress = []
        all_state_args = []
        prev_reset_mask_bool = False  # Track previous frame's reset for pose_feat initialization
        for i, _view in enumerate(views):
            view = to_gpu(_view, device)
            device = view["img"].device
            batch_size = view["img"].shape[0]
            img_mask = view["img_mask"].reshape(
                -1, batch_size
            )  # Shape: (1, batch_size)
            ray_mask = view["ray_mask"].reshape(
                -1, batch_size
            )  # Shape: (1, batch_size)
            imgs = view["img"].unsqueeze(0)  # Shape: (1, batch_size, C, H, W)
            ray_maps = view["ray_map"].unsqueeze(
                0
            )  # Shape: (num_views, batch_size, H, W, C)
            shapes = (
                view["true_shape"].unsqueeze(0)
                if "true_shape" in view
                else torch.tensor(view["img"].shape[-2:], device=device)
                .unsqueeze(0)
                .repeat(batch_size, 1)
                .unsqueeze(0)
            )  # Shape: (num_views, batch_size, 2)
            imgs = imgs.view(
                -1, *imgs.shape[2:]
            )  # Shape: (num_views * batch_size, C, H, W)
            ray_maps = ray_maps.view(
                -1, *ray_maps.shape[2:]
            )  # Shape: (num_views * batch_size, H, W, C)
            shapes = shapes.view(-1, 2).to(
                imgs.device
            )  # Shape: (num_views * batch_size, 2)
            img_masks_flat = img_mask.view(-1)  # Shape: (num_views * batch_size)
            ray_masks_flat = ray_mask.view(-1)
            selected_imgs = imgs[img_masks_flat]
            selected_shapes = shapes[img_masks_flat]
            if selected_imgs.size(0) > 0:
                img_out, img_pos, _ = self._encode_image(selected_imgs, selected_shapes)
            else:
                img_out, img_pos = None, None
            ray_maps = ray_maps.permute(0, 3, 1, 2)  # Change shape to (N, C, H, W)
            selected_ray_maps = ray_maps[ray_masks_flat]
            selected_shapes_ray = shapes[ray_masks_flat]
            if selected_ray_maps.size(0) > 0:
                ray_out, ray_pos, _ = self._encode_ray_map(
                    selected_ray_maps, selected_shapes_ray
                )
            else:
                ray_out, ray_pos = None, None

            shape = shapes
            if img_out is not None and ray_out is None:
                feat_i = img_out[-1]
                pos_i = img_pos
            elif img_out is None and ray_out is not None:
                feat_i = ray_out[-1]
                pos_i = ray_pos
            elif img_out is not None and ray_out is not None:
                feat_i = img_out[-1] + ray_out[-1]
                pos_i = img_pos
            else:
                raise NotImplementedError

            reset_mask = view.get("reset", None)
            if reset_mask is not None:
                if isinstance(reset_mask, torch.Tensor):
                    reset_mask_bool = reset_mask.item() if reset_mask.numel() == 1 else reset_mask.any().item()
                else:
                    reset_mask_bool = bool(reset_mask)
            else:
                reset_mask_bool = False

            # Check if loop closure is enabled
            _enable_lc = getattr(self.config, 'enable_loop_closure', False)

            if i == 0:
                state_feat, state_pos = self._init_state(feat_i, pos_i)
                mem = self.pose_retriever.mem.expand(feat_i.shape[0], -1, -1)
                init_state_feat = state_feat.clone()
                init_mem = mem.clone()
                # Initialize Keyframe Memory Bank device (but don't clear - preserve long-term memory)
                if self.keyframe_memory_bank is not None:
                    self.keyframe_memory_bank.device = device
                # Reset loop closure keyframe DB for new sequence
                if _enable_lc and self.loop_closure_keyframe_db is not None:
                    self.loop_closure_keyframe_db.reset(feat_i.shape[0], device=feat_i.device)
                    self.loop_closures = []  # Clear previous loop closures

            # NOTE: Keyframe Memory Bank is NOT cleared on reset - it serves as long-term memory
            # that persists across segments to help with re-localization

            if self.pose_head_flag:
                global_img_feat_i = self._get_img_level_feat(feat_i)

                # Retrieve keyframes from long-term memory if available
                keyframe_features = None
                if self.keyframe_memory_bank is not None and self.keyframe_memory_bank.size > 0:
                    proj_query_feat = self.pose_retriever.proj_q(global_img_feat_i)
                    keyframe_features, _ = self.keyframe_memory_bank.retrieve_top_k(
                        query_feat=proj_query_feat,
                        query_time=i,
                        k=self.config.keyframe_memory_top_k,
                        lambda_time=self.config.keyframe_memory_lambda_time
                    )
                    if i % 50 == 0:
                        print(f"[KMB Debug] Frame {i}: retrieved {keyframe_features.shape[1]} keyframes from bank size {self.keyframe_memory_bank.size}")

                if i == 0 or prev_reset_mask_bool:
                    # After reset: use keyframe memory to initialize pose_feat if available
                    if keyframe_features is not None:
                        # Use keyframe features to provide initial pose context
                        # inquire_with_keyframes uses init_mem since local mem was just reset
                        pose_feat_i = self.pose_retriever.inquire_with_keyframes(
                            global_img_feat_i, mem, keyframe_features
                        )
                        if prev_reset_mask_bool:
                            print(f"[KMB] Frame {i}: Reset - initialized pose_feat from {keyframe_features.shape[1]} keyframes")
                    else:
                        # No keyframe memory available, fall back to pose_token
                        pose_feat_i = self.pose_token.expand(feat_i.shape[0], -1, -1)
                else:
                    # Normal frame: use both local memory and keyframe memory
                    if keyframe_features is not None:
                        pose_feat_i = self.pose_retriever.inquire_with_keyframes(
                            global_img_feat_i, mem, keyframe_features
                        )
                    else:
                        pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem)
                pose_pos_i = -torch.ones(
                    feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
                )
            else:
                pose_feat_i = None
                pose_pos_i = None
            new_state_feat, dec, self_attn_state, cross_attn_state, self_attn_img, cross_attn_img = self._recurrent_rollout(
                state_feat,
                state_pos,
                feat_i,
                pos_i,
                pose_feat_i,
                pose_pos_i,
                init_state_feat,
                img_mask=view["img_mask"],
                reset_mask=view["reset"],
                update=view.get("update", None),
                return_attn=True,
            )
            out_pose_feat_i = dec[-1][:, 0:1]

            # Add current frame to Keyframe Memory Bank if enabled
            # Project global_feat to match memory format (proj_q projects to v_dim)
            # Use diversity-based sampling: only add if sufficiently different from existing keyframes
            if self.keyframe_memory_bank is not None and self.pose_head_flag:
                proj_global_feat = self.pose_retriever.proj_q(global_img_feat_i)
                # First frame is always added (force_add=True), others use diversity check
                added = self.keyframe_memory_bank.add(
                    frame_idx=i,
                    global_feat=proj_global_feat,
                    pose_feat=out_pose_feat_i,
                    force_add=(i == 0)  # Always add first frame as anchor
                )
                if added and i % 50 == 0:
                    print(f"[KMB] Frame {i}: Added to memory bank (size={self.keyframe_memory_bank.size})")

            # update mem
            new_mem = self.pose_retriever.update_mem(
                mem, global_img_feat_i, out_pose_feat_i
            )

            assert len(dec) == self.dec_depth + 1
            head_input = [
                dec[0].float(),
                dec[self.dec_depth * 2 // 4][:, 1:].float(),
                dec[self.dec_depth * 3 // 4][:, 1:].float(),
                dec[self.dec_depth].float(),
            ]
            res = self._downstream_head(head_input, shape, pos=pos_i)

            # Loop closure detection and memory recall (inference only)
            if _enable_lc and self.pose_head_flag and i > 0 and self.loop_closure_keyframe_db is not None:
                current_pose_enc = res.get("camera_pose", None)
                if current_pose_enc is not None:
                    # Convert pose encoding to 7-dim pose (tx, ty, tz, qw, qx, qy, qz)
                    from src.dust3r.utils.camera import pose_encoding_to_camera
                    current_pose_mat = pose_encoding_to_camera(current_pose_enc.clone())  # [B, 4, 4]
                    # Extract translation and rotation (as quaternion)
                    trans = current_pose_mat[:, :3, 3]  # [B, 3]
                    rot_mat = current_pose_mat[:, :3, :3]  # [B, 3, 3]
                    # Convert rotation matrix to quaternion
                    import roma
                    quat = roma.rotmat_to_unitquat(rot_mat)  # [B, 4] (x, y, z, w)
                    # Reorder to (w, x, y, z) for consistency
                    quat = torch.cat([quat[:, 3:4], quat[:, :3]], dim=-1)  # [B, 4]
                    current_pose_7d = torch.cat([trans, quat], dim=-1)  # [B, 7]

                    # Detect loop closure with memory recall
                    _threshold = getattr(self.config, 'loop_closure_threshold', 0.85)
                    _min_gap = getattr(self.config, 'loop_closure_min_frame_gap', 30)
                    loop_detected, hist_state_feat, hist_mem, confidence, matched_frame_id = \
                        self.loop_closure_keyframe_db.query_with_memory(
                            global_img_feat_i, i, min_frame_gap=_min_gap, threshold=_threshold
                        )

                    # Apply memory recall if loop is detected
                    if loop_detected.any() and hist_state_feat is not None and hist_mem is not None:
                        # Memory fusion: blend current state with historical state
                        # Higher confidence = more weight to historical memory
                        blend_weight = confidence * loop_detected.float()  # [B]
                        blend_weight = blend_weight[:, None, None]  # [B, 1, 1]

                        # Fuse state_feat: current * (1-w) + historical * w
                        new_state_feat = new_state_feat * (1 - blend_weight * 0.3) + hist_state_feat * (blend_weight * 0.3)
                        # Fuse mem: current * (1-w) + historical * w
                        new_mem = new_mem * (1 - blend_weight * 0.3) + hist_mem * (blend_weight * 0.3)

                        # Store loop closure information for visualization
                        conf_val = confidence.item() if isinstance(confidence, torch.Tensor) else confidence
                        frame_id_val = matched_frame_id.item() if isinstance(matched_frame_id, torch.Tensor) else matched_frame_id

                        self.loop_closures.append({
                            'current_idx': i,
                            'matched_idx': int(frame_id_val),
                            'confidence': float(conf_val),
                        })
                        print(f"[Loop Closure + Memory Recall] Frame {i}: Recalled memory from frame {int(frame_id_val)} (confidence={conf_val:.3f}, blend={conf_val*0.3:.3f})")

                    # Add keyframe at fixed intervals (with memory state)
                    _kf_interval = getattr(self.config, 'loop_closure_keyframe_interval', 10)
                    if i % _kf_interval == 0:
                        self.loop_closure_keyframe_db.add_keyframe(
                            global_img_feat_i,
                            current_pose_7d,
                            i,
                            state_feat=new_state_feat,  # Save current state
                            mem=new_mem,                 # Save current memory
                        )

            res_cpu = to_cpu(res)
            ress.append(res_cpu)
            img_mask = view["img_mask"]
            update = view.get("update", None)
            if update is not None:
                update_mask = (
                    img_mask & update
                )  # if don't update, then whatever img_mask
            else:
                update_mask = img_mask
            update_mask = update_mask[:, None, None].float()

            # update with learning rate
            if i == 0 or prev_reset_mask_bool:
                update_mask1 = update_mask
            else:
                if self.config.model_update_type == "cut3r":
                    update_mask1 = update_mask
                elif self.config.model_update_type == "ttt3r":
                    cross_attn_state = rearrange(torch.cat(cross_attn_state, dim=0), 'l h nstate nimg -> 1 nstate nimg (l h)') # [12, 16, 768, 1 + 576] -> [1, 768, 1 + 576, 12*16]
                    state_query_img_key = cross_attn_state.mean(dim=(-1, -2))
                    update_mask1 = update_mask * torch.sigmoid(state_query_img_key)[..., None] * 1.0
                else:
                    raise ValueError(f"Invalid model type: {self.config.model_update_type}")

            update_mask2 = update_mask
            state_feat = new_state_feat * update_mask1 + state_feat * (
                1 - update_mask1
            )  # update global state
            mem = new_mem * update_mask2 + mem * (
                1 - update_mask2
            )  # then update local state

            reset_mask_tensor = view.get("reset", None)
            if reset_mask_tensor is not None:
                reset_mask_tensor = reset_mask_tensor[:, None, None].float()
                state_feat = init_state_feat * reset_mask_tensor + state_feat * (
                    1 - reset_mask_tensor
                )
                mem = init_mem * reset_mask_tensor + mem * (1 - reset_mask_tensor)

            # Update prev_reset_mask_bool for next iteration (original TTT3R behavior)
            prev_reset_mask_bool = reset_mask_bool

        if ret_state:
            return ress, views, all_state_args
        return ress, views

if __name__ == "__main__":
    print(ARCroco3DStereo.mro())
    cfg = ARCroco3DStereoConfig(
        state_size=256,
        pos_embed="RoPE100",
        rgb_head=True,
        pose_head=True,
        img_size=(224, 224),
        head_type="linear",
        output_mode="pts3d+pose",
        depth_mode=("exp", -inf, inf),
        conf_mode=("exp", 1, inf),
        pose_mode=("exp", -inf, inf),
        enc_embed_dim=1024,
        enc_depth=24,
        enc_num_heads=16,
        dec_embed_dim=768,
        dec_depth=12,
        dec_num_heads=12,
    )
    ARCroco3DStereo(cfg)
