#!/usr/bin/env python3
"""
Neural SPAI Generator
Learns to generate a sparse matrix G such that A^-1 ≈ G * G^T + εI.
Architecture: 1D Windowed Transformer.
Loss: Unsupervised Scale-Invariant Aligned Identity (SAI) Loss via Stochastic Trace Estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import argparse
import struct
import time

# --- 1. Physics Helper: Stencil Computation ---

def compute_laplacian_stencil(neighbors, layers, num_nodes):
    """
    Computes the explicit stencil weights (Matrix A) for the negative Laplacian.
    This is used to construct the 'Physics' input features and for the Loss function.
    Returns: A_diag [N, 1], A_off [N, 24]
    """
    device = neighbors.device
    N = num_nodes
    IDX_AIR = N
    IDX_WALL = N + 1
    
    # Expand inputs for broadcasting
    # layers: [N, 1] -> 2^layer gives cell size dx
    dx = torch.pow(2.0, layers.float()).view(N, 1, 1)
    
    # Safe neighbor lookup (clamp invalid indices to 0 for gathering)
    safe_nbrs = neighbors.long().clone()
    safe_nbrs[safe_nbrs >= N] = 0
    
    # Get neighbor layers
    layers_flat = layers.view(-1)
    nbr_layers = layers_flat[safe_nbrs] # [N, 24]
    dx_nbrs = torch.pow(2.0, nbr_layers.float()).view(N, 6, 4)
    
    # Reshape neighbors to [N, 6 faces, 4 sub-neighbors]
    raw_nbrs = neighbors.view(N, 6, 4)
    is_wall = (raw_nbrs == IDX_WALL)
    is_air  = (raw_nbrs == IDX_AIR)
    is_fluid = (raw_nbrs < N)
    
    # --- Logic for Mixed-Resolution Interfaces ---
    # We need to distinguish between 'Coarse/Same' neighbors (Slot 0) and 'Fine' neighbors (Slots 0-3)
    
    # Slot 0 Logic: Active if neighbor is fluid AND (Same Layer OR Coarser Layer)
    self_layer_exp = layers.view(N, 1, 1).expand(N, 6, 4)
    is_coarse_or_same = is_fluid & (nbr_layers.view(N, 6, 4) >= self_layer_exp)
    slot0_active = is_coarse_or_same[:, :, 0] # Boolean [N, 6]
    
    # Calculate Distances & Weights
    # Case 1: Coarse/Same (Distance between centers)
    dist_c = torch.maximum(0.5 * (dx + dx_nbrs), torch.tensor(1e-6, device=device))
    weight_c = (dx**2) / dist_c
    
    # Case 2: Fine/Air (Distance to interface or fine neighbor center)
    # For Dirichlet (Air), distance is effectively dx/2
    dx_target = dx_nbrs.clone()
    dx_target[is_air] = dx.expand(N, 6, 4)[is_air] * 0.5 
    
    dist_s = torch.maximum(0.5 * (dx + dx_target), torch.tensor(1e-6, device=device))
    weight_s = (dx**2 / 4.0) / dist_s # Area is 1/4th
    
    # Accumulate A (Off-Diagonals are negative)
    A_off = torch.zeros(N, 6, 4, device=device)
    
    # Apply Coarse/Same Weights (Only to Slot 0)
    mask_c = slot0_active.unsqueeze(2).expand(N, 6, 4).clone()
    mask_c[:, :, 1:] = False # Only valid for slot 0
    A_off[mask_c] = -weight_c[mask_c]
    
    # Apply Fine/Air Weights (Valid for all slots if not blocked by wall or coarse neighbor)
    mask_sub = (~slot0_active.unsqueeze(2).expand(N, 6, 4)) & (~is_wall)
    A_off[mask_sub] = -weight_s[mask_sub]
    
    # Flatten
    A_off_flat = A_off.view(N, 24)
    
    # Diagonal is sum of absolute off-diagonals (Laplacian property)
    # Includes weights to Air (Dirichlet)
    A_diag = torch.sum(torch.abs(A_off_flat), dim=1, keepdim=True)
    
    # Zero out flux to Air in the sparse matrix
    # (Because p_air is fixed at 0, it doesn't appear as a column in the system matrix)
    is_air_flat = (neighbors == IDX_AIR)
    A_off_flat[is_air_flat] = 0.0
    
    return A_diag, A_off_flat

# --- 2. Architecture: Hierarchical V-Cycle Transformer ---

# --- 2.1 RoPE (Rotary Positional Embeddings) ---

def apply_rope(x, freqs_cis):
    """
    Apply Rotary Position Embedding to input tensor.
    
    Args:
        x: [B, nhead, N, head_dim] query or key tensor
        freqs_cis: [N, head_dim // 2] complex frequencies (real and imag parts)
    
    Returns:
        x_rope: [B, nhead, N, head_dim] with RoPE applied
    """
    B, nhead, N, head_dim = x.shape
    
    # Split x into pairs: [B, nhead, N, head_dim//2, 2]
    x_reshaped = x.view(B, nhead, N, head_dim // 2, 2)
    
    # Extract real and imaginary parts of frequencies
    # freqs_cis is [N, head_dim//2, 2] where [:, :, 0] is cos, [:, :, 1] is sin
    if len(freqs_cis.shape) == 3:
        # Standard case: [N, head_dim//2, 2]
        cos_part = freqs_cis[:, :, 0]  # [N, head_dim//2]
        sin_part = freqs_cis[:, :, 1]  # [N, head_dim//2]
    else:
        # If it's a complex tensor (2D), convert to real representation
        cos_part = freqs_cis.real if hasattr(freqs_cis, 'real') else freqs_cis
        sin_part = freqs_cis.imag if hasattr(freqs_cis, 'imag') else torch.zeros_like(freqs_cis)
    
    # Expand for broadcasting: [1, 1, N, head_dim//2]
    cos_part = cos_part.unsqueeze(0).unsqueeze(0)  # [1, 1, N, head_dim//2]
    sin_part = sin_part.unsqueeze(0).unsqueeze(0)  # [1, 1, N, head_dim//2]
    
    # Apply rotation: [x0, x1] -> [x0*cos - x1*sin, x0*sin + x1*cos]
    x0, x1 = x_reshaped[..., 0], x_reshaped[..., 1]
    x_rotated = torch.stack([
        x0 * cos_part - x1 * sin_part,
        x0 * sin_part + x1 * cos_part
    ], dim=-1)  # [B, nhead, N, head_dim//2, 2]
    
    # Reshape back to [B, nhead, N, head_dim]
    return x_rotated.view(B, nhead, N, head_dim)


def precompute_freqs_cis(seq_len, head_dim, base=10000.0, device='cpu'):
    """
    Precompute frequency matrix for RoPE.
    
    Returns:
        freqs_cis: [seq_len, head_dim // 2, 2] where [:, :, 0] is cos, [:, :, 1] is sin
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)  # [seq_len, head_dim // 2]
    
    # Compute cos and sin
    cos_freqs = torch.cos(freqs)  # [seq_len, head_dim // 2]
    sin_freqs = torch.sin(freqs)  # [seq_len, head_dim // 2]
    
    # Stack: [seq_len, head_dim // 2, 2]
    freqs_cis = torch.stack([cos_freqs, sin_freqs], dim=-1)
    return freqs_cis


# --- 2.2 Scheduler: Calculate Depth and Padding Schedule ---

class HierarchicalScheduler:
    """
    Pre-computes the depth and padding schedule for deterministic V-cycle execution.
    """
    def __init__(self, k=32, min_leaf_size=128):
        """
        Args:
            k: Branching factor (compression rate per level)
            min_leaf_size: Minimum sequence length at bottom level
        """
        self.k = k
        self.min_leaf_size = min_leaf_size
    
    def compute_schedule(self, seq_len):
        """
        Compute depth and padding schedule for a given sequence length.
        
        Returns:
            depth: int, number of recursive levels
            padding_schedule: list of ints, padding needed at each level (from fine to coarse)
        """
        depth = 0
        padding_schedule = []
        current_len = seq_len
        
        # Calculate depth
        while current_len > self.min_leaf_size:
            remainder = current_len % self.k
            padding = (self.k - remainder) % self.k
            padding_schedule.append(padding)
            current_len = (current_len + padding) // self.k
            depth += 1
        
        # If depth is 0, we skip recursion (seq_len <= min_leaf_size)
        if depth == 0:
            return 0, []
        
        return depth, padding_schedule


# --- 2.3 Components: Down-Sampler, Bottleneck, Up-Sampler ---

class DownSampler(nn.Module):
    """
    Encoder: Localized mixing within groups of k tokens before pooling.
    """
    def __init__(self, d_model, k, expansion=2):
        super().__init__()
        self.k = k
        self.d_model = d_model
        
        # Local mixer: MLP that operates on groups of k tokens
        self.local_mixer = nn.Sequential(
            nn.Linear(d_model, d_model * expansion),
            nn.ReLU(),
            nn.Linear(d_model * expansion, d_model)
        )
        
        # Optional: Layer norm
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        """
        x: [B, N, D]
        Returns: [B, N//k, D] (after pooling)
        """
        B, N, D = x.shape
        
        # Reshape to [B, N//k, k, D] for local mixing
        # First, ensure N is divisible by k (padding should be handled upstream)
        assert N % self.k == 0, f"N ({N}) must be divisible by k ({self.k})"
        
        x_grouped = x.view(B, N // self.k, self.k, D)
        
        # Local mixing: apply MLP to each group
        x_mixed = self.local_mixer(x_grouped)  # [B, N//k, k, D]
        
        # Pool: take mean across k dimension
        x_pooled = x_mixed.mean(dim=2)  # [B, N//k, D]
        
        # Normalize
        x_pooled = self.norm(x_pooled)
        
        return x_pooled


class BottleneckAttention(nn.Module):
    """
    Global attention at the coarsest level with RoPE.
    """
    def __init__(self, d_model, nhead, max_seq_len=512):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        
        # Standard multi-head attention components
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Precompute RoPE frequencies (will be recomputed if seq_len > max_seq_len)
        self.max_seq_len = max_seq_len
        self.register_buffer('freqs_cis', precompute_freqs_cis(max_seq_len, self.head_dim))
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        """
        x: [B, N, D] where N is small (<= min_leaf_size)
        Returns: [B, N, D]
        """
        B, N, D = x.shape
        
        # Precompute RoPE if needed
        if N > self.max_seq_len:
            freqs_cis = precompute_freqs_cis(N, self.head_dim, device=x.device)
        else:
            freqs_cis = self.freqs_cis[:N]
        
        # Normalize
        x_norm = self.norm(x)
        
        # Project to Q, K, V
        q = self.q_proj(x_norm).view(B, N, self.nhead, self.head_dim)
        k = self.k_proj(x_norm).view(B, N, self.nhead, self.head_dim)
        v = self.v_proj(x_norm).view(B, N, self.nhead, self.head_dim)
        
        # Reshape for attention: [B, nhead, N, head_dim]
        q = q.transpose(1, 2)  # [B, nhead, N, head_dim]
        k = k.transpose(1, 2)  # [B, nhead, N, head_dim]
        v = v.transpose(1, 2)  # [B, nhead, N, head_dim]
        
        # Apply RoPE to Q and K
        q_rope = apply_rope(q, freqs_cis)
        k_rope = apply_rope(k, freqs_cis)
        
        # Scaled dot-product attention
        scores = torch.matmul(q_rope, k_rope.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)  # [B, nhead, N, head_dim]
        
        # Reshape back: [B, N, nhead, head_dim] -> [B, N, D]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, D)
        
        # Output projection
        output = self.out_proj(attn_output)
        
        # Residual connection
        return x + output


class UpSampler(nn.Module):
    """
    Decoder/Fuse: Expands coarse features and merges with skip connections.
    """
    def __init__(self, d_model, k, expansion=2):
        super().__init__()
        self.k = k
        self.d_model = d_model
        
        # Up-projection: prepares coarse features for expansion
        self.up_proj = nn.Linear(d_model, d_model)
        
        # Fusion MLP: merges (Fine, UpSampled_Coarse)
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model * expansion),
            nn.ReLU(),
            nn.Linear(d_model * expansion, d_model)
        )
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x_coarse, x_fine):
        """
        x_coarse: [B, N_coarse, D] from lower level
        x_fine: [B, N_fine, D] skip connection (N_fine = N_coarse * k)
        Returns: [B, N_fine, D]
        """
        B, N_coarse, D = x_coarse.shape
        _, N_fine, _ = x_fine.shape
        
        assert N_fine == N_coarse * self.k, f"Size mismatch: {N_fine} != {N_coarse} * {self.k}"
        
        # Up-project coarse features
        x_coarse_proj = self.up_proj(x_coarse)  # [B, N_coarse, D]
        
        # Expand: repeat each token k times
        x_coarse_expanded = x_coarse_proj.repeat_interleave(self.k, dim=1)  # [B, N_fine, D]
        
        # Concatenate fine and expanded coarse
        x_concat = torch.cat([x_fine, x_coarse_expanded], dim=-1)  # [B, N_fine, 2*D]
        
        # Fuse
        x_fused = self.fusion(x_concat)  # [B, N_fine, D]
        
        # Normalize
        x_fused = self.norm(x_fused)
        
        return x_fused


# --- 2.4 Main Architecture: Hierarchical V-Cycle Transformer ---

class HierarchicalVCycleTransformer(nn.Module):
    """
    Deterministic Hierarchical (V-Cycle) Transformer.
    Replaces sliding window with multigrid approach.
    """
    def __init__(self, input_dim=58, d_model=32, nhead=4, k=32, min_leaf_size=128, max_octree_depth=12):
        super().__init__()
        self.d_model = d_model
        self.k = k
        self.min_leaf_size = min_leaf_size
        self.scheduler = HierarchicalScheduler(k=k, min_leaf_size=min_leaf_size)
        
        # 1. Feature Projection
        self.feature_proj = nn.Linear(input_dim, d_model)
        
        # 2. Layer Embedding (physics scale)
        self.layer_embed = nn.Embedding(max_octree_depth, d_model)
        
        # 3. V-Cycle Components (one per level, but we'll reuse them)
        # Down-samplers
        self.down_samplers = nn.ModuleList()
        # Bottleneck (only one, used at the bottom)
        self.bottleneck = BottleneckAttention(d_model, nhead)
        # Up-samplers
        self.up_samplers = nn.ModuleList()
        
        # We'll create components dynamically based on max depth
        # For now, create enough for reasonable depths (e.g., 5 levels max)
        max_depth = 5
        for _ in range(max_depth):
            self.down_samplers.append(DownSampler(d_model, k))
            self.up_samplers.append(UpSampler(d_model, k))
        
        # 4. Output Head
        self.norm_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 25)
        
        # Break symmetry with noise
        with torch.no_grad():
            # Initialize weights with small variance to ensure gradient flow
            self.head.weight.normal_(0.0, 0.001)
            
            # Initialize bias with uniform noise to prevent "Zero Trap"
            # We want off-diagonals to be non-zero to force the optimizer to fix them
            self.head.bias.uniform_(-0.01, 0.01)
            
            # Hard-set the diagonal to a good starting point (Identity-ish)
            self.head.bias[0].fill_(1.0)
    
    def forward(self, x, layers):
        """
        x: [Batch, N, 58]
        layers: [Batch, N]
        """
        B, N, C = x.shape
        
        # A. Embedding
        h = self.feature_proj(x) + self.layer_embed(layers)
        
        # B. Compute schedule
        depth, padding_schedule = self.scheduler.compute_schedule(N)
        
        # If depth is 0, apply bottleneck directly
        if depth == 0:
            h = self.bottleneck(h)
            coeffs = self.head(self.norm_out(h))
            return coeffs
        
        # C. Phase A: Downward Pass (Restriction)
        skip_stack = []  # Store skip connections (post-pad, for fusion)
        residual_stack = []  # Store pre-pad versions (for residual connections)
        current = h
        
        for level in range(depth):
            # 1. Save pre-pad version for residual connection
            residual_stack.append(current.clone())
            
            # 2. Pad
            padding = padding_schedule[level]
            if padding > 0:
                current = F.pad(current, (0, 0, 0, padding))
            
            # 3. Save post-pad version (skip connection for fusion)
            skip_stack.append(current.clone())
            
            # 4. Local Mix & Pool (Down-Sample)
            down_sampler = self.down_samplers[level] if level < len(self.down_samplers) else self.down_samplers[-1]
            current = down_sampler(current)
        
        # D. Phase B: Bottleneck (Global Attention)
        current = self.bottleneck(current)
        
        # E. Phase C: Upward Pass (Prolongation)
        for level in range(depth - 1, -1, -1):  # Reverse order
            # 1. Retrieve skip connection (post-pad) and residual (pre-pad)
            x_fine = skip_stack.pop()
            x_residual = residual_stack.pop()
            
            # 2. Up-Sample and Fuse
            up_sampler = self.up_samplers[level] if level < len(self.up_samplers) else self.up_samplers[-1]
            current = up_sampler(current, x_fine)
            
            # 3. Crop padding
            padding = padding_schedule[level]
            if padding > 0:
                current = current[:, :current.shape[1] - padding, :]
            
            # 4. Residual connection (add original input before encoder at this level)
            current = current + x_residual
        
        # F. Output Prediction
        coeffs = self.head(self.norm_out(current))  # [B, N, 25]
        return coeffs


# Alias for backward compatibility
SPAIGenerator = HierarchicalVCycleTransformer

# --- 3. Unsupervised SAI Loss (Stochastic Trace Estimation) ---

class SAILoss(nn.Module):
    def __init__(self, epsilon=1e-4, num_probe_vectors=1):
        super().__init__()
        self.epsilon = epsilon
        self.num_probe_vectors = num_probe_vectors

    def forward(self, G_coeffs, A_diag, A_off, neighbors, valid_mask):
        """
        Computes Loss = || (1/||A||) * A * (G G^T + eps I) - I ||_F^2
        Uses multiple random probe vectors and averages the losses.
        """
        B, N, _ = G_coeffs.shape
        device = G_coeffs.device
        
        # 1. Estimate ||A|| (Mean of absolute non-zero entries) for Scale Invariance
        # We average over active nodes only
        active_count = valid_mask.sum()
        norm_A = (torch.abs(A_diag).sum() + torch.abs(A_off).sum()) / (active_count * 25 + 1e-6)
        
        # 2. Sample multiple Random Vectors w ~ N(0, 1) and average losses
        total_loss = 0.0
        
        for _ in range(self.num_probe_vectors):
            w = torch.randn(B, N, 1, device=device) * valid_mask
            
            # 3. Compute v = (G G^T + eps I) * w
            #    v = G * (G^T * w) + eps * w
            
            # Step A: u = G^T * w (Scatter)
            u = self.apply_GT(w, G_coeffs, neighbors, N)
            
            # Step B: v_part = G * u (Gather)
            v_part = self.apply_G(u, G_coeffs, neighbors, N)
            
            # Step C: Add epsilon regularization
            v = v_part + self.epsilon * w
            
            # 4. Compute z = (1/||A||) * A * v
            #    y = A * v
            #    z = y / norm_A
            y = self.apply_A(v, A_diag, A_off, neighbors, N)
            z = y / (norm_A + 1e-8)
            
            # 5. Loss = || z - w ||^2 
            # (Since target is Identity, A * M^-1 * w should equal w)
            diff = (z - w) * valid_mask
            loss = (diff ** 2).sum() / (valid_mask.sum() + 1e-6)
            total_loss += loss
        
        # Average over all probe vectors
        return total_loss / self.num_probe_vectors

    # --- Sparse Matrix Ops (Differentiable) ---

    def apply_G(self, x, coeffs, neighbors, N):
        """ 
        Gather (Standard SpMV): y_i = G_ii * x_i + sum(G_ij * x_j) 
        """
        B = x.shape[0]
        
        # Diagonal part
        res = x * coeffs[:, :, 0:1] # coeffs[0] is diag
        
        # Off-diagonal part
        safe_nbrs = neighbors.clone()
        safe_nbrs[safe_nbrs >= N] = 0 # Clamp invalid indices for gather
        
        # Flatten batch for indexing
        batch_idx = torch.arange(B, device=x.device).view(B, 1, 1) * N
        flat_idx = (safe_nbrs + batch_idx).view(-1)
        
        x_flat = x.view(-1, 1)
        x_nbrs = x_flat[flat_idx].view(B, N, 24)
        
        # Mask out 'Air' and 'Wall' neighbors (indices >= N)
        # neighbors is [B, N, 24], mask should be [B, N, 24] to match x_nbrs
        mask = (neighbors < N).float()
        x_nbrs = x_nbrs * mask
        
        # Weighted sum
        res += (x_nbrs * coeffs[:, :, 1:]).sum(dim=2, keepdim=True)
        return res

    def apply_GT(self, x, coeffs, neighbors, N):
        """ 
        Scatter (Transpose SpMV): y_j += G_ij * x_i 
        Value x_i * G_ij is added to node neighbors[i][j]
        """
        B = x.shape[0]
        
        # Diagonal part (Symmetric)
        res = x * coeffs[:, :, 0:1]
        
        # Off-diagonal scatter
        # Values to send: x[i] * G_ij
        vals = x * coeffs[:, :, 1:] # [B, N, 24]
        
        # Targets
        target_idx = neighbors.clone()
        mask = target_idx < N # Only scatter to fluid nodes
        
        # Flatten for index_add_
        batch_idx = torch.arange(B, device=x.device).view(B, 1, 1) * N
        flat_target = (target_idx + batch_idx).view(-1)
        flat_vals = vals.view(-1)
        flat_mask = mask.view(-1)
        
        # Result container
        res_flat = res.view(-1) # Modify in-place if possible, but index_add_ is safer
        
        # Filter valids
        valid_indices = flat_target[flat_mask]
        valid_values = flat_vals[flat_mask]
        
        if valid_indices.numel() > 0:
            res_flat.index_add_(0, valid_indices, valid_values)
            
        return res_flat.view(B, N, 1)

    def apply_A(self, x, A_diag, A_off, neighbors, N):
        """ Apply Physics Matrix A """
        coeffs = torch.cat([A_diag, A_off], dim=2)
        return self.apply_G(x, coeffs, neighbors, N)

# --- 4. Dataset ---

class FluidGraphDataset(Dataset):
    def __init__(self, root_dirs):
        self.frame_paths = []
        for d in root_dirs:
            d = Path(d)
            if not d.exists():
                continue
                
            # Check if there are Run_* directories (nested structure)
            all_items = list(d.iterdir())
            run_dirs = sorted([f for f in all_items if f.is_dir() and f.name.startswith('Run_')])
            
            if run_dirs:
                # Nested structure: TestData/Run_*/frame_*
                for run_dir in run_dirs:
                    frames = sorted([f for f in run_dir.iterdir() if f.is_dir() and f.name.startswith('frame_')])
                    self.frame_paths.extend(frames)
            else:
                # Flat structure: TestData/frame_* (backward compatibility)
                frames = sorted([f for f in all_items if f.is_dir() and f.name.startswith('frame_')])
                self.frame_paths.extend(frames)

        # Pre-scan for Max Nodes to handle padding
        max_n = 0
        for p in self.frame_paths:
            try:
                with open(p / "meta.txt", 'r') as f:
                    for line in f:
                        if 'numNodes' in line: max_n = max(max_n, int(line.split(':')[1].strip()))
            except: continue
        
        # Round up to window_size for window efficiency
        # Note: This assumes window_size=256, update if changed
        window_size = 256
        self.max_nodes = ((max_n + window_size - 1) // window_size) * window_size
        print(f"Dataset: {len(self.frame_paths)} frames. Max Nodes padded to: {self.max_nodes}")
        
        self.node_dtype = np.dtype([
            ('position', '3<f4'), ('velocity', '3<f4'), ('face_vels', '6<f4'),
            ('mass', '<f4'), ('layer', '<u4'), ('morton', '<u4'), ('active', '<u4')
        ])

    def __len__(self): return len(self.frame_paths)

    def __getitem__(self, idx):
        frame_path = self.frame_paths[idx]
        
        # 1. Read Metadata
        num_nodes = 0
        with open(frame_path / "meta.txt", 'r') as f:
            for line in f:
                if 'numNodes' in line: num_nodes = int(line.split(':')[1].strip())
        if num_nodes == 0: return self.__getitem__((idx+1)%len(self)) # Skip empty

        # 2. Read Binaries
        def read_bin(name, dtype):
            return np.fromfile(frame_path / name, dtype=dtype)
            
        raw_nodes = read_bin("nodes.bin", self.node_dtype)
        raw_nbrs = read_bin("neighbors.bin", np.uint32).reshape(num_nodes, 24)
        
        # 3. Compute Physics (Matrix A) on CPU for Input Features
        t_nbrs = torch.from_numpy(raw_nbrs.astype(np.int64))
        t_layers = torch.from_numpy(raw_nodes['layer'][:num_nodes].astype(np.int64)).unsqueeze(1)
        A_diag, A_off = compute_laplacian_stencil(t_nbrs, t_layers, num_nodes)
        
        # 4. Build Input Features (Dim 58)
        # Pos (3)
        pos = raw_nodes['position'][:num_nodes] / 1024.0 # Normalized
        
        # Wall Flags (6) - Reduced from 24
        # We check the 6 canonical directions. 
        # Neighbor layout is: [Left:0-3, Right:4-7, Down:8-11, Up:12-15, Front:16-19, Back:20-23]
        # If the *first* slot of a face is Wall, the whole face is Wall.
        IDX_WALL = num_nodes + 1
        is_wall_full = (raw_nbrs == IDX_WALL)
        wall_indices = [0, 4, 8, 12, 16, 20] # Indices of first sub-neighbor per face
        wall_flags = is_wall_full[:, wall_indices].astype(np.float32) # [N, 6]
        
        # Air Flags (24) - Kept full resolution
        IDX_AIR = num_nodes
        air_flags = (raw_nbrs == IDX_AIR).astype(np.float32) # [N, 24]
        
        # Matrix A (25)
        # A_diag (1) + A_off (24)
        
        # Assemble Feature Vector
        features_np = np.column_stack([
            pos,                  # 3
            wall_flags,           # 6
            air_flags,            # 24
            A_diag.numpy(),       # 1
            A_off.numpy()         # 24
        ]) 
        # Total: 58
        
        # 5. Padding
        padded_features = np.zeros((self.max_nodes, 58), dtype=np.float32)
        padded_features[:num_nodes] = features_np
        
        padded_layers = np.zeros((self.max_nodes,), dtype=np.int64)
        padded_layers[:num_nodes] = t_layers.squeeze().numpy()
        
        # Pad neighbors with 'Invalid' index (self.max_nodes) so they are masked out
        padded_nbrs = np.full((self.max_nodes, 24), self.max_nodes, dtype=np.int64)
        padded_nbrs[:num_nodes] = raw_nbrs
        
        # Pad Matrix A for Loss
        padded_A_diag = np.zeros((self.max_nodes, 1), dtype=np.float32)
        padded_A_diag[:num_nodes] = A_diag.numpy()
        padded_A_off = np.zeros((self.max_nodes, 24), dtype=np.float32)
        padded_A_off[:num_nodes] = A_off.numpy()
        
        # Mask
        valid_mask = np.zeros((self.max_nodes, 1), dtype=np.float32)
        valid_mask[:num_nodes] = 1.0
        
        return {
            'x': padded_features,
            'layers': padded_layers,
            'neighbors': padded_nbrs,
            'A_diag': padded_A_diag,
            'A_off': padded_A_off,
            'mask': valid_mask
        }

# --- 5. Export Utility ---

def export_weights(model, path):
    print(f"Exporting packed 16-bit weights to {path}...")
    with open(path, 'wb') as f:
        # 1. Header: Standard 32-bit floats/ints
        # Format: [reserved1, reserved2, d_model, nhead, num_levels, input_dim]
        # For hierarchical architecture, num_levels is the max depth of down/up samplers
        num_levels = len(model.down_samplers)
        nhead = model.bottleneck.nhead
        header = struct.pack('<ffiiii', 
                           0.0, 0.0, 
                           model.d_model, 
                           nhead,
                           num_levels, 
                           58)
        f.write(header)
        
        # 2. Helper to write packed tensor
        def write_packed_tensor(tensor):
            # A. Transpose 2D weights (Input-Major for HLSL)
            if len(tensor.shape) == 2:
                # Transpose: [Out, In] -> [In, Out]
                t_data = tensor.t().detach().cpu().numpy().flatten()
            else:
                t_data = tensor.detach().cpu().numpy().flatten()
            
            # B. Pad if length is odd (Need even count to pack pairs)
            if len(t_data) % 2 != 0:
                t_data = np.append(t_data, 0.0)
                
            # C. Convert to float16, then view as uint32
            # This packs two consecutive float16s into one 32-bit integer
            # Low 16 bits = First float, High 16 bits = Second float
            data_packed = t_data.astype(np.float16).view(np.uint32)
            
            f.write(data_packed.tobytes())

        # --- Write Weights (Hierarchical V-Cycle Architecture) ---
        
        # 1. Feature Projection
        write_packed_tensor(model.feature_proj.weight)
        write_packed_tensor(model.feature_proj.bias)
        write_packed_tensor(model.layer_embed.weight)

        # 2. Down-Samplers (one per level)
        for down_sampler in model.down_samplers:
            # Local mixer MLP
            write_packed_tensor(down_sampler.local_mixer[0].weight)
            write_packed_tensor(down_sampler.local_mixer[0].bias)
            write_packed_tensor(down_sampler.local_mixer[2].weight)
            write_packed_tensor(down_sampler.local_mixer[2].bias)
            write_packed_tensor(down_sampler.norm.weight)
            write_packed_tensor(down_sampler.norm.bias)

        # 3. Bottleneck Attention (Global Attention with RoPE)
        bottleneck = model.bottleneck
        write_packed_tensor(bottleneck.q_proj.weight)
        write_packed_tensor(bottleneck.q_proj.bias)
        write_packed_tensor(bottleneck.k_proj.weight)
        write_packed_tensor(bottleneck.k_proj.bias)
        write_packed_tensor(bottleneck.v_proj.weight)
        write_packed_tensor(bottleneck.v_proj.bias)
        write_packed_tensor(bottleneck.out_proj.weight)
        write_packed_tensor(bottleneck.out_proj.bias)
        write_packed_tensor(bottleneck.norm.weight)
        write_packed_tensor(bottleneck.norm.bias)

        # 4. Up-Samplers (one per level)
        for up_sampler in model.up_samplers:
            # Up-projection
            write_packed_tensor(up_sampler.up_proj.weight)
            write_packed_tensor(up_sampler.up_proj.bias)
            # Fusion MLP
            write_packed_tensor(up_sampler.fusion[0].weight)
            write_packed_tensor(up_sampler.fusion[0].bias)
            write_packed_tensor(up_sampler.fusion[2].weight)
            write_packed_tensor(up_sampler.fusion[2].bias)
            write_packed_tensor(up_sampler.norm.weight)
            write_packed_tensor(up_sampler.norm.bias)

        # 5. Output Head
        write_packed_tensor(model.norm_out.weight)
        write_packed_tensor(model.norm_out.bias)
        write_packed_tensor(model.head.weight)
        write_packed_tensor(model.head.bias)
        
    print("Export complete.")

# --- 6. Training Loop ---

def train(args):
    # Use Metal (MPS) on macOS, CUDA on Linux/Windows, CPU as fallback
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")
    
    # Init Data - ensure path is absolute
    data_path = Path(args.data_folder)
    if not data_path.is_absolute():
        # If relative, make it relative to script directory
        # Script is in Assets/Scripts/, so parent.parent = Assets/
        script_dir = Path(__file__).parent
        data_path = script_dir.parent / "StreamingAssets" / "TestData"
    
    # FluidGraphDataset expects root directories that contain frame_* subdirectories
    dataset = FluidGraphDataset([data_path])
    if len(dataset) == 0:
        print("No data found! Check path.")
        return
    
    # Determine if we should use random frame sampling per epoch
    use_random_sampling = args.frames_per_epoch > 0
        
    # Create base loader (will be recreated each epoch if using random sampling)
    if use_random_sampling:
        # Initial loader with full dataset (will be replaced each epoch)
        loader = None
    else:
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Init Model (Hierarchical V-Cycle Transformer)
    model = HierarchicalVCycleTransformer(
        input_dim=58, 
        d_model=args.d_model, 
        nhead=args.nhead if hasattr(args, 'nhead') else 4,
        k=args.k if hasattr(args, 'k') else 32,
        min_leaf_size=args.min_leaf_size if hasattr(args, 'min_leaf_size') else 128
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = SAILoss(num_probe_vectors=args.num_probe_vectors)
    
    # Initialize Mixed Precision Training (AMP)
    # GradScaler is only needed for CUDA to handle potential underflows
    use_amp = device.type in ['cuda', 'mps']
    scaler = None
    if device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
        print("AMP enabled with GradScaler (CUDA)")
    elif device.type == 'mps':
        print("AMP enabled (MPS - no GradScaler needed)")
    else:
        print("AMP disabled (CPU)")
    
    # Compile the model and the loss function for kernel fusion
    # This works best on Linux/Windows NVIDIA GPUs. On Mac MPS, support is experimental.
    try:
        model = torch.compile(model)
        criterion = torch.compile(criterion)
        print("Model and loss function compiled with torch.compile")
    except Exception as e:
        print(f"Warning: torch.compile failed ({e}), continuing without compilation")
    
    # Paths
    script_dir = Path(__file__).parent
    weights_path = script_dir / "model_weights.bytes"
    
    print("Starting Training...")
    
    # Create CUDA events for timing (only on CUDA)
    use_cuda_events = device.type == 'cuda'
    if use_cuda_events:
        start_event = torch.cuda.Event(enable_timing=True)
        fwd_end_event = torch.cuda.Event(enable_timing=True)
        loss_end_event = torch.cuda.Event(enable_timing=True)
    
    epoch_times = []
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        
        # If using random frame sampling, create a new sampler and loader for this epoch
        if use_random_sampling:
            # Randomly sample indices
            num_frames = min(args.frames_per_epoch, len(dataset))
            indices = torch.randperm(len(dataset))[:num_frames].tolist()
            sampler = SubsetRandomSampler(indices)
            loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, shuffle=False)
            print(f"Epoch {epoch+1}: Using {num_frames} randomly selected frames")
        
        # Accumulators for average times
        total_fwd_time = 0.0
        total_loss_time = 0.0
        
        for batch in loader:
            x = batch['x'].to(device)
            layers = batch['layers'].to(device)
            nbrs = batch['neighbors'].to(device)
            A_diag = batch['A_diag'].to(device)
            A_off = batch['A_off'].to(device)
            mask = batch['mask'].to(device)
            
            optimizer.zero_grad()
            
            # --- Start Timing ---
            if use_cuda_events:
                start_event.record()
            else:
                fwd_start = time.time()
            
            # 1. Forward Pass with AMP
            if use_amp:
                # Use new torch.amp.autocast() API with device type
                device_type = 'cuda' if device.type == 'cuda' else 'mps'
                with torch.amp.autocast(device_type=device_type):
                    G_coeffs = model(x, layers)
            else:
                G_coeffs = model(x, layers)
            
            if use_cuda_events:
                fwd_end_event.record()
            else:
                fwd_end = time.time()
            
            # 2. Loss Calculation with AMP
            if use_amp:
                # Use new torch.amp.autocast() API with device type
                device_type = 'cuda' if device.type == 'cuda' else 'mps'
                with torch.amp.autocast(device_type=device_type):
                    loss = criterion(G_coeffs, A_diag, A_off, nbrs, mask)
            else:
                loss = criterion(G_coeffs, A_diag, A_off, nbrs, mask)
            
            if use_cuda_events:
                loss_end_event.record()
            else:
                loss_end = time.time()
            # --------------------
            
            # 3. Backward Pass with AMP Scaler (CUDA only)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            total_loss += loss.item()
            
            # Synchronize to get accurate times
            if use_cuda_events:
                torch.cuda.synchronize()
                # Calculate elapsed times in milliseconds
                fwd = start_event.elapsed_time(fwd_end_event)
                loss = fwd_end_event.elapsed_time(loss_end_event)
            else:
                # Use time.time() for MPS/CPU (convert to milliseconds)
                fwd = (fwd_end - fwd_start) * 1000.0
                loss = (loss_end - fwd_end) * 1000.0
            
            total_fwd_time += fwd
            total_loss_time += loss
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        avg_time = sum(epoch_times) / len(epoch_times)
        
        # Calculate average times per batch
        avg_fwd = total_fwd_time / len(loader)
        avg_loss_time = total_loss_time / len(loader)
        
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.6f} | Time: {epoch_time:.2f}s (forward: {avg_fwd:.1f}s, loss: {avg_loss_time:.1f}s) | Avg Time: {avg_time:.2f}s")
        
        if (epoch + 1) % 5 == 0:
            export_weights(model, weights_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Default to the StreamingAssets/TestData folder
    # Script is in Assets/Scripts/, so parent.parent = Assets/
    default_data_path = Path(__file__).parent.parent / "StreamingAssets" / "TestData"
    parser.add_argument('--data_folder', type=str, default=str(default_data_path))
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--layers', type=int, default=4, help='Deprecated: kept for compatibility, not used in hierarchical architecture')
    parser.add_argument('--nhead', type=int, default=4, help='Number of attention heads in bottleneck')
    parser.add_argument('--k', type=int, default=32, help='Branching factor (compression rate per level)')
    parser.add_argument('--min_leaf_size', type=int, default=128, help='Minimum sequence length at bottom level')
    parser.add_argument('--num_probe_vectors', type=int, default=4)
    parser.add_argument('--frames_per_epoch', type=int, default=0, 
                        help='Number of random frames to use per epoch. If 0, uses all frames.')
    args = parser.parse_args()
    
    train(args)