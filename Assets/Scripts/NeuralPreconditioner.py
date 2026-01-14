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
from torch.utils.data import Dataset, DataLoader
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

# --- 2. Architecture: SPAI Generator ---

class SPAIGenerator(nn.Module):
    # CHANGE 1: Default d_model=32 (matches paper's ~24k parameters)
    def __init__(self, input_dim=58, d_model=32, num_layers=4, nhead=4, window_size=256, max_octree_depth=12):
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size
        
        # 1. Feature Projection
        # Input: Pos(3) + Wall(6) + Air(24) + A_diag(1) + A_off(24) = 58
        self.feature_proj = nn.Linear(input_dim, d_model)
        
        # 2. Embeddings
        # Encodes the scale of the physics (Layer 10 vs Layer 5)
        self.layer_embed = nn.Embedding(max_octree_depth, d_model)
        # Encodes position within the local window (0-511)
        self.window_pos_embed = nn.Parameter(torch.randn(1, window_size, d_model) * 0.02)
        
        # 3. Backbone: Windowed Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 2, 
            dropout=0.0, 
            activation='gelu', 
            batch_first=True, 
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Output Head
        # Predicts Matrix G coefficients (1 Diag + 24 Off-Diag)
        self.norm_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 25) 
        
        # Init Strategy: Start close to Identity (G_diag=1, G_off=0)
        with torch.no_grad():
            self.head.weight.fill_(0.0)
            self.head.bias.fill_(0.0)
            self.head.bias[0] = 1.0 

    def forward(self, x, layers):
        """
        x: [Batch, N, 58]
        layers: [Batch, N]
        """
        B, N, C = x.shape
        
        # A. Embedding
        h = self.feature_proj(x) + self.layer_embed(layers)
        
        # B. Windowing
        # Pad N to multiple of window_size
        pad_len = (self.window_size - (N % self.window_size)) % self.window_size
        if pad_len > 0:
            h = F.pad(h, (0, 0, 0, pad_len))
            
        N_padded = h.shape[1]
        num_windows = N_padded // self.window_size
        
        # Reshape to [Batch * NumWindows, WindowSize, d_model]
        h_windows = h.view(B * num_windows, self.window_size, self.d_model)
        
        # Add Window Position Embedding (Broadcasts)
        h_windows = h_windows + self.window_pos_embed
        
        # C. Transformer (Local Attention)
        h_encoded = self.encoder(h_windows)
        
        # D. Un-Window
        h_flat = h_encoded.view(B, N_padded, self.d_model)
        
        # Remove Padding
        if pad_len > 0:
            h_flat = h_flat[:, :N, :]
            
        # E. Output Prediction
        coeffs = self.head(self.norm_out(h_flat)) # [B, N, 25]
        return coeffs

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
    print(f"Exporting weights to {path}...")
    with open(path, 'wb') as f:
        # 1. Header: p_mean (float), p_std (float), d_model (int), heads (int), layers (int), input_dim (int)
        # We don't have p_mean/std here, writing 0.0
        header = struct.pack('<ffiiii', 
                           0.0, 0.0, 
                           model.d_model, 
                           model.encoder.layers[0].self_attn.num_heads,  # num_heads from model
                           len(model.encoder.layers), 
                           58) # Input dim is fixed at 58
        f.write(header)
        
        # Helper to write tensor
        def write_tensor(tensor):
            # TRANSPOSE 2D WEIGHTS for HLSL (Input-Major) compatibility
            if len(tensor.shape) == 2:
                t_data = tensor.t().cpu().detach().numpy().astype(np.float32)
            else:
                t_data = tensor.cpu().detach().numpy().astype(np.float32)
            f.write(t_data.tobytes())

        # --- 1. Feature Projection ---
        write_tensor(model.feature_proj.weight) # Will be Transposed [58, d_model]
        write_tensor(model.feature_proj.bias)
        write_tensor(model.layer_embed.weight)    # [12, d_model]
        write_tensor(model.window_pos_embed)      # [1, window_size, d_model]

        # --- 2. Transformer Layers ---
        for i, layer in enumerate(model.encoder.layers):
            # Attention In (Q,K,V packed)
            # PyTorch: [3*d_model, d_model] -> Transpose to [d_model, 3*d_model]
            # This lines up perfectly with HLSL reading stride of 3*d_model
            write_tensor(layer.self_attn.in_proj_weight) 
            write_tensor(layer.self_attn.in_proj_bias)
            
            # Attention Out
            write_tensor(layer.self_attn.out_proj.weight)
            write_tensor(layer.self_attn.out_proj.bias)
            
            # Norm 1
            write_tensor(layer.norm1.weight)
            write_tensor(layer.norm1.bias)
            
            # FFN 1 (Linear 1)
            write_tensor(layer.linear1.weight)
            write_tensor(layer.linear1.bias)
            
            # FFN 2 (Linear 2)
            write_tensor(layer.linear2.weight)
            write_tensor(layer.linear2.bias)
            
            # Norm 2
            write_tensor(layer.norm2.weight)
            write_tensor(layer.norm2.bias)

        # --- 3. Output Head ---
        write_tensor(model.norm_out.weight)
        write_tensor(model.norm_out.bias)
        write_tensor(model.head.weight)
        write_tensor(model.head.bias)
        
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
        
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Init Model
    model = SPAIGenerator(input_dim=58, d_model=args.d_model, num_layers=args.layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = SAILoss(num_probe_vectors=args.num_probe_vectors)
    
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
            
            # 1. Forward Pass (Model Inference)
            G_coeffs = model(x, layers)
            
            if use_cuda_events:
                fwd_end_event.record()
            else:
                fwd_end = time.time()
            
            # 2. Loss Calculation (Physics / Sparse Matrix Ops)
            loss = criterion(G_coeffs, A_diag, A_off, nbrs, mask)
            
            if use_cuda_events:
                loss_end_event.record()
            else:
                loss_end = time.time()
            # --------------------
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            # Synchronize to get accurate times
            if use_cuda_events:
                torch.cuda.synchronize()
                # Calculate elapsed times in milliseconds
                fwd_ms = start_event.elapsed_time(fwd_end_event)
                loss_ms = fwd_end_event.elapsed_time(loss_end_event)
            else:
                # Use time.time() for MPS/CPU (convert to milliseconds)
                fwd_ms = (fwd_end - fwd_start) * 1000.0
                loss_ms = (loss_end - fwd_end) * 1000.0
            
            total_fwd_time += fwd_ms
            total_loss_time += loss_ms
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        avg_time = sum(epoch_times) / len(epoch_times)
        
        # Calculate average times per batch
        avg_fwd = total_fwd_time / len(loader)
        avg_loss_time = total_loss_time / len(loader)
        
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.6f} | Time: {epoch_time:.2f}s (forward: {avg_fwd:.1f}ms, loss: {avg_loss_time:.1f}ms) | Avg Time: {avg_time:.2f}s")
        
        if (epoch + 1) % 5 == 0:
            export_weights(model, weights_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Default to the StreamingAssets/TestData folder
    # Script is in Assets/Scripts/, so parent.parent = Assets/
    default_data_path = Path(__file__).parent.parent / "StreamingAssets" / "TestData"
    parser.add_argument('--data_folder', type=str, default=str(default_data_path))
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--num_probe_vectors', type=int, default=1)
    args = parser.parse_args()
    
    train(args)