#!/usr/bin/env python3
"""
MatrixStatistics.py
Analyzes the sparsity pattern of the A matrix (Laplacian) from recorded fluid simulation data.
Focuses on a frame 1/4 of the way through the dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.sparse import csr_matrix, coo_matrix
import torch
from NeuralPreconditioner import FluidGraphDataset, compute_laplacian_stencil

def reconstruct_sparse_matrix(A_diag, A_off, neighbors, num_nodes):
    """
    Reconstructs a sparse matrix A from diagonal, off-diagonal, and neighbor information.
    Boundaries (Air/Wall) are not counted as neighbors but tracked separately.
    
    Args:
        A_diag: [N, 1] diagonal values
        A_off: [N, 24] off-diagonal values (one per neighbor)
        neighbors: [N, 24] neighbor indices
        num_nodes: actual number of nodes (excluding padding)
    
    Returns:
        A: scipy.sparse.csr_matrix of shape (num_nodes, num_nodes)
        boundary_stats: dict with counts of Air and Wall boundaries
    """
    N = num_nodes
    IDX_AIR = N
    IDX_WALL = N + 1
    
    # Track boundaries separately
    air_count = 0
    wall_count = 0
    invalid_count = 0
    
    # Build COO format (row, col, data)
    rows = []
    cols = []
    data = []
    
    # Add diagonal entries
    for i in range(N):
        rows.append(i)
        cols.append(i)
        data.append(A_diag[i, 0])
    
    # Add off-diagonal entries (only for valid fluid nodes)
    for i in range(N):
        for k in range(24):
            neighbor_idx = neighbors[i, k]
            
            # Classify neighbor type
            if neighbor_idx == IDX_AIR:
                air_count += 1
            elif neighbor_idx == IDX_WALL:
                wall_count += 1
            elif neighbor_idx > IDX_WALL:
                invalid_count += 1
            elif 0 <= neighbor_idx < N:
                # Valid fluid node - add to matrix
                val = A_off[i, k]
                # Only add non-zero values
                if abs(val) > 1e-10:
                    rows.append(i)
                    cols.append(int(neighbor_idx))
                    data.append(val)
    
    # Create sparse matrix
    A = coo_matrix((data, (rows, cols)), shape=(N, N))
    # Convert to CSR for efficient operations
    A = A.tocsr()
    
    boundary_stats = {
        'air_boundaries': air_count,
        'wall_boundaries': wall_count,
        'invalid_indices': invalid_count,
        'total_boundary_slots': air_count + wall_count + invalid_count
    }
    
    return A, boundary_stats

def analyze_sparsity_pattern(A, num_nodes):
    """
    Analyzes the sparsity pattern of matrix A.
    
    Returns:
        dict with statistics
    """
    stats = {}
    
    # Basic properties
    stats['shape'] = A.shape
    stats['nnz'] = A.nnz  # Number of non-zeros
    stats['density'] = A.nnz / (A.shape[0] * A.shape[1])
    stats['avg_nnz_per_row'] = A.nnz / A.shape[0]
    
    # Diagonal statistics
    diag = A.diagonal()
    stats['diag_min'] = diag.min()
    stats['diag_max'] = diag.max()
    stats['diag_mean'] = diag.mean()
    stats['diag_std'] = diag.std()
    
    # Off-diagonal statistics
    A_off = A.copy()
    A_off.setdiag(0)  # Remove diagonal
    off_diag_values = A_off.data
    if len(off_diag_values) > 0:
        stats['off_diag_min'] = off_diag_values.min()
        stats['off_diag_max'] = off_diag_values.max()
        stats['off_diag_mean'] = off_diag_values.mean()
        stats['off_diag_std'] = off_diag_values.std()
        stats['off_diag_abs_mean'] = np.abs(off_diag_values).mean()
    else:
        stats['off_diag_min'] = 0
        stats['off_diag_max'] = 0
        stats['off_diag_mean'] = 0
        stats['off_diag_std'] = 0
        stats['off_diag_abs_mean'] = 0
    
    # Connectivity analysis
    # Count non-zeros per row (degree of each node)
    row_nnz = np.diff(A.indptr)  # Number of non-zeros per row
    stats['min_degree'] = row_nnz.min()
    stats['max_degree'] = row_nnz.max()
    stats['mean_degree'] = row_nnz.mean()
    stats['std_degree'] = row_nnz.std()
    
    # Symmetry check (A should be symmetric for Laplacian)
    A_sym_diff = A - A.T
    stats['symmetry_error'] = np.abs(A_sym_diff.data).max() if A_sym_diff.nnz > 0 else 0.0
    
    # Check for isolated nodes (zero diagonal and no connections)
    isolated = (diag == 0) & (row_nnz == 1)  # Only diagonal entry
    stats['num_isolated'] = isolated.sum()
    
    return stats

def visualize_sparsity_pattern(A, num_nodes, frame_idx):
    """
    Visualizes the sparsity pattern of matrix A.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Full sparsity pattern (may be large, so sample if needed)
    ax = axes[0, 0]
    if num_nodes > 2000:
        # Sample a 2000x2000 submatrix for visualization
        sample_size = 2000
        A_sample = A[:sample_size, :sample_size]
        ax.spy(A_sample, markersize=0.5, precision=1e-10)
        ax.set_title(f"Sparsity Pattern (Top-Left {sample_size}x{sample_size} of {num_nodes}x{num_nodes})")
    else:
        ax.spy(A, markersize=0.5, precision=1e-10)
        ax.set_title(f"Sparsity Pattern ({num_nodes}x{num_nodes})")
    ax.set_xlabel("Column Index")
    ax.set_ylabel("Row Index")
    
    # 2. Histogram of non-zero values
    ax = axes[0, 1]
    nnz_values = A.data
    ax.hist(nnz_values, bins=100, alpha=0.7, edgecolor='black')
    ax.set_xlabel("Non-zero Value")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Non-zero Values")
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # 3. Histogram of diagonal values
    ax = axes[1, 0]
    diag = A.diagonal()
    ax.hist(diag, bins=50, alpha=0.7, edgecolor='black', color='green')
    ax.set_xlabel("Diagonal Value")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Diagonal Values")
    ax.grid(True, alpha=0.3)
    
    # 4. Histogram of off-diagonal values
    ax = axes[1, 1]
    A_off = A.copy()
    A_off.setdiag(0)
    off_diag_values = A_off.data
    if len(off_diag_values) > 0:
        ax.hist(off_diag_values, bins=100, alpha=0.7, edgecolor='black', color='red')
        ax.set_xlabel("Off-diagonal Value")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Off-diagonal Values")
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No off-diagonal values", ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Distribution of Off-diagonal Values")
    
    plt.tight_layout()
    plt.show()

def main():
    # Setup paths
    script_dir = Path(__file__).parent
    default_data_path = script_dir.parent / "StreamingAssets" / "TestData"
    
    # Load dataset
    dataset = FluidGraphDataset([default_data_path])
    if len(dataset) == 0:
        print(f"Error: No data found at {default_data_path}")
        return
    
    # Get frame at 1/4 of the way through
    frame_idx = len(dataset) // 4
    print(f"Analyzing frame {frame_idx} of {len(dataset)} (1/4 of the way through)")
    
    # Load the frame
    batch = dataset[frame_idx]
    
    # Extract data (remove padding)
    num_nodes = int(batch['mask'].sum())
    print(f"Number of nodes: {num_nodes}")
    
    A_diag = batch['A_diag'][:num_nodes]  # [N, 1]
    A_off = batch['A_off'][:num_nodes]    # [N, 24]
    neighbors = batch['neighbors'][:num_nodes]  # [N, 24]
    
    # Reconstruct sparse matrix
    print("Reconstructing sparse matrix A...")
    A, boundary_stats = reconstruct_sparse_matrix(A_diag, A_off, neighbors, num_nodes)
    
    # Analyze sparsity pattern
    print("Analyzing sparsity pattern...")
    stats = analyze_sparsity_pattern(A, num_nodes)
    
    # Print statistics
    print("\n" + "="*60)
    print("MATRIX A SPARSITY STATISTICS")
    print("="*60)
    print(f"Matrix Shape: {stats['shape']}")
    print(f"Number of Non-zeros: {stats['nnz']:,}")
    print(f"Density: {stats['density']:.6f} ({stats['density']*100:.4f}%)")
    print(f"Average Non-zeros per Row: {stats['avg_nnz_per_row']:.2f}")
    print()
    print("Diagonal Statistics:")
    print(f"  Min: {stats['diag_min']:.6f}")
    print(f"  Max: {stats['diag_max']:.6f}")
    print(f"  Mean: {stats['diag_mean']:.6f}")
    print(f"  Std: {stats['diag_std']:.6f}")
    print()
    print("Off-diagonal Statistics:")
    print(f"  Min: {stats['off_diag_min']:.6f}")
    print(f"  Max: {stats['off_diag_max']:.6f}")
    print(f"  Mean: {stats['off_diag_mean']:.6f}")
    print(f"  Mean |Value|: {stats['off_diag_abs_mean']:.6f}")
    print(f"  Std: {stats['off_diag_std']:.6f}")
    print()
    print("Connectivity (Degree per Node):")
    print(f"  Min Degree: {stats['min_degree']}")
    print(f"  Max Degree: {stats['max_degree']}")
    print(f"  Mean Degree: {stats['mean_degree']:.2f}")
    print(f"  Std Degree: {stats['std_degree']:.2f}")
    print()
    print("Other Properties:")
    print(f"  Symmetry Error (max |A - A^T|): {stats['symmetry_error']:.2e}")
    print(f"  Isolated Nodes: {stats['num_isolated']}")
    print()
    print("Boundary Statistics (not counted as neighbors):")
    print(f"  Air Boundaries (Dirichlet): {boundary_stats['air_boundaries']:,}")
    print(f"  Wall Boundaries (Neumann): {boundary_stats['wall_boundaries']:,}")
    print(f"  Invalid/Invalid Indices: {boundary_stats['invalid_indices']:,}")
    print(f"  Total Boundary Slots: {boundary_stats['total_boundary_slots']:,}")
    print(f"  (Out of {num_nodes * 24:,} total neighbor slots)")
    print("="*60)
    
    # Visualize
    print("\nGenerating visualizations...")
    visualize_sparsity_pattern(A, num_nodes, frame_idx)
    
    # Additional analysis: Bandwidth
    # Compute bandwidth (maximum distance from diagonal)
    bandwidth = 0
    for i in range(A.shape[0]):
        row_start = A.indptr[i]
        row_end = A.indptr[i+1]
        if row_end > row_start:
            cols = A.indices[row_start:row_end]
            max_dist = np.max(np.abs(cols - i))
            bandwidth = max(bandwidth, max_dist)
    
    print(f"\nMatrix Bandwidth: {bandwidth}")
    print(f"  (Maximum distance from diagonal to any non-zero)")
    
    # Check if matrix is approximately banded
    if bandwidth < num_nodes * 0.1:
        print(f"  Matrix appears to be banded (bandwidth << N)")
    else:
        print(f"  Matrix is not strongly banded")

if __name__ == "__main__":
    main()
