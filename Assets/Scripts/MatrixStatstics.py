#!/usr/bin/env python3
"""
Reconstruct matrix A from saved data and visualize sparsity pattern.
Ignores boundaries (air/wall nodes).
"""

import numpy as np
import torch
import matplotlib
# Use Agg backend for non-interactive (works everywhere)
# If user wants interactive, they can set MPLBACKEND environment variable
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from pathlib import Path
import sys
import traceback

# Import the compute_laplacian_stencil function from NeuralPreconditioner
sys.path.insert(0, str(Path(__file__).parent))
from NeuralPreconditioner import compute_laplacian_stencil

def load_frame_data(frame_path):
    """Load data from a saved frame."""
    frame_path = Path(frame_path)
    
    # Read metadata
    num_nodes = 0
    with open(frame_path / "meta.txt", 'r') as f:
        for line in f:
            if 'numNodes' in line:
                num_nodes = int(line.split(':')[1].strip())
    
    if num_nodes == 0:
        raise ValueError("numNodes is 0 or not found in metadata")
    
    # Define node dtype (matches NeuralPreconditioner)
    node_dtype = np.dtype([
        ('position', '3<f4'), ('velocity', '3<f4'), ('face_vels', '6<f4'),
        ('mass', '<f4'), ('layer', '<u4'), ('morton', '<u4'), ('active', '<u4')
    ])
    
    # Read binary files
    raw_nodes = np.fromfile(frame_path / "nodes.bin", dtype=node_dtype)
    
    # Neighbors buffer uses SoA (Structure of Arrays) layout:
    # [AllNodes_Slot0, AllNodes_Slot1, ..., AllNodes_Slot23]
    # So we need to reshape as (24, num_nodes) first, then transpose to (num_nodes, 24)
    raw_nbrs = np.fromfile(frame_path / "neighbors.bin", dtype=np.uint32).reshape(24, num_nodes).T
    
    return raw_nodes, raw_nbrs, num_nodes

def print_buffer_samples(raw_nodes, raw_nbrs, num_nodes, num_samples=10):
    """Print sample entries from decoded buffers for debugging."""
    print(f"\n=== Sample Buffer Entries (showing {min(num_samples, num_nodes)} of {num_nodes} nodes from middle of list) ===\n")
    
    # Calculate start index to sample from the middle
    start_idx = max(0, (num_nodes - num_samples) // 2)
    end_idx = min(num_nodes, start_idx + num_samples)
    
    print(f"Sampling nodes {start_idx} to {end_idx-1} (middle of list)\n")
    
    # Print sample nodes from the middle
    for i in range(start_idx, end_idx):
        node = raw_nodes[i]
        print(f"Node {i}:")
        print(f"  Position: ({node['position'][0]:.4f}, {node['position'][1]:.4f}, {node['position'][2]:.4f})")
        print(f"  Velocity: ({node['velocity'][0]:.4f}, {node['velocity'][1]:.4f}, {node['velocity'][2]:.4f})")
        print(f"  Mass: {node['mass']:.4f}")
        print(f"  Layer: {node['layer']}")
        print(f"  Morton Code: {node['morton']}")
        print(f"  Active: {node['active']}")
        
        # Print neighbors for this node
        neighbors = raw_nbrs[i]
        print(f"  Neighbors (24 slots):")
        # Group by face (6 faces, 4 neighbors each)
        all_valid_neighbors = []
        for face in range(6):
            face_start = face * 4
            face_nbrs = neighbors[face_start:face_start+4]
            face_names = ['left', 'right', 'bottom', 'top', 'front', 'back']
            print(f"    {face_names[face]}: {face_nbrs[0]}, {face_nbrs[1]}, {face_nbrs[2]}, {face_nbrs[3]}")
            # Show which are valid (less than num_nodes)
            valid = [int(n) for n in face_nbrs if n < num_nodes]
            if valid:
                print(f"      -> Valid fluid neighbors: {valid}")
                all_valid_neighbors.extend(valid)
            elif face_nbrs[0] == num_nodes:
                print(f"      -> All invalid (no neighbor)")
            elif face_nbrs[0] == num_nodes + 1:
                print(f"      -> Wall boundary")
        
        # Analyze neighbor index distribution
        if all_valid_neighbors:
            min_nbr = min(all_valid_neighbors)
            max_nbr = max(all_valid_neighbors)
            avg_nbr = sum(all_valid_neighbors) / len(all_valid_neighbors)
            print(f"  Neighbor index stats: min={min_nbr}, max={max_nbr}, avg={avg_nbr:.1f}, current_node={i}")
            if min_nbr < i * 0.1:  # If neighbors are much lower than current index
                print(f"    ⚠️  WARNING: Neighbors are mostly low-index (current node is {i})")
                # Check morton codes and spatial distances of low-index neighbors
                print(f"    Checking low-index neighbors:")
                current_pos = np.array([node['position'][0], node['position'][1], node['position'][2]])
                for nbr_idx in sorted(set(all_valid_neighbors))[:5]:  # Check first 5 unique neighbors
                    if nbr_idx < num_nodes:
                        nbr_node = raw_nodes[nbr_idx]
                        nbr_morton = nbr_node['morton']
                        nbr_layer = nbr_node['layer']
                        nbr_pos = np.array([nbr_node['position'][0], nbr_node['position'][1], nbr_node['position'][2]])
                        distance = np.linalg.norm(current_pos - nbr_pos)
                        print(f"      Neighbor {nbr_idx}: morton={nbr_morton}, layer={nbr_layer}, pos=({nbr_pos[0]:.1f}, {nbr_pos[1]:.1f}, {nbr_pos[2]:.1f}), distance={distance:.1f}")
            if max_nbr > i * 1.5:  # If neighbors are much higher than current index
                print(f"    ⚠️  WARNING: Neighbors are mostly high-index (current node is {i})")
        print()
    
    # Print some statistics
    print(f"=== Buffer Statistics ===")
    print(f"Total nodes: {num_nodes}")
    
    # Count neighbor types
    valid_count = 0
    invalid_count = 0
    wall_count = 0
    for i in range(num_nodes):
        for j in range(24):
            nbr = raw_nbrs[i, j]
            if nbr < num_nodes:
                valid_count += 1
            elif nbr == num_nodes:
                invalid_count += 1
            elif nbr == num_nodes + 1:
                wall_count += 1
    
    print(f"Neighbor slots:")
    print(f"  Valid fluid neighbors: {valid_count}")
    print(f"  Invalid (no neighbor): {invalid_count}")
    print(f"  Wall boundaries: {wall_count}")
    print(f"  Total slots: {num_nodes * 24}")
    print()

def reconstruct_matrix_A(raw_nodes, raw_nbrs, num_nodes):
    """Reconstruct sparse matrix A from node and neighbor data."""
    # Convert to torch tensors
    t_nbrs = torch.from_numpy(raw_nbrs.astype(np.int64))
    t_layers = torch.from_numpy(raw_nodes['layer'][:num_nodes].astype(np.int64)).unsqueeze(1)
    
    # Compute Laplacian stencil (diagonal and off-diagonal weights)
    A_diag, A_off = compute_laplacian_stencil(t_nbrs, t_layers, num_nodes)
    
    # Convert to numpy
    A_diag_np = A_diag.numpy().flatten()
    A_off_np = A_off.numpy()
    nbrs_np = raw_nbrs.astype(np.int64)
    
    # Build sparse matrix (COO format)
    rows = []
    cols = []
    data = []
    
    # Add diagonal entries
    for i in range(num_nodes):
        if abs(A_diag_np[i]) > 1e-9:
            rows.append(i)
            cols.append(i)
            data.append(float(A_diag_np[i]))
    
    # Add off-diagonal entries (ignoring boundaries: air/wall/invalid)
    IDX_AIR = num_nodes
    IDX_WALL = num_nodes + 1
    
    for i in range(num_nodes):
        for k in range(24):
            col = nbrs_np[i, k]
            # Only include valid fluid neighbors (not air/wall/invalid)
            if col < num_nodes:  # Valid fluid neighbor
                val = A_off_np[i, k]
                if abs(val) > 1e-9:
                    rows.append(i)
                    cols.append(int(col))
                    data.append(float(val))
    
    # Create sparse COO matrix
    if len(data) > 0:
        A_sparse = coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
        # Convert to CSR for better performance
        A_sparse = A_sparse.tocsr()
    else:
        raise ValueError("No non-zero entries found in matrix A")
    
    return A_sparse, A_diag_np, A_off_np

def compute_neighbor_distances(raw_nodes, raw_nbrs, num_nodes):
    """Compute average Euclidean distance from each node to its valid neighbors."""
    distances = np.zeros(num_nodes)
    
    for i in range(num_nodes):
        node_pos = np.array([raw_nodes[i]['position'][0], 
                            raw_nodes[i]['position'][1], 
                            raw_nodes[i]['position'][2]])
        
        neighbor_distances = []
        for j in range(24):
            nbr_idx = raw_nbrs[i, j]
            if nbr_idx < num_nodes:  # Valid neighbor
                nbr_pos = np.array([raw_nodes[nbr_idx]['position'][0],
                                   raw_nodes[nbr_idx]['position'][1],
                                   raw_nodes[nbr_idx]['position'][2]])
                dist = np.linalg.norm(node_pos - nbr_pos)
                neighbor_distances.append(dist)
        
        if neighbor_distances:
            distances[i] = np.mean(neighbor_distances)
        else:
            distances[i] = np.nan  # No valid neighbors
    
    return distances

def visualize_sparsity_pattern(A_sparse, raw_nodes, raw_nbrs, num_nodes, save_path=None):
    """Visualize the sparsity pattern of matrix A, colored by average neighbor distance."""
    print(f"Matrix A: {num_nodes} x {num_nodes}")
    print(f"Non-zero entries: {A_sparse.nnz}")
    print(f"Sparsity: {(1 - A_sparse.nnz / (num_nodes * num_nodes)) * 100:.2f}%")
    
    # Compute neighbor distances
    print("Computing neighbor distances...")
    neighbor_distances = compute_neighbor_distances(raw_nodes, raw_nbrs, num_nodes)
    
    # Statistics on distances
    valid_distances = neighbor_distances[~np.isnan(neighbor_distances)]
    if len(valid_distances) > 0:
        print(f"Neighbor distance stats:")
        print(f"  Mean: {np.mean(valid_distances):.2f}")
        print(f"  Median: {np.median(valid_distances):.2f}")
        print(f"  Min: {np.min(valid_distances):.2f}")
        print(f"  Max: {np.max(valid_distances):.2f}")
        print(f"  Nodes with neighbors > 100 units away: {np.sum(valid_distances > 100)}")
        print(f"  Nodes with neighbors > 500 units away: {np.sum(valid_distances > 500)}")
    
    # Create figure with two subplots: A and A^-1
    fig, axes = plt.subplots(1, 2, figsize=(24, 12))
    
    # Plot 1: Sparsity pattern of A colored by neighbor distance
    ax1 = axes[0]
    
    # Convert sparse matrix to COO format for coloring
    A_coo = A_sparse.tocoo()
    
    # For each non-zero entry, get the row index and find its average neighbor distance
    row_distances = neighbor_distances[A_coo.row]
    
    # Create scatter plot colored by distance
    scatter1 = ax1.scatter(A_coo.col, A_coo.row, c=row_distances, 
                         s=0.1, cmap='viridis_r', vmin=0, vmax=np.percentile(valid_distances, 95))
    ax1.set_xlabel('Column Index (Neighbor Node)')
    ax1.set_ylabel('Row Index (Current Node)')
    ax1.set_title(f'Matrix A - Sparsity Pattern\n({num_nodes}x{num_nodes})')
    ax1.invert_yaxis()  # Match spy plot orientation
    ax1.set_aspect('auto')
    
    # Add colorbar
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Average Euclidean Distance to Neighbors', rotation=270, labelpad=20)
    
    # Plot 2: Inverse of A (A^-1) using sparse LU factorization
    ax2 = axes[1]
    
    print("Computing inverse of A (using sparse LU factorization)...")
    try:
        from scipy.sparse.linalg import splu
        
        # Use sparse LU factorization to efficiently compute the full inverse
        print(f"  Computing sparse LU factorization...")
        lu = splu(A_sparse.tocsc())  # Convert to CSC format for efficient factorization
        
        print(f"  Computing all {num_nodes} columns of A^-1...")
        A_inv = np.zeros((num_nodes, num_nodes))
        
        # Compute all columns by solving A * x = e_i for each unit vector
        for col_idx in range(num_nodes):
            if (col_idx + 1) % 500 == 0:
                print(f"    Computing column {col_idx+1}/{num_nodes}...")
            # Solve A * x = e_col_idx
            e = np.zeros(num_nodes)
            e[col_idx] = 1.0
            x = lu.solve(e)
            A_inv[:, col_idx] = x
        
        # Visualize the inverse
        # Use log scale for better visualization of small values
        A_inv_abs = np.abs(A_inv)
        A_inv_log = np.log10(A_inv_abs + 1e-10)  # Add small value to avoid log(0)
        
        im = ax2.imshow(A_inv_log, cmap='viridis', aspect='auto', origin='upper')
        ax2.set_xlabel('Column Index')
        ax2.set_ylabel('Row Index')
        ax2.set_title(f'Inverse of A (log10 |A^-1|)\n({num_nodes}x{num_nodes})')
        
        # Add colorbar
        cbar2 = plt.colorbar(im, ax=ax2)
        cbar2.set_label('log10(|A^-1|)', rotation=270, labelpad=20)
        
        # Print statistics on inverse
        print(f"Inverse of A statistics:")
        print(f"  Max |A^-1|: {np.max(A_inv_abs):.2e}")
        print(f"  Min |A^-1|: {np.min(A_inv_abs[A_inv_abs > 0]):.2e}")
        print(f"  Mean |A^-1|: {np.mean(A_inv_abs):.2e}")
        
    except ImportError:
        print(f"  Error: scipy.sparse.linalg.splu not available, falling back to dense inverse")
        # Fallback to original method for small matrices
        if num_nodes < 2000:
            A_dense = A_sparse.toarray()
            A_inv = np.linalg.inv(A_dense)
            A_inv_abs = np.abs(A_inv)
            A_inv_log = np.log10(A_inv_abs + 1e-10)
            im = ax2.imshow(A_inv_log, cmap='viridis', aspect='auto', origin='upper')
            ax2.set_title(f'Inverse of A (log10 |A^-1|)\n({num_nodes}x{num_nodes})')
            cbar2 = plt.colorbar(im, ax=ax2)
            cbar2.set_label('log10(|A^-1|)', rotation=270, labelpad=20)
        else:
            ax2.text(0.5, 0.5, f'Matrix too large\n({num_nodes}x{num_nodes})\nfor dense inverse', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=14)
            ax2.set_title('Inverse of A (Not Computed)')
    except MemoryError:
        print(f"  Error: Not enough memory for sparse LU factorization")
        ax2.text(0.5, 0.5, f'Not enough memory\n({num_nodes}x{num_nodes})', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=14)
        ax2.set_title('Inverse of A (Not Computed)')
    except Exception as e:
        print(f"  Error computing approximate inverse: {e}")
        import traceback
        traceback.print_exc()
        ax2.text(0.5, 0.5, f'Error computing inverse:\n{str(e)}', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Inverse of A (Error)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved sparsity pattern to {save_path}")
    else:
        # With Agg backend, we can't show interactively, so this shouldn't be reached
        # but handle it gracefully
        import tempfile
        import os
        temp_file = os.path.join(tempfile.gettempdir(), 'sparsity_pattern.png')
        plt.savefig(temp_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {temp_file}")
    
    plt.close(fig)  # Close to free memory
    return fig

def analyze_matrix_properties(A_sparse):
    """Analyze properties of matrix A."""
    num_nodes = A_sparse.shape[0]
    
    print("\n=== Matrix A Properties ===")
    print(f"Size: {num_nodes} x {num_nodes}")
    print(f"Non-zero entries: {A_sparse.nnz}")
    print(f"Sparsity: {(1 - A_sparse.nnz / (num_nodes * num_nodes)) * 100:.2f}%")
    print(f"Average non-zeros per row: {A_sparse.nnz / num_nodes:.2f}")
    
    # Check symmetry
    A_sym = A_sparse - A_sparse.T
    max_asymmetry = np.abs(A_sym.data).max() if A_sym.nnz > 0 else 0.0
    print(f"Max asymmetry: {max_asymmetry:.2e}")
    if max_asymmetry < 1e-6:
        print("Matrix is symmetric (within tolerance)")
    else:
        print("Matrix is NOT symmetric")
    
    # Diagonal dominance (using sparse operations to avoid memory issues)
    try:
        diag = A_sparse.diagonal()
        # Compute row sums using sparse operations (absolute values)
        # Convert to COO to work with absolute values easily
        A_coo = A_sparse.tocoo()
        A_abs_coo = coo_matrix((np.abs(A_coo.data), (A_coo.row, A_coo.col)), shape=A_sparse.shape)
        A_abs_csr = A_abs_coo.tocsr()
        row_sums = np.array(A_abs_csr.sum(axis=1)).flatten()
        off_diag_sums = row_sums - np.abs(diag)
        dominance = np.abs(diag) - off_diag_sums
        min_dominance = dominance.min()
        print(f"Diagonal dominance (min): {min_dominance:.2e}")
        if min_dominance > 0:
            print("Matrix is diagonally dominant")
        else:
            print(f"Matrix is NOT diagonally dominant (min dominance: {min_dominance:.2e})")
    except Exception as e:
        print(f"Could not compute diagonal dominance: {e}")
        import traceback
        traceback.print_exc()
    
    # Condition number estimate (for small matrices)
    if num_nodes < 1000:
        try:
            A_dense = A_sparse.toarray()
            cond_num = np.linalg.cond(A_dense)
            print(f"Condition number: {cond_num:.2e}")
        except Exception as e:
            print(f"Could not compute condition number: {e}")

def main():
    """Main function to load data, reconstruct A, and visualize."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Reconstruct matrix A and visualize sparsity pattern')
    parser.add_argument('--frame', type=str, 
                       default='StreamingAssets/TestData/Run_2026-01-24_11-51-49/frame_0300',
                       help='Path to frame directory (relative to Assets folder, default: frame_0300 - 1/4 through recording)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for sparsity pattern image (default: auto-open saved image, no saving to file)')
    parser.add_argument('--analyze', action='store_true', default=True,
                       help='Print detailed matrix analysis (default: enabled)')
    parser.add_argument('--no-analyze', dest='analyze', action='store_false',
                       help='Disable detailed matrix analysis')
    
    args = parser.parse_args()
    
    # Construct full path
    script_dir = Path(__file__).parent.absolute()
    assets_dir = script_dir.parent if script_dir.name == 'Scripts' else script_dir
    
    # Handle both relative and absolute paths
    if Path(args.frame).is_absolute():
        frame_path = Path(args.frame)
    else:
        # Try relative to Assets directory first
        frame_path = assets_dir / args.frame
        if not frame_path.exists():
            # Try relative to current working directory
            frame_path = Path(args.frame)
    
    if not frame_path.exists():
        print(f"Error: Frame path does not exist: {frame_path}")
        print(f"Looking for: {frame_path}")
        return
    
    print(f"Loading data from: {frame_path}")
    
    # Load data
    try:
        raw_nodes, raw_nbrs, num_nodes = load_frame_data(frame_path)
        print(f"Loaded {num_nodes} nodes")
        
        # Print sample buffer entries for debugging
        print_buffer_samples(raw_nodes, raw_nbrs, num_nodes, num_samples=5)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Reconstruct matrix A
    try:
        print("Reconstructing matrix A...")
        A_sparse, A_diag, A_off = reconstruct_matrix_A(raw_nodes, raw_nbrs, num_nodes)
        print("Matrix A reconstructed successfully")
    except Exception as e:
        print(f"Error reconstructing matrix A: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Analyze matrix properties (default is True)
    analyze_matrix_properties(A_sparse)
    
    # Visualize sparsity pattern
    try:
        print("Visualizing sparsity pattern...")
        # If no output specified, save to temp file and open it
        if args.output is None:
            import tempfile
            import os
            import subprocess
            temp_file = os.path.join(tempfile.gettempdir(), 'sparsity_pattern.png')
            visualize_sparsity_pattern(A_sparse, raw_nodes, raw_nbrs, num_nodes, save_path=temp_file)
            # Try to open the image
            try:
                subprocess.run(['open', temp_file], check=False)  # macOS
                print(f"Opened sparsity pattern: {temp_file}")
            except:
                try:
                    subprocess.run(['xdg-open', temp_file], check=False)  # Linux
                    print(f"Opened sparsity pattern: {temp_file}")
                except:
                    print(f"Sparsity pattern saved to: {temp_file}")
        else:
            visualize_sparsity_pattern(A_sparse, raw_nodes, raw_nbrs, num_nodes, save_path=args.output)
        print("Visualization complete")
    except Exception as e:
        print(f"Error visualizing sparsity pattern: {e}")
        traceback.print_exc()
        # Still print basic stats even if visualization fails
        print(f"\nBasic stats:")
        print(f"  Matrix size: {A_sparse.shape[0]} x {A_sparse.shape[1]}")
        print(f"  Non-zeros: {A_sparse.nnz}")
        print(f"  Sparsity: {(1 - A_sparse.nnz / (A_sparse.shape[0] * A_sparse.shape[1])) * 100:.2f}%")

if __name__ == '__main__':
    main()
