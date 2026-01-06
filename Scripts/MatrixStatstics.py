#!/usr/bin/env python3
"""
Matrix Analysis Script for Fluid Simulation Data
Analyzes the Laplacian matrix properties before solving to understand
what can make the conjugate gradient solver faster.
"""

import numpy as np
import struct
import os
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, csc_matrix, diags
from scipy.sparse.linalg import eigsh
import warnings
warnings.filterwarnings('ignore')

class MatrixAnalyzer:
    def __init__(self, data_folder):
        self.data_folder = Path(data_folder)
        self.num_nodes = None
        self.nodes = None
        self.neighbors = None
        self.divergence = None
        self.pressure = None
        self.metadata = {}
        
    def load_metadata(self, frame_path):
        """Load metadata from meta.txt"""
        meta_file = frame_path / "meta.txt"
        if not meta_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_file}")
        
        with open(meta_file, 'r') as f:
            for line in f:
                if ':' in line:
                    key, value = line.strip().split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Try to parse as number
                    try:
                        if '.' in value:
                            self.metadata[key] = float(value)
                        else:
                            self.metadata[key] = int(value)
                    except ValueError:
                        # Parse vector values
                        if ' ' in value:
                            try:
                                self.metadata[key] = [float(x) for x in value.split()]
                            except ValueError:
                                self.metadata[key] = value
                        else:
                            self.metadata[key] = value
        
        self.num_nodes = self.metadata.get('numNodes', 0)
        print(f"Loaded metadata: {self.num_nodes} nodes")
        return self.metadata
    
    def load_binary_buffer(self, filepath, dtype, count=None):
        """Load binary buffer file"""
        with open(filepath, 'rb') as f:
            data = f.read()
        
        if dtype == np.uint32:
            arr = np.frombuffer(data, dtype=np.uint32)
        elif dtype == np.float32:
            arr = np.frombuffer(data, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
        
        if count is not None:
            arr = arr[:count]
        
        return arr
    
    def load_nodes(self, frame_path):
        """Load nodes buffer (64 bytes per node)"""
        nodes_file = frame_path / "nodes.bin"
        if not nodes_file.exists():
            raise FileNotFoundError(f"Nodes file not found: {nodes_file}")
        
        # Node structure: position(12) + velocity(12) + faceVelocities(24) + mass(4) + layer(4) + mortonCode(4) + active(4) = 64 bytes
        with open(nodes_file, 'rb') as f:
            data = f.read()
        
        num_nodes = len(data) // 64
        nodes = []
        
        for i in range(num_nodes):
            offset = i * 64
            # Read position (3 floats)
            pos = struct.unpack('3f', data[offset:offset+12])
            offset += 12
            # Read velocity (3 floats)
            vel = struct.unpack('3f', data[offset:offset+12])
            offset += 12
            # Read face velocities (6 floats)
            face_vels = struct.unpack('6f', data[offset:offset+24])
            offset += 24
            # Read mass (1 float)
            mass = struct.unpack('f', data[offset:offset+4])[0]
            offset += 4
            # Read layer (1 uint)
            layer = struct.unpack('I', data[offset:offset+4])[0]
            offset += 4
            # Read mortonCode (1 uint)
            morton_code = struct.unpack('I', data[offset:offset+4])[0]
            offset += 4
            # Read active (1 uint)
            active = struct.unpack('I', data[offset:offset+4])[0]
            
            nodes.append({
                'position': np.array(pos),
                'velocity': np.array(vel),
                'face_velocities': np.array(face_vels),
                'mass': mass,
                'layer': layer,
                'morton_code': morton_code,
                'active': active
            })
        
        self.nodes = np.array(nodes, dtype=object)
        print(f"Loaded {len(self.nodes)} nodes")
        return self.nodes
    
    def load_neighbors(self, frame_path):
        """Load neighbors buffer (24 uints per node, or infer stride from file size)"""
        neighbors_file = frame_path / "neighbors.bin"
        if not neighbors_file.exists():
            raise FileNotFoundError(f"Neighbors file not found: {neighbors_file}")
        
        # Infer stride from file size if num_nodes is known
        file_size = neighbors_file.stat().st_size
        total_ints = file_size // 4  # Each index is 4 bytes (uint32)
        
        if self.num_nodes is not None and self.num_nodes > 0:
            stride = total_ints // self.num_nodes
            print(f"Inferred neighbor stride from file size: {stride} ints per node")
        else:
            # Default to 24 if we don't know num_nodes yet
            stride = 24
            print(f"Using default neighbor stride: {stride} ints per node")
        
        # Load data
        neighbors = self.load_binary_buffer(neighbors_file, np.uint32)
        
        # Reshape to (num_nodes, stride)
        if self.num_nodes is not None:
            expected_size = self.num_nodes * stride
            if neighbors.size != expected_size:
                print(f"Warning: Neighbor file size mismatch. Expected {expected_size}, got {neighbors.size}. Resizing...")
                neighbors = neighbors[:expected_size] if neighbors.size > expected_size else np.pad(neighbors, (0, expected_size - neighbors.size))
            self.neighbors = neighbors.reshape(self.num_nodes, stride)
        else:
            # If we don't know num_nodes, infer it
            num_nodes = total_ints // stride
            self.neighbors = neighbors[:num_nodes * stride].reshape(num_nodes, stride)
            print(f"Inferred {num_nodes} nodes from neighbor file")
        
        print(f"Loaded neighbors: shape {self.neighbors.shape}")
        return self.neighbors
    
    def load_divergence(self, frame_path):
        """Load divergence buffer (1 float per node)"""
        divergence_file = frame_path / "divergence.bin"
        if not divergence_file.exists():
            raise FileNotFoundError(f"Divergence file not found: {divergence_file}")
        
        self.divergence = self.load_binary_buffer(divergence_file, np.float32, self.num_nodes)
        print(f"Loaded divergence: shape {self.divergence.shape}")
        return self.divergence
    
    def load_pressure(self, frame_path):
        """Load pressure buffer (1 float per node)"""
        pressure_file = frame_path / "pressure.bin"
        if not pressure_file.exists():
            print("Warning: Pressure file not found, skipping")
            return None
        
        self.pressure = self.load_binary_buffer(pressure_file, np.float32, self.num_nodes)
        print(f"Loaded pressure: shape {self.pressure.shape}")
        return self.pressure
    
    def load_frame(self, frame_index):
        """Load all data for a specific frame"""
        frame_path = self.data_folder / f"frame_{frame_index:04d}"
        
        if not frame_path.exists():
            raise FileNotFoundError(f"Frame {frame_index} not found: {frame_path}")
        
        print(f"\n{'='*60}")
        print(f"Loading frame {frame_index}")
        print(f"{'='*60}")
        
        self.load_metadata(frame_path)
        self.load_nodes(frame_path)
        self.load_neighbors(frame_path)
        self.load_divergence(frame_path)
        self.load_pressure(frame_path)
        
        return True
    
    def reconstruct_matrix(self):
        """
        Reconstruct the sparse Laplacian matrix A from neighbors.
        The matrix is applied as: Ap = -Laplacian(p)
        Based on CGSolver.compute ApplyLaplacian kernel.
        """
        print("\n" + "="*60)
        print("Reconstructing sparse matrix A")
        print("="*60)
        
        num_nodes = self.num_nodes
        # Use dictionary to accumulate values (handles duplicates)
        matrix_dict = {}
        
        print("Building matrix entries...")
        for i in range(num_nodes):
            if (i + 1) % 5000 == 0:
                print(f"  Processing node {i+1}/{num_nodes}")
            node_i = self.nodes[i]
            dx_i = 2.0 ** node_i['layer']
            
            if not np.isfinite(dx_i):
                dx_i = 1e-6
            
            neighbor_base = i * 24
            diagonal_sum = 0.0  # Track diagonal contribution
            
            # Loop over 6 faces
            for d in range(6):
                face_base = neighbor_base + d * 4
                neighbor_idx = self.neighbors[i, d * 4]  # First slot: same/parent layer (index d*4 in the 24-element array)
                
                # Case 1: True Boundary (Neumann BC)
                if neighbor_idx == num_nodes + 1:
                    # Zero flux, no contribution
                    continue
                
                # Case 2: Same-layer or Parent (coarser) neighbor
                elif neighbor_idx < num_nodes and self.nodes[neighbor_idx]['layer'] >= node_i['layer']:
                    node_j = self.nodes[neighbor_idx]
                    dx_j = 2.0 ** node_j['layer']
                    
                    if not np.isfinite(dx_j):
                        dx_j = 1e-6
                    
                    # Distance between cell centers
                    distance = max(0.5 * (dx_i + dx_j), 1e-6)
                    
                    # Face area is the area of cell i's face
                    a_shared = dx_i * dx_i
                    
                    # Weight: area divided by distance
                    w = a_shared / distance
                    
                    # Off-diagonal: w * p_j
                    key = (i, neighbor_idx)
                    matrix_dict[key] = matrix_dict.get(key, 0.0) + w
                    
                    # Diagonal: -w * p_i
                    diagonal_sum -= w
                
                # Case 3 or 4: Child (finer) neighbors or Dirichlet boundary
                else:
                    face_area = dx_i * dx_i
                    child_face_area = face_area / 4.0
                    
                    for k in range(4):
                        child_idx = self.neighbors[i, d * 4 + k]  # Slots 0-3 for this face
                        
                        # Case 3: Child (finer) neighbor
                        if child_idx < num_nodes:
                            node_k = self.nodes[child_idx]
                            dx_k = 2.0 ** node_k['layer']
                            
                            if not np.isfinite(dx_k):
                                dx_k = 1e-6
                            
                            distance = max(0.5 * (dx_i + dx_k), 1e-6)
                            w = child_face_area / distance
                            
                            # Off-diagonal: w * p_k
                            key = (i, child_idx)
                            matrix_dict[key] = matrix_dict.get(key, 0.0) + w
                            
                            # Diagonal: -w * p_i
                            diagonal_sum -= w
                        
                        # Case 4: Dirichlet boundary
                        elif child_idx == num_nodes:
                            dx_k = dx_i * 0.5
                            distance = max(0.5 * (dx_i + dx_k), 1e-6)
                            w = child_face_area / distance
                            
                            # Diagonal: -w * p_i (boundary has zero pressure)
                            diagonal_sum -= w
            
            # Add diagonal entry
            key = (i, i)
            matrix_dict[key] = matrix_dict.get(key, 0.0) + diagonal_sum
        
        print("Converting to sparse matrix...")
        # Convert dictionary to sparse matrix format
        if len(matrix_dict) == 0:
            raise ValueError("Matrix dictionary is empty!")
        
        rows, cols, data = zip(*[(r, c, v) for (r, c), v in matrix_dict.items()])
        print(f"  Created {len(matrix_dict)} matrix entries")
        
        # Create sparse matrix (negative Laplacian)
        A = csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
        
        # Apply negative sign (as per CGSolver.compute line 193)
        A = -A
        
        # Apply deltaTime scaling (assuming deltaTime = 1/frameRate)
        delta_time = 1.0 / self.metadata.get('frameRate', 60.0)
        A = A * delta_time
        
        print(f"Matrix shape: {A.shape}")
        print(f"Number of nonzeros: {A.nnz}")
        print(f"Sparsity: {(1 - A.nnz / (num_nodes * num_nodes)) * 100:.2f}%")
        
        self.matrix = A
        return A
    
    def analyze_banding(self):
        """Analyze the banding structure of the matrix"""
        print("\n" + "="*60)
        print("BANDING ANALYSIS")
        print("="*60)
        
        A = self.matrix
        num_nodes = A.shape[0]
        
        # Calculate bandwidth
        max_upper_bandwidth = 0
        max_lower_bandwidth = 0
        
        for i in range(num_nodes):
            row = A.getrow(i)
            if row.nnz > 0:
                col_indices = row.indices
                upper_band = max(col_indices[col_indices > i] - i) if np.any(col_indices > i) else 0
                lower_band = max(i - col_indices[col_indices < i]) if np.any(col_indices < i) else 0
                max_upper_bandwidth = max(max_upper_bandwidth, upper_band)
                max_lower_bandwidth = max(max_lower_bandwidth, lower_band)
        
        bandwidth = max_upper_bandwidth + max_lower_bandwidth + 1
        print(f"Maximum upper bandwidth: {max_upper_bandwidth}")
        print(f"Maximum lower bandwidth: {max_lower_bandwidth}")
        print(f"Total bandwidth: {bandwidth}")
        print(f"Bandwidth ratio: {bandwidth / num_nodes * 100:.2f}%")
        
        # Analyze distance distribution
        distances = []
        for i in range(min(1000, num_nodes)):  # Sample for speed
            row = A.getrow(i)
            if row.nnz > 0:
                col_indices = row.indices
                for j in col_indices:
                    if j != i:
                        distances.append(abs(j - i))
        
        if len(distances) > 0:
            distances = np.array(distances)
            print(f"\nDistance statistics (sample of {len(distances)} entries):")
            print(f"  Mean distance: {np.mean(distances):.2f}")
            print(f"  Median distance: {np.median(distances):.2f}")
            print(f"  Std deviation: {np.std(distances):.2f}")
            print(f"  Min distance: {np.min(distances)}")
            print(f"  Max distance: {np.max(distances)}")
            print(f"  95th percentile: {np.percentile(distances, 95):.2f}")
            distance_stats = distances
        else:
            distance_stats = None
        
        return {
            'max_upper_bandwidth': max_upper_bandwidth,
            'max_lower_bandwidth': max_lower_bandwidth,
            'total_bandwidth': bandwidth,
            'distance_stats': distance_stats
        }
    
    def analyze_symmetry(self):
        """Check if matrix is symmetric"""
        print("\n" + "="*60)
        print("SYMMETRY ANALYSIS")
        print("="*60)
        
        A = self.matrix
        # Check symmetry: A should equal A^T
        diff = A - A.T
        max_diff = np.abs(diff.data).max() if diff.nnz > 0 else 0.0
        
        print(f"Maximum |A - A^T|: {max_diff:.2e}")
        
        if max_diff < 1e-6:
            print("Matrix is SYMMETRIC (within tolerance)")
            is_symmetric = True
        else:
            print("Matrix is NOT symmetric")
            is_symmetric = False
        
        return {'is_symmetric': is_symmetric, 'max_diff': max_diff}
    
    def analyze_diagonal_dominance(self):
        """Analyze diagonal dominance"""
        print("\n" + "="*60)
        print("DIAGONAL DOMINANCE ANALYSIS")
        print("="*60)
        
        A = self.matrix
        diagonal = A.diagonal()
        row_sums = np.abs(A).sum(axis=1).A1
        off_diag_sums = row_sums - np.abs(diagonal)
        
        # Strict diagonal dominance: |A_ii| > sum(|A_ij|) for j != i
        is_strictly_dd = np.abs(diagonal) > off_diag_sums
        num_strictly_dd = np.sum(is_strictly_dd)
        
        # Weak diagonal dominance: |A_ii| >= sum(|A_ij|) for j != i
        is_weakly_dd = np.abs(diagonal) >= off_diag_sums
        num_weakly_dd = np.sum(is_weakly_dd)
        
        # Diagonal dominance ratio
        dd_ratios = np.abs(diagonal) / (off_diag_sums + 1e-10)
        
        print(f"Strictly diagonally dominant rows: {num_strictly_dd} / {A.shape[0]} ({num_strictly_dd/A.shape[0]*100:.1f}%)")
        print(f"Weakly diagonally dominant rows: {num_weakly_dd} / {A.shape[0]} ({num_weakly_dd/A.shape[0]*100:.1f}%)")
        print(f"\nDiagonal dominance ratio statistics:")
        print(f"  Mean: {np.mean(dd_ratios):.4f}")
        print(f"  Median: {np.median(dd_ratios):.4f}")
        print(f"  Min: {np.min(dd_ratios):.4f}")
        print(f"  Max: {np.max(dd_ratios):.4f}")
        print(f"  25th percentile: {np.percentile(dd_ratios, 25):.4f}")
        print(f"  75th percentile: {np.percentile(dd_ratios, 75):.4f}")
        
        return {
            'num_strictly_dd': num_strictly_dd,
            'num_weakly_dd': num_weakly_dd,
            'dd_ratios': dd_ratios
        }
    
    def analyze_condition_number(self):
        """Estimate condition number"""
        print("\n" + "="*60)
        print("CONDITION NUMBER ANALYSIS")
        print("="*60)
        
        A = self.matrix
        
        # For large sparse matrices, compute a few extreme eigenvalues
        num_eigenvals = min(5, max(2, A.shape[0] // 10000))  # Adaptive based on size
        print(f"Computing {num_eigenvals} extreme eigenvalues (this may take a while)...")
        try:
            # Compute smallest and largest eigenvalues
            eigenvals_small, _ = eigsh(A, k=num_eigenvals, which='SA', maxiter=1000, tol=1e-4)
            eigenvals_large, _ = eigsh(A, k=num_eigenvals, which='LA', maxiter=1000, tol=1e-4)
            
            lambda_min = np.min(eigenvals_small)
            lambda_max = np.max(eigenvals_large)
            
            print(f"Smallest eigenvalue (approx): {lambda_min:.6e}")
            print(f"Largest eigenvalue (approx): {lambda_max:.6e}")
            
            if abs(lambda_min) > 1e-10:
                condition_number = abs(lambda_max / lambda_min)
                print(f"Condition number (approx): {condition_number:.2e}")
            else:
                print("Warning: Matrix appears singular or near-singular!")
                condition_number = np.inf
            
            return {
                'lambda_min': lambda_min,
                'lambda_max': lambda_max,
                'condition_number': condition_number,
                'eigenvals_small': eigenvals_small,
                'eigenvals_large': eigenvals_large
            }
        except Exception as e:
            print(f"Error computing eigenvalues: {e}")
            return None
    
    def analyze_eigenvalue_distribution(self, num_eigenvals=50):
        """Analyze eigenvalue distribution"""
        print("\n" + "="*60)
        print("EIGENVALUE DISTRIBUTION ANALYSIS")
        print("="*60)
        
        A = self.matrix
        # Adaptive number of eigenvalues based on matrix size
        num_eigenvals = min(num_eigenvals, max(10, A.shape[0] // 1000))
        num_eigenvals = min(num_eigenvals, A.shape[0] - 1)
        
        print(f"Computing {num_eigenvals} eigenvalues...")
        try:
            eigenvals, _ = eigsh(A, k=num_eigenvals, which='LM', maxiter=1000, tol=1e-4)
            eigenvals = np.sort(eigenvals)
            
            print(f"\nEigenvalue statistics:")
            print(f"  Min: {np.min(eigenvals):.6e}")
            print(f"  Max: {np.max(eigenvals):.6e}")
            print(f"  Mean: {np.mean(eigenvals):.6e}")
            print(f"  Median: {np.median(eigenvals):.6e}")
            print(f"  Std: {np.std(eigenvals):.6e}")
            
            # Check if all eigenvalues are positive (positive definiteness)
            if np.all(eigenvals > 0):
                print("\nMatrix appears POSITIVE DEFINITE (all eigenvalues > 0)")
            elif np.all(eigenvals >= 0):
                print("\nMatrix appears POSITIVE SEMI-DEFINITE (all eigenvalues >= 0)")
            else:
                print(f"\nMatrix has negative eigenvalues (min: {np.min(eigenvals):.6e})")
            
            return eigenvals
        except Exception as e:
            print(f"Error computing eigenvalues: {e}")
            return None
    
    def analyze_sparsity_pattern(self):
        """Analyze sparsity pattern"""
        print("\n" + "="*60)
        print("SPARSITY PATTERN ANALYSIS")
        print("="*60)
        
        A = self.matrix
        num_nodes = A.shape[0]
        
        # Nonzeros per row
        nnz_per_row = A.getnnz(axis=1)
        
        print(f"Nonzeros per row statistics:")
        print(f"  Mean: {np.mean(nnz_per_row):.2f}")
        print(f"  Median: {np.median(nnz_per_row):.2f}")
        print(f"  Min: {np.min(nnz_per_row)}")
        print(f"  Max: {np.max(nnz_per_row)}")
        print(f"  Std: {np.std(nnz_per_row):.2f}")
        
        # Average number of neighbors
        print(f"\nAverage neighbors per node: {np.mean(nnz_per_row) - 1:.2f}")
        
        return {'nnz_per_row': nnz_per_row}
    
    def analyze_vector_statistics(self):
        """Analyze divergence and pressure vector statistics"""
        print("\n" + "="*60)
        print("VECTOR STATISTICS ANALYSIS")
        print("="*60)
        
        if self.divergence is None:
            print("Divergence vector not loaded")
            return None
        
        stats = {}
        
        # Divergence statistics (the "b" vector in Ax=b)
        print("\nDivergence (Source Term) Statistics:")
        stats['div_min'] = np.min(self.divergence)
        stats['div_max'] = np.max(self.divergence)
        stats['div_mean'] = np.mean(self.divergence)
        stats['div_std'] = np.std(self.divergence)
        stats['div_l2'] = np.linalg.norm(self.divergence)
        stats['div_rms'] = stats['div_l2'] / np.sqrt(len(self.divergence))
        
        print(f"  Range:  {stats['div_min']:.4e} to {stats['div_max']:.4e}")
        print(f"  Mean:   {stats['div_mean']:.4e}")
        print(f"  Std:    {stats['div_std']:.4e}")
        print(f"  L2 norm: {stats['div_l2']:.4e}")
        print(f"  RMS:    {stats['div_rms']:.4e}")
        
        # Pressure statistics (the "x" vector)
        if self.pressure is not None:
            print("\nPressure (Solution) Statistics:")
            stats['press_min'] = np.min(self.pressure)
            stats['press_max'] = np.max(self.pressure)
            stats['press_mean'] = np.mean(self.pressure)
            stats['press_std'] = np.std(self.pressure)
            stats['press_l2'] = np.linalg.norm(self.pressure)
            
            print(f"  Range:  {stats['press_min']:.4e} to {stats['press_max']:.4e}")
            print(f"  Mean:   {stats['press_mean']:.4e}")
            print(f"  Std:    {stats['press_std']:.4e}")
            print(f"  L2 norm: {stats['press_l2']:.4e}")
        else:
            print("\nPressure vector not loaded")
        
        return stats
    
    def analyze_isolated_nodes(self):
        """Detect isolated/orphan nodes (nodes with no valid neighbors)"""
        print("\n" + "="*60)
        print("ISOLATED NODE ANALYSIS")
        print("="*60)
        
        if self.neighbors is None:
            print("Neighbors not loaded")
            return None
        
        num_nodes = self.num_nodes
        
        # Count valid neighbors (neighbors that are within valid node range)
        # Boundary markers: numNodes = Dirichlet, numNodes+1 = Neumann
        valid_mask = (self.neighbors >= 0) & (self.neighbors < num_nodes)
        neighbor_counts = np.sum(valid_mask, axis=1)
        
        isolated_nodes = np.sum(neighbor_counts == 0)
        low_connectivity = np.sum(neighbor_counts < 3)  # Less than 3 neighbors
        
        print(f"Total nodes: {num_nodes}")
        print(f"Isolated nodes (0 neighbors): {isolated_nodes} ({isolated_nodes/num_nodes*100:.2f}%)")
        print(f"Low connectivity (<3 neighbors): {low_connectivity} ({low_connectivity/num_nodes*100:.2f}%)")
        print(f"Average neighbors per node: {np.mean(neighbor_counts):.2f}")
        print(f"Min neighbors: {np.min(neighbor_counts)}")
        print(f"Max neighbors: {np.max(neighbor_counts)}")
        
        if isolated_nodes > 0:
            print(f"\nWarning: {isolated_nodes} isolated nodes detected.")
            print("These may be inactive/air nodes or indicate data issues.")
        
        return {
            'isolated_nodes': isolated_nodes,
            'low_connectivity': low_connectivity,
            'neighbor_counts': neighbor_counts,
            'avg_neighbors': np.mean(neighbor_counts),
            'min_neighbors': np.min(neighbor_counts),
            'max_neighbors': np.max(neighbor_counts)
        }
    
    def estimate_residual(self):
        """
        Estimate residual r = b - Ax using simplified Laplacian assumption.
        This gives a rough convergence check without full matrix-vector multiply.
        """
        print("\n" + "="*60)
        print("RESIDUAL ESTIMATION")
        print("="*60)
        
        if self.pressure is None or self.divergence is None or self.neighbors is None:
            print("Required data not loaded (pressure, divergence, or neighbors)")
            return None
        
        num_nodes = self.num_nodes
        
        # Count valid neighbors
        valid_mask = (self.neighbors >= 0) & (self.neighbors < num_nodes)
        neighbor_counts = np.sum(valid_mask, axis=1)
        
        # Create padded pressure array to handle boundary indices safely
        padded_press = np.append(self.pressure, 0.0)
        safe_neighbors = self.neighbors.copy()
        # Replace invalid neighbors with index pointing to padding (last element)
        safe_neighbors[~valid_mask] = len(self.pressure)  # Points to the 0.0 padding value
        
        # Compute sum of neighbor pressures for each node
        # This is a simplified approximation assuming uniform weights
        # Only sum valid neighbors
        neighbor_press = padded_press[safe_neighbors]
        sum_neighbor_press = np.sum(neighbor_press * valid_mask, axis=1)
        
        # Simplified Laplacian: A_ii * p_i - sum(p_neighbors) ≈ divergence
        # For standard Poisson, A_ii ≈ neighbor_count
        diagonal = neighbor_counts.astype(np.float32)
        
        # Compute approximate LHS: diagonal * pressure - sum(neighbor pressures)
        lhs = diagonal * self.pressure - sum_neighbor_press
        
        # Residual = LHS - divergence
        residual = lhs - self.divergence
        
        stats = {
            'residual_mean': np.mean(np.abs(residual)),
            'residual_max': np.max(np.abs(residual)),
            'residual_l2': np.linalg.norm(residual),
            'residual_rms': np.linalg.norm(residual) / np.sqrt(num_nodes)
        }
        
        print(f"Estimated residual statistics (simplified Laplacian assumption):")
        print(f"  Mean |residual|: {stats['residual_mean']:.4e}")
        print(f"  Max |residual|:  {stats['residual_max']:.4e}")
        print(f"  L2 norm:         {stats['residual_l2']:.4e}")
        print(f"  RMS:              {stats['residual_rms']:.4e}")
        
        if stats['residual_mean'] > 1.0:
            print("\nNote: High residual estimate suggests:")
            print("  - Matrix contains non-uniform weights (T-junction coefficients)")
            print("  - Divergence units may be scaled differently")
            print("  - This is a simplified estimate; actual residual requires full A*x")
        elif stats['residual_mean'] < 1e-6:
            print("\nNote: Very low residual suggests good convergence.")
        
        return stats
    
    def analyze_sliding_window(self, max_window_size=None):
        """
        Optimized sliding window analysis using CDF approach.
        Analyzes matrix ordering quality - well-ordered matrices have neighbors close together.
        """
        print("\n" + "="*60)
        print("SLIDING WINDOW ANALYSIS")
        print("="*60)
        
        if self.neighbors is None:
            print("Neighbors not loaded")
            return None
        
        num_nodes = self.num_nodes
        
        # Create an index array [0, 1, 2, ... N] for vectorized distance calc
        node_indices = np.arange(num_nodes).reshape(num_nodes, 1)
        
        # Mask invalid neighbors
        # Neighbors >= num_nodes are boundaries (Dirichlet/Neumann)
        # Neighbors < 0 are empty/padding
        valid_mask = (self.neighbors >= 0) & (self.neighbors < num_nodes)
        
        # Count connectivity
        neighbor_counts = np.sum(valid_mask, axis=1)
        avg_neighbors = np.mean(neighbor_counts)
        isolated_nodes = np.sum(neighbor_counts == 0)
        boundary_nodes = np.sum((neighbor_counts > 0) & (neighbor_counts < 6))
        
        print(f"Average Neighbors per Row: {avg_neighbors:.2f}")
        print(f"Isolated Nodes (Pure Diagonal): {isolated_nodes} ({(isolated_nodes/num_nodes)*100:.2f}%)")
        print(f"Boundary/Irregular Nodes (<6 neighbors): {boundary_nodes} ({(boundary_nodes/num_nodes)*100:.2f}%)")
        print(f"  -> These 'Exceptions' break the standard diagonal bands but improve condition number (Dirichlet).")
        
        # Calculate distance |neighbor_index - self_index|
        # Set invalid neighbors to the node's own index so distance is 0
        safe_neighbors = self.neighbors.copy()
        # Broadcast node indices to match neighbors shape and set invalid ones
        node_indices_broadcast = np.broadcast_to(node_indices, safe_neighbors.shape)
        safe_neighbors[~valid_mask] = node_indices_broadcast[~valid_mask]
        
        distances = np.abs(safe_neighbors - node_indices_broadcast)
        
        # The "Bandwidth" required for a node is the MAX distance to any of its neighbors
        # If we want a window size W that covers ALL neighbors for this node, W must be >= max_dist
        max_dist_per_node = np.max(distances, axis=1)
        
        # OPTIMIZATION: Sort the distances to create a CDF (Cumulative Distribution Function)
        # This replaces the slow loop over window sizes.
        sorted_bandwidths = np.sort(max_dist_per_node)
        
        # The Y-axis is simply the percentile (0 to 100)
        # The X-axis is the Window Size (sorted_bandwidths)
        y_percentiles = np.arange(len(sorted_bandwidths)) / max(1, len(sorted_bandwidths) - 1) * 100
        
        # Calculate key statistics
        p90_idx = np.searchsorted(y_percentiles, 90)
        p90_val = sorted_bandwidths[p90_idx] if p90_idx < len(sorted_bandwidths) else sorted_bandwidths[-1]
        
        p99_idx = np.searchsorted(y_percentiles, 99)
        p99_val = sorted_bandwidths[p99_idx] if p99_idx < len(sorted_bandwidths) else sorted_bandwidths[-1]
        
        print(f"\nKey Statistics:")
        print(f"  Window size for 90% capture: {p90_val}")
        print(f"  Window size for 99% capture: {p99_val}")
        print(f"  Max Bandwidth (100%):        {sorted_bandwidths[-1]}")
        print(f"  Mean max distance:           {np.mean(max_dist_per_node):.2f}")
        print(f"  Median max distance:         {np.median(max_dist_per_node):.2f}")
        
        # Analyze power-of-2 window sizes
        print(f"\n{'='*60}")
        print("POWER-OF-2 WINDOW SIZE ANALYSIS")
        print(f"{'='*60}")
        max_window_needed = int(np.ceil(sorted_bandwidths[-1]))
        
        # Generate power-of-2 window sizes up to max_window_needed
        power_of_2_windows = []
        window_size = 1
        while window_size <= max_window_needed:
            power_of_2_windows.append(window_size)
            window_size *= 2
        
        # Also include common sizes like 256, 512 if they're relevant
        if 256 <= max_window_needed and 256 not in power_of_2_windows:
            power_of_2_windows.append(256)
        if 512 <= max_window_needed and 512 not in power_of_2_windows:
            power_of_2_windows.append(512)
        if 1024 <= max_window_needed and 1024 not in power_of_2_windows:
            power_of_2_windows.append(1024)
        
        power_of_2_windows = sorted(set(power_of_2_windows))
        
        print(f"\n{'Window Size':<15} {'Nodes In':<15} {'Nodes Out':<15} {'Percent Captured':<20}")
        print(f"{'-'*15} {'-'*15} {'-'*15} {'-'*20}")
        
        for window_size in power_of_2_windows:
            nodes_in = np.sum(max_dist_per_node <= window_size)
            nodes_out = num_nodes - nodes_in
            percent_captured = (nodes_in / num_nodes) * 100.0
            print(f"{window_size:<15} {nodes_in:<15} {nodes_out:<15} {percent_captured:<20.2f}%")
        
        results = {
            'window_sizes': sorted_bandwidths,
            'percentages': y_percentiles,
            'max_distances': max_dist_per_node,
            'p90_window': p90_val,
            'p99_window': p99_val,
            'max_window': sorted_bandwidths[-1],
            'power_of_2_analysis': {
                'window_sizes': power_of_2_windows,
                'nodes_in': [np.sum(max_dist_per_node <= w) for w in power_of_2_windows],
                'nodes_out': [num_nodes - np.sum(max_dist_per_node <= w) for w in power_of_2_windows],
                'percent_captured': [(np.sum(max_dist_per_node <= w) / num_nodes) * 100.0 for w in power_of_2_windows]
            }
        }
        
        return results
    
    def visualize_sliding_window(self, results):
        """Visualize sliding window analysis"""
        if results is None:
            return
        
        sorted_bw = results['window_sizes']
        y_perc = results['percentages']
        p90_val = results.get('p90_window', None)
        p99_val = results.get('p99_window', None)
        
        plt.figure(figsize=(10, 6))
        plt.plot(sorted_bw, y_perc, linewidth=2, label='Node Coverage')
        
        # Add reference lines
        if p90_val is not None:
            plt.axvline(x=p90_val, color='r', linestyle='--', alpha=0.7, label=f'90% Coverage (Window: {p90_val})')
        
        if p99_val is not None:
            plt.axvline(x=p99_val, color='g', linestyle='--', alpha=0.7, label=f'99% Coverage (Window: {p99_val})')
        
        plt.title("Preconditioning Window Analysis\n(Percent of Nodes fully contained within Window Size)", fontsize=12, fontweight='bold')
        plt.xlabel("Window Size (Index Distance)", fontsize=11)
        plt.ylabel("% of Nodes Fully Covered", fontsize=11)
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        
        plt.show()
        plt.close()
    
    def analyze_numnodes_distribution(self):
        """
        Analyze numNodes across all frames and create a histogram showing frequency distribution.
        """
        print("\n" + "="*60)
        print("NUMNODES DISTRIBUTION ANALYSIS")
        print("="*60)
        
        # Find all frame directories
        frame_dirs = sorted([d for d in self.data_folder.iterdir() 
                            if d.is_dir() and d.name.startswith('frame_')])
        
        if not frame_dirs:
            print("Error: No frame directories found")
            return None
        
        print(f"Found {len(frame_dirs)} frames")
        print("Loading numNodes from all frames...")
        
        num_nodes_list = []
        frame_indices = []
        
        for frame_dir in frame_dirs:
            try:
                # Extract frame index from directory name
                frame_name = frame_dir.name
                frame_index = int(frame_name.split('_')[1])
                frame_indices.append(frame_index)
                
                # Load metadata to get numNodes
                meta_file = frame_dir / "meta.txt"
                if meta_file.exists():
                    metadata = {}
                    with open(meta_file, 'r') as f:
                        for line in f:
                            if ':' in line:
                                key, value = line.strip().split(':', 1)
                                key = key.strip()
                                value = value.strip()
                                
                                if key == 'numNodes':
                                    try:
                                        num_nodes = int(value)
                                        num_nodes_list.append(num_nodes)
                                        break
                                    except ValueError:
                                        pass
            except Exception as e:
                print(f"Warning: Could not load frame {frame_dir.name}: {e}")
                continue
        
        if not num_nodes_list:
            print("Error: No numNodes data found")
            return None
        
        num_nodes_array = np.array(num_nodes_list)
        frame_indices_array = np.array(frame_indices)
        
        print(f"\nLoaded {len(num_nodes_list)} frames")
        print(f"numNodes statistics:")
        print(f"  Min:    {np.min(num_nodes_array)}")
        print(f"  Max:    {np.max(num_nodes_array)}")
        print(f"  Mean:   {np.mean(num_nodes_array):.2f}")
        print(f"  Median: {np.median(num_nodes_array):.2f}")
        print(f"  Std:    {np.std(num_nodes_array):.2f}")
        
        return {
            'num_nodes': num_nodes_array,
            'frame_indices': frame_indices_array,
            'stats': {
                'min': np.min(num_nodes_array),
                'max': np.max(num_nodes_array),
                'mean': np.mean(num_nodes_array),
                'median': np.median(num_nodes_array),
                'std': np.std(num_nodes_array)
            }
        }
    
    def visualize_numnodes_distribution(self, results, save_path=None):
        """Visualize numNodes vs frequency histogram"""
        if results is None:
            return
        
        num_nodes = results['num_nodes']
        stats = results['stats']
        
        # Create figure with two subplots: histogram and time series
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Histogram: numNodes vs frequency
        ax1.hist(num_nodes, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax1.axvline(stats['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {stats['mean']:.0f}")
        ax1.axvline(stats['median'], color='green', linestyle='--', linewidth=2, label=f"Median: {stats['median']:.0f}")
        ax1.set_xlabel('numNodes', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Frequency (Number of Frames)', fontsize=12, fontweight='bold')
        ax1.set_title('numNodes Distribution Across All Frames', fontsize=14, fontweight='bold')
        ax1.grid(True, which='both', linestyle='--', alpha=0.5)
        ax1.legend()
        
        # Time series: numNodes over frame index
        frame_indices = results['frame_indices']
        ax2.plot(frame_indices, num_nodes, 'b-', linewidth=1.5, alpha=0.7)
        ax2.axhline(stats['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {stats['mean']:.0f}")
        ax2.set_xlabel('Frame Index', fontsize=12, fontweight='bold')
        ax2.set_ylabel('numNodes', fontsize=12, fontweight='bold')
        ax2.set_title('numNodes Over Time (Frame Index)', fontsize=14, fontweight='bold')
        ax2.grid(True, which='both', linestyle='--', alpha=0.5)
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_matrix(self, save_path=None, max_size=1000):
        """Visualize the matrix sparsity pattern"""
        print("\n" + "="*60)
        print("VISUALIZING MATRIX SPARSITY PATTERN")
        print("="*60)
        
        A = self.matrix
        num_nodes = A.shape[0]
        
        # Downsample if too large
        if num_nodes > max_size:
            print(f"Downsampling matrix from {num_nodes} to {max_size} for visualization")
            indices = np.linspace(0, num_nodes-1, max_size, dtype=int)
            A_vis = A[np.ix_(indices, indices)]
        else:
            A_vis = A
        
        plt.figure(figsize=(12, 10))
        plt.spy(A_vis, markersize=0.5, precision=1e-10)
        plt.title(f'Matrix Sparsity Pattern (n={A.shape[0]}, nnz={A.nnz})')
        plt.xlabel('Column index')
        plt.ylabel('Row index')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def full_analysis(self, frame_index, visualize=True):
        """Run full analysis on a frame"""
        # Load data
        self.load_frame(frame_index)
        
        # Reconstruct matrix
        self.reconstruct_matrix()
        
        # Run all analyses
        results = {}
        results['vector_stats'] = self.analyze_vector_statistics()
        results['isolated'] = self.analyze_isolated_nodes()
        results['residual'] = self.estimate_residual()
        results['banding'] = self.analyze_banding()
        results['symmetry'] = self.analyze_symmetry()
        results['diagonal_dominance'] = self.analyze_diagonal_dominance()
        results['sparsity'] = self.analyze_sparsity_pattern()
        
        # These are computationally expensive, so run them last
        # Skip for very large matrices to avoid memory issues
        if self.num_nodes < 100000:
            print("\n" + "="*60)
            print("Running expensive eigenvalue computations...")
            print("="*60)
            try:
                results['condition'] = self.analyze_condition_number()
            except Exception as e:
                print(f"Warning: Could not compute condition number: {e}")
                results['condition'] = None
            
            try:
                results['eigenvalues'] = self.analyze_eigenvalue_distribution()
            except Exception as e:
                print(f"Warning: Could not compute eigenvalue distribution: {e}")
                results['eigenvalues'] = None
        else:
            print("\n" + "="*60)
            print("Skipping eigenvalue computations (matrix too large)")
            print("="*60)
            results['condition'] = None
            results['eigenvalues'] = None
        
        # Sliding window analysis
        print("\n" + "="*60)
        print("Running sliding window analysis...")
        print("="*60)
        results['sliding_window'] = self.analyze_sliding_window()
        
        # Visualize
        if visualize:
            vis_path = self.data_folder / f"matrix_visualization_frame_{frame_index:04d}.png"
            self.visualize_matrix(save_path=vis_path)
        
        # Always visualize sliding window (it's fast and informative)
        if results['sliding_window'] is not None:
            self.visualize_sliding_window(results['sliding_window'])
        
        return results


def main():
    import sys
    import traceback
    
    try:
        # Find the simulation data folder
        script_dir = Path(__file__).parent
        assets_dir = script_dir.parent
        data_folder = assets_dir / "StreamingAssets" / "SimulationData" / "Run_2025-11-23_21-59-47"
        
        if not data_folder.exists():
            print(f"Error: Data folder not found: {data_folder}")
            return
        
        # Find middle frame
        frame_dirs = sorted([d for d in data_folder.iterdir() if d.is_dir() and d.name.startswith('frame_')])
        if not frame_dirs:
            print("Error: No frame directories found")
            return
        
        # Get middle frame
        middle_frame_idx = len(frame_dirs) // 2
        frame_name = frame_dirs[middle_frame_idx].name
        frame_index = int(frame_name.split('_')[1])
        
        print(f"Analyzing frame {frame_index} (middle frame)")
        
        # Create analyzer
        analyzer = MatrixAnalyzer(data_folder)
        
        # First, analyze numNodes distribution across all frames
        print("\n" + "="*60)
        print("ANALYZING NUMNODES DISTRIBUTION")
        print("="*60)
        numnodes_results = analyzer.analyze_numnodes_distribution()
        if numnodes_results:
            analyzer.visualize_numnodes_distribution(numnodes_results, save_path=None)
        
        # Then run full analysis on middle frame
        results = analyzer.full_analysis(frame_index, visualize=False)  # Disable visualization for now
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print("\nSummary:")
        print(f"  Matrix size: {analyzer.matrix.shape[0]} x {analyzer.matrix.shape[1]}")
        print(f"  Nonzeros: {analyzer.matrix.nnz}")
        print(f"  Sparsity: {(1 - analyzer.matrix.nnz / (analyzer.matrix.shape[0] * analyzer.matrix.shape[1])) * 100:.2f}%")
        
        if results['condition']:
            print(f"  Condition number: {results['condition']['condition_number']:.2e}")
        
        if results['symmetry']['is_symmetric']:
            print("  Matrix is symmetric")
        else:
            print(f"  Matrix is NOT symmetric (max diff: {results['symmetry']['max_diff']:.2e})")
            print("    Note: CG solver typically requires symmetric matrices.")
            print("    Consider using MINRES or GMRES for non-symmetric matrices.")
        
        print(f"\nKey findings for solver optimization:")
        print(f"  - Bandwidth: {results['banding']['total_bandwidth']} ({results['banding']['total_bandwidth']/analyzer.matrix.shape[0]*100:.1f}% of matrix size)")
        print(f"  - Diagonal dominance: {results['diagonal_dominance']['num_strictly_dd']}/{analyzer.matrix.shape[0]} rows ({results['diagonal_dominance']['num_strictly_dd']/analyzer.matrix.shape[0]*100:.1f}%)")
        if results['condition']:
            print(f"  - Condition number: {results['condition']['condition_number']:.2e}")
            if results['condition']['condition_number'] > 1000:
                print("    Warning: High condition number may slow convergence. Consider preconditioning.")
        
        print(f"\nAnalysis complete!")
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

