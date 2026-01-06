# Neural-Accelerated Fluid Simulation: Sparse Approximate Inverse Preconditioning on GPU

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Unity](https://img.shields.io/badge/Unity-2022.3%2B-black)](https://unity.com/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

A high-performance, large-scale fluid simulation framework implementing a novel **Learning-based Sparse Approximate Inverse (SPAI)** preconditioner for the Conjugate Gradient pressure solver.

This project demonstrates a hybrid offline/online architecture: a Python-based neural network learns the inverse topology of the Laplacian operator on a sparse octree, and a Unity/HLSL simulation applies this learned preconditioner in real-time. The result is a simulation capable of handling **1,048,576 particles** with significantly reduced convergence times compared to standard industry baselines.

## Performance Benchmarks

All statistics represent a stress-test scenario with **1,048,576 particles**, using a sparse octree dynamic grid ranging from **Level 4 ($2^4$) to Level 10 ($2^{10}$)** resolution.

### 1. Neural vs. Jacobi (Standard GPU Baseline)
*The Jacobi preconditioner is the standard for GPU-based CG solvers due to its parallel nature. The Neural SPAI outperforms it by capturing non-diagonal dependencies while maintaining parallelism.*

| Metric | Reduction | Value Change |
|--------|-----------|--------------|
| **CG Iterations** | **44.6%** | 56 $\to$ 31 |
| **Solve Time** | **17.9%** | 28ms $\to$ 23ms |
| **Total Frame Time** | **11.3%** | 53ms $\to$ 47ms |

### 2. Neural vs. None (Raw CG Solver)
*Highlights the raw efficiency gain of the neural preconditioner against an unconditioned solver.*

| Metric | Reduction | Value Change |
|--------|-----------|--------------|
| **CG Iterations** | **74.6%** | 122 $\to$ 31 |
| **Solve Time** | **75.0%** | 92ms $\to$ 23ms |
| **Total Frame Time** | **54.4%** | 103ms $\to$ 47ms |

---

## Core Technical Innovations

### 1. Neural SPAI Preconditioner (Transformer on GPU)

Traditional preconditioners like Incomplete Cholesky (ICC) are effective but inherently sequential (solving triangular systems), making them poor candidates for massive GPU parallelism.

This project implements a **Sparse Approximate Inverse (SPAI)** approach. Instead of solving $Mz = r$, we approximate the inverse matrix $G \approx A^{-1}$ directly using a neural network.
* **Architecture:** A 1D Windowed Transformer (implemented in `NeuralPreconditioner.py`) that processes the linearized Morton-coded octree.
* **Inference:** The network weights are exported to compute buffers. `Preconditioner.compute` runs a custom **fused Transformer layer** directly in HLSL, performing Attention and Feed-Forward steps to generate the preconditioner matrix $G$ on the fly.
* **Math:** The network minimizes the Frobenius norm $\|I - GA\|_F$. Applying the preconditioner becomes a simple Matrix-Vector multiplication ($z = Gr$), which is perfectly parallelizable and avoids the "wavefront" serialization of triangular solves.

### 2. Pointer-less Sparse Octree & Morton Coding

To manage $10^6$ particles without the overhead of pointer-chasing, the simulation utilizes a **linearized sparse octree**.
* **Spatial Hashing:** Particles are hashed using 30-bit Morton Codes (Z-order curve), interleaving bits of X, Y, and Z coordinates.
* **Physical Layout:** `Nodes.compute` and `RadixSort.compute` organize data such that spatially adjacent nodes are likely adjacent in memory.
* **Resolution:** The grid adapts dynamically, allocating high-resolution cells (Level 10) only near the fluid interface, while using coarse cells (Level 4) for the interior/bulk, optimizing the degree of freedom for the pressure solve.

### 3. PIC/FLIP Fluid Solver
The simulation uses a hybrid **Particle-In-Cell (PIC)** and **Fluid-Implicit-Particle (FLIP)** method to solve the Incompressible Euler equations.
* **Advection:** Lagrangian particle advection prevents numerical dissipation.
* **Grid Projection:** Particle velocities are scattered to the sparse octree leaves using a tri-linear interpolation kernel.
* **Pressure Projection:** The core bottleneck. We solve $\nabla \cdot (\nabla p) = \nabla \cdot u^*$ (Poisson equation) using the Preconditioned Conjugate Gradient (PCG) method.
* **Update:** The calculated pressure gradient corrects the velocity field, ensuring the fluid remains divergence-free (incompressible).

---

## Implementation Details

### Kernel Memory Access & Coalescing
A major focus of the `CGSolver.compute` implementation is memory coalescing. Because the nodes are sorted via Radix Sort (Morton order) prior to the solve:
1.  **Cache Hit Rate:** Neighboring cells in 3D space are often neighbors in the 1D buffer. When a thread calculates the Laplacian for node $i$, the data for neighbors ($i \pm 1, i \pm \text{row}$) is likely already in the L2 cache.
2.  **Divergence:** Warp divergence is minimized because active fluid nodes are packed contiguously in the `nodesBuffer`.

### The Custom Radix Sort (`RadixSort.compute`)
Unity's default sorting is insufficient for this data scale. A custom **GPU Radix Sort** was implemented:
* **Algorithm:** Least Significant Digit (LSD) Radix Sort.
* **Prefix Sums:** Uses a hierarchical scan (Hillis-Steele) to calculate offsets for parallel scattering.
* **Throughput:** capable of sorting 1M+ particle keys per frame to rebuild the octree structure dynamically every timestep.

### The Physics-Informed Loss Function
The training script (`NeuralPreconditioner.py`) does not just learn a mapping; it enforces physical consistency.
* **Input:** The local stencil of the Laplacian matrix $A$ (derived from grid topology).
* **Output:** The non-zero coefficients of the inverse factor $G$.
* **Loss:** Unsupervised Scale-Invariant Aligned Identity (SAI) Loss via Stochastic Trace Estimation. It trains the network to produce a $G$ such that $G A \approx I$, without needing expensive ground-truth matrix inversions.

## Project Structure

* **`FluidSimulator.cs`**: The CPU-side orchestrator. Manages compute buffer dispatch, timestep logic, and data transfer between the C# simulation and the Python training loop.
* **`CGSolver.compute`**: The heavy lifter. Contains the matrix-free Laplacian operator, the Dot Product reduction kernels, and the PCG iteration loop.
* **`Preconditioner.compute`**: The neural inference engine written in HLSL. Loads trained weights and predicts the preconditioner matrix per-node.
* **`Particles.compute`**: Handles the Lagrangian particle integration, wall boundary conditions, and advection.
* **`NeuralPreconditioner.py`**: PyTorch implementation of the Transformer model. Handles data loading from the simulation and exporting weights back to Unity.

## Summary

This project represents a deep dive into **GPGPU optimization** and **Scientific Machine Learning (SciML)**.

**Key Technical Skills Demonstrated:**
* **Advanced HLSL:** Implementing Transformers and Conjugate Gradient solvers in raw compute shaders.
* **Memory Optimization:** Designing data structures (Morton-ordered linear trees) specifically for GPU cache hierarchy.
* **Numerical Methods:** Implementation of PIC/FLIP and PCG solvers.
* **Neural Systems:** Bridging the gap between offline PyTorch training and real-time Unity inference.

---
*This work corresponds to the implementation files found in the `scripts` folder.*