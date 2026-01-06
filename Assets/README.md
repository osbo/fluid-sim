# Neural-Accelerated Fluid Simulation: Sparse Approximate Inverse Preconditioning on GPU

**Abstract:** This project implements a high-performance Eulerian-Lagrangian fluid simulation for incompressible flows, targeting massive particle counts ($N > 10^6$) on consumer hardware. The core contribution is a hybrid numerical-neural solver that replaces standard preconditioners (e.g., Incomplete Cholesky, Jacobi) with a learned Sparse Approximate Inverse (SPAI). By training a Transformer-based architecture to predict the inverse topology of the discrete Laplacian on a linearized sparse octree, we achieve a significant reduction in convergence time for the Conjugate Gradient (CG) pressure projection.

## 1. Performance Analysis

The following benchmarks were conducted on a stress-test scenario involving **1,048,576 particles**, utilizing a dynamic sparse octree with effective resolutions ranging from **Level 4 ($2^4$)** in the bulk fluid to **Level 10 ($2^{10}$)** at the interface.

### 1.1 Neural SPAI vs. Jacobi (Standard GPU Baseline)
The Jacobi preconditioner ($M = \text{diag}(A)$) is the industry standard for GPU-based CG solvers due to its trivial parallelization. The Neural SPAI demonstrates superior convergence properties by capturing off-diagonal spectral dependencies while maintaining $O(1)$ parallel inference cost.

| Metric | Reduction | Value Change |
|--------|-----------|--------------|
| **CG Iterations** | **44.6%** | $56 \to 31$ |
| **Solve Time** | **17.9%** | $28\text{ms} \to 23\text{ms}$ |
| **Total Frame Time** | **11.3%** | $53\text{ms} \to 47\text{ms}$ |

### 1.2 Neural SPAI vs. Unconditioned CG
This comparison isolates the efficiency of the learned preconditioner against the raw Krylov subspace method.

| Metric | Reduction | Value Change |
|--------|-----------|--------------|
| **CG Iterations** | **74.6%** | $122 \to 31$ |
| **Solve Time** | **75.0%** | $92\text{ms} \to 23\text{ms}$ |
| **Total Frame Time** | **54.4%** | $103\text{ms} \to 47\text{ms}$ |

---

## 2. Algorithmic Methodology

The simulation employs a standard **PIC/FLIP (Particle-In-Cell / Fluid-Implicit-Particle)** discretization. The governing equations are the incompressible Euler equations, split into advection and projection steps.

### 2.1 Adaptive Sparse Octree & Spatial Hashing
To handle the domain $\Omega$, we avoid pointer-based tree structures which induce cache-thrashing on GPUs. Instead, we utilize a **linearized octree** based on Z-order curves (Morton Codes).

* **Initialization:** Particles are hashed into 30-bit Morton codes. A custom **GPU Radix Sort** (`RadixSort.compute`) arranges particles to ensure spatial locality corresponds to memory locality.
* **Adaptive Resolution (Level Set):** The grid resolution is not uniform. We employ a distance-based refinement strategy:
    * **Level 10 ($2^{10}$):** Allocated strictly at the fluid interface (Zero Level Set), maximizing detail where surface tension and visual artifacts are most prominent.
    * **Level 4 ($2^4$):** Allocated for the deep bulk fluid.
    * **Grading:** Resolution degrades based on the Manhattan distance from the surface.
* **2:1 Balance Constraint:** To ensure numerical stability in the finite difference stencil, we enforce a strict 2:1 balance. No cell is allowed to be adjacent to a neighbor more than one level coarser or finer. This simplifies the T-junction interpolation during the Laplacian construction.

### 2.2 Staggered Grid & Velocity Interpolation
The simulation utilizes a MAC (Marker-and-Cell) grid arrangement.
* **Dual Representation:** Velocity is stored on particles (Lagrangian) and grid faces (Eulerian).
* **Face Velocity Sharing:** The `Node` struct contains storage for 6 faces. During the `ProcessNodes` kernel, velocity is scattered from particles to leaves. Critically, to enforce continuity, shared faces between adjacent nodes (e.g., Node $i$'s right face and Node $j$'s left face) are averaged.
* **Divergence:** The divergence $\nabla \cdot \mathbf{u}$ is computed on this staggered grid. The 2:1 constraint ensures that flux computations across resolution changes (fine-to-coarse boundaries) remain conservative.

### 2.3 Boundary Conditions
* **Solid Walls:** Handled via a discrete epsilon-buffer method in `Particles.compute`. Particles approaching the domain boundary ($x < 0$ or $x > 1024$) are reflected with a restitution coefficient (bounce damping) of 0.3. A spatial buffer of $\epsilon = 0.05$ prevents particles from becoming "stuck" in the zero-velocity wall condition during interpolation.
* **Pressure Solve:**
    * **Free Surface:** Dirichlet boundary condition ($p=0$) is applied at the interface (ghost nodes).
    * **Solid Boundaries:** Neumann boundary condition ($\frac{\partial p}{\partial n} = 0$) is enforced by modifying the matrix diagonal in `CGSolver.compute`.

---

## 3. The Neural Preconditioner

The bottleneck of incompressible fluid simulation is solving the Poisson equation $A p = \mathbf{d}$, where $A$ is the sparse, symmetric positive-definite Laplacian matrix. We propose a learning-based alternative to incomplete factorizations.

### 3.1 Sparse Approximate Inverse (SPAI)
Instead of solving a linear system for the preconditioner step (e.g., $Mz=r$), we approximate the inverse matrix explicitly: $G \approx A^{-1}$.
The preconditioner application becomes a simple sparse matrix-vector multiplication (SpMV): $z = G r$.

### 3.2 Transformer Architecture (`NeuralPreconditioner.py`)
* **Input:** The local geometric topology of the octree. The network receives a sequence of Morton codes and level information, representing the linearized tree.
* **Model:** A 1D Windowed Transformer. The attention mechanism effectively captures the non-local pressure propagation characteristics of the fluid.
* **Loss Function:** We utilize an unsupervised **Scale-Invariant Aligned Identity (SAI) Loss** via Stochastic Trace Estimation. The network minimizes the Frobenius norm $\| I - GA \|_F$ without requiring ground-truth inverses, making training tractable for large systems.

### 3.3 GPU Inference (`Preconditioner.compute`)
The trained weights are exported to compute buffers. The inference is implemented as a **Fused Transformer Layer** in HLSL. This custom kernel performs the Attention and Feed-Forward steps directly on the GPU, avoiding CPU-GPU synchronization latency and allowing the preconditioner to adapt dynamically to the changing grid topology every frame.

---

## 4. Implementation Details

### 4.1 Kernel Memory Access
The solver is optimized for the SIMT (Single Instruction, Multiple Threads) architecture of modern GPUs.
* **Coalescing:** By sorting nodes via Morton codes, we ensure that threads with adjacent `dispatchID` access adjacent memory addresses in `nodesBuffer`. This maximizes L2 cache hit rates and memory bandwidth utilization.
* **Warp Divergence:** The `CGSolver.compute` kernels are structured to minimize branching. Active fluid nodes are packed contiguously, ensuring that warps remain fully occupied during the iterative solve.

### 4.2 Code Structure
* **`FluidSimulator.cs`**: Orchestrator. Manages the lifecycle of Compute Buffers and dispatches kernels based on the current simulation state.
* **`CGSolver.compute`**: Implements the Preconditioned Conjugate Gradient (PCG) algorithm. Contains the matrix-free Laplacian operator which reconstructs the stencil $A$ on-the-fly using neighbor indices.
* **`Preconditioner.compute`**: The neural inference engine.
* **`Nodes.compute`**: Handles the bottom-up construction of the octree, including leaf creation, neighbor finding, and hierarchical coarsening.
* **`RadixSort.compute`**: A custom implementation of parallel Radix Sort, utilizing a Hillis-Steele scan for prefix sum calculation, essential for the fast rebuilding of the linear octree.