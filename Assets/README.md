# Neural-Accelerated Fluid Simulation: Learned SPAI Preconditioning on GPU

**Abstract:** This project implements a high-performance Eulerian-Lagrangian fluid simulation for incompressible flows, targeting massive particle counts ($N > 10^6$) on consumer hardware. The core contribution is a hybrid numerical-neural solver that replaces standard preconditioners (e.g., Incomplete Cholesky, Jacobi) with a learned **Sparse Approximate Inverse (SPAI)**. By training a Transformer-based architecture to predict the inverse topology of the discrete Laplacian on a linearized sparse octree, we achieve a significant reduction in convergence time for the Conjugate Gradient (CG) pressure projection.

Based on the methodology presented by **Yang et al. (2025)** in *"Learning Sparse Approximate Inverse Preconditioners for Conjugate Gradient Solvers on GPUs"*.

## 1. Performance Statistics

Benchmarks were conducted on a stress-test scenario with **1,048,576 particles**, utilizing a dynamic sparse octree with cell sizes ranging from **Level 4 ($2^4$)** to **Level 10 ($2^{10}$)**.

### 1.1 Neural SPAI vs. Jacobi (Standard GPU Baseline)
The Jacobi preconditioner ($M = \text{diag}(A)$) is the industry standard for GPU-based CG due to its trivial parallelization. The Neural SPAI outperforms it by capturing non-diagonal spectral dependencies while maintaining similar parallelism.

| Metric | Reduction | Value Change |
|--------|-----------|--------------|
| **Frame Time** | **11.3%** | $53\text{ms} \to 47\text{ms}$ |
| **Solve Time** | **17.9%** | $28\text{ms} \to 23\text{ms}$ |
| **CG Iterations** | **44.6%** | $56 \to 31$ |

### 1.2 Neural SPAI vs. None (Raw CG)
This comparison highlights the raw algorithmic efficiency of the learned preconditioner against an unconditioned Krylov subspace solver.

| Metric | Reduction | Value Change |
|--------|-----------|--------------|
| **Frame Time** | **54.4%** | $103\text{ms} \to 47\text{ms}$ |
| **Solve Time** | **75.0%** | $92\text{ms} \to 23\text{ms}$ |
| **CG Iterations** | **74.6%** | $122 \to 31$ |

---

## 2. Mathematical Methodology

### 2.1 The Poisson Problem
The bottleneck of the simulation is solving the pressure projection step of the incompressible Euler equations:
$$\nabla \cdot (\frac{1}{\rho} \nabla p) = \frac{\nabla \cdot \mathbf{u}^*}{\Delta t}$$
Discretized on our adaptive grid, this yields the linear system $A p = \mathbf{d}$, where $A$ is the sparse, symmetric positive-definite (SPD) Laplacian matrix.

### 2.2 Preconditioned Conjugate Gradient (PCG)
Standard CG convergence depends on the spectral condition number $\kappa(A)$. We solve the preconditioned system:
$$M^{-1} A p = M^{-1} \mathbf{d}$$
where $M$ is a preconditioner such that $\kappa(M^{-1}A) \ll \kappa(A)$.

### 2.3 Sparse Approximate Inverse (SPAI)
Traditional preconditioners like Incomplete Cholesky (ICC) require forward-backward substitution ($L y = r$, $L^T z = y$), which is inherently serial and poorly suited for GPUs.
Instead of defining $M$, we learn the explicit inverse factor $G \approx L^{-1}$ such that:
$$A^{-1} \approx G G^T$$
This transforms the preconditioning step from a triangular solve into a **Sparse Matrix-Vector Multiplication (SpMV)**:
$$z = G (G^T r)$$
This operation is perfectly parallelizable, allowing the Neural Preconditioner to fully saturate GPU memory bandwidth.

---

## 3. Domain Discretization: Linearized Sparse Octree

To manage $10^6$ particles without the overhead of pointer-chasing, the simulation utilizes a **linearized sparse octree** structured along a Z-order curve (Morton Code).

### 3.1 Adaptive Resolution Levels
The grid follows a power-of-two scaling where **Level 0** represents maximum detail ($1^3$ voxel) and **Level 10** represents the coarse root bounds ($1024^3$ voxel).
* **Current Configuration:** The simulation actively utilizes levels 4 through 10.
* **Refinement Strategy:** A distance-based heuristic allocates high-resolution leaf nodes near the fluid interface (Zero Level Set) and rapidly coarsens into the bulk fluid.

### 3.2 The 2:1 Balanced Neighbor Buffer
To facilitate finite difference stencils on a non-uniform grid, we enforce a **2:1 balance constraint**: adjacent cells may differ by at most one level of resolution.

Unlike standard uniform grids with 6 neighbors (Von Neumann neighborhood), a balanced octree node requires a significantly more complex neighbor list.
* **The 24-Slot Problem:** When a node at Level $L$ borders nodes at Level $L-1$ (finer), a single face can be shared by up to 4 smaller neighbors.
* **Implementation:** `Nodes.compute` utilizes a flat buffer stride of **24 integers per node** (6 faces $\times$ 4 max sub-neighbors).
* **Topological Search:** The `FindNeighbors` kernel traverses the Morton curve to populate these slots. During the matrix-free Laplacian application, the solver iterates these 24 slots, weighting fluxes by the intersecting surface area to ensure conservation of mass across resolution changes.

---

## 4. The Neural Preconditioner Architecture

The core innovation is the replacement of numerical preconditioning heuristics with a learned parametric model implemented in `NeuralPreconditioner.py`.

### 4.1 SWIN-Like Windowed Attention

Standard Transformers have $O(N^2)$ complexity, which is intractable for $N=10^6$ nodes. We employ a **1D Windowed Transformer** architecture similar to Swin Transformer, but applied to the 1D Z-order curve rather than 2D image patches.

* **Window Size:** 64 nodes.
* **Stride:** 32 nodes (50% overlap).
* **Locality:** Statistical analysis of the Morton curve dataset shows that **~77% of a node's spatial neighbors** fall within this sliding window. This allows the attention mechanism to effectively capture local pressure dependencies (the stencil of $A$) without global attention.

### 4.2 Model Specifications
The architecture is designed to be lightweight enough for real-time inference while deep enough to learn the Laplacian topology:
* **Embedding Dimension ($d_{model}$):** 32
* **Attention Heads:** 4
* **Encoder Layers:** 2
* **Parameter Count:** ~12k (Extremely compact for GPU L1 cache residency).

### 4.3 Unsupervised SAI Loss
The network is trained using **Stochastic Trace Estimation** to minimize the Frobenius norm of the residual identity, without ever computing the expensive ground-truth inverse:
$$\mathcal{L} = \| I - G A \|_F^2$$
This unsupervised approach allows the model to learn valid preconditioners solely from the grid topology inputs (Morton codes and Levels).

### 4.4 Inference: Fused HLSL Kernel
To avoid the latency of context-switching between Unity and Python, the inference is implemented directly in `Preconditioner.compute` as a **Fused Transformer Layer**.
* **Operation:** The kernel manually performs the `Embedding` $\to$ `LayerNorm` $\to$ `MHSA` $\to$ `FFN` sequence.
* **Parallelism:** Each thread processes one window. The 50% stride ensures that boundary artifacts at window edges are averaged out in the overlapping regions.

---

## 5. Implementation Details

### 5.1 Boundary Conditions

**Particle (Lagrangian) Boundaries:**
Handled in `Particles.compute`, the simulation enforces containment within $[0, 1024]^3$.
* **Reflective Walls:** Particles violating bounds ($x < 0$ or $x > 1024$) are reflected.
* **Restitution:** A coefficient of $0.3$ is applied to the velocity perpendicular to the wall (bounce damping).
* **$\epsilon$-Buffer:** Particles are clamped to a spatial buffer of $\epsilon = 0.05$ from the boundary. This prevents numerical "sticking" where interpolation kernels might sample strictly zero velocity at the exact wall coordinate.

**Pressure (Eulerian) Boundaries:**
Handled in `CGSolver.compute` during the Laplacian construction.
* **Dirichlet ($p=0$):** Applied at the free surface. If a neighbor index $\ge$ `numNodes` (ghost node), it is treated as an air cell with atmospheric pressure $p=0$. This adds a term to the diagonal of $A$ and the RHS, enforcing the condition.
* **Neumann ($\frac{\partial p}{\partial n} = 0$):** Applied at solid walls. If a neighbor is physically missing (outside the domain), the flux across that face is zero. The `ComputeDiagonal` kernel simply omits this face from the coefficient sum, effectively enforcing the no-flow condition naturally via the finite volume discretization.

### 5.2 Matrix-Free PCG Solver
The system solves $A p = \mathbf{d}$ without explicitly constructing $A$.
* **Operator $A(\cdot)$:** The Laplacian is evaluated on-the-fly using the `nodesBuffer` and the 24-slot `neighborsBuffer`.
* **Flux Calculation:** For every face, flux is computed as:
  $$\text{Flux}_{ij} = \frac{\text{Area}_{face}}{\max(\Delta x_{ij}, \epsilon_{dist})} (p_i - p_j)$$
  where $\Delta x_{ij}$ accounts for the potentially different cell sizes of neighbors $i$ and $j$.

### 5.3 Custom Radix Sort
To support the dynamic reconstruction of the Octree every frame, a custom **GPU Radix Sort** (`RadixSort.compute`) was implemented.
* **Algorithm:** Least Significant Digit (LSD) Radix Sort.
* **Prefix Sums:** Uses a hierarchical Hillis-Steele scan to calculate offsets for parallel scattering.
* **Throughput:** Capable of sorting 1M+ particle keys per frame to rebuild the octree structure dynamically.

## 6. References

**Zherui Yang, Zhehao Li, Kangbo Lyu, Yixuan Li, Tao Du, Ligang Liu.** *Learning Sparse Approximate Inverse Preconditioners for Conjugate Gradient Solvers on GPUs.* arXiv preprint arXiv:2510.27517, 2025.