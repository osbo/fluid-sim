# Neural-Accelerated Fluid Simulation: Learned SPAI Preconditioning on GPU

**Abstract:** This project implements a high-performance Eulerian-Lagrangian fluid simulation (PIC/FLIP) on the GPU, featuring a novel **Neural Sparse Approximate Inverse (SPAI)** preconditioner for the Conjugate Gradient pressure solver. By treating the linearized octree as a sequence, a Windowed Transformer predicts the inverse topology of the Laplacian operator, replacing traditional sequential preconditioners (e.g., Incomplete Cholesky) with a fully parallelizable matrix-vector product.

Based on the methodology presented by **Yang et al. (2025)** in *"Learning Sparse Approximate Inverse Preconditioners for Conjugate Gradient Solvers on GPUs"*.

## 1. Performance Statistics

Benchmarks were conducted on a stress-test scenario with **1,048,576 particles**, utilizing a dynamic sparse octree with cell sizes ranging from **Level 4 ($2^4$)** to **Level 10 ($2^{10}$)**.

### 1.1 Neural SPAI vs. Jacobi (Standard GPU Baseline)
The Jacobi preconditioner is the standard for GPU-based CG due to its $O(1)$ parallel nature. The Neural SPAI outperforms it by capturing non-diagonal spectral dependencies while maintaining similar parallelism.

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

## 2. Domain Discretization: Linearized Sparse Octree

The simulation domain is discretized using a pointer-less, linearized octree structure to maximize GPU memory coalescence.

### 2.1 Adaptive Resolution Levels
The grid follows a power-of-two scaling where **Level 0** represents maximum detail ($1^3$ voxel) and **Level 10** represents the coarse root bounds ($1024^3$ voxel).
* **Current Configuration:** The simulation actively utilizes levels 4 through 10.
* **Refinement Strategy:** A distance-based heuristic allocates high-resolution leaf nodes near the fluid interface (Zero Level Set) and rapidly coarsens into the bulk fluid to reduce the degrees of freedom for the Poisson solve.

### 2.2 The 2:1 Balanced Neighbor Buffer
To facilitate finite difference stencils on a non-uniform grid, we enforce a **2:1 balance constraint**: adjacent cells may differ by at most one level of resolution.

Unlike standard uniform grids with 6 neighbors (Von Neumann neighborhood), a balanced octree node requires a significantly more complex neighbor list.
* **The 24-Slot Problem:** When a node at Level $L$ borders nodes at Level $L-1$ (finer), a single face can be shared by up to 4 smaller neighbors.
* **Implementation:** `Nodes.compute` utilizes a flat buffer stride of **24 integers per node** (6 faces $\times$ 4 max sub-neighbors).
* **Topological Search:** The `FindNeighbors` kernel traverses the Morton curve to populate these slots. During the matrix-free Laplacian application, the solver iterates these 24 slots, weighting fluxes by the intersecting surface area to ensure conservation of mass across resolution changes.

---

## 3. The Neural Preconditioner Architecture

The core innovation is the replacement of numerical preconditioning heuristics with a learned parametric model. This model approximates $G \approx A^{-1}$ such that the condition number $\kappa(GA) \approx 1$.

### 3.1 SWIN-Like Windowed Attention

Standard Transformers have $O(N^2)$ complexity, which is intractable for $N=10^6$ nodes. We employ a **1D Windowed Transformer** architecture similar to Swin Transformer, but applied to the 1D Z-order curve (Morton code sequence) rather than 2D image patches.

* **Window Size:** 64 nodes.
* **Stride:** 32 nodes (50% overlap).
* **Local-Global Approximation:** Due to the locality-preserving properties of the Morton curve, spatially adjacent nodes are highly likely to be adjacent in the index buffer. Statistical analysis of the dataset shows that **~77% of a node's spatial neighbors** fall within this sliding window, allowing the attention mechanism to capture local pressure dependencies effectively.

### 3.2 Model Specifications (`NeuralPreconditioner.py`)
The architecture is designed to be lightweight enough for real-time inference while deep enough to learn the Laplacian topology:
* **Embedding Dimension ($d_{model}$):** 32
* **Attention Heads:** 4
* **Encoder Layers:** 2
* **Parameter Count:** ~12k (Extremely compact for GPU L1 cache residency).

### 3.3 Inference: Fused HLSL Kernel
To avoid the latency of context-switching between Unity and Python/TensorRT, the inference is implemented directly in `Preconditioner.compute` as a **Fused Transformer Layer**.
* **Operation:** The kernel manually performs the Input Embedding $\to$ LayerNorm $\to$ Multi-Head Self-Attention (MHSA) $\to$ Feed-Forward Network (FFN) sequence.
* **Parallelism:** Each thread processes one window. The 50% stride ensures that boundary artifacts at window edges are averaged out in the overlapping regions.

---

## 4. Physics Solver Details

### 4.1 Boundary Conditions
The simulation handles domain boundaries ($x,y,z \in [0, 1024]$) with a specific compliant collision response:
* **Reflective Walls:** Particles violating bounds ($x < 0$ or $x > 1024$) are reflected.
* **Restitution:** A coefficient of $0.3$ is applied to the velocity perpendicular to the wall (bounce damping).
* **$\epsilon$-Buffer:** To prevent numerical sticking where interpolation kernels sample strictly zero velocity at the exact wall coordinate, particles are clamped to a spatial buffer of $\epsilon = 0.05$ from the boundary.

### 4.2 Matrix-Free PCG Solver (`CGSolver.compute`)
The system solves $A p = \mathbf{d}$ (Poisson Equation). The matrix $A$ is never explicitly constructed.
* **Operator $A(\cdot)$:** The Laplacian is evaluated on-the-fly using the `nodesBuffer` and the 24-slot `neighborsBuffer`.
* **Preconditioner Application:** The PCG step $z = M^{-1}r$ is replaced by $z = G r$, where $G$ is the output of the Neural Transformer. This reduces the preconditioning step to a simple sparse matrix-vector multiplication (SpMV), eliminating the serial dependency of forward-backward substitution found in ICC.

## 5. References

**Zherui Yang, Zhehao Li, Kangbo Lyu, Yixuan Li, Tao Du, Ligang Liu.** *Learning Sparse Approximate Inverse Preconditioners for Conjugate Gradient Solvers on GPUs.* arXiv preprint arXiv:2510.27517, 2025.