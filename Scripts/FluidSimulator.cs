using UnityEngine;
using UnityEngine.InputSystem;
using System;
using System.Linq;
using System.Runtime.InteropServices;
using UnityEngine.Rendering;

public class FluidSimulator : MonoBehaviour
{
    public BoxCollider simulationBounds;
    public BoxCollider fluidInitialBounds;
    
    public ComputeShader radixSortShader;
    public ComputeShader particlesShader;
    public ComputeShader nodesPrefixSumsShader;
    public ComputeShader nodesShader;
    public ComputeShader cgSolverShader;
    public int numParticles;
    
    // CG Solver parameters
    public int maxCgIterations;
    public float convergenceThreshold;
    
    // Simulation parameters
    private float maxDetailCellSize;
    
    private RadixSort radixSort;
    private int initializeParticlesKernel;
    private int createLeavesKernel;
    private int processNodesKernel;
    private int markUniqueParticlesKernel;
    private int markUniquesPrefixKernel;
    private int markActiveNodesKernel;
    private int scatterUniquesKernel;
    private int scatterActivesKernel;
    private int copyNodesKernel;
    private int writeUniqueCountKernel;
    private int writeNodeCountKernel;
    private int radixPrefixSumKernelId;
    private int radixPrefixFixupKernelId;
    private int findNeighborsKernel;
    private int interpolateFaceVelocitiesKernel;
    private int copyFaceVelocitiesKernel;
    private int calculateDivergenceKernel;
    private int applyLaplacianKernel;
    private int axpyKernel;
    private int dotProductKernel;
    private int applyPressureGradientKernel;
    private int updateParticlesKernel;
    private int applyGravityKernel;
    private int enforceBoundaryConditionsKernel;
    private int solvePressureIterationKernel;
    private int initializePressureBuffersKernel;
    private int initializePhiKernel;
    private int propagatePhiKernel;
    
    // GPU Buffers
    private ComputeBuffer particlesBuffer;
    private ComputeBuffer indicators;
    private ComputeBuffer prefixSums;
    private ComputeBuffer aux;
    private ComputeBuffer aux2;
    private ComputeBuffer uniqueIndices;
    private ComputeBuffer uniqueCount;
    private ComputeBuffer nodeCount;
    private ComputeBuffer auxSmall;
    private ComputeBuffer nodesBuffer;
    private ComputeBuffer nodesBufferOld;
    private ComputeBuffer tempNodesBuffer;
    private ComputeBuffer neighborsBuffer;
    private ComputeBuffer divergenceBuffer;
    private ComputeBuffer pBuffer;
    private ComputeBuffer ApBuffer;
    private ComputeBuffer residualBuffer;
    private ComputeBuffer pressureBuffer;
    private ComputeBuffer phiBuffer;
    private ComputeBuffer phiBuffer_Read;
    private ComputeBuffer dirtyFlagBuffer;
    
    // add previous active node count, but that can be a cpu variable 
    
    // Number of nodes, active nodes, and unique active nodes
    private int numNodes;
    
    // Public accessors for external scripts (e.g., FluidMeshRenderer)
    public ComputeBuffer NodesBuffer => nodesBuffer;
    public ComputeBuffer PhiBuffer => phiBuffer;
    public int NumNodes => numNodes;
    private int numUniqueNodes; // rename to numUniqueNodes
    private int layer;
    public float gravity;
    public float frameRate;
    public int minLayer;
    public int maxLayer;

    private bool hasShownWaitMessage = false;
    private int frameNumber = 0;
    
    private System.Diagnostics.Stopwatch totalOctreeSw;
    
    // Simulation parameters (calculated in InitializeParticleSystem)
    private Vector3 mortonNormalizationFactor;
    private float mortonMaxValue;
    private Vector3 simulationBoundsMin;
    private Vector3 simulationBoundsMax;
    private Vector3 fluidInitialBoundsMin;
    private Vector3 fluidInitialBoundsMax;

    private struct faceVelocities
    {
        public float left;
        public float right;
        public float bottom;
        public float top;
        public float front;
        public float back;
    }
    
    // Particle struct (must match compute shader)
    private struct Particle
    {
        public Vector3 position;    // 12 bytes
        public Vector3 velocity;    // 12 bytes
        public uint layer;          // 4 bytes
        public uint mortonCode;     // 4 bytes
    }

    // Node struct (must match compute shader)
    private struct Node
    {
        public Vector3 position;    // 12 bytes
        public Vector3 velocity;    // 12 bytes
        public faceVelocities velocities; // 6*4 = 24 bytes
        public float mass;             // 4 bytes
        public uint layer;          // 4 bytes
        public uint mortonCode;     // 4 bytes
        public uint active;         // 4 bytes
    }

    private Node[] nodesCPU;
    private Particle[] particlesCPU;
    private string str;

    // Material to render particles as points (assign a shader like Custom/ParticlesPoints)
    public Material particlesMaterial;
    
    // Training data recorder
    public TrainingDataRecorder recorder;

    void OnEnable()
    {
        RenderPipelineManager.endCameraRendering += OnEndCameraRendering;
    }

    void OnDisable()
    {
        RenderPipelineManager.endCameraRendering -= OnEndCameraRendering;
    }

    private void OnEndCameraRendering(ScriptableRenderContext ctx, Camera cam)
    {
        DrawParticles(cam);
    }

    void Start()
    {
        InitializeParticleSystem();
        // InitializeInitialParticles();
        
        // Auto-find recorder if not assigned
        if (recorder == null)
        {
            recorder = FindFirstObjectByType<TrainingDataRecorder>();
            if (recorder == null)
            {
                Debug.LogWarning("TrainingDataRecorder is not assigned and not found in scene. No training data will be recorded.");
            }
        }
        
        // Start recording automatically if recorder is assigned
        if (recorder != null)
        {
            recorder.StartNewRun();
        }
    }

    private void InitializeInitialParticles()
    {
        // Set 10 random particles to layer 0
        particlesBuffer.GetData(particlesCPU);

        // Create array of indices and shuffle
        int[] indices = new int[numParticles];
        for (int i = 0; i < numParticles; i++) {
            indices[i] = i;
        }
        
        // Fisher-Yates shuffle
        System.Random rng = new System.Random();
        for (int i = indices.Length - 1; i > 0; i--) {
            int j = rng.Next(0, i + 1);
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }

        // for (int i = 0; i < numParticles/1024 && i < numParticles; i++) {
        for (int i = 0; (uint)i < 1000 && i < numParticles; i++) {
            particlesCPU[indices[i]].layer = (uint)minLayer;
            particlesCPU[indices[i]].velocity = new Vector3(100.0f, 0.0f, 0.0f);
        }

        particlesBuffer.SetData(particlesCPU);
    }

    void Update()
    {
        // // Check for space key press to advance frame
        // if (Keyboard.current == null || !Keyboard.current.spaceKey.wasPressedThisFrame)
        // {
        //     if (!hasShownWaitMessage)
        //     {
        //         Debug.Log("Press SPACE to advance simulation frame");
        //         hasShownWaitMessage = true;
        //     }
        //     return; // Wait for space key press
        // }
        
        // // Reset wait message flag when frame advances
        // hasShownWaitMessage = false;

        layer = minLayer;
        
        // Start frame timing
        var frameSw = System.Diagnostics.Stopwatch.StartNew();
        
        // Calculate maxDetailCellSize for volume calculations
        // Use the normalized simulation bounds (simulationBoundsMin is now Vector3.zero)
        Vector3 simulationSize = simulationBoundsMax;
        maxDetailCellSize = Mathf.Min(simulationSize.x, simulationSize.y, simulationSize.z) / 1024.0f;

        // Step 1: Sort particles
        var sortSw = System.Diagnostics.Stopwatch.StartNew();
        SortParticles();
        sortSw.Stop();

        // Step 2: Find unique particles and create leaves
        var findUniqueSw = System.Diagnostics.Stopwatch.StartNew();
        findUniqueParticles();
        findUniqueSw.Stop();

        var createLeavesSw = System.Diagnostics.Stopwatch.StartNew();
        CreateLeaves();
        createLeavesSw.Stop();

        // Step 3: Layer loop (layers 1-10)
        var layerLoopSw = System.Diagnostics.Stopwatch.StartNew();
        for (layer = layer + 1; layer <= maxLayer; layer++)
        {
            
            findUniqueNodes();
            ProcessNodes();
            compactNodes();
        }
        layerLoopSw.Stop();

        // Step 4: Find neighbors
        var findNeighborsSw = System.Diagnostics.Stopwatch.StartNew();
        findNeighbors();
        findNeighborsSw.Stop();

        // Step 5: Compute level set (distance field)
        var computeLevelSetSw = System.Diagnostics.Stopwatch.StartNew();
        // ComputeLevelSet();
        computeLevelSetSw.Stop();

        // Step 6: Store old velocities for FLIP method
        var storeOldVelocitiesSw = System.Diagnostics.Stopwatch.StartNew();
        StoreOldVelocities();
        storeOldVelocitiesSw.Stop();

        // Step 7: Apply gravity on the grid
        var applyGravitySw = System.Diagnostics.Stopwatch.StartNew();
        ApplyGravity();
        applyGravitySw.Stop();

        // Step 8: Enforce boundary conditions
        var enforceBoundarySw = System.Diagnostics.Stopwatch.StartNew();
        EnforceBoundaryConditions();
        enforceBoundarySw.Stop();

        // Step 9: Solve pressure
        var solvePressureSw = System.Diagnostics.Stopwatch.StartNew();
        SolvePressure();
        solvePressureSw.Stop();

        // Step 10: Update particles
        var updateParticlesSw = System.Diagnostics.Stopwatch.StartNew();
        UpdateParticles();
        updateParticlesSw.Stop();

        // Frame timing summary
        frameSw.Stop();
        frameNumber++;
        Debug.Log($"Frame {frameNumber} Summary:\n" +
                 $"• Total Frame: {frameSw.Elapsed.TotalMilliseconds:F2} ms\n" +
                 $"• Sort: {sortSw.Elapsed.TotalMilliseconds:F2} ms\n" +
                 $"• Find Unique: {findUniqueSw.Elapsed.TotalMilliseconds:F2} ms\n" +
                 $"• Create Leaves: {createLeavesSw.Elapsed.TotalMilliseconds:F2} ms\n" +
                 $"• Layer Loop: {layerLoopSw.Elapsed.TotalMilliseconds:F2} ms\n" +
                 $"• Find Neighbors: {findNeighborsSw.Elapsed.TotalMilliseconds:F2} ms\n" +
                 $"• Compute Level Set: {computeLevelSetSw.Elapsed.TotalMilliseconds:F2} ms\n" +
                 $"• Store Old Velocities: {storeOldVelocitiesSw.Elapsed.TotalMilliseconds:F2} ms\n" +
                 $"• Apply Gravity: {applyGravitySw.Elapsed.TotalMilliseconds:F2} ms\n" +
                 $"• Enforce Boundaries: {enforceBoundarySw.Elapsed.TotalMilliseconds:F2} ms\n" +
                 $"• Solve Pressure: {solvePressureSw.Elapsed.TotalMilliseconds:F2} ms\n" +
                 $"• Update Particles: {updateParticlesSw.Elapsed.TotalMilliseconds:F2} ms");
    }

    private void StoreOldVelocities()
    {
        // Release and recreate nodesBufferOld each frame since numNodes changes
        nodesBufferOld?.Release();
        nodesBufferOld = new ComputeBuffer(numNodes, sizeof(float) * 3 + sizeof(float) * 3 + sizeof(float) * 6 + sizeof(float) + sizeof(uint) * 3);
        
        // Copy current nodesBuffer to nodesBufferOld
        if (nodesBuffer != null && nodesBufferOld != null)
        {
            // Get data from nodesBuffer
            Node[] nodesData = new Node[numNodes];
            nodesBuffer.GetData(nodesData);
            
            // Set data to nodesBufferOld
            nodesBufferOld.SetData(nodesData);
        }
    }

    private void UpdateParticles()
    {
        if (particlesShader == null)
        {
            Debug.LogError("Particles compute shader is not assigned. Please assign `particlesShader` in the inspector.");
            return;
        }

        updateParticlesKernel = particlesShader.FindKernel("UpdateParticles");

        // need: nodes buffer, nodes buffer old, particles buffer, numNodes, numParticles
        particlesShader.SetBuffer(updateParticlesKernel, "nodesBuffer", nodesBuffer);
        particlesShader.SetBuffer(updateParticlesKernel, "nodesBufferOld", nodesBufferOld);
        particlesShader.SetBuffer(updateParticlesKernel, "particlesBuffer", particlesBuffer);
        particlesShader.SetInt("numNodes", numNodes);
        particlesShader.SetInt("numParticles", numParticles);
        particlesShader.SetFloat("deltaTime", (1 / frameRate));
        particlesShader.SetFloat("gravity", gravity);
        particlesShader.SetFloat("maxDetailCellSize", maxDetailCellSize);
        particlesShader.SetVector("mortonNormalizationFactor", mortonNormalizationFactor);
        particlesShader.SetFloat("mortonMaxValue", mortonMaxValue);
        particlesShader.SetVector("simulationBoundsMin", simulationBoundsMin);
        particlesShader.SetVector("simulationBoundsMax", simulationBoundsMax);
        particlesShader.SetInt("minLayer", minLayer);
        particlesShader.SetInt("maxLayer", maxLayer);
        int threadGroups = Mathf.CeilToInt(numParticles / 512.0f);
        particlesShader.Dispatch(updateParticlesKernel, threadGroups, 1, 1);
    }

    private void SolvePressure()
    {
        var totalSolveSw = System.Diagnostics.Stopwatch.StartNew();
        
        if (cgSolverShader == null)
        {
            Debug.LogError("CGSolver compute shader is not assigned. Please assign `cgSolverShader` in the inspector.");
            return;
        }

        // Find kernel indices
        var kernelFindSw = System.Diagnostics.Stopwatch.StartNew();
        int calculateDivergenceKernel = cgSolverShader.FindKernel("CalculateDivergence");
        int applyLaplacianKernel = cgSolverShader.FindKernel("ApplyLaplacian");
        int axpyKernel = cgSolverShader.FindKernel("Axpy");
        int dotProductKernel = cgSolverShader.FindKernel("DotProduct");
        kernelFindSw.Stop();
        
        if (calculateDivergenceKernel < 0 || applyLaplacianKernel < 0 || axpyKernel < 0 || dotProductKernel < 0)
        {
            Debug.LogError("One or more CG solver kernels not found. Check CGSolver.compute shader compilation.");
            return;
        }

        // Initialize buffers - release and recreate each frame since numNodes changes
        var bufferInitSw = System.Diagnostics.Stopwatch.StartNew();
        divergenceBuffer?.Release();
        residualBuffer?.Release();
        pBuffer?.Release();
        ApBuffer?.Release();
        pressureBuffer?.Release();
        
        divergenceBuffer = new ComputeBuffer(numNodes, sizeof(float));
        residualBuffer = new ComputeBuffer(numNodes, sizeof(float));
        pBuffer = new ComputeBuffer(numNodes, sizeof(float));
        ApBuffer = new ComputeBuffer(numNodes, sizeof(float));
        pressureBuffer = new ComputeBuffer(numNodes, sizeof(float));
        bufferInitSw.Stop();

        // --- Step 1: Calculate Divergence and Initialize ---
        var divergenceSw = System.Diagnostics.Stopwatch.StartNew();
        cgSolverShader.SetBuffer(calculateDivergenceKernel, "nodesBuffer", nodesBuffer);
        cgSolverShader.SetBuffer(calculateDivergenceKernel, "divergenceBuffer", divergenceBuffer);
        cgSolverShader.SetBuffer(calculateDivergenceKernel, "neighborsBuffer", neighborsBuffer);
        cgSolverShader.SetInt("numNodes", numNodes);
        Dispatch(calculateDivergenceKernel, numNodes);
        divergenceSw.Stop();

        // Initialize: r = b, p = r (since initial pressure x = 0)
        var initSw = System.Diagnostics.Stopwatch.StartNew();
        CopyBuffer(divergenceBuffer, residualBuffer);
        CopyBuffer(divergenceBuffer, pBuffer);

        float r_dot_r = GpuDotProduct(residualBuffer, residualBuffer);
        if (r_dot_r < convergenceThreshold) return; // Already converged
        initSw.Stop();

        // Initialize residual tracking
        float initialResidual = r_dot_r;
        float previousResidual = r_dot_r;
        int totalIterations = 0;

        // --- Step 2: Main CG Loop ---
        var cgLoopSw = System.Diagnostics.Stopwatch.StartNew();
        for (int i = 0; i < maxCgIterations; i++)
        {
            // Calculate Ap = A * p
            cgSolverShader.SetBuffer(applyLaplacianKernel, "nodesBuffer", nodesBuffer);
            cgSolverShader.SetBuffer(applyLaplacianKernel, "neighborsBuffer", neighborsBuffer);
            cgSolverShader.SetBuffer(applyLaplacianKernel, "pBuffer", pBuffer);
            cgSolverShader.SetBuffer(applyLaplacianKernel, "ApBuffer", ApBuffer);
            cgSolverShader.SetBuffer(applyLaplacianKernel, "neighborsBuffer", neighborsBuffer);
            cgSolverShader.SetInt("numNodes", numNodes);
            cgSolverShader.SetFloat("deltaTime", (1 / frameRate));
            Dispatch(applyLaplacianKernel, numNodes);

            // Calculate alpha with safety checks
            float p_dot_Ap = GpuDotProduct(pBuffer, ApBuffer);
            
            
            // Safety check for ill-conditioned matrix
            if (Math.Abs(p_dot_Ap) < 1e-12f)
            {
                Debug.LogError($"CG Solver: Matrix is singular or ill-conditioned! p_dot_Ap = {p_dot_Ap:E6}");
                break;
            }
            
            if (p_dot_Ap <= 0.0f)
            {
                Debug.LogError($"CG Solver: Matrix is not positive definite! p_dot_Ap = {p_dot_Ap:E6}");
                break;
            }
            
            float alpha = r_dot_r / p_dot_Ap;
            
            // Safety check for alpha with more reasonable bounds
            if (Math.Abs(alpha) > 1e10f)
            {
                Debug.LogError($"CG Solver: Alpha is too large! alpha = {alpha:E6}, r_dot_r = {r_dot_r:E6}, p_dot_Ap = {p_dot_Ap:E6}");
                Debug.LogError($"This suggests the matrix is still ill-conditioned. Consider preconditioning or different discretization.");
                break;
            }

            // Update pressure: x = x + alpha * p
            UpdateVector(pressureBuffer, pBuffer, alpha);

            // Update residual: r = r - alpha * Ap
            UpdateVector(residualBuffer, ApBuffer, -alpha);

            // Check for convergence
            float r_dot_r_new = GpuDotProduct(residualBuffer, residualBuffer);
            
            // Early stopping for diverging solver
            if (r_dot_r_new > 10.0f * initialResidual)
            {
                totalIterations = i + 1;
                // Debug.LogError($"CG Solver diverged after {i + 1} iterations. Residual grew to {r_dot_r_new / initialResidual:F1}x initial value.");
                break;
            }
            
            if (r_dot_r_new < convergenceThreshold) 
            {
                totalIterations = i + 1;
                break;
            }

            // Update search direction: p = r + (r_new_dot_r_new / r_dot_r) * p
            float beta = r_dot_r_new / r_dot_r;
            UpdateVector(pBuffer, residualBuffer, 1.0f, beta); // Special AXPY: p = r + beta * p

            r_dot_r = r_dot_r_new;
            previousResidual = r_dot_r_new;
            totalIterations = i + 1;
        }
        cgLoopSw.Stop();
        
        // // Create clean summary report
        // float finalResidual = r_dot_r;
        // float totalImprovement = ((initialResidual - finalResidual) / initialResidual) * 100.0f;
        // float residualRatio = finalResidual / initialResidual;
        
        // string summary = $"CG Solver Summary: Iterations {totalIterations}/{maxCgIterations}, Total {((totalImprovement > 0) ? "Convergence" : "Divergence")} {totalImprovement:F2}%, Residual Ratio {residualRatio:E3}x initial\n";
        // summary += $"• Iterations: {totalIterations}/{maxCgIterations}\n";
        // summary += $"• Initial residual: {initialResidual:E6}\n";
        // summary += $"• Final residual: {finalResidual:E6}\n";
        // summary += $"• Total improvement: {totalImprovement:F2}%\n";
        // summary += $"• Residual ratio: {residualRatio:E3}x initial\n";
        
        // if (totalImprovement > 0)
        //     summary += $"• Status: Converged successfully\n";
        // else if (totalImprovement > -10)
        //     summary += $"• Status: Converged (slight increase)\n";
        // else
        //     summary += $"• Status: Diverged ({Math.Abs(totalImprovement):F1}% increase)\n";
            
        // Debug.Log(summary);

        // Save training data: At this point, pressureBuffer contains the converged result (Target).
        // nodesBuffer and neighborsBuffer contain the Geometry (Input).
        // divergenceBuffer contains the RHS (Input).
        if (recorder != null && recorder.isRecording)
        {
            // Get actual world-space bounds (not normalized)
            Vector3 simBoundsMin = simulationBounds != null ? simulationBounds.bounds.min : Vector3.zero;
            Vector3 simBoundsMax = simulationBounds != null ? simulationBounds.bounds.max : Vector3.zero;
            Vector3 fluidBoundsMin = fluidInitialBounds != null ? fluidInitialBounds.bounds.min : Vector3.zero;
            Vector3 fluidBoundsMax = fluidInitialBounds != null ? fluidInitialBounds.bounds.max : Vector3.zero;
            
            recorder.SaveFrame(nodesBuffer, neighborsBuffer, divergenceBuffer, pressureBuffer, numNodes,
                minLayer, maxLayer, gravity, numParticles, maxCgIterations, convergenceThreshold, frameRate,
                simBoundsMin, simBoundsMax, fluidBoundsMin, fluidBoundsMax);
        }

        // --- Step 3: Apply pressure to velocities ---
        var pressureGradientSw = System.Diagnostics.Stopwatch.StartNew();
        ApplyPressureGradient();
        pressureGradientSw.Stop();
        
        // Final timing summary
        totalSolveSw.Stop();
        // Debug.Log($"SolvePressure Timing Summary:\n" +
        //          $"• Total: {totalSolveSw.Elapsed.TotalMilliseconds:F2} ms\n" +
        //          $"• Kernel Find: {kernelFindSw.Elapsed.TotalMilliseconds:F2} ms\n" +
        //          $"• Buffer Init: {bufferInitSw.Elapsed.TotalMilliseconds:F2} ms\n" +
        //          $"• Divergence: {divergenceSw.Elapsed.TotalMilliseconds:F2} ms\n" +
        //          $"• Initialization: {initSw.Elapsed.TotalMilliseconds:F2} ms\n" +
        //          $"• CG Loop: {cgLoopSw.Elapsed.TotalMilliseconds:F2} ms\n" +
        //          $"• Pressure Gradient: {pressureGradientSw.Elapsed.TotalMilliseconds:F2} ms");
    }

    private void ApplyPressureGradient()
    {
        if (nodesShader == null)
        {
            Debug.LogError("Nodes compute shader is not assigned.");
            return;
        }

        applyPressureGradientKernel = nodesShader.FindKernel("ApplyPressureGradient");
        
        // Set the necessary buffers and parameters
        nodesShader.SetBuffer(applyPressureGradientKernel, "nodesBuffer", nodesBuffer);
        nodesShader.SetBuffer(applyPressureGradientKernel, "tempNodesBuffer", tempNodesBuffer);
        nodesShader.SetBuffer(applyPressureGradientKernel, "neighborsBuffer", neighborsBuffer);
        nodesShader.SetBuffer(applyPressureGradientKernel, "pressureBuffer", pressureBuffer); // The result from the CG solve!
        
        nodesShader.SetInt("numNodes", numNodes);
        nodesShader.SetFloat("deltaTime", (1 / frameRate)); // Use the same deltaTime as in advection
        nodesShader.SetFloat("maxDetailCellSize", maxDetailCellSize);
        nodesShader.SetInt("minLayer", minLayer);

        // Dispatch the kernel to update velocities
        int threadGroups = Mathf.CeilToInt(numNodes / 512.0f);
        nodesShader.Dispatch(applyPressureGradientKernel, threadGroups, 1, 1);

        copyFaceVelocitiesKernel = nodesShader.FindKernel("copyFaceVelocities");
        nodesShader.SetBuffer(copyFaceVelocitiesKernel, "nodesBuffer", nodesBuffer);
        nodesShader.SetBuffer(copyFaceVelocitiesKernel, "tempNodesBuffer", tempNodesBuffer);
        nodesShader.SetInt("numNodes", numNodes);
        nodesShader.Dispatch(copyFaceVelocitiesKernel, threadGroups, 1, 1);
    }

    // Helper for dispatching kernels
    private void Dispatch(int kernel, int count) 
    {
        int threadGroups = Mathf.CeilToInt(count / 512.0f);
        cgSolverShader.Dispatch(kernel, threadGroups, 1, 1);
    }

    // Helper for copying buffer data
    private void CopyBuffer(ComputeBuffer source, ComputeBuffer destination)
    {
        if (source.count != destination.count) return;
        
        float[] data = new float[source.count];
        source.GetData(data);
        destination.SetData(data);
    }

    // Helper for GPU-side dot product
    private float GpuDotProduct(ComputeBuffer bufferA, ComputeBuffer bufferB)
    {
        if (cgSolverShader == null) return 0.0f;

        int dotProductKernel = cgSolverShader.FindKernel("DotProduct");
        
        // Set up buffers for dot product
        cgSolverShader.SetBuffer(dotProductKernel, "xBuffer", bufferA);
        cgSolverShader.SetBuffer(dotProductKernel, "yBuffer", bufferB);
        cgSolverShader.SetBuffer(dotProductKernel, "divergenceBuffer", divergenceBuffer); // Required for output
        cgSolverShader.SetInt("numNodes", numNodes);
        
        // Dispatch dot product kernel
        int threadGroups = Mathf.CeilToInt(numNodes / 512.0f);
        cgSolverShader.Dispatch(dotProductKernel, threadGroups, 1, 1);
        
        // Read back the result from divergenceBuffer (reused as temporary output)
        float[] result = new float[threadGroups];
        divergenceBuffer.GetData(result);
        
        // Sum up all partial results
        float total = 0.0f;
        for (int i = 0; i < threadGroups; i++)
        {
            total += result[i];
        }
        
        return total;
    }

    // Helper for AXPY operations: y = a*x + y
    private void UpdateVector(ComputeBuffer yBuffer, ComputeBuffer xBuffer, float a)
    {
        if (cgSolverShader == null) return;

        int axpyKernel = cgSolverShader.FindKernel("Axpy");
        
        cgSolverShader.SetBuffer(axpyKernel, "xBuffer", xBuffer);
        cgSolverShader.SetBuffer(axpyKernel, "yBuffer", yBuffer);
        cgSolverShader.SetFloat("a", a);
        cgSolverShader.SetInt("numNodes", numNodes);
        
        Dispatch(axpyKernel, numNodes);
    }

    // Special AXPY for p = r + beta * p
    private void UpdateVector(ComputeBuffer pBuffer, ComputeBuffer rBuffer, float rCoeff, float pCoeff)
    {
        if (cgSolverShader == null) return;

        // Find the new Scale kernel (and the existing Axpy kernel)
        int scaleKernel = cgSolverShader.FindKernel("Scale");
        int axpyKernel = cgSolverShader.FindKernel("Axpy");
        
        // --- CORRECTED LOGIC ---

        // First, scale p by beta: p_new = beta * p_old
        cgSolverShader.SetBuffer(scaleKernel, "yBuffer", pBuffer);
        cgSolverShader.SetFloat("a", pCoeff); // pCoeff is beta
        cgSolverShader.SetInt("numNodes", numNodes);
        Dispatch(scaleKernel, numNodes);
        
        // Then, add r: p_new = r_new + p_new (which is now beta * p_old)
        cgSolverShader.SetBuffer(axpyKernel, "xBuffer", rBuffer);
        cgSolverShader.SetBuffer(axpyKernel, "yBuffer", pBuffer);
        cgSolverShader.SetFloat("a", rCoeff); // rCoeff is 1.0
        cgSolverShader.SetInt("numNodes", numNodes);
        Dispatch(axpyKernel, numNodes);
    }
    
    private void InitializeParticleSystem()
    {
        // Get kernel index
        if (particlesShader == null)
        {
            Debug.LogError("Particles compute shader is not assigned. Please assign `particlesShader` in the inspector.");
            return;
        }
        initializeParticlesKernel = particlesShader.FindKernel("InitializeParticles");
        
        // Create buffers
        particlesBuffer = new ComputeBuffer(numParticles, sizeof(float) * 3 + sizeof(float) * 3 + sizeof(uint) + sizeof(uint)); // 12 + 12 + 4 + 4 = 32 bytes
        particlesCPU = new Particle[numParticles];

        // Set buffer data to compute shader
        particlesShader.SetBuffer(initializeParticlesKernel, "particlesBuffer", particlesBuffer);
        particlesShader.SetInt("count", numParticles);
        particlesShader.SetInt("minLayer", minLayer);
        particlesShader.SetInt("maxLayer", maxLayer);
        // Calculate bounds
        simulationBoundsMin = simulationBounds.bounds.min;
        simulationBoundsMax = simulationBounds.bounds.max;
        
        // Get fluid initial bounds from the GameObject's transform
        fluidInitialBoundsMin = fluidInitialBounds.bounds.min;
        fluidInitialBoundsMax = fluidInitialBounds.bounds.max;
        
        // Normalize bounds to ensure all particle positions are positive
        // Shift all bounds to be positive by subtracting the simulation bounds minimum
        fluidInitialBoundsMin -= simulationBoundsMin;
        fluidInitialBoundsMax -= simulationBoundsMin;
        simulationBoundsMin = Vector3.zero; // Now simulation bounds start at origin
        simulationBoundsMax -= simulationBounds.bounds.min; // Shift simulation bounds max accordingly
        
        // Calculate morton code normalization factors on CPU
        Vector3 simulationSize = simulationBoundsMax;
        
        // For 32-bit Morton codes: 10 bits per axis = 1024 possible values (0-1023)
        // Normalize each axis to 0-1023 range
        mortonNormalizationFactor = new Vector3(
            1023.0f / simulationSize.x,
            1023.0f / simulationSize.y,
            1023.0f / simulationSize.z
        );
        
        // Max morton value is 1023 for each axis
        mortonMaxValue = 1023.0f;
        
        // Calculate grid dimensions for even particle distribution
        Vector3 fluidInitialSize = fluidInitialBoundsMax - fluidInitialBoundsMin;
        
		// Calculate optimal grid dimensions to fit numParticles as evenly as possible (3D)
		// Start with cube root and adjust for aspect ratio
		float cubeRoot = Mathf.Pow(numParticles, 1.0f / 3.0f);
        
        // Calculate aspect ratio normalized dimensions
        float maxSize = Mathf.Max(fluidInitialSize.x, fluidInitialSize.y, fluidInitialSize.z);
        Vector3 normalizedSize = fluidInitialSize / maxSize;
        
		Vector3Int gridDimensions = new Vector3Int(
			Mathf.Max(1, Mathf.RoundToInt(cubeRoot * normalizedSize.x)),
			Mathf.Max(1, Mathf.RoundToInt(cubeRoot * normalizedSize.y)),
			Mathf.Max(1, Mathf.RoundToInt(cubeRoot * normalizedSize.z))
		);
        
        // Ensure we have enough grid cells for all particles
        // If we have too few cells, increase dimensions
		while (gridDimensions.x * gridDimensions.y * gridDimensions.z < numParticles)
		{
			// Increase the smallest dimension to approach the target count
			if (gridDimensions.x <= gridDimensions.y && gridDimensions.x <= gridDimensions.z)
				gridDimensions.x++;
			else if (gridDimensions.y <= gridDimensions.x && gridDimensions.y <= gridDimensions.z)
				gridDimensions.y++;
			else
				gridDimensions.z++;
		}
        
        // If we have too many cells, reduce dimensions
		while (gridDimensions.x * gridDimensions.y * gridDimensions.z > numParticles)
		{
			// Decrease the largest dimension (but not below 1)
			if (gridDimensions.x >= gridDimensions.y && gridDimensions.x >= gridDimensions.z)
				gridDimensions.x = Mathf.Max(1, gridDimensions.x - 1);
			else if (gridDimensions.y >= gridDimensions.x && gridDimensions.y >= gridDimensions.z)
				gridDimensions.y = Mathf.Max(1, gridDimensions.y - 1);
			else
				gridDimensions.z = Mathf.Max(1, gridDimensions.z - 1);
		}
        
        // Calculate grid spacing to fill the entire fluid bounds
		Vector3 actualGridSpacing = new Vector3(
			fluidInitialSize.x / Mathf.Max(1, gridDimensions.x),
			fluidInitialSize.y / Mathf.Max(1, gridDimensions.y),
			fluidInitialSize.z / Mathf.Max(1, gridDimensions.z)
		);
        
        // Set bounds parameters to compute shader
        particlesShader.SetVector("simulationBoundsMin", simulationBoundsMin);
        particlesShader.SetVector("simulationBoundsMax", simulationBoundsMax);
        particlesShader.SetVector("fluidInitialBoundsMin", fluidInitialBoundsMin);
        particlesShader.SetVector("fluidInitialBoundsMax", fluidInitialBoundsMax);
        particlesShader.SetVector("simulationSize", simulationSize);
        particlesShader.SetVector("mortonNormalizationFactor", mortonNormalizationFactor);
        particlesShader.SetFloat("mortonMaxValue", mortonMaxValue);
        particlesShader.SetInt("minLayer", minLayer);
        
        // Set grid parameters to compute shader
        particlesShader.SetInts("gridDimensions", new int[] { (int)gridDimensions.x, (int)gridDimensions.y, (int)gridDimensions.z });
        particlesShader.SetVector("gridSpacing", actualGridSpacing);
        
        // Dispatch the kernel
        particlesShader.SetFloat("dispersionPower", 4.0f); // higher = stronger skew
        particlesShader.SetFloat("boundaryThreshold", 0.01f); // near-left threshold in normalized X
        int threadGroups = Mathf.CeilToInt(numParticles / 512.0f);
        particlesShader.Dispatch(initializeParticlesKernel, threadGroups, 1, 1);
    }
    
    // Public method for ScenarioManager to reset the simulation
    public void ResetSimulation()
    {
        InitializeParticleSystem();
    }
    
    private void SortParticles()
    {
        // Release and recreate radix sort each frame since numParticles changes
        radixSort?.ReleaseBuffers();
        radixSort = new RadixSort(radixSortShader, (uint)numParticles);
        
        // Sort the particles directly by their morton codes
        radixSort.Sort(particlesBuffer, particlesBuffer, (uint)numParticles);
    }

    private void findUniqueParticles()
    {
        if (numParticles == 0) return;

        if (nodesPrefixSumsShader == null)
        {
            Debug.LogError("Leaves compute shader is not assigned. Please assign `nodesPrefixSumsShader` in the inspector.");
            return;
        }

        // Find kernels
        markUniqueParticlesKernel = nodesPrefixSumsShader.FindKernel("markUniqueParticles");
        markUniquesPrefixKernel = nodesPrefixSumsShader.FindKernel("markUniquesPrefix");
        scatterUniquesKernel = nodesPrefixSumsShader.FindKernel("scatterUniques");
        writeUniqueCountKernel = nodesPrefixSumsShader.FindKernel("writeUniqueCount");

        if (markUniqueParticlesKernel < 0 || markUniquesPrefixKernel < 0 || scatterUniquesKernel < 0 || writeUniqueCountKernel < 0)
        {
            Debug.LogError("One or more kernels not found in Leaves.compute. Verify #pragma kernel names and shader assignment.");
            return;
        }

        // Release and recreate buffers each frame since numParticles changes
        indicators?.Release();
        prefixSums?.Release();
        aux?.Release();
        aux2?.Release();
        uniqueIndices?.Release();
        uniqueCount?.Release();

        // Allocate leaves buffers
        indicators = new ComputeBuffer(numParticles, sizeof(uint));
        prefixSums = new ComputeBuffer(numParticles, sizeof(uint));

        uint tgSize = 512u;
        uint numThreadgroups = (uint)((numParticles + (tgSize * 2) - 1) / (tgSize * 2));
        uint auxSize = (uint)System.Math.Max(1, (int)numThreadgroups);
        aux = new ComputeBuffer((int)auxSize, sizeof(uint));
        aux2 = new ComputeBuffer((int)auxSize, sizeof(uint));

        uniqueIndices = new ComputeBuffer(numParticles, sizeof(uint));
        uniqueCount = new ComputeBuffer(1, sizeof(uint));

        int prefixBits = layer * 3;

        // mark uniques
        nodesPrefixSumsShader.SetBuffer(markUniqueParticlesKernel, "sortedParticles", particlesBuffer);
        nodesPrefixSumsShader.SetBuffer(markUniqueParticlesKernel, "indicators", indicators);
        nodesPrefixSumsShader.SetInt("len", numParticles);
		nodesPrefixSumsShader.SetInt("prefixBits", prefixBits);
        int groupsLinear = (numParticles + 511) / 512;
        nodesPrefixSumsShader.Dispatch(markUniqueParticlesKernel, groupsLinear, 1, 1);

        // Reuse proven radix scan kernels for indicators scan
        radixPrefixSumKernelId = radixSortShader.FindKernel("prefixSum");
        radixPrefixFixupKernelId = radixSortShader.FindKernel("prefixFixup");

        // First-level scan: indicators -> prefixSums
        radixSortShader.SetBuffer(radixPrefixSumKernelId, "input", indicators);
        radixSortShader.SetBuffer(radixPrefixSumKernelId, "output", prefixSums);
        radixSortShader.SetBuffer(radixPrefixSumKernelId, "aux", aux);
        radixSortShader.SetInt("len", numParticles);
        radixSortShader.SetInt("zeroff", 1);
        radixSortShader.Dispatch(radixPrefixSumKernelId, (int)numThreadgroups, 1, 1);

        if (numThreadgroups > 1)
        {
            // Scan aux -> aux2
            if (auxSmall == null) auxSmall = new ComputeBuffer(1, sizeof(uint));
            uint auxThreadgroups = 1; // aux length is small
            radixSortShader.SetBuffer(radixPrefixSumKernelId, "input", aux);
            radixSortShader.SetBuffer(radixPrefixSumKernelId, "output", aux2);
            radixSortShader.SetBuffer(radixPrefixSumKernelId, "aux", auxSmall);
            radixSortShader.SetInt("len", (int)auxSize);
            radixSortShader.SetInt("zeroff", 1);
            radixSortShader.Dispatch(radixPrefixSumKernelId, (int)auxThreadgroups, 1, 1);

            // Fixup: add scanned aux into prefixSums
            radixSortShader.SetBuffer(radixPrefixFixupKernelId, "input", prefixSums);
            radixSortShader.SetBuffer(radixPrefixFixupKernelId, "aux", aux2);
            radixSortShader.SetInt("len", numParticles);
            radixSortShader.Dispatch(radixPrefixFixupKernelId, (int)numThreadgroups, 1, 1);
        }

        // scatterUniques uniques -> uniqueIndices (store corresponding particle index)
        nodesPrefixSumsShader.SetBuffer(scatterUniquesKernel, "indicators", indicators);
        nodesPrefixSumsShader.SetBuffer(scatterUniquesKernel, "prefixSums", prefixSums);
        nodesPrefixSumsShader.SetBuffer(scatterUniquesKernel, "sortedParticles", particlesBuffer);
        nodesPrefixSumsShader.SetBuffer(scatterUniquesKernel, "uniqueIndices", uniqueIndices);
        nodesPrefixSumsShader.SetInt("len", numParticles);
        nodesPrefixSumsShader.Dispatch(scatterUniquesKernel, groupsLinear, 1, 1);

        // write unique count
        nodesPrefixSumsShader.SetBuffer(writeUniqueCountKernel, "indicators", indicators);
        nodesPrefixSumsShader.SetBuffer(writeUniqueCountKernel, "prefixSums", prefixSums);
        nodesPrefixSumsShader.SetBuffer(writeUniqueCountKernel, "uniqueCount", uniqueCount);
        nodesPrefixSumsShader.SetInt("len", numParticles);
        nodesPrefixSumsShader.Dispatch(writeUniqueCountKernel, 1, 1, 1);

        // Get unique count
        uint[] numNodesCpu = new uint[1];
        uniqueCount.GetData(numNodesCpu);
        numNodes = (int)numNodesCpu[0];
    }

    private void CreateLeaves()
    {
        // Dispatch numNodes threads
        if (nodesShader == null)
        {
            Debug.LogError("Nodes compute shader is not assigned. Please assign `nodesShader` in the inspector.");
            return;
        }

        if (numNodes == 0) return;

        // Find kernel
        createLeavesKernel = nodesShader.FindKernel("CreateLeaves");

        // Release and recreate node buffers each frame since numNodes changes
        nodesBuffer?.Release();
        tempNodesBuffer?.Release();
        nodesBuffer = new ComputeBuffer(numNodes, sizeof(float) * 3 + sizeof(float) * 3 + sizeof(float) * 6 + sizeof(float) + sizeof(uint) * 3);
        tempNodesBuffer = new ComputeBuffer(numNodes, sizeof(float) * 3 + sizeof(float) * 3 + sizeof(float) * 6 + sizeof(float) + sizeof(uint) * 3);

        // Set buffer data to compute shader
        nodesShader.SetBuffer(createLeavesKernel, "particlesBuffer", particlesBuffer);
        nodesShader.SetBuffer(createLeavesKernel, "nodesBuffer", nodesBuffer);
        nodesShader.SetBuffer(createLeavesKernel, "uniqueIndices", uniqueIndices);
        nodesShader.SetInt("numNodes", numNodes);
        nodesShader.SetInt("numParticles", numParticles);
        nodesShader.SetInt("minLayer", minLayer);
        nodesShader.SetInt("maxLayer", maxLayer);

        // Dispatch the kernel
        int threadGroups = Mathf.CeilToInt(numNodes / 512.0f);
        nodesShader.Dispatch(createLeavesKernel, threadGroups, 1, 1);
    }

    private void findUniqueNodes() // in for loop
    {
        // dispatch numActiveNodes threads, find unique-prefix active nodes
        if (nodesPrefixSumsShader == null)
        {
            Debug.LogError("NodesPrefixSums compute shader is not assigned. Please assign `nodesPrefixSumsShader` in the inspector.");
            return;
        }

        if (numNodes == 0) return;

        ClearUniqueBuffers();

        // Calculate prefix bits: shift right by 3 * layer bits
        int prefixBits = layer * 3;

		// Mark uniques with prefix comparison (active indices)
		if (markUniquesPrefixKernel == 0)
		{
			markUniquesPrefixKernel = nodesPrefixSumsShader.FindKernel("markUniquesPrefix");
		}
		nodesPrefixSumsShader.SetBuffer(markUniquesPrefixKernel, "nodesBuffer", nodesBuffer);
		nodesPrefixSumsShader.SetBuffer(markUniquesPrefixKernel, "indicators", indicators);
		nodesPrefixSumsShader.SetInt("len", numNodes);
		nodesPrefixSumsShader.SetInt("prefixBits", prefixBits);
		int groupsLinear = (numNodes + 511) / 512;
		nodesPrefixSumsShader.Dispatch(markUniquesPrefixKernel, groupsLinear, 1, 1);

        // Reuse proven radix scan kernels for indicators scan
        uint tgSize = 512u;
        uint numThreadgroups = (uint)((numNodes + (tgSize * 2) - 1) / (tgSize * 2));
        uint auxSize = (uint)System.Math.Max(1, (int)numThreadgroups);

        // First-level scan: indicators -> prefixSums
		radixSortShader.SetBuffer(radixPrefixSumKernelId, "input", indicators);
		radixSortShader.SetBuffer(radixPrefixSumKernelId, "output", prefixSums);
		radixSortShader.SetBuffer(radixPrefixSumKernelId, "aux", aux);
		radixSortShader.SetInt("len", numNodes);
		radixSortShader.SetInt("zeroff", 1);
		radixSortShader.Dispatch(radixPrefixSumKernelId, (int)numThreadgroups, 1, 1);

        if (numThreadgroups > 1)
        {
            // Scan aux -> aux2
            if (auxSmall == null) auxSmall = new ComputeBuffer(1, sizeof(uint));
            uint auxThreadgroups = 1; // aux length is small
			radixSortShader.SetBuffer(radixPrefixSumKernelId, "input", aux);
			radixSortShader.SetBuffer(radixPrefixSumKernelId, "output", aux2);
			radixSortShader.SetBuffer(radixPrefixSumKernelId, "aux", auxSmall);
			radixSortShader.SetInt("len", (int)auxSize);
			radixSortShader.SetInt("zeroff", 1);
			radixSortShader.Dispatch(radixPrefixSumKernelId, (int)auxThreadgroups, 1, 1);

            // Fixup: add scanned aux into prefixSums
			radixSortShader.SetBuffer(radixPrefixFixupKernelId, "input", prefixSums);
			radixSortShader.SetBuffer(radixPrefixFixupKernelId, "aux", aux2);
			radixSortShader.SetInt("len", numNodes);
			radixSortShader.Dispatch(radixPrefixFixupKernelId, (int)numThreadgroups, 1, 1);
        }

        // scatterUniques unique indices
        nodesPrefixSumsShader.SetBuffer(scatterUniquesKernel, "indicators", indicators);
        nodesPrefixSumsShader.SetBuffer(scatterUniquesKernel, "prefixSums", prefixSums);
        nodesPrefixSumsShader.SetBuffer(scatterUniquesKernel, "uniqueIndices", uniqueIndices);
        nodesPrefixSumsShader.SetInt("len", numNodes);
        nodesPrefixSumsShader.Dispatch(scatterUniquesKernel, groupsLinear, 1, 1);

        // Write unique count
        nodesPrefixSumsShader.SetBuffer(writeUniqueCountKernel, "indicators", indicators);
        nodesPrefixSumsShader.SetBuffer(writeUniqueCountKernel, "prefixSums", prefixSums);
        nodesPrefixSumsShader.SetBuffer(writeUniqueCountKernel, "uniqueCount", uniqueCount);
        nodesPrefixSumsShader.SetInt("len", numNodes);
        nodesPrefixSumsShader.Dispatch(writeUniqueCountKernel, 1, 1, 1);

        uint[] uniqueCountCpu = new uint[1];
        uniqueCount.GetData(uniqueCountCpu);
        numUniqueNodes = (int)uniqueCountCpu[0];
    }

    private void ProcessNodes() // in for loop
    {
        // dispatch numUniqueActiveNodes threads, process nodes
        // Find the kernel for processing nodes at this level
        int processNodesKernel = nodesShader.FindKernel("ProcessNodes");
        
        // Set buffers for the node processing kernel
        nodesShader.SetBuffer(processNodesKernel, "uniqueIndices", uniqueIndices);
        nodesShader.SetBuffer(processNodesKernel, "nodesBuffer", nodesBuffer);
        nodesShader.SetInt("numUniqueNodes", numUniqueNodes);
        nodesShader.SetInt("numNodes", numNodes);
        nodesShader.SetInt("layer", layer);
        nodesShader.SetInt("minLayer", minLayer);
        nodesShader.SetInt("maxLayer", maxLayer);
        
        // Dispatch one thread per unique node
        int threadGroups = Mathf.CeilToInt(numUniqueNodes / 512.0f);
        nodesShader.Dispatch(processNodesKernel, threadGroups, 1, 1);
    }

    private void compactNodes() // in for loop
    {
        // dispatch numNodes threads, find active nodes
        if (nodesPrefixSumsShader == null)
        {
            Debug.LogError("NodesPrefixSums compute shader is not assigned. Please assign `nodesPrefixSumsShader` in the inspector.");
            return;
        }

        if (numNodes == 0)
        {
            Debug.LogError("Num nodes is 0. Please assign `numNodes` in the inspector.");
            return;
        }

        ClearActiveBuffers();

        // Initialize buffers if not already created
        if (nodeCount == null)
        {
            nodeCount = new ComputeBuffer(1, sizeof(uint));
        }
        if (uniqueCount == null)
        {
            uniqueCount = new ComputeBuffer(1, sizeof(uint));
        }

        // Find kernels
        markActiveNodesKernel = nodesPrefixSumsShader.FindKernel("markActiveNodes");
        scatterActivesKernel = nodesPrefixSumsShader.FindKernel("scatterActives");
        copyNodesKernel = nodesPrefixSumsShader.FindKernel("copyNodes");
        writeNodeCountKernel = nodesPrefixSumsShader.FindKernel("writeNodeCount");

        if (markActiveNodesKernel < 0 || scatterActivesKernel < 0 || copyNodesKernel < 0 || writeNodeCountKernel < 0)
        {
            Debug.LogError("One or more kernels not found in NodesPrefixSums.compute. Verify #pragma kernel names and shader assignment.");
            return;
        }

        // Set buffers for the node processing kernel
        nodesPrefixSumsShader.SetBuffer(markActiveNodesKernel, "nodesBuffer", nodesBuffer);
        nodesPrefixSumsShader.SetBuffer(markActiveNodesKernel, "indicators", indicators);
        nodesPrefixSumsShader.SetInt("len", numNodes);
        int groupsLinear = Mathf.Max(1, (numNodes + 511) / 512);
        nodesPrefixSumsShader.Dispatch(markActiveNodesKernel, groupsLinear, 1, 1);

        uint tgSize = 512u;
        uint numThreadgroups = (uint)((numNodes + (tgSize * 2) - 1) / (tgSize * 2));
        uint auxSize = (uint)System.Math.Max(1, (int)numThreadgroups);

        // // Debug: Print aux buffer size and thread group info
        // Debug.Log($"Layer {layer}: Prefix sum parameters - numNodes: {numNodes}, numThreadgroups: {numThreadgroups}, auxSize: {auxSize}");

		radixSortShader.SetBuffer(radixPrefixSumKernelId, "input", indicators);
		radixSortShader.SetBuffer(radixPrefixSumKernelId, "output", prefixSums);
		radixSortShader.SetBuffer(radixPrefixSumKernelId, "aux", aux);
		radixSortShader.SetInt("len", numNodes);
		radixSortShader.SetInt("zeroff", 1);
		radixSortShader.Dispatch(radixPrefixSumKernelId, (int)numThreadgroups, 1, 1);

        if (numThreadgroups > 1)
        {

            // Scan aux -> aux2
            if (auxSmall == null) auxSmall = new ComputeBuffer(1, sizeof(uint));
            uint auxThreadgroups = 1; // aux length is small
			radixSortShader.SetBuffer(radixPrefixSumKernelId, "input", aux);
			radixSortShader.SetBuffer(radixPrefixSumKernelId, "output", aux2);
			radixSortShader.SetBuffer(radixPrefixSumKernelId, "aux", auxSmall);
			radixSortShader.SetInt("len", (int)auxSize);
			radixSortShader.SetInt("zeroff", 1);
			radixSortShader.Dispatch(radixPrefixSumKernelId, (int)auxThreadgroups, 1, 1);

            // Fixup: add scanned aux into prefixSums
			radixSortShader.SetBuffer(radixPrefixFixupKernelId, "input", prefixSums);
			radixSortShader.SetBuffer(radixPrefixFixupKernelId, "aux", aux2);
			radixSortShader.SetInt("len", numNodes);
			radixSortShader.Dispatch(radixPrefixFixupKernelId, (int)numThreadgroups, 1, 1);
        }

        // scatterActives active indices
        nodesPrefixSumsShader.SetBuffer(scatterActivesKernel, "indicators", indicators);
        nodesPrefixSumsShader.SetBuffer(scatterActivesKernel, "prefixSums", prefixSums);
        nodesPrefixSumsShader.SetBuffer(scatterActivesKernel, "nodesBuffer", nodesBuffer);
        nodesPrefixSumsShader.SetBuffer(scatterActivesKernel, "tempNodesBuffer", tempNodesBuffer);
        nodesPrefixSumsShader.SetInt("len", numNodes);
        int scatterGroups = Mathf.Max(1, (numNodes + 511) / 512);
        nodesPrefixSumsShader.Dispatch(scatterActivesKernel, scatterGroups, 1, 1);

        // Write active count
        nodesPrefixSumsShader.SetBuffer(writeNodeCountKernel, "indicators", indicators);
        nodesPrefixSumsShader.SetBuffer(writeNodeCountKernel, "prefixSums", prefixSums);
        nodesPrefixSumsShader.SetBuffer(writeNodeCountKernel, "nodeCount", nodeCount);
        nodesPrefixSumsShader.SetInt("len", numNodes);
        nodesPrefixSumsShader.Dispatch(writeNodeCountKernel, 1, 1, 1);

        uint[] nodeCountCpu = new uint[1];
        nodeCount.GetData(nodeCountCpu);
        numNodes = (int)nodeCountCpu[0];

        // copy nodes
        nodesPrefixSumsShader.SetBuffer(copyNodesKernel, "nodesBuffer", nodesBuffer);
        nodesPrefixSumsShader.SetBuffer(copyNodesKernel, "tempNodesBuffer", tempNodesBuffer);
        nodesPrefixSumsShader.SetInt("len", numNodes);
        int copyGroups = Mathf.Max(1, (numNodes + 511) / 512);
        nodesPrefixSumsShader.Dispatch(copyNodesKernel, copyGroups, 1, 1);
    }

    private void findNeighbors()
    {
        if (nodesShader == null)
        {
            Debug.LogError("Nodes compute shader is not assigned. Please assign `nodesShader` in the inspector.");
            return;
        }

        // Release and recreate neighbors buffer each frame since numNodes changes
        neighborsBuffer?.Release();
        neighborsBuffer = new ComputeBuffer(numNodes, sizeof(uint) * 24);

        findNeighborsKernel = nodesShader.FindKernel("findNeighbors");
        nodesShader.SetBuffer(findNeighborsKernel, "nodesBuffer", nodesBuffer);
        nodesShader.SetBuffer(findNeighborsKernel, "tempNodesBuffer", tempNodesBuffer);
        nodesShader.SetBuffer(findNeighborsKernel, "neighborsBuffer", neighborsBuffer);
        nodesShader.SetInt("numNodes", numNodes);
        int threadGroups = Mathf.CeilToInt(numNodes / 512.0f);
        nodesShader.Dispatch(findNeighborsKernel, threadGroups, 1, 1);

        interpolateFaceVelocitiesKernel = nodesShader.FindKernel("interpolateFaceVelocities");
        nodesShader.SetBuffer(interpolateFaceVelocitiesKernel, "nodesBuffer", nodesBuffer);
        nodesShader.SetBuffer(interpolateFaceVelocitiesKernel, "tempNodesBuffer", tempNodesBuffer);
        nodesShader.SetBuffer(interpolateFaceVelocitiesKernel, "neighborsBuffer", neighborsBuffer);
        nodesShader.SetInt("numNodes", numNodes);
        nodesShader.Dispatch(interpolateFaceVelocitiesKernel, threadGroups, 1, 1);

        // Bring the new, velocity interpolated face velocities from tempNodesBuffer back to nodesBuffer
        copyFaceVelocitiesKernel = nodesShader.FindKernel("copyFaceVelocities");
        nodesShader.SetBuffer(copyFaceVelocitiesKernel, "nodesBuffer", nodesBuffer);
        nodesShader.SetBuffer(copyFaceVelocitiesKernel, "tempNodesBuffer", tempNodesBuffer);
        nodesShader.SetInt("numNodes", numNodes);
        nodesShader.Dispatch(copyFaceVelocitiesKernel, threadGroups, 1, 1);
    }

    private void ComputeLevelSet()
    {
        if (nodesShader == null)
        {
            Debug.LogError("Nodes compute shader is not assigned. Please assign `nodesShader` in the inspector.");
            return;
        }

        // Release and recreate buffers each frame since numNodes changes
        phiBuffer?.Release();
        phiBuffer_Read?.Release();
        dirtyFlagBuffer?.Release();
        
        phiBuffer = new ComputeBuffer(numNodes, sizeof(float));
        phiBuffer_Read = new ComputeBuffer(numNodes, sizeof(float));
        dirtyFlagBuffer = new ComputeBuffer(1, sizeof(uint));

        // Find kernels
        initializePhiKernel = nodesShader.FindKernel("InitializePhi");
        propagatePhiKernel = nodesShader.FindKernel("PropagatePhi");

        if (initializePhiKernel < 0 || propagatePhiKernel < 0)
        {
            Debug.LogError("One or more level set kernels not found. Check Nodes.compute shader compilation.");
            return;
        }

        int dispatchSize = Mathf.CeilToInt(numNodes / 512.0f);

        // 1. Initialize the Phi field
        nodesShader.SetBuffer(initializePhiKernel, "phiBuffer", phiBuffer);
        nodesShader.SetBuffer(initializePhiKernel, "neighborsBuffer", neighborsBuffer);
        nodesShader.SetInt("numNodes", numNodes);
        nodesShader.Dispatch(initializePhiKernel, dispatchSize, 1, 1);

        // Copy initial phi values to read buffer
        CopyBuffer(phiBuffer, phiBuffer_Read);

        // 2. Iteratively Propagate (Relax) the Phi field
        ComputeBuffer readBuffer = phiBuffer_Read;
        ComputeBuffer writeBuffer = phiBuffer; // Start with phiBuffer as write buffer
        uint[] dirtyData = { 0 };

        int maxIterations = 32; // Safety break, e.g., max grid dimension
        for (int i = 0; i < maxIterations; i++)
        {
            // Swap buffers (ping-pong)
            (readBuffer, writeBuffer) = (writeBuffer, readBuffer);

            // Reset the dirty flag to 0 before dispatching
            dirtyData[0] = 0;
            dirtyFlagBuffer.SetData(dirtyData);

            // Set buffers for the PropagatePhi kernel
            nodesShader.SetBuffer(propagatePhiKernel, "phiBuffer_Read", readBuffer);
            nodesShader.SetBuffer(propagatePhiKernel, "phiBuffer", writeBuffer);
            nodesShader.SetBuffer(propagatePhiKernel, "dirtyFlagBuffer", dirtyFlagBuffer);
            nodesShader.SetBuffer(propagatePhiKernel, "nodesBuffer", nodesBuffer);
            nodesShader.SetBuffer(propagatePhiKernel, "neighborsBuffer", neighborsBuffer);
            nodesShader.SetInt("numNodes", numNodes);

            // Run the propagation
            nodesShader.Dispatch(propagatePhiKernel, dispatchSize, 1, 1);

            // Check the dirty flag
            dirtyFlagBuffer.GetData(dirtyData);

            if (dirtyData[0] == 0)
            {
                // No nodes changed their phi value. We are done.
                break; 
            }
        }

        // At this point, `writeBuffer` holds the final, correct phi values.
        // If the final write was to phiBuffer_Read, copy it back to phiBuffer
        if (writeBuffer == phiBuffer_Read)
        {
            CopyBuffer(phiBuffer_Read, phiBuffer);
        }

        // float[] phiCPU = new float[numNodes];
        // phiBuffer.GetData(phiCPU);

        // float maxPhi = phiCPU.Max();
        // float minPhi = phiCPU.Min();
        // str = $"max phi: {maxPhi}, min phi: {minPhi}, max-min: {maxPhi - minPhi}";
        // Debug.Log(str);
    }

    private void ApplyGravity()
    {
        if (nodesShader == null) return;

        applyGravityKernel = nodesShader.FindKernel("ApplyGravity");
        if (applyGravityKernel < 0)
        {
            Debug.LogError("ApplyGravity kernel not found in Nodes.compute shader.");
            return;
        }

        nodesShader.SetBuffer(applyGravityKernel, "nodesBuffer", nodesBuffer);
        nodesShader.SetInt("numNodes", numNodes);
        nodesShader.SetFloat("gravity", gravity);
        nodesShader.SetFloat("deltaTime", (1 / frameRate));
        int threadGroups = Mathf.CeilToInt(numNodes / 512.0f);
        nodesShader.Dispatch(applyGravityKernel, threadGroups, 1, 1);
    }

    private void EnforceBoundaryConditions()
    {
        if (nodesShader == null) return;
        
        enforceBoundaryConditionsKernel = nodesShader.FindKernel("EnforceBoundaryConditions");
        if (enforceBoundaryConditionsKernel < 0)
        {
            Debug.LogError("EnforceBoundaryConditions kernel not found in Nodes.compute shader.");
            return;
        }
        
        nodesShader.SetBuffer(enforceBoundaryConditionsKernel, "nodesBuffer", nodesBuffer);
        nodesShader.SetInt("numNodes", numNodes);
        
        int threadGroups = Mathf.CeilToInt(numNodes / 512.0f);
        nodesShader.Dispatch(enforceBoundaryConditionsKernel, threadGroups, 1, 1);
    }

    private void ClearUniqueBuffers()
    {
        if (nodesPrefixSumsShader == null)
        {
            Debug.LogError("NodesPrefixSums compute shader is not assigned. Please assign `nodesPrefixSumsShader` in the inspector.");
            return;
        }
        if (numParticles == 0) return;

        int clearUniqueBuffersKernel = nodesPrefixSumsShader.FindKernel("clearUniqueBuffers");
        nodesPrefixSumsShader.SetBuffer(clearUniqueBuffersKernel, "indicators", indicators);
        nodesPrefixSumsShader.SetBuffer(clearUniqueBuffersKernel, "uniqueIndices", uniqueIndices);
        nodesPrefixSumsShader.SetInt("len", numParticles);
        int groupsLinear = (numParticles + 511) / 512;
        nodesPrefixSumsShader.Dispatch(clearUniqueBuffersKernel, groupsLinear, 1, 1);
    }

    private void ClearActiveBuffers()
    {
        if (nodesPrefixSumsShader == null)
        {
            Debug.LogError("NodesPrefixSums compute shader is not assigned. Please assign `nodesPrefixSumsShader` in the inspector.");
            return;
        }
        if (numParticles == 0) return;
        
        int clearActiveBuffersKernel = nodesPrefixSumsShader.FindKernel("clearActiveBuffers");
        nodesPrefixSumsShader.SetBuffer(clearActiveBuffersKernel, "indicators", indicators);
        nodesPrefixSumsShader.SetInt("len", numParticles);
        int groupsLinear = (numParticles + 511) / 512;
        nodesPrefixSumsShader.Dispatch(clearActiveBuffersKernel, groupsLinear, 1, 1);
    }

    // Simple debug visualization using Gizmos
    private void OnDrawGizmos()
    {
        if (nodesBuffer == null) return;
        
        // Calculate the maximum detail cell size (smallest possible cell)
        // With 10 bits per axis, we have 1024 possible values (0-1023)
        // The maximum detail cell size is simulation bounds divided by 1024
        // Use the normalized simulation bounds (simulationBoundsMin is now Vector3.zero)
        Vector3 simulationSize = simulationBounds.bounds.size;
        // float maxDetailCellSize = Mathf.Min(simulationSize.x, simulationSize.y, simulationSize.z) / 1024.0f;
        float maxDetailCellSize = Mathf.Min(simulationSize.x, simulationSize.y, simulationSize.z) / 1024.0f;

        // Define 11 colors for different layers (0-10)
        Color[] layerColors = new Color[]
        {
            new Color(1f, 0f, 0f),     // Red - Layer 0
            new Color(1f, 0.3f, 0f),   // Orange-red - Layer 1
            new Color(1f, 0.6f, 0f),   // Orange - Layer 2
            new Color(1f, 1f, 0f),     // Yellow - Layer 3
            new Color(0.5f, 1f, 0f),   // Yellow-green - Layer 4
            new Color(0f, 1f, 0f),     // Green - Layer 5
            new Color(0f, 1f, 0.5f),   // Blue-green - Layer 6
            new Color(0f, 1f, 1f),     // Cyan - Layer 7
            new Color(0f, 0.5f, 1f),   // Light blue - Layer 8
            new Color(0f, 0f, 1f),     // Blue - Layer 9
            new Color(0.5f, 0f, 1f)    // Violet - Layer 10
        };

        // nodesCPU = new Node[numNodes];
        // nodesBuffer.GetData(nodesCPU);

        // uint[] neighborsCPU = new uint[numNodes * 24];
        // neighborsBuffer.GetData(neighborsCPU);

        // for (int i = 0; i < numNodes; i++) {
        //     Node node = nodesCPU[i];
        //     // Gizmos.color = layerColors[(int)node.layer];
        //     // Gizmos.DrawWireCube(DecodeMorton3D(node), Vector3.one * Mathf.Max(maxDetailCellSize * Mathf.Pow(2, node.layer), 0.01f));
        //     Gizmos.color = new Color(0.0f, 0.5f, 1.0f, 1.0f);
        //     bool isBoundary = false;
        //     uint neighborBaseIndex = (uint)i * 24;
        //     for (int d = 0; d < 6; d++) {
        //         uint baseFaceIndex = neighborBaseIndex + (uint)d * 4;
        //         uint n0_idx = neighborsCPU[baseFaceIndex];
        //         if (n0_idx == numNodes + 1) {
        //             isBoundary = true;
        //             break;
        //         }
        //     }
        //     if (node.layer == minLayer || isBoundary) {
        //         Gizmos.DrawCube(DecodeMorton3D(node), Vector3.one * Mathf.Max(maxDetailCellSize * Mathf.Pow(2, node.layer), 0.01f));
        //     }
        // }
    }

    private void DrawParticles(Camera cam)
    {
        if (particlesMaterial == null || particlesBuffer == null || numParticles <= 0) return;
        if (cam == null) return;

        particlesMaterial.SetBuffer("_Particles", particlesBuffer);
        particlesMaterial.SetFloat("_PointSize", 2.0f);
        particlesMaterial.SetInt("_MinLayer", minLayer);
        particlesMaterial.SetInt("_MaxLayer", maxLayer);
        
        // Add simulation bounds for denormalizing particle positions
        particlesMaterial.SetVector("_SimulationBoundsMin", simulationBounds.bounds.min);
        particlesMaterial.SetVector("_SimulationBoundsMax", simulationBounds.bounds.max);
        
        // Calculate depth fade parameters based on simulation bounds and camera position
        Vector3 cameraPos = cam.transform.position;
        Bounds bounds = simulationBounds.bounds;
        
        // Calculate distance from camera to nearest and farthest points of the bounds
        Vector3 boundsCenter = bounds.center;
        Vector3 boundsSize = bounds.size;
        
        // Find the nearest point on the bounds to the camera
        Vector3 nearestPoint = new Vector3(
            Mathf.Clamp(cameraPos.x, bounds.min.x, bounds.max.x),
            Mathf.Clamp(cameraPos.y, bounds.min.y, bounds.max.y),
            Mathf.Clamp(cameraPos.z, bounds.min.z, bounds.max.z)
        );
        
        // Calculate distances
        float distanceToNearest = Vector3.Distance(cameraPos, nearestPoint);
        float distanceToCenter = Vector3.Distance(cameraPos, boundsCenter);
        float boundsDiagonal = boundsSize.magnitude; // Diagonal length of the bounds
        
        // Calculate fade start: start fading slightly before the nearest point
        // Subtract half the diagonal to account for particles that might be at the edge
        float fadeStart = Mathf.Max(0.0f, distanceToNearest - boundsDiagonal * 0.3f);
        
        // Calculate fade end: end fading at the farthest point plus some margin
        // Use the diagonal to estimate the farthest possible distance
        float fadeEnd = distanceToCenter + boundsDiagonal * 0.8f;
        
        // Ensure fade end is always greater than fade start
        if (fadeEnd <= fadeStart)
        {
            fadeEnd = fadeStart + boundsDiagonal * 0.5f;
        }
        
        particlesMaterial.SetFloat("_DepthFadeStart", fadeStart);
        particlesMaterial.SetFloat("_DepthFadeEnd", fadeEnd);

        particlesMaterial.SetPass(0);
        Graphics.DrawProceduralNow(MeshTopology.Points, numParticles, 1);
    }

    private Vector3 DecodeMorton3D(Node node)
    {
        int gridResolution = (int)Mathf.Pow(2, 10 - node.layer);
        float cellSize = 1024.0f / gridResolution;
        Vector3 quantizedPos = new Vector3(
            Mathf.Floor(node.position.x / cellSize) * cellSize + cellSize * 0.5f,
            Mathf.Floor(node.position.y / cellSize) * cellSize + cellSize * 0.5f,
            Mathf.Floor(node.position.z / cellSize) * cellSize + cellSize * 0.5f
        );
        Vector3 simulationSize = simulationBounds.bounds.max - simulationBounds.bounds.min;
        
        // Convert back to world coordinates
        Vector3 quantizedWorldPos = simulationBounds.bounds.min + new Vector3(
            quantizedPos.x / 1024.0f * simulationSize.x,
            quantizedPos.y / 1024.0f * simulationSize.y,
            quantizedPos.z / 1024.0f * simulationSize.z
        );

        return quantizedWorldPos;
    }

    void OnDestroy()
    {
        // Clean up buffers
        radixSort?.ReleaseBuffers();
        particlesBuffer?.Release();
        indicators?.Release();
        prefixSums?.Release();
        aux?.Release();
        aux2?.Release();
        auxSmall?.Release();
        uniqueIndices?.Release();
        uniqueCount?.Release();
        nodeCount?.Release();
        nodesBuffer?.Release();
        nodesBufferOld?.Release();
        tempNodesBuffer?.Release();
        neighborsBuffer?.Release();
        divergenceBuffer?.Release();
        residualBuffer?.Release();
        pBuffer?.Release();
        ApBuffer?.Release();
        pressureBuffer?.Release();
        phiBuffer?.Release();
        phiBuffer_Read?.Release();
        dirtyFlagBuffer?.Release();
    }
}

public class RadixSort
{
    private ComputeShader sortShader;
    private int prefixSumKernel;
    private int prefixFixupKernel;
    private int splitPrepKernel;
    private int splitScatterKernel;
    private int copyParticlesKernel;
    private int clearBuffer32Kernel;

    private ComputeBuffer tempParticles;
    private ComputeBuffer tempParticlesB;
    private ComputeBuffer auxBuffer;
    private ComputeBuffer aux2Buffer;
    private ComputeBuffer auxSmallBuffer;
    private ComputeBuffer eBuffer;
    private ComputeBuffer fBuffer;

    private uint maxLength;

    public RadixSort(ComputeShader shader, uint maxLength)
    {
        this.maxLength = maxLength;
        this.sortShader = shader;

        prefixSumKernel = sortShader.FindKernel("prefixSum");
        prefixFixupKernel = sortShader.FindKernel("prefixFixup");
        splitPrepKernel = sortShader.FindKernel("split_prep");
        splitScatterKernel = sortShader.FindKernel("split_scatter");
        copyParticlesKernel = sortShader.FindKernel("copyParticles");
        clearBuffer32Kernel = sortShader.FindKernel("clearBuffer32");

        // Calculate particle struct size (3*4 + 3*4 + 4 + 4 = 32 bytes)
        int particleSize = 3 * 4 + 3 * 4 + 4 + 4; // position(12) + velocity(12) + layer(4) + mortonCode(4)
        tempParticles = new ComputeBuffer((int)maxLength, particleSize, ComputeBufferType.Default);
        tempParticlesB = new ComputeBuffer((int)maxLength, particleSize, ComputeBufferType.Default);
        eBuffer = new ComputeBuffer((int)maxLength, sizeof(uint), ComputeBufferType.Default);
        fBuffer = new ComputeBuffer((int)maxLength, sizeof(uint), ComputeBufferType.Default);

        uint threadgroupSize = 512;
        uint numThreadgroups = (maxLength + (threadgroupSize * 2) - 1) / (threadgroupSize * 2);
        uint requiredAuxSize = System.Math.Max(1, numThreadgroups);
        auxBuffer = new ComputeBuffer((int)requiredAuxSize, sizeof(uint), ComputeBufferType.Default);
        aux2Buffer = new ComputeBuffer((int)requiredAuxSize, sizeof(uint), ComputeBufferType.Default);
        auxSmallBuffer = new ComputeBuffer(1, sizeof(uint), ComputeBufferType.Default);
    }

    public void ReleaseBuffers()
    {
        tempParticles.Release();
        tempParticlesB.Release();
        auxBuffer.Release();
        aux2Buffer.Release();
        auxSmallBuffer.Release();
        eBuffer.Release();
        fBuffer.Release();
    }

    public void Sort(ComputeBuffer inputParticles, ComputeBuffer outputParticles, uint actualCount)
    {
        if (actualCount == 0) return;

        ClearBuffer(tempParticles, (uint)tempParticles.count);
        ClearBuffer(tempParticlesB, (uint)tempParticlesB.count);

        int threadGroupSize = 512;
        int threadGroups = (int)((actualCount + threadGroupSize - 1) / threadGroupSize);

        sortShader.SetBuffer(copyParticlesKernel, "inputParticles", inputParticles);
        sortShader.SetBuffer(copyParticlesKernel, "outputParticles", tempParticles);
        sortShader.SetInt("count", (int)actualCount);
        sortShader.Dispatch(copyParticlesKernel, threadGroups, 1, 1);

        ComputeBuffer particlesIn = tempParticles;
        ComputeBuffer particlesOut = tempParticlesB;

        for (int i = 0; i < 32; i++)
        {
            EncodeSplit(particlesIn, particlesOut, (uint)i, actualCount);

            (particlesIn, particlesOut) = (particlesOut, particlesIn);
        }

        sortShader.SetBuffer(copyParticlesKernel, "inputParticles", particlesIn);
        sortShader.SetBuffer(copyParticlesKernel, "outputParticles", outputParticles);
        sortShader.SetInt("count", (int)actualCount);
        sortShader.Dispatch(copyParticlesKernel, threadGroups, 1, 1);
    }

    private void EncodeSplit(ComputeBuffer inputParticles, ComputeBuffer outputParticles, uint bit, uint count)
    {
        sortShader.SetBuffer(splitPrepKernel, "inputParticles", inputParticles);
        sortShader.SetInt("bit", (int)bit);
        sortShader.SetBuffer(splitPrepKernel, "e", eBuffer);
        sortShader.SetInt("count", (int)count);
        int threadGroups = (int)((count + 512 - 1) / 512);
        sortShader.Dispatch(splitPrepKernel, threadGroups, 1, 1);

        EncodeScan(eBuffer, fBuffer, count, bit);

        sortShader.SetBuffer(splitScatterKernel, "inputParticles", inputParticles);
        sortShader.SetBuffer(splitScatterKernel, "outputParticles", outputParticles);
        sortShader.SetInt("bit", (int)bit);
        sortShader.SetBuffer(splitScatterKernel, "e", eBuffer);
        sortShader.SetBuffer(splitScatterKernel, "f", fBuffer);
        sortShader.SetInt("count", (int)count);
        sortShader.Dispatch(splitScatterKernel, threadGroups, 1, 1);
    }

    private void EncodeScan(ComputeBuffer input, ComputeBuffer output, uint length, uint bit)
    {
        if (length == 0) return;

        uint threadgroupSize = 512;
        uint numThreadgroups = (length + (threadgroupSize * 2) - 1) / (threadgroupSize * 2);
        uint zeroff = 1;

        if (numThreadgroups <= 1)
        {
            sortShader.SetBuffer(prefixSumKernel, "input", input);
            sortShader.SetBuffer(prefixSumKernel, "output", output);
            sortShader.SetBuffer(prefixSumKernel, "aux", auxBuffer);
            sortShader.SetInt("len", (int)length);
            sortShader.SetInt("zeroff", (int)zeroff);
            sortShader.Dispatch(prefixSumKernel, 1, 1, 1);
        }
        else
        {
            sortShader.SetBuffer(prefixSumKernel, "input", input);
            sortShader.SetBuffer(prefixSumKernel, "output", output);
            sortShader.SetBuffer(prefixSumKernel, "aux", auxBuffer);
            sortShader.SetInt("len", (int)length);
            sortShader.SetInt("zeroff", (int)zeroff);
            sortShader.Dispatch(prefixSumKernel, (int)numThreadgroups, 1, 1);

            uint auxLength = numThreadgroups;
            sortShader.SetBuffer(prefixSumKernel, "input", auxBuffer);
            sortShader.SetBuffer(prefixSumKernel, "output", aux2Buffer);
            sortShader.SetBuffer(prefixSumKernel, "aux", auxSmallBuffer);
            sortShader.SetInt("len", (int)auxLength);
            sortShader.SetInt("zeroff", (int)zeroff);
            sortShader.Dispatch(prefixSumKernel, 1, 1, 1);

            sortShader.SetBuffer(prefixFixupKernel, "input", output);
            sortShader.SetBuffer(prefixFixupKernel, "aux", aux2Buffer);
            sortShader.SetInt("len", (int)length);
            sortShader.Dispatch(prefixFixupKernel, (int)numThreadgroups, 1, 1);
        }
    }


    private void ClearBuffer(ComputeBuffer buffer, uint count)
    {
        // Use clearBuffer64 for particle buffers (32-byte structures)
        // and clearBuffer32 for uint buffers
        if (buffer.stride == 32) // Particle buffer
        {
            int clearBuffer64Kernel = sortShader.FindKernel("clearBuffer64");
            sortShader.SetBuffer(clearBuffer64Kernel, "outputParticles", buffer);
            sortShader.SetInt("count", (int)count);
            int threadGroups = (int)((count + 511) / 512);
            sortShader.Dispatch(clearBuffer64Kernel, threadGroups, 1, 1);
        }
        else // uint buffer
        {
            sortShader.SetBuffer(clearBuffer32Kernel, "output", buffer);
            sortShader.SetInt("count", (int)count);
            int threadGroups = (int)((count + 511) / 512);
            sortShader.Dispatch(clearBuffer32Kernel, threadGroups, 1, 1);
        }
    }
}