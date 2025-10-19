using UnityEngine;
using UnityEngine.InputSystem;
using System;
using UnityEngine.Rendering;

public class FluidSimulator : MonoBehaviour
{
    [SerializeField] private BoxCollider simulationBounds;
    [SerializeField] private GameObject fluidInitialBounds;
    
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
    private int pullVelocitiesKernel;
    private int copyFaceVelocitiesKernel;
    private int calculateDivergenceKernel;
    private int applyLaplacianKernel;
    private int axpyKernel;
    private int dotProductKernel;
    private int applyPressureGradientKernel;
    private int updateParticlesKernel;
    private int applyGravityKernel;
    
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
    private ComputeBuffer tempNodesBuffer;
    private ComputeBuffer neighborsBuffer;
    private ComputeBuffer divergenceBuffer;
    private ComputeBuffer pBuffer;
    private ComputeBuffer ApBuffer;
    private ComputeBuffer residualBuffer;
    private ComputeBuffer pressureBuffer;
    
    // add previous active node count, but that can be a cpu variable 
    
    // Number of nodes, active nodes, and unique active nodes
    private int numNodes;
    private int numUniqueNodes; // rename to numUniqueNodes
    private int layer;
    public float gravity;
    public float frameRate;
    public int minLayer;
    public int maxLayer;
    public int surfaceLayer;
    public float velocitySensitivity;
    private float divergenceMultiplier = 50.0f;

    private string str;
    private int activeCount;
    private int inactiveCount;
    private bool hasShownWaitMessage = false;
    
    private System.Diagnostics.Stopwatch totalOctreeSw;
    
    // Simulation parameters (calculated in InitializeParticleSystem)
    private Vector3 mortonNormalizationFactor;
    private float mortonMaxValue;
    private Vector3 simulationBoundsMin;
    private Vector3 simulationBoundsMax;

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
        public faceVelocities velocities; // 6*4 = 24 bytes
        public float mass;             // 4 bytes
        public uint layer;          // 4 bytes
        public uint mortonCode;     // 4 bytes
        public uint active;         // 4 bytes
    }

    private Node[] nodesCPU;
    private Particle[] particlesCPU;

    // Material to render particles as points (assign a shader like Custom/ParticlesPoints)
    public Material particlesMaterial;

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
        Vector3 simulationBoundsMin = simulationBounds.bounds.min;
        Vector3 simulationBoundsMax = simulationBounds.bounds.max;
        Vector3 simulationSize = simulationBoundsMax - simulationBoundsMin;
        maxDetailCellSize = Mathf.Min(simulationSize.x, simulationSize.y, simulationSize.z) / 1024.0f;

        // Frame loop: SortParticles -> findUniqueParticles -> CreateLeaves -> layer loop -> pullVelocities -> SolvePressure -> UpdateParticles

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
            var totalLayerSw = System.Diagnostics.Stopwatch.StartNew();
            
            var findUniqueNodesSw = System.Diagnostics.Stopwatch.StartNew();
            findUniqueNodes();
            findUniqueNodesSw.Stop();
            
            var processNodesSw = System.Diagnostics.Stopwatch.StartNew();
            ProcessNodes();
            processNodesSw.Stop();
            
            var compactNodesSw = System.Diagnostics.Stopwatch.StartNew();
            compactNodes();
            compactNodesSw.Stop();

            totalLayerSw.Stop();
            // Debug.Log($"Layer {layer}: {numNodes} nodes, Total: {totalLayerSw.Elapsed.TotalMilliseconds:F2} ms (Find Unique: {findUniqueNodesSw.Elapsed.TotalMilliseconds:F2} ms, Process: {processNodesSw.Elapsed.TotalMilliseconds:F2} ms, Compact: {compactNodesSw.Elapsed.TotalMilliseconds:F2} ms)");
        }
        layerLoopSw.Stop();

        // Step 4: Pull velocities
        var pullVelocitiesSw = System.Diagnostics.Stopwatch.StartNew();
        pullVelocities();
        pullVelocitiesSw.Stop();

        // Step 4.5: Apply gravity on the grid
        var applyGravitySw = System.Diagnostics.Stopwatch.StartNew();
        ApplyGravity();
        applyGravitySw.Stop();

        // Step 5: Solve pressure
        var solvePressureSw = System.Diagnostics.Stopwatch.StartNew();
        SolvePressure();
        solvePressureSw.Stop();

        nodesCPU = new Node[numNodes];
        nodesBuffer.GetData(nodesCPU);
        string str = "Node velocities:\n";
        for (int i = 0; i < 40 && i < numNodes; i++)
        {
            str += $"Node {i}: Layer {nodesCPU[i].layer}, Position {nodesCPU[i].position}, Morton Code {nodesCPU[i].mortonCode}, Velocities (Left: {nodesCPU[i].velocities.left}, Right: {nodesCPU[i].velocities.right}, Bottom: {nodesCPU[i].velocities.bottom}, Top: {nodesCPU[i].velocities.top}, Front: {nodesCPU[i].velocities.front}, Back: {nodesCPU[i].velocities.back})\n";
        }
        Debug.Log(str);

        // Step 6: Update particles
        var updateParticlesSw = System.Diagnostics.Stopwatch.StartNew();
        UpdateParticles();
        updateParticlesSw.Stop();

        particlesCPU = new Particle[numParticles];
        particlesBuffer.GetData(particlesCPU);
        for (int i = 0; i < numParticles; i++)
        {
            if (float.IsNaN(particlesCPU[i].velocity.x) || float.IsNaN(particlesCPU[i].velocity.y) || float.IsNaN(particlesCPU[i].velocity.z))
            {
                Debug.Log($"Particle {i} has NaN velocity\n");
                break;
            }
        }

        // Draw particles as points using the particles buffer
        // Moved to OnRenderObject for reliable rendering timing

        // Frame timing summary
        frameSw.Stop();
        Debug.Log($"Frame Summary:\n" +
                 $"• Total Frame: {frameSw.Elapsed.TotalMilliseconds:F2} ms\n" +
                 $"• Sort: {sortSw.Elapsed.TotalMilliseconds:F2} ms\n" +
                 $"• Find Unique: {findUniqueSw.Elapsed.TotalMilliseconds:F2} ms\n" +
                 $"• Create Leaves: {createLeavesSw.Elapsed.TotalMilliseconds:F2} ms\n" +
                 $"• Layer Loop: {layerLoopSw.Elapsed.TotalMilliseconds:F2} ms\n" +
                 $"• Pull Velocities: {pullVelocitiesSw.Elapsed.TotalMilliseconds:F2} ms\n" +
                 $"• Apply Gravity: {applyGravitySw.Elapsed.TotalMilliseconds:F2} ms\n" +
                 $"• Solve Pressure: {solvePressureSw.Elapsed.TotalMilliseconds:F2} ms\n" +
                 $"• Update Particles: {updateParticlesSw.Elapsed.TotalMilliseconds:F2} ms");
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

    private void UpdateParticles()
    {
        if (particlesShader == null)
        {
            Debug.LogError("Particles compute shader is not assigned. Please assign `particlesShader` in the inspector.");
            return;
        }

        updateParticlesKernel = particlesShader.FindKernel("UpdateParticles");

        // need: nodes buffer, particles buffer, numNodes, numParticles
        particlesShader.SetBuffer(updateParticlesKernel, "nodesBuffer", nodesBuffer);
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
        cgSolverShader.SetInt("numNodes", numNodes);
        cgSolverShader.SetFloat("maxDetailCellSize", maxDetailCellSize);
        Dispatch(calculateDivergenceKernel, numNodes);
        divergenceSw.Stop();

        // Debug: print total divergence
        float totalDivergence = 0.0f;
        float[] divergenceCPU = new float[numNodes];
        divergenceBuffer.GetData(divergenceCPU);
        for (int i = 0; i < numNodes; i++)
        {
            totalDivergence += divergenceCPU[i];
        }
        Debug.Log($"Total divergence: {totalDivergence}");

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
            cgSolverShader.SetInt("numNodes", numNodes);
            cgSolverShader.SetFloat("maxDetailCellSize", maxDetailCellSize);
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
            
            float alpha = r_dot_r / (p_dot_Ap + 1e-12f);
            
            // Safety check for alpha with more reasonable bounds
            if (Math.Abs(alpha) > 1e3f)
            {
                Debug.LogError($"CG Solver: Alpha is too large! alpha = {alpha:E6}, r_dot_r = {r_dot_r:E6}, p_dot_Ap = {p_dot_Ap:E6}");
                Debug.LogError($"This suggests the matrix is still ill-conditioned. Consider preconditioning or different discretization.");
                break;
            }
            
            // Additional safety: limit alpha to prevent overshooting
            // Use a more conservative cap based on the residual magnitude
            float maxAlpha = Math.Min(1.0f, 1e6f / Math.Max(r_dot_r, 1e-6f));
            alpha = Math.Min(alpha, maxAlpha);

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
        
        // Create clean summary report
        float finalResidual = r_dot_r;
        float totalImprovement = ((initialResidual - finalResidual) / initialResidual) * 100.0f;
        float residualRatio = finalResidual / initialResidual;
        
        string summary = $"CG Solver Summary: Iterations {totalIterations}/{maxCgIterations}, Total {((totalImprovement > 0) ? "Convergence" : "Divergence")} {totalImprovement:F2}%, Residual Ratio {residualRatio:E3}x initial\n";
        summary += $"• Iterations: {totalIterations}/{maxCgIterations}\n";
        summary += $"• Initial residual: {initialResidual:E6}\n";
        summary += $"• Final residual: {finalResidual:E6}\n";
        summary += $"• Total improvement: {totalImprovement:F2}%\n";
        summary += $"• Residual ratio: {residualRatio:E3}x initial\n";
        
        if (totalImprovement > 0)
            summary += $"• Status: Converged successfully\n";
        else if (totalImprovement > -10)
            summary += $"• Status: Converged (slight increase)\n";
        else
            summary += $"• Status: Diverged ({Math.Abs(totalImprovement):F1}% increase)\n";
            
        Debug.Log(summary);

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
        nodesShader.SetFloat("velocitySensitivity", velocitySensitivity);
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

        int axpyKernel = cgSolverShader.FindKernel("Axpy");
        
        // First: p = beta * p
        cgSolverShader.SetBuffer(axpyKernel, "xBuffer", pBuffer);
        cgSolverShader.SetBuffer(axpyKernel, "yBuffer", pBuffer);
        cgSolverShader.SetFloat("a", pCoeff);
        cgSolverShader.SetInt("numNodes", numNodes);
        Dispatch(axpyKernel, numNodes);
        
        // Then: p = r + p (where p is now beta * p)
        cgSolverShader.SetBuffer(axpyKernel, "xBuffer", rBuffer);
        cgSolverShader.SetBuffer(axpyKernel, "yBuffer", pBuffer);
        cgSolverShader.SetFloat("a", rCoeff);
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
        Vector3 fluidInitialBoundsMin = fluidInitialBounds.transform.position - fluidInitialBounds.transform.localScale * 0.5f;
        Vector3 fluidInitialBoundsMax = fluidInitialBounds.transform.position + fluidInitialBounds.transform.localScale * 0.5f;
        
        // Calculate morton code normalization factors on CPU
        Vector3 simulationSize = simulationBoundsMax - simulationBoundsMin;
        
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
        nodesBuffer = new ComputeBuffer(numNodes, sizeof(float) * 3 + sizeof(float) * 6 + sizeof(float) + sizeof(uint) * 3);
        tempNodesBuffer = new ComputeBuffer(numNodes, sizeof(float) * 3 + sizeof(float) * 6 + sizeof(float) + sizeof(uint) * 3);

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
        nodesShader.SetInt("surfaceLayer", surfaceLayer);
        
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

    private void pullVelocities()
    {
        if (nodesShader == null)
        {
            Debug.LogError("Nodes compute shader is not assigned. Please assign `nodesShader` in the inspector.");
            return;
        }

        // Release and recreate neighbors buffer each frame since numNodes changes
        neighborsBuffer?.Release();
        neighborsBuffer = new ComputeBuffer(numNodes, sizeof(uint) * 24);

        pullVelocitiesKernel = nodesShader.FindKernel("pullVelocities");
        nodesShader.SetBuffer(pullVelocitiesKernel, "nodesBuffer", nodesBuffer);
        nodesShader.SetBuffer(pullVelocitiesKernel, "tempNodesBuffer", tempNodesBuffer);
        nodesShader.SetBuffer(pullVelocitiesKernel, "neighborsBuffer", neighborsBuffer);
        nodesShader.SetInt("numNodes", numNodes);
        int threadGroups = Mathf.CeilToInt(numNodes / 512.0f);
        nodesShader.Dispatch(pullVelocitiesKernel, threadGroups, 1, 1);

        copyFaceVelocitiesKernel = nodesShader.FindKernel("copyFaceVelocities");
        nodesShader.SetBuffer(copyFaceVelocitiesKernel, "nodesBuffer", nodesBuffer);
        nodesShader.SetBuffer(copyFaceVelocitiesKernel, "tempNodesBuffer", tempNodesBuffer);
        nodesShader.SetInt("numNodes", numNodes);
        nodesShader.Dispatch(copyFaceVelocitiesKernel, threadGroups, 1, 1);
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

        // Node[] nodesCPU = new Node[numNodes];
        // nodesBuffer.GetData(nodesCPU);
        
        // Calculate the maximum detail cell size (smallest possible cell)
        // With 10 bits per axis, we have 1024 possible values (0-1023)
        // The maximum detail cell size is simulation bounds divided by 1024
        Vector3 simulationBoundsMin = simulationBounds.bounds.min;
        Vector3 simulationBoundsMax = simulationBounds.bounds.max;
        Vector3 simulationSize = simulationBoundsMax - simulationBoundsMin;
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

            // new Color(1f, 0f, 0f),     // Red - Layer 0
            // new Color(0f, 1f, 0f),     // Green - Layer 5
            // new Color(1f, 0.3f, 0f),   // Orange-red - Layer 1
            // new Color(0f, 1f, 0.5f),   // Blue-green - Layer 6
            // new Color(1f, 0.6f, 0f),   // Orange - Layer 2
            // new Color(0f, 1f, 1f),     // Cyan - Layer 7
            // new Color(1f, 1f, 0f),     // Yellow - Layer 3
            // new Color(0f, 0.5f, 1f),   // Light blue - Layer 8
            // new Color(0.5f, 1f, 0f),   // Yellow-green - Layer 4
            // new Color(0f, 0f, 1f),     // Blue - Layer 9
            // new Color(0.5f, 0f, 1f)    // Violet - Layer 10
        };

        nodesCPU = new Node[numNodes];
        nodesBuffer.GetData(nodesCPU);
        
        for (int i = 0; i < numNodes; i++)
        {
            Node node = nodesCPU[i];
            int layerIndex = Mathf.Clamp((int)node.layer, 0, layerColors.Length - 1);
            Gizmos.color = layerColors[layerIndex];
            // float divergence = node.velocities.right - node.velocities.left + node.velocities.top - node.velocities.bottom + node.velocities.front - node.velocities.back;
            // float volume = Mathf.Pow(8, node.layer);
            // float divergenceNormalized = divergence * divergenceMultiplier / volume;
            // float hue = Mathf.Clamp(divergenceNormalized+0.5f, 0, 1);
            // Gizmos.color = Color.HSVToRGB(hue, 1, 1);
            // Gizmos.DrawWireCube(DecodeMorton3D(node), Vector3.one * Mathf.Max(maxDetailCellSize * Mathf.Pow(2, node.layer), 0.01f));
        }

        // // Read neighbors as a flat uint buffer: 24 uints per node (left4,right4,bottom4,top4,front4,back4)
        // if (neighborsBuffer == null) return;
        // uint[] neighborsCPU = new uint[numNodes * 24];
        // neighborsBuffer.GetData(neighborsCPU);
        // Gizmos.color = new Color(1, 0, 0, 1);
        // if (selectedNode >= numNodes) return;
        // Gizmos.DrawCube(DecodeMorton3D(nodesCPU[selectedNode]), Vector3.one * Mathf.Max(maxDetailCellSize * Mathf.Pow(2, Mathf.Min(nodesCPU[selectedNode].layer, layer)), 0.01f));
        // Gizmos.DrawWireCube(DecodeMorton3D(nodesCPU[selectedNode]), Vector3.one * Mathf.Max(maxDetailCellSize * Mathf.Pow(2, Mathf.Min(nodesCPU[selectedNode].layer, layer)), 0.01f));
        // int nb = selectedNode * 24;
        // for (int i = 0; i < 24; i++) {
        //     uint idx = neighborsCPU[nb + i];
        //     if (idx >= numNodes) continue;
        //     Gizmos.color = new Color(0, 1, 0, 0.25f);
        //     Gizmos.DrawCube(DecodeMorton3D(nodesCPU[idx]), Vector3.one * Mathf.Max(maxDetailCellSize * Mathf.Pow(2, Mathf.Min(nodesCPU[idx].layer, layer)), 0.01f));
        //     Gizmos.color = new Color(0, 1, 0, 1);
        //     Gizmos.DrawWireCube(DecodeMorton3D(nodesCPU[idx]), Vector3.one * Mathf.Max(maxDetailCellSize * Mathf.Pow(2, Mathf.Min(nodesCPU[idx].layer, layer)), 0.01f));
        // }


        // Particle[] particlesCPU = new Particle[numParticles];
        // particlesBuffer.GetData(particlesCPU);

        // for (int i = 0; i < numParticles; i++)
        // {
        //     Particle particle = particlesCPU[i];
        //     int layerIndex = Mathf.Clamp((int)particle.layer, 0, layerColors.Length - 1);
        //     Gizmos.color = layerColors[layerIndex];
        //     Gizmos.DrawCube(particle.position, Vector3.one * 0.25f);
        // }
    }

    private void DrawParticles(Camera cam)
    {
        if (particlesMaterial == null || particlesBuffer == null || numParticles <= 0) return;
        if (cam == null) return;

        particlesMaterial.SetBuffer("_Particles", particlesBuffer);
        particlesMaterial.SetFloat("_PointSize", 2.0f);

        particlesMaterial.SetPass(0);
        Graphics.DrawProceduralNow(MeshTopology.Points, numParticles, 1);
    }

    // remove legacy OnRenderObject path
    // void OnRenderObject() { DrawParticles(Camera.current != null ? Camera.current : Camera.main); }

    private Vector3 DecodeMorton3D(Node node)
    {
        // Step 1: Quantize the node position to the appropriate level of detail for this layer
        Vector3 simulationBoundsMin = simulationBounds.bounds.min;
        Vector3 simulationBoundsMax = simulationBounds.bounds.max;
        Vector3 simulationSize = simulationBoundsMax - simulationBoundsMin;
        
        // Calculate the grid resolution for this layer
        // Layer 0: finest detail (1024 cells per axis)
        // Layer 10: coarsest detail (1 cell per axis)
        int gridResolution = (int)Mathf.Pow(2, 10 - node.layer);
        
        // Normalize the node position to morton code range (0-1023)
        Vector3 mortonNormalizationFactor = new Vector3(
            1023.0f / simulationSize.x,
            1023.0f / simulationSize.y,
            1023.0f / simulationSize.z
        );
        
        Vector3 normalizedPos = new Vector3(
            (node.position.x - simulationBoundsMin.x) * mortonNormalizationFactor.x,
            (node.position.y - simulationBoundsMin.y) * mortonNormalizationFactor.y,
            (node.position.z - simulationBoundsMin.z) * mortonNormalizationFactor.z
        );
        
        // Quantize to the grid resolution for this layer
        // This snaps the position to the nearest grid cell at the appropriate level of detail
        float cellSize = 1024.0f / gridResolution;
        Vector3 quantizedPos = new Vector3(
            Mathf.Floor(normalizedPos.x / cellSize) * cellSize + cellSize * 0.5f,
            Mathf.Floor(normalizedPos.y / cellSize) * cellSize + cellSize * 0.5f,
            Mathf.Floor(normalizedPos.z / cellSize) * cellSize + cellSize * 0.5f
        );
        
        // Convert back to world coordinates
        Vector3 quantizedWorldPos = simulationBoundsMin + new Vector3(
            quantizedPos.x / mortonNormalizationFactor.x,
            quantizedPos.y / mortonNormalizationFactor.y,
            quantizedPos.z / mortonNormalizationFactor.z
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
        tempNodesBuffer?.Release();
        neighborsBuffer?.Release();
        divergenceBuffer?.Release();
        residualBuffer?.Release();
        pBuffer?.Release();
        ApBuffer?.Release();
        pressureBuffer?.Release();
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
        sortShader.SetBuffer(clearBuffer32Kernel, "output", buffer);
        sortShader.SetInt("count", (int)count);
        int threadGroups = (int)((count + 511) / 512);
        sortShader.Dispatch(clearBuffer32Kernel, threadGroups, 1, 1);
    }
}