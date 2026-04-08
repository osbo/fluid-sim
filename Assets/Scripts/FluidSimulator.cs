using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.InputSystem;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using System.IO;
using Unity.Profiling;

// Main FluidSimulator class - split into partial classes for better organization
// Core simulation logic: particle management, main loop, initialization
// See: FluidEnums.cs, FluidRenderer.cs, FluidPreconditioner.cs,
//      FluidOctree.cs, FluidSolver.cs, FluidLeafOnlyInputs.cs, FluidLeafOnlyWeights.cs,
//      FluidLeafOnlyPrecondApply.cs, FluidSimulator.LoadEditorAssets.cs, RadixSort.cs
public partial class FluidSimulator : MonoBehaviour
{
    BoxCollider simulationBounds;
    BoxCollider fluidInitialBounds;

    ComputeShader radixSortShader;
    ComputeShader particlesShader;
    ComputeShader nodesPrefixSumsShader;
    ComputeShader nodesShader;
    ComputeShader cgSolverShader;
    ComputeShader csrBuilderShader;
    public int numParticles = 1048576;

    // CG Solver parameters
    public int maxCgIterations = 400;
    public float convergenceThreshold = 1e-05f;
    [Tooltip("Unused: PCG always runs maxCgIterations with no mid-loop convergence check. Enable per-frame debug timing log to print final ||r||² after the solve.")]
    public int cgConvergenceCheckInterval = 5;
    
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
    private int writeUniqueCountKernel;
    private int writeNodeCountKernel;
    private int radixPrefixSumKernelId;
    private int radixPrefixFixupKernelId;
    private int radixPrefixSumSolverMainKernelId;
    private int radixPrefixSumSolverAuxKernelId;
    private int radixPrefixFixupSolverKernelId;
    private int buildSolverIndirectArgsKernelId;
    private int findNeighborsKernel;
    private int findReverseKernel;
    private int interpolateFaceVelocitiesKernel;
    private int copyFaceVelocitiesKernel;
    private int calculateDivergenceKernel;
    private int axpyKernel;
    private int dotProductKernel;
    private int applyPressureGradientKernel;
    private int updateParticlesKernel;
    private int applyExternalForcesKernel;
    private int solvePressureIterationKernel;
    private int initializePressureBuffersKernel;
    private int initializePhiKernel;
    private int propagatePhiKernel;
    private int calculateDensityGradientKernel;
    // Cached solver kernel IDs (set once in InitSolverKernels)
    private int buildMatrixAKernelId;
    private int spmvCsrKernelId;
    private int applyJacobiKernelId;
    private int globalReduceSumKernelId;
    private int copyFloatKernelId;
    private int scaleKernelId;
    private int computeAlphaKernelId;
    private int computeBetaKernelId;
    private int storeRhoFromDotKernelId;
    private int axpyAlphaKernelId;
    private int axpyNegAlphaKernelId;
    private int scaleAddBetaKernelId;
    // Cached CSR builder kernel IDs
    private int csrCountNnzKernelId;
    private int csrFinalizeRowPtrKernelId;
    private int csrFillKernelId;
    // Cached octree/prefix-sum kernel IDs
    private int copyNodesKernelId;
    private int writeDispatchArgsKernelId;
    private int writeUniqueCountKernelId2;  // alias for writeUniqueCountKernel (already a field above)
    private int writeNodeCountKernelId;
    private int scatterActivesKernelId;
    private int markActiveNodesKernelId;
    private int npsPrefixSumKernelId;
    private int npsPrefixFixupKernelId;
    private int writeDispatchArgsFromCountKernelId;
    private int copyUintBufferKernelId;
    
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
    // AoS Node struct in GPU memory; morton-heavy searches also use mortonCodesBuffer (see Nodes.compute). A full SoA split
    // (positions, layers, masses, …) would further cut bandwidth in kernels that only need a subset of fields.
    private ComputeBuffer nodesBuffer;
    private ComputeBuffer nodesBufferOld;
    private ComputeBuffer tempNodesBuffer;
    private ComputeBuffer mortonCodesBuffer;
    private ComputeBuffer neighborsBuffer;
    public ComputeBuffer reverseNeighborsBuffer;
    private ComputeBuffer divergenceBuffer;
    private ComputeBuffer pBuffer;
    private ComputeBuffer ApBuffer;
    private ComputeBuffer residualBuffer;
    private ComputeBuffer pressureBuffer;
    // NEW: Buffer for standard Laplacian Matrix A
    // Format: [25 * numNodes] where Slot 0 = Diagonal, 1-24 = Neighbors
    private ComputeBuffer matrixABuffer;
    private ComputeBuffer phiBuffer;
    private ComputeBuffer phiBuffer_Read;
    private ComputeBuffer dirtyFlagBuffer;
    private ComputeBuffer zVectorBuffer; // Preconditioned residual vector 'z' for PCG
    private ComputeBuffer cgAlphaBuffer;
    private ComputeBuffer cgBetaBuffer;
    private ComputeBuffer cgRhoBuffer;
    /// <summary>12 uints: 4× DispatchIndirect triples — [0] SpMV/256-thread, [1] 512-thread, [2] 1-group, [3] prefix-pass1.
    /// Replaces cgPcgIndirectArgsBuffer; first 9 uints are written by BuildSolverIndirectArgs each frame.</summary>
    private ComputeBuffer solverIndirectArgsBuffer;
    private ComputeBuffer solverReductionCount256Buffer; // ceil(n/256) for GlobalReduceSum after SpMV
    private ComputeBuffer solverReductionCount512Buffer; // ceil(n/512) for GlobalReduceSum after DotProduct
    private ComputeBuffer solverPrefixLenBuffer;         // [0]=numNodes [1]=pfx1, for CSR prefix sum kernels
    private ComputeBuffer diffusionGradientBuffer; // Precomputed normalized density gradient per node
    private ComputeBuffer dispatchArgsBuffer;       // 3-uint indirect dispatch args for DispatchIndirect
    private ComputeBuffer particlePrefixElementCountBuffer; // [0] = numParticles for find-unique prefix scans
    private int maxNodesCapacity;
    private int maxPrefixThreadGroups;
    private int maxAuxBlocks;
    private readonly uint[] gpuNodeCountReadback = new uint[1];
    // CSR matrix representation of A
    private ComputeBuffer nnzPerNode;   // Per-row nnz counts
    private ComputeBuffer csrRowPtr;    // Row pointer (size numNodes + 1)
    private ComputeBuffer csrColIndices;
    private ComputeBuffer csrRowIndices; // Explicit row indices (edge_index[0])
    private ComputeBuffer csrValues;
    
    // Helper array to avoid allocating every frame for GPU reduction
    private float[] reductionResult = new float[1];
    
    
    // Number of nodes, active nodes, and unique active nodes
    private int numNodes;
    
    // Public accessors for external scripts (e.g., FluidMeshRenderer)
    public ComputeBuffer NodesBuffer => nodesBuffer;
    public ComputeBuffer ParticlesBuffer => particlesBuffer;
    public ComputeBuffer PhiBuffer => phiBuffer;
    public int NumNodes => numNodes;
    public int NumParticlesForRender => numParticles;
    public Bounds WorldSimulationBounds => simulationBounds != null ? simulationBounds.bounds : new Bounds(Vector3.zero, Vector3.one);
    public float MaxDetailCellSize => maxDetailCellSize;
    private int numUniqueNodes; // rename to numUniqueNodes
    private int layer;
    public float gravity = 100.0f;
    public bool useRealTime = false; // When true, uses Time.deltaTime instead of fixed frameRate
    public float frameRate = 30.0f;
    [Tooltip("Finest octree layer for the sim; collider voxel grid uses R = 2^(10 - minLayer) along each axis (same as Nodes.compute kMortonAxisBits).")]
    [Range(0, 10)] public int minLayer = 4;
    [Range(0, 10)] public int maxLayer = 10;

    /// <summary>Must match kMortonAxisBits in Nodes.compute.</summary>
    public const int MortonAxisBits = 10;

    /// <summary>Collider voxel resolution per axis: 2^(MortonAxisBits - minLayer).</summary>
    public int SolidVoxelResolution => 1 << (MortonAxisBits - Mathf.Clamp(minLayer, 0, MortonAxisBits));
    [Tooltip("When per-frame debug timing log is enabled: each frame, GPU readback of nodesBuffer and layer histogram text for the summary. No effect when that master toggle is off.")]
    public bool logLayerCountsPerFrame = true;

    [Header("Debug: frame stats / logging")]
    [Tooltip("Master toggle for expensive per-frame diagnostics. When off (default): no frame summary Debug.Log, no cumulative timing averages, no layer-count readback, and GPU sync for section timing is skipped. Stopwatches still run (cheap); use Unity Profiler FluidSim.* markers. Turn on only while profiling.")]
    public bool enablePerFrameDebugTimingLog = false;

    [Header("Debug: section timing vs GPU")]
    [Tooltip("Only when per-frame debug timing log is enabled. Each timed simulation step runs AsyncGPUReadback (tiny slices) before and after the step so Stopwatch ms ≈ GPU work for that step. Very slow. Off = timers measure mostly CPU enqueue time.")]
    public bool gpuSyncForSectionTimingReport = false;

    public PreconditionerType preconditioner = PreconditionerType.Jacobi;

    private int frameNumber = 0;
    private double cumulativeFrameTimeMs = 0.0;
    private double cumulativeCgIterations = 0.0;
    private int cgSolveFrameCount = 0;
    private float averageCgIterations = 0.0f;
    private int lastCgIterations = 0;
    
    // Pause/resume functionality
    private bool isPaused = false;
    
    private System.Diagnostics.Stopwatch totalOctreeSw;
    private double lastRenderTimeMs;

    // Profiler markers for Unity Profiler GPU timeline
    static readonly ProfilerMarker s_FrameMarker            = new ProfilerMarker("FluidSim.Frame");
    static readonly ProfilerMarker s_SortMarker             = new ProfilerMarker("FluidSim.Sort");
    static readonly ProfilerMarker s_FindUniqueMarker       = new ProfilerMarker("FluidSim.FindUnique");
    static readonly ProfilerMarker s_CreateLeavesMarker     = new ProfilerMarker("FluidSim.CreateLeaves");
    static readonly ProfilerMarker s_LayerLoopMarker        = new ProfilerMarker("FluidSim.LayerLoop");
    static readonly ProfilerMarker s_FindNeighborsMarker    = new ProfilerMarker("FluidSim.FindNeighbors");
    static readonly ProfilerMarker s_GradientsMarker        = new ProfilerMarker("FluidSim.CalculateGradients");
    static readonly ProfilerMarker s_StoreVelocitiesMarker  = new ProfilerMarker("FluidSim.StoreOldVelocities");
    static readonly ProfilerMarker s_ExternalForcesMarker   = new ProfilerMarker("FluidSim.ApplyExternalForces");
    static readonly ProfilerMarker s_NodeCountReadbackMarker= new ProfilerMarker("FluidSim.NodeCountReadback");
    static readonly ProfilerMarker s_SolvePressureMarker    = new ProfilerMarker("FluidSim.SolvePressure");
    static readonly ProfilerMarker s_UpdateParticlesMarker  = new ProfilerMarker("FluidSim.UpdateParticles");
    static readonly ProfilerMarker s_GpuTimingSyncMarker    = new ProfilerMarker("FluidSim.GpuTimingSync");
    
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
    private readonly int[] layerCountHistogramScratch = new int[16];
    private readonly int[] spatialDupPerLayerScratch = new int[16];
    private readonly Dictionary<ulong, int> spatialOverlapFirstIndexScratch = new Dictionary<ulong, int>(65536);
    private readonly StringBuilder layerCountLogSb = new StringBuilder(128);
    private Particle[] particlesCPU;
    private string str;

    // Solid colliders (static mesh obstacles)
    [Header("Static Colliders")]
    [Tooltip("Parent GameObject whose MeshFilter children define static solid obstacles.")]
    public GameObject collidersRoot;
    private ComputeBuffer solidVoxelsBuffer;
    public ComputeBuffer SolidVoxelsBuffer => solidVoxelsBuffer;

    // Training data recorder
    public TrainingDataRecorder recorder;

    internal void SetLastRenderTimeMs(double ms) => lastRenderTimeMs = ms;

    // Helper method for resizing buffers
    private void ResizeBuffer(ref ComputeBuffer buffer, int count, int stride)
    {
        if (buffer != null && buffer.count >= count) return; // Capacity is sufficient

        buffer?.Release();
        // Allocate next power of two to prevent frequent resizing
        int newSize = Mathf.NextPowerOfTwo(Mathf.Max(count, 512));
        buffer = new ComputeBuffer(newSize, stride);
    }
    
    private void ValidateOctreeLayers()
    {
        // ProcessNodes first pass uses layer == minLayer + 1 and requires node.layer >= layer.
        // The layer loop runs for layer in (minLayer + 1) .. maxLayer inclusive.
        if (maxLayer <= minLayer)
        {
            Debug.LogWarning($"FluidSimulator: maxLayer ({maxLayer}) must be > minLayer ({minLayer}). Clamping maxLayer to {minLayer + 1}.");
            maxLayer = minLayer + 1;
        }
    }

    private void OnValidate()
    {
        ValidateOctreeLayers();
    }

    void Awake()
    {
        LoadEditorDefaultComputeShaders();
        ResolveBoundsColliders();
    }

    void ResolveBoundsColliders()
    {
        if (simulationBounds == null)
        {
            var go = GameObject.Find("Simulation Bounds");
            if (go != null)
                simulationBounds = go.GetComponent<BoxCollider>();
        }
        if (fluidInitialBounds == null)
        {
            var go = GameObject.Find("Fluid Initial Bounds");
            if (go != null)
                fluidInitialBounds = go.GetComponent<BoxCollider>();
        }
        if (collidersRoot == null)
        {
            var go = GameObject.Find("Collider");
            if (go != null)
                collidersRoot = go;
        }
    }

    void Start()
    {
        ValidateOctreeLayers();
        InitializeParticleSystem();
        // InitializeInitialParticles();

        BakeAndUploadSolidVoxels();

        // Cache all kernel IDs once so hot paths never call FindKernel
        InitSolverKernels();
        InitOctreeKernels();
        TryLoadLeafOnlyWeightsFromDisk();

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

    private void BakeAndUploadSolidVoxels()
    {
        int R = SolidVoxelResolution;
        int total = R * R * R;

        uint[] voxels = (collidersRoot != null)
            ? MeshColliderBaker.Bake(collidersRoot, simulationBounds.bounds, R)
            : new uint[total];

        solidVoxelsBuffer?.Release();
        solidVoxelsBuffer = new ComputeBuffer(total, sizeof(uint));
        solidVoxelsBuffer.SetData(voxels);

        // Teleport any particles that initialized inside solid voxels
        if (particlesShader != null)
        {
            int kernel = particlesShader.FindKernel("RemoveSolidParticles");
            particlesShader.SetBuffer(kernel, "particlesBuffer", particlesBuffer);
            particlesShader.SetBuffer(kernel, "solidVoxelsBuffer", solidVoxelsBuffer);
            particlesShader.SetInt("solidVoxelResolution", R);
            particlesShader.SetInt("numParticles", numParticles);
            particlesShader.SetVector("fluidInitialBoundsMin", fluidInitialBoundsMin);
            int groups = Mathf.CeilToInt(numParticles / 512.0f);
            particlesShader.Dispatch(kernel, groups, 1, 1);
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
            particlesCPU[indices[i]].velocity = new Vector3(100.0f, 0.0f, 0.0f);
        }

        particlesBuffer.SetData(particlesCPU);
    }

    /// <summary>Optional: tiny async readbacks + wait so section Stopwatches approximate GPU time (see <see cref="gpuSyncForSectionTimingReport"/>).</summary>
    private void GpuTimingReportFlushGpu(bool includePressureBuffer = false)
    {
        if (!enablePerFrameDebugTimingLog || !gpuSyncForSectionTimingReport)
            return;
        using (s_GpuTimingSyncMarker.Auto())
        {
            if (particlesBuffer != null)
            {
                var r = AsyncGPUReadback.Request(particlesBuffer, 4, 0);
                r.WaitForCompletion();
            }
            if (nodeCount != null)
            {
                var r = AsyncGPUReadback.Request(nodeCount, sizeof(uint), 0);
                r.WaitForCompletion();
            }
            if (includePressureBuffer && pressureBuffer != null)
            {
                var r = AsyncGPUReadback.Request(pressureBuffer, sizeof(float), 0);
                r.WaitForCompletion();
            }
        }
    }

    void Update()
    {
        // Check for pause/resume toggle (space bar)
        if (Keyboard.current != null && Keyboard.current.spaceKey.wasPressedThisFrame)
        {
            isPaused = !isPaused;
            Debug.Log(isPaused ? "Simulation PAUSED - Camera and rendering still active" : "Simulation RESUMED");
        }
        
        // Skip simulation steps if paused (rendering will continue via OnEndCameraRendering)
        if (isPaused)
        {
            return;
        }
        
        layer = minLayer;
        
        var frameSw = System.Diagnostics.Stopwatch.StartNew();
        using var _ = s_FrameMarker.Auto();

        // Calculate maxDetailCellSize for volume calculations
        // Use the normalized simulation bounds (simulationBoundsMin is now Vector3.zero)
        Vector3 simulationSize = simulationBoundsMax;
        maxDetailCellSize = Mathf.Min(simulationSize.x, simulationSize.y, simulationSize.z) / 1024.0f;

        var sortSw = new System.Diagnostics.Stopwatch();
        var findUniqueSw = new System.Diagnostics.Stopwatch();
        var createLeavesSw = new System.Diagnostics.Stopwatch();
        var layerLoopSw = new System.Diagnostics.Stopwatch();
        var findNeighborsSw = new System.Diagnostics.Stopwatch();
        var calculateGradientsSw = new System.Diagnostics.Stopwatch();
        var computeLevelSetSw = new System.Diagnostics.Stopwatch();
        var storeOldVelocitiesSw = new System.Diagnostics.Stopwatch();
        var applyExternalForcesSw = new System.Diagnostics.Stopwatch();

        // GPU labels: one ExecuteCommandBuffer when this scope ends, *before* nodeCount readback below (nested inner scopes do not flush).
        using (new GpuProfileSection(this, "FluidSim.PrePressure"))
        {
        // Step 1: Sort particles
        GpuTimingReportFlushGpu(false);
        sortSw.Restart();
        using (new GpuProfileSection(this, "FluidSim.Sort"))
        using (s_SortMarker.Auto())
            SortParticles();
        GpuTimingReportFlushGpu(false);
        sortSw.Stop();

        // Step 2: Find unique particles and create leaves
        GpuTimingReportFlushGpu(false);
        findUniqueSw.Restart();
        using (new GpuProfileSection(this, "FluidSim.FindUnique"))
        using (s_FindUniqueMarker.Auto())
            findUniqueParticles();
        GpuTimingReportFlushGpu(false);
        findUniqueSw.Stop();

        GpuTimingReportFlushGpu(false);
        createLeavesSw.Restart();
        using (new GpuProfileSection(this, "FluidSim.CreateLeaves"))
        using (s_CreateLeavesMarker.Auto())
            CreateLeaves();
        GpuTimingReportFlushGpu(false);
        createLeavesSw.Stop();

        // Step 3: Layer loop (layers 1-10)
        GpuTimingReportFlushGpu(false);
        layerLoopSw.Restart();
        using (new GpuProfileSection(this, "FluidSim.LayerLoop"))
        using (s_LayerLoopMarker.Auto())
        {
            for (layer = layer + 1; layer <= maxLayer; layer++)
            {
                findUniqueNodes();
                ProcessNodes();
                compactNodes();
                // mortonCodesBuffer is refreshed inside scatterActives (fused with compaction).
            }
        }
        GpuTimingReportFlushGpu(false);
        layerLoopSw.Stop();

        // Step 4: Find neighbors (GPU: WriteIndirectArgsFromCount + DispatchIndirect — no CPU count needed)
        GpuTimingReportFlushGpu(false);
        findNeighborsSw.Restart();
        using (new GpuProfileSection(this, "FluidSim.FindNeighbors"))
        using (s_FindNeighborsMarker.Auto())
            findNeighbors();
        GpuTimingReportFlushGpu(false);
        findNeighborsSw.Stop();

        // Step 4.5: Calculate density gradients (indirect from GPU nodeCount)
        GpuTimingReportFlushGpu(false);
        calculateGradientsSw.Restart();
        using (new GpuProfileSection(this, "FluidSim.CalculateGradients"))
        using (s_GradientsMarker.Auto())
            CalculateDensityGradients();
        GpuTimingReportFlushGpu(false);
        calculateGradientsSw.Stop();

        // Step 5: Compute level set (distance field)
        GpuTimingReportFlushGpu(false);
        computeLevelSetSw.Restart();
        // ComputeLevelSet();
        GpuTimingReportFlushGpu(false);
        computeLevelSetSw.Stop();

        // Step 6: Store old velocities for FLIP method
        GpuTimingReportFlushGpu(false);
        storeOldVelocitiesSw.Restart();
        using (new GpuProfileSection(this, "FluidSim.StoreOldVelocities"))
        using (s_StoreVelocitiesMarker.Auto())
            StoreOldVelocities();
        GpuTimingReportFlushGpu(false);
        storeOldVelocitiesSw.Stop();

        // Step 7: Apply external forces (indirect from GPU nodeCount)
        GpuTimingReportFlushGpu(false);
        applyExternalForcesSw.Restart();
        using (new GpuProfileSection(this, "FluidSim.ApplyExternalForces"))
        using (s_ExternalForcesMarker.Auto())
            ApplyExternalForces();
        GpuTimingReportFlushGpu(false);
        applyExternalForcesSw.Stop();
        }

        // One uint sync readback: CPU must have correct numNodes this frame for SolvePressure (LeafOnly),
        // BuildLayerCountsSummaryForLog, rendering instance counts, etc. GPU indirect dispatches already use nodeCount buffer.
        using (s_NodeCountReadbackMarker.Auto())
        {
            nodeCount.GetData(gpuNodeCountReadback);
            numNodes = (int)gpuNodeCountReadback[0];
        }
        string layerCountsLine = null;
        if (enablePerFrameDebugTimingLog && logLayerCountsPerFrame)
            layerCountsLine = BuildLayerCountsSummaryForLog();

        // Step 9: Solve pressure (GPU labels are sub-scopes inside SolvePressure — LeafOnly/Neural use immediate dispatches)
        var solvePressureSw = new System.Diagnostics.Stopwatch();
        GpuTimingReportFlushGpu(false);
        solvePressureSw.Restart();
        using (s_SolvePressureMarker.Auto())
            SolvePressure();
        GpuTimingReportFlushGpu(true);
        solvePressureSw.Stop();

        // Step 10: Update particles
        var updateParticlesSw = new System.Diagnostics.Stopwatch();
        GpuTimingReportFlushGpu(false);
        updateParticlesSw.Restart();
        using (new GpuProfileSection(this, "FluidSim.UpdateParticles"))
        using (s_UpdateParticlesMarker.Auto())
            UpdateParticles();
        GpuTimingReportFlushGpu(false);
        updateParticlesSw.Stop();

        // Frame timing summary (optional; Console + string build are costly on the main thread)
        frameSw.Stop();
        if (enablePerFrameDebugTimingLog)
        {
            frameNumber++;
            cumulativeFrameTimeMs += frameSw.Elapsed.TotalMilliseconds;
            double averageFrameTimeMs = cumulativeFrameTimeMs / Math.Max(1, frameNumber);
            string averageCgIterationsText = cgSolveFrameCount > 0 ? averageCgIterations.ToString("F2") : "N/A";
            string timingModeNote = gpuSyncForSectionTimingReport
                ? "• Timing mode: GPU sync ON (section ms include AsyncGPUReadback waits; not a full device idle guarantee).\n"
                : "";
            Debug.Log($"Frame {frameNumber} Summary:\n" +
                     $"• Total Frame: {frameSw.Elapsed.TotalMilliseconds:F2} ms\n" +
                     $"• Avg Frame Time: {averageFrameTimeMs:F2} ms\n" +
                     (timingModeNote.Length > 0 ? timingModeNote : "") +
                     $"• # Nodes: {numNodes}\n" +
                     (layerCountsLine != null ? $"• {layerCountsLine}\n" : "") +
                     $"• Sort: {sortSw.Elapsed.TotalMilliseconds:F2} ms\n" +
                     $"• Find Unique: {findUniqueSw.Elapsed.TotalMilliseconds:F2} ms\n" +
                     $"• Create Leaves: {createLeavesSw.Elapsed.TotalMilliseconds:F2} ms\n" +
                     $"• Layer Loop: {layerLoopSw.Elapsed.TotalMilliseconds:F2} ms\n" +
                     $"• Find Neighbors: {findNeighborsSw.Elapsed.TotalMilliseconds:F2} ms\n" +
                     $"• Calculate Gradients: {calculateGradientsSw.Elapsed.TotalMilliseconds:F2} ms\n" +
                     $"• Compute Level Set: {computeLevelSetSw.Elapsed.TotalMilliseconds:F2} ms\n" +
                     $"• Store Old Velocities: {storeOldVelocitiesSw.Elapsed.TotalMilliseconds:F2} ms\n" +
                     $"• Apply External Forces: {applyExternalForcesSw.Elapsed.TotalMilliseconds:F2} ms\n" +
                     $"• Solve Pressure: {solvePressureSw.Elapsed.TotalMilliseconds:F2} ms\n" +
                     $"• Update Particles: {updateParticlesSw.Elapsed.TotalMilliseconds:F2} ms\n" +
                     $"• Rendering: {lastRenderTimeMs:F2} ms\n" +
                     $"• CG Iterations: {lastCgIterations}\n" +
                     $"• Avg CG Iterations: {averageCgIterationsText}");
        }
    }

    /// <summary>Layer histogram after <see cref="numNodes"/> readback; caller must gate with <see cref="enablePerFrameDebugTimingLog"/> and <see cref="logLayerCountsPerFrame"/>.</summary>
    private string BuildLayerCountsSummaryForLog()
    {
        if (!logLayerCountsPerFrame)
            return null;
        if (numNodes <= 0 || nodesBuffer == null)
            return "Nodes per layer: (none)";

        if (nodesCPU == null || nodesCPU.Length < numNodes)
            nodesCPU = new Node[Mathf.NextPowerOfTwo(Mathf.Max(numNodes, 512))];

        nodesBuffer.GetData(nodesCPU, 0, 0, numNodes);

        AnalyzeNodesCpuForLayerHistogramAndCellDupCount(activeNodesOnly: false, out int dupCount);

        int[] h = layerCountHistogramScratch;
        layerCountLogSb.Clear();
        layerCountLogSb.Append("Nodes per layer:");
        bool any = false;
        for (int L = 0; L < h.Length; L++)
        {
            if (h[L] == 0) continue;
            layerCountLogSb.Append(' ').Append('L').Append(L).Append('=').Append(h[L]);
            any = true;
        }

        if (!any)
            layerCountLogSb.Append(" (no counts — unexpected)");

        if (dupCount > 0)
        {
            int[] dupPerLayer = spatialDupPerLayerScratch;
            layerCountLogSb.Append(" | OVERLAPS(same cell @ layer): ").Append(dupCount).Append(" (");
            bool firstDup = true;
            for (int L = 0; L < dupPerLayer.Length; L++)
            {
                if (dupPerLayer[L] == 0) continue;
                if (!firstDup) layerCountLogSb.Append(", ");
                layerCountLogSb.Append('L').Append(L).Append('=').Append(dupPerLayer[L]);
                firstDup = false;
            }
            layerCountLogSb.Append(") [key = (layer, morton>>3*layer)]");
        }

        return layerCountLogSb.ToString();
    }

    /// <summary>Solver cell identity: same <paramref name="layerIndex"/> and same morton prefix at that scale.</summary>
    private static ulong SpatialCellKeyAtLayer(uint mortonCode, int layerIndex)
    {
        int L = layerIndex;
        if (L < 0) L = 0;
        int shift = L * 3;
        uint prefix = shift >= 32 ? 0u : mortonCode >> shift;
        return ((ulong)(uint)L << 32) | prefix;
    }

    /// <summary>Matches GPU octree cell identity: same layer and same morton prefix at that layer = same spatial cell.</summary>
    private static ulong SpatialCellKey(in Node n) =>
        SpatialCellKeyAtLayer(n.mortonCode, (int)n.layer);

    /// <summary>Assumes <see cref="nodesCPU"/> is filled for <see cref="numNodes"/>. Clears and fills layer histogram and duplicate-cell counts (same layer + morton prefix).</summary>
    private void AnalyzeNodesCpuForLayerHistogramAndCellDupCount(bool activeNodesOnly, out int dupCount)
    {
        int[] h = layerCountHistogramScratch;
        int[] dupPerLayer = spatialDupPerLayerScratch;
        Array.Clear(h, 0, h.Length);
        Array.Clear(dupPerLayer, 0, dupPerLayer.Length);
        spatialOverlapFirstIndexScratch.Clear();

        dupCount = 0;

        for (int i = 0; i < numNodes; i++)
        {
            ref Node n = ref nodesCPU[i];
            if (activeNodesOnly && n.active == 0)
                continue;

            int L = (int)n.layer;
            if (L >= 0 && L < h.Length)
                h[L]++;

            ulong key = SpatialCellKey(in n);
            if (spatialOverlapFirstIndexScratch.ContainsKey(key))
            {
                dupCount++;
                if (L >= 0 && L < dupPerLayer.Length)
                    dupPerLayer[L]++;
            }
            else
                spatialOverlapFirstIndexScratch[key] = i;
        }
    }

    private void CalculateDensityGradients()
    {
        if (nodesShader == null)
        {
            Debug.LogError("Nodes compute shader is not assigned.");
            return;
        }

        BindNodesOctreeCounts();
        WriteIndirectArgsFromCountBuffer(nodeCount);
        nodesShader.SetBuffer(calculateDensityGradientKernel, "nodesBuffer", nodesBuffer);
        nodesShader.SetBuffer(calculateDensityGradientKernel, "neighborsBuffer", neighborsBuffer);
        nodesShader.SetBuffer(calculateDensityGradientKernel, "diffusionGradientBuffer", diffusionGradientBuffer);
        GpuProfileDispatchIndirect(nodesShader, calculateDensityGradientKernel, dispatchArgsBuffer, 0);
    }

    private void StoreOldVelocities()
    {
        const int nodeStride = sizeof(float) * 3 + sizeof(float) * 3 + sizeof(float) * 6 + sizeof(float) + sizeof(uint) * 3;
        ResizeBuffer(ref nodesBufferOld, maxNodesCapacity, nodeStride);

        // GPU copy: copyNodes kernel writes tempNodesBuffer[i] → nodesBuffer[i],
        // so bind src→tempNodesBuffer slot, dst→nodesBuffer slot.
        BindNpsPrefixCount(nodeCount);
        nodesPrefixSumsShader.SetBuffer(copyNodesKernelId, "tempNodesBuffer", nodesBuffer);
        nodesPrefixSumsShader.SetBuffer(copyNodesKernelId, "nodesBuffer", nodesBufferOld);
        int copyGroups = Mathf.Max(1, (maxNodesCapacity + 511) / 512);
        GpuProfileDispatchCompute(nodesPrefixSumsShader, copyNodesKernelId, copyGroups, 1, 1);
    }

    private void UpdateParticles()
    {
        if (particlesShader == null)
        {
            Debug.LogError("Particles compute shader is missing.");
            return;
        }

        updateParticlesKernel = particlesShader.FindKernel("UpdateParticles");

        // need: nodes buffer, nodes buffer old, particles buffer, numNodes, numParticles
        particlesShader.SetBuffer(updateParticlesKernel, "nodesBuffer", nodesBuffer);
        particlesShader.SetBuffer(updateParticlesKernel, "mortonCodesBuffer", mortonCodesBuffer);
        particlesShader.SetBuffer(updateParticlesKernel, "nodesBufferOld", nodesBufferOld);
        particlesShader.SetBuffer(updateParticlesKernel, "particlesBuffer", particlesBuffer);
        particlesShader.SetBuffer(updateParticlesKernel, "diffusionGradientBuffer", diffusionGradientBuffer);
        particlesShader.SetBuffer(updateParticlesKernel, "nodeCountBuffer", nodeCount);
        particlesShader.SetBuffer(updateParticlesKernel, "solidVoxelsBuffer", solidVoxelsBuffer);
            particlesShader.SetInt("solidVoxelResolution", SolidVoxelResolution);
        particlesShader.SetInt("numParticles", numParticles);
        float deltaTime = useRealTime ? Time.deltaTime : (1 / frameRate);
        particlesShader.SetFloat("deltaTime", deltaTime);
        particlesShader.SetFloat("gravity", gravity);
        particlesShader.SetFloat("maxDetailCellSize", maxDetailCellSize);
        particlesShader.SetVector("mortonNormalizationFactor", mortonNormalizationFactor);
        particlesShader.SetFloat("mortonMaxValue", mortonMaxValue);
        particlesShader.SetVector("simulationBoundsMin", simulationBoundsMin);
        particlesShader.SetVector("simulationBoundsMax", simulationBoundsMax);
        particlesShader.SetInt("minLayer", minLayer);
        particlesShader.SetInt("maxLayer", maxLayer);
        int threadGroups = Mathf.CeilToInt(numParticles / 512.0f);
        GpuProfileDispatchCompute(particlesShader, updateParticlesKernel, threadGroups, 1, 1);
    }

    private void InitializeParticleSystem()
    {
        if (simulationBounds == null || fluidInitialBounds == null)
        {
            Debug.LogError(
                "FluidSimulator: assign simulation bounds by placing BoxColliders on GameObjects named \"Simulation Bounds\" and \"Fluid Initial Bounds\", or add them to the scene.");
            return;
        }

        // Get kernel index
        if (particlesShader == null)
        {
            Debug.LogError("Particles compute shader is missing. In the Editor it loads from Assets/Scripts/Particles.compute automatically.");
            return;
        }
        initializeParticlesKernel = particlesShader.FindKernel("InitializeParticles");
        
        // Create buffers
        particlesBuffer = new ComputeBuffer(numParticles, sizeof(float) * 3 + sizeof(float) * 3 + sizeof(uint)); // 12 + 12 + 4 = 28 bytes
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

        AllocateOctreeBuffersToCapacity();

        radixSort?.ReleaseBuffers();
        radixSort = null;
        if (radixSortShader != null)
            radixSort = new RadixSort(radixSortShader, (uint)numParticles);
    }

    /// <summary>Preallocate octree/prefix buffers so the update loop never resizes or stalls on GPU count readbacks mid-pipeline.</summary>
    private void AllocateOctreeBuffersToCapacity()
    {
        maxNodesCapacity = Mathf.Max(512, Mathf.CeilToInt(numParticles * 1.5f));
        maxPrefixThreadGroups = Mathf.Max(1, (maxNodesCapacity + 1023) / 1024);
        maxAuxBlocks = maxPrefixThreadGroups;

        int nodeStride = sizeof(float) * 3 + sizeof(float) * 3 + sizeof(float) * 6 + sizeof(float) + sizeof(uint) * 3;

        indicators?.Release();
        prefixSums?.Release();
        uniqueIndices?.Release();
        aux?.Release();
        aux2?.Release();
        nodesBuffer?.Release();
        tempNodesBuffer?.Release();
        mortonCodesBuffer?.Release();
        neighborsBuffer?.Release();
        reverseNeighborsBuffer?.Release();
        diffusionGradientBuffer?.Release();
        particlePrefixElementCountBuffer?.Release();

        indicators = new ComputeBuffer(maxNodesCapacity, sizeof(uint));
        prefixSums = new ComputeBuffer(maxNodesCapacity, sizeof(uint));
        uniqueIndices = new ComputeBuffer(maxNodesCapacity, sizeof(uint));
        aux = new ComputeBuffer(maxAuxBlocks, sizeof(uint));
        aux2 = new ComputeBuffer(maxAuxBlocks, sizeof(uint));
        if (uniqueCount == null) uniqueCount = new ComputeBuffer(1, sizeof(uint));
        if (nodeCount == null) nodeCount = new ComputeBuffer(1, sizeof(uint));
        if (auxSmall == null) auxSmall = new ComputeBuffer(1, sizeof(uint));
        if (dispatchArgsBuffer == null)
            dispatchArgsBuffer = new ComputeBuffer(3, sizeof(uint), ComputeBufferType.IndirectArguments);

        nodesBuffer = new ComputeBuffer(maxNodesCapacity, nodeStride);
        tempNodesBuffer = new ComputeBuffer(maxNodesCapacity, nodeStride);
        mortonCodesBuffer = new ComputeBuffer(maxNodesCapacity, sizeof(uint));
        neighborsBuffer = new ComputeBuffer(maxNodesCapacity * 24, sizeof(uint));
        reverseNeighborsBuffer = new ComputeBuffer(maxNodesCapacity * 24, sizeof(uint));
        diffusionGradientBuffer = new ComputeBuffer(maxNodesCapacity, sizeof(float) * 3);
        particlePrefixElementCountBuffer = new ComputeBuffer(1, sizeof(uint));

        // Pre-allocate solver buffers to max capacity so SolvePressure never resizes mid-frame.
        int solverCap = Mathf.NextPowerOfTwo(Mathf.Max(maxNodesCapacity, 512));
        int maxNnz    = solverCap * 25;
        divergenceBuffer?.Release(); divergenceBuffer = new ComputeBuffer(solverCap, sizeof(float));
        residualBuffer?.Release();   residualBuffer   = new ComputeBuffer(solverCap, sizeof(float));
        pBuffer?.Release();          pBuffer          = new ComputeBuffer(solverCap, sizeof(float));
        ApBuffer?.Release();         ApBuffer         = new ComputeBuffer(solverCap, sizeof(float));
        pressureBuffer?.Release();   pressureBuffer   = new ComputeBuffer(solverCap, sizeof(float));
        zVectorBuffer?.Release();    zVectorBuffer    = new ComputeBuffer(solverCap, sizeof(float));
        matrixABuffer?.Release();    matrixABuffer    = new ComputeBuffer(maxNodesCapacity * 25, sizeof(float));
        nnzPerNode?.Release();       nnzPerNode       = new ComputeBuffer(solverCap, sizeof(uint));
        csrRowPtr?.Release();        csrRowPtr        = new ComputeBuffer(solverCap + 1, sizeof(uint));
        csrColIndices?.Release();    csrColIndices    = new ComputeBuffer(maxNnz, sizeof(uint));
        csrRowIndices?.Release();    csrRowIndices    = new ComputeBuffer(maxNnz, sizeof(uint));
        csrValues?.Release();        csrValues        = new ComputeBuffer(maxNnz, sizeof(float));

        // Solver indirect dispatch args (4 triples = 12 uints):
        //   [0-2]  256-thread (BuildMatrixA, divergence, CSR, ApplyPressure)
        //   [3-5]  512-thread (vector ops: axpy, scale, dot, copy, jacobi)
        //   [6-8]  1-group    (GlobalReduceSum, ComputeAlpha/Beta, StoreRho)
        //   [9-11] pfx1-groups (CSR exclusive prefix sum passes 1 & 2)
        solverIndirectArgsBuffer?.Release();
        solverIndirectArgsBuffer = new ComputeBuffer(12, sizeof(uint), ComputeBufferType.IndirectArguments);
        solverReductionCount256Buffer?.Release();
        solverReductionCount256Buffer = new ComputeBuffer(1, sizeof(uint));
        solverReductionCount512Buffer?.Release();
        solverReductionCount512Buffer = new ComputeBuffer(1, sizeof(uint));
        solverPrefixLenBuffer?.Release();
        solverPrefixLenBuffer = new ComputeBuffer(2, sizeof(uint));
    }
    
    // Public method for ScenarioManager to reset the simulation
    public void ResetSimulation()
    {
        InitializeParticleSystem();
    }
    
    private void SortParticles()
    {
        if (radixSort == null || radixSortShader == null)
            return;
        radixSort.Sort(particlesBuffer, particlesBuffer, (uint)numParticles, emitGpuDebugLabels ? fluidSimGpuProfileCmd : null);
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
        reverseNeighborsBuffer?.Release();
        diffusionGradientBuffer?.Release();
        dispatchArgsBuffer?.Release();
        particlePrefixElementCountBuffer?.Release();
        divergenceBuffer?.Release();
        residualBuffer?.Release();
        cgAlphaBuffer?.Release();
        cgBetaBuffer?.Release();
        cgRhoBuffer?.Release();
        solverIndirectArgsBuffer?.Release();
        solverReductionCount256Buffer?.Release();
        solverReductionCount512Buffer?.Release();
        solverPrefixLenBuffer?.Release();
        pBuffer?.Release();
        ApBuffer?.Release();
        pressureBuffer?.Release();
        matrixABuffer?.Release();
        nnzPerNode?.Release();
        csrRowPtr?.Release();
        csrColIndices?.Release();
        csrRowIndices?.Release();
        csrValues?.Release();
        phiBuffer?.Release();
        phiBuffer_Read?.Release();
        dirtyFlagBuffer?.Release();
        mortonCodesBuffer?.Release();
        solidVoxelsBuffer?.Release();
        ReleasePreconditionerBuffers();
        ReleaseLeafOnlyBuffers();
        ReleaseLeafOnlyWeightsBuffers();
        ReleaseFluidSimGpuProfileCmd();
    }
}
