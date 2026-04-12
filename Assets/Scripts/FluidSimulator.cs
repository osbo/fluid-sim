using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Serialization;
using UnityEngine.InputSystem;
using System;
using System.Collections;
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
    [Tooltip("Optional mesh whose interior defines the fluid spawn volume. If unset, falls back to the Fluid Initial Bounds box.")]
    public MeshFilter fluidInitialMesh;
    [Tooltip("Initial velocity applied to all spawned particles.")]
    public Vector3 initialVelocity = Vector3.zero;

    ComputeShader radixSortShader;
    ComputeShader particlesShader;
    ComputeShader nodesPrefixSumsShader;
    ComputeShader nodesShader;
    ComputeShader cgSolverShader;
    ComputeShader csrBuilderShader;
    ComputeShader uniformGridShader;
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
    private int updateParticlesUniformKernel = -1;
    private int applyExternalForcesKernel;
    private int applyViscosityKernelId;
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
    private int buildMatrixAUniformKernelId;
    private int csrCountNnzUniformKernelId;
    private int csrFillUniformKernelId;
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
    private int uniformGridClearDenseKernel;
    private int uniformGridBinParticlesKernel;

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
    /// <summary>GridMode.Uniform only: 6 face neighbors per node, SoA stride <see cref="maxNodesCapacity"/>.</summary>
    private ComputeBuffer uniformNeighborsBuffer;
    public ComputeBuffer reverseNeighborsBuffer;
    private ComputeBuffer divergenceBuffer;
    private ComputeBuffer pBuffer;
    private ComputeBuffer ApBuffer;
    private ComputeBuffer residualBuffer;
    private ComputeBuffer pressureBuffer;
    // NEW: Buffer for standard Laplacian Matrix A
    // Format: [25 * numNodes] where Slot 0 = Diagonal, 1-24 = Neighbors
    private ComputeBuffer matrixABuffer;
    /// <summary>GridMode.Uniform only: 7 floats per node (diagonal at i, off-diagonals at (1..6)*stride+i).</summary>
    private ComputeBuffer uniformMatrixABuffer;
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

    // Uniform grid hash (GridMode.Uniform): dense N³ maps + compact active list + per-node accumulation.
    private ComputeBuffer uniformCellCountsBuffer;
    private ComputeBuffer uniformDenseIndexMapBuffer;
    private ComputeBuffer uniformActiveMortonListBuffer;
    private ComputeBuffer uniformActiveNodeCountBuffer;
    private ComputeBuffer uniformNodeAccumBuffer; // 7 * kSplatStripes * maxNodesCapacity uints (see UniformGrid.compute)

    public const int UniformNeighborSlotCount = 6;
    public const int UniformMatrixSlotCount = 7;

    private int maxNodesCapacity;
    private int maxPrefixThreadGroups;
    private int maxAuxBlocks;
    private readonly uint[] gpuNodeCountReadback = new uint[1];
    private uint[] uniformNeighborDebugFacesScratch;
    private uint[] uniformNeighborDebugMortonScratch;
    private uint[] uniformNeighborDebugDenseScratch;
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

    public ComputeBuffer UniformCellCountsBuffer => uniformCellCountsBuffer;
    public ComputeBuffer UniformDenseIndexMapBuffer => uniformDenseIndexMapBuffer;
    public ComputeBuffer UniformActiveMortonListBuffer => uniformActiveMortonListBuffer;
    public ComputeBuffer UniformActiveNodeCountBuffer => uniformActiveNodeCountBuffer;
    /// <summary>Capacity of <see cref="UniformActiveMortonListBuffer"/>; also max compact active cells per frame for uniform binning.</summary>
    public int MaxUniformActiveCells => maxNodesCapacity;
    public Bounds WorldSimulationBounds => simulationBounds != null ? simulationBounds.bounds : new Bounds(Vector3.zero, Vector3.one);
    public float MaxDetailCellSize => maxDetailCellSize;
    private int numUniqueNodes; // rename to numUniqueNodes
    private int layer;
    public float gravity = 100.0f;

    [Header("Viscosity")]
    [Tooltip("Kinematic viscosity ν in sim-units²/s. 0 = inviscid. Use small values for water-like fluids; larger for honey/glycerol.\nExplicit stability limit: ν·dt/dx_min² < 1/6.")]
    public float kinematicViscosity = 0.0f;
    [Tooltip("FLIP/PIC blend. 1.0 = pure FLIP (recommended when kinematicViscosity > 0). 0.95 = slight PIC damping (legacy).")]
    [Range(0f, 1f)] public float flipRatio = 1.0f;

    public bool useRealTime = false; // When true, uses Time.deltaTime instead of fixed frameRate
    public float frameRate = 30.0f;
    [Tooltip("Finest octree layer for the sim; collider voxel grid uses R = 2^(10 - minLayer) along each axis (same as Nodes.compute kMortonAxisBits).")]
    [Range(0, 10)] public int minLayer = 4;
    [Range(0, 10)] public int maxLayer = 10;
    [Tooltip("Octree-equivalent layer L (same as Nodes minLayer / node.layer for cell size): sim cell width = 2^L. Uniform grid uses k = (MortonAxisBits − L) bits per axis, N = 2^k cells per axis, so each bin matches a level-L octree cell.")]
    [Range(0, 10)]
    [FormerlySerializedAs("uniformGridResolutionLog2")]
    public int uniformGridCellLayer = 3;

    /// <summary>Must match kMortonAxisBits in Nodes.compute.</summary>
    public const int MortonAxisBits = 10;

    /// <summary>Collider voxel resolution per axis: 2^(MortonAxisBits - minLayer).</summary>
    public int SolidVoxelResolution => 1 << (MortonAxisBits - Mathf.Clamp(minLayer, 0, MortonAxisBits));

    /// <summary>k = MortonAxisBits − L; N = 2^k cells per axis (Morton fold matches octree cells of width 2^L).</summary>
    public int UniformGridBinsPerAxisLog2 => MortonAxisBits - Mathf.Clamp(uniformGridCellLayer, 0, MortonAxisBits);

    /// <summary>Uniform hash grid: N = 2^k cells per axis.</summary>
    public int UniformGridCellsPerAxis { get; private set; }

    /// <summary>Uniform hash grid: N³ total dense cell slots.</summary>
    public int UniformGridCellCount { get; private set; }
    [Tooltip("When per-frame debug timing log is enabled: each frame, GPU readback of nodesBuffer and layer histogram text for the summary. No effect when that master toggle is off.")]
    public bool logLayerCountsPerFrame = true;

    [Header("Debug: frame stats / logging")]
    [Tooltip("Master toggle for expensive per-frame diagnostics. When off (default): no frame summary Debug.Log, no cumulative timing averages, no layer-count readback, and no AsyncGPUReadback fences around timed sections. When on: each section’s timer includes the GPU fence before it; summary logs after WaitForEndOfFrame with full wall-frame ms (aligns with Game view). Very slow.")]
    public bool enablePerFrameDebugTimingLog = false;

    [Header("Debug: uniform grid neighbors")]
    [Tooltip("GridMode.Uniform only: after FindNeighbors, read back stencil and log reciprocity, domain-wall, and denseIndexMap checks. Traces +/− per axis (faces 0/1=x, 2/3=y, 4/5=z). Very expensive (large GPU readback each frame).")]
    public bool debugValidateUniformNeighbors = false;
    [Tooltip("GPU readback: first N node slots per face (must be ≥ active nodes for full reciprocity). Increase if recipSkipped > 0.")]
    [Min(1)]
    public int debugUniformNeighborsMaxNodes = 2097152;
    [Tooltip("Skip full denseIndexMap readback when uniform N³ exceeds this (still runs reciprocity + wall checks).")]
    [Min(1024)]
    public int debugUniformNeighborsMaxDenseCells = 2097152;
    [Tooltip("Max example lines printed per category (reciprocity / wall / dense).")]
    [Range(0, 64)]
    public int debugUniformNeighborsMaxExamples = 16;

    public PreconditionerType preconditioner = PreconditionerType.Jacobi;

    [Header("Grid")]
    [Tooltip("Octree: adaptive refinement. Uniform: fixed cell size 2^uniformGridCellLayer, same pressure/FLIP pipeline with a 6-neighbor stencil.")]
    public GridMode gridMode = GridMode.Octree;

    private int frameNumber = 0;
    private double cumulativeFrameTimeMs = 0.0;
    private double cumulativeUnityUnscaledDeltaMsForTimingLog = 0.0;
    private double cumulativeFullFrameWallMs = 0.0;
    private int fullFrameWallSamples = 0;
    private double lastTimingEndOfFrameRealtime = -1.0;
    private bool hasPendingFrameTimingLog;
    private FrameTimingLogPending frameTimingLogPending;

    private struct FrameTimingLogPending
    {
        public int frameNumber;
        public double simUpdateMs;
        public double unityUnscaledDeltaMs;
        public double averageSimUpdateMs;
        public double averageUnityUnscaledMs;
        public double sortMs, findUniqueMs, createLeavesMs, layerLoopMs, findNeighborsMs;
        public double uniformGridBuildMs;
        public double calculateGradientsMs, computeLevelSetMs, storeOldVelocitiesMs, applyExternalForcesMs;
        public double solvePressureMs, updateParticlesMs, miscPipelineMs;
        public bool useOctree;
        public int numNodes;
        public string layerCountsLine;
        public string averageCgIterationsText;
        public int lastCgIterations;
    }
    private double cumulativeCgIterations = 0.0;
    private int cgSolveFrameCount = 0;
    private float averageCgIterations = 0.0f;
    private int lastCgIterations = 0;

    /// <summary>Running averages for <c>[PCG]</c> line: sample count and sums (only while <see cref="enablePerFrameDebugTimingLog"/> is on).</summary>
    private int pcgResidualLogSampleCount = 0;
    private double cumulativePcgResidualSqLogSum = 0.0;
    private double cumulativePcgRelResidualLogSum = 0.0;
    private double cumulativePcgItersLogSum = 0.0;
    
    // Pause/resume functionality
    private bool isPaused = false;

    
    private double lastRenderTimeMs;

    // Profiler markers for Unity Profiler GPU timeline
    static readonly ProfilerMarker s_FrameMarker            = new ProfilerMarker("FluidSim.Frame");
    static readonly ProfilerMarker s_SortMarker             = new ProfilerMarker("FluidSim.Sort");
    static readonly ProfilerMarker s_FindUniqueMarker       = new ProfilerMarker("FluidSim.FindUnique");
    static readonly ProfilerMarker s_CreateLeavesMarker     = new ProfilerMarker("FluidSim.CreateLeaves");
    static readonly ProfilerMarker s_LayerLoopMarker        = new ProfilerMarker("FluidSim.LayerLoop");
    static readonly ProfilerMarker s_FindNeighborsMarker    = new ProfilerMarker("FluidSim.FindNeighbors");
    static readonly ProfilerMarker s_UniformGridBuildMarker = new ProfilerMarker("FluidSim.UniformGridBuild");
    static readonly ProfilerMarker s_GradientsMarker        = new ProfilerMarker("FluidSim.CalculateGradients");
    static readonly ProfilerMarker s_StoreVelocitiesMarker  = new ProfilerMarker("FluidSim.StoreOldVelocities");
    static readonly ProfilerMarker s_ExternalForcesMarker   = new ProfilerMarker("FluidSim.ApplyExternalForces");
    static readonly ProfilerMarker s_NodeCountReadbackMarker= new ProfilerMarker("FluidSim.NodeCountReadback");
    static readonly ProfilerMarker s_SolvePressureMarker    = new ProfilerMarker("FluidSim.SolvePressure");
    static readonly ProfilerMarker s_UpdateParticlesMarker  = new ProfilerMarker("FluidSim.UpdateParticles");
    static readonly ProfilerMarker s_MiscPipelineMarker     = new ProfilerMarker("FluidSim.MiscPipeline");
    static readonly ProfilerMarker s_GpuTimingSyncMarker    = new ProfilerMarker("FluidSim.GpuTimingSync");

    static readonly string s_FrameTimingModeNote =
        "• Timing: GPU sync ON (each section includes pre-step AsyncGPUReadback fence; not full device idle).\n" +
        "• Uniform build and octree Find Neighbors both fence nodesBuffer after their GPU tail (default flush only hits particles + nodeCount).\n" +
        "• Full frame wall = realtime between successive WaitForEndOfFrame (~what Game view reflects).\n" +
        "• FluidSimulator.Update excludes other scripts, editor, vsync wait; Rendering line is this frame’s FluidRenderer camera work (before EOF).\n";
    
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
    /// <summary>Avoids <c>SetData(new uint[] {{ ... }})</c> allocations on hot paths (uniform prefix, node count, recording).</summary>
    private readonly uint[] gpuUintScratch1 = new uint[1];
    /// <summary>Pooled when <see cref="debugValidateUniformNeighbors"/> is enabled (5×6 face counters per frame).</summary>
    private int[] uniformNeighborsDebugFaceStats30;
    private StringBuilder uniformNeighborsDebugSbRecip;
    private StringBuilder uniformNeighborsDebugSbWall;
    private StringBuilder uniformNeighborsDebugSbDense;
    private StringBuilder uniformNeighborsDebugSbSummary;
    private Particle[] particlesCPU;
    private string str;

    // Solid colliders (static mesh obstacles)
    [Header("Static Colliders")]
    [Tooltip("When off: no mesh bake, no R³ voxel/SDF GPU memory (only 1-element stubs), and compute skips solid/SDF logic.")]
    public bool useColliders = true;
    [Tooltip("Parent GameObject whose MeshFilter children define static solid obstacles.")]
    public GameObject collidersRoot;

    /// <summary>Resolution passed to solid/SDF shaders: full R when <see cref="useColliders"/> is on, else 1 (stub buffer).</summary>
    internal int ColliderGridResolution => useColliders ? SolidVoxelResolution : 1;

    private ComputeBuffer solidSDFBuffer;
    public  ComputeBuffer SolidSDFBuffer      => solidSDFBuffer;
    private ComputeBuffer initialPositionsBuffer;
    /// <summary>Per-voxel occupancy (0/1) for debug draw; same R³ layout as <see cref="SolidSDFBuffer"/>.</summary>
    private ComputeBuffer solidVoxelsBuffer;
    public  ComputeBuffer SolidVoxelsBuffer     => solidVoxelsBuffer;

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
        uniformGridCellLayer = Mathf.Clamp(uniformGridCellLayer, 0, MortonAxisBits);
        if (Application.isPlaying && particlesBuffer != null)
            BakeAndUploadSolidVoxels();
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
        InitUniformGridKernels();
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

        StartCoroutine(FrameTimingLogEndOfFrameRoutine());
    }

    private void BakeAndUploadSolidVoxels()
    {
        const int stubElements = 1;

        if (!useColliders)
        {
            solidSDFBuffer?.Release();
            solidSDFBuffer = new ComputeBuffer(stubElements, sizeof(float));
            solidSDFBuffer.SetData(new float[] { 1e9f });
            solidVoxelsBuffer?.Release();
            solidVoxelsBuffer = new ComputeBuffer(stubElements, sizeof(uint));
            solidVoxelsBuffer.SetData(new uint[] { 0u });
            return;
        }

        int R = SolidVoxelResolution;
        int total = R * R * R;

        MeshColliderBaker.BakeResult bake = collidersRoot != null
            ? MeshColliderBaker.Bake(collidersRoot, simulationBounds.bounds, R)
            : new MeshColliderBaker.BakeResult
            {
                Solid = new uint[total],
                Normals = new Vector3[total],
                SDF = new float[total]
            };

        solidSDFBuffer?.Release();
        solidSDFBuffer = new ComputeBuffer(total, sizeof(float));
        solidSDFBuffer.SetData(bake.SDF);

        solidVoxelsBuffer?.Release();
        solidVoxelsBuffer = new ComputeBuffer(total, sizeof(uint));
        solidVoxelsBuffer.SetData(bake.Solid);

        // Teleport any particles that initialized inside solid voxels
        if (particlesShader != null)
        {
            int kernel = particlesShader.FindKernel("RemoveSolidParticles");
            particlesShader.SetBuffer(kernel, "particlesBuffer", particlesBuffer);
            particlesShader.SetBuffer(kernel, "solidSDFBuffer", solidSDFBuffer);
            particlesShader.SetInt("useColliders", 1);
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

    /// <summary>When <see cref="enablePerFrameDebugTimingLog"/> is on: tiny AsyncGPUReadback + wait so section Stopwatches approximate GPU time for that step.</summary>
    private void GpuTimingReportFlushGpu(bool includePressureBuffer = false)
    {
        if (!enablePerFrameDebugTimingLog)
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

    /// <summary>
    /// Uniform-grid build finishes with normalize + FindNeighbors + face-velocity kernels that write/read <see cref="nodesBuffer"/>.
    /// <see cref="GpuTimingReportFlushGpu"/> only fences <c>particlesBuffer</c> and <c>nodeCount</c>, so the next timed section’s opening flush
    /// would wait for that tail GPU work and incorrectly charge it to e.g. Calculate Gradients. One byte readback on <c>nodesBuffer</c> ties the
    /// full build to the Uniform Grid Build stopwatch when debug timing is on.
    /// </summary>
    private void GpuTimingFenceAfterUniformGridBuildGpuComplete()
    {
        if (!enablePerFrameDebugTimingLog || nodesBuffer == null)
            return;
        using (s_GpuTimingSyncMarker.Auto())
        {
            var r = AsyncGPUReadback.Request(nodesBuffer, sizeof(float), 0);
            r.WaitForCompletion();
        }
    }

    /// <summary>Octree <c>findNeighbors</c> ends with copy-face-vel → <see cref="nodesBuffer"/>; fence that so the next section does not absorb it.</summary>
    private void GpuTimingFenceAfterFindNeighborsOctree()
    {
        if (!enablePerFrameDebugTimingLog || nodesBuffer == null)
            return;
        using (s_GpuTimingSyncMarker.Auto())
        {
            var r = AsyncGPUReadback.Request(nodesBuffer, sizeof(float), 0);
            r.WaitForCompletion();
        }
    }

    void Update()
    {
        if (Keyboard.current != null && Keyboard.current.rKey.wasPressedThisFrame)
        {
            ResetSimulation();
            Debug.Log("Simulation reset to initial state (R).");
        }

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

        var frameSw = System.Diagnostics.Stopwatch.StartNew();
        using var _ = s_FrameMarker.Auto();

        // Calculate maxDetailCellSize for volume calculations
        // Use the normalized simulation bounds (simulationBoundsMin is now Vector3.zero)
        Vector3 simulationSize = simulationBoundsMax;
        maxDetailCellSize = Mathf.Min(simulationSize.x, simulationSize.y, simulationSize.z) / 1024.0f;

        bool useOctreePipeline = gridMode == GridMode.Octree;

        var sortSw = new System.Diagnostics.Stopwatch();
        var findUniqueSw = new System.Diagnostics.Stopwatch();
        var createLeavesSw = new System.Diagnostics.Stopwatch();
        var layerLoopSw = new System.Diagnostics.Stopwatch();
        var findNeighborsSw = new System.Diagnostics.Stopwatch();
        var calculateGradientsSw = new System.Diagnostics.Stopwatch();
        var computeLevelSetSw = new System.Diagnostics.Stopwatch();
        var storeOldVelocitiesSw = new System.Diagnostics.Stopwatch();
        var applyExternalForcesSw = new System.Diagnostics.Stopwatch();
        var solvePressureSw = new System.Diagnostics.Stopwatch();
        var updateParticlesSw = new System.Diagnostics.Stopwatch();
        var uniformGridBuildSw = new System.Diagnostics.Stopwatch();
        var miscPipelineSw = new System.Diagnostics.Stopwatch();

        string layerCountsLine = null;

        if (useOctreePipeline)
        {
            layer = minLayer;

            // Each section: GPU fence (when timing log on) + work. Fences are attributed to the section they precede.
            sortSw.Restart();
            GpuTimingReportFlushGpu(false);
            using (s_SortMarker.Auto())
                SortParticles();
            sortSw.Stop();

            findUniqueSw.Restart();
            GpuTimingReportFlushGpu(false);
            using (s_FindUniqueMarker.Auto())
                findUniqueParticles();
            findUniqueSw.Stop();

            createLeavesSw.Restart();
            GpuTimingReportFlushGpu(false);
            using (s_CreateLeavesMarker.Auto())
                CreateLeaves();
            createLeavesSw.Stop();

            layerLoopSw.Restart();
            GpuTimingReportFlushGpu(false);
            using (s_LayerLoopMarker.Auto())
            {
                for (layer = layer + 1; layer <= maxLayer; layer++)
                {
                    findUniqueNodes();
                    ProcessNodes();
                    compactNodes();
                }
            }
            layerLoopSw.Stop();

            findNeighborsSw.Restart();
            GpuTimingReportFlushGpu(false);
            using (s_FindNeighborsMarker.Auto())
                findNeighbors();
            GpuTimingFenceAfterFindNeighborsOctree();
            findNeighborsSw.Stop();

            calculateGradientsSw.Restart();
            GpuTimingReportFlushGpu(false);
            using (s_GradientsMarker.Auto())
                CalculateDensityGradients();
            calculateGradientsSw.Stop();

            computeLevelSetSw.Restart();
            GpuTimingReportFlushGpu(false);
            // ComputeLevelSet();
            computeLevelSetSw.Stop();

            storeOldVelocitiesSw.Restart();
            GpuTimingReportFlushGpu(false);
            using (s_StoreVelocitiesMarker.Auto())
                StoreOldVelocities();
            storeOldVelocitiesSw.Stop();

            applyExternalForcesSw.Restart();
            GpuTimingReportFlushGpu(false);
            using (s_ExternalForcesMarker.Auto())
                ApplyExternalForces();
            applyExternalForcesSw.Stop();

            if (enablePerFrameDebugTimingLog)
                miscPipelineSw.Restart();
            using (s_MiscPipelineMarker.Auto())
            {
                GpuTimingReportFlushGpu(false);
                if (kinematicViscosity > 0f)
                    ApplyViscosity();
                using (s_NodeCountReadbackMarker.Auto())
                {
                    nodeCount.GetData(gpuNodeCountReadback);
                    numNodes = (int)gpuNodeCountReadback[0];
                }
                if (enablePerFrameDebugTimingLog && logLayerCountsPerFrame)
                    layerCountsLine = BuildLayerCountsSummaryForLog();
            }
            if (enablePerFrameDebugTimingLog)
                miscPipelineSw.Stop();

            solvePressureSw.Restart();
            GpuTimingReportFlushGpu(false);
            using (s_SolvePressureMarker.Auto())
                SolvePressure();
            GpuTimingReportFlushGpu(true);
            solvePressureSw.Stop();

            updateParticlesSw.Restart();
            GpuTimingReportFlushGpu(false);
            using (s_UpdateParticlesMarker.Auto())
                UpdateParticles();
            GpuTimingReportFlushGpu(false);
            updateParticlesSw.Stop();
        }
        else
        {
            uniformGridBuildSw.Restart();
            GpuTimingReportFlushGpu(false);
            using (s_UniformGridBuildMarker.Auto())
                DispatchUniformGridClearAndBin();
            GpuTimingFenceAfterUniformGridBuildGpuComplete();
            uniformGridBuildSw.Stop();

            if (numNodes > 0)
            {
                calculateGradientsSw.Restart();
                GpuTimingReportFlushGpu(false);
                using (s_GradientsMarker.Auto())
                    CalculateDensityGradients();
                calculateGradientsSw.Stop();

                computeLevelSetSw.Restart();
                GpuTimingReportFlushGpu(false);
                computeLevelSetSw.Stop();

                storeOldVelocitiesSw.Restart();
                GpuTimingReportFlushGpu(false);
                using (s_StoreVelocitiesMarker.Auto())
                    StoreOldVelocities();
                storeOldVelocitiesSw.Stop();

                applyExternalForcesSw.Restart();
                GpuTimingReportFlushGpu(false);
                using (s_ExternalForcesMarker.Auto())
                    ApplyExternalForces();
                applyExternalForcesSw.Stop();

                if (enablePerFrameDebugTimingLog)
                    miscPipelineSw.Restart();
                using (s_MiscPipelineMarker.Auto())
                {
                    GpuTimingReportFlushGpu(false);
                    if (kinematicViscosity > 0f)
                        ApplyViscosity();
                    // numNodes already set in DispatchUniformGridClearAndBin (no redundant nodeCount GetData).
                    if (enablePerFrameDebugTimingLog && logLayerCountsPerFrame)
                        layerCountsLine = BuildLayerCountsSummaryForLog();
                }
                if (enablePerFrameDebugTimingLog)
                    miscPipelineSw.Stop();

                solvePressureSw.Restart();
                GpuTimingReportFlushGpu(false);
                using (s_SolvePressureMarker.Auto())
                    SolvePressure();
                GpuTimingReportFlushGpu(true);
                solvePressureSw.Stop();
            }

            updateParticlesSw.Restart();
            GpuTimingReportFlushGpu(false);
            using (s_UpdateParticlesMarker.Auto())
                UpdateParticles();
            GpuTimingReportFlushGpu(false);
            updateParticlesSw.Stop();
        }

        frameSw.Stop();
        if (enablePerFrameDebugTimingLog)
        {
            frameNumber++;
            double simMs = frameSw.Elapsed.TotalMilliseconds;
            cumulativeFrameTimeMs += simMs;
            double unityDeltaMs = Time.unscaledDeltaTime * 1000.0;
            cumulativeUnityUnscaledDeltaMsForTimingLog += unityDeltaMs;

            string averageCgIterationsText = cgSolveFrameCount > 0 ? averageCgIterations.ToString("F2") : "N/A";
            hasPendingFrameTimingLog = true;
            frameTimingLogPending = new FrameTimingLogPending
            {
                frameNumber = frameNumber,
                simUpdateMs = simMs,
                unityUnscaledDeltaMs = unityDeltaMs,
                averageSimUpdateMs = cumulativeFrameTimeMs / frameNumber,
                averageUnityUnscaledMs = cumulativeUnityUnscaledDeltaMsForTimingLog / frameNumber,
                sortMs = sortSw.Elapsed.TotalMilliseconds,
                findUniqueMs = findUniqueSw.Elapsed.TotalMilliseconds,
                createLeavesMs = createLeavesSw.Elapsed.TotalMilliseconds,
                layerLoopMs = layerLoopSw.Elapsed.TotalMilliseconds,
                findNeighborsMs = findNeighborsSw.Elapsed.TotalMilliseconds,
                uniformGridBuildMs = uniformGridBuildSw.Elapsed.TotalMilliseconds,
                calculateGradientsMs = calculateGradientsSw.Elapsed.TotalMilliseconds,
                computeLevelSetMs = computeLevelSetSw.Elapsed.TotalMilliseconds,
                storeOldVelocitiesMs = storeOldVelocitiesSw.Elapsed.TotalMilliseconds,
                applyExternalForcesMs = applyExternalForcesSw.Elapsed.TotalMilliseconds,
                solvePressureMs = solvePressureSw.Elapsed.TotalMilliseconds,
                updateParticlesMs = updateParticlesSw.Elapsed.TotalMilliseconds,
                miscPipelineMs = miscPipelineSw.Elapsed.TotalMilliseconds,
                useOctree = useOctreePipeline,
                numNodes = numNodes,
                layerCountsLine = layerCountsLine,
                averageCgIterationsText = averageCgIterationsText,
                lastCgIterations = lastCgIterations
            };
        }
    }

    IEnumerator FrameTimingLogEndOfFrameRoutine()
    {
        var waitEnd = new WaitForEndOfFrame();
        while (enabled)
        {
            yield return waitEnd;
            if (!enablePerFrameDebugTimingLog || !hasPendingFrameTimingLog)
                continue;

            hasPendingFrameTimingLog = false;
            FrameTimingLogPending p = frameTimingLogPending;

            double now = Time.realtimeSinceStartupAsDouble;
            double fullWallMs = double.NaN;
            if (lastTimingEndOfFrameRealtime >= 0.0)
            {
                fullWallMs = (now - lastTimingEndOfFrameRealtime) * 1000.0;
                cumulativeFullFrameWallMs += fullWallMs;
                fullFrameWallSamples++;
            }
            lastTimingEndOfFrameRealtime = now;

            double avgFullWall = fullFrameWallSamples > 0 ? cumulativeFullFrameWallMs / fullFrameWallSamples : 0.0;

            double sumListed = p.miscPipelineMs + p.updateParticlesMs + p.solvePressureMs + p.applyExternalForcesMs
                + p.storeOldVelocitiesMs + p.computeLevelSetMs + p.calculateGradientsMs;
            if (p.useOctree)
                sumListed += p.sortMs + p.findUniqueMs + p.createLeavesMs + p.layerLoopMs + p.findNeighborsMs;
            else
                sumListed += p.uniformGridBuildMs;

            double drift = p.simUpdateMs - sumListed;

            string fullWallBlock = double.IsNaN(fullWallMs)
                ? "• Full frame wall (EOF→EOF): n/a (first sample)\n"
                : $"• Full frame wall (realtime EOF→EOF, ~Game view): {fullWallMs:F2} ms\n" +
                  $"• Avg full frame wall: {avgFullWall:F2} ms\n";

            string pipelineSection = p.useOctree
                ? $"• Sort: {p.sortMs:F2} ms\n" +
                  $"• Find Unique: {p.findUniqueMs:F2} ms\n" +
                  $"• Create Leaves: {p.createLeavesMs:F2} ms\n" +
                  $"• Layer Loop: {p.layerLoopMs:F2} ms\n" +
                  $"• Find Neighbors: {p.findNeighborsMs:F2} ms\n"
                : $"• Uniform Grid Build: {p.uniformGridBuildMs:F2} ms\n" +
                  "  (N³ clear, bin, splat+normalize, neighbors, face velocities)\n";

            Debug.Log($"Frame {p.frameNumber} Summary:\n" +
                fullWallBlock +
                $"• Unity unscaledDeltaTime (engine last-frame interval): {p.unityUnscaledDeltaMs:F2} ms\n" +
                $"• Avg unscaledDeltaTime: {p.averageUnityUnscaledMs:F2} ms\n" +
                $"• FluidSimulator.Update (wall): {p.simUpdateMs:F2} ms\n" +
                $"• Avg FluidSimulator.Update: {p.averageSimUpdateMs:F2} ms\n" +
                $"• Sum of sections below: {sumListed:F2} ms (drift vs Update: {drift:+0.00;-0.00;0.00} ms)\n" +
                s_FrameTimingModeNote +
                $"• Grid: {(p.useOctree ? "Octree" : "Uniform")}\n" +
                $"• # Nodes: {p.numNodes}\n" +
                (p.layerCountsLine != null ? $"• {p.layerCountsLine}\n" : "") +
                pipelineSection +
                $"• Misc (viscosity, node readback, layer histogram): {p.miscPipelineMs:F2} ms\n" +
                $"• Calculate Gradients: {p.calculateGradientsMs:F2} ms\n" +
                $"• Compute Level Set: {p.computeLevelSetMs:F2} ms\n" +
                $"• Store Old Velocities: {p.storeOldVelocitiesMs:F2} ms\n" +
                $"• Apply External Forces: {p.applyExternalForcesMs:F2} ms\n" +
                $"• Solve Pressure: {p.solvePressureMs:F2} ms\n" +
                $"• Update Particles: {p.updateParticlesMs:F2} ms\n" +
                $"• Rendering (FluidRenderer this frame, before EOF): {lastRenderTimeMs:F2} ms\n" +
                $"• CG Iterations: {p.lastCgIterations}\n" +
                $"• Avg CG Iterations: {p.averageCgIterationsText}");
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
        if (gridMode == GridMode.Uniform)
        {
            if (uniformGridShader == null || uniformGridCalculateDensityGradientKernel < 0)
            {
                Debug.LogError("FluidSimulator: UniformGrid.compute kernel UniformGrid_CalculateDensityGradient is missing.");
                return;
            }
            if (numNodes <= 0)
                return;

            uniformGridShader.SetInt("numNodesCapacity", maxNodesCapacity);
            uniformGridShader.SetBuffer(uniformGridCalculateDensityGradientKernel, "nodeCountBuffer", nodeCount);
            uniformGridShader.SetBuffer(uniformGridCalculateDensityGradientKernel, "nodesBuffer", nodesBuffer);
            uniformGridShader.SetBuffer(uniformGridCalculateDensityGradientKernel, "uniformNeighborsBuffer", uniformNeighborsBuffer);
            uniformGridShader.SetBuffer(uniformGridCalculateDensityGradientKernel, "diffusionGradientBuffer", diffusionGradientBuffer);
            WriteIndirectArgsFromCountBuffer(nodeCount);
            uniformGridShader.DispatchIndirect(uniformGridCalculateDensityGradientKernel, dispatchArgsBuffer, 0);
            return;
        }

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
        nodesShader.DispatchIndirect(calculateDensityGradientKernel, dispatchArgsBuffer, 0);
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
        nodesPrefixSumsShader.Dispatch(copyNodesKernelId, copyGroups, 1, 1);
    }

    private void UpdateParticles()
    {
        if (particlesShader == null)
        {
            Debug.LogError("Particles compute shader is missing.");
            return;
        }

        int kernel = gridMode == GridMode.Uniform ? updateParticlesUniformKernel : updateParticlesKernel;
        if (kernel < 0)
        {
            Debug.LogError(
                gridMode == GridMode.Uniform
                    ? "Particles.compute kernel UpdateParticlesUniform not found (reimport shader)."
                    : "Particles.compute kernel UpdateParticles not found (reimport shader).");
            return;
        }

        particlesShader.SetBuffer(kernel, "nodesBuffer", nodesBuffer);
        if (gridMode == GridMode.Octree)
            particlesShader.SetBuffer(kernel, "mortonCodesBuffer", mortonCodesBuffer);
        particlesShader.SetBuffer(kernel, "nodesBufferOld", nodesBufferOld);
        particlesShader.SetBuffer(kernel, "particlesBuffer", particlesBuffer);
        particlesShader.SetBuffer(kernel, "diffusionGradientBuffer", diffusionGradientBuffer);
        particlesShader.SetBuffer(kernel, "nodeCountBuffer", nodeCount);
        particlesShader.SetBuffer(kernel, "solidSDFBuffer", solidSDFBuffer);
        particlesShader.SetInt("useColliders", useColliders ? 1 : 0);
        particlesShader.SetInt("solidVoxelResolution", ColliderGridResolution);
        particlesShader.SetInt("numParticles", numParticles);
        float deltaTime = useRealTime ? Time.deltaTime : (1 / frameRate);
        particlesShader.SetFloat("deltaTime", deltaTime);
        particlesShader.SetFloat("gravity", gravity);
        particlesShader.SetFloat("flip_ratio", flipRatio);
        particlesShader.SetFloat("maxDetailCellSize", maxDetailCellSize);
        particlesShader.SetVector("mortonNormalizationFactor", mortonNormalizationFactor);
        particlesShader.SetFloat("mortonMaxValue", mortonMaxValue);
        particlesShader.SetVector("simulationBoundsMin", simulationBoundsMin);
        particlesShader.SetVector("simulationBoundsMax", simulationBoundsMax);
        particlesShader.SetInt("minLayer", minLayer);
        particlesShader.SetInt("maxLayer", maxLayer);
        if (gridMode == GridMode.Uniform)
        {
            particlesShader.SetBuffer(kernel, "uniformDenseIndexMap", uniformDenseIndexMapBuffer);
            particlesShader.SetInt("uniformGridLog2K", UniformGridBinsPerAxisLog2);
            particlesShader.SetInt("uniformGridCellCount", UniformGridCellCount);
            particlesShader.SetInt("uniformGridCellLayer", Mathf.Clamp(uniformGridCellLayer, 0, MortonAxisBits));
        }
        int threadGroups = Mathf.CeilToInt(numParticles / 512.0f);
        particlesShader.Dispatch(kernel, threadGroups, 1, 1);
    }

    private void InitializeParticleSystem()
    {
        bool hasMesh = fluidInitialMesh != null;
        if (simulationBounds == null || (!hasMesh && fluidInitialBounds == null))
        {
            Debug.LogError(
                "FluidSimulator: assign Simulation Bounds, and either a Fluid Initial Mesh or a Fluid Initial Bounds box collider.");
            return;
        }

        if (particlesShader == null)
        {
            Debug.LogError("Particles compute shader is missing.");
            return;
        }
        initializeParticlesKernel = particlesShader.FindKernel("InitializeParticles");

        // Create particle buffer
        particlesBuffer?.Release();
        particlesBuffer = new ComputeBuffer(numParticles, sizeof(float) * 3 + sizeof(float) * 3 + sizeof(uint));
        particlesCPU = new Particle[numParticles];

        // Compute sim-space transform: world → 0..1023 on each axis
        simulationBoundsMin = simulationBounds.bounds.min;
        simulationBoundsMax = simulationBounds.bounds.max;
        simulationBoundsMax -= simulationBoundsMin;  // size
        simulationBoundsMin  = Vector3.zero;

        Vector3 simulationSize = simulationBoundsMax;
        const float minSimExtent = 1e-4f;
        simulationSize.x = Mathf.Max(simulationSize.x, minSimExtent);
        simulationSize.y = Mathf.Max(simulationSize.y, minSimExtent);
        simulationSize.z = Mathf.Max(simulationSize.z, minSimExtent);
        mortonNormalizationFactor = new Vector3(
            1023.0f / simulationSize.x,
            1023.0f / simulationSize.y,
            1023.0f / simulationSize.z);
        mortonMaxValue = 1023.0f;

        // Also keep fluidInitialBoundsMin for RemoveSolidParticles fallback
        fluidInitialBoundsMin = (fluidInitialBounds != null ? fluidInitialBounds.bounds.min : simulationBounds.bounds.min)
                                - simulationBounds.bounds.min;
        fluidInitialBoundsMax = (fluidInitialBounds != null ? fluidInitialBounds.bounds.max : simulationBounds.bounds.max)
                                - simulationBounds.bounds.min;

        // --- Generate initial positions in sim space (0–1023) ---
        Vector3[] worldPositions;
        if (hasMesh)
        {
            worldPositions = MeshColliderBaker.GenerateVolumePoints(
                fluidInitialMesh, simulationBounds.bounds, numParticles);
        }
        else
        {
            // Box fallback: uniform grid over fluidInitialBounds
            Bounds box = fluidInitialBounds.bounds;
            float cubeRoot = Mathf.Pow(numParticles, 1f / 3f);
            Vector3 size = box.size;
            float maxSide = Mathf.Max(size.x, Mathf.Max(size.y, size.z));
            Vector3 norm = size / maxSide;
            int nx = Mathf.Max(1, Mathf.RoundToInt(cubeRoot * norm.x));
            int ny = Mathf.Max(1, Mathf.RoundToInt(cubeRoot * norm.y));
            int nz = Mathf.Max(1, Mathf.RoundToInt(cubeRoot * norm.z));
            // Adjust so nx*ny*nz >= numParticles
            while (nx * ny * nz < numParticles)
            {
                if (nx <= ny && nx <= nz) nx++;
                else if (ny <= nx && ny <= nz) ny++;
                else nz++;
            }
            Vector3 step = new Vector3(size.x / nx, size.y / ny, size.z / nz);
            worldPositions = new Vector3[numParticles];
            for (int i = 0; i < numParticles; i++)
            {
                int iz = i / (nx * ny), rem = i % (nx * ny), iy = rem / nx, ix = rem % nx;
                worldPositions[i] = box.min + new Vector3((ix + 0.5f) * step.x, (iy + 0.5f) * step.y, (iz + 0.5f) * step.z);
            }
        }

        if (worldPositions == null || worldPositions.Length == 0)
        {
            Debug.LogWarning(
                "FluidSimulator: no initial world positions (empty mesh volume). Using uniform grid in Fluid Initial Bounds or simulation bounds.");
            Bounds fb = simulationBounds.bounds;
            if (fluidInitialBounds != null)
                fb = fluidInitialBounds.bounds;
            worldPositions = MeshColliderBaker.UniformGridWorldPositions(fb, numParticles);
        }

        // Convert world positions → sim space
        Vector3 simOrigin = simulationBounds.bounds.min;
        var simPositions = new Vector3[numParticles];
        int wpLen = worldPositions.Length;
        for (int i = 0; i < numParticles; i++)
        {
            Vector3 local = worldPositions[i % wpLen] - simOrigin;
            simPositions[i] = new Vector3(
                local.x * mortonNormalizationFactor.x,
                local.y * mortonNormalizationFactor.y,
                local.z * mortonNormalizationFactor.z);
            simPositions[i] = Vector3.Max(simPositions[i], Vector3.one * 0.5f);
            simPositions[i] = Vector3.Min(simPositions[i], Vector3.one * 1023.5f);
        }

        initialPositionsBuffer?.Release();
        initialPositionsBuffer = new ComputeBuffer(numParticles, sizeof(float) * 3);
        initialPositionsBuffer.SetData(simPositions);

        // Dispatch init kernel
        particlesShader.SetBuffer(initializeParticlesKernel, "particlesBuffer", particlesBuffer);
        particlesShader.SetBuffer(initializeParticlesKernel, "initialPositionsBuffer", initialPositionsBuffer);
        particlesShader.SetVector("initialVelocity", initialVelocity);
        particlesShader.SetInt("count", numParticles);
        particlesShader.SetInt("minLayer", minLayer);
        particlesShader.SetInt("maxLayer", maxLayer);
        particlesShader.SetVector("simulationBoundsMin", simulationBoundsMin);
        particlesShader.SetVector("simulationBoundsMax", simulationBoundsMax);
        particlesShader.SetVector("simulationSize", simulationSize);
        particlesShader.SetVector("mortonNormalizationFactor", mortonNormalizationFactor);
        particlesShader.SetFloat("mortonMaxValue", mortonMaxValue);
        int threadGroups = Mathf.CeilToInt(numParticles / 512.0f);
        particlesShader.Dispatch(initializeParticlesKernel, threadGroups, 1, 1);

        updateParticlesKernel = particlesShader.FindKernel("UpdateParticles");
        updateParticlesUniformKernel = particlesShader.FindKernel("UpdateParticlesUniform");

        AllocateOctreeBuffersToCapacity();
    }

    /// <summary>Preallocate octree/prefix buffers so the update loop never resizes or stalls on GPU count readbacks mid-pipeline.</summary>
    private void AllocateOctreeBuffersToCapacity()
    {
        int newCapacity = Mathf.Max(512, Mathf.CeilToInt(numParticles * 1.5f));
        if (newCapacity <= maxNodesCapacity && nodesBuffer != null)
            return;
        maxNodesCapacity = newCapacity;
        GetUniformGridDims(out _, out _, out int uniformCellCount);
        int prefixCapacity = Mathf.Max(maxNodesCapacity, uniformCellCount);
        maxPrefixThreadGroups = Mathf.Max(1, (prefixCapacity + 1023) / 1024);
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

        indicators = new ComputeBuffer(prefixCapacity, sizeof(uint));
        prefixSums = new ComputeBuffer(prefixCapacity, sizeof(uint));
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
        uniformMatrixABuffer?.Release();
        uniformMatrixABuffer = new ComputeBuffer(maxNodesCapacity * UniformMatrixSlotCount, sizeof(float));
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

        AllocateUniformGridBuffers();
    }

    /// <summary>
    /// Dense N³ hash arrays (cell counts, Morton → dense node id, exclusive-prefix offsets), compact active Morton list,
    /// atomic active cell counter, and a full particle scratch buffer for counting-sort scatter.
    /// </summary>
    /// <summary>N³ cell count for current <see cref="uniformGridCellLayer"/>; matches GPU uniform grid.</summary>
    private void GetUniformGridDims(out int k, out int nPerAxis, out int cellCount)
    {
        k = UniformGridBinsPerAxisLog2;
        nPerAxis = 1 << k;
        long cellCountLong = (long)nPerAxis * nPerAxis * nPerAxis;
        if (cellCountLong > int.MaxValue)
        {
            Debug.LogError("FluidSimulator: uniform grid N³ exceeds int.MaxValue; clamping k.");
            k = 8;
            nPerAxis = 1 << k;
        }

        cellCount = checked(nPerAxis * nPerAxis * nPerAxis);
    }

    private void AllocateUniformGridBuffers()
    {
        GetUniformGridDims(out int k, out int nPerAxis, out int cellCount);
        UniformGridCellsPerAxis = nPerAxis;
        UniformGridCellCount = cellCount;

        uniformCellCountsBuffer?.Release();
        uniformDenseIndexMapBuffer?.Release();
        uniformActiveMortonListBuffer?.Release();
        uniformActiveNodeCountBuffer?.Release();
        uniformNodeAccumBuffer?.Release();

        uniformCellCountsBuffer = new ComputeBuffer(cellCount, sizeof(uint));
        uniformDenseIndexMapBuffer = new ComputeBuffer(cellCount, sizeof(uint));

        // Upper bound on nonempty cells for a single frame matches the octree node budget.
        uniformActiveMortonListBuffer = new ComputeBuffer(maxNodesCapacity, sizeof(uint));
        uniformActiveNodeCountBuffer = new ComputeBuffer(1, sizeof(uint));
        const int uniformGridSplatStripes = 8; // must match kSplatStripes in UniformGrid.compute
        uniformNodeAccumBuffer = new ComputeBuffer(7 * uniformGridSplatStripes * maxNodesCapacity, sizeof(uint));

        uniformNeighborsBuffer?.Release();
        uniformNeighborsBuffer = new ComputeBuffer(maxNodesCapacity * UniformNeighborSlotCount, sizeof(uint));

        radixSort?.ReleaseBuffers();
        radixSort = radixSortShader != null
            ? new RadixSort(radixSortShader, (uint)numParticles, (uint)maxAuxBlocks)
            : null;
    }
    
    void ResetSimulationAccumulators()
    {
        frameNumber = 0;
        cumulativeFrameTimeMs = 0.0;
        cumulativeUnityUnscaledDeltaMsForTimingLog = 0.0;
        cumulativeFullFrameWallMs = 0.0;
        fullFrameWallSamples = 0;
        lastTimingEndOfFrameRealtime = -1.0;
        hasPendingFrameTimingLog = false;
        cumulativeCgIterations = 0.0;
        cgSolveFrameCount = 0;
        averageCgIterations = 0.0f;
        lastCgIterations = 0;
        pcgResidualLogSampleCount = 0;
        cumulativePcgResidualSqLogSum = 0.0;
        cumulativePcgRelResidualLogSum = 0.0;
        cumulativePcgItersLogSum = 0.0;
        lastRenderTimeMs = 0.0;
    }

    /// <summary>Re-runs particle initialization, solid-voxel fixup, and clears timing/recording state (same as scene start).</summary>
    public void ResetSimulation()
    {
        ResolveBoundsColliders();
        if (simulationBounds == null || fluidInitialBounds == null)
        {
            Debug.LogWarning("FluidSimulator: cannot reset — assign Simulation Bounds and Fluid Initial Bounds colliders.");
            return;
        }
        if (particlesShader == null)
        {
            Debug.LogWarning("FluidSimulator: cannot reset — particles compute shader is missing.");
            return;
        }

        InitializeParticleSystem();
        if (particlesBuffer == null)
            return;

        BakeAndUploadSolidVoxels();
        ResetSimulationAccumulators();

        if (recorder != null)
            recorder.StartNewRun();
    }
    
    private void SortParticles()
    {
        if (radixSort == null || radixSortShader == null)
            return;
        radixSort.Sort(particlesBuffer, particlesBuffer, (uint)numParticles);
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
        uniformNeighborsBuffer?.Release();
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
        uniformMatrixABuffer?.Release();
        nnzPerNode?.Release();
        csrRowPtr?.Release();
        csrColIndices?.Release();
        csrRowIndices?.Release();
        csrValues?.Release();
        phiBuffer?.Release();
        phiBuffer_Read?.Release();
        dirtyFlagBuffer?.Release();
        mortonCodesBuffer?.Release();
        solidSDFBuffer?.Release();
        initialPositionsBuffer?.Release();
        uniformCellCountsBuffer?.Release();
        uniformDenseIndexMapBuffer?.Release();
        uniformActiveMortonListBuffer?.Release();
        uniformActiveNodeCountBuffer?.Release();
        uniformNodeAccumBuffer?.Release();
        ReleasePreconditionerBuffers();
        ReleaseLeafOnlyBuffers();
        ReleaseLeafOnlyWeightsBuffers();
    }
}
