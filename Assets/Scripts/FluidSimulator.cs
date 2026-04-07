using UnityEngine;
using UnityEngine.InputSystem;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using UnityEngine.Rendering;
using System.IO;

// Main FluidSimulator class - split into partial classes for better organization
// Core simulation logic: particle management, main loop, initialization
// See: FluidEnums.cs, FluidRendering.cs, FluidPreconditioner.cs,
//      FluidOctree.cs, FluidSolver.cs, FluidLeafOnlyInputs.cs, FluidLeafOnlyWeights.cs,
//      FluidLeafOnlyPrecondApply.cs, RadixSort.cs
public partial class FluidSimulator : MonoBehaviour
{
    public BoxCollider simulationBounds;
    public BoxCollider fluidInitialBounds;
    
    public ComputeShader radixSortShader;
    public ComputeShader particlesShader;
    public ComputeShader nodesPrefixSumsShader;
    public ComputeShader nodesShader;
    public ComputeShader cgSolverShader;
    public ComputeShader csrBuilderShader; // NEW: CSR construction shader
    public int numParticles = 1048576;
    
    // CG Solver parameters
    public int maxCgIterations = 400;
    public float convergenceThreshold = 1e-05f;
    
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
    private int extractMortonCodesKernel;
    private int calculateDensityGradientKernel;
    // Cached solver kernel IDs (set once in InitSolverKernels)
    private int buildMatrixAKernelId;
    private int applyMatrixAndDotKernelId;
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
    private ComputeBuffer diffusionGradientBuffer; // Precomputed normalized density gradient per node
    private ComputeBuffer dispatchArgsBuffer;       // 3-uint indirect dispatch args for DispatchIndirect
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
    public ComputeBuffer PhiBuffer => phiBuffer;
    public int NumNodes => numNodes;
    private int numUniqueNodes; // rename to numUniqueNodes
    private int layer;
    public float gravity = 100.0f;
    public bool useRealTime = false; // When true, uses Time.deltaTime instead of fixed frameRate
    public float frameRate = 30.0f;
    [Range(0, 10)] public int minLayer = 4;
    [Range(0, 10)] public int maxLayer = 10;
    [Tooltip("Each frame: GPU readback of nodesBuffer and Debug.Log of how many nodes sit at each layer index (after findNeighbors).")]
    public bool logLayerCountsPerFrame = true;
    [Header("Debug: spatial cell overlaps")]
    [Tooltip("When overlaps exist, emit a second log with sample node pairs (index, morton, layer, active, mass, position). Overlap = same (layer, mortonCode >> 3*layer).")]
    public bool logSpatialOverlapDetails = false;
    [Tooltip("After CreateLeaves: readback + leaf-pack stats (duplicate full morton, minLayer buckets, coarse maxLayer noise). Does not use the solver cell key.")]
    public bool diagSpatialOverlapAfterCreateLeaves = false;
    [Tooltip("After SortParticles: readback + morton monotonicity / duplicate morton counts (O(n) on GPU buffer).")]
    public bool diagInspectParticleMortonAfterSort = false;
    [Tooltip("Append deinterleaved grid (x,y,z) and cellMin@layer to overlap sample lines.")]
    public bool diagSpatialOverlapVerboseTraces = false;
    [Tooltip("GPU readback + overlap check after each compactNodes inside the layer loop. Very heavy.")]
    public bool diagSpatialOverlapAfterEachLayerCompact = false;
    [Tooltip("GPU readback + overlap check after each ProcessNodes, before compact. Distinguishes 'ProcessNodes wrote dup actives' vs 'compact scatter' issues. Very heavy.")]
    public bool diagSpatialOverlapAfterProcessNodes = false;
    public PreconditionerType preconditioner = PreconditionerType.Jacobi;

    private bool hasShownWaitMessage = false;
    private int frameNumber = 0;
    private double cumulativeFrameTimeMs = 0.0;
    private double cumulativeCgIterations = 0.0;
    private int cgSolveFrameCount = 0;
    private float averageCgIterations = 0.0f;
    private int lastCgIterations = 0;
    
    // Pause/resume functionality
    private bool isPaused = false;
    
    private System.Diagnostics.Stopwatch totalOctreeSw;
    private System.Diagnostics.Stopwatch renderSw = new System.Diagnostics.Stopwatch();
    private double lastRenderTimeMs = 0.0;
    
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
    /// <summary>Reused for uint-keyed probes (full morton, particle counts). Cleared per use.</summary>
    private readonly Dictionary<uint, int> inspectUintKeyScratch = new Dictionary<uint, int>(65536);
    private readonly List<(int firstIdx, int secondIdx)> spatialOverlapPairScratch = new List<(int, int)>(16);
    private readonly StringBuilder layerCountLogSb = new StringBuilder(128);
    private readonly StringBuilder spatialOverlapDetailSb = new StringBuilder(512);
    private Particle[] particlesCPU;
    private string str;

    // Training data recorder
    public TrainingDataRecorder recorder;
    
    [Header("Rendering")]

    // Rendering mode enum
    public RenderingMode renderingMode = RenderingMode.Thickness;
    
    // Thickness source selection
    public ThicknessSource thicknessSource = ThicknessSource.Nodes;

    public Light mainLight; // Assign your Directional Light
    
    public ComputeShader blurCompute; // Assign BilateralBlur.compute here
    
    // Material to render particles as points (assign a shader like Custom/ParticlesPoints)
    public Material particlesMaterial;
    
    // Material to render particles with depth shader (assign a shader like Fluid/ParticleDepth)
    public Material particleDepthMaterial;
    
    // Material to render nodes with thickness shader (assign a shader like Custom/NodeThickness)
    public Material nodeThicknessMaterial;
    
    // Material to render nodes with wireframe shader (assign a shader like Custom/NodeWireframe)
    public Material nodeWireframeMaterial;
    
    // Material to render particles with thickness shader (assign a shader like Custom/ParticleThickness)
    public Material particleThicknessMaterial;
    
    // Material to generate normals from depth (assign a shader like Fluid/NormalsFromDepth)
    public Material normalMaterial; // Assign "Fluid/NormalsFromDepth" in Inspector

    public Material compositeMaterial; // Assign "Fluid/FluidRender" here
    
    
    // Debug display material for visualizing textures
    public Material debugDisplayMaterial; // Assign "Custom/DebugTextureDisplay" in Inspector

    public Color fluidColor = new Color(0.2f, 0.6f, 1.0f); // Input for extinction
    public bool useSkybox = true; // Toggle between cubemap and gradient sky
    public Cubemap skyboxTexture; // Skybox cubemap for environment sampling
    public Color skyHorizonColor = new Color(0.0f, 0.0f, 0.0f); // Dark color at horizon (looking down)
    public Color skyTopColor = new Color(1.0f, 1.0f, 1.0f); // Light color at top (looking up)
    [Range(-20.0f, 20.0f)] public float sunIntensity = -2.3f; // Exponent: actual intensity = exp(value)
    [Range(0.0f, 500.0f)] public float sunSharpness = 53.0f; // Sun highlight sharpness (power exponent)
    // Calculated depth values (updated each frame based on camera position)
    private float calculatedDepthDisplayScale = 0.0f;  // Exponent for depth scale
    private float calculatedDepthMinValue = 0.0f;     // Exponent for depth min
    private float calculatedMinDepth = 0.0f;          // Actual min depth value
    private float calculatedMaxDepth = 0.0f;          // Actual max depth value
    [Range(-20.0f, 20.0f)] public float thicknessScaleNodes = -8.9f; // Exponent: actual scale = exp(value) for nodes
    [Range(-20.0f, 20.0f)] public float thicknessScaleParticles = -9.6f; // Exponent: actual scale = exp(value) for particles

    [Range(0, 100)] public int depthBlurRadius = 6;
    [Range(0.0001f, 10.0f)] public float depthBlurThreshold = 1.24f;
    [Range(0, 100)] public int thicknessBlurRadius = 22;
    [Range(0.0001f, 1.0f)] public float particleRadius = 0.066f; // Radius for particle points (world space)
    [Range(0.0001f, 1.0f)] public float depthRadius = 0.279f; // Radius for depth quads (world space)
    [Range(0.0001f, 1.0f)] public float thicknessRadius = 0.776f; // Radius for particle thickness quads (world space)
    [Range(0, 20)] public float absorptionStrength = 5.8f;
    [Range(0.0f, 1.0f)] public float depthOfFieldStrength = 0.076f;
    [Range(0.0f, 1.0f)] public float reflectionStrength = 0.633f; // Multiplier for reflection intensity
    [Range(0.0f, 1.0f)] public float reflectionTint = 0.418f; // Amount to mix refraction color into reflection
    [Range(0.0f, 1.0f)] public float fresnelClamp = 0.8f; // Maximum fresnel weight to keep water visible
    [Range(0, 2)] public float refractionScale = 0.57f; // Scales normal map intensity (0.0 = flat mirror, 1.0 = normal, 2.0 = super distorted)
    
    // Helper method for resizing buffers
    private void ResizeBuffer(ref ComputeBuffer buffer, int count, int stride)
    {
        if (buffer != null && buffer.count >= count) return; // Capacity is sufficient

        buffer?.Release();
        // Allocate next power of two to prevent frequent resizing
        int newSize = Mathf.NextPowerOfTwo(Mathf.Max(count, 512));
        buffer = new ComputeBuffer(newSize, stride);
    }
    
    // Mesh for instanced node rendering (quad for billboard-style rendering)
    private Mesh quadMesh;
    private ComputeBuffer argsBuffer;
    private ComputeBuffer particleArgsBuffer;
    private CommandBuffer thicknessCmd;
    private CommandBuffer depthCmd;
    private RenderTexture thicknessTexture;
    private RenderTexture fluidDepthTexture; // Stores the final smooth depth for normals
    private RenderTexture rawDepthTexture; // Stores unblurred depth for visualization
    private RenderTexture fluidNormalTexture; // Stores the surface normals (RGB = slope vectors)
    private RenderTexture fluidThicknessTexture; // Smoothed thickness
    private RenderTexture nodesTexture; // Stores wireframe nodes rendering
    
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

    void Start()
    {
        ValidateOctreeLayers();
        InitializeParticleSystem();
        // InitializeInitialParticles();

        // Cache all kernel IDs once so hot paths never call FindKernel
        InitSolverKernels();
        InitOctreeKernels();
        TryLoadLeafOnlyWeightsFromDisk();

        // Create quad mesh and args buffers for node and particle rendering
        CreateQuadMesh();
        CreateArgsBuffer();
        CreateParticleArgsBuffer();
        
        // Create thickness render texture
        if (thicknessTexture == null)
        {
            thicknessTexture = new RenderTexture(Screen.width, Screen.height, 0, RenderTextureFormat.RFloat);
            thicknessTexture.enableRandomWrite = true;
            thicknessTexture.Create();
        }
        
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
            particlesCPU[indices[i]].velocity = new Vector3(100.0f, 0.0f, 0.0f);
        }

        particlesBuffer.SetData(particlesCPU);
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
        
        // Calculate maxDetailCellSize for volume calculations
        // Use the normalized simulation bounds (simulationBoundsMin is now Vector3.zero)
        Vector3 simulationSize = simulationBoundsMax;
        maxDetailCellSize = Mathf.Min(simulationSize.x, simulationSize.y, simulationSize.z) / 1024.0f;

        // Step 1: Sort particles
        var sortSw = System.Diagnostics.Stopwatch.StartNew();
        SortParticles();
        sortSw.Stop();
        if (diagInspectParticleMortonAfterSort)
            LogParticleMortonInspection("After SortParticles");

        // Step 2: Find unique particles and create leaves
        var findUniqueSw = System.Diagnostics.Stopwatch.StartNew();
        findUniqueParticles();
        findUniqueSw.Stop();

        var createLeavesSw = System.Diagnostics.Stopwatch.StartNew();
        CreateLeaves();
        createLeavesSw.Stop();
        if (diagSpatialOverlapAfterCreateLeaves)
            LogCreateLeavesPackInspection();

        // Step 3: Layer loop (layers 1-10)
        var layerLoopSw = System.Diagnostics.Stopwatch.StartNew();
        for (layer = layer + 1; layer <= maxLayer; layer++)
        {
            
            findUniqueNodes();
            ProcessNodes();
            if (diagSpatialOverlapAfterProcessNodes)
                LogSpatialOverlapStage($"After ProcessNodes before compact (octree layer={layer})", maxSamplePairs: 12, activeNodesOnly: true, verboseTraces: diagSpatialOverlapVerboseTraces);
            compactNodes();

            // --- NEW: Refresh SoA Buffer after compaction ---
            // The nodes have moved/shrunk, so we must extract the codes again 
            // for the next iteration (and for findNeighbors later)
            if (mortonCodesBuffer != null && nodesBuffer != null)
            {
                nodesShader.SetBuffer(extractMortonCodesKernel, "nodesBuffer", nodesBuffer);
                nodesShader.SetBuffer(extractMortonCodesKernel, "mortonCodesBuffer", mortonCodesBuffer);
                nodesShader.SetInt("numNodes", numNodes);
                int groups = Mathf.CeilToInt(numNodes / 512.0f);
                nodesShader.Dispatch(extractMortonCodesKernel, groups, 1, 1);
            }
        }
        layerLoopSw.Stop();

        // Step 4: Find neighbors
        var findNeighborsSw = System.Diagnostics.Stopwatch.StartNew();
        findNeighbors();
        findNeighborsSw.Stop();

        string layerCountsLine = BuildLayerCountsSummaryForLog();

        // Step 4.5: Calculate density gradients
        var calculateGradientsSw = System.Diagnostics.Stopwatch.StartNew();
        CalculateDensityGradients();
        calculateGradientsSw.Stop();

        // Step 5: Compute level set (distance field)
        var computeLevelSetSw = System.Diagnostics.Stopwatch.StartNew();
        // ComputeLevelSet();
        computeLevelSetSw.Stop();

        // Step 6: Store old velocities for FLIP method
        var storeOldVelocitiesSw = System.Diagnostics.Stopwatch.StartNew();
        StoreOldVelocities();
        storeOldVelocitiesSw.Stop();

        // Step 7: Apply external forces on the grid
        var applyExternalForcesSw = System.Diagnostics.Stopwatch.StartNew();
        ApplyExternalForces();
        applyExternalForcesSw.Stop();

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
        cumulativeFrameTimeMs += frameSw.Elapsed.TotalMilliseconds;
        double averageFrameTimeMs = cumulativeFrameTimeMs / Math.Max(1, frameNumber);
        string averageCgIterationsText = cgSolveFrameCount > 0 ? averageCgIterations.ToString("F2") : "N/A";
        Debug.Log($"Frame {frameNumber} Summary:\n" +
                 $"• Total Frame: {frameSw.Elapsed.TotalMilliseconds:F2} ms\n" +
                 $"• Avg Frame Time: {averageFrameTimeMs:F2} ms\n" +
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

    /// <summary>Non-null when <see cref="logLayerCountsPerFrame"/> is true: nodes-per-layer after findNeighbors (GPU readback).</summary>
    private string BuildLayerCountsSummaryForLog()
    {
        if (!logLayerCountsPerFrame)
            return null;
        if (numNodes <= 0 || nodesBuffer == null)
            return "Nodes per layer: (none)";

        if (nodesCPU == null || nodesCPU.Length < numNodes)
            nodesCPU = new Node[Mathf.NextPowerOfTwo(Mathf.Max(numNodes, 512))];

        nodesBuffer.GetData(nodesCPU, 0, 0, numNodes);

        AnalyzeNodesCpuForHistogramAndSpatialOverlaps(
            recordSamplePairs: logSpatialOverlapDetails,
            maxSamplePairs: 16,
            activeNodesOnly: false,
            out int dupCount,
            out _);

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

        if (logSpatialOverlapDetails && dupCount > 0)
            Debug.LogWarning(BuildSpatialOverlapSamplesMessage("After findNeighbors (frame summary readback)", diagSpatialOverlapVerboseTraces));

        return layerCountLogSb.ToString();
    }

    /// <summary>Same convention as <c>Nodes.compute</c> / wireframe: 10 bits per axis.</summary>
    private static void DeinterleaveMorton(uint m, out uint gx, out uint gy, out uint gz)
    {
        gx = 0;
        gy = 0;
        gz = 0;
        for (int i = 0; i < 10; i++)
        {
            gx |= ((m >> (3 * i + 0)) & 1u) << i;
            gy |= ((m >> (3 * i + 1)) & 1u) << i;
            gz |= ((m >> (3 * i + 2)) & 1u) << i;
        }
    }

    private static uint CellMortonMinCorner(uint mortonCode, int layer)
    {
        int shift = layer * 3;
        if (shift <= 0) return mortonCode;
        if (shift >= 32) return 0u;
        uint mask = (1u << shift) - 1u;
        return mortonCode & ~mask;
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

    private void AppendNodeTraceLine(StringBuilder sb, int idx, in Node n, bool verbose)
    {
        int L = (int)n.layer;
        int sh = Mathf.Clamp(L, 0, 15) * 3;
        uint prefix = sh >= 32 ? 0u : n.mortonCode >> sh;
        sb.Append("    [").Append(idx).Append("] layer=").Append(L)
            .Append(" morton=0x").Append(n.mortonCode.ToString("X8"))
            .Append(" prefix@L=0x").Append(prefix.ToString("X8"))
            .Append(" active=").Append(n.active)
            .Append(" mass=").Append(n.mass.ToString("G6", System.Globalization.CultureInfo.InvariantCulture))
            .Append(" pos=").Append(n.position.ToString("F4"));
        if (!verbose)
        {
            sb.Append('\n');
            return;
        }
        DeinterleaveMorton(n.mortonCode, out uint gx, out uint gy, out uint gz);
        sb.Append("\n      grid(full morton)=( ").Append(gx).Append(", ").Append(gy).Append(", ").Append(gz).Append(" )");
        uint cellMc = CellMortonMinCorner(n.mortonCode, L);
        DeinterleaveMorton(cellMc, out uint cx, out uint cy, out uint cz);
        uint side = L >= 31 ? 0u : (1u << Mathf.Clamp(L, 0, 30));
        sb.Append(" | cellMin@storedL=( ").Append(cx).Append(", ").Append(cy).Append(", ").Append(cz)
            .Append(" ) cellSideGrid=").Append(side).Append('\n');
    }

    /// <summary>Assumes <see cref="nodesCPU"/> is filled for <see cref="numNodes"/>. Clears and fills histogram + overlap scratch.</summary>
    /// <param name="activeNodesOnly">If true, only nodes with <c>active != 0</c> (skip merged/inactive slots still present before compact).</param>
    private void AnalyzeNodesCpuForHistogramAndSpatialOverlaps(bool recordSamplePairs, int maxSamplePairs, bool activeNodesOnly, out int dupCount, out int mortonOrderInversions)
    {
        int[] h = layerCountHistogramScratch;
        int[] dupPerLayer = spatialDupPerLayerScratch;
        Array.Clear(h, 0, h.Length);
        Array.Clear(dupPerLayer, 0, dupPerLayer.Length);
        spatialOverlapFirstIndexScratch.Clear();
        spatialOverlapPairScratch.Clear();

        dupCount = 0;
        mortonOrderInversions = 0;
        uint prevMorton = 0;
        bool havePrev = false;

        for (int i = 0; i < numNodes; i++)
        {
            ref Node n = ref nodesCPU[i];
            if (activeNodesOnly && n.active == 0)
                continue;

            int L = (int)n.layer;
            if (L >= 0 && L < h.Length)
                h[L]++;

            if (!activeNodesOnly)
            {
                if (havePrev && n.mortonCode < prevMorton)
                    mortonOrderInversions++;
                prevMorton = n.mortonCode;
                havePrev = true;
            }

            ulong key = SpatialCellKey(in n);
            if (spatialOverlapFirstIndexScratch.TryGetValue(key, out int firstI))
            {
                dupCount++;
                if (L >= 0 && L < dupPerLayer.Length)
                    dupPerLayer[L]++;
                if (recordSamplePairs && spatialOverlapPairScratch.Count < maxSamplePairs)
                    spatialOverlapPairScratch.Add((firstI, i));
            }
            else
                spatialOverlapFirstIndexScratch[key] = i;
        }
    }

    private string BuildSpatialOverlapSamplesMessage(string stageTag, bool verbose)
    {
        spatialOverlapDetailSb.Clear();
        spatialOverlapDetailSb.Append("[SpatialOverlap] ").Append(stageTag)
            .Append(" — duplicate (layer, morton prefix) cells; samples follow.\n");
        foreach (var p in spatialOverlapPairScratch)
        {
            spatialOverlapDetailSb.Append("  pair: ").Append(p.firstIdx).Append(" vs ").Append(p.secondIdx).Append('\n');
            AppendNodeTraceLine(spatialOverlapDetailSb, p.firstIdx, nodesCPU[p.firstIdx], verbose);
            AppendNodeTraceLine(spatialOverlapDetailSb, p.secondIdx, nodesCPU[p.secondIdx], verbose);
        }
        return spatialOverlapDetailSb.ToString();
    }

    /// <summary>Optional pipeline checkpoints: GPU readback + overlap stats (and optional morton-order check).</summary>
    /// <param name="activeNodesOnly">Use true after ProcessNodes (inactive siblings still in the buffer until compact).</param>
    private void LogSpatialOverlapStage(string stageTag, int maxSamplePairs = 12, bool activeNodesOnly = false, bool verboseTraces = false)
    {
        if (numNodes <= 0 || nodesBuffer == null)
        {
            Debug.Log($"{stageTag} — numNodes=0, skip overlap check.");
            return;
        }

        if (nodesCPU == null || nodesCPU.Length < numNodes)
            nodesCPU = new Node[Mathf.NextPowerOfTwo(Mathf.Max(numNodes, 512))];

        nodesBuffer.GetData(nodesCPU, 0, 0, numNodes);
        AnalyzeNodesCpuForHistogramAndSpatialOverlaps(
            recordSamplePairs: true,
            maxSamplePairs: maxSamplePairs,
            activeNodesOnly: activeNodesOnly,
            out int dupCount,
            out int inv);

        if (dupCount == 0)
        {
            if (!activeNodesOnly && inv > 0)
                Debug.LogWarning($"[SpatialOverlap] {stageTag} — no duplicate cells but mortonOrderInversions={inv} (expected sorted mortonCodesBuffer order), numNodes={numNodes}");
            else
                Debug.Log($"{stageTag} — spatial overlaps=0, mortonOrderInversions={(activeNodesOnly ? "n/a" : inv.ToString())}, numNodes={numNodes}, activeOnly={activeNodesOnly}");
            return;
        }

        spatialOverlapDetailSb.Clear();
        spatialOverlapDetailSb.Append("[SpatialOverlap] ").Append(stageTag)
            .Append(" numNodes=").Append(numNodes)
            .Append(" activeOnly=").Append(activeNodesOnly)
            .Append(" duplicateCells=").Append(dupCount)
            .Append(" mortonOrderInversions=").Append(activeNodesOnly ? "n/a" : inv.ToString())
            .Append(activeNodesOnly ? "\n" : " (inversions imply buffer not sorted by mortonCode)\n");
        foreach (var p in spatialOverlapPairScratch)
        {
            spatialOverlapDetailSb.Append("  pair: ").Append(p.firstIdx).Append(" vs ").Append(p.secondIdx).Append('\n');
            AppendNodeTraceLine(spatialOverlapDetailSb, p.firstIdx, nodesCPU[p.firstIdx], verboseTraces);
            AppendNodeTraceLine(spatialOverlapDetailSb, p.secondIdx, nodesCPU[p.secondIdx], verboseTraces);
        }
        Debug.LogWarning(spatialOverlapDetailSb.ToString());
    }

    /// <summary>CreateLeaves: all leaves share <c>node.layer == maxLayer</c>, so solver-style (layer,prefix) collisions are expected noise. Report meaningful invariants instead.</summary>
    private void LogCreateLeavesPackInspection()
    {
        if (numNodes <= 0 || nodesBuffer == null)
        {
            Debug.Log("[LeafInspect] After CreateLeaves: numNodes=0, skip.");
            return;
        }

        if (nodesCPU == null || nodesCPU.Length < numNodes)
            nodesCPU = new Node[Mathf.NextPowerOfTwo(Mathf.Max(numNodes, 512))];

        nodesBuffer.GetData(nodesCPU, 0, 0, numNodes);

        inspectUintKeyScratch.Clear();
        int duplicateFullMortonExtra = 0;
        int sampleA = -1, sampleB = -1;
        for (int i = 0; i < numNodes; i++)
        {
            uint mc = nodesCPU[i].mortonCode;
            if (inspectUintKeyScratch.TryGetValue(mc, out int firstI))
            {
                duplicateFullMortonExtra++;
                if (sampleA < 0)
                {
                    sampleA = firstI;
                    sampleB = i;
                }
            }
            else
                inspectUintKeyScratch[mc] = i;
        }

        spatialOverlapFirstIndexScratch.Clear();
        int dupMinLayerBuckets = 0;
        int minLayerSampleA = -1, minLayerSampleB = -1;
        for (int i = 0; i < numNodes; i++)
        {
            ulong k = SpatialCellKeyAtLayer(nodesCPU[i].mortonCode, minLayer);
            if (spatialOverlapFirstIndexScratch.TryGetValue(k, out int firstI))
            {
                dupMinLayerBuckets++;
                if (minLayerSampleA < 0)
                {
                    minLayerSampleA = firstI;
                    minLayerSampleB = i;
                }
            }
            else
                spatialOverlapFirstIndexScratch[k] = i;
        }

        spatialOverlapFirstIndexScratch.Clear();
        int coarseCollisionsAtMaxLayer = 0;
        for (int i = 0; i < numNodes; i++)
        {
            ulong k = SpatialCellKeyAtLayer(nodesCPU[i].mortonCode, maxLayer);
            if (spatialOverlapFirstIndexScratch.ContainsKey(k))
                coarseCollisionsAtMaxLayer++;
            else
                spatialOverlapFirstIndexScratch[k] = i;
        }

        int mortonInv = 0;
        for (int i = 1; i < numNodes; i++)
        {
            if (nodesCPU[i].mortonCode < nodesCPU[i - 1].mortonCode)
                mortonInv++;
        }

        Array.Clear(layerCountHistogramScratch, 0, layerCountHistogramScratch.Length);
        for (int i = 0; i < numNodes; i++)
        {
            int L = (int)nodesCPU[i].layer;
            if (L >= 0 && L < layerCountHistogramScratch.Length)
                layerCountHistogramScratch[L]++;
        }

        spatialOverlapDetailSb.Clear();
        spatialOverlapDetailSb.Append("[LeafInspect] After CreateLeaves numNodes=").Append(numNodes)
            .Append("\n  duplicateFullMortonExtra=").Append(duplicateFullMortonExtra)
            .Append(" (expect 0 — two leaves with identical particle morton)")
            .Append("\n  duplicateMinLayerBuckets=").Append(dupMinLayerBuckets)
            .Append(" (expect 0 — should match findUniqueParticles @ minLayer)")
            .Append("\n  coarseBucketCollisionsAtMaxLayer=").Append(coarseCollisionsAtMaxLayer)
            .Append(" (expected large; every node.layer=maxLayer so many leaves share a coarse key — not a bug)")
            .Append("\n  mortonOrderInversions=").Append(mortonInv)
            .Append(" (expect 0 if sorted by morton)")
            .Append("\n  storedLayerHistogram: ");
        bool anyL = false;
        for (int L = 0; L < layerCountHistogramScratch.Length; L++)
        {
            if (layerCountHistogramScratch[L] == 0) continue;
            if (anyL) spatialOverlapDetailSb.Append(' ');
            spatialOverlapDetailSb.Append('L').Append(L).Append('=').Append(layerCountHistogramScratch[L]);
            anyL = true;
        }
        spatialOverlapDetailSb.Append('\n');

        if (duplicateFullMortonExtra > 0 || dupMinLayerBuckets > 0 || mortonInv > 0)
        {
            if (duplicateFullMortonExtra > 0 && sampleA >= 0)
            {
                spatialOverlapDetailSb.Append("  sample duplicate full morton: indices ").Append(sampleA).Append(" vs ").Append(sampleB).Append('\n');
                AppendNodeTraceLine(spatialOverlapDetailSb, sampleA, nodesCPU[sampleA], diagSpatialOverlapVerboseTraces);
                AppendNodeTraceLine(spatialOverlapDetailSb, sampleB, nodesCPU[sampleB], diagSpatialOverlapVerboseTraces);
            }
            if (dupMinLayerBuckets > 0 && minLayerSampleA >= 0)
            {
                spatialOverlapDetailSb.Append("  sample duplicate minLayer=").Append(minLayer).Append(" bucket: indices ")
                    .Append(minLayerSampleA).Append(" vs ").Append(minLayerSampleB).Append('\n');
                AppendNodeTraceLine(spatialOverlapDetailSb, minLayerSampleA, nodesCPU[minLayerSampleA], diagSpatialOverlapVerboseTraces);
                AppendNodeTraceLine(spatialOverlapDetailSb, minLayerSampleB, nodesCPU[minLayerSampleB], diagSpatialOverlapVerboseTraces);
            }
            Debug.LogWarning(spatialOverlapDetailSb.ToString());
        }
        else
            Debug.Log(spatialOverlapDetailSb.ToString());
    }

    private void LogParticleMortonInspection(string stageTag)
    {
        if (particlesBuffer == null || numParticles <= 0)
        {
            Debug.Log($"[ParticleInspect] {stageTag}: no particles, skip.");
            return;
        }

        if (particlesCPU == null || particlesCPU.Length < numParticles)
            particlesCPU = new Particle[numParticles];

        particlesBuffer.GetData(particlesCPU, 0, 0, numParticles);

        int sortInversions = 0;
        for (int i = 1; i < numParticles; i++)
        {
            if (particlesCPU[i].mortonCode < particlesCPU[i - 1].mortonCode)
                sortInversions++;
        }

        int consecutiveEqual = 0;
        for (int i = 1; i < numParticles; i++)
        {
            if (particlesCPU[i].mortonCode == particlesCPU[i - 1].mortonCode)
                consecutiveEqual++;
        }

        inspectUintKeyScratch.Clear();
        for (int i = 0; i < numParticles; i++)
        {
            uint m = particlesCPU[i].mortonCode;
            inspectUintKeyScratch.TryGetValue(m, out int c);
            inspectUintKeyScratch[m] = c + 1;
        }

        int distinctMorton = inspectUintKeyScratch.Count;
        int mortonValuesWithCountGt1 = 0;
        foreach (var kv in inspectUintKeyScratch)
        {
            if (kv.Value > 1)
                mortonValuesWithCountGt1++;
        }

        int extraParticleSlots = numParticles - distinctMorton;

        spatialOverlapDetailSb.Clear();
        spatialOverlapDetailSb.Append("[ParticleInspect] ").Append(stageTag)
            .Append(" numParticles=").Append(numParticles)
            .Append("\n  morton sort inversions=").Append(sortInversions)
            .Append(" (expect 0 after radix sort — if >0, sort is broken)")
            .Append("\n  consecutive equal morton pairs=").Append(consecutiveEqual)
            .Append("\n  distinct morton codes=").Append(distinctMorton)
            .Append("\n  morton codes with count>1=").Append(mortonValuesWithCountGt1)
            .Append("\n  extra particle slots vs distinct=").Append(extraParticleSlots)
            .Append(" (= sum of (count-1) over duplicated codes)")
            .Append("\n  Note: many particles sharing one morton is normal — EncodeMorton3D uses 10 bits/axis (~1M cells);")
            .Append("\n  ~1M particles pack into fewer distinct codes, so duplicates do not imply a sort or leaf bug by themselves.")
            .Append('\n');

        if (sortInversions > 0)
            Debug.LogWarning(spatialOverlapDetailSb.ToString());
        else
            Debug.Log(spatialOverlapDetailSb.ToString());
    }

    private void CalculateDensityGradients()
    {
        if (nodesShader == null)
        {
            Debug.LogError("Nodes compute shader is not assigned.");
            return;
        }

        nodesShader.SetBuffer(calculateDensityGradientKernel, "nodesBuffer", nodesBuffer);
        nodesShader.SetBuffer(calculateDensityGradientKernel, "neighborsBuffer", neighborsBuffer);
        nodesShader.SetBuffer(calculateDensityGradientKernel, "diffusionGradientBuffer", diffusionGradientBuffer);
        nodesShader.SetInt("numNodes", numNodes);
        
        int groups = Mathf.CeilToInt(numNodes / 512.0f);
        nodesShader.Dispatch(calculateDensityGradientKernel, groups, 1, 1);
    }

    private void StoreOldVelocities()
    {
        const int nodeStride = sizeof(float) * 3 + sizeof(float) * 3 + sizeof(float) * 6 + sizeof(float) + sizeof(uint) * 3;
        ResizeBuffer(ref nodesBufferOld, numNodes, nodeStride);

        // GPU copy: copyNodes kernel writes tempNodesBuffer[i] → nodesBuffer[i],
        // so bind src→tempNodesBuffer slot, dst→nodesBuffer slot.
        nodesPrefixSumsShader.SetBuffer(copyNodesKernelId, "tempNodesBuffer", nodesBuffer);
        nodesPrefixSumsShader.SetBuffer(copyNodesKernelId, "nodesBuffer", nodesBufferOld);
        nodesPrefixSumsShader.SetInt("len", numNodes);
        nodesPrefixSumsShader.Dispatch(copyNodesKernelId, Mathf.CeilToInt(numNodes / 512.0f), 1, 1);
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
        particlesShader.SetBuffer(updateParticlesKernel, "mortonCodesBuffer", mortonCodesBuffer);
        particlesShader.SetBuffer(updateParticlesKernel, "nodesBufferOld", nodesBufferOld);
        particlesShader.SetBuffer(updateParticlesKernel, "particlesBuffer", particlesBuffer);
        particlesShader.SetBuffer(updateParticlesKernel, "diffusionGradientBuffer", diffusionGradientBuffer);
        particlesShader.SetInt("numNodes", numNodes);
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
        particlesShader.Dispatch(updateParticlesKernel, threadGroups, 1, 1);
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
        
        if (nodesShader != null)
        {
            extractMortonCodesKernel = nodesShader.FindKernel("ExtractMortonCodes");
        }
        
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
        divergenceBuffer?.Release();
        residualBuffer?.Release();
        cgAlphaBuffer?.Release();
        cgBetaBuffer?.Release();
        cgRhoBuffer?.Release();
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
        ReleasePreconditionerBuffers();
        ReleaseLeafOnlyBuffers();
        ReleaseLeafOnlyWeightsBuffers();

        // Clean up render textures
        if (fluidDepthTexture != null) fluidDepthTexture.Release();
        if (fluidNormalTexture != null) fluidNormalTexture.Release();
        if (nodesTexture != null) nodesTexture.Release();
    }
}
