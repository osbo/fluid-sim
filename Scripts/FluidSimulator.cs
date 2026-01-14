using UnityEngine;
using UnityEngine.InputSystem;
using System;
using System.Linq;
using System.Runtime.InteropServices;
using UnityEngine.Rendering;
using System.IO;
using System.Collections.Generic;

// Main FluidSimulator class - split into partial classes for better organization
// Core simulation logic: particle management, main loop, initialization
// See: FluidEnums.cs, FluidRendering.cs, FluidPreconditioner.cs,
//      FluidOctree.cs, FluidSolver.cs, RadixSort.cs
public partial class FluidSimulator : MonoBehaviour
{
    public BoxCollider simulationBounds;
    public BoxCollider fluidInitialBounds;
    
    public ComputeShader radixSortShader;
    public ComputeShader particlesShader;
    public ComputeShader nodesPrefixSumsShader;
    public ComputeShader nodesShader;
    public ComputeShader cgSolverShader;
    public ComputeShader preconditionerShader; // Assign Preconditioner.compute in Inspector
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
    private int copyNodesKernel;
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
    private ComputeBuffer phiBuffer;
    private ComputeBuffer phiBuffer_Read;
    private ComputeBuffer dirtyFlagBuffer;
    private ComputeBuffer tokenBuffer; // For neural preconditioner token assembly
    private ComputeBuffer tokenBufferOut; // Output buffer for transformer layers
    private ComputeBuffer matrixGBuffer; // Output buffer for preconditioner matrix G
    private ComputeBuffer zBuffer; // Intermediate buffer for PCG preconditioner application (used as 'u' in shader)
    private ComputeBuffer zVectorBuffer; // Preconditioned residual vector 'z' for PCG
    private ComputeBuffer scatterIndicesBuffer; // NEW: Pre-computed scatter indices for optimization
    private ComputeBuffer diagonalBuffer; // Diagonal of Laplacian matrix A for Jacobi preconditioning
    private ComputeBuffer bufferQ; // Q buffer for attention
    private ComputeBuffer bufferK; // K buffer for attention
    private ComputeBuffer bufferV; // V buffer for attention
    private ComputeBuffer bufferAttn; // Attention output buffer
    private ComputeBuffer diffusionGradientBuffer; // Precomputed normalized density gradient per node
    public TextAsset modelWeightsAsset; // Assign model_weights.bytes from Assets/Scripts/ in Inspector
    private Dictionary<string, ComputeBuffer> weightBuffers = new Dictionary<string, ComputeBuffer>();
    private float p_mean;
    private float p_std;
    private int d_model, num_heads, num_layers, input_dim;
    
    // Helper array to avoid allocating every frame for GPU reduction
    private float[] reductionResult = new float[1];
    
    // Model architecture constants
    private const int WINDOW_SIZE = 256;  // Window size for attention (down from 512)
    private const int NUM_HEADS = 4;     // Number of attention heads
    
    // add previous active node count, but that can be a cpu variable 
    
    // Number of nodes, active nodes, and unique active nodes
    private int numNodes;
    
    // Public accessors for external scripts (e.g., FluidMeshRenderer)
    public ComputeBuffer NodesBuffer => nodesBuffer;
    public ComputeBuffer PhiBuffer => phiBuffer;
    public int NumNodes => numNodes;
    private int numUniqueNodes; // rename to numUniqueNodes
    private int layer;
    public float gravity = 100.0f;
    public float frameRate = 30.0f;
    public int minLayer = 4;
    public int maxLayer = 10;
    public PreconditionerType preconditioner = PreconditionerType.Neural;

    private bool hasShownWaitMessage = false;
    private int frameNumber = 0;
    private double cumulativeFrameTimeMs = 0.0;
    private double cumulativeCgIterations = 0.0;
    private int cgSolveFrameCount = 0;
    private float averageCgIterations = 0.0f;
    private int lastCgIterations = 0;
    
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
    
    // Material to render particles with thickness shader (assign a shader like Custom/ParticleThickness)
    public Material particleThicknessMaterial;
    
    // Material to generate normals from depth (assign a shader like Fluid/NormalsFromDepth)
    public Material normalMaterial; // Assign "Fluid/NormalsFromDepth" in Inspector

    public Material compositeMaterial; // Assign "Fluid/FluidRender" here
    
    
    // Debug display material for visualizing textures
    public Material debugDisplayMaterial; // Assign "Custom/DebugTextureDisplay" in Inspector

    public Color fluidColor = new Color(0.2f, 0.6f, 1.0f); // Input for extinction
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

    [Range(0, 100)] public int depthBlurRadius = 11;
    [Range(0.0001f, 10.0f)] public float depthBlurThreshold = 2.0f;
    [Range(0, 100)] public int thicknessBlurRadius = 22;
    [Range(0.0001f, 1.0f)] public float particleRadius = 0.066f; // Radius for particle points (world space)
    [Range(0.0001f, 1.0f)] public float depthRadius = 0.082f; // Radius for depth quads (world space)
    [Range(0.0001f, 1.0f)] public float thicknessRadius = 0.776f; // Radius for particle thickness quads (world space)
    [Range(0, 20)] public float absorptionStrength = 4.1f;
    [Range(0.0f, 1.0f)] public float depthOfFieldStrength = 0.41f;
    [Range(0, 2)] public float refractionScale = 1.0f;
    
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
    
    void Start()
    {
        InitializeParticleSystem();
        // InitializeInitialParticles();
        
        // Load neural preconditioner model metadata
        LoadModelMetadata();
        
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

    private void CalculateDensityGradients()
    {
        if (nodesShader == null)
        {
            Debug.LogError("Nodes compute shader is not assigned.");
            return;
        }

        calculateDensityGradientKernel = nodesShader.FindKernel("CalculateDensityGradient");
        nodesShader.SetBuffer(calculateDensityGradientKernel, "nodesBuffer", nodesBuffer);
        nodesShader.SetBuffer(calculateDensityGradientKernel, "neighborsBuffer", neighborsBuffer);
        nodesShader.SetBuffer(calculateDensityGradientKernel, "diffusionGradientBuffer", diffusionGradientBuffer);
        nodesShader.SetInt("numNodes", numNodes);
        
        int groups = Mathf.CeilToInt(numNodes / 512.0f);
        nodesShader.Dispatch(calculateDensityGradientKernel, groups, 1, 1);
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
        particlesShader.SetBuffer(updateParticlesKernel, "mortonCodesBuffer", mortonCodesBuffer);
        particlesShader.SetBuffer(updateParticlesKernel, "nodesBufferOld", nodesBufferOld);
        particlesShader.SetBuffer(updateParticlesKernel, "particlesBuffer", particlesBuffer);
        particlesShader.SetBuffer(updateParticlesKernel, "diffusionGradientBuffer", diffusionGradientBuffer);
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
        divergenceBuffer?.Release();
        residualBuffer?.Release();
        pBuffer?.Release();
        ApBuffer?.Release();
        pressureBuffer?.Release();
        phiBuffer?.Release();
        phiBuffer_Read?.Release();
        dirtyFlagBuffer?.Release();
        mortonCodesBuffer?.Release();
        ReleasePreconditionerBuffers();
        foreach(var b in weightBuffers.Values) b?.Release();
        weightBuffers.Clear();
        
        // Clean up render textures
        if (fluidDepthTexture != null) fluidDepthTexture.Release();
        if (fluidNormalTexture != null) fluidNormalTexture.Release();
    }
}
