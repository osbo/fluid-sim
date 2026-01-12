using UnityEngine;
using UnityEngine.InputSystem;
using System;
using System.Linq;
using System.Runtime.InteropServices;
using UnityEngine.Rendering;
using System.IO;
using System.Collections.Generic;

public enum RenderingMode
{
    Particles,
    Depth,
    Thickness,
    BlurredDepth,
    Normal,
    Composite
}

public enum PreconditionerType
{
    None,
    Neural,
    Jacobi
}

public enum ThicknessSource
{
    Nodes,
    Particles
}

public class FluidSimulator : MonoBehaviour
{
    public BoxCollider simulationBounds;
    public BoxCollider fluidInitialBounds;
    
    public ComputeShader radixSortShader;
    public ComputeShader particlesShader;
    public ComputeShader nodesPrefixSumsShader;
    public ComputeShader nodesShader;
    public ComputeShader cgSolverShader;
    public ComputeShader preconditionerShader; // Assign Preconditioner.compute in Inspector
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
    public TextAsset modelWeightsAsset; // Assign model_weights.bytes from Assets/Scripts/ in Inspector
    private Dictionary<string, ComputeBuffer> weightBuffers = new Dictionary<string, ComputeBuffer>();
    private float p_mean;
    private float p_std;
    private int d_model, num_heads, num_layers, input_dim;
    
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
    public float gravity;
    public float frameRate;
    public int minLayer;
    public int maxLayer;
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

    // Training data recorder
    public TrainingDataRecorder recorder;
    
    [Header("Rendering")]

    // Rendering mode enum
    public RenderingMode renderingMode = RenderingMode.Particles;
    
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
    public Color skyHorizonColor = new Color(0.1f, 0.1f, 0.15f); // Dark color at horizon (looking down)
    public Color skyTopColor = new Color(0.4f, 0.6f, 0.9f); // Light color at top (looking up)
    [Range(-20.0f, 20.0f)] public float sunIntensity = 0.0f; // Exponent: actual intensity = exp(value)
    [Range(0.0f, 500.0f)] public float sunSharpness = 500.0f; // Sun highlight sharpness (power exponent)
    // Calculated depth values (updated each frame based on camera position)
    private float calculatedDepthDisplayScale = 0.0f;  // Exponent for depth scale
    private float calculatedDepthMinValue = 0.0f;     // Exponent for depth min
    private float calculatedMinDepth = 0.0f;          // Actual min depth value
    private float calculatedMaxDepth = 0.0f;          // Actual max depth value
    [Range(-20.0f, 20.0f)] public float thicknessScaleNodes = 0.0f; // Exponent: actual scale = exp(value) for nodes
    [Range(-20.0f, 20.0f)] public float thicknessScaleParticles = 0.0f; // Exponent: actual scale = exp(value) for particles

    [Range(0, 100)] public int blurRadius = 5;
    [Range(0.0001f, 10.0f)] public float blurDepthFalloff = 1.0f;
    [Range(0.0001f, 1.0f)] public float particleRadius = 0.01f; // Radius for particle points (world space)
    [Range(0.0001f, 1.0f)] public float depthRadius = 0.01f; // Radius for depth quads (world space)
    [Range(0.0001f, 1.0f)] public float thicknessRadius = 0.01f; // Radius for particle thickness quads (world space)
    [Range(0, 20)] public float absorptionStrength = 1.0f;
    [Range(0, 2)] public float refractionScale = 1.0f;
    
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
    

    void OnEnable()
    {
        RenderPipelineManager.endCameraRendering += OnEndCameraRendering;
        
        // Create command buffers for thickness and depth rendering (like working project)
        if (thicknessCmd == null)
        {
            thicknessCmd = new CommandBuffer();
            thicknessCmd.name = "Node Thickness Render";
        }
        
        if (depthCmd == null)
        {
            depthCmd = new CommandBuffer();
            depthCmd.name = "Particle Depth Render";
        }
    }

    void OnDisable()
    {
        RenderPipelineManager.endCameraRendering -= OnEndCameraRendering;
        
        // Release args buffers
        if (argsBuffer != null)
        {
            argsBuffer.Release();
            argsBuffer = null;
        }
        
        if (particleArgsBuffer != null)
        {
            particleArgsBuffer.Release();
            particleArgsBuffer = null;
        }
        
        // Release command buffers
        if (thicknessCmd != null)
        {
            thicknessCmd.Release();
            thicknessCmd = null;
        }
        
        if (depthCmd != null)
        {
            depthCmd.Release();
            depthCmd = null;
        }
        
        // Release render textures
        if (thicknessTexture != null)
        {
            thicknessTexture.Release();
            thicknessTexture = null;
        }
        
        if (rawDepthTexture != null)
        {
            rawDepthTexture.Release();
            rawDepthTexture = null;
        }
    }

    private void ResizeBuffer(ref ComputeBuffer buffer, int count, int stride)
    {
        if (buffer != null && buffer.count >= count) return; // Capacity is sufficient

        buffer?.Release();
        // Allocate next power of two to prevent frequent resizing
        int newSize = Mathf.NextPowerOfTwo(Mathf.Max(count, 512));
        buffer = new ComputeBuffer(newSize, stride);
    }

    private void OnEndCameraRendering(ScriptableRenderContext ctx, Camera cam)
    {
        // Start rendering timing
        renderSw.Restart();
        
        // --- 1. Render Thickness (Raw) ---
        // Needed for: Thickness, Composite
        if (renderingMode == RenderingMode.Thickness || renderingMode == RenderingMode.Composite)
        {
            RenderThickness(cam);
            
            // If Composite, we also need to BLUR the thickness
            if (renderingMode == RenderingMode.Composite)
            {
                 // Create smooth thickness texture if needed
                 if (fluidThicknessTexture == null || fluidThicknessTexture.width != cam.pixelWidth)
                 {
                     if (fluidThicknessTexture != null) fluidThicknessTexture.Release();
                     RenderTextureDescriptor desc = new RenderTextureDescriptor(cam.pixelWidth, cam.pixelHeight, RenderTextureFormat.RFloat, 0);
                     desc.enableRandomWrite = true;
                     fluidThicknessTexture = new RenderTexture(desc);
                     fluidThicknessTexture.Create();
                 }
                 
                 // Reuse your existing blur function on the thickness texture
                 // Note: We use the thickness texture as both source and depth for the blur weights 
                 // (Self-guided blur preserves thickness edges)
                 RunBilateralBlur(cam, thicknessTexture, fluidThicknessTexture);
            }
        }
        
        // --- 2. Render Depth (Raw) ---
        // Needed for: Depth, BlurredDepth, Normal, Composite
        if (renderingMode != RenderingMode.Thickness) 
        {
            DrawParticles(cam, ctx); // Renders to rawDepthTexture or screen
        }
        
        // --- 3. Blur Depth & Gen Normals ---
        // Needed for: BlurredDepth, Normal, Composite
        if (renderingMode == RenderingMode.BlurredDepth || renderingMode == RenderingMode.Normal || renderingMode == RenderingMode.Composite)
        {
            // Initialize fluidDepthTexture if needed
            if (fluidDepthTexture == null || fluidDepthTexture.width != cam.pixelWidth || fluidDepthTexture.height != cam.pixelHeight)
            {
                if (fluidDepthTexture != null) fluidDepthTexture.Release();
                RenderTextureDescriptor desc = new RenderTextureDescriptor(cam.pixelWidth, cam.pixelHeight, RenderTextureFormat.RFloat, 0);
                desc.enableRandomWrite = true;
                fluidDepthTexture = new RenderTexture(desc);
                fluidDepthTexture.Create();
            }
            
            // 1. Blur Depth (Raw -> FluidDepth)
            RunBilateralBlur(cam, rawDepthTexture, fluidDepthTexture);
            
            // 2. Generate Normals (FluidDepth -> FluidNormal)
            RenderNormals(cam);
        }
        
        // --- 4. Final Display / Composite ---
        if (renderingMode == RenderingMode.Thickness)
        {
            float thicknessScale = thicknessSource == ThicknessSource.Nodes 
                ? thicknessScaleNodes 
                : thicknessScaleParticles;
            DebugBlit(thicknessTexture, thicknessScale, 0.0f, false);
        }
        else if (renderingMode == RenderingMode.Depth)
        {
            // Calculate depth parameters for debug visualization
            if (cam != null && simulationBounds != null)
            {
                Vector3 cameraPos = cam.transform.position;
                Bounds bounds = simulationBounds.bounds;
                float boundsDiagonal = bounds.size.magnitude;
                CalculateDepthParameters(cameraPos, bounds, boundsDiagonal);
            }
            DebugBlit(rawDepthTexture, calculatedDepthDisplayScale, calculatedDepthMinValue);
        }
        else if (renderingMode == RenderingMode.BlurredDepth)
        {
            // Calculate depth parameters for debug visualization
            if (cam != null && simulationBounds != null)
            {
                Vector3 cameraPos = cam.transform.position;
                Bounds bounds = simulationBounds.bounds;
                float boundsDiagonal = bounds.size.magnitude;
                CalculateDepthParameters(cameraPos, bounds, boundsDiagonal);
            }
            DebugBlit(fluidDepthTexture, calculatedDepthDisplayScale, calculatedDepthMinValue);
        }
        else if (renderingMode == RenderingMode.Normal)
        {
            Graphics.Blit(fluidNormalTexture, (RenderTexture)null);
        }
        else if (renderingMode == RenderingMode.Composite)
        {
            RenderComposite(cam);
        }
        
        // Stop rendering timing and store the result
        renderSw.Stop();
        lastRenderTimeMs = renderSw.Elapsed.TotalMilliseconds;
    }

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
            particlesCPU[indices[i]].layer = (uint)minLayer;
            particlesCPU[indices[i]].velocity = new Vector3(100.0f, 0.0f, 0.0f);
        }

        particlesBuffer.SetData(particlesCPU);
    }

    private void LoadModelMetadata()
    {
        if (modelWeightsAsset == null)
        {
            Debug.LogWarning("Model weights asset is not assigned. Please assign model_weights.bytes in the Inspector.");
            return;
        }

        byte[] fileBytes = modelWeightsAsset.bytes;
        
        if (fileBytes == null || fileBytes.Length < 24)
        {
            Debug.LogWarning("Model weights file is too small or invalid. Neural preconditioner will not work.");
            return;
        }

        try
        {
            using (BinaryReader reader = new BinaryReader(new MemoryStream(fileBytes)))
            {
                // 1. Header
                p_mean = reader.ReadSingle();
                p_std = reader.ReadSingle();
                d_model = reader.ReadInt32();
                num_heads = reader.ReadInt32();
                num_layers = reader.ReadInt32();
                input_dim = reader.ReadInt32();

                // 2. Helper to read float arrays into ComputeBuffers
                // Note: PyTorch exports float32. 
                ComputeBuffer ReadBuffer(int count)
                {
                    float[] data = new float[count];
                    for (int i = 0; i < count; i++) data[i] = reader.ReadSingle();
                    
                    ComputeBuffer buffer = new ComputeBuffer(count, sizeof(float));
                    buffer.SetData(data);
                    return buffer;
                }

                // 3. Load weights into dictionary (Order matches NeuralPreconditioner.py export_weights)
                // Clear old buffers if reloading
                foreach(var b in weightBuffers.Values) b?.Release();
                weightBuffers.Clear();

                // Stem weights
                weightBuffers["feature_proj.weight"] = ReadBuffer(input_dim * d_model); // [58, 128] -> transposed
                weightBuffers["feature_proj.bias"] = ReadBuffer(d_model);
                weightBuffers["layer_embed"] = ReadBuffer(12 * d_model); // [12, 128]
                weightBuffers["window_pos_embed"] = ReadBuffer(WINDOW_SIZE * d_model); // [1, WINDOW_SIZE, d_model] flattened

                // Transformer layers
                for (int i = 0; i < num_layers; i++)
                {
                    string prefix = $"layer_{i}.";
                    // Order matches NeuralPreconditioner.py export_weights function
                    weightBuffers[prefix + "in_proj_w"] = ReadBuffer(d_model * 3 * d_model); // [128, 384] transposed
                    weightBuffers[prefix + "in_proj_b"] = ReadBuffer(3 * d_model); // [384]
                    weightBuffers[prefix + "out_proj_w"] = ReadBuffer(d_model * d_model); // [128, 128] transposed
                    weightBuffers[prefix + "out_proj_b"] = ReadBuffer(d_model); // [128]
                    weightBuffers[prefix + "norm1_w"] = ReadBuffer(d_model);
                    weightBuffers[prefix + "norm1_b"] = ReadBuffer(d_model);
                    weightBuffers[prefix + "linear1_w"] = ReadBuffer(d_model * 2 * d_model); // [128, 256] transposed
                    weightBuffers[prefix + "linear1_b"] = ReadBuffer(2 * d_model); // [256]
                    weightBuffers[prefix + "linear2_w"] = ReadBuffer(2 * d_model * d_model); // [256, 128] transposed
                    weightBuffers[prefix + "linear2_b"] = ReadBuffer(d_model); // [128]
                    weightBuffers[prefix + "norm2_w"] = ReadBuffer(d_model);
                    weightBuffers[prefix + "norm2_b"] = ReadBuffer(d_model);
                }

                // Head weights
                weightBuffers["norm_out.weight"] = ReadBuffer(d_model);
                weightBuffers["norm_out.bias"] = ReadBuffer(d_model);
                weightBuffers["head.weight"] = ReadBuffer(d_model * 25); // [128, 25] transposed
                weightBuffers["head.bias"] = ReadBuffer(25);

                Debug.Log("Loaded all weights into dictionary.");
            }
            
            Debug.Log("Loaded " + num_layers + " Transformer layers.");
            
            Debug.Log($"Loaded Model Config: Mean={p_mean}, Std={p_std}, d_model={d_model}, Heads={num_heads}, Layers={num_layers}, InputDim={input_dim}");
        }
        catch (Exception e)
        {
            Debug.LogError($"Failed to load model metadata: {e.Message}");
        }
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
                 $"• Compute Level Set: {computeLevelSetSw.Elapsed.TotalMilliseconds:F2} ms\n" +
                 $"• Store Old Velocities: {storeOldVelocitiesSw.Elapsed.TotalMilliseconds:F2} ms\n" +
                 $"• Apply External Forces: {applyExternalForcesSw.Elapsed.TotalMilliseconds:F2} ms\n" +
                 $"• Solve Pressure: {solvePressureSw.Elapsed.TotalMilliseconds:F2} ms\n" +
                 $"• Update Particles: {updateParticlesSw.Elapsed.TotalMilliseconds:F2} ms\n" +
                 $"• Rendering: {lastRenderTimeMs:F2} ms\n" +
                 $"• CG Iterations: {lastCgIterations}\n" +
                 $"• Avg CG Iterations: {averageCgIterationsText}");
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
        
        if (cgSolverShader == null)
        {
            Debug.LogError("CGSolver compute shader is not assigned. Please assign `cgSolverShader` in the inspector.");
            return;
        }

        // --- Kernel Lookups ---
        int calculateDivergenceKernel = cgSolverShader.FindKernel("CalculateDivergence");
        int applyLaplacianAndDotKernel = cgSolverShader.FindKernel("ApplyLaplacianAndDot");
        int axpyKernel = cgSolverShader.FindKernel("Axpy");
        int dotProductKernel = cgSolverShader.FindKernel("DotProduct");
        // Preconditioner Kernels from CGSolver.compute
        int precomputeIndicesKernel = cgSolverShader.FindKernel("PrecomputeIndices");
        int applySparseGTKernel = cgSolverShader.FindKernel("ApplySparseGT"); 
        int applySparseGAndDotKernel = cgSolverShader.FindKernel("ApplySparseGAndDot");
        int clearBufferFloatKernel = cgSolverShader.FindKernel("ClearBufferFloat");
        int applyJacobiKernel = cgSolverShader.FindKernel("ApplyJacobi");
        int computeDiagonalKernel = cgSolverShader.FindKernel("ComputeDiagonal");
        
        if (calculateDivergenceKernel < 0 || applyLaplacianAndDotKernel < 0 || axpyKernel < 0 || dotProductKernel < 0)
        {
            Debug.LogError("One or more CG solver kernels not found. Check CGSolver.compute shader compilation.");
            return;
        }

        // --- Buffer Init (Grow-only) ---
        int requiredSize = Mathf.NextPowerOfTwo(Mathf.Max(numNodes, 512));
        
        if (divergenceBuffer == null || divergenceBuffer.count < requiredSize)
        {
            divergenceBuffer?.Release(); 
            divergenceBuffer = new ComputeBuffer(requiredSize, sizeof(float));
        }
        if (residualBuffer == null || residualBuffer.count < requiredSize)
        {
            residualBuffer?.Release(); 
            residualBuffer = new ComputeBuffer(requiredSize, sizeof(float));
        }
        if (pBuffer == null || pBuffer.count < requiredSize)
        {
            pBuffer?.Release(); 
            pBuffer = new ComputeBuffer(requiredSize, sizeof(float));
        }
        if (ApBuffer == null || ApBuffer.count < requiredSize)
        {
            ApBuffer?.Release(); 
            ApBuffer = new ComputeBuffer(requiredSize, sizeof(float));
        }
        if (pressureBuffer == null || pressureBuffer.count < requiredSize)
        {
            pressureBuffer?.Release(); 
            pressureBuffer = new ComputeBuffer(requiredSize, sizeof(float));
        }
        if (zVectorBuffer == null || zVectorBuffer.count < requiredSize)
        {
            zVectorBuffer?.Release(); 
            zVectorBuffer = new ComputeBuffer(requiredSize, sizeof(float));
        }
        if (scatterIndicesBuffer == null || scatterIndicesBuffer.count < requiredSize * 24)
        {
            scatterIndicesBuffer?.Release();
            scatterIndicesBuffer = new ComputeBuffer(requiredSize * 24, 4); // SoA: stride is 4 bytes (uint)
        }
        // Always allocate diagonalBuffer (needed for Neural input OR Jacobi)
        if (diagonalBuffer == null || diagonalBuffer.count < requiredSize)
        {
            diagonalBuffer?.Release();
            diagonalBuffer = new ComputeBuffer(requiredSize, sizeof(float));
        }

        // --- Step 1: Init (r = b) ---
        cgSolverShader.SetBuffer(calculateDivergenceKernel, "nodesBuffer", nodesBuffer);
        cgSolverShader.SetBuffer(calculateDivergenceKernel, "divergenceBuffer", divergenceBuffer);
        cgSolverShader.SetBuffer(calculateDivergenceKernel, "neighborsBuffer", neighborsBuffer);
        cgSolverShader.SetInt("numNodes", numNodes);
        Dispatch(calculateDivergenceKernel, numNodes);

        // Compute Diagonal EARLY (Before Neural Inference or Jacobi)
        // This is needed for both Neural (as input feature) and Jacobi (as preconditioner)
        if (computeDiagonalKernel >= 0)
        {
            cgSolverShader.SetBuffer(computeDiagonalKernel, "nodesBuffer", nodesBuffer);
            cgSolverShader.SetBuffer(computeDiagonalKernel, "neighborsBuffer", neighborsBuffer);
            cgSolverShader.SetBuffer(computeDiagonalKernel, "diagonalBuffer", diagonalBuffer);
            cgSolverShader.SetInt("numNodes", numNodes);

            // CHANGED: Use 256.0f divisor to match [numthreads(256, 1, 1)]
            int groups = Mathf.CeilToInt(numNodes / 256.0f); 
            cgSolverShader.Dispatch(computeDiagonalKernel, groups, 1, 1);
        }

        // Generate Matrix G (Neural Inference) or Use Diagonal (Jacobi)
        if (preconditioner == PreconditionerType.Neural && preconditionerShader != null)
        {
            RunNeuralPreconditioner(); // Fills matrixGBuffer (now has access to ready-made diagonalBuffer)
            
            // --- Precompute Optimization ---
            cgSolverShader.SetBuffer(precomputeIndicesKernel, "neighborsBuffer", neighborsBuffer);
            cgSolverShader.SetBuffer(precomputeIndicesKernel, "scatterIndicesBuffer", scatterIndicesBuffer);
            cgSolverShader.SetInt("numNodes", numNodes);
            Dispatch(precomputeIndicesKernel, numNodes);
        }
        // Note: Jacobi preconditioner uses the diagonalBuffer computed above, no additional dispatch needed

        // Init Pressure x=0
        float[] zeros = new float[numNodes];
        pressureBuffer.SetData(zeros);

        // r = b
        CopyBuffer(divergenceBuffer, residualBuffer);

        // --- Step 2: Preconditioned Initialization ---
        // z = M^-1 * r AND rho = r . z (Fused)
        float rho = ApplyPreconditionerAndDot(
            residualBuffer, zVectorBuffer, 
            applySparseGTKernel, applySparseGAndDotKernel, applySparseGAndDotKernel, 
            clearBufferFloatKernel, applyJacobiKernel
        );

        // p = z
        CopyBuffer(zVectorBuffer, pBuffer);

        float initialResidual = GpuDotProduct(residualBuffer, residualBuffer); // For convergence check only
        if (initialResidual < convergenceThreshold)
        {
            //totalSolveSw.Stop(); // This is no longer needed after removing totalSolveSw
            return;
        }

        // Reset CG loop timers
        laplacianSw.Reset();
        dotProductSw.Reset();
        updateVectorSw.Reset();

        // --- Step 3: PCG Loop ---
        int totalIterations = 0;
        
        for (int k = 0; k < maxCgIterations; k++)
        {
            // 1. FUSED: Ap = A * p AND p · Ap (in single kernel)
            laplacianSw.Start();
            cgSolverShader.SetBuffer(applyLaplacianAndDotKernel, "nodesBuffer", nodesBuffer);
            cgSolverShader.SetBuffer(applyLaplacianAndDotKernel, "neighborsBuffer", neighborsBuffer);
            cgSolverShader.SetBuffer(applyLaplacianAndDotKernel, "pBuffer", pBuffer);
            cgSolverShader.SetBuffer(applyLaplacianAndDotKernel, "ApBuffer", ApBuffer);
            cgSolverShader.SetBuffer(applyLaplacianAndDotKernel, "divergenceBuffer", divergenceBuffer); // For reduction output
            cgSolverShader.SetFloat("deltaTime", (1 / frameRate));
            cgSolverShader.SetInt("numNodes", numNodes);
            int groups = Mathf.CeilToInt(numNodes / 256.0f);
            cgSolverShader.Dispatch(applyLaplacianAndDotKernel, groups, 1, 1);
            laplacianSw.Stop();

            // 2. CPU Side Reduction (same as GpuDotProduct helper)
            dotProductSw.Start();
            float[] result = new float[groups];
            divergenceBuffer.GetData(result); // Only read back small buffer
            float p_dot_Ap = 0.0f;
            for (int i = 0; i < groups; i++) p_dot_Ap += result[i];
            dotProductSw.Stop();
            
            if (Mathf.Abs(p_dot_Ap) < 1e-12f)
            {
                Debug.LogError($"PCG Solver: Matrix is singular or ill-conditioned! p_dot_Ap = {p_dot_Ap:E6}");
                break;
            }
            
            if (p_dot_Ap <= 0.0f)
            {
                Debug.LogError($"PCG Solver: Matrix is not positive definite! p_dot_Ap = {p_dot_Ap:E6}");
                break;
            }
            
            float alpha = rho / p_dot_Ap;
            
            if (Mathf.Abs(alpha) > 1e10f)
            {
                Debug.LogError($"PCG Solver: Alpha is too large! alpha = {alpha:E6}, rho = {rho:E6}, p_dot_Ap = {p_dot_Ap:E6}");
                break;
            }

            // 3. x = x + alpha * p
            updateVectorSw.Start();
            UpdateVector(pressureBuffer, pBuffer, alpha);
            updateVectorSw.Stop();

            // 4. r = r - alpha * Ap
            updateVectorSw.Start();
            UpdateVector(residualBuffer, ApBuffer, -alpha);
            updateVectorSw.Stop();

            // Check Convergence (using true residual r.r)
            // Only check every 5 iterations to reduce CPU-GPU transfer
            if (k % 5 == 0)
            {
                dotProductSw.Start();
                float r_dot_r = GpuDotProduct(residualBuffer, residualBuffer);
                dotProductSw.Stop();
                
                if (r_dot_r > 10.0f * initialResidual)
                {
                    totalIterations = k + 1;
                    break;
                }
                
                if (r_dot_r < convergenceThreshold)
                {
                    totalIterations = k + 1;
                    break;
                }
            }

            // --- PRECONDITIONER STEP ---
            // 5. z_new = M^-1 * r_new AND rho_new = r_new . z_new (Fused)
            dotProductSw.Start();
            float rho_new = ApplyPreconditionerAndDot(
                residualBuffer, zVectorBuffer, 
                applySparseGTKernel, applySparseGAndDotKernel, applySparseGAndDotKernel, 
                clearBufferFloatKernel, applyJacobiKernel
            );
            dotProductSw.Stop();

            // 7. beta = rho_new / rho
            float beta = rho_new / rho;

            // 8. p = z_new + beta * p
            updateVectorSw.Start();
            UpdateVector(pBuffer, zVectorBuffer, 1.0f, beta);
            updateVectorSw.Stop();

            rho = rho_new;
            totalIterations = k + 1;
        }
        
        // Update running statistics
        cgSolveFrameCount++;
        cumulativeCgIterations += totalIterations;
        averageCgIterations = (float)(cumulativeCgIterations / Math.Max(1, cgSolveFrameCount));
        lastCgIterations = totalIterations;
        
        // Summary report

        // Save training data
        if (recorder != null && recorder.isRecording)
        {
            Vector3 simBoundsMin = simulationBounds != null ? simulationBounds.bounds.min : Vector3.zero;
            Vector3 simBoundsMax = simulationBounds != null ? simulationBounds.bounds.max : Vector3.zero;
            Vector3 fluidBoundsMin = fluidInitialBounds != null ? fluidInitialBounds.bounds.min : Vector3.zero;
            Vector3 fluidBoundsMax = fluidInitialBounds != null ? fluidInitialBounds.bounds.max : Vector3.zero;
            
            recorder.SaveFrame(nodesBuffer, neighborsBuffer, divergenceBuffer, pressureBuffer, numNodes,
                minLayer, maxLayer, gravity, numParticles, maxCgIterations, convergenceThreshold, frameRate,
                simBoundsMin, simBoundsMax, fluidBoundsMin, fluidBoundsMax);
        }

        // --- Step 4: Apply pressure to velocities ---
        ApplyPressureGradient();
        
        // Final timing summary
    }

    // --- Helper to Run the Preconditioner Pipeline ---
    private void ApplyPreconditioner(ComputeBuffer r, ComputeBuffer z_out, int kGT, int kG, int kClear, int kJacobi)
    {
        if (preconditioner == PreconditionerType.None)
        {
            // Identity fallback: z = r
            CopyBuffer(r, z_out);
            return;
        }
        else if (preconditioner == PreconditionerType.Jacobi)
        {
            // Jacobi preconditioning: z = D^-1 * r
            // where D is the diagonal of the Laplacian matrix A
            if (kJacobi >= 0 && diagonalBuffer != null)
            {
                cgSolverShader.SetBuffer(kJacobi, "xBuffer", r);
                cgSolverShader.SetBuffer(kJacobi, "yBuffer", z_out);
                cgSolverShader.SetBuffer(kJacobi, "diagonalBuffer", diagonalBuffer);
                cgSolverShader.SetInt("numNodes", numNodes);
                Dispatch(kJacobi, numNodes);
            }
            else
            {
                // Fallback to identity if kernel not found
                CopyBuffer(r, z_out);
            }
            return;
        }
        else if (preconditioner == PreconditionerType.Neural)
        {
            if (matrixGBuffer == null)
            {
                // Identity fallback: z = r
                CopyBuffer(r, z_out);
                return;
            }

            // Ensure zBuffer is allocated (used as intermediate 'u' in shader)
            int requiredSize = Mathf.NextPowerOfTwo(Mathf.Max(numNodes, 512));
            if (zBuffer == null || zBuffer.count < requiredSize)
            {
                zBuffer?.Release();
                zBuffer = new ComputeBuffer(requiredSize, 4);
            }

            // 1. Clear Intermediate 'zBuffer' (used as 'u' in Shader)
            cgSolverShader.SetBuffer(kClear, "zBuffer", zBuffer); 
            cgSolverShader.SetInt("numNodes", numNodes);
            Dispatch(kClear, numNodes);

            // 2. u = G^T * r  (Scatter)
            // CGSolver.compute: ApplySparseGT reads 'xBuffer' (r), writes 'zBuffer' (u)
            cgSolverShader.SetBuffer(kGT, "xBuffer", r);
            cgSolverShader.SetBuffer(kGT, "zBuffer", zBuffer); // Intermediate
            cgSolverShader.SetBuffer(kGT, "matrixGBuffer", matrixGBuffer);
            cgSolverShader.SetBuffer(kGT, "neighborsBuffer", neighborsBuffer);
            cgSolverShader.SetBuffer(kGT, "reverseNeighborsBuffer", reverseNeighborsBuffer);
            cgSolverShader.SetBuffer(kGT, "scatterIndicesBuffer", scatterIndicesBuffer); // Bind precomputed indices
            cgSolverShader.SetInt("numNodes", numNodes);
            Dispatch(kGT, numNodes);

            // 3. z = G * u + eps * r (Gather)
            // CGSolver.compute: fused ApplySparseGAndDot is used elsewhere; here we use the non-dot variant if desired.
            cgSolverShader.SetBuffer(kG, "zBuffer", zBuffer); // Intermediate input
            cgSolverShader.SetBuffer(kG, "xBuffer", r);       // For epsilon skip connection
            cgSolverShader.SetBuffer(kG, "yBuffer", z_out);   // Final Output
            cgSolverShader.SetBuffer(kG, "matrixGBuffer", matrixGBuffer);
            cgSolverShader.SetBuffer(kG, "neighborsBuffer", neighborsBuffer);
            cgSolverShader.SetInt("numNodes", numNodes);
            Dispatch(kG, numNodes);
        }
        else
        {
            // Unknown preconditioner type, fallback to identity
            CopyBuffer(r, z_out);
        }
    }

    // --- Fused Preconditioner + Dot Product ---
    // Performs M^-1 * r AND computes r · z in a single pass to reduce memory round-trips
    private float ApplyPreconditionerAndDot(ComputeBuffer r, ComputeBuffer z_out, int kGT, int kG, int kG_Dot, int kClear, int kJacobi)
    {
        float dotResult = 0.0f;

        if (preconditioner == PreconditionerType.Neural && matrixGBuffer != null)
        {
            // 1. Clear Intermediate
            cgSolverShader.SetBuffer(kClear, "zBuffer", zBuffer);
            cgSolverShader.SetInt("numNodes", numNodes);
            Dispatch(kClear, numNodes);

            // 2. Step 1: u = G^T * r (Standard Scatter)
            cgSolverShader.SetBuffer(kGT, "xBuffer", r);
            cgSolverShader.SetBuffer(kGT, "zBuffer", zBuffer);
            cgSolverShader.SetBuffer(kGT, "matrixGBuffer", matrixGBuffer);
            cgSolverShader.SetBuffer(kGT, "neighborsBuffer", neighborsBuffer);
            cgSolverShader.SetBuffer(kGT, "reverseNeighborsBuffer", reverseNeighborsBuffer);
            cgSolverShader.SetBuffer(kGT, "scatterIndicesBuffer", scatterIndicesBuffer);
            cgSolverShader.SetInt("numNodes", numNodes);
            Dispatch(kGT, numNodes);

            // 3. Step 2 + Dot: z = G * u + eps * r AND rho = dot(r, z)
            cgSolverShader.SetBuffer(kG_Dot, "zBuffer", zBuffer);       // u (Input)
            cgSolverShader.SetBuffer(kG_Dot, "xBuffer", r);             // r (Input for eps & dot)
            cgSolverShader.SetBuffer(kG_Dot, "yBuffer", z_out);         // z (Output)
            cgSolverShader.SetBuffer(kG_Dot, "matrixGBuffer", matrixGBuffer);
            cgSolverShader.SetBuffer(kG_Dot, "neighborsBuffer", neighborsBuffer);
            cgSolverShader.SetBuffer(kG_Dot, "divergenceBuffer", divergenceBuffer); // Scratch for reduction
            cgSolverShader.SetInt("numNodes", numNodes);
            
                        int groups = Mathf.CeilToInt(numNodes / 256.0f);
            
            
            
                        // --- [FIX START] ADD THIS LINE ---
            
                        cgSolverShader.Dispatch(kG_Dot, groups, 1, 1);
            
                        // --- [FIX END] ---
            
            
            
                        // 4. CPU Reduction
            
                        // Only read back the partial sums (tiny read)
            
                        float[] partials = new float[groups];
            divergenceBuffer.GetData(partials);
            for(int i=0; i<groups; i++) dotResult += partials[i];
        }
        else
        {
            // Fallback for Jacobi / None: Run standard logic + separate dot product
            ApplyPreconditioner(r, z_out, kGT, kG, kClear, kJacobi);
            dotResult = GpuDotProduct(r, z_out);
        }

        return dotResult;
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
        int threadGroups = Mathf.CeilToInt(numNodes / 256.0f);
        nodesShader.Dispatch(applyPressureGradientKernel, threadGroups, 1, 1);

        // SWAP the C# references
        ComputeBuffer swap = nodesBuffer;
        nodesBuffer = tempNodesBuffer;
        tempNodesBuffer = swap;
    }

    // Helper for dispatching kernels
    private void Dispatch(int kernel, int count) 
    {
        int threadGroups = Mathf.CeilToInt(count / 512.0f);
        cgSolverShader.Dispatch(kernel, threadGroups, 1, 1);
    }

    // Timing variables for preconditioner
    private System.Diagnostics.Stopwatch featSw = new System.Diagnostics.Stopwatch();
    private System.Diagnostics.Stopwatch qkvSw = new System.Diagnostics.Stopwatch();
    private System.Diagnostics.Stopwatch attnSw = new System.Diagnostics.Stopwatch();
    private System.Diagnostics.Stopwatch ffnSw = new System.Diagnostics.Stopwatch();
    private System.Diagnostics.Stopwatch headSw = new System.Diagnostics.Stopwatch();
    
    // Timing variables for CG loop
    private System.Diagnostics.Stopwatch laplacianSw = new System.Diagnostics.Stopwatch();
    private System.Diagnostics.Stopwatch dotProductSw = new System.Diagnostics.Stopwatch();
    private System.Diagnostics.Stopwatch updateVectorSw = new System.Diagnostics.Stopwatch();

    private void RunNeuralPreconditioner()
    {
        if (preconditionerShader == null)
        {
            return;
        }

        // --- NEURAL PRECONDITIONER INFERENCE ---
        // Align to window size for attention
        int paddedNodes = Mathf.CeilToInt(numNodes / (float)WINDOW_SIZE) * WINDOW_SIZE;
        int windowGroups = paddedNodes / WINDOW_SIZE;

        // 1. Resize Buffers (UPDATED SIZES)
        int requiredSize = Mathf.NextPowerOfTwo(Mathf.Max(numNodes, WINDOW_SIZE));
        int requiredPaddedSize = Mathf.CeilToInt(requiredSize / (float)WINDOW_SIZE) * WINDOW_SIZE;
        
        // MatrixG: [N, 25] -> SoA layout: [25 * N] with stride 4
        if (matrixGBuffer == null || matrixGBuffer.count < requiredSize * 25)
        {
            matrixGBuffer?.Release();
            matrixGBuffer = new ComputeBuffer(requiredSize * 25, 4); // SoA: stride is 4 bytes (float)
        }
        if (zBuffer == null || zBuffer.count < requiredSize)
        {
            zBuffer?.Release();
            zBuffer = new ComputeBuffer(requiredSize, 4);
        }
        
        // Internal buffers: Use d_model loaded from the .bytes file
        int current_d_model = this.d_model; 
        int stride = current_d_model * 4; 
        
        if (tokenBuffer == null || tokenBuffer.count < requiredPaddedSize * current_d_model)
        {
            tokenBuffer?.Release();
            tokenBuffer = new ComputeBuffer(requiredPaddedSize * current_d_model, 4);
        }
        if (tokenBufferOut == null || tokenBufferOut.count < requiredPaddedSize * current_d_model)
        {
            tokenBufferOut?.Release();
            tokenBufferOut = new ComputeBuffer(requiredPaddedSize * current_d_model, 4);
        }
        if (bufferQ == null || bufferQ.count < requiredPaddedSize * current_d_model)
        {
            bufferQ?.Release();
            bufferQ = new ComputeBuffer(requiredPaddedSize * current_d_model, 4);
        }
        if (bufferK == null || bufferK.count < requiredPaddedSize * current_d_model)
        {
            bufferK?.Release();
            bufferK = new ComputeBuffer(requiredPaddedSize * current_d_model, 4);
        }
        if (bufferV == null || bufferV.count < requiredPaddedSize * current_d_model)
        {
            bufferV?.Release();
            bufferV = new ComputeBuffer(requiredPaddedSize * current_d_model, 4);
        }
        if (bufferAttn == null || bufferAttn.count < requiredPaddedSize * current_d_model)
        {
            bufferAttn?.Release();
            bufferAttn = new ComputeBuffer(requiredPaddedSize * current_d_model, 4);
        }

        // Reset timers
        featSw.Reset();
        qkvSw.Reset();
        attnSw.Reset();
        ffnSw.Reset();
        headSw.Reset();

        // 1. Compute Features
        featSw.Start();
        int kFeat = preconditionerShader.FindKernel("ComputeFeatures");
        preconditionerShader.SetBuffer(kFeat, "nodesBuffer", nodesBuffer);
        preconditionerShader.SetBuffer(kFeat, "neighborsBuffer", neighborsBuffer);
        preconditionerShader.SetBuffer(kFeat, "tokenBuffer", tokenBuffer);
        preconditionerShader.SetBuffer(kFeat, "diagonalBuffer", diagonalBuffer); // Bind pre-computed diagonal buffer
        preconditionerShader.SetInt("numNodes", numNodes);
        preconditionerShader.SetInt("maxNodes", paddedNodes);
        preconditionerShader.SetInt("windowSize", WINDOW_SIZE); // Pass window size to shader
        
        // Set Shader Constants
        preconditionerShader.SetInt("d_model", current_d_model);
        preconditionerShader.SetInt("d_ffn", current_d_model * 2);
        preconditionerShader.SetInt("numHeads", NUM_HEADS); // Pass num heads to shader
        
        // Set Feature/Embed weights
        if (weightBuffers.ContainsKey("feature_proj.weight"))
            preconditionerShader.SetBuffer(kFeat, "weightsFeatureProj", weightBuffers["feature_proj.weight"]);
        if (weightBuffers.ContainsKey("feature_proj.bias"))
            preconditionerShader.SetBuffer(kFeat, "biasFeatureProj", weightBuffers["feature_proj.bias"]);
        if (weightBuffers.ContainsKey("layer_embed"))
            preconditionerShader.SetBuffer(kFeat, "weightsLayerEmbed", weightBuffers["layer_embed"]);
        if (weightBuffers.ContainsKey("window_pos_embed"))
            preconditionerShader.SetBuffer(kFeat, "weightsPosEmbed", weightBuffers["window_pos_embed"]);
        
        // Dispatch features using Window Groups
        preconditionerShader.Dispatch(kFeat, windowGroups, 1, 1);
        featSw.Stop();

        // Loop Layers (num_layers should be 2 from metadata)
        int kFused = preconditionerShader.FindKernel("FusedTransformerLayer");

        for (int i = 0; i < num_layers; i++)
        {
            string p = $"layer_{i}.";

            qkvSw.Start(); // reuse qkvSw to time the fused layer

            // Input / Output buffers (ping-pong)
            preconditionerShader.SetBuffer(kFused, "tokenBuffer", tokenBuffer);
            preconditionerShader.SetBuffer(kFused, "tokenBufferOut", tokenBufferOut);

            // Global uniforms
            preconditionerShader.SetInt("numNodes", numNodes);
            preconditionerShader.SetInt("maxNodes", paddedNodes);

            // Transformer weights for this layer
            if (weightBuffers.ContainsKey(p + "norm1_w"))
                preconditionerShader.SetBuffer(kFused, "w_norm1", weightBuffers[p + "norm1_w"]);
            if (weightBuffers.ContainsKey(p + "norm1_b"))
                preconditionerShader.SetBuffer(kFused, "b_norm1", weightBuffers[p + "norm1_b"]);

            if (weightBuffers.ContainsKey(p + "in_proj_w"))
                preconditionerShader.SetBuffer(kFused, "w_attn_in", weightBuffers[p + "in_proj_w"]);
            if (weightBuffers.ContainsKey(p + "in_proj_b"))
                preconditionerShader.SetBuffer(kFused, "b_attn_in", weightBuffers[p + "in_proj_b"]);

            if (weightBuffers.ContainsKey(p + "out_proj_w"))
                preconditionerShader.SetBuffer(kFused, "w_attn_out", weightBuffers[p + "out_proj_w"]);
            if (weightBuffers.ContainsKey(p + "out_proj_b"))
                preconditionerShader.SetBuffer(kFused, "b_attn_out", weightBuffers[p + "out_proj_b"]);

            if (weightBuffers.ContainsKey(p + "norm2_w"))
                preconditionerShader.SetBuffer(kFused, "w_norm2", weightBuffers[p + "norm2_w"]);
            if (weightBuffers.ContainsKey(p + "norm2_b"))
                preconditionerShader.SetBuffer(kFused, "b_norm2", weightBuffers[p + "norm2_b"]);

            if (weightBuffers.ContainsKey(p + "linear1_w"))
                preconditionerShader.SetBuffer(kFused, "w_ffn1", weightBuffers[p + "linear1_w"]);
            if (weightBuffers.ContainsKey(p + "linear1_b"))
                preconditionerShader.SetBuffer(kFused, "b_ffn1", weightBuffers[p + "linear1_b"]);
            if (weightBuffers.ContainsKey(p + "linear2_w"))
                preconditionerShader.SetBuffer(kFused, "w_ffn2", weightBuffers[p + "linear2_w"]);
            if (weightBuffers.ContainsKey(p + "linear2_b"))
                preconditionerShader.SetBuffer(kFused, "b_ffn2", weightBuffers[p + "linear2_b"]);

            // Dispatch fused kernel: 1 group per window, WINDOW_SIZE threads per group
            preconditionerShader.Dispatch(kFused, windowGroups, 1, 1);
            qkvSw.Stop();

            // Swap Buffers
            var temp = tokenBuffer;
            tokenBuffer = tokenBufferOut;
            tokenBufferOut = temp;
        }

        // 4. Head
        headSw.Start();
        int kHead = preconditionerShader.FindKernel("PredictHead");
        preconditionerShader.SetBuffer(kHead, "tokenBufferOut", tokenBuffer); // Note: swapped buffer is now input
        preconditionerShader.SetBuffer(kHead, "matrixGBuffer", matrixGBuffer);
        preconditionerShader.SetInt("numNodes", numNodes);
        preconditionerShader.SetInt("d_model", current_d_model);
        preconditionerShader.SetInt("d_ffn", current_d_model * 2);
        
        if (weightBuffers.ContainsKey("norm_out.weight"))
            preconditionerShader.SetBuffer(kHead, "w_normOut", weightBuffers["norm_out.weight"]);
        if (weightBuffers.ContainsKey("norm_out.bias"))
            preconditionerShader.SetBuffer(kHead, "b_normOut", weightBuffers["norm_out.bias"]);
        if (weightBuffers.ContainsKey("head.weight"))
            preconditionerShader.SetBuffer(kHead, "w_head", weightBuffers["head.weight"]);
        if (weightBuffers.ContainsKey("head.bias"))
            preconditionerShader.SetBuffer(kHead, "b_head", weightBuffers["head.bias"]);
        
        // Dispatch Head: 1 Group per Node (32 threads)
        // Handle large node counts by dispatching in batches (max 65535 thread groups per dimension)
        const int MAX_THREAD_GROUPS = 65535;
        int remainingNodes = paddedNodes;
        int offset = 0;
        while (remainingNodes > 0)
        {
            int batchSize = Mathf.Min(remainingNodes, MAX_THREAD_GROUPS);
            preconditionerShader.SetInt("headOffset", offset);
            preconditionerShader.Dispatch(kHead, batchSize, 1, 1);
            offset += batchSize;
            remainingNodes -= batchSize;
        }
        headSw.Stop();
    }

    private void ReleasePreconditionerBuffers()
    {
        bufferQ?.Release();
        bufferK?.Release();
        bufferV?.Release();
        bufferAttn?.Release();
        tokenBuffer?.Release();
        tokenBufferOut?.Release();
        matrixGBuffer?.Release();
        zBuffer?.Release();
        zVectorBuffer?.Release();
        scatterIndicesBuffer?.Release();
        diagonalBuffer?.Release();
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
        
        if (nodesShader != null)
        {
            extractMortonCodesKernel = nodesShader.FindKernel("ExtractMortonCodes");
        }
        
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

        // Use ResizeBuffer to prevent frequent reallocations
        ResizeBuffer(ref nodesBuffer, numNodes, sizeof(float) * 3 + sizeof(float) * 3 + sizeof(float) * 6 + sizeof(float) + sizeof(uint) * 3);
        ResizeBuffer(ref tempNodesBuffer, numNodes, sizeof(float) * 3 + sizeof(float) * 3 + sizeof(float) * 6 + sizeof(float) + sizeof(uint) * 3);
        ResizeBuffer(ref mortonCodesBuffer, numNodes, sizeof(uint));

        // Set buffer data to compute shader
        nodesShader.SetBuffer(createLeavesKernel, "particlesBuffer", particlesBuffer);
        nodesShader.SetBuffer(createLeavesKernel, "nodesBuffer", nodesBuffer);
        nodesShader.SetBuffer(createLeavesKernel, "uniqueIndices", uniqueIndices);
        nodesShader.SetBuffer(createLeavesKernel, "mortonCodesBuffer", mortonCodesBuffer);
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
        nodesShader.SetBuffer(processNodesKernel, "mortonCodesBuffer", mortonCodesBuffer);
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

        // Use ResizeBuffer to prevent frequent reallocations
        ResizeBuffer(ref neighborsBuffer, numNodes * 24, sizeof(uint));

        findNeighborsKernel = nodesShader.FindKernel("findNeighbors");
        nodesShader.SetBuffer(findNeighborsKernel, "nodesBuffer", nodesBuffer);
        nodesShader.SetBuffer(findNeighborsKernel, "tempNodesBuffer", tempNodesBuffer);
        nodesShader.SetBuffer(findNeighborsKernel, "neighborsBuffer", neighborsBuffer);
        nodesShader.SetBuffer(findNeighborsKernel, "mortonCodesBuffer", mortonCodesBuffer);
        nodesShader.SetInt("numNodes", numNodes);
        int threadGroups = Mathf.CeilToInt(numNodes / 512.0f);
        nodesShader.Dispatch(findNeighborsKernel, threadGroups, 1, 1);

        // Find reverse connections (run once per frame after findNeighbors)
        // Use ResizeBuffer to prevent frequent reallocations
        ResizeBuffer(ref reverseNeighborsBuffer, numNodes * 24, sizeof(uint));
        
        findReverseKernel = nodesShader.FindKernel("FindReverseConnections");
        nodesShader.SetBuffer(findReverseKernel, "reverseNeighborsBuffer", reverseNeighborsBuffer);
        nodesShader.SetBuffer(findReverseKernel, "neighborsBuffer", neighborsBuffer);
        nodesShader.SetInt("numNodes", numNodes);
        nodesShader.Dispatch(findReverseKernel, threadGroups, 1, 1);

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

    private void ApplyExternalForces()
    {
        if (nodesShader == null) return;

        applyExternalForcesKernel = nodesShader.FindKernel("ApplyExternalForces");
        if (applyExternalForcesKernel < 0)
        {
            Debug.LogError("ApplyExternalForces kernel not found in Nodes.compute shader.");
            return;
        }

        nodesShader.SetBuffer(applyExternalForcesKernel, "nodesBuffer", nodesBuffer);
        nodesShader.SetInt("numNodes", numNodes);
        nodesShader.SetFloat("gravity", gravity);
        nodesShader.SetFloat("deltaTime", (1 / frameRate));
        int threadGroups = Mathf.CeilToInt(numNodes / 512.0f);
        nodesShader.Dispatch(applyExternalForcesKernel, threadGroups, 1, 1);
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

        // // uint[] neighborsCPU = new uint[numNodes * 24];
        // // neighborsBuffer.GetData(neighborsCPU);

        // for (int i = 0; i < numNodes; i++) {
        //     Node node = nodesCPU[i];
        //     Gizmos.color = layerColors[(int)node.layer];
        //     Gizmos.DrawWireCube(DecodeMorton3D(node), Vector3.one * Mathf.Max(maxDetailCellSize * Mathf.Pow(2, node.layer), 0.01f));
        //     // Gizmos.color = new Color(0.0f, 0.5f, 1.0f, 1.0f);
        //     // bool isBoundary = false;
        //     // uint neighborBaseIndex = (uint)i * 24;
        //     // for (int d = 0; d < 6; d++) {
        //     //     uint baseFaceIndex = neighborBaseIndex + (uint)d * 4;
        //     //     uint n0_idx = neighborsCPU[baseFaceIndex];
        //     //     if (n0_idx == numNodes + 1) {
        //     //         isBoundary = true;
        //     //         break;
        //     //     }
        //     // }
        //     // if (node.layer == minLayer || isBoundary) {
        //     //     Gizmos.DrawCube(DecodeMorton3D(node), Vector3.one * Mathf.Max(maxDetailCellSize * Mathf.Pow(2, node.layer), 0.01f));
        //     // }
        // }
    }

    private void DrawParticles(Camera cam, ScriptableRenderContext ctx = default)
    {
        if (particlesBuffer == null || numParticles <= 0) return;
        if (cam == null) return;

        Material currentMaterial = null;
        bool useBlur = false;

        switch (renderingMode)
        {
            case RenderingMode.Particles:
                currentMaterial = particlesMaterial;
                break;
            case RenderingMode.Depth:
                currentMaterial = particleDepthMaterial;
                break;
            // FIX: Handle Normal mode here so it generates the depth texture required!
            case RenderingMode.BlurredDepth:
            case RenderingMode.Normal: 
            case RenderingMode.Composite: // [FIX] Add this
                currentMaterial = particleDepthMaterial; // Use depth shader as base
                useBlur = true;
                break;
        }

        if (currentMaterial == null) return;

        // Create quad mesh and args buffer if they don't exist (needed for Depth mode)
        if (quadMesh == null)
        {
            CreateQuadMesh();
        }
        
        // Update particle args buffer with current particle count (needed for Depth mode)
        if (particleArgsBuffer == null || particleArgsBuffer.count != 5)
        {
            CreateParticleArgsBuffer();
        }
        else
        {
            // Update instance count in args buffer
            uint[] args = new uint[5];
            particleArgsBuffer.GetData(args);
            args[1] = (uint)numParticles; // Update instance count
            particleArgsBuffer.SetData(args);
        }

        // --- Common Material Setup ---
        currentMaterial.SetBuffer("_Particles", particlesBuffer);
        
        // Set radius for Particles mode or Depth modes
        if (renderingMode == RenderingMode.Particles)
        {
            currentMaterial.SetFloat("_Radius", particleRadius);
        }
        else if (renderingMode == RenderingMode.Depth || renderingMode == RenderingMode.BlurredDepth || renderingMode == RenderingMode.Normal || renderingMode == RenderingMode.Composite)
        {
            currentMaterial.SetFloat("_Radius", depthRadius);
        }
        currentMaterial.SetInt("_MinLayer", minLayer);
        currentMaterial.SetInt("_MaxLayer", maxLayer);
        currentMaterial.SetVector("_SimulationBoundsMin", simulationBounds.bounds.min);
        currentMaterial.SetVector("_SimulationBoundsMax", simulationBounds.bounds.max);
        
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
        
        // Calculate distancesx
        float distanceToNearest = Vector3.Distance(cameraPos, nearestPoint);
        float distanceToCenter = Vector3.Distance(cameraPos, boundsCenter);
        float boundsDiagonal = boundsSize.magnitude; // Diagonal length of the bounds
        
        // Calculate depth min/max for normalizing particles inside the simulation bounds to 0-1 range
        CalculateDepthParameters(cameraPos, bounds, boundsDiagonal);
        
        // Set depth scale and min value for Particles mode (calculated dynamically)
        if (renderingMode == RenderingMode.Particles)
        {
            // Get the calculated values (convert from exponents to actual values)
            float calculatedMinValue = Mathf.Exp(calculatedDepthMinValue);
            float calculatedScale = Mathf.Exp(calculatedDepthDisplayScale);
            currentMaterial.SetFloat("_Scale", calculatedScale);
            currentMaterial.SetFloat("_MinValue", calculatedMinValue);
        }
        
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
        
        currentMaterial.SetFloat("_DepthFadeStart", fadeStart);
        currentMaterial.SetFloat("_DepthFadeEnd", fadeEnd);
        
        // Calculate depth min/max for depth visualization (for depth and blurred depth materials)
        // Reuse the already calculated minDepth and maxDepth values
        if (renderingMode == RenderingMode.Depth || renderingMode == RenderingMode.BlurredDepth || renderingMode == RenderingMode.Normal || renderingMode == RenderingMode.Composite)
        {
            currentMaterial.SetFloat("_DepthMin", calculatedMinDepth);
            currentMaterial.SetFloat("_DepthMax", calculatedMaxDepth);
        }

        // For Composite mode, we render to rawDepthTexture directly (blur happens in OnEndCameraRendering)
        // For BlurredDepth mode, we use RenderBlurredDepth which handles its own blur
        if (useBlur && blurCompute != null && renderingMode != RenderingMode.Composite)
        {
            RenderBlurredDepth(cam, currentMaterial);
        }
        else
        {
            // [FIX] Update condition to include all modes that need the depth texture
            if (renderingMode == RenderingMode.Depth || 
                renderingMode == RenderingMode.BlurredDepth || 
                renderingMode == RenderingMode.Normal || 
                renderingMode == RenderingMode.Composite)
            {
                // Ensure raw depth texture exists and matches screen size
                if (rawDepthTexture == null || rawDepthTexture.width != cam.pixelWidth || rawDepthTexture.height != cam.pixelHeight)
                {
                    if (rawDepthTexture != null)
                    {
                        rawDepthTexture.Release();
                    }
                    rawDepthTexture = new RenderTexture(cam.pixelWidth, cam.pixelHeight, 24, RenderTextureFormat.RFloat);
                    rawDepthTexture.enableRandomWrite = true;
                    rawDepthTexture.Create();
                }
                
                depthCmd.Clear();
                depthCmd.SetRenderTarget(rawDepthTexture);
                // Clear to 'Far' (10,000)
                depthCmd.ClearRenderTarget(true, true, new Color(10000.0f, 10000.0f, 10000.0f, 1.0f));
                depthCmd.DrawMeshInstancedIndirect(quadMesh, 0, currentMaterial, 0, particleArgsBuffer);
                Graphics.ExecuteCommandBuffer(depthCmd);
                
                // Only debug blit if we are strictly in Depth mode
                if (renderingMode == RenderingMode.Depth)
                {
                    DebugBlit(rawDepthTexture, calculatedDepthDisplayScale, calculatedDepthMinValue);
                }
            }
            else
            {
                // For Particles mode, use instanced rendering with quad mesh
                if (particlesBuffer == null || numParticles <= 0) return;
                
                // Update particle args buffer with current particle count
                if (particleArgsBuffer == null || particleArgsBuffer.count != 5)
                {
                    CreateParticleArgsBuffer();
                }
                else
                {
                    // Update instance count in args buffer
                    uint[] args = new uint[5];
                    particleArgsBuffer.GetData(args);
                    args[1] = (uint)numParticles; // Update instance count
                    particleArgsBuffer.SetData(args);
                }
                
                // Use CommandBuffer to ensure correct RenderTarget and avoid strict bounds culling
                depthCmd.Clear();
                
                // Target the Camera's active buffer (Screen)
                depthCmd.SetRenderTarget(BuiltinRenderTextureType.CameraTarget);
                
                // Draw mesh without strict bounds checks
                depthCmd.DrawMeshInstancedIndirect(quadMesh, 0, currentMaterial, 0, particleArgsBuffer);
                
                Graphics.ExecuteCommandBuffer(depthCmd);
            }
        }
    }

    private void RenderThickness(Camera cam)
    {
        if (cam == null) return;

        // Create quad mesh if it doesn't exist
        if (quadMesh == null)
        {
            CreateQuadMesh();
        }

        // Ensure thickness texture exists and matches screen size
        if (thicknessTexture == null || thicknessTexture.width != cam.pixelWidth || thicknessTexture.height != cam.pixelHeight)
        {
            if (thicknessTexture != null)
            {
                thicknessTexture.Release();
            }
            thicknessTexture = new RenderTexture(cam.pixelWidth, cam.pixelHeight, 0, RenderTextureFormat.RFloat);
            thicknessTexture.enableRandomWrite = true;
            thicknessTexture.Create();
        }

        // Use CommandBuffer like working project (this may fix Metal binding issues)
        thicknessCmd.Clear();
        
        // Set render target to thickness texture and clear to black (0 thickness)
        thicknessCmd.SetRenderTarget(thicknessTexture);
        thicknessCmd.ClearRenderTarget(true, true, Color.black);

        // Switch between nodes and particles based on thicknessSource enum
        if (thicknessSource == ThicknessSource.Nodes)
        {
            // Render using nodes
            if (nodesBuffer == null || numNodes <= 0) return;
            if (nodeThicknessMaterial == null) return;

            // Update args buffer with current node count
            if (argsBuffer == null || argsBuffer.count != 5)
            {
                CreateArgsBuffer();
            }
            else
            {
                // Update instance count in args buffer
                uint[] args = new uint[5];
                argsBuffer.GetData(args);
                args[1] = (uint)numNodes; // Update instance count
                argsBuffer.SetData(args);
            }

            // Set up material properties for NodeThickness shader
            nodeThicknessMaterial.SetBuffer("_Nodes", nodesBuffer);
            nodeThicknessMaterial.SetVector("_SimulationBoundsMin", simulationBounds.bounds.min);
            nodeThicknessMaterial.SetVector("_SimulationBoundsMax", simulationBounds.bounds.max);
            nodeThicknessMaterial.SetFloat("_PointSize", maxDetailCellSize); // Use maxDetailCellSize as base size
            nodeThicknessMaterial.SetColor("_Color", new Color(0.0f, 0.5f, 1.0f, 1.0f)); // Blue color

            // Draw nodes to thickness texture
            thicknessCmd.DrawMeshInstancedIndirect(
                quadMesh,
                0,
                nodeThicknessMaterial,
                0,  // shader pass index
                argsBuffer
            );
        }
        else // ThicknessSource.Particles
        {
            // Render using particles
            if (particlesBuffer == null || numParticles <= 0) return;
            if (particleThicknessMaterial == null) return;

            // Update particle args buffer with current particle count
            if (particleArgsBuffer == null || particleArgsBuffer.count != 5)
            {
                CreateParticleArgsBuffer();
            }
            else
            {
                // Update instance count in args buffer
                uint[] args = new uint[5];
                particleArgsBuffer.GetData(args);
                args[1] = (uint)numParticles; // Update instance count
                particleArgsBuffer.SetData(args);
            }

            // Set up material properties for ParticleThickness shader
            particleThicknessMaterial.SetBuffer("_Particles", particlesBuffer);
            particleThicknessMaterial.SetVector("_SimulationBoundsMin", simulationBounds.bounds.min);
            particleThicknessMaterial.SetVector("_SimulationBoundsMax", simulationBounds.bounds.max);
            particleThicknessMaterial.SetFloat("_Radius", thicknessRadius);

            // Draw particles to thickness texture
            thicknessCmd.DrawMeshInstancedIndirect(
                quadMesh,
                0,
                particleThicknessMaterial,
                0,  // shader pass index
                particleArgsBuffer
            );
        }
        
        // Execute command buffer
        Graphics.ExecuteCommandBuffer(thicknessCmd);

        // Visualization is handled in OnEndCameraRendering
    }
    
    private void RenderNormals(Camera cam)
    {
        int width = cam.pixelWidth;
        int height = cam.pixelHeight;

        // 1. Ensure Texture Exists (Same as before)
        if (fluidNormalTexture == null || fluidNormalTexture.width != width || fluidNormalTexture.height != height)
        {
            if (fluidNormalTexture != null) fluidNormalTexture.Release();
            RenderTextureDescriptor desc = new RenderTextureDescriptor(width, height, RenderTextureFormat.ARGBHalf, 0);
            fluidNormalTexture = new RenderTexture(desc);
            fluidNormalTexture.Create();
        }

        // 2. Setup Material
        if (normalMaterial != null && fluidDepthTexture != null)
        {
            // Keep this: View -> World
            normalMaterial.SetMatrix("_CameraInvViewMatrix", cam.cameraToWorldMatrix);
            
            // FIX: Remove _CameraInvProjMatrix. 
            // Instead, pass the intrinsic camera parameters needed to unproject linear depth.
            // Unity's built-in projection params (zBufferParams, etc) handle part of this, 
            // but for manual linear depth, we just need FOV and Aspect.
            
            float fovY = Mathf.Deg2Rad * cam.fieldOfView;
            float aspect = cam.aspect;
            float tanHalfFovY = Mathf.Tan(fovY * 0.5f);
            
            // Pass these to the shader
            normalMaterial.SetVector("_CameraParams", new Vector4(aspect * tanHalfFovY, tanHalfFovY, 0, 0));
            normalMaterial.SetTexture("_MainTex", fluidDepthTexture);
            
            Graphics.Blit(fluidDepthTexture, fluidNormalTexture, normalMaterial);
        }
    }
    
    private void DebugBlit(RenderTexture source, float scaleExponent, float minValueExponent = float.MinValue, bool useExpForMin = true)
    {
        if (debugDisplayMaterial == null || source == null) return;

        // Restore the screen as the render target
        Graphics.SetRenderTarget(null);
        
        // Convert exponent to actual scale: exp(scaleExponent)
        float scale = Mathf.Exp(scaleExponent);
        debugDisplayMaterial.SetFloat("_Scale", scale);
        
        // Set min value
        float minValue;
        if (minValueExponent == float.MinValue)
        {
            minValue = 0.0f; // Default to 0 if not provided
        }
        else if (useExpForMin)
        {
            minValue = Mathf.Exp(minValueExponent); // Use exp for depth
        }
        else
        {
            minValue = minValueExponent; // Use directly for thickness (should be 0.0f)
        }
        debugDisplayMaterial.SetFloat("_MinValue", minValue);
        
        // Draw to screen
        Graphics.Blit(source, (RenderTexture)null, debugDisplayMaterial);
    }

    private void CalculateDepthParameters(Vector3 cameraPos, Bounds bounds, float boundsDiagonal)
    {
        // Calculate depth min/max for normalizing particles inside the simulation bounds to 0-1 range
        // Min depth: distance to nearest point on bounds (particles can't be closer)
        Vector3 nearestPoint = new Vector3(
            Mathf.Clamp(cameraPos.x, bounds.min.x, bounds.max.x),
            Mathf.Clamp(cameraPos.y, bounds.min.y, bounds.max.y),
            Mathf.Clamp(cameraPos.z, bounds.min.z, bounds.max.z)
        );
        float minDepth = Vector3.Distance(cameraPos, nearestPoint);
        
        // Max depth: find the farthest corner of the bounds (farthest point where particles can be)
        Vector3[] corners = new Vector3[]
        {
            bounds.min,
            new Vector3(bounds.max.x, bounds.min.y, bounds.min.z),
            new Vector3(bounds.min.x, bounds.max.y, bounds.min.z),
            new Vector3(bounds.max.x, bounds.max.y, bounds.min.z),
            new Vector3(bounds.min.x, bounds.min.y, bounds.max.z),
            new Vector3(bounds.max.x, bounds.min.y, bounds.max.z),
            new Vector3(bounds.min.x, bounds.max.y, bounds.max.z),
            bounds.max
        };
        
        float maxDepth = 0.0f;
        foreach (Vector3 corner in corners)
        {
            float distance = Vector3.Distance(cameraPos, corner);
            maxDepth = Mathf.Max(maxDepth, distance);
        }
        
        // Ensure max is greater than min
        if (maxDepth <= minDepth)
        {
            maxDepth = minDepth + boundsDiagonal * 0.5f;
        }
        
        // Store the actual depth values for use in depth visualization
        calculatedMinDepth = minDepth;
        calculatedMaxDepth = maxDepth;
        
        // Calculate depth scale and min value dynamically to map depth to 0-1 range
        // Formula: displayVal = (depth - minDepth) * (1.0 / (maxDepth - minDepth))
        // So: _MinValue = minDepth, _Scale = 1.0 / (maxDepth - minDepth)
        // Store as exponents: depthMinValue = ln(minDepth), depthDisplayScale = ln(1.0 / (maxDepth - minDepth))
        float calculatedMinValue = Mathf.Max(0.001f, minDepth); // Ensure > 0 for log
        float depthRange = maxDepth - minDepth;
        float calculatedScale = depthRange > 0.001f ? (1.0f / depthRange) : 1.0f; // Avoid division by zero
        
        // Store as exponents (for compatibility with shaders that use exp())
        calculatedDepthMinValue = Mathf.Log(calculatedMinValue);
        calculatedDepthDisplayScale = Mathf.Log(calculatedScale);
    }

    private void CreateQuadMesh()
    {
        // Create a simple quad mesh for billboard rendering
        Vector3[] vertices = new Vector3[4]
        {
            new Vector3(-0.5f, -0.5f, 0),
            new Vector3(0.5f, -0.5f, 0),
            new Vector3(-0.5f, 0.5f, 0),
            new Vector3(0.5f, 0.5f, 0)
        };
        
        int[] triangles = new int[6]
        {
            0, 2, 1,
            2, 3, 1
        };
        
        Vector2[] uv = new Vector2[4]
        {
            new Vector2(0, 0),
            new Vector2(1, 0),
            new Vector2(0, 1),
            new Vector2(1, 1)
        };
        
        quadMesh = new Mesh();
        quadMesh.vertices = vertices;
        quadMesh.triangles = triangles;
        quadMesh.uv = uv;
        quadMesh.RecalculateNormals();
    }
    
    private void CreateArgsBuffer()
    {
        if (quadMesh == null) CreateQuadMesh();
        
        const int stride = sizeof(uint);
        const int numArgs = 5;
        const int subMeshIndex = 0;
        
        uint[] args = new uint[numArgs];
        args[0] = (uint)quadMesh.GetIndexCount(subMeshIndex);
        args[1] = (uint)numNodes;
        args[2] = (uint)quadMesh.GetIndexStart(subMeshIndex);
        args[3] = (uint)quadMesh.GetBaseVertex(subMeshIndex);
        args[4] = 0; // start instance location
        
        if (argsBuffer != null)
        {
            argsBuffer.Release();
        }
        argsBuffer = new ComputeBuffer(numArgs, stride, ComputeBufferType.IndirectArguments);
        argsBuffer.SetData(args);
    }
    
    private void CreateParticleArgsBuffer()
    {
        if (quadMesh == null) CreateQuadMesh();
        
        const int stride = sizeof(uint);
        const int numArgs = 5;
        const int subMeshIndex = 0;
        
        uint[] args = new uint[numArgs];
        args[0] = (uint)quadMesh.GetIndexCount(subMeshIndex);
        args[1] = (uint)numParticles;
        args[2] = (uint)quadMesh.GetIndexStart(subMeshIndex);
        args[3] = (uint)quadMesh.GetBaseVertex(subMeshIndex);
        args[4] = 0; // start instance location
        
        if (particleArgsBuffer != null)
        {
            particleArgsBuffer.Release();
        }
        particleArgsBuffer = new ComputeBuffer(numArgs, stride, ComputeBufferType.IndirectArguments);
        particleArgsBuffer.SetData(args);
    }

    private void RenderBlurredDepth(Camera cam, Material mat)
    {
        int width = cam.pixelWidth;
        int height = cam.pixelHeight;

        // 1. Initialize persistent texture (fluidDepthTexture)
        // Check if null or size changed
        if (fluidDepthTexture == null || fluidDepthTexture.width != width || fluidDepthTexture.height != height)
        {
            if (fluidDepthTexture != null) fluidDepthTexture.Release();
            
            // Use descriptor to properly ensure UAV flag is set BEFORE creation
            RenderTextureDescriptor desc = new RenderTextureDescriptor(width, height, RenderTextureFormat.RFloat, 0);
            desc.enableRandomWrite = true; // REQUIRED for Compute Shader output
            
            fluidDepthTexture = new RenderTexture(desc);
            fluidDepthTexture.Create();
        }

        // 2. Create rawDepth (Input for blur)
        // IMPORTANT: enableRandomWrite MUST be FALSE here because this texture has a Depth Buffer (24).
        // It is only read by the Compute Shader (as _SourceTexture), so it doesn't need UAV.
        RenderTextureDescriptor rawDesc = new RenderTextureDescriptor(width, height, RenderTextureFormat.RFloat, 24);
        rawDesc.enableRandomWrite = false; 
        RenderTexture rawDepth = RenderTexture.GetTemporary(rawDesc);

        // 3. Create tempBlur (Intermediate Ping-Pong buffer)
        // This needs UAV because the Compute Shader writes to it. No Depth Buffer needed (0).
        RenderTextureDescriptor blurDesc = new RenderTextureDescriptor(width, height, RenderTextureFormat.RFloat, 0);
        blurDesc.enableRandomWrite = true; 
        RenderTexture tempBlur = RenderTexture.GetTemporary(blurDesc);
        
        // --- RENDERING ---
        
        // A. Render Particles to rawDepth
        depthCmd.Clear();
        depthCmd.SetRenderTarget(rawDepth);
        depthCmd.ClearRenderTarget(true, true, Color.black); // Clear to "far away"
        depthCmd.DrawMeshInstancedIndirect(quadMesh, 0, mat, 0, particleArgsBuffer);
        Graphics.ExecuteCommandBuffer(depthCmd);

        // B. Horizontal Blur (rawDepth -> tempBlur)
        int kernelHandleH = blurCompute.FindKernel("HorizontalBlur");
        blurCompute.SetTexture(kernelHandleH, "_SourceTexture", rawDepth);
        blurCompute.SetTexture(kernelHandleH, "_DestinationTexture", tempBlur);
        blurCompute.SetFloat("_BlurRadius", blurRadius);
        blurCompute.SetFloat("_BlurDepthFalloff", blurDepthFalloff);
        blurCompute.SetVector("_Resolution", new Vector2(width, height));
        
        int groupsX = Mathf.CeilToInt(width / 8.0f);
        int groupsY = Mathf.CeilToInt(height / 8.0f);
        blurCompute.Dispatch(kernelHandleH, groupsX, groupsY, 1);

        // C. Vertical Blur (tempBlur -> fluidDepthTexture)
        int kernelHandleV = blurCompute.FindKernel("VerticalBlur");
        blurCompute.SetTexture(kernelHandleV, "_SourceTexture", tempBlur);
        blurCompute.SetTexture(kernelHandleV, "_DestinationTexture", fluidDepthTexture);
        blurCompute.SetFloat("_BlurRadius", blurRadius);
        blurCompute.SetFloat("_BlurDepthFalloff", blurDepthFalloff);
        blurCompute.SetVector("_Resolution", new Vector2(width, height));
        
        blurCompute.Dispatch(kernelHandleV, groupsX, groupsY, 1);

        // D. Visualize
        if (renderingMode == RenderingMode.BlurredDepth)
        {
            // Calculate depth parameters for debug visualization
            if (cam != null && simulationBounds != null)
            {
                Vector3 cameraPos = cam.transform.position;
                Bounds bounds = simulationBounds.bounds;
                float boundsDiagonal = bounds.size.magnitude;
                CalculateDepthParameters(cameraPos, bounds, boundsDiagonal);
            }
            DebugBlit(fluidDepthTexture, calculatedDepthDisplayScale, calculatedDepthMinValue);
        }

        // E. Cleanup
        RenderTexture.ReleaseTemporary(rawDepth);
        RenderTexture.ReleaseTemporary(tempBlur);
    }

    // Extracted from your previous RenderBlurredDepth to allow reuse
    private void RunBilateralBlur(Camera cam, RenderTexture source, RenderTexture dest)
    {
        if (blurCompute == null || source == null || dest == null) return;
        
        int width = cam.pixelWidth;
        int height = cam.pixelHeight;
        
        // Temp buffer
        RenderTextureDescriptor blurDesc = new RenderTextureDescriptor(width, height, RenderTextureFormat.RFloat, 0);
        blurDesc.enableRandomWrite = true; 
        RenderTexture tempBlur = RenderTexture.GetTemporary(blurDesc);

        // H Blur
        int kH = blurCompute.FindKernel("HorizontalBlur");
        blurCompute.SetTexture(kH, "_SourceTexture", source);
        blurCompute.SetTexture(kH, "_DestinationTexture", tempBlur);
        blurCompute.SetFloat("_BlurRadius", blurRadius);
        blurCompute.SetFloat("_BlurDepthFalloff", blurDepthFalloff);
        blurCompute.SetVector("_Resolution", new Vector2(width, height));
        blurCompute.Dispatch(kH, Mathf.CeilToInt(width/8.0f), Mathf.CeilToInt(height/8.0f), 1);
        
        // V Blur
        int kV = blurCompute.FindKernel("VerticalBlur");
        blurCompute.SetTexture(kV, "_SourceTexture", tempBlur);
        blurCompute.SetTexture(kV, "_DestinationTexture", dest);
        blurCompute.SetFloat("_BlurRadius", blurRadius);
        blurCompute.SetFloat("_BlurDepthFalloff", blurDepthFalloff);
        blurCompute.SetVector("_Resolution", new Vector2(width, height));
        blurCompute.Dispatch(kV, Mathf.CeilToInt(width/8.0f), Mathf.CeilToInt(height/8.0f), 1);
        
        RenderTexture.ReleaseTemporary(tempBlur);
    }

    private void RenderComposite(Camera cam)
    {
        if (compositeMaterial == null) return;

        // 1. Bind Textures
        compositeMaterial.SetTexture("_MainTex", null); // Or camera target if you grab it
        compositeMaterial.SetTexture("_NormalTex", fluidNormalTexture);
        compositeMaterial.SetTexture("_DepthTex", fluidDepthTexture); // Smooth
        compositeMaterial.SetTexture("_DepthRawTex", rawDepthTexture); // Raw
        compositeMaterial.SetTexture("_ThicknessTex", fluidThicknessTexture); // Smooth

        // 2. Bind Parameters
        Vector3 sunDir = mainLight != null ? -mainLight.transform.forward : Vector3.up;
        float sunInt = Mathf.Exp(sunIntensity);
        
        // Calculate extinction from color (Logarithm of inverse color)
        Vector3 extinction = new Vector3(
            -Mathf.Log(fluidColor.r), 
            -Mathf.Log(fluidColor.g), 
            -Mathf.Log(fluidColor.b)
        ) * absorptionStrength;

        compositeMaterial.SetVector("extinctionCoefficients", extinction);
        compositeMaterial.SetVector("dirToSun", sunDir);
        compositeMaterial.SetVector("boundsSize", simulationBounds.bounds.size);
        compositeMaterial.SetFloat("refractionMultiplier", refractionScale);
        compositeMaterial.SetFloat("sunIntensity", sunInt);
        compositeMaterial.SetFloat("sunSharpness", sunSharpness);
        compositeMaterial.SetVector("skyHorizonColor", skyHorizonColor);
        compositeMaterial.SetVector("skyTopColor", skyTopColor);

        // Calculate depth parameters for composite rendering
        Vector3 cameraPos = cam.transform.position;
        Bounds bounds = simulationBounds.bounds;
        float boundsDiagonal = bounds.size.magnitude;
        CalculateDepthParameters(cameraPos, bounds, boundsDiagonal);
        
        // Use the calculated values (already stored as exponents)
        compositeMaterial.SetFloat("depthDisplayScale", calculatedDepthDisplayScale);
        compositeMaterial.SetFloat("depthMinValue", calculatedDepthMinValue);
        float thicknessScale = thicknessSource == ThicknessSource.Nodes 
            ? thicknessScaleNodes 
            : thicknessScaleParticles;
        compositeMaterial.SetFloat("thicknessDisplayScale", thicknessScale);
        
        // Environment (Simple checkerboard settings)
        compositeMaterial.SetVector("floorPos", new Vector3(0, -5, 0)); // Adjust as needed
        compositeMaterial.SetVector("floorSize", new Vector3(75, 0.1f, 75));
        compositeMaterial.SetColor("tileCol1", new Color(0.8f, 0.8f, 0.8f));
        compositeMaterial.SetColor("tileCol2", new Color(0.4f, 0.4f, 0.4f));

        // 3. Blit to Screen
        Graphics.Blit(null, (RenderTexture)null, compositeMaterial);
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
        reverseNeighborsBuffer?.Release();
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