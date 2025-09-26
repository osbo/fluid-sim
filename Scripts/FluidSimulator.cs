using UnityEngine;

public class FluidSimulator : MonoBehaviour
{
    [SerializeField] private BoxCollider simulationBounds;
    [SerializeField] private GameObject fluidInitialBounds;
    
    public ComputeShader radixSortShader;
    public ComputeShader particlesShader;
    public ComputeShader nodesPrefixSumsShader;
    public ComputeShader nodesShader;
    public int numParticles;
    
    private RadixSort radixSort;
    private int initializeParticlesKernel;
    private int createLeavesKernel;
    private int processLeavesKernel;
    private int processNodesKernel;
    private int markUniqueParticlesKernel;
    private int markUniquesPrefixKernel;
	private int markActiveUniquesPrefixKernel;
    private int markActiveNodesKernel;
    private int scatterUniquesKernel;
    private int scatterActivesKernel;
    private int writeUniqueCountKernel;
    private int writeActiveCountKernel;
    private int radixPrefixSumKernelId;
    private int radixPrefixFixupKernelId;
    
    // GPU Buffers
    private ComputeBuffer particlesBuffer;
    private ComputeBuffer mortonCodesBuffer;
    private ComputeBuffer particleIndicesBuffer;
    private ComputeBuffer nodeFlagsBuffer; // Packed: 00(unique)(active)
    private ComputeBuffer sortedMortonCodes;
    private ComputeBuffer sortedParticleIndices;
    private ComputeBuffer indicators;
    private ComputeBuffer prefixSums;
    private ComputeBuffer aux;
    private ComputeBuffer aux2;
    private ComputeBuffer uniqueIndices;
    private ComputeBuffer activeIndices;
    private ComputeBuffer uniqueCount;
    private ComputeBuffer activeCount;
    private ComputeBuffer auxSmall;
    private ComputeBuffer nodesBuffer;
    private ComputeBuffer nodeMortonCodes;
    private ComputeBuffer nodeIndices;
    
    // Number of nodes, active nodes, and unique active nodes
    private int numNodes;
    private int numActiveNodes;
    private int numUniqueActiveNodes;
    private int layer;
    
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
        public faceVelocities velocities; // 6*4 = 24 bytes
        public uint layer;          // 4 bytes
        public uint mortonCode;     // 4 bytes
    }

    // Node struct (must match compute shader)
    private struct Node
    {
        public Vector3 position;    // 12 bytes
        public faceVelocities velocities; // 6*4 = 24 bytes
        public uint layer;          // 4 bytes
        public uint mortonCode;     // 4 bytes
    }

    void Start()
    {
        InitializeParticleSystem();

        // Debug: print numParticles
        Debug.Log($"Num Particles: {numParticles}");

        // Debug: check for NaN in particles
        Particle[] particlesCPU = new Particle[numParticles];
        particlesBuffer.GetData(particlesCPU);
        string str = $"NaN in particles:\n";
        for (int i = 0; i < numParticles; i++) {
            if (float.IsNaN(particlesCPU[i].position.x) || float.IsNaN(particlesCPU[i].position.y) || float.IsNaN(particlesCPU[i].position.z)) {
                str += $"Particle {i}: Position is NaN\n";
            }
        }
        Debug.Log(str);

        AdvectParticles();

        SortParticles();
        PrefixSumParticlesCodes();

        // Debug: print numNodes
        Debug.Log($"Num Unique Morton Codes (Layer 0: 30 bits): {numNodes}");

        CreateLeaves();

        // Debug: print nodes buffer
        Node[] nodesCPU = new Node[numNodes];
        nodesBuffer.GetData(nodesCPU);
        uint[] flagsCPU = new uint[numNodes];
        nodeFlagsBuffer.GetData(flagsCPU);
        int leafCount = 0;
        int internalCount = 0;
        int displayNum = 20;
        str = $"Layer 0: First {displayNum} leaves and {displayNum} internal nodes:\n";
        for (int i = 0; i < numNodes; i++)
        {
            if (nodesCPU[i].layer == 0) {
                if (leafCount < displayNum) {
                    str += $"Leaf {i}: Morton Code: {nodesCPU[i].mortonCode}, Layer: {nodesCPU[i].layer}, Position: {nodesCPU[i].position}, Active: {flagsCPU[i]}\n";
                    leafCount++;
                }
            } else {
                if (internalCount < displayNum) {
                    str += $"Internal {i}: Morton Code: {nodesCPU[i].mortonCode}, Layer: {nodesCPU[i].layer}, Position: {nodesCPU[i].position}, Active: {flagsCPU[i]}\n";
                    internalCount++;
                }
            }
            if (float.IsNaN(nodesCPU[i].position.x) || float.IsNaN(nodesCPU[i].position.y) || float.IsNaN(nodesCPU[i].position.z)) {
                str += $"Node {i}: Position is NaN\n";
            }
        }
        Debug.Log(str);

        PrefixSumLeavesCodes();

        // Debug: print numUniqueActiveNodes
        Debug.Log($"Num Unique Morton Codes (Layer 1: 27 bits): {numUniqueActiveNodes}");

        ProcessLeaves();

        // Debug: print nodes buffer
        nodesBuffer.GetData(nodesCPU);
        nodeFlagsBuffer.GetData(flagsCPU);
        leafCount = 0;
        internalCount = 0;
        displayNum = 20;
        str = $"Layer 1: First {displayNum} leaves and {displayNum} internal nodes:\n";
        for (int i = 0; i < numNodes; i++)
        {
            if (nodesCPU[i].layer == 0) {
                if (leafCount < displayNum) {
                    str += $"Leaf {i}: Morton Code: {nodesCPU[i].mortonCode}, Layer: {nodesCPU[i].layer}, Position: {nodesCPU[i].position}, Active: {flagsCPU[i]}\n";
                    leafCount++;
                }
            } else {
                if (internalCount < displayNum) {
                    str += $"Internal {i}: Morton Code: {nodesCPU[i].mortonCode}, Layer: {nodesCPU[i].layer}, Position: {nodesCPU[i].position}, Active: {flagsCPU[i]}\n";
                    internalCount++;
                }
            }
        }
        Debug.Log(str);

        for (layer = 2; layer <= 10; layer++)
        // for (layer = 2; layer <= 7; layer++)
        {
            prefixSumActiveNodes();

            // Debug: print numActiveNodes
            Debug.Log($"Num Active Nodes (Layer {layer}): {numActiveNodes}");

            // Debug: print activeIndices
            uint[] activeIndicesCPU = new uint[numActiveNodes];
            activeIndices.GetData(activeIndicesCPU);
            str = $"Layer {layer}: First 20 active indices:\n";
            for (int i = 0; i < 20; i++)
            {
                str += $"Active Index {i}: {activeIndicesCPU[i]}\n";
            }
            Debug.Log(str);

            prefixSumActiveNodesCodes();

            // Debug: print numUniqueActiveNodes
            Debug.Log($"Num Unique Morton Codes (Layer {layer}: {30 - layer * 3} bits): {numUniqueActiveNodes}");

            // Debug: print unique active indices
            uint[] uniqueIndicesCPU = new uint[numActiveNodes];
            uniqueIndices.GetData(uniqueIndicesCPU);
            str = $"Layer {layer}: First 20 unique active indices:\n";
            for (int i = 0; i < 20; i++)
            {
                str += $"Unique Active Index {i}: {uniqueIndicesCPU[i]} (Particle Index: {activeIndicesCPU[uniqueIndicesCPU[i]]})\n";
            }
            Debug.Log(str);

            ProcessNodes();

            // Debug: print nodes buffer
            nodesBuffer.GetData(nodesCPU);
            nodeFlagsBuffer.GetData(flagsCPU);
            leafCount = 0;
            internalCount = 0;
            displayNum = 20;
            int activeCount = 0;
            int inactiveCount = 0;
            str = $"Layer {layer}: First {displayNum} leaves and {displayNum} internal nodes:\n";
            for (int i = 0; i < numNodes; i++)
            {
                if (flagsCPU[i] == 1) {
                    if (nodesCPU[i].layer == 0) {
                        if (leafCount < displayNum) {
                            str += $"Leaf {i}: Morton Code: {nodesCPU[i].mortonCode}, Layer: {nodesCPU[i].layer}, Position: {nodesCPU[i].position}, Active: {flagsCPU[i]}\n";
                            leafCount++;
                        }
                    } else {
                        if (internalCount < displayNum) {
                            str += $"Internal {i}: Morton Code: {nodesCPU[i].mortonCode}, Layer: {nodesCPU[i].layer}, Position: {nodesCPU[i].position}, Active: {flagsCPU[i]}\n";
                                internalCount++;
                        }
                    }
                    activeCount++;
                } else {
                    inactiveCount++;
                }
            }
            str += $"Active nodes: {activeCount}\n";
            str += $"Inactive nodes: {inactiveCount}\n";
            Debug.Log(str);
        }
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
        particlesBuffer = new ComputeBuffer(numParticles, sizeof(float) * 3 + sizeof(float) * 6 + sizeof(uint) + sizeof(uint)); // 12 + 24 + 4 + 4 = 44 bytes
        mortonCodesBuffer = new ComputeBuffer(numParticles, sizeof(uint));
        particleIndicesBuffer = new ComputeBuffer(numParticles, sizeof(uint));
        // nodeFlagsBuffer created later in CreateNodes() with correct size
        sortedMortonCodes = new ComputeBuffer(numParticles, sizeof(uint));
        sortedParticleIndices = new ComputeBuffer(numParticles, sizeof(uint));

        // Set buffer data to compute shader
        particlesShader.SetBuffer(initializeParticlesKernel, "particlesBuffer", particlesBuffer);
        particlesShader.SetBuffer(initializeParticlesKernel, "mortonCodesBuffer", mortonCodesBuffer);
        particlesShader.SetBuffer(initializeParticlesKernel, "particleIndicesBuffer", particleIndicesBuffer);
        // nodeFlagsBuffer not set here - created later in CreateNodes()
        particlesShader.SetInt("count", numParticles);
        
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
        
        // Calculate optimal grid dimensions to fit numParticles as evenly as possible
        // Start with cubic root and adjust for aspect ratio
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
            if (gridDimensions.x <= gridDimensions.y && gridDimensions.x <= gridDimensions.z)
                gridDimensions.x++;
            else if (gridDimensions.y <= gridDimensions.z)
                gridDimensions.y++;
            else
                gridDimensions.z++;
        }
        
        // If we have too many cells, reduce dimensions
        while (gridDimensions.x * gridDimensions.y * gridDimensions.z > numParticles)
        {
            if (gridDimensions.x >= gridDimensions.y && gridDimensions.x >= gridDimensions.z)
                gridDimensions.x = Mathf.Max(1, gridDimensions.x - 1);
            else if (gridDimensions.y >= gridDimensions.z)
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
        
        // Set grid parameters to compute shader
        particlesShader.SetInts("gridDimensions", new int[] { (int)gridDimensions.x, (int)gridDimensions.y, (int)gridDimensions.z });
        particlesShader.SetVector("gridSpacing", actualGridSpacing);
        
        // Dispatch the kernel
        particlesShader.SetFloat("dispersionPower", 4.0f); // higher = stronger skew
        particlesShader.SetFloat("boundaryThreshold", 0.01f); // near-left threshold in normalized X
        int threadGroups = Mathf.CeilToInt(numParticles / 512.0f);
        particlesShader.Dispatch(initializeParticlesKernel, threadGroups, 1, 1);
    }

    private void AdvectParticles()
    {
        if (particlesShader == null)
        {
            Debug.LogError("Particles compute shader is not assigned. Please assign `particlesShader` in the inspector.");
            return;
        }

        int advectParticlesKernel = particlesShader.FindKernel("AdvectParticles");

        float deltaTime = (1/60.0f);
        float gravity = 9.81f;
        
        particlesShader.SetBuffer(advectParticlesKernel, "particlesBuffer", particlesBuffer);
        particlesShader.SetBuffer(advectParticlesKernel, "mortonCodesBuffer", mortonCodesBuffer);
        particlesShader.SetBuffer(advectParticlesKernel, "particleIndicesBuffer", particleIndicesBuffer);
        particlesShader.SetInt("numParticles", numParticles);
        particlesShader.SetFloat("deltaTime", deltaTime);
        particlesShader.SetFloat("gravity", gravity);
        particlesShader.SetVector("mortonNormalizationFactor", mortonNormalizationFactor);
        particlesShader.SetFloat("mortonMaxValue", mortonMaxValue);
        particlesShader.SetVector("simulationBoundsMin", simulationBoundsMin);
        particlesShader.SetVector("simulationBoundsMax", simulationBoundsMax);
        int threadGroups = Mathf.CeilToInt(numParticles / 512.0f);
        particlesShader.Dispatch(advectParticlesKernel, threadGroups, 1, 1);
    }
    
    private void SortParticles()
    {
        // Initialize radix sort
        radixSort = new RadixSort(radixSortShader, (uint)numParticles);
        
        // Sort the morton codes and their corresponding indices
        radixSort.Sort(mortonCodesBuffer, particleIndicesBuffer, sortedMortonCodes, sortedParticleIndices, (uint)numParticles);
    }

    private void PrefixSumParticlesCodes()
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

        // Allocate leaves buffers
        indicators = new ComputeBuffer(numParticles, sizeof(uint));
        prefixSums = new ComputeBuffer(numParticles, sizeof(uint));

        uint tgSize = 512u;
        uint numThreadgroups = (uint)((numParticles + (tgSize * 2) - 1) / (tgSize * 2));
        uint auxSize = (uint)System.Math.Max(1, (int)numThreadgroups);
        aux = new ComputeBuffer((int)auxSize, sizeof(uint));
        aux2 = new ComputeBuffer((int)auxSize, sizeof(uint));

        uniqueIndices = new ComputeBuffer(numParticles, sizeof(uint));
        activeIndices = new ComputeBuffer(numParticles, sizeof(uint));
        uniqueCount = new ComputeBuffer(1, sizeof(uint));
        activeCount = new ComputeBuffer(1, sizeof(uint));

        // mark uniques
        nodesPrefixSumsShader.SetBuffer(markUniqueParticlesKernel, "sortedMortonCodes", sortedMortonCodes);
        nodesPrefixSumsShader.SetBuffer(markUniqueParticlesKernel, "indicators", indicators);
        nodesPrefixSumsShader.SetInt("len", numParticles);
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
        nodesPrefixSumsShader.SetBuffer(scatterUniquesKernel, "sortedIndices", sortedParticleIndices);
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
        numActiveNodes = (int)numNodesCpu[0];
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

        // Create node buffers
        nodesBuffer = new ComputeBuffer(numNodes, sizeof(float) * 3 + sizeof(float) * 6 + sizeof(uint) + sizeof(uint)); // 12 + 24 + 4 + 4 = 44 bytes
        nodeMortonCodes = new ComputeBuffer(numNodes, sizeof(uint));
        nodeIndices = new ComputeBuffer(numNodes, sizeof(uint));
        nodeFlagsBuffer = new ComputeBuffer(numNodes, sizeof(uint)); // 4 bytes (packed flags)

        // Set buffer data to compute shader
        nodesShader.SetBuffer(createLeavesKernel, "particlesBuffer", particlesBuffer);
        nodesShader.SetBuffer(createLeavesKernel, "sortedParticleIndices", sortedParticleIndices);
        nodesShader.SetBuffer(createLeavesKernel, "uniqueIndices", uniqueIndices);
        nodesShader.SetBuffer(createLeavesKernel, "nodesBuffer", nodesBuffer);
        nodesShader.SetBuffer(createLeavesKernel, "nodeMortonCodes", nodeMortonCodes);
        nodesShader.SetBuffer(createLeavesKernel, "nodeIndices", nodeIndices);
        nodesShader.SetBuffer(createLeavesKernel, "nodeFlagsBuffer", nodeFlagsBuffer);
        nodesShader.SetInt("numNodes", numNodes);
        nodesShader.SetInt("numParticles", numParticles);

        // Dispatch the kernel
        int threadGroups = Mathf.CeilToInt(numNodes / 512.0f);
        nodesShader.Dispatch(createLeavesKernel, threadGroups, 1, 1);
    }

    private void PrefixSumLeavesCodes()
    {
        // dispatch numNodes threads, find unique-prefix leaves (all are active)
        if (nodesPrefixSumsShader == null)
        {
            Debug.LogError("NodesPrefixSums compute shader is not assigned. Please assign `nodesPrefixSumsShader` in the inspector.");
            return;
        }

        ClearUniqueBuffers();

        int prefixBits = 3;

		nodesPrefixSumsShader.SetBuffer(markUniquesPrefixKernel, "nodeMortonCodes", nodeMortonCodes);
		nodesPrefixSumsShader.SetBuffer(markUniquesPrefixKernel, "indicators", indicators);
		nodesPrefixSumsShader.SetInt("len", numNodes);
		nodesPrefixSumsShader.SetInt("prefixBits", prefixBits);
		int groupsLinear = (numNodes + 511) / 512;
		nodesPrefixSumsShader.Dispatch(markUniquesPrefixKernel, groupsLinear, 1, 1);

        uint tgSize = 512u;
        uint numThreadgroups = (uint)((numNodes + (tgSize * 2) - 1) / (tgSize * 2));
        uint auxSize = (uint)System.Math.Max(1, (int)numThreadgroups);

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
        numUniqueActiveNodes = (int)uniqueCountCpu[0];
    }

    private void ProcessLeaves()
    {
        // dispatch numUniqueActiveNodes threads
        if (nodesShader == null)
        {
            Debug.LogError("Nodes compute shader is not assigned. Please assign `nodesShader` in the inspector.");
            return;
        }

        if (numUniqueActiveNodes == 0) return;

        // Find kernel
        processLeavesKernel = nodesShader.FindKernel("ProcessLeaves");

        // Set buffers for the node processing kernel
        nodesShader.SetBuffer(processLeavesKernel, "particlesBuffer", particlesBuffer);
        nodesShader.SetBuffer(processLeavesKernel, "sortedMortonCodes", sortedMortonCodes);
        nodesShader.SetBuffer(processLeavesKernel, "sortedParticleIndices", sortedParticleIndices);
        nodesShader.SetBuffer(processLeavesKernel, "uniqueIndices", uniqueIndices);
        nodesShader.SetBuffer(processLeavesKernel, "uniqueCount", uniqueCount);
        nodesShader.SetBuffer(processLeavesKernel, "nodesBuffer", nodesBuffer);
        nodesShader.SetBuffer(processLeavesKernel, "nodeMortonCodes", nodeMortonCodes);
        nodesShader.SetBuffer(processLeavesKernel, "nodeIndices", nodeIndices);
        nodesShader.SetBuffer(processLeavesKernel, "nodeFlagsBuffer", nodeFlagsBuffer);
        nodesShader.SetInt("numUniqueNodes", numUniqueActiveNodes);
        nodesShader.SetInt("numActiveNodes", numActiveNodes);
        nodesShader.SetInt("numNodes", numNodes);
        nodesShader.SetInt("numParticles", numParticles);
        nodesShader.SetInt("layer", 1);

        // Dispatch one thread per unique node
        int threadGroups = Mathf.CeilToInt(numUniqueActiveNodes / 512.0f);
        nodesShader.Dispatch(processLeavesKernel, threadGroups, 1, 1);
    }

    private void prefixSumActiveNodes() // in for loop
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

        // Find kernels
        markActiveNodesKernel = nodesPrefixSumsShader.FindKernel("markActiveNodes");
        scatterActivesKernel = nodesPrefixSumsShader.FindKernel("scatterActives");
        writeActiveCountKernel = nodesPrefixSumsShader.FindKernel("writeActiveCount");

        if (markActiveNodesKernel < 0 || scatterActivesKernel < 0 || writeActiveCountKernel < 0)
        {
            Debug.LogError("One or more kernels not found in NodesPrefixSums.compute. Verify #pragma kernel names and shader assignment.");
            return;
        }

        // // Debug: Print input node flags before processing
        // uint[] nodeFlagsCPU = new uint[numNodes];
        // nodeFlagsBuffer.GetData(nodeFlagsCPU);
        // string debugStr = $"Layer {layer}: First 20 node flags (input):\n";
        // for (int i = 0; i < 20 && i < numNodes; i++)
        // {
        //     debugStr += $"Node {i}: Flag={nodeFlagsCPU[i]} (active={nodeFlagsCPU[i] & 1u})\n";
        // }
        // Debug.Log(debugStr);

        // Set buffers for the node processing kernel
        nodesPrefixSumsShader.SetBuffer(markActiveNodesKernel, "nodeFlagsBuffer", nodeFlagsBuffer);
        nodesPrefixSumsShader.SetBuffer(markActiveNodesKernel, "indicators", indicators);
        nodesPrefixSumsShader.SetInt("len", numNodes);
        int groupsLinear = Mathf.Max(1, (numNodes + 511) / 512);
        nodesPrefixSumsShader.Dispatch(markActiveNodesKernel, groupsLinear, 1, 1);
        
        // // Debug: print the indicators after markActiveNodes
        // uint[] indicatorsCPU = new uint[numNodes];
        // indicators.GetData(indicatorsCPU);
        // debugStr = $"Layer {layer}: First 20 indicators (after markActiveNodes):\n";
        // for (int i = 0; i < 20 && i < numNodes; i++)
        // {
        //     debugStr += $"Indicator {i}: {indicatorsCPU[i]}\n";
        // }
        // Debug.Log(debugStr);

        // // Count active indicators manually for verification
        // int manualActiveCount = 0;
        // for (int i = 0; i < numNodes; i++)
        // {
        //     if (indicatorsCPU[i] == 1) manualActiveCount++;
        // }
        // Debug.Log($"Layer {layer}: Manual count of active indicators: {manualActiveCount}");

        uint tgSize = 512u;
        uint numThreadgroups = (uint)((numNodes + (tgSize * 2) - 1) / (tgSize * 2));
        uint auxSize = (uint)System.Math.Max(1, (int)numThreadgroups);

        // Debug: Print aux buffer size and thread group info
        Debug.Log($"Layer {layer}: Prefix sum parameters - numNodes: {numNodes}, numThreadgroups: {numThreadgroups}, auxSize: {auxSize}");

		radixSortShader.SetBuffer(radixPrefixSumKernelId, "input", indicators);
		radixSortShader.SetBuffer(radixPrefixSumKernelId, "output", prefixSums);
		radixSortShader.SetBuffer(radixPrefixSumKernelId, "aux", aux);
		radixSortShader.SetInt("len", numNodes);
		radixSortShader.SetInt("zeroff", 1);
		radixSortShader.Dispatch(radixPrefixSumKernelId, (int)numThreadgroups, 1, 1);

        // // Debug: print prefix sums after first level scan
        // uint[] prefixSumsCPU = new uint[numNodes];
        // prefixSums.GetData(prefixSumsCPU);
        // debugStr = $"Layer {layer}: First 20 prefix sums (after first level scan):\n";
        // for (int i = 0; i < 20 && i < numNodes; i++)
        // {
        //     debugStr += $"PrefixSum {i}: {prefixSumsCPU[i]}\n";
        // }
        // Debug.Log(debugStr);

        if (numThreadgroups > 1)
        {
            // // Debug: print aux buffer before second level scan
            // uint[] auxCPU = new uint[auxSize];
            // aux.GetData(auxCPU);
            // debugStr = $"Layer {layer}: Aux buffer (before second level scan):\n";
            // for (int i = 0; i < auxSize && i < 20; i++)
            // {
            //     debugStr += $"Aux {i}: {auxCPU[i]}\n";
            // }
            // Debug.Log(debugStr);

            // Scan aux -> aux2
            if (auxSmall == null) auxSmall = new ComputeBuffer(1, sizeof(uint));
            uint auxThreadgroups = 1; // aux length is small
			radixSortShader.SetBuffer(radixPrefixSumKernelId, "input", aux);
			radixSortShader.SetBuffer(radixPrefixSumKernelId, "output", aux2);
			radixSortShader.SetBuffer(radixPrefixSumKernelId, "aux", auxSmall);
			radixSortShader.SetInt("len", (int)auxSize);
			radixSortShader.SetInt("zeroff", 1);
			radixSortShader.Dispatch(radixPrefixSumKernelId, (int)auxThreadgroups, 1, 1);

            // // Debug: print aux2 buffer after second level scan
            // uint[] aux2CPU = new uint[auxSize];
            // aux2.GetData(aux2CPU);
            // debugStr = $"Layer {layer}: Aux2 buffer (after second level scan):\n";
            // for (int i = 0; i < auxSize && i < 20; i++)
            // {
            //     debugStr += $"Aux2 {i}: {aux2CPU[i]}\n";
            // }
            // Debug.Log(debugStr);

            // Fixup: add scanned aux into prefixSums
			radixSortShader.SetBuffer(radixPrefixFixupKernelId, "input", prefixSums);
			radixSortShader.SetBuffer(radixPrefixFixupKernelId, "aux", aux2);
			radixSortShader.SetInt("len", numNodes);
			radixSortShader.Dispatch(radixPrefixFixupKernelId, (int)numThreadgroups, 1, 1);
        }

        // // Debug: print final prefix sums after fixup
        // prefixSums.GetData(prefixSumsCPU);
        // debugStr = $"Layer {layer}: First 20 prefix sums (after fixup):\n";
        // for (int i = 0; i < 20 && i < numNodes; i++)
        // {
        //     debugStr += $"PrefixSum {i}: {prefixSumsCPU[i]}\n";
        // }
        // Debug.Log(debugStr);

        // scatterActives active indices
        nodesPrefixSumsShader.SetBuffer(scatterActivesKernel, "indicators", indicators);
        nodesPrefixSumsShader.SetBuffer(scatterActivesKernel, "prefixSums", prefixSums);
        nodesPrefixSumsShader.SetBuffer(scatterActivesKernel, "activeIndices", activeIndices);
        nodesPrefixSumsShader.SetInt("len", numNodes);
        int scatterGroups = Mathf.Max(1, (numNodes + 511) / 512);
        nodesPrefixSumsShader.Dispatch(scatterActivesKernel, scatterGroups, 1, 1);

        // // Debug: print active indices after scatter
        // uint[] activeIndicesCPU = new uint[numNodes];
        // activeIndices.GetData(activeIndicesCPU);
        // debugStr = $"Layer {layer}: First 20 active indices (after scatterActives):\n";
        // for (int i = 0; i < 20 && i < numNodes; i++)
        // {
        //     debugStr += $"ActiveIndex {i}: {activeIndicesCPU[i]}\n";
        // }
        // Debug.Log(debugStr);

        // Write active count
        nodesPrefixSumsShader.SetBuffer(writeActiveCountKernel, "indicators", indicators);
        nodesPrefixSumsShader.SetBuffer(writeActiveCountKernel, "prefixSums", prefixSums);
        nodesPrefixSumsShader.SetBuffer(writeActiveCountKernel, "activeCount", activeCount);
        nodesPrefixSumsShader.SetInt("len", numNodes);
        nodesPrefixSumsShader.Dispatch(writeActiveCountKernel, 1, 1, 1);

        // // Debug: print active count calculation details
        // uint lastIndicator = indicatorsCPU[numNodes - 1];
        // uint lastPrefixSum = prefixSumsCPU[numNodes - 1];
        // uint calculatedCount = lastPrefixSum + lastIndicator;
        // Debug.Log($"Layer {layer}: Active count calculation - lastIndicator: {lastIndicator}, lastPrefixSum: {lastPrefixSum}, calculated: {calculatedCount}");

        uint[] activeCountCpu = new uint[1];
        activeCount.GetData(activeCountCpu);
        numActiveNodes = (int)activeCountCpu[0];

        // // Debug: print final active count
        // Debug.Log($"Layer {layer}: Final active count from GPU: {numActiveNodes}");
        // Debug.Log($"Layer {layer}: Manual count verification: {manualActiveCount}");
        
        // if (numActiveNodes != manualActiveCount)
        // {
        //     Debug.LogWarning($"Layer {layer}: Mismatch between GPU count ({numActiveNodes}) and manual count ({manualActiveCount})");
        // }
    }

    private void prefixSumActiveNodesCodes() // in for loop
    {
        // dispatch numActiveNodes threads, find unique-prefix active nodes
        if (nodesPrefixSumsShader == null)
        {
            Debug.LogError("NodesPrefixSums compute shader is not assigned. Please assign `nodesPrefixSumsShader` in the inspector.");
            return;
        }

        if (numActiveNodes == 0) return;

        ClearUniqueBuffers();

        // Calculate prefix bits: shift right by 3 * layer bits
        int prefixBits = layer * 3;

		// Mark uniques with prefix comparison (active indices)
		if (markActiveUniquesPrefixKernel == 0)
		{
			markActiveUniquesPrefixKernel = nodesPrefixSumsShader.FindKernel("markActiveUniquesPrefix");
		}
		nodesPrefixSumsShader.SetBuffer(markActiveUniquesPrefixKernel, "nodeMortonCodes", nodeMortonCodes);
		nodesPrefixSumsShader.SetBuffer(markActiveUniquesPrefixKernel, "indicators", indicators);
		nodesPrefixSumsShader.SetBuffer(markActiveUniquesPrefixKernel, "activeIndices", activeIndices);
		nodesPrefixSumsShader.SetInt("len", numActiveNodes);
		nodesPrefixSumsShader.SetInt("prefixBits", prefixBits);
		int groupsLinear = (numActiveNodes + 511) / 512;
		nodesPrefixSumsShader.Dispatch(markActiveUniquesPrefixKernel, groupsLinear, 1, 1);

        // // Debug: print indicators after markActiveUniquesPrefix
        // uint[] indicatorsCPU = new uint[numActiveNodes];
        // indicators.GetData(indicatorsCPU);
        // string debugStr = $"Layer {layer}: First 20 indicators (after markActiveUniquesPrefix):\n";
        // for (int i = 0; i < 20 && i < numActiveNodes; i++)
        // {
        //     debugStr += $"Indicator {i}: {indicatorsCPU[i]}\n";
        // }
        // Debug.Log(debugStr);

        // Reuse proven radix scan kernels for indicators scan
        uint tgSize = 512u;
        uint numThreadgroups = (uint)((numActiveNodes + (tgSize * 2) - 1) / (tgSize * 2));
        uint auxSize = (uint)System.Math.Max(1, (int)numThreadgroups);

        // First-level scan: indicators -> prefixSums
		radixSortShader.SetBuffer(radixPrefixSumKernelId, "input", indicators);
		radixSortShader.SetBuffer(radixPrefixSumKernelId, "output", prefixSums);
		radixSortShader.SetBuffer(radixPrefixSumKernelId, "aux", aux);
		radixSortShader.SetInt("len", numActiveNodes);
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
			radixSortShader.SetInt("len", numActiveNodes);
			radixSortShader.Dispatch(radixPrefixFixupKernelId, (int)numThreadgroups, 1, 1);
        }

        // scatterUniques unique indices
        nodesPrefixSumsShader.SetBuffer(scatterUniquesKernel, "indicators", indicators);
        nodesPrefixSumsShader.SetBuffer(scatterUniquesKernel, "prefixSums", prefixSums);
        nodesPrefixSumsShader.SetBuffer(scatterUniquesKernel, "uniqueIndices", uniqueIndices);
        nodesPrefixSumsShader.SetInt("len", numActiveNodes);
        nodesPrefixSumsShader.Dispatch(scatterUniquesKernel, groupsLinear, 1, 1);

        // Write unique count
        nodesPrefixSumsShader.SetBuffer(writeUniqueCountKernel, "indicators", indicators);
        nodesPrefixSumsShader.SetBuffer(writeUniqueCountKernel, "prefixSums", prefixSums);
        nodesPrefixSumsShader.SetBuffer(writeUniqueCountKernel, "uniqueCount", uniqueCount);
        nodesPrefixSumsShader.SetInt("len", numActiveNodes);
        nodesPrefixSumsShader.Dispatch(writeUniqueCountKernel, 1, 1, 1);

        uint[] uniqueCountCpu = new uint[1];
        uniqueCount.GetData(uniqueCountCpu);
        numUniqueActiveNodes = (int)uniqueCountCpu[0];
    }

    private void ProcessNodes() // in for loop
    {
        // dispatch numUniqueActiveNodes threads, process nodes
        // Find the kernel for processing nodes at this level
        int processNodesKernel = nodesShader.FindKernel("ProcessNodes");
        
        // Set buffers for the node processing kernel
        nodesShader.SetBuffer(processNodesKernel, "particlesBuffer", particlesBuffer);
        nodesShader.SetBuffer(processNodesKernel, "sortedMortonCodes", sortedMortonCodes);
        nodesShader.SetBuffer(processNodesKernel, "sortedParticleIndices", sortedParticleIndices);
        nodesShader.SetBuffer(processNodesKernel, "uniqueIndices", uniqueIndices);
        nodesShader.SetBuffer(processNodesKernel, "activeIndices", activeIndices);
        nodesShader.SetBuffer(processNodesKernel, "uniqueCount", uniqueCount);
        nodesShader.SetBuffer(processNodesKernel, "activeCount", activeCount);
        nodesShader.SetBuffer(processNodesKernel, "nodesBuffer", nodesBuffer);
        nodesShader.SetBuffer(processNodesKernel, "nodeMortonCodes", nodeMortonCodes);
        nodesShader.SetBuffer(processNodesKernel, "nodeIndices", nodeIndices);
        nodesShader.SetBuffer(processNodesKernel, "nodeFlagsBuffer", nodeFlagsBuffer);
        nodesShader.SetInt("numUniqueActiveNodes", numUniqueActiveNodes);
        nodesShader.SetInt("numActiveNodes", numActiveNodes);
        nodesShader.SetInt("numNodes", numNodes);
        nodesShader.SetInt("numParticles", numParticles);
        nodesShader.SetInt("layer", layer);
        
        // Dispatch one thread per unique node
        int threadGroups = Mathf.CeilToInt(numUniqueActiveNodes / 512.0f);
        nodesShader.Dispatch(processNodesKernel, threadGroups, 1, 1);
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
        nodesPrefixSumsShader.SetBuffer(clearActiveBuffersKernel, "activeIndices", activeIndices);
        nodesPrefixSumsShader.SetInt("len", numParticles);
        int groupsLinear = (numParticles + 511) / 512;
        nodesPrefixSumsShader.Dispatch(clearActiveBuffersKernel, groupsLinear, 1, 1);
    }

    // Simple debug visualization using Gizmos
    private void OnDrawGizmos()
    {
        if (nodesBuffer == null) return;

        Node[] nodesCPU = new Node[numNodes];
        nodesBuffer.GetData(nodesCPU);
        uint[] activeIndicesCpu = new uint[numActiveNodes];
        activeIndices.GetData(activeIndicesCpu);
        
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
            Color.red,      // Layer 0
            Color.green,    // Layer 1
            Color.blue,     // Layer 2
            Color.yellow,   // Layer 3
            Color.magenta,  // Layer 4
            Color.cyan,     // Layer 5
            Color.white,    // Layer 6
            Color.gray,     // Layer 7
            new Color(1f, 0.5f, 0f), // Orange - Layer 8
            new Color(0.5f, 0f, 1f), // Purple - Layer 9
            new Color(0f, 0.5f, 0.5f) // Teal - Layer 10
        };
        
        // for (int i = 0; i < numActiveNodes; i++)
        // {
        //     Node node = nodesCPU[activeIndicesCpu[i]];
        //     int layerIndex = Mathf.Clamp((int)node.layer, 0, layerColors.Length - 1);
        //     Gizmos.color = layerColors[layerIndex];
        //     Gizmos.DrawWireCube(DecodeMorton3D(node), Vector3.one * Mathf.Max(maxDetailCellSize * Mathf.Pow(2, node.layer), 0.01f));
        // }

        Particle[] particlesCPU = new Particle[numParticles];
        particlesBuffer.GetData(particlesCPU);

        for (int i = 0; i < numParticles; i++)
        {
            Particle particle = particlesCPU[i];
            int layerIndex = Mathf.Clamp((int)particle.layer, 0, layerColors.Length - 1);
            Gizmos.color = layerColors[layerIndex];
            Gizmos.DrawCube(particle.position, Vector3.one * 0.25f);
        }
    }

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
        mortonCodesBuffer?.Release();
        particleIndicesBuffer?.Release();
        nodeFlagsBuffer?.Release();
        sortedMortonCodes?.Release();
        sortedParticleIndices?.Release();
        indicators?.Release();
        prefixSums?.Release();
        aux?.Release();
        aux2?.Release();
        auxSmall?.Release();
        uniqueIndices?.Release();
        uniqueCount?.Release();
        nodesBuffer?.Release();
        nodeMortonCodes?.Release();
        nodeIndices?.Release();
    }
}

public class RadixSort
{
    private ComputeShader sortShader;
    private int prefixSumKernel;
    private int prefixFixupKernel;
    private int splitPrepKernel;
    private int splitScatterKernel;
    private int copyBufferKernel;
    private int copyIndicesKernel;
    private int clearBuffer32Kernel;

    private ComputeBuffer tempKeys;
    private ComputeBuffer tempIndices;
    private ComputeBuffer tempKeysB;
    private ComputeBuffer tempIndicesB;
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
        copyBufferKernel = sortShader.FindKernel("copyBuffer");
        copyIndicesKernel = sortShader.FindKernel("copyIndices");
        clearBuffer32Kernel = sortShader.FindKernel("clearBuffer32");

        tempKeys = new ComputeBuffer((int)maxLength, sizeof(uint), ComputeBufferType.Default);
        tempIndices = new ComputeBuffer((int)maxLength, sizeof(uint), ComputeBufferType.Default);
        tempKeysB = new ComputeBuffer((int)maxLength, sizeof(uint), ComputeBufferType.Default);
        tempIndicesB = new ComputeBuffer((int)maxLength, sizeof(uint), ComputeBufferType.Default);
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
        tempKeys.Release();
        tempIndices.Release();
        tempKeysB.Release();
        tempIndicesB.Release();
        auxBuffer.Release();
        aux2Buffer.Release();
        auxSmallBuffer.Release();
        eBuffer.Release();
        fBuffer.Release();
    }

    public void Sort(ComputeBuffer inputKeys, ComputeBuffer inputIndices, ComputeBuffer outputKeys, ComputeBuffer outputIndices, uint actualCount)
    {
        if (actualCount == 0) return;

        ClearBuffer(tempKeys, (uint)tempKeys.count);
        ClearBuffer(tempIndices, (uint)tempIndices.count);
        ClearBuffer(tempKeysB, (uint)tempKeysB.count);
        ClearBuffer(tempIndicesB, (uint)tempIndicesB.count);

        int threadGroupSize = 512;
        int threadGroups = (int)((actualCount + threadGroupSize - 1) / threadGroupSize);

        sortShader.SetBuffer(copyBufferKernel, "input", inputKeys);
        sortShader.SetBuffer(copyBufferKernel, "output", tempKeys);
        sortShader.SetInt("count", (int)actualCount);
        sortShader.Dispatch(copyBufferKernel, threadGroups, 1, 1);

        sortShader.SetBuffer(copyIndicesKernel, "inputIndices", inputIndices);
        sortShader.SetBuffer(copyIndicesKernel, "outputIndices", tempIndices);
        sortShader.SetInt("count", (int)actualCount);
        sortShader.Dispatch(copyIndicesKernel, threadGroups, 1, 1);

        ComputeBuffer keysIn = tempKeys;
        ComputeBuffer keysOut = tempKeysB;
        ComputeBuffer indicesIn = tempIndices;
        ComputeBuffer indicesOut = tempIndicesB;

        for (int i = 0; i < 32; i++)
        {
            EncodeSplit(keysIn, indicesIn, keysOut, indicesOut, (uint)i, actualCount);

            (keysIn, keysOut) = (keysOut, keysIn);
            (indicesIn, indicesOut) = (indicesOut, indicesIn);
        }

        sortShader.SetBuffer(copyBufferKernel, "input", keysIn);
        sortShader.SetBuffer(copyBufferKernel, "output", outputKeys);
        sortShader.SetInt("count", (int)actualCount);
        sortShader.Dispatch(copyBufferKernel, threadGroups, 1, 1);

        sortShader.SetBuffer(copyIndicesKernel, "inputIndices", indicesIn);
        sortShader.SetBuffer(copyIndicesKernel, "outputIndices", outputIndices);
        sortShader.SetInt("count", (int)actualCount);
        sortShader.Dispatch(copyIndicesKernel, threadGroups, 1, 1);
    }

    private void EncodeSplit(ComputeBuffer input, ComputeBuffer inputIndices, ComputeBuffer output, ComputeBuffer outputIndices, uint bit, uint count)
    {
        sortShader.SetBuffer(splitPrepKernel, "input", input);
        sortShader.SetInt("bit", (int)bit);
        sortShader.SetBuffer(splitPrepKernel, "e", eBuffer);
        sortShader.SetInt("count", (int)count);
        int threadGroups = (int)((count + 512 - 1) / 512);
        sortShader.Dispatch(splitPrepKernel, threadGroups, 1, 1);

        EncodeScan(eBuffer, fBuffer, count, bit);

        sortShader.SetBuffer(splitScatterKernel, "input", input);
        sortShader.SetBuffer(splitScatterKernel, "inputIndices", inputIndices);
        sortShader.SetBuffer(splitScatterKernel, "output", output);
        sortShader.SetBuffer(splitScatterKernel, "outputIndices", outputIndices);
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