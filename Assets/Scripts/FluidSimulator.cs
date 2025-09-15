using UnityEngine;

public class FluidSimulator : MonoBehaviour
{
    [SerializeField] private BoxCollider simulationBounds;
    [SerializeField] private GameObject fluidInitialBounds;
    
    public ComputeShader radixSortShader;
    public ComputeShader fluidKernelsShader;
    public ComputeShader leavesShader;
    public ComputeShader nodesShader;
    public int numParticles = 10000;
    
    private RadixSort radixSort;
    private int initializeParticlesKernel;
    private int markUniquesKernel;
    private int leavesPrefixSumKernel;      // not used after simplification, but kept for reference
    private int leavesPrefixFixupKernel;    // not used after simplification, but kept for reference
    private int scatterUniquesKernel;
    private int writeUniqueCountKernel;
    // Reuse proven radix kernels for scan
    private int radixPrefixSumKernelId;
    private int radixPrefixFixupKernelId;
    private int createNodesKernel;
    
    // GPU Buffers
    private ComputeBuffer particlesBuffer;
    private ComputeBuffer mortonCodesBuffer;
    private ComputeBuffer particleIndicesBuffer;
    private ComputeBuffer nodeFlagsBuffer; // Packed: 00(unique)(active)
    private ComputeBuffer sortedMortonCodes;
    private ComputeBuffer sortedParticleIndices;
    private ComputeBuffer uniqueIndicators;
    private ComputeBuffer leavesPrefixSums;
    private ComputeBuffer leavesAux;
    private ComputeBuffer leavesAux2;
    private ComputeBuffer uniqueIndices;
    private ComputeBuffer uniqueCount;
    private ComputeBuffer leavesAuxSmall;
    private ComputeBuffer nodesBuffer;
    private ComputeBuffer nodeMortonCodes;
    private ComputeBuffer nodeIndices;
    
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
        public uint layer;          // 4 bytes
        public uint mortonCode;     // 4 bytes
    }

    void Start()
    {
        InitializeParticleSystem();
        SortParticles();
        PrefixSumLeaves();
        CreateNodes();
    }
    
    private void InitializeParticleSystem()
    {
        // Get kernel index
        initializeParticlesKernel = fluidKernelsShader.FindKernel("InitializeParticles");
        
        // Create buffers
        particlesBuffer = new ComputeBuffer(numParticles, sizeof(float) * 3 + sizeof(float) * 3 + sizeof(uint) + sizeof(uint)); // 32 bytes
        mortonCodesBuffer = new ComputeBuffer(numParticles, sizeof(uint));
        particleIndicesBuffer = new ComputeBuffer(numParticles, sizeof(uint));
        // nodeFlagsBuffer created later in CreateNodes() with correct size
        sortedMortonCodes = new ComputeBuffer(numParticles, sizeof(uint));
        sortedParticleIndices = new ComputeBuffer(numParticles, sizeof(uint));
        
        // Set buffer data to compute shader
        fluidKernelsShader.SetBuffer(initializeParticlesKernel, "particlesBuffer", particlesBuffer);
        fluidKernelsShader.SetBuffer(initializeParticlesKernel, "mortonCodesBuffer", mortonCodesBuffer);
        fluidKernelsShader.SetBuffer(initializeParticlesKernel, "particleIndicesBuffer", particleIndicesBuffer);
        // nodeFlagsBuffer not set here - created later in CreateNodes()
        fluidKernelsShader.SetInt("count", numParticles);
        
        // Calculate bounds
        Vector3 simulationBoundsMin = simulationBounds.bounds.min;
        Vector3 simulationBoundsMax = simulationBounds.bounds.max;
        
        // Get fluid initial bounds from the GameObject's transform
        Vector3 fluidInitialBoundsMin = fluidInitialBounds.transform.position - fluidInitialBounds.transform.localScale * 0.5f;
        Vector3 fluidInitialBoundsMax = fluidInitialBounds.transform.position + fluidInitialBounds.transform.localScale * 0.5f;
        
        // Calculate morton code normalization factors on CPU
        Vector3 simulationSize = simulationBoundsMax - simulationBoundsMin;
        
        // For 32-bit Morton codes: 10 bits per axis = 1024 possible values (0-1023)
        // Normalize each axis to 0-1023 range
        Vector3 mortonNormalizationFactor = new Vector3(
            1023.0f / simulationSize.x,
            1023.0f / simulationSize.y,
            1023.0f / simulationSize.z
        );
        
        // Max morton value is 1023 for each axis
        float mortonMaxValue = 1023.0f;
        
        // Calculate grid dimensions for even particle distribution
        Vector3 fluidInitialSize = fluidInitialBoundsMax - fluidInitialBoundsMin;
        
        // Calculate grid dimensions that will fit numParticles particles
        // Use the same major order as morton code (Z, Y, X)
        float volumePerParticle = (fluidInitialSize.x * fluidInitialSize.y * fluidInitialSize.z) / numParticles;
        float gridSpacing = Mathf.Pow(volumePerParticle, 1.0f / 3.0f);
        
        Vector3Int gridDimensions = new Vector3Int(
            Mathf.Max(1, Mathf.RoundToInt(fluidInitialSize.x / gridSpacing)),
            Mathf.Max(1, Mathf.RoundToInt(fluidInitialSize.y / gridSpacing)),
            Mathf.Max(1, Mathf.RoundToInt(fluidInitialSize.z / gridSpacing))
        );
        
        // Adjust grid dimensions to ensure we don't exceed numParticles
        while (gridDimensions.x * gridDimensions.y * gridDimensions.z > numParticles)
        {
            if (gridDimensions.x > 1) gridDimensions.x--;
            else if (gridDimensions.y > 1) gridDimensions.y--;
            else if (gridDimensions.z > 1) gridDimensions.z--;
            else break;
        }
        
        // Calculate actual grid spacing based on final dimensions
        Vector3 actualGridSpacing = new Vector3(
            fluidInitialSize.x / Mathf.Max(1, gridDimensions.x - 1),
            fluidInitialSize.y / Mathf.Max(1, gridDimensions.y - 1),
            fluidInitialSize.z / Mathf.Max(1, gridDimensions.z - 1)
        );
        
        // Set bounds parameters to compute shader
        fluidKernelsShader.SetVector("simulationBoundsMin", simulationBoundsMin);
        fluidKernelsShader.SetVector("simulationBoundsMax", simulationBoundsMax);
        fluidKernelsShader.SetVector("fluidInitialBoundsMin", fluidInitialBoundsMin);
        fluidKernelsShader.SetVector("fluidInitialBoundsMax", fluidInitialBoundsMax);
        fluidKernelsShader.SetVector("mortonNormalizationFactor", mortonNormalizationFactor);
        fluidKernelsShader.SetFloat("mortonMaxValue", mortonMaxValue);
        
        // Set grid parameters to compute shader
        fluidKernelsShader.SetInts("gridDimensions", new int[] { (int)gridDimensions.x, (int)gridDimensions.y, (int)gridDimensions.z });
        fluidKernelsShader.SetVector("gridSpacing", actualGridSpacing);
        
        // Dispatch the kernel
        int threadGroups = Mathf.CeilToInt(numParticles / 64.0f);
        fluidKernelsShader.Dispatch(initializeParticlesKernel, threadGroups, 1, 1);
        
        // Debug.Log($"Initialized {numParticles} particles with bounds: simulation({simulationBoundsMin} to {simulationBoundsMax}), fluid initial({fluidInitialBoundsMin} to {fluidInitialBoundsMax})");
        // Debug.Log($"Morton normalization factors: {mortonNormalizationFactor}, max value: {mortonMaxValue}");
        // Debug.Log($"Grid dimensions: {gridDimensions}, grid spacing: {actualGridSpacing}");
    }
    
    private void SortParticles()
    {
        // Initialize radix sort
        radixSort = new RadixSort(radixSortShader, (uint)numParticles);
        
        // Sort the morton codes and their corresponding indices
        radixSort.Sort(mortonCodesBuffer, particleIndicesBuffer, sortedMortonCodes, sortedParticleIndices, (uint)numParticles);
        
        // Debug: Log sorted codes and indices
        uint[] sortedMortonCodesArray = new uint[numParticles];
        sortedMortonCodes.GetData(sortedMortonCodesArray);
        
        uint[] sortedIndicesArray = new uint[numParticles];
        sortedParticleIndices.GetData(sortedIndicesArray);
        
        // // Get particle data
        // Particle[] particlesArray = new Particle[numParticles];
        // particlesBuffer.GetData(particlesArray);
        
        string sorted_output = "Sorted Morton Codes (first 20): ";
        string indices_output = "Corresponding Indices (first 20): ";
        for (int i = 0; i < Mathf.Min(20, numParticles); i++)
        {
            sorted_output += sortedMortonCodesArray[i] + " ";
            indices_output += sortedIndicesArray[i] + " ";
        }
        Debug.Log(sorted_output);
        Debug.Log(indices_output);
        
        // // Print particle data for first 20 sorted particles
        // Debug.Log("=== First 20 Sorted Particles ===");
        // for (int i = 0; i < Mathf.Min(20, numParticles); i++)
        // {
        //     uint particleIndex = sortedIndicesArray[i];
        //     Particle particle = particlesArray[particleIndex];
            
        //     Debug.Log($"Sorted Position {i}: Particle[{particleIndex}] - " +
        //              $"Pos=({particle.position.x:F3}, {particle.position.y:F3}, {particle.position.z:F3}), " +
        //              $"Vel=({particle.velocity.x:F3}, {particle.velocity.y:F3}, {particle.velocity.z:F3}), " +
        //              $"Layer={particle.layer}, MortonCode={particle.mortonCode}");
        // }
    }

    private void PrefixSumLeaves()
    {
        int count = numParticles;
        if (count == 0) return;

        if (leavesShader == null)
        {
            Debug.LogError("Leaves compute shader is not assigned. Please assign `leavesShader` in the inspector.");
            return;
        }

        // Find kernels
        markUniquesKernel = leavesShader.FindKernel("markUniques");
        leavesPrefixSumKernel = leavesShader.FindKernel("prefixSum");
        leavesPrefixFixupKernel = leavesShader.FindKernel("prefixFixup");
        scatterUniquesKernel = leavesShader.FindKernel("scatterUniques");
        writeUniqueCountKernel = leavesShader.FindKernel("writeUniqueCount");

        if (markUniquesKernel < 0 || leavesPrefixSumKernel < 0 || leavesPrefixFixupKernel < 0 ||
            scatterUniquesKernel < 0 || writeUniqueCountKernel < 0)
        {
            Debug.LogError("One or more kernels not found in Leaves.compute. Verify #pragma kernel names and shader assignment.");
            return;
        }

        // Allocate leaves buffers
        uniqueIndicators = new ComputeBuffer(count, sizeof(uint));
        leavesPrefixSums = new ComputeBuffer(count, sizeof(uint));

        uint tgSize = 512u;
        uint numThreadgroups = (uint)((count + (tgSize * 2) - 1) / (tgSize * 2));
        uint auxSize = (uint)System.Math.Max(1, (int)numThreadgroups);
        leavesAux = new ComputeBuffer((int)auxSize, sizeof(uint));
        leavesAux2 = new ComputeBuffer((int)auxSize, sizeof(uint));

        uniqueIndices = new ComputeBuffer(count, sizeof(uint));
        uniqueCount = new ComputeBuffer(1, sizeof(uint));

        // mark uniques
        leavesShader.SetBuffer(markUniquesKernel, "sortedMortonCodes", sortedMortonCodes);
        leavesShader.SetBuffer(markUniquesKernel, "uniqueIndicators", uniqueIndicators);
        leavesShader.SetInt("count", count);
        int groupsLinear = (count + 511) / 512;
        leavesShader.Dispatch(markUniquesKernel, groupsLinear, 1, 1);

        // Reuse proven radix scan kernels for uniqueIndicators scan
        radixPrefixSumKernelId = radixSortShader.FindKernel("prefixSum");
        radixPrefixFixupKernelId = radixSortShader.FindKernel("prefixFixup");

        // First-level scan: uniqueIndicators -> leavesPrefixSums
        radixSortShader.SetBuffer(radixPrefixSumKernelId, "input", uniqueIndicators);
        radixSortShader.SetBuffer(radixPrefixSumKernelId, "output", leavesPrefixSums);
        radixSortShader.SetBuffer(radixPrefixSumKernelId, "aux", leavesAux);
        radixSortShader.SetInt("len", count);
        radixSortShader.SetInt("zeroff", 1);
        radixSortShader.Dispatch(radixPrefixSumKernelId, (int)numThreadgroups, 1, 1);

        if (numThreadgroups > 1)
        {
            // Scan aux -> leavesAux2
            if (leavesAuxSmall == null) leavesAuxSmall = new ComputeBuffer(1, sizeof(uint));
            uint auxThreadgroups = 1; // aux length is small
            radixSortShader.SetBuffer(radixPrefixSumKernelId, "input", leavesAux);
            radixSortShader.SetBuffer(radixPrefixSumKernelId, "output", leavesAux2);
            radixSortShader.SetBuffer(radixPrefixSumKernelId, "aux", leavesAuxSmall);
            radixSortShader.SetInt("len", (int)auxSize);
            radixSortShader.SetInt("zeroff", 1);
            radixSortShader.Dispatch(radixPrefixSumKernelId, (int)auxThreadgroups, 1, 1);

            // Fixup: add scanned aux into leavesPrefixSums
            radixSortShader.SetBuffer(radixPrefixFixupKernelId, "input", leavesPrefixSums);
            radixSortShader.SetBuffer(radixPrefixFixupKernelId, "aux", leavesAux2);
            radixSortShader.SetInt("len", count);
            radixSortShader.Dispatch(radixPrefixFixupKernelId, (int)numThreadgroups, 1, 1);
        }

        // scatter uniques -> uniqueIndices (store corresponding particle index)
        leavesShader.SetBuffer(scatterUniquesKernel, "uniqueIndicators", uniqueIndicators);
        leavesShader.SetBuffer(scatterUniquesKernel, "prefixSums", leavesPrefixSums);
        leavesShader.SetBuffer(scatterUniquesKernel, "sortedIndices", sortedParticleIndices);
        leavesShader.SetBuffer(scatterUniquesKernel, "uniqueIndices", uniqueIndices);
        leavesShader.SetInt("count", count);
        leavesShader.Dispatch(scatterUniquesKernel, groupsLinear, 1, 1);

        // write unique count
        leavesShader.SetBuffer(writeUniqueCountKernel, "uniqueIndicators", uniqueIndicators);
        leavesShader.SetBuffer(writeUniqueCountKernel, "prefixSums", leavesPrefixSums);
        leavesShader.SetBuffer(writeUniqueCountKernel, "uniqueCount", uniqueCount);
        leavesShader.SetInt("count", count);
        leavesShader.Dispatch(writeUniqueCountKernel, 1, 1, 1);

        // Debug output (disabled in production)
        uint[] uniqueCountCpu = new uint[1];
        uniqueCount.GetData(uniqueCountCpu);
        int uniques = (int)uniqueCountCpu[0];
        Debug.Log($"Particles created: {numParticles}, Unique morton codes: {uniques}");
        int toPrint = Mathf.Min(20, uniques);
        uint[] uniqueIdxCpu = new uint[toPrint];
        uniqueIndices.GetData(uniqueIdxCpu, 0, 0, toPrint);
        uint[] sortedCodesCpu = new uint[Mathf.Min(count, 200)];
        sortedMortonCodes.GetData(sortedCodesCpu, 0, 0, Mathf.Min(count, 200));
        string uIdx = "First unique indices (particle indices): ";
        string uCodes = "First unique morton codes: ";
        for (int i = 0; i < toPrint; i++)
        {
            uIdx += uniqueIdxCpu[i] + " ";
            uCodes += sortedCodesCpu[i] + " ";
        }
        Debug.Log(uIdx);
        Debug.Log(uCodes);
        VerifyLeavesOnCPU(count, uniques);
    }

    private void CreateNodes()
    {
        if (nodesShader == null)
        {
            Debug.LogError("Nodes compute shader is not assigned. Please assign `nodesShader` in the inspector.");
            return;
        }

        // Get unique count from GPU
        uint[] uniqueCountCpu = new uint[1];
        uniqueCount.GetData(uniqueCountCpu);
        int numUniqueNodes = (int)uniqueCountCpu[0];

        if (numUniqueNodes == 0) return;

        // Find kernel
        createNodesKernel = nodesShader.FindKernel("CreateNodes");

        // Create node buffers
        nodesBuffer = new ComputeBuffer(numUniqueNodes, sizeof(float) * 3 + sizeof(float) * 3 + sizeof(uint) + sizeof(uint)); // 32 bytes
        nodeMortonCodes = new ComputeBuffer(numUniqueNodes, sizeof(uint));
        nodeIndices = new ComputeBuffer(numUniqueNodes, sizeof(uint));
        nodeFlagsBuffer = new ComputeBuffer(numUniqueNodes, sizeof(uint)); // 4 bytes (packed flags)

        // Set buffer data to compute shader
        nodesShader.SetBuffer(createNodesKernel, "particlesBuffer", particlesBuffer);
        nodesShader.SetBuffer(createNodesKernel, "sortedParticleIndices", sortedParticleIndices);
        nodesShader.SetBuffer(createNodesKernel, "nodesBuffer", nodesBuffer);
        nodesShader.SetBuffer(createNodesKernel, "nodeMortonCodes", nodeMortonCodes);
        nodesShader.SetBuffer(createNodesKernel, "nodeIndices", nodeIndices);
        nodesShader.SetBuffer(createNodesKernel, "nodeFlagsBuffer", nodeFlagsBuffer);
        nodesShader.SetInt("numUniqueNodes", numUniqueNodes);
        nodesShader.SetInt("numParticles", numParticles);

        // Dispatch the kernel
        int threadGroups = Mathf.CeilToInt(numUniqueNodes / 64.0f);
        nodesShader.Dispatch(createNodesKernel, threadGroups, 1, 1);

        Debug.Log($"Created {numUniqueNodes} nodes from {numParticles} particles");
        
        // Debug: Print first few nodes and aggregated nodes
        PrintNodeDebugInfo(numUniqueNodes);
    }
    
    private void PrintNodeDebugInfo(int numUniqueNodes)
    {
        if (numUniqueNodes == 0) return;
        
        // Read back node data
        Node[] nodes = new Node[Mathf.Min(numUniqueNodes, 20)]; // First 20 nodes
        uint[] nodeMortonCodesData = new uint[Mathf.Min(numUniqueNodes, 20)];
        uint[] nodeIndicesData = new uint[Mathf.Min(numUniqueNodes, 20)];
        
        nodesBuffer.GetData(nodes, 0, 0, nodes.Length);
        nodeMortonCodes.GetData(nodeMortonCodesData, 0, 0, nodeMortonCodesData.Length);
        nodeIndices.GetData(nodeIndicesData, 0, 0, nodeIndicesData.Length);
        
        // Print first few nodes
        Debug.Log("=== First 10 Nodes ===");
        for (int i = 0; i < Mathf.Min(10, nodes.Length); i++)
        {
            Debug.Log($"Node {i}: Pos=({nodes[i].position.x:F2}, {nodes[i].position.y:F2}, {nodes[i].position.z:F2}), " +
                     $"Vel=({nodes[i].velocity.x:F2}, {nodes[i].velocity.y:F2}, {nodes[i].velocity.z:F2}), " +
                     $"Layer={nodes[i].layer}, MortonCode={nodeMortonCodesData[i]}");
        }
        
        // Find nodes that aggregated multiple particles by checking particle counts
        Debug.Log("=== Nodes with Multiple Particles ===");
        int aggregatedCount = 0;
        
        // Read back particle data to analyze aggregation
        uint[] sortedMortonCodesData = new uint[numParticles];
        uint[] sortedParticleIndicesData = new uint[numParticles];
        uint[] uniqueIndicesData = new uint[numUniqueNodes];
        
        sortedMortonCodes.GetData(sortedMortonCodesData);
        sortedParticleIndices.GetData(sortedParticleIndicesData);
        uniqueIndices.GetData(uniqueIndicesData);
        
        for (int i = 0; i < numUniqueNodes && aggregatedCount < 10; i++)
        {
            uint mortonCode = nodeMortonCodesData[i];
            uint firstParticleIndex = uniqueIndicesData[i];
            
            // Count how many particles share this morton code
            int particleCount = 1;
            for (int j = (int)firstParticleIndex + 1; j < numParticles; j++)
            {
                if (sortedMortonCodesData[j] == mortonCode)
                {
                    particleCount++;
                }
                else
                {
                    break; // Consecutive particles with same code
                }
            }
            
            if (particleCount > 1)
            {
                Debug.Log($"Node {i}: Aggregated {particleCount} particles, MortonCode={mortonCode}, " +
                         $"Pos=({nodes[i].position.x:F2}, {nodes[i].position.y:F2}, {nodes[i].position.z:F2})");
                aggregatedCount++;
            }
        }
        
        if (aggregatedCount == 0)
        {
            Debug.Log("No nodes aggregated multiple particles (all nodes represent single particles)");
        }
    }

    private void VerifyLeavesOnCPU(int count, int uniques)
    {
        // Read back required GPU buffers
        uint[] codes = new uint[count];
        uint[] sortedIdx = new uint[count];
        sortedMortonCodes.GetData(codes);
        sortedParticleIndices.GetData(sortedIdx);

        uint[] gpuUniqueParticleIdx = new uint[uniques];
        if (uniques > 0)
        {
            uniqueIndices.GetData(gpuUniqueParticleIdx, 0, 0, uniques);
        }

        // Build CPU reference list: particle indices at the start of each unique code run
        System.Collections.Generic.List<uint> cpuUniqueParticleIdx = new System.Collections.Generic.List<uint>(uniques);
        for (int i = 0; i < count; i++)
        {
            if (i == 0 || codes[i] != codes[i - 1])
            {
                cpuUniqueParticleIdx.Add(sortedIdx[i]);
            }
        }

        // Compare counts
        if (cpuUniqueParticleIdx.Count != uniques)
        {
            Debug.LogError($"Leaves verification FAILED: GPU uniques={uniques}, CPU uniques={cpuUniqueParticleIdx.Count}");
            return;
        }

        // Compare lists element-wise (order should match the sorted order)
        int maxCheck = cpuUniqueParticleIdx.Count;
        for (int i = 0; i < maxCheck; i++)
        {
            if (gpuUniqueParticleIdx[i] != cpuUniqueParticleIdx[i])
            {
                Debug.LogError($"Leaves verification FAILED at {i}: GPU particleIdx={gpuUniqueParticleIdx[i]} vs CPU={cpuUniqueParticleIdx[i]}");
                return;
            }
        }

        Debug.Log("Leaves verification: OK (GPU unique indices match CPU reference)");

        // Also print the Morton codes that are not unique (distinct values with frequency > 1)
        System.Collections.Generic.List<uint> nonUniqueCodes = new System.Collections.Generic.List<uint>();
        int idx = 0;
        while (idx < count)
        {
            uint code = codes[idx];
            int runStart = idx;
            while (idx + 1 < count && codes[idx + 1] == code) idx++;
            int runLen = idx - runStart + 1;
            if (runLen > 1)
            {
                nonUniqueCodes.Add(code);
            }
            idx++;
        }

        string nonUniqueMsg = "Non-unique Morton codes (distinct, count=" + nonUniqueCodes.Count + "): ";
        for (int i = 0; i < nonUniqueCodes.Count; i++)
        {
            nonUniqueMsg += nonUniqueCodes[i] + (i + 1 < nonUniqueCodes.Count ? " " : "");
        }
        Debug.Log(nonUniqueMsg);
    }

    // Simple debug visualization using Gizmos
    private void OnDrawGizmos()
    {
        if (particlesBuffer == null) return;
        
        // Read particle data back to CPU for visualization
        Particle[] particles = new Particle[numParticles];
        particlesBuffer.GetData(particles);
        
        // Set gizmo color to blue
        Gizmos.color = Color.blue;
        
        // Draw each particle as a small sphere
        for (int i = 0; i < numParticles; i++)
        {
            float size = Mathf.Pow(8.0f, particles[i].layer) * 0.002f; // Scale down the size
            Gizmos.DrawSphere(particles[i].position, size);
        }
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
        uniqueIndicators?.Release();
        leavesPrefixSums?.Release();
        leavesAux?.Release();
        leavesAux2?.Release();
        leavesAuxSmall?.Release();
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
        int threadGroups = (int)((count + 63) / 64);
        sortShader.Dispatch(clearBuffer32Kernel, threadGroups, 1, 1);
    }
}