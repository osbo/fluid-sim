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
    
    // Number of unique nodes (set during CreateNodes)
    private int numUniqueNodes;
    
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
    }
    
    private void SortParticles()
    {
        // Initialize radix sort
        radixSort = new RadixSort(radixSortShader, (uint)numParticles);
        
        // Sort the morton codes and their corresponding indices
        radixSort.Sort(mortonCodesBuffer, particleIndicesBuffer, sortedMortonCodes, sortedParticleIndices, (uint)numParticles);
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
        numUniqueNodes = (int)uniqueCountCpu[0];

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
        nodesShader.SetBuffer(createNodesKernel, "uniqueIndices", uniqueIndices);
        nodesShader.SetBuffer(createNodesKernel, "nodesBuffer", nodesBuffer);
        nodesShader.SetBuffer(createNodesKernel, "nodeMortonCodes", nodeMortonCodes);
        nodesShader.SetBuffer(createNodesKernel, "nodeIndices", nodeIndices);
        nodesShader.SetBuffer(createNodesKernel, "nodeFlagsBuffer", nodeFlagsBuffer);
        nodesShader.SetInt("numUniqueNodes", numUniqueNodes);
        nodesShader.SetInt("numParticles", numParticles);

        // Dispatch the kernel
        int threadGroups = Mathf.CeilToInt(numUniqueNodes / 64.0f);
        nodesShader.Dispatch(createNodesKernel, threadGroups, 1, 1);
    }

    // Simple debug visualization using Gizmos
    private void OnDrawGizmos()
    {
        if (nodesBuffer == null) return;

        Node[] nodes = new Node[numUniqueNodes];
        nodesBuffer.GetData(nodes);

        // Define 10 colors for different layers
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
            new Color(0.5f, 0f, 1f)  // Purple - Layer 9
        };
        
        for (int i = 0; i < numUniqueNodes; i++)
        {
            int layerIndex = Mathf.Clamp((int)nodes[i].layer, 0, layerColors.Length - 1);
            Gizmos.color = layerColors[layerIndex];
            Gizmos.DrawWireCube(DecodeMorton3D(nodes[i].mortonCode), Vector3.one * 0.02f);
        }
    }

    private Vector3 DecodeMorton3D(uint mortonCode)
    {
        uint x = 0, y = 0, z = 0;
        for (int i = 0; i < 10; i++)
        {
            x |= ((mortonCode >> (i * 3 + 0)) & 1) << i;
            y |= ((mortonCode >> (i * 3 + 1)) & 1) << i;
            z |= ((mortonCode >> (i * 3 + 2)) & 1) << i;
        }
        
        // Convert normalized coordinates (0-1023) back to world space
        Vector3 normalizedPos = new Vector3(x, y, z);
        Vector3 simulationBoundsMin = simulationBounds.bounds.min;
        Vector3 simulationBoundsMax = simulationBounds.bounds.max;
        Vector3 simulationSize = simulationBoundsMax - simulationBoundsMin;
        
        // Reverse the normalization: worldPos = normalizedPos / mortonNormalizationFactor + simulationBoundsMin
        Vector3 mortonNormalizationFactor = new Vector3(
            1023.0f / simulationSize.x,
            1023.0f / simulationSize.y,
            1023.0f / simulationSize.z
        );
        
        Vector3 worldPos = new Vector3(
            normalizedPos.x / mortonNormalizationFactor.x,
            normalizedPos.y / mortonNormalizationFactor.y,
            normalizedPos.z / mortonNormalizationFactor.z
        ) + simulationBoundsMin;
        
        return worldPos;
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