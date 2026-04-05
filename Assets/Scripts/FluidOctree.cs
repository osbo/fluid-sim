using UnityEngine;

// Octree/Node component of FluidSimulator
public partial class FluidSimulator : MonoBehaviour
{
    // Cache all octree/prefix-sum kernel IDs once in Start() so the layer loop never calls FindKernel.
    private void InitOctreeKernels()
    {
        if (nodesPrefixSumsShader == null || nodesShader == null || radixSortShader == null)
        {
            Debug.LogError("One or more octree shaders not assigned. Cannot cache kernel IDs.");
            return;
        }

        markUniqueParticlesKernel  = nodesPrefixSumsShader.FindKernel("markUniqueParticles");
        markUniquesPrefixKernel    = nodesPrefixSumsShader.FindKernel("markUniquesPrefix");
        scatterUniquesKernel       = nodesPrefixSumsShader.FindKernel("scatterUniques");
        writeUniqueCountKernel     = nodesPrefixSumsShader.FindKernel("writeUniqueCount");
        writeNodeCountKernelId     = nodesPrefixSumsShader.FindKernel("writeNodeCount");
        markActiveNodesKernelId    = nodesPrefixSumsShader.FindKernel("markActiveNodes");
        scatterActivesKernelId     = nodesPrefixSumsShader.FindKernel("scatterActives");
        writeDispatchArgsKernelId  = nodesPrefixSumsShader.FindKernel("WriteDispatchArgs");

        createLeavesKernel         = nodesShader.FindKernel("CreateLeaves");
        processNodesKernel         = nodesShader.FindKernel("ProcessNodes");
        findNeighborsKernel        = nodesShader.FindKernel("findNeighbors");
        findReverseKernel          = nodesShader.FindKernel("FindReverseConnections");
        interpolateFaceVelocitiesKernel = nodesShader.FindKernel("interpolateFaceVelocities");
        copyFaceVelocitiesKernel   = nodesShader.FindKernel("copyFaceVelocities");
        extractMortonCodesKernel   = nodesShader.FindKernel("ExtractMortonCodes");
        initializePhiKernel        = nodesShader.FindKernel("InitializePhi");
        propagatePhiKernel         = nodesShader.FindKernel("PropagatePhi");

        // dispatchArgsBuffer: 3 uints for DispatchIndirect (threadGroupsX, 1, 1)
        if (dispatchArgsBuffer == null)
            dispatchArgsBuffer = new ComputeBuffer(3, sizeof(uint), ComputeBufferType.IndirectArguments);
    }

    private void findUniqueParticles()
    {
        if (numParticles == 0) return;

        if (nodesPrefixSumsShader == null)
        {
            Debug.LogError("NodesPrefixSums compute shader is not assigned.");
            return;
        }

        // Grow-only buffer allocation — numParticles is fixed, no need to release/recreate each frame.
        int np = numParticles;
        uint tgSize = 512u;
        uint numThreadgroups = (uint)((np + (tgSize * 2) - 1) / (tgSize * 2));
        uint auxSize = (uint)System.Math.Max(1, (int)numThreadgroups);

        ResizeBuffer(ref indicators,    np, sizeof(uint));
        ResizeBuffer(ref prefixSums,    np, sizeof(uint));
        ResizeBuffer(ref aux,           (int)auxSize, sizeof(uint));
        ResizeBuffer(ref aux2,          (int)auxSize, sizeof(uint));
        ResizeBuffer(ref uniqueIndices, np, sizeof(uint));
        if (uniqueCount == null) uniqueCount = new ComputeBuffer(1, sizeof(uint));
        if (auxSmall == null)   auxSmall   = new ComputeBuffer(1, sizeof(uint));

        int prefixBits = layer * 3;
        int groupsLinear = (np + 511) / 512;

        // Mark unique particles
        nodesPrefixSumsShader.SetBuffer(markUniqueParticlesKernel, "sortedParticles", particlesBuffer);
        nodesPrefixSumsShader.SetBuffer(markUniqueParticlesKernel, "indicators", indicators);
        nodesPrefixSumsShader.SetInt("len", np);
        nodesPrefixSumsShader.SetInt("prefixBits", prefixBits);
        nodesPrefixSumsShader.Dispatch(markUniqueParticlesKernel, groupsLinear, 1, 1);

        // Prefix scan
        RunPrefixScan(indicators, prefixSums, np, numThreadgroups, auxSize);

        // Scatter unique indices
        nodesPrefixSumsShader.SetBuffer(scatterUniquesKernel, "indicators", indicators);
        nodesPrefixSumsShader.SetBuffer(scatterUniquesKernel, "prefixSums", prefixSums);
        nodesPrefixSumsShader.SetBuffer(scatterUniquesKernel, "sortedParticles", particlesBuffer);
        nodesPrefixSumsShader.SetBuffer(scatterUniquesKernel, "uniqueIndices", uniqueIndices);
        nodesPrefixSumsShader.SetInt("len", np);
        nodesPrefixSumsShader.Dispatch(scatterUniquesKernel, groupsLinear, 1, 1);

        // Write unique count to GPU buffer
        nodesPrefixSumsShader.SetBuffer(writeUniqueCountKernel, "indicators", indicators);
        nodesPrefixSumsShader.SetBuffer(writeUniqueCountKernel, "prefixSums", prefixSums);
        nodesPrefixSumsShader.SetBuffer(writeUniqueCountKernel, "uniqueCount", uniqueCount);
        nodesPrefixSumsShader.SetInt("len", np);
        nodesPrefixSumsShader.Dispatch(writeUniqueCountKernel, 1, 1, 1);

        // One readback needed: numNodes drives CreateLeaves dispatch + layer loop initialization.
        uint[] numNodesCpu = new uint[1];
        uniqueCount.GetData(numNodesCpu);
        numNodes = (int)numNodesCpu[0];
    }

    private void CreateLeaves()
    {
        if (nodesShader == null)
        {
            Debug.LogError("Nodes compute shader is not assigned.");
            return;
        }
        if (numNodes == 0) return;

        ResizeBuffer(ref nodesBuffer,     numNodes, sizeof(float) * 3 + sizeof(float) * 3 + sizeof(float) * 6 + sizeof(float) + sizeof(uint) * 3);
        ResizeBuffer(ref tempNodesBuffer, numNodes, sizeof(float) * 3 + sizeof(float) * 3 + sizeof(float) * 6 + sizeof(float) + sizeof(uint) * 3);
        ResizeBuffer(ref mortonCodesBuffer, numNodes, sizeof(uint));

        nodesShader.SetBuffer(createLeavesKernel, "particlesBuffer", particlesBuffer);
        nodesShader.SetBuffer(createLeavesKernel, "nodesBuffer", nodesBuffer);
        nodesShader.SetBuffer(createLeavesKernel, "uniqueIndices", uniqueIndices);
        nodesShader.SetBuffer(createLeavesKernel, "mortonCodesBuffer", mortonCodesBuffer);
        nodesShader.SetInt("numNodes", numNodes);
        nodesShader.SetInt("numParticles", numParticles);
        nodesShader.SetInt("minLayer", minLayer);
        nodesShader.SetInt("maxLayer", maxLayer);

        int threadGroups = Mathf.CeilToInt(numNodes / 512.0f);
        nodesShader.Dispatch(createLeavesKernel, threadGroups, 1, 1);
    }

    // Marks unique node prefixes, prefix-scans, scatters, writes uniqueCount to GPU,
    // then writes DispatchIndirect args so ProcessNodes can run without a CPU readback.
    private void findUniqueNodes()
    {
        if (nodesPrefixSumsShader == null)
        {
            Debug.LogError("NodesPrefixSums compute shader is not assigned.");
            return;
        }
        if (numNodes == 0) return;

        int prefixBits = layer * 3;
        int groupsLinear = (numNodes + 511) / 512;

        uint tgSize = 512u;
        uint numThreadgroups = (uint)((numNodes + (tgSize * 2) - 1) / (tgSize * 2));
        uint auxSize = (uint)System.Math.Max(1, (int)numThreadgroups);

        nodesPrefixSumsShader.SetBuffer(markUniquesPrefixKernel, "nodesBuffer", nodesBuffer);
        nodesPrefixSumsShader.SetBuffer(markUniquesPrefixKernel, "indicators", indicators);
        nodesPrefixSumsShader.SetInt("len", numNodes);
        nodesPrefixSumsShader.SetInt("prefixBits", prefixBits);
        nodesPrefixSumsShader.Dispatch(markUniquesPrefixKernel, groupsLinear, 1, 1);

        RunPrefixScan(indicators, prefixSums, numNodes, numThreadgroups, auxSize);

        nodesPrefixSumsShader.SetBuffer(scatterUniquesKernel, "indicators", indicators);
        nodesPrefixSumsShader.SetBuffer(scatterUniquesKernel, "prefixSums", prefixSums);
        nodesPrefixSumsShader.SetBuffer(scatterUniquesKernel, "uniqueIndices", uniqueIndices);
        nodesPrefixSumsShader.SetInt("len", numNodes);
        nodesPrefixSumsShader.Dispatch(scatterUniquesKernel, groupsLinear, 1, 1);

        nodesPrefixSumsShader.SetBuffer(writeUniqueCountKernel, "indicators", indicators);
        nodesPrefixSumsShader.SetBuffer(writeUniqueCountKernel, "prefixSums", prefixSums);
        nodesPrefixSumsShader.SetBuffer(writeUniqueCountKernel, "uniqueCount", uniqueCount);
        nodesPrefixSumsShader.SetInt("len", numNodes);
        nodesPrefixSumsShader.Dispatch(writeUniqueCountKernel, 1, 1, 1);

        // Compute DispatchIndirect args on GPU: no CPU readback of numUniqueNodes needed.
        nodesPrefixSumsShader.SetBuffer(writeDispatchArgsKernelId, "uniqueCount", uniqueCount);
        nodesPrefixSumsShader.SetBuffer(writeDispatchArgsKernelId, "dispatchArgsBuffer", dispatchArgsBuffer);
        nodesPrefixSumsShader.Dispatch(writeDispatchArgsKernelId, 1, 1, 1);
    }

    // ProcessNodes uses DispatchIndirect: numUniqueNodes stays on GPU, no readback.
    private void ProcessNodes()
    {
        nodesShader.SetBuffer(processNodesKernel, "uniqueIndices", uniqueIndices);
        nodesShader.SetBuffer(processNodesKernel, "nodesBuffer", nodesBuffer);
        nodesShader.SetBuffer(processNodesKernel, "mortonCodesBuffer", mortonCodesBuffer);
        nodesShader.SetBuffer(processNodesKernel, "uniqueCount", uniqueCount);
        nodesShader.SetInt("numNodes", numNodes);
        nodesShader.SetInt("layer", layer);
        nodesShader.SetInt("minLayer", minLayer);
        nodesShader.SetInt("maxLayer", maxLayer);
        nodesShader.DispatchIndirect(processNodesKernel, dispatchArgsBuffer, 0);
    }

    private void compactNodes()
    {
        if (nodesPrefixSumsShader == null)
        {
            Debug.LogError("NodesPrefixSums compute shader is not assigned.");
            return;
        }
        if (numNodes == 0)
        {
            Debug.LogError("Num nodes is 0.");
            return;
        }
        if (nodeCount == null) nodeCount = new ComputeBuffer(1, sizeof(uint));

        int groupsLinear = Mathf.Max(1, (numNodes + 511) / 512);

        nodesPrefixSumsShader.SetBuffer(markActiveNodesKernelId, "nodesBuffer", nodesBuffer);
        nodesPrefixSumsShader.SetBuffer(markActiveNodesKernelId, "indicators", indicators);
        nodesPrefixSumsShader.SetInt("len", numNodes);
        nodesPrefixSumsShader.Dispatch(markActiveNodesKernelId, groupsLinear, 1, 1);

        uint tgSize = 512u;
        uint numThreadgroups = (uint)((numNodes + (tgSize * 2) - 1) / (tgSize * 2));
        uint auxSize = (uint)System.Math.Max(1, (int)numThreadgroups);

        RunPrefixScan(indicators, prefixSums, numNodes, numThreadgroups, auxSize);

        nodesPrefixSumsShader.SetBuffer(scatterActivesKernelId, "indicators", indicators);
        nodesPrefixSumsShader.SetBuffer(scatterActivesKernelId, "prefixSums", prefixSums);
        nodesPrefixSumsShader.SetBuffer(scatterActivesKernelId, "nodesBuffer", nodesBuffer);
        nodesPrefixSumsShader.SetBuffer(scatterActivesKernelId, "tempNodesBuffer", tempNodesBuffer);
        nodesPrefixSumsShader.SetInt("len", numNodes);
        int scatterGroups = Mathf.Max(1, (numNodes + 511) / 512);
        nodesPrefixSumsShader.Dispatch(scatterActivesKernelId, scatterGroups, 1, 1);

        nodesPrefixSumsShader.SetBuffer(writeNodeCountKernelId, "indicators", indicators);
        nodesPrefixSumsShader.SetBuffer(writeNodeCountKernelId, "prefixSums", prefixSums);
        nodesPrefixSumsShader.SetBuffer(writeNodeCountKernelId, "nodeCount", nodeCount);
        nodesPrefixSumsShader.SetInt("len", numNodes);
        nodesPrefixSumsShader.Dispatch(writeNodeCountKernelId, 1, 1, 1);

        // Readback of numNodes is unavoidable here: it gates the next layer's dispatch size
        // and drives extractMortonCodes / findNeighbors after the loop.
        uint[] nodeCountCpu = new uint[1];
        nodeCount.GetData(nodeCountCpu);
        numNodes = (int)nodeCountCpu[0];

        (nodesBuffer, tempNodesBuffer) = (tempNodesBuffer, nodesBuffer);
    }

    private void findNeighbors()
    {
        if (nodesShader == null)
        {
            Debug.LogError("Nodes compute shader is not assigned.");
            return;
        }

        ResizeBuffer(ref neighborsBuffer,        numNodes * 24, sizeof(uint));
        ResizeBuffer(ref diffusionGradientBuffer, numNodes,     sizeof(float) * 3);

        int threadGroups = Mathf.CeilToInt(numNodes / 512.0f);

        nodesShader.SetBuffer(findNeighborsKernel, "nodesBuffer", nodesBuffer);
        nodesShader.SetBuffer(findNeighborsKernel, "tempNodesBuffer", tempNodesBuffer);
        nodesShader.SetBuffer(findNeighborsKernel, "neighborsBuffer", neighborsBuffer);
        nodesShader.SetBuffer(findNeighborsKernel, "mortonCodesBuffer", mortonCodesBuffer);
        nodesShader.SetInt("numNodes", numNodes);
        nodesShader.Dispatch(findNeighborsKernel, threadGroups, 1, 1);

        ResizeBuffer(ref reverseNeighborsBuffer, numNodes * 24, sizeof(uint));

        nodesShader.SetBuffer(findReverseKernel, "reverseNeighborsBuffer", reverseNeighborsBuffer);
        nodesShader.SetBuffer(findReverseKernel, "neighborsBuffer", neighborsBuffer);
        nodesShader.SetInt("numNodes", numNodes);
        nodesShader.Dispatch(findReverseKernel, threadGroups, 1, 1);

        nodesShader.SetBuffer(interpolateFaceVelocitiesKernel, "nodesBuffer", nodesBuffer);
        nodesShader.SetBuffer(interpolateFaceVelocitiesKernel, "tempNodesBuffer", tempNodesBuffer);
        nodesShader.SetBuffer(interpolateFaceVelocitiesKernel, "neighborsBuffer", neighborsBuffer);
        nodesShader.SetInt("numNodes", numNodes);
        nodesShader.Dispatch(interpolateFaceVelocitiesKernel, threadGroups, 1, 1);

        nodesShader.SetBuffer(copyFaceVelocitiesKernel, "nodesBuffer", nodesBuffer);
        nodesShader.SetBuffer(copyFaceVelocitiesKernel, "tempNodesBuffer", tempNodesBuffer);
        nodesShader.SetInt("numNodes", numNodes);
        nodesShader.Dispatch(copyFaceVelocitiesKernel, threadGroups, 1, 1);
    }

    private void ComputeLevelSet()
    {
        if (nodesShader == null)
        {
            Debug.LogError("Nodes compute shader is not assigned.");
            return;
        }

        phiBuffer?.Release();
        phiBuffer_Read?.Release();
        dirtyFlagBuffer?.Release();

        phiBuffer       = new ComputeBuffer(numNodes, sizeof(float));
        phiBuffer_Read  = new ComputeBuffer(numNodes, sizeof(float));
        dirtyFlagBuffer = new ComputeBuffer(1, sizeof(uint));

        int dispatchSize = Mathf.CeilToInt(numNodes / 512.0f);

        nodesShader.SetBuffer(initializePhiKernel, "phiBuffer", phiBuffer);
        nodesShader.SetBuffer(initializePhiKernel, "neighborsBuffer", neighborsBuffer);
        nodesShader.SetInt("numNodes", numNodes);
        nodesShader.Dispatch(initializePhiKernel, dispatchSize, 1, 1);

        // GPU copy phi → phi_Read (no CPU round-trip)
        GpuCopyBuffer(phiBuffer, phiBuffer_Read);

        ComputeBuffer readBuffer  = phiBuffer_Read;
        ComputeBuffer writeBuffer = phiBuffer;
        uint[] dirtyData = { 0 };

        int maxIterations = 32;
        for (int i = 0; i < maxIterations; i++)
        {
            (readBuffer, writeBuffer) = (writeBuffer, readBuffer);

            dirtyData[0] = 0;
            dirtyFlagBuffer.SetData(dirtyData);

            nodesShader.SetBuffer(propagatePhiKernel, "phiBuffer_Read", readBuffer);
            nodesShader.SetBuffer(propagatePhiKernel, "phiBuffer", writeBuffer);
            nodesShader.SetBuffer(propagatePhiKernel, "dirtyFlagBuffer", dirtyFlagBuffer);
            nodesShader.SetBuffer(propagatePhiKernel, "nodesBuffer", nodesBuffer);
            nodesShader.SetBuffer(propagatePhiKernel, "neighborsBuffer", neighborsBuffer);
            nodesShader.SetInt("numNodes", numNodes);
            nodesShader.Dispatch(propagatePhiKernel, dispatchSize, 1, 1);

            dirtyFlagBuffer.GetData(dirtyData);
            if (dirtyData[0] == 0) break;
        }

        if (writeBuffer == phiBuffer_Read)
            GpuCopyBuffer(phiBuffer_Read, phiBuffer);
    }

    // Shared helper: runs a two-level exclusive prefix scan (same logic used in three places).
    private void RunPrefixScan(ComputeBuffer input, ComputeBuffer output, int len,
                                uint numThreadgroups, uint auxSize)
    {
        radixSortShader.SetBuffer(radixPrefixSumKernelId, "input", input);
        radixSortShader.SetBuffer(radixPrefixSumKernelId, "output", output);
        radixSortShader.SetBuffer(radixPrefixSumKernelId, "aux", aux);
        radixSortShader.SetInt("len", len);
        radixSortShader.SetInt("zeroff", 1);
        radixSortShader.Dispatch(radixPrefixSumKernelId, (int)numThreadgroups, 1, 1);

        if (numThreadgroups > 1)
        {
            radixSortShader.SetBuffer(radixPrefixSumKernelId, "input", aux);
            radixSortShader.SetBuffer(radixPrefixSumKernelId, "output", aux2);
            radixSortShader.SetBuffer(radixPrefixSumKernelId, "aux", auxSmall);
            radixSortShader.SetInt("len", (int)auxSize);
            radixSortShader.SetInt("zeroff", 1);
            radixSortShader.Dispatch(radixPrefixSumKernelId, 1, 1, 1);

            radixSortShader.SetBuffer(radixPrefixFixupKernelId, "input", output);
            radixSortShader.SetBuffer(radixPrefixFixupKernelId, "aux", aux2);
            radixSortShader.SetInt("len", len);
            radixSortShader.Dispatch(radixPrefixFixupKernelId, (int)numThreadgroups, 1, 1);
        }
    }

    // Deinterleave morton code to get 3D grid coordinates (matches compute shader)
    private Vector3Int DeinterleaveMorton(uint m)
    {
        uint x = 0, y = 0, z = 0;
        for (int i = 0; i < 10; i++)
        {
            x |= ((m >> (3 * i + 0)) & 1) << i;
            y |= ((m >> (3 * i + 1)) & 1) << i;
            z |= ((m >> (3 * i + 2)) & 1) << i;
        }
        return new Vector3Int((int)x, (int)y, (int)z);
    }

    private Vector3 DecodeMorton3D(Node node)
    {
        uint shift = (uint)(node.layer * 3);
        uint cellMortonCode = node.mortonCode & ~((1u << (int)shift) - 1);
        Vector3Int cellGridMin = DeinterleaveMorton(cellMortonCode);

        float cellSideLength = Mathf.Pow(2, node.layer);
        Vector3 cellCenter = new Vector3(
            cellGridMin.x + cellSideLength * 0.5f,
            cellGridMin.y + cellSideLength * 0.5f,
            cellGridMin.z + cellSideLength * 0.5f
        );

        Vector3 simulationSize = simulationBounds.bounds.max - simulationBounds.bounds.min;
        return simulationBounds.bounds.min + new Vector3(
            cellCenter.x / 1024.0f * simulationSize.x,
            cellCenter.y / 1024.0f * simulationSize.y,
            cellCenter.z / 1024.0f * simulationSize.z
        );
    }
}
