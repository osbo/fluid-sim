using UnityEngine;

// Octree/Node component of FluidSimulator
public partial class FluidSimulator : MonoBehaviour
{
    bool _loggedMissingOctreeBuffers;

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
        writeDispatchArgsFromCountKernelId = nodesPrefixSumsShader.FindKernel("WriteDispatchArgsFromCount");
        copyUintBufferKernelId     = nodesPrefixSumsShader.FindKernel("CopyUintBuffer");
        npsPrefixSumKernelId       = nodesPrefixSumsShader.FindKernel("prefixSum");
        npsPrefixFixupKernelId     = nodesPrefixSumsShader.FindKernel("prefixFixup");
        copyNodesKernelId          = nodesPrefixSumsShader.FindKernel("copyNodes");

        createLeavesKernel         = nodesShader.FindKernel("CreateLeaves");
        processNodesKernel         = nodesShader.FindKernel("ProcessNodes");
        findNeighborsKernel        = nodesShader.FindKernel("findNeighbors");
        findReverseKernel          = nodesShader.FindKernel("FindReverseConnections");
        interpolateFaceVelocitiesKernel = nodesShader.FindKernel("interpolateFaceVelocities");
        copyFaceVelocitiesKernel   = nodesShader.FindKernel("copyFaceVelocities");
        initializePhiKernel        = nodesShader.FindKernel("InitializePhi");
        propagatePhiKernel         = nodesShader.FindKernel("PropagatePhi");

        if (dispatchArgsBuffer == null)
            dispatchArgsBuffer = new ComputeBuffer(3, sizeof(uint), ComputeBufferType.IndirectArguments);
    }

    private void BindNpsPrefixCount(ComputeBuffer prefixLenSource)
    {
        nodesPrefixSumsShader.SetBuffer(markUniqueParticlesKernel, "prefixElementCount", prefixLenSource);
        nodesPrefixSumsShader.SetBuffer(markUniquesPrefixKernel, "prefixElementCount", prefixLenSource);
        nodesPrefixSumsShader.SetBuffer(markActiveNodesKernelId, "prefixElementCount", prefixLenSource);
        nodesPrefixSumsShader.SetBuffer(scatterUniquesKernel, "prefixElementCount", prefixLenSource);
        nodesPrefixSumsShader.SetBuffer(scatterActivesKernelId, "prefixElementCount", prefixLenSource);
        nodesPrefixSumsShader.SetBuffer(writeUniqueCountKernel, "prefixElementCount", prefixLenSource);
        nodesPrefixSumsShader.SetBuffer(writeNodeCountKernelId, "prefixElementCount", prefixLenSource);
        nodesPrefixSumsShader.SetBuffer(copyNodesKernelId, "prefixElementCount", prefixLenSource);
        nodesPrefixSumsShader.SetBuffer(npsPrefixSumKernelId, "prefixElementCount", prefixLenSource);
        nodesPrefixSumsShader.SetBuffer(npsPrefixFixupKernelId, "prefixElementCount", prefixLenSource);
    }

    private void WriteIndirectArgsFromCountBuffer(ComputeBuffer countBuffer)
    {
        nodesPrefixSumsShader.SetBuffer(writeDispatchArgsFromCountKernelId, "dispatchArgsFromCountSrc", countBuffer);
        nodesPrefixSumsShader.SetBuffer(writeDispatchArgsFromCountKernelId, "dispatchArgsBuffer", dispatchArgsBuffer);
        nodesPrefixSumsShader.Dispatch(writeDispatchArgsFromCountKernelId, 1, 1, 1);
    }

    private void GpuCopyUint1(ComputeBuffer dst, ComputeBuffer src)
    {
        nodesPrefixSumsShader.SetBuffer(copyUintBufferKernelId, "copyUintSrc", src);
        nodesPrefixSumsShader.SetBuffer(copyUintBufferKernelId, "copyUintDst", dst);
        nodesPrefixSumsShader.Dispatch(copyUintBufferKernelId, 1, 1, 1);
    }

    private void BindNodesOctreeCounts()
    {
        nodesShader.SetInt("numNodesCapacity", maxNodesCapacity);
        nodesShader.SetBuffer(processNodesKernel, "nodeCountBuffer", nodeCount);
        nodesShader.SetBuffer(findNeighborsKernel, "nodeCountBuffer", nodeCount);
        nodesShader.SetBuffer(findReverseKernel, "nodeCountBuffer", nodeCount);
        nodesShader.SetBuffer(interpolateFaceVelocitiesKernel, "nodeCountBuffer", nodeCount);
        nodesShader.SetBuffer(copyFaceVelocitiesKernel, "nodeCountBuffer", nodeCount);
        nodesShader.SetBuffer(initializePhiKernel, "nodeCountBuffer", nodeCount);
        nodesShader.SetBuffer(propagatePhiKernel, "nodeCountBuffer", nodeCount);
        nodesShader.SetBuffer(calculateDensityGradientKernel, "nodeCountBuffer", nodeCount);
        nodesShader.SetBuffer(applyPressureGradientKernel, "nodeCountBuffer", nodeCount);
        nodesShader.SetBuffer(applyExternalForcesKernel, "nodeCountBuffer", nodeCount);
        nodesShader.SetBuffer(applyViscosityKernelId, "nodeCountBuffer", nodeCount);
    }

    private void findUniqueParticles()
    {
        if (numParticles == 0) return;

        if (particlePrefixElementCountBuffer == null || particlesBuffer == null ||
            indicators == null || prefixSums == null || uniqueIndices == null ||
            uniqueCount == null || dispatchArgsBuffer == null || nodeCount == null)
        {
            if (!_loggedMissingOctreeBuffers)
            {
                _loggedMissingOctreeBuffers = true;
                Debug.LogError("FluidSimulator: octree/prefix buffers are not allocated (did particle init fail?). Skipping findUniqueParticles.");
            }
            return;
        }

        if (nodesPrefixSumsShader == null)
        {
            Debug.LogError("NodesPrefixSums compute shader is not assigned.");
            return;
        }

        int np = numParticles;
        int prefixBits = layer * 3;
        int groupsLinear = (np + 511) / 512;

        particlePrefixElementCountBuffer.SetData(new uint[] { (uint)np });
        BindNpsPrefixCount(particlePrefixElementCountBuffer);

        nodesPrefixSumsShader.SetBuffer(markUniqueParticlesKernel, "sortedParticles", particlesBuffer);
        nodesPrefixSumsShader.SetBuffer(markUniqueParticlesKernel, "indicators", indicators);
        nodesPrefixSumsShader.SetInt("prefixBits", prefixBits);
        nodesPrefixSumsShader.Dispatch(markUniqueParticlesKernel, groupsLinear, 1, 1);

        RunPrefixScan(indicators, prefixSums, particlePrefixElementCountBuffer);

        nodesPrefixSumsShader.SetBuffer(scatterUniquesKernel, "indicators", indicators);
        nodesPrefixSumsShader.SetBuffer(scatterUniquesKernel, "prefixSums", prefixSums);
        nodesPrefixSumsShader.SetBuffer(scatterUniquesKernel, "sortedParticles", particlesBuffer);
        nodesPrefixSumsShader.SetBuffer(scatterUniquesKernel, "uniqueIndices", uniqueIndices);
        nodesPrefixSumsShader.Dispatch(scatterUniquesKernel, groupsLinear, 1, 1);

        nodesPrefixSumsShader.SetBuffer(writeUniqueCountKernel, "indicators", indicators);
        nodesPrefixSumsShader.SetBuffer(writeUniqueCountKernel, "prefixSums", prefixSums);
        nodesPrefixSumsShader.SetBuffer(writeUniqueCountKernel, "uniqueCount", uniqueCount);
        nodesPrefixSumsShader.Dispatch(writeUniqueCountKernel, 1, 1, 1);

        nodesPrefixSumsShader.SetBuffer(writeDispatchArgsKernelId, "uniqueCount", uniqueCount);
        nodesPrefixSumsShader.SetBuffer(writeDispatchArgsKernelId, "dispatchArgsBuffer", dispatchArgsBuffer);
        nodesPrefixSumsShader.Dispatch(writeDispatchArgsKernelId, 1, 1, 1);

        GpuCopyUint1(nodeCount, uniqueCount);
    }

    private void CreateLeaves()
    {
        if (nodesShader == null)
        {
            Debug.LogError("Nodes compute shader is not assigned.");
            return;
        }
        if (numParticles == 0) return;

        BindNodesOctreeCounts();
        nodesShader.SetBuffer(createLeavesKernel, "particlesBuffer", particlesBuffer);
        nodesShader.SetBuffer(createLeavesKernel, "nodesBuffer", nodesBuffer);
        nodesShader.SetBuffer(createLeavesKernel, "uniqueIndices", uniqueIndices);
        nodesShader.SetBuffer(createLeavesKernel, "mortonCodesBuffer", mortonCodesBuffer);
        nodesShader.SetBuffer(createLeavesKernel, "uniqueCount", uniqueCount);
        nodesShader.SetInt("numParticles", numParticles);
        nodesShader.SetInt("minLayer", minLayer);
        nodesShader.SetInt("maxLayer", maxLayer);
        nodesShader.DispatchIndirect(createLeavesKernel, dispatchArgsBuffer, 0);
    }

    private void findUniqueNodes()
    {
        if (nodesPrefixSumsShader == null)
        {
            Debug.LogError("NodesPrefixSums compute shader is not assigned.");
            return;
        }

        int prefixBits = layer * 3;

        BindNpsPrefixCount(nodeCount);
        WriteIndirectArgsFromCountBuffer(nodeCount);

        nodesPrefixSumsShader.SetBuffer(markUniquesPrefixKernel, "mortonCodesBuffer", mortonCodesBuffer);
        nodesPrefixSumsShader.SetBuffer(markUniquesPrefixKernel, "indicators", indicators);
        nodesPrefixSumsShader.SetInt("prefixBits", prefixBits);
        nodesPrefixSumsShader.DispatchIndirect(markUniquesPrefixKernel, dispatchArgsBuffer, 0);

        RunPrefixScan(indicators, prefixSums, nodeCount);

        nodesPrefixSumsShader.SetBuffer(scatterUniquesKernel, "indicators", indicators);
        nodesPrefixSumsShader.SetBuffer(scatterUniquesKernel, "prefixSums", prefixSums);
        nodesPrefixSumsShader.SetBuffer(scatterUniquesKernel, "uniqueIndices", uniqueIndices);
        nodesPrefixSumsShader.DispatchIndirect(scatterUniquesKernel, dispatchArgsBuffer, 0);

        nodesPrefixSumsShader.SetBuffer(writeUniqueCountKernel, "indicators", indicators);
        nodesPrefixSumsShader.SetBuffer(writeUniqueCountKernel, "prefixSums", prefixSums);
        nodesPrefixSumsShader.SetBuffer(writeUniqueCountKernel, "uniqueCount", uniqueCount);
        nodesPrefixSumsShader.Dispatch(writeUniqueCountKernel, 1, 1, 1);

        nodesPrefixSumsShader.SetBuffer(writeDispatchArgsKernelId, "uniqueCount", uniqueCount);
        nodesPrefixSumsShader.SetBuffer(writeDispatchArgsKernelId, "dispatchArgsBuffer", dispatchArgsBuffer);
        nodesPrefixSumsShader.Dispatch(writeDispatchArgsKernelId, 1, 1, 1);
    }

    private void ProcessNodes()
    {
        BindNodesOctreeCounts();
        nodesShader.SetBuffer(processNodesKernel, "uniqueIndices", uniqueIndices);
        nodesShader.SetBuffer(processNodesKernel, "nodesBuffer", nodesBuffer);
        nodesShader.SetBuffer(processNodesKernel, "mortonCodesBuffer", mortonCodesBuffer);
        nodesShader.SetBuffer(processNodesKernel, "uniqueCount", uniqueCount);
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

        BindNpsPrefixCount(nodeCount);
        WriteIndirectArgsFromCountBuffer(nodeCount);

        nodesPrefixSumsShader.SetBuffer(markActiveNodesKernelId, "nodesBuffer", nodesBuffer);
        nodesPrefixSumsShader.SetBuffer(markActiveNodesKernelId, "indicators", indicators);
        nodesPrefixSumsShader.DispatchIndirect(markActiveNodesKernelId, dispatchArgsBuffer, 0);

        RunPrefixScan(indicators, prefixSums, nodeCount);

        nodesPrefixSumsShader.SetBuffer(scatterActivesKernelId, "indicators", indicators);
        nodesPrefixSumsShader.SetBuffer(scatterActivesKernelId, "prefixSums", prefixSums);
        nodesPrefixSumsShader.SetBuffer(scatterActivesKernelId, "nodesBuffer", nodesBuffer);
        nodesPrefixSumsShader.SetBuffer(scatterActivesKernelId, "tempNodesBuffer", tempNodesBuffer);
        nodesPrefixSumsShader.SetBuffer(scatterActivesKernelId, "mortonCodesBuffer", mortonCodesBuffer);
        nodesPrefixSumsShader.DispatchIndirect(scatterActivesKernelId, dispatchArgsBuffer, 0);

        nodesPrefixSumsShader.SetBuffer(writeNodeCountKernelId, "indicators", indicators);
        nodesPrefixSumsShader.SetBuffer(writeNodeCountKernelId, "prefixSums", prefixSums);
        nodesPrefixSumsShader.SetBuffer(writeNodeCountKernelId, "nodeCount", nodeCount);
        nodesPrefixSumsShader.Dispatch(writeNodeCountKernelId, 1, 1, 1);

        (nodesBuffer, tempNodesBuffer) = (tempNodesBuffer, nodesBuffer);
    }

    private void findNeighbors()
    {
        if (nodesShader == null)
        {
            Debug.LogError("Nodes compute shader is not assigned.");
            return;
        }

        BindNodesOctreeCounts();
        WriteIndirectArgsFromCountBuffer(nodeCount);

        nodesShader.SetBuffer(findNeighborsKernel, "nodesBuffer", nodesBuffer);
        nodesShader.SetBuffer(findNeighborsKernel, "tempNodesBuffer", tempNodesBuffer);
        nodesShader.SetBuffer(findNeighborsKernel, "neighborsBuffer", neighborsBuffer);
        nodesShader.SetBuffer(findNeighborsKernel, "mortonCodesBuffer", mortonCodesBuffer);
        nodesShader.SetBuffer(findNeighborsKernel, "solidSDFBuffer", solidSDFBuffer);
        nodesShader.SetInt("useColliders", useColliders ? 1 : 0);
        nodesShader.SetInt("solidVoxelResolution", ColliderGridResolution);
        nodesShader.DispatchIndirect(findNeighborsKernel, dispatchArgsBuffer, 0);

        nodesShader.SetBuffer(findReverseKernel, "reverseNeighborsBuffer", reverseNeighborsBuffer);
        nodesShader.SetBuffer(findReverseKernel, "neighborsBuffer", neighborsBuffer);
        nodesShader.DispatchIndirect(findReverseKernel, dispatchArgsBuffer, 0);

        nodesShader.SetBuffer(interpolateFaceVelocitiesKernel, "nodesBuffer", nodesBuffer);
        nodesShader.SetBuffer(interpolateFaceVelocitiesKernel, "tempNodesBuffer", tempNodesBuffer);
        nodesShader.SetBuffer(interpolateFaceVelocitiesKernel, "neighborsBuffer", neighborsBuffer);
        nodesShader.DispatchIndirect(interpolateFaceVelocitiesKernel, dispatchArgsBuffer, 0);

        nodesShader.SetBuffer(copyFaceVelocitiesKernel, "nodesBuffer", nodesBuffer);
        nodesShader.SetBuffer(copyFaceVelocitiesKernel, "tempNodesBuffer", tempNodesBuffer);
        nodesShader.DispatchIndirect(copyFaceVelocitiesKernel, dispatchArgsBuffer, 0);
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

        BindNodesOctreeCounts();
        nodesShader.SetBuffer(initializePhiKernel, "phiBuffer", phiBuffer);
        nodesShader.SetBuffer(initializePhiKernel, "neighborsBuffer", neighborsBuffer);
        nodesShader.Dispatch(initializePhiKernel, dispatchSize, 1, 1);

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
            nodesShader.Dispatch(propagatePhiKernel, dispatchSize, 1, 1);

            dirtyFlagBuffer.GetData(dirtyData);
            if (dirtyData[0] == 0) break;
        }

        if (writeBuffer == phiBuffer_Read)
            GpuCopyBuffer(phiBuffer_Read, phiBuffer);
    }

    private void RunPrefixScan(ComputeBuffer input, ComputeBuffer output, ComputeBuffer prefixLenSource)
    {
        BindNpsPrefixCount(prefixLenSource);

        nodesPrefixSumsShader.SetBuffer(npsPrefixSumKernelId, "indicators", input);
        nodesPrefixSumsShader.SetBuffer(npsPrefixSumKernelId, "prefixSums", output);
        nodesPrefixSumsShader.SetBuffer(npsPrefixSumKernelId, "aux", aux);
        nodesPrefixSumsShader.Dispatch(npsPrefixSumKernelId, maxPrefixThreadGroups, 1, 1);

        if (maxPrefixThreadGroups > 1)
        {
            radixSortShader.SetBuffer(radixPrefixSumKernelId, "input", aux);
            radixSortShader.SetBuffer(radixPrefixSumKernelId, "output", aux2);
            radixSortShader.SetBuffer(radixPrefixSumKernelId, "aux", auxSmall);
            radixSortShader.SetInt("len", maxAuxBlocks);
            radixSortShader.SetInt("zeroff", 1);
            radixSortShader.Dispatch(radixPrefixSumKernelId, 1, 1, 1);

            nodesPrefixSumsShader.SetBuffer(npsPrefixFixupKernelId, "prefixSums", output);
            nodesPrefixSumsShader.SetBuffer(npsPrefixFixupKernelId, "aux2", aux2);
            nodesPrefixSumsShader.Dispatch(npsPrefixFixupKernelId, maxPrefixThreadGroups, 1, 1);
        }
    }

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
