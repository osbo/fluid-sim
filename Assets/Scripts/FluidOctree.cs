using UnityEngine;

// Octree/Node component of FluidSimulator
public partial class FluidSimulator : MonoBehaviour
{
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
        ResizeBuffer(ref diffusionGradientBuffer, numNodes, sizeof(float) * 3);

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

        nodesCPU = new Node[numNodes];
        nodesBuffer.GetData(nodesCPU);

        

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
}
