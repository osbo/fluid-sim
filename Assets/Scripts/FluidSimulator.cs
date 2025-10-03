using UnityEngine;
using UnityEngine.InputSystem;

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
    private int pullVelocitiesKernel;
    private int copyFaceVelocitiesKernel;
    
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
    private ComputeBuffer tempNodesBuffer;
    // add previous active node count, but that can be a cpu variable 
    
    // Number of nodes, active nodes, and unique active nodes
    private int numNodes;
    private int numUniqueNodes; // rename to numUniqueNodes
    private int layer;
    public int initialLayer;

    private string str;
    private int activeCount;
    private int inactiveCount;
    
    // Stopwatch to measure total octree construction time (from advection to loop end)
    private System.Diagnostics.Stopwatch totalOctreeSw;
    
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
        public Vector3 velocity;    // 12 bytes
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
        public uint active;         // 4 bytes
    }

    void Start()
    {
        layer = initialLayer;

        InitializeParticleSystem();

        // Set 10 random particles to layer 0
        Particle[] particles = new Particle[numParticles];
        particlesBuffer.GetData(particles);

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

        // Set first 10 shuffled indices to layer 0
        for (int i = 0; i < 10 && i < numParticles; i++) {
            particles[indices[i]].layer = 0;
        }

        // particles[9*numParticles/16].layer = 0;

        particlesBuffer.SetData(particles);

        // Start total octree construction timer (advection -> full build loop end)
        totalOctreeSw = System.Diagnostics.Stopwatch.StartNew();

		// Particle compute timing: Advect + Sort
		{
			var particleSw = System.Diagnostics.Stopwatch.StartNew();
			AdvectParticles();
			SortParticles();
			particleSw.Stop();
			Debug.Log($"Particle compute time (advect+sort): {particleSw.Elapsed.TotalMilliseconds:F2} ms");
		}

		// Layer 0 compute timing: unique leaves + create leaves
		{
			var layer0Sw = System.Diagnostics.Stopwatch.StartNew();
			findUniqueParticles();
			CreateLeaves();
			layer0Sw.Stop();
			Debug.Log($"Layer {layer}: {numNodes} nodes, {layer0Sw.Elapsed.TotalMilliseconds:F2} ms");
		}

		StartCoroutine(PostCreateLeavesFlow());
    }

	private System.Collections.IEnumerator PostCreateLeavesFlow()
	{
		// Calculate maxDetailCellSize for volume calculations
		Vector3 simulationBoundsMin = simulationBounds.bounds.min;
		Vector3 simulationBoundsMax = simulationBounds.bounds.max;
		Vector3 simulationSize = simulationBoundsMax - simulationBoundsMin;
		float maxDetailCellSize = Mathf.Min(simulationSize.x, simulationSize.y, simulationSize.z) / 1024.0f;
		
		for (layer = layer + 1; layer <= 10; layer++)
		{
			// Wait for Space key press to proceed to next layer, then wait for release (new Input System)
			// yield return new WaitUntil(() => Keyboard.current != null && Keyboard.current.spaceKey.wasPressedThisFrame);
			// yield return new WaitUntil(() => Keyboard.current != null && !Keyboard.current.spaceKey.isPressed);
            
			System.Diagnostics.Stopwatch sw = System.Diagnostics.Stopwatch.StartNew();
			findUniqueNodes();
			
			ProcessNodes();
			
			compactNodes();

			sw.Stop();
			Debug.Log($"Layer {layer}: {numNodes} nodes, {sw.Elapsed.TotalMilliseconds:F2} ms");
		}

        // Stop and log total octree construction time
        if (totalOctreeSw != null)
        {
            totalOctreeSw.Stop();
            Debug.Log($"Total octree construction: {numParticles} particles to {numNodes} nodes ({100.0f - 100.0f*(float)numNodes/numParticles:F2}% reduction), {totalOctreeSw.Elapsed.TotalMilliseconds:F2} ms");
        }

        // start new timer
        var pullVelocitiesSw = System.Diagnostics.Stopwatch.StartNew();

        // // Debug: print first 20 nodes
        // Node[] nodesCPU = new Node[numNodes];
        // nodesBuffer.GetData(nodesCPU);
        // string str = "Middle 40 nodes:\n";
        // for (int i = (int)(numNodes*0.875); i < (int)(numNodes*0.875 + 40); i++)
        // {   
        //     // int index = UnityEngine.Random.Range(0, numNodes);
        //     int index = i;
        //     Node node = nodesCPU[index];
        //     str += $"Layer: {node.layer}, Morton Code: {node.mortonCode}, Position: {node.position}, Velocities: (left: {node.velocities.left}, right: {node.velocities.right}, bottom: {node.velocities.bottom}, top: {node.velocities.top}, front: {node.velocities.front}, back: {node.velocities.back})\n";
        // }
        // Debug.Log(str);

        pullVelocities();

        pullVelocitiesSw.Stop();
        Debug.Log($"Pull velocities time: {pullVelocitiesSw.Elapsed.TotalMilliseconds:F2} ms");

        Node[] nodesCPU = new Node[numNodes];
        nodesBuffer.GetData(nodesCPU);
        str = "Middle 40 nodes after pullVelocities:\n";
        // for (int i = (int)(numNodes*0.875); i < (int)(numNodes*0.875 + 40); i++)
        // int numPrinted = 40;
        for (int i = 0; i < 40; i++)
        {   
            int index = UnityEngine.Random.Range(0, numNodes);
            // if (numPrinted == 40) break;
            // int index = i;
            Node node = nodesCPU[index];
            // if (node.velocities.left + node.velocities.right + node.velocities.bottom + node.velocities.top + node.velocities.front + node.velocities.back == 0) continue;
            // numPrinted++;
            float divergence = node.velocities.left + node.velocities.right + node.velocities.bottom + node.velocities.top + node.velocities.front + node.velocities.back;
            float volume = Mathf.Pow(8, node.layer) * 0.0001f;
            float divergenceNormalized = divergence / volume;
            str += $"Layer: {node.layer}, Morton Code: {node.mortonCode}, Position: {node.position}, Velocities: (left: {node.velocities.left}, right: {node.velocities.right}, bottom: {node.velocities.bottom}, top: {node.velocities.top}, front: {node.velocities.front}, back: {node.velocities.back}), Divergence: {divergence}, Volume: {volume}, Divergence Normalized: {divergenceNormalized}\n";
            // str += $"Layer: {node.layer}, Morton Code: {node.mortonCode}, Position: {node.position}, Neighbor Morton Codes: (left: {node.velocities.left}, right: {node.velocities.right}, bottom: {node.velocities.bottom}, top: {node.velocities.top}, front: {node.velocities.front}, back: {node.velocities.back})\n";
        }
        Debug.Log(str);

		yield break;
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
        particlesBuffer = new ComputeBuffer(numParticles, sizeof(float) * 3 + sizeof(float) * 3 + sizeof(uint) + sizeof(uint)); // 12 + 12 + 4 + 4 = 32 bytes

        // Set buffer data to compute shader
        particlesShader.SetBuffer(initializeParticlesKernel, "particlesBuffer", particlesBuffer);
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

        // Create node buffers (release previous ones if they exist)
        nodesBuffer?.Release();
        tempNodesBuffer?.Release();
        nodesBuffer = new ComputeBuffer(numNodes, sizeof(float) * 3 + sizeof(float) * 6 + sizeof(uint) * 3); // 12 + 24 + 4 + 4 + 4 = 48 bytes
        tempNodesBuffer = new ComputeBuffer(numNodes, sizeof(float) * 3 + sizeof(float) * 6 + sizeof(uint) * 3); // 12 + 24 + 4 + 4 + 4 = 48 bytes

        // Set buffer data to compute shader
        nodesShader.SetBuffer(createLeavesKernel, "particlesBuffer", particlesBuffer);
        nodesShader.SetBuffer(createLeavesKernel, "nodesBuffer", nodesBuffer);
        nodesShader.SetBuffer(createLeavesKernel, "uniqueIndices", uniqueIndices);
        nodesShader.SetInt("numNodes", numNodes);
        nodesShader.SetInt("numParticles", numParticles);

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
        nodesShader.SetInt("numUniqueNodes", numUniqueNodes);
        nodesShader.SetInt("numNodes", numNodes);
        nodesShader.SetInt("layer", layer);
        
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

    private void pullVelocities()
    {
        if (nodesShader == null)
        {
            Debug.LogError("Nodes compute shader is not assigned. Please assign `nodesShader` in the inspector.");
            return;
        }

        pullVelocitiesKernel = nodesShader.FindKernel("pullVelocities");
        nodesShader.SetBuffer(pullVelocitiesKernel, "nodesBuffer", nodesBuffer);
        nodesShader.SetBuffer(pullVelocitiesKernel, "tempNodesBuffer", tempNodesBuffer);
        nodesShader.SetInt("numNodes", numNodes);
        int threadGroups = Mathf.CeilToInt(numNodes / 512.0f);
        nodesShader.Dispatch(pullVelocitiesKernel, threadGroups, 1, 1);

        copyFaceVelocitiesKernel = nodesShader.FindKernel("copyFaceVelocities");
        nodesShader.SetBuffer(copyFaceVelocitiesKernel, "nodesBuffer", nodesBuffer);
        nodesShader.SetBuffer(copyFaceVelocitiesKernel, "tempNodesBuffer", tempNodesBuffer);
        nodesShader.SetInt("numNodes", numNodes);
        nodesShader.Dispatch(copyFaceVelocitiesKernel, threadGroups, 1, 1);
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

        Node[] nodesCPU = new Node[numNodes];
        nodesBuffer.GetData(nodesCPU);
        
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

            // new Color(1f, 0f, 0f),     // Red - Layer 0
            // new Color(0f, 1f, 0f),     // Green - Layer 5
            // new Color(1f, 0.3f, 0f),   // Orange-red - Layer 1
            // new Color(0f, 1f, 0.5f),   // Blue-green - Layer 6
            // new Color(1f, 0.6f, 0f),   // Orange - Layer 2
            // new Color(0f, 1f, 1f),     // Cyan - Layer 7
            // new Color(1f, 1f, 0f),     // Yellow - Layer 3
            // new Color(0f, 0.5f, 1f),   // Light blue - Layer 8
            // new Color(0.5f, 1f, 0f),   // Yellow-green - Layer 4
            // new Color(0f, 0f, 1f),     // Blue - Layer 9
            // new Color(0.5f, 0f, 1f)    // Violet - Layer 10
        };
        
        for (int i = 0; i < numNodes; i++)
        {
            Node node = nodesCPU[i];
            int layerIndex = Mathf.Clamp((int)Mathf.Min(node.layer, layer), 0, layerColors.Length - 1);
            Gizmos.color = layerColors[layerIndex];
            float divergence = node.velocities.left + node.velocities.right + node.velocities.bottom + node.velocities.top + node.velocities.front + node.velocities.back;
            float volume = Mathf.Pow(8, node.layer) * 0.01f;
            float divergenceNormalized = divergence / volume;
            float hue = Mathf.Clamp(divergenceNormalized+0.5f, 0, 1);
            Gizmos.color = Color.HSVToRGB(hue, 1, 1);
            Gizmos.DrawWireCube(DecodeMorton3D(node), Vector3.one * Mathf.Max(maxDetailCellSize * Mathf.Pow(2, Mathf.Min(node.layer, layer)), 0.01f));
        }

        // Particle[] particlesCPU = new Particle[numParticles];
        // particlesBuffer.GetData(particlesCPU);

        // for (int i = 0; i < numParticles; i++)
        // {
        //     Particle particle = particlesCPU[i];
        //     int layerIndex = Mathf.Clamp((int)particle.layer, 0, layerColors.Length - 1);
        //     Gizmos.color = layerColors[layerIndex];
        //     Gizmos.DrawCube(particle.position, Vector3.one * 0.25f);
        // }
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
        int gridResolution = (int)Mathf.Pow(2, 10 - Mathf.Min(node.layer, layer));
        
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
        indicators?.Release();
        prefixSums?.Release();
        aux?.Release();
        aux2?.Release();
        auxSmall?.Release();
        uniqueIndices?.Release();
        uniqueCount?.Release();
        nodeCount?.Release();
        nodesBuffer?.Release();
        tempNodesBuffer?.Release();
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
        sortShader.SetBuffer(clearBuffer32Kernel, "output", buffer);
        sortShader.SetInt("count", (int)count);
        int threadGroups = (int)((count + 511) / 512);
        sortShader.Dispatch(clearBuffer32Kernel, threadGroups, 1, 1);
    }
}