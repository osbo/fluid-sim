using UnityEngine;

public class FluidSimulator : MonoBehaviour
{
    [SerializeField] private BoxCollider simulationBounds;
    [SerializeField] private GameObject fluidInitialBounds;
    
    // Particle system parameters
    private const int PARTICLE_COUNT = 50000;
    
    // Compute shader and kernel
    [SerializeField] private ComputeShader fluidKernels;
    [SerializeField] private ComputeShader radixSortShader;
    private int initializeParticlesKernel;
    
    // Radix sort kernel indices
    private int splitPrepKernel;
    private int scatterElementsKernel;
    private int copyBufferKernel;
    
    // GPU Buffers
    private ComputeBuffer particlesBuffer;
    private ComputeBuffer mortonCodesBuffer;  // Separate morton codes buffer
    private ComputeBuffer particleIndicesBuffer;  // Separate indices buffer
    private ComputeBuffer nodeFlagsBuffer; // Packed: 00(unique)(active)
    
    // Radix sort buffers (separate keys and indices like the working Metal implementation)
    private ComputeBuffer inputKeysBuffer;
    private ComputeBuffer inputIndicesBuffer;
    private ComputeBuffer outputKeysBuffer;
    private ComputeBuffer outputIndicesBuffer;
    private ComputeBuffer eBuffer;  // Bit flags
    private ComputeBuffer fBuffer;  // Prefix sums
    
    // Particle struct (must match compute shader)
    private struct Particle
    {
        public Vector3 position;    // 12 bytes
        public Vector3 velocity;    // 12 bytes
        public uint layer;          // 4 bytes
        public uint mortonCode;     // 4 bytes
    }
    
    
    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        InitializeParticleSystem();
        
        // Test radix sort functionality
        Debug.Log("=== Testing Radix Sort ===");
        VerifySort(); // Should show unsorted initially
        
        RadixSort(); // Sort the morton codes
        VerifySort(); // Should show sorted after
    }
    
    private void InitializeParticleSystem()
    {
        // Get kernel indices
        initializeParticlesKernel = fluidKernels.FindKernel("InitializeParticles");
        
        // Get radix sort kernel indices
        if (radixSortShader != null)
        {
            splitPrepKernel = radixSortShader.FindKernel("SplitPrep");
            scatterElementsKernel = radixSortShader.FindKernel("ScatterElements");
            copyBufferKernel = radixSortShader.FindKernel("CopyBuffer");
        }
        
        // Create buffers
        particlesBuffer = new ComputeBuffer(PARTICLE_COUNT, sizeof(float) * 3 + sizeof(float) * 3 + sizeof(uint) + sizeof(uint)); // 32 bytes
        mortonCodesBuffer = new ComputeBuffer(PARTICLE_COUNT, sizeof(uint)); // 4 bytes (morton codes only)
        particleIndicesBuffer = new ComputeBuffer(PARTICLE_COUNT, sizeof(uint)); // 4 bytes (indices only)
        nodeFlagsBuffer = new ComputeBuffer(PARTICLE_COUNT, sizeof(uint)); // 4 bytes (packed flags)
        
        // Create radix sort buffers (separate keys and indices)
        inputKeysBuffer = new ComputeBuffer(PARTICLE_COUNT, sizeof(uint)); // Morton codes
        inputIndicesBuffer = new ComputeBuffer(PARTICLE_COUNT, sizeof(uint)); // Particle indices
        outputKeysBuffer = new ComputeBuffer(PARTICLE_COUNT, sizeof(uint)); // Output morton codes
        outputIndicesBuffer = new ComputeBuffer(PARTICLE_COUNT, sizeof(uint)); // Output indices
        eBuffer = new ComputeBuffer(PARTICLE_COUNT, sizeof(uint)); // Bit flags (0 or 1)
        fBuffer = new ComputeBuffer(PARTICLE_COUNT, sizeof(uint)); // Prefix sums
        
        // Set buffer data to compute shader
        fluidKernels.SetBuffer(initializeParticlesKernel, "particlesBuffer", particlesBuffer);
        fluidKernels.SetBuffer(initializeParticlesKernel, "mortonCodesBuffer", mortonCodesBuffer);
        fluidKernels.SetBuffer(initializeParticlesKernel, "particleIndicesBuffer", particleIndicesBuffer);
        fluidKernels.SetBuffer(initializeParticlesKernel, "nodeFlagsBuffer", nodeFlagsBuffer);
        
        // Calculate bounds
        Vector3 simulationBoundsMin = simulationBounds.bounds.min;
        Vector3 simulationBoundsMax = simulationBounds.bounds.max;
        
        // Get fluid initial bounds from the GameObject's transform
        Vector3 fluidInitialBoundsMin = fluidInitialBounds.transform.position - fluidInitialBounds.transform.localScale * 0.5f;
        Vector3 fluidInitialBoundsMax = fluidInitialBounds.transform.position + fluidInitialBounds.transform.localScale * 0.5f;
        
        // Calculate morton code normalization factors on CPU
        Vector3 simulationSize = simulationBoundsMax - simulationBoundsMin;
        
        // Find the longest axis of simulation bounds
        float maxAxis = Mathf.Max(simulationSize.x, Mathf.Max(simulationSize.y, simulationSize.z));
        
        // Calculate normalization factor (2^21 for longest axis)
        float normalizationFactor = Mathf.Pow(2.0f, 21.0f) / maxAxis;
        Vector3 mortonNormalizationFactor = new Vector3(normalizationFactor, normalizationFactor, normalizationFactor);
        
        // Calculate max morton value
        float mortonMaxValue = Mathf.Pow(2.0f, 21.0f) - 1.0f;
        
        // Calculate grid dimensions for even particle distribution
        Vector3 fluidInitialSize = fluidInitialBoundsMax - fluidInitialBoundsMin;
        
        // Calculate grid dimensions that will fit PARTICLE_COUNT particles
        // Use the same major order as morton code (Z, Y, X)
        float volumePerParticle = (fluidInitialSize.x * fluidInitialSize.y * fluidInitialSize.z) / PARTICLE_COUNT;
        float gridSpacing = Mathf.Pow(volumePerParticle, 1.0f / 3.0f);
        
        Vector3Int gridDimensions = new Vector3Int(
            Mathf.Max(1, Mathf.RoundToInt(fluidInitialSize.x / gridSpacing)),
            Mathf.Max(1, Mathf.RoundToInt(fluidInitialSize.y / gridSpacing)),
            Mathf.Max(1, Mathf.RoundToInt(fluidInitialSize.z / gridSpacing))
        );
        
        // Adjust grid dimensions to ensure we don't exceed PARTICLE_COUNT
        while (gridDimensions.x * gridDimensions.y * gridDimensions.z > PARTICLE_COUNT)
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
        fluidKernels.SetVector("simulationBoundsMin", simulationBoundsMin);
        fluidKernels.SetVector("simulationBoundsMax", simulationBoundsMax);
        fluidKernels.SetVector("fluidInitialBoundsMin", fluidInitialBoundsMin);
        fluidKernels.SetVector("fluidInitialBoundsMax", fluidInitialBoundsMax);
        fluidKernels.SetVector("mortonNormalizationFactor", mortonNormalizationFactor);
        fluidKernels.SetFloat("mortonMaxValue", mortonMaxValue);
        
        // Set grid parameters to compute shader
        fluidKernels.SetInts("gridDimensions", new int[] { (int)gridDimensions.x, (int)gridDimensions.y, (int)gridDimensions.z });
        fluidKernels.SetVector("gridSpacing", actualGridSpacing);
        
        // Dispatch the kernel
        int threadGroups = Mathf.CeilToInt(PARTICLE_COUNT / 64.0f);
        fluidKernels.Dispatch(initializeParticlesKernel, threadGroups, 1, 1);
        
        // Debug.Log($"Initialized {PARTICLE_COUNT} particles with bounds: simulation({simulationBoundsMin} to {simulationBoundsMax}), fluid initial({fluidInitialBoundsMin} to {fluidInitialBoundsMax}), morton normalization factor: {mortonNormalizationFactor}, max morton value: {mortonMaxValue}, grid dimensions: {gridDimensions}, grid spacing: {actualGridSpacing}");
        
        // Read back and print first 10 elements of each buffer
        // ReadAndPrintBufferData();
    }

    // Update is called once per frame
    void Update()
    {
        
    }
    
    // Simple radix sort for morton codes
    public void RadixSort()
    {
        if (radixSortShader == null) return;
        
        // No copying needed! Use the existing separate buffers directly
        // mortonCodesBuffer and particleIndicesBuffer are already separate
        
        // Set element count
        radixSortShader.SetInt("elementCount", PARTICLE_COUNT);
        
        // Perform radix sort (32 passes for 32-bit morton codes)
        for (int bit = 0; bit < 32; bit++)
        {
            // Set current bit
            radixSortShader.SetInt("currentBit", bit);
            
            // Set buffers for all kernels - use the main buffers directly
            radixSortShader.SetBuffer(splitPrepKernel, "inputKeys", mortonCodesBuffer);
            radixSortShader.SetBuffer(splitPrepKernel, "eBuffer", eBuffer);
            
            radixSortShader.SetBuffer(scatterElementsKernel, "inputKeys", mortonCodesBuffer);
            radixSortShader.SetBuffer(scatterElementsKernel, "inputIndices", particleIndicesBuffer);
            radixSortShader.SetBuffer(scatterElementsKernel, "outputKeys", outputKeysBuffer);
            radixSortShader.SetBuffer(scatterElementsKernel, "outputIndices", outputIndicesBuffer);
            radixSortShader.SetBuffer(scatterElementsKernel, "eBuffer", eBuffer);
            radixSortShader.SetBuffer(scatterElementsKernel, "fBuffer", fBuffer);
            
            radixSortShader.SetBuffer(copyBufferKernel, "inputKeys", outputKeysBuffer);
            radixSortShader.SetBuffer(copyBufferKernel, "inputIndices", outputIndicesBuffer);
            radixSortShader.SetBuffer(copyBufferKernel, "outputKeys", mortonCodesBuffer);
            radixSortShader.SetBuffer(copyBufferKernel, "outputIndices", particleIndicesBuffer);
            
            // Dispatch kernels
            int threadGroups = Mathf.CeilToInt(PARTICLE_COUNT / 256.0f);
            
            // Step 1: Prepare bit flags
            radixSortShader.Dispatch(splitPrepKernel, threadGroups, 1, 1);
            
            // Step 2: Calculate prefix sums (simple scan)
            CalculatePrefixSum();
            
            // Step 3: Scatter elements
            radixSortShader.Dispatch(scatterElementsKernel, threadGroups, 1, 1);
            
            // Step 4: Copy output back to input for next iteration
            radixSortShader.Dispatch(copyBufferKernel, threadGroups, 1, 1);
        }
        
        // No copying needed! The main buffers (mortonCodesBuffer and particleIndicesBuffer) 
        // are already sorted in place
    }
    
    // Simple prefix sum calculation (CPU-based for now)
    private void CalculatePrefixSum()
    {
        uint[] eData = new uint[PARTICLE_COUNT];
        eBuffer.GetData(eData);
        
        uint[] fData = new uint[PARTICLE_COUNT];
        uint sum = 0;
        for (int i = 0; i < PARTICLE_COUNT; i++)
        {
            fData[i] = sum;
            sum += eData[i];
        }
        
        fBuffer.SetData(fData);
    }
    
    // Verify that the morton codes are properly sorted
    public void VerifySort()
    {
        if (mortonCodesBuffer == null || particleIndicesBuffer == null) return;
        
        // Read morton codes and indices from GPU
        uint[] mortonCodes = new uint[PARTICLE_COUNT];
        uint[] particleIndices = new uint[PARTICLE_COUNT];
        mortonCodesBuffer.GetData(mortonCodes);
        particleIndicesBuffer.GetData(particleIndices);
        
        // Check if morton codes are in ascending order
        bool isSorted = true;
        uint firstOutOfOrder = 0;
        uint previousMortonCode = 0;
        
        for (int i = 0; i < PARTICLE_COUNT; i++)
        {
            if (i > 0 && mortonCodes[i] < previousMortonCode)
            {
                isSorted = false;
                firstOutOfOrder = (uint)i;
                break;
            }
            previousMortonCode = mortonCodes[i];
        }
        
        // Print verification result
        if (isSorted)
        {
            Debug.Log("✅ SORT VERIFICATION: YES - Morton codes are properly sorted");
        }
        else
        {
            Debug.LogError($"❌ SORT VERIFICATION: NO - Morton codes are NOT sorted. First out-of-order element at index {firstOutOfOrder}");
            Debug.LogError($"   Previous morton code: {mortonCodes[firstOutOfOrder - 1]}, Current: {mortonCodes[firstOutOfOrder]}");
        }
        
        // Additional verification: Check that all particle indices are still present
        bool allIndicesPresent = true;
        bool[] indexFound = new bool[PARTICLE_COUNT];
        
        for (int i = 0; i < PARTICLE_COUNT; i++)
        {
            if (particleIndices[i] >= PARTICLE_COUNT)
            {
                allIndicesPresent = false;
                Debug.LogError($"❌ INDEX VERIFICATION: NO - Invalid particle index {particleIndices[i]} at position {i}");
                break;
            }
            indexFound[particleIndices[i]] = true;
        }
        
        if (allIndicesPresent)
        {
            // Check for missing indices
            for (int i = 0; i < PARTICLE_COUNT; i++)
            {
                if (!indexFound[i])
                {
                    allIndicesPresent = false;
                    Debug.LogError($"❌ INDEX VERIFICATION: NO - Missing particle index {i}");
                    break;
                }
            }
        }
        
        if (allIndicesPresent)
        {
            Debug.Log("✅ INDEX VERIFICATION: YES - All particle indices are present and valid");
        }
    }
    
    // Simple debug visualization using Gizmos
    private void OnDrawGizmos()
    {
        if (particlesBuffer == null) return;
        
        // Read particle data back to CPU for visualization
        Particle[] particles = new Particle[PARTICLE_COUNT];
        particlesBuffer.GetData(particles);
        
        // Set gizmo color to blue
        Gizmos.color = Color.blue;
        
        // Draw each particle as a small sphere
        for (int i = 0; i < PARTICLE_COUNT; i++)
        {
            float size = Mathf.Pow(8.0f, particles[i].layer) * 0.002f; // Scale down the size
            Gizmos.DrawSphere(particles[i].position, size);
        }
    }
    
    private void ReadAndPrintBufferData()
    {
        const int elementsToRead = 10;
        
        // Read particles buffer
        Particle[] particles = new Particle[elementsToRead];
        particlesBuffer.GetData(particles, 0, 0, elementsToRead);
        
        // Read morton codes buffer
        MortonCode[] mortonCodes = new MortonCode[elementsToRead];
        mortonCodesBuffer.GetData(mortonCodes, 0, 0, elementsToRead);
        
        // Read node flags buffer (packed: 00(unique)(active))
        uint[] nodeFlags = new uint[elementsToRead];
        nodeFlagsBuffer.GetData(nodeFlags, 0, 0, elementsToRead);
        
        // Print buffer data
        Debug.Log("=== Buffer Data (First 10 Elements) ===");
        
        for (int i = 0; i < elementsToRead; i++)
        {
            Debug.Log($"Index {i}:");
            Debug.Log($"  Particle: pos=({particles[i].position.x:F3}, {particles[i].position.y:F3}, {particles[i].position.z:F3}), " +
                     $"vel=({particles[i].velocity.x:F3}, {particles[i].velocity.y:F3}, {particles[i].velocity.z:F3}), " +
                     $"layer={particles[i].layer}, mortonCode={particles[i].mortonCode}");
            Debug.Log($"  MortonCode: mortonCode={mortonCodes[i].mortonCode}, index={mortonCodes[i].index}");
            // Unpack bit flags: 00(unique)(active)
            bool isActive = (nodeFlags[i] & 0x1) != 0;      // Bit 0: active
            bool isUnique = (nodeFlags[i] & 0x2) != 0;      // Bit 1: unique
            Debug.Log($"  NodeFlags: {nodeFlags[i]} (Active: {isActive}, Unique: {isUnique})");
        }
    }
    
    // MortonCode struct to match compute shader
    private struct MortonCode
    {
        public uint mortonCode;
        public uint index;
    }
    
    private void OnDestroy()
    {
        // Clean up buffers
        particlesBuffer?.Release();
        mortonCodesBuffer?.Release();
        particleIndicesBuffer?.Release();
        nodeFlagsBuffer?.Release();
        
        // Clean up radix sort buffers
        inputKeysBuffer?.Release();
        inputIndicesBuffer?.Release();
        outputKeysBuffer?.Release();
        outputIndicesBuffer?.Release();
        eBuffer?.Release();
        fBuffer?.Release();
    }
}
