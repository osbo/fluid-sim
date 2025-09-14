using UnityEngine;
using System;

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
    
    // GPU Buffers
    private ComputeBuffer particlesBuffer;
    private ComputeBuffer mortonCodesBuffer;
    private ComputeBuffer particleIndicesBuffer;
    private ComputeBuffer nodeFlagsBuffer; // Packed: 00(unique)(active)
    
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
    }
    
    private void InitializeParticleSystem()
    {
        // Get kernel indices
        initializeParticlesKernel = fluidKernels.FindKernel("InitializeParticles");
        
        // Create buffers
        particlesBuffer = new ComputeBuffer(PARTICLE_COUNT, sizeof(float) * 3 + sizeof(float) * 3 + sizeof(uint) + sizeof(uint)); // 32 bytes
        mortonCodesBuffer = new ComputeBuffer(PARTICLE_COUNT, sizeof(uint));
        particleIndicesBuffer = new ComputeBuffer(PARTICLE_COUNT, sizeof(uint));
        nodeFlagsBuffer = new ComputeBuffer(PARTICLE_COUNT, sizeof(uint)); // 4 bytes (packed flags)
        
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
    }

    // Update is called once per frame
    void Update()
    {
        
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
        uint[] mortonCodes = new uint[elementsToRead];
        mortonCodesBuffer.GetData(mortonCodes, 0, 0, elementsToRead);

        // Read particle indices buffer
        uint[] particleIndices = new uint[elementsToRead];
        particleIndicesBuffer.GetData(particleIndices, 0, 0, elementsToRead);

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
            Debug.Log($"  Morton Code: {mortonCodes[i]}, Particle Index: {particleIndices[i]}");
            // Unpack bit flags: 00(unique)(active)
            bool isActive = (nodeFlags[i] & 0x1) != 0;      // Bit 0: active
            bool isUnique = (nodeFlags[i] & 0x2) != 0;      // Bit 1: unique
            Debug.Log($"  NodeFlags: {nodeFlags[i]} (Active: {isActive}, Unique: {isUnique})");
        }
    }
    
    private void OnDestroy()
    {
        // Clean up buffers
        particlesBuffer?.Release();
        mortonCodesBuffer?.Release();
        particleIndicesBuffer?.Release();
        nodeFlagsBuffer?.Release();
    }
}