using UnityEngine;
using System.IO;
using System;

public class TrainingDataRecorder : MonoBehaviour
{
    [Header("Recording Settings")]
    [Tooltip("When checked, begins a new run folder and saves up to Max Frames counting from this moment. Uncheck anytime to stop without saving the current solver step.")]
    public bool record = true;
    public int maxFrames = 600;
    [Tooltip("Persist one snapshot every N solver steps while recording (1 = every step).")]
    [Min(1)]
    public int saveEvery = 1;
    public string baseFolder = "SimulationData";
    
    public bool isRecording { get; private set; } = false;
    private string currentRunFolder;
    private int frameIndex = 0;
    /// <summary>Solver steps since this run started; used with <see cref="saveEvery"/>.</summary>
    private int recordSimStep;
    private byte[] saveBytesScratch;
    /// <summary>Last serialized <see cref="record"/>; used to detect Inspector toggles so max-frames stop does not auto-start a new run.</summary>
    private bool recordPrev;

    // Struct matching your Compute Shader Node exactly for byte-alignment
    // Node struct: Vector3 position (12) + Vector3 velocity (12) + faceVelocities (24) + float mass (4) + uint layer (4) + uint mortonCode (4) + uint active (4) = 64 bytes total
    [System.Serializable]
    public struct NodeData // 64 bytes total
    {
        public Vector3 position;    // 12
        public Vector3 velocity;    // 12
        // Face velocities (6 floats)
        public float left, right, bottom, top, front, back; // 24
        public float mass;          // 4
        public uint layer;          // 4
        public uint mortonCode;     // 4
        public uint active;         // 4
    }

    void OnEnable()
    {
        recordPrev = record;
    }

    void Update()
    {
        if (record == recordPrev)
            return;
        if (record)
            StartNewRun();
        else
            StopRecording();
        recordPrev = record;
    }

    public void StartNewRun()
    {
        if (!record)
        {
            isRecording = false;
            return;
        }

        StopRecording();

        string timestamp = System.DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss");
        currentRunFolder = Path.Combine(Application.streamingAssetsPath, baseFolder, "Run_" + timestamp);
        Directory.CreateDirectory(currentRunFolder);
        frameIndex = 0;
        recordSimStep = 0;
        isRecording = true;
        Debug.Log($"Started recording to: {currentRunFolder} (max {maxFrames} frames from this moment)");
    }

    public void SaveFrame(ComputeBuffer nodes, ComputeBuffer neighbors, ComputeBuffer divergence, ComputeBuffer pressure,
        ComputeBuffer nodeDensity, ComputeBuffer diffusionGradient, ComputeBuffer rowIndices, ComputeBuffer colIndices, ComputeBuffer csrValues, int totalNNZ,
        int numNodes, int minLayer, int maxLayer, float gravity, int numParticles, int maxCgIterations, float convergenceThreshold, float frameRate,
        Vector3 simulationBoundsMin, Vector3 simulationBoundsMax, Vector3 fluidInitialBoundsMin, Vector3 fluidInitialBoundsMax,
        int neighborUintsPerNode = 24)
    {
        if (!isRecording)
            return;
        // Inspector cleared Record before Update: do not write this timestep.
        if (!record)
        {
            StopRecording();
            return;
        }

        // Check if we've reached the max frame count before saving
        if (frameIndex >= maxFrames)
        {
            StopRecording();
            return;
        }

        recordSimStep++;
        int every = Mathf.Max(1, saveEvery);
        if ((recordSimStep - 1) % every != 0)
            return;

        string framePath = Path.Combine(currentRunFolder, $"frame_{frameIndex:D4}");
        Directory.CreateDirectory(framePath);

        // 1. Save Metadata
        string metadata = $"numNodes: {numNodes}\n" +
                         $"saveEvery: {every}\n" +
                         $"recordingSimStep: {recordSimStep}\n" +
                         $"minLayer: {minLayer}\n" +
                         $"maxLayer: {maxLayer}\n" +
                         $"gravity: {gravity}\n" +
                         $"numParticles: {numParticles}\n" +
                         $"maxCgIterations: {maxCgIterations}\n" +
                         $"convergenceThreshold: {convergenceThreshold}\n" +
                         $"frameRate: {frameRate}\n" +
                         $"simulationBoundsMin: {simulationBoundsMin.x} {simulationBoundsMin.y} {simulationBoundsMin.z}\n" +
                         $"simulationBoundsMax: {simulationBoundsMax.x} {simulationBoundsMax.y} {simulationBoundsMax.z}\n" +
                         $"fluidInitialBoundsMin: {fluidInitialBoundsMin.x} {fluidInitialBoundsMin.y} {fluidInitialBoundsMin.z}\n" +
                         $"fluidInitialBoundsMax: {fluidInitialBoundsMax.x} {fluidInitialBoundsMax.y} {fluidInitialBoundsMax.z}\n";
        
        File.WriteAllText(Path.Combine(framePath, "meta.txt"), metadata);

        // 2. Save Buffers directly as raw bytes
        // Node buffer: 64 bytes per node
        // Structure: Vector3 position (12) + Vector3 velocity (12) + 6 face velocities (24) + float mass (4) + uint layer (4) + uint mortonCode (4) + uint active (4) = 64 bytes
        SaveBuffer(nodes, numNodes, 64, Path.Combine(framePath, "nodes.bin"));
        SaveBuffer(neighbors, numNodes, neighborUintsPerNode * sizeof(uint), Path.Combine(framePath, "neighbors.bin"));
        // Divergence buffer: 1 float per node = 4 bytes
        SaveBuffer(divergence, numNodes, sizeof(float), Path.Combine(framePath, "divergence.bin"));
        // Pressure buffer: 1 float per node = 4 bytes
        SaveBuffer(pressure, numNodes, sizeof(float), Path.Combine(framePath, "pressure.bin"));
        // P2G-averaged fluid density per node (multiphase); 1 float per node = 4 bytes
        if (nodeDensity != null)
            SaveBuffer(nodeDensity, numNodes, sizeof(float), Path.Combine(framePath, "node_density.bin"));
        // Diffusion gradient: float3 per node = 12 bytes
        if (diffusionGradient != null)
            SaveBuffer(diffusionGradient, numNodes, 3 * sizeof(float), Path.Combine(framePath, "diffusion_gradient.bin"));

        // Save the Sparse Graph (Edge Index)
        // We save exactly totalNNZ elements. Stride is sizeof(uint) = 4.
        SaveBuffer(rowIndices, totalNNZ, sizeof(uint), Path.Combine(framePath, "edge_index_rows.bin"));
        SaveBuffer(colIndices, totalNNZ, sizeof(uint), Path.Combine(framePath, "edge_index_cols.bin"));

        // Save the exact physics weights "A" (matrix values)
        SaveBuffer(csrValues, totalNNZ, sizeof(float), Path.Combine(framePath, "A_values.bin"));

        Debug.Log($"Saved frame {frameIndex}");
        frameIndex++;
        
        // Check if we've reached max frames after incrementing
        if (frameIndex >= maxFrames)
        {
            StopRecording();
        }
    }

    private void SaveBuffer(ComputeBuffer buffer, int count, int stride, string path)
    {
        // For data generation, blocking GetData is acceptable to ensure sync.
        // Use a temp array to read. 
        // OPTIMIZATION: In production, reuse these arrays instead of allocating new ones every frame.
        
        if (buffer == null)
        {
            Debug.LogError($"Cannot save buffer: buffer is null for path {path}");
            return;
        }

        if (buffer.count < count)
        {
            Debug.LogWarning($"Buffer count ({buffer.count}) is less than requested count ({count}). Reading only {buffer.count} elements.");
            count = buffer.count;
        }

        int nBytes = count * stride;
        if (saveBytesScratch == null || saveBytesScratch.Length < nBytes)
            saveBytesScratch = new byte[Math.Max(nBytes, saveBytesScratch?.Length * 2 ?? 65536)];

        buffer.GetData(saveBytesScratch, 0, 0, nBytes);
        using (var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Read))
            fs.Write(saveBytesScratch, 0, nBytes);
    }

    public void StopRecording()
    {
        if (!isRecording)
            return;

        isRecording = false;
        Debug.Log($"Stopped recording. Total frames saved: {frameIndex}/{maxFrames}");
    }
}

