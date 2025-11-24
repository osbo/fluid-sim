using UnityEngine;
using System.IO;

public class TrainingDataRecorder : MonoBehaviour
{
    [Header("Recording Settings")]
    public bool record = true;
    public int maxFrames = 600;
    public string baseFolder = "SimulationData";
    
    public bool isRecording { get; private set; } = false;
    private string currentRunFolder;
    private int frameIndex = 0;

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

    public void StartNewRun()
    {
        if (!record)
        {
            isRecording = false;
            return;
        }
        
        string timestamp = System.DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss");
        currentRunFolder = Path.Combine(Application.streamingAssetsPath, baseFolder, "Run_" + timestamp);
        Directory.CreateDirectory(currentRunFolder);
        frameIndex = 0;
        isRecording = true;
        Debug.Log($"Started recording to: {currentRunFolder} (max {maxFrames} frames)");
    }

    public void SaveFrame(ComputeBuffer nodes, ComputeBuffer neighbors, ComputeBuffer divergence, ComputeBuffer pressure, int numNodes,
        int minLayer, int maxLayer, float gravity, int numParticles, int maxCgIterations, float convergenceThreshold, float frameRate,
        Vector3 simulationBoundsMin, Vector3 simulationBoundsMax, Vector3 fluidInitialBoundsMin, Vector3 fluidInitialBoundsMax)
    {
        if (!isRecording || !record) return;
        
        // Check if we've reached the max frame count before saving
        if (frameIndex >= maxFrames)
        {
            StopRecording();
            return;
        }

        string framePath = Path.Combine(currentRunFolder, $"frame_{frameIndex:D4}");
        Directory.CreateDirectory(framePath);

        // 1. Save Metadata
        string metadata = $"numNodes: {numNodes}\n" +
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
        // Neighbors buffer: 24 uints per node = 24 * 4 = 96 bytes
        SaveBuffer(neighbors, numNodes, 24 * sizeof(uint), Path.Combine(framePath, "neighbors.bin"));
        // Divergence buffer: 1 float per node = 4 bytes
        SaveBuffer(divergence, numNodes, sizeof(float), Path.Combine(framePath, "divergence.bin"));
        // Pressure buffer: 1 float per node = 4 bytes
        SaveBuffer(pressure, numNodes, sizeof(float), Path.Combine(framePath, "pressure.bin"));

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

        // We read as bytes to avoid struct marshalling overhead
        byte[] rawBytes = new byte[count * stride];
        buffer.GetData(rawBytes, 0, 0, count * stride);
        File.WriteAllBytes(path, rawBytes);
    }

    public void StopRecording()
    {
        if (!isRecording) return; // Already stopped
        
        isRecording = false;
        Debug.Log($"Stopped recording. Total frames saved: {frameIndex}/{maxFrames}");
    }
}

