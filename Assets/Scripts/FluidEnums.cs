using UnityEngine;

public enum RenderingMode
{
    Particles,
    ParticlesVelocity,
    Nodes,
    Depth,
    Thickness,
    BlurredDepth,
    BlurredThickness,
    Normal,
    Composite
}

public enum PreconditionerType
{
    None,
    Neural,
    Jacobi
}

/// <summary>How PCG applies the pressure Laplacian each iteration. CSR is still built when <see cref="PreconditionerType.Neural"/> or the training recorder is capturing frames (needs edge lists).</summary>
public enum PressureSolveMatrixFormat
{
    /// <summary><c>ApplyMatrixAndDot</c> on SoA stencil; skips CSR build if nothing else needs it.</summary>
    Structured,
    /// <summary><c>SpMV_CSR</c>; builds CSR every solve.</summary>
    Csr
}

public enum ThicknessSource
{
    Nodes,
    Particles
}

public struct faceVelocities
{
    public float left;
    public float right;
    public float bottom;
    public float top;
    public float front;
    public float back;
}

// Particle struct (must match compute shader)
public struct Particle
{
    public Vector3 position;    // 12 bytes
    public Vector3 velocity;    // 12 bytes
    public uint mortonCode;     // 4 bytes
}

// Node struct (must match compute shader)
public struct Node
{
    public Vector3 position;    // 12 bytes
    public Vector3 velocity;    // 12 bytes
    public faceVelocities velocities; // 6*4 = 24 bytes
    public float mass;             // 4 bytes
    public uint layer;          // 4 bytes
    public uint mortonCode;     // 4 bytes
    public uint active;         // 4 bytes
}
