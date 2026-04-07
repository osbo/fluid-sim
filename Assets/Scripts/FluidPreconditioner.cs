using UnityEngine;

// Preconditioner hooks for PCG. Legacy sparse-G / V-cycle path removed; LeafOnly matrix-free apply TBD.
public partial class FluidSimulator : MonoBehaviour
{
    private static bool warnedNeuralPackedFallback;

    private void ApplyPreconditioner(ComputeBuffer r, ComputeBuffer z_out, int kJacobi)
    {
        if (preconditioner == PreconditionerType.None)
        {
            CopyBuffer(r, z_out);
            return;
        }

        if (preconditioner == PreconditionerType.Neural)
        {
            if (TryDispatchLeafOnlyPrecondPackedApply(r, z_out))
                return;
            if (!warnedNeuralPackedFallback)
            {
                warnedNeuralPackedFallback = true;
                Debug.Log(
                    "PreconditionerType.Neural: LeafOnly packed GPU apply did not run (Editor auto-loads LeafOnlyPrecondApply.compute; " +
                    "packed data uploads after layer-1 forward; assign shader in player builds; weights + checkpoint layout must match). " +
                    "Falling back to Jacobi. Set preconditioner to Jacobi to silence.");
            }
        }

        if (preconditioner == PreconditionerType.Jacobi || preconditioner == PreconditionerType.Neural)
        {
            if (kJacobi >= 0 && matrixABuffer != null)
            {
                cgSolverShader.SetBuffer(kJacobi, "xBuffer", r);
                cgSolverShader.SetBuffer(kJacobi, "yBuffer", z_out);
                cgSolverShader.SetBuffer(kJacobi, "matrixABuffer", matrixABuffer);
                cgSolverShader.SetInt("numNodes", numNodes);
                Dispatch(kJacobi, numNodes);
            }
            else
            {
                CopyBuffer(r, z_out);
            }
            return;
        }

        CopyBuffer(r, z_out);
    }

    private void ApplyPreconditionerPcgIterationGpu(ComputeBuffer r, ComputeBuffer z_out, int kJacobi)
    {
        ApplyPreconditioner(r, z_out, kJacobi);
        GpuDotProductReduceNoReadback(r, z_out);
    }

    /// <summary>Same as <see cref="ApplyPreconditionerPcgIterationGpu"/> but uses <see cref="cgPcgIndirectArgsBuffer"/> for 512-thread and copy paths (PCG indirect early-out).</summary>
    private void ApplyPreconditionerPcgIterationGpuIndirect(ComputeBuffer r, ComputeBuffer z_out, int kJacobi, int groups512Uniform)
    {
        if (preconditioner == PreconditionerType.None)
        {
            GpuCopyBufferIndirect(r, z_out);
        }
        else if (preconditioner == PreconditionerType.Jacobi)
        {
            if (kJacobi >= 0 && matrixABuffer != null)
            {
                cgSolverShader.SetBuffer(kJacobi, "xBuffer", r);
                cgSolverShader.SetBuffer(kJacobi, "yBuffer", z_out);
                cgSolverShader.SetBuffer(kJacobi, "matrixABuffer", matrixABuffer);
                cgSolverShader.SetInt("numNodes", numNodes);
                cgSolverShader.DispatchIndirect(kJacobi, cgPcgIndirectArgsBuffer, CgIndirectArgsOffsetVec512);
            }
            else
                GpuCopyBufferIndirect(r, z_out);
        }
        else
            GpuCopyBufferIndirect(r, z_out);

        GpuDotProductReduceNoReadbackIndirect(r, z_out, groups512Uniform);
    }

    private void ApplyPreconditionerInitStoreRhoGpu(ComputeBuffer r, ComputeBuffer z_out, int kJacobi)
    {
        ApplyPreconditionerPcgIterationGpu(r, z_out, kJacobi);
        DispatchStoreRhoFromDot();
    }

    private void ReleasePreconditionerBuffers()
    {
        zVectorBuffer?.Release();
    }
}
