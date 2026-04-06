using UnityEngine;

// Preconditioner hooks for PCG. Legacy sparse-G / V-cycle path removed; LeafOnly matrix-free apply TBD.
public partial class FluidSimulator : MonoBehaviour
{
    private static bool warnedNeuralFallback;

    private void ApplyPreconditioner(ComputeBuffer r, ComputeBuffer z_out, int kJacobi)
    {
        if (preconditioner == PreconditionerType.None)
        {
            CopyBuffer(r, z_out);
            return;
        }

        if (preconditioner == PreconditionerType.Neural)
        {
            if (!warnedNeuralFallback)
            {
                warnedNeuralFallback = true;
                Debug.LogWarning(
                    "PreconditionerType.Neural: old sparse-G preconditioner was removed. Using Jacobi until LeafOnly (matrix-free) is integrated.");
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
