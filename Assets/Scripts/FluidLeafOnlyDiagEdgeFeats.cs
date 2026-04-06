using UnityEngine;

/// <summary>
/// GPU build of diagonal dense <c>edge_feats</c> (K, L, L, 4) — <c>build_diag_dense_edge_feats_from_positions</c> with no block column.
/// Channels 0–2: <c>pos[bL+j]-pos[bL+i]</c>; channel 3: mean in-leaf <c>A_ij</c> using CSR values ÷ <c>scale_A</c>
/// (<see cref="LeafOnlyCsrValuesScaledBuffer"/>), matching dataset <c>edge_values</c> and PyTorch <c>build_diag_dense_edge_feats</c>.
/// </summary>
public partial class FluidSimulator : MonoBehaviour
{
    [Tooltip("LeafOnlyDiagEdgeFeats.compute — dense diag edge grid for attention; leave null to skip.")]
    public ComputeShader leafOnlyDiagEdgeFeatsShader;

    private int leafOnlyDiagGeomKernel;
    private int leafOnlyDiagClearScratchKernel;
    private int leafOnlyDiagAccumASerialKernel;
    private int leafOnlyDiagWriteCh3Kernel;

    private ComputeBuffer leafOnlyDiagEdgeFeats;
    private ComputeBuffer leafOnlyDiagASumScratch;
    private ComputeBuffer leafOnlyDiagACountScratch;

    private int leafOnlyDiagCapCells;

    public ComputeBuffer LeafOnlyDiagEdgeFeatsBuffer => leafOnlyDiagEdgeFeats;

    private void InitLeafOnlyDiagEdgeFeatsKernels()
    {
        if (leafOnlyDiagEdgeFeatsShader == null)
            return;
        leafOnlyDiagGeomKernel = leafOnlyDiagEdgeFeatsShader.FindKernel("LeafOnly_DiagEdgeFeatGeom");
        leafOnlyDiagClearScratchKernel = leafOnlyDiagEdgeFeatsShader.FindKernel("LeafOnly_DiagEdgeFeatClearScratch");
        leafOnlyDiagAccumASerialKernel = leafOnlyDiagEdgeFeatsShader.FindKernel("LeafOnly_DiagEdgeFeatAccumA_Serial");
        leafOnlyDiagWriteCh3Kernel = leafOnlyDiagEdgeFeatsShader.FindKernel("LeafOnly_DiagEdgeFeatWriteCh3");
    }

    private void EnsureLeafOnlyDiagEdgeFeatsBuffers(int numLeaves, int leafSize)
    {
        int cells = numLeaves * leafSize * leafSize;
        int outFloats = cells * 4;
        if (leafOnlyDiagEdgeFeats != null && leafOnlyDiagCapCells >= cells)
            return;

        leafOnlyDiagEdgeFeats?.Release();
        leafOnlyDiagASumScratch?.Release();
        leafOnlyDiagACountScratch?.Release();

        leafOnlyDiagEdgeFeats = new ComputeBuffer(outFloats, sizeof(float));
        leafOnlyDiagASumScratch = new ComputeBuffer(cells, sizeof(float));
        leafOnlyDiagACountScratch = new ComputeBuffer(cells, sizeof(uint));
        leafOnlyDiagCapCells = cells;
    }

    private static int LeafOnlyDiagGroups256(int total)
    {
        return Mathf.Max(1, Mathf.CeilToInt(total / 256f));
    }

    private void BindLeafOnlyDiagEdgeFeatsCommon(int kernel)
    {
        leafOnlyDiagEdgeFeatsShader.SetBuffer(kernel, "leafXPacked", leafOnlyXPacked);
        leafOnlyDiagEdgeFeatsShader.SetBuffer(kernel, "leafDiagEdgeFeatsOut", leafOnlyDiagEdgeFeats);
    }

    /// <summary>Build diag <c>edge_feats</c> on GPU; log <c>diag_edge_feats</c> parity (full tensor stats + head).</summary>
    private void DispatchLeafOnlyDiagEdgeFeatsAndLogParity(int nPad, int csrNnz)
    {
        const string phase = "unity";
        if (leafOnlyDiagEdgeFeatsShader == null)
            return;
        if (leafOnlyXPacked == null || csrRowIndices == null || csrColIndices == null || leafOnlyCsrValuesScaled == null)
            return;

        int L = LeafOnlyLeafSize;
        if (nPad <= 0 || (nPad % L) != 0)
            return;

        int K = nPad / L;
        EnsureLeafOnlyDiagEdgeFeatsBuffers(K, L);

        leafOnlyDiagEdgeFeatsShader.SetInt("leafNPadded", nPad);
        leafOnlyDiagEdgeFeatsShader.SetInt("leafLeafSize", L);
        leafOnlyDiagEdgeFeatsShader.SetInt("leafNumLeaves", K);
        leafOnlyDiagEdgeFeatsShader.SetInt("leafCsrNnz", Mathf.Max(0, csrNnz));

        int cells = K * L * L;

        BindLeafOnlyDiagEdgeFeatsCommon(leafOnlyDiagGeomKernel);
        leafOnlyDiagEdgeFeatsShader.Dispatch(leafOnlyDiagGeomKernel, LeafOnlyDiagGroups256(cells), 1, 1);

        leafOnlyDiagEdgeFeatsShader.SetBuffer(leafOnlyDiagClearScratchKernel, "leafDiagA_Sum", leafOnlyDiagASumScratch);
        leafOnlyDiagEdgeFeatsShader.SetBuffer(leafOnlyDiagClearScratchKernel, "leafDiagA_Count", leafOnlyDiagACountScratch);
        leafOnlyDiagEdgeFeatsShader.Dispatch(leafOnlyDiagClearScratchKernel, LeafOnlyDiagGroups256(cells), 1, 1);

        leafOnlyDiagEdgeFeatsShader.SetBuffer(leafOnlyDiagAccumASerialKernel, "leafDiagCsrRows", csrRowIndices);
        leafOnlyDiagEdgeFeatsShader.SetBuffer(leafOnlyDiagAccumASerialKernel, "leafDiagCsrCols", csrColIndices);
        leafOnlyDiagEdgeFeatsShader.SetBuffer(leafOnlyDiagAccumASerialKernel, "leafDiagCsrVals", leafOnlyCsrValuesScaled);
        leafOnlyDiagEdgeFeatsShader.SetBuffer(leafOnlyDiagAccumASerialKernel, "leafDiagA_Sum", leafOnlyDiagASumScratch);
        leafOnlyDiagEdgeFeatsShader.SetBuffer(leafOnlyDiagAccumASerialKernel, "leafDiagA_Count", leafOnlyDiagACountScratch);
        leafOnlyDiagEdgeFeatsShader.Dispatch(leafOnlyDiagAccumASerialKernel, 1, 1, 1);

        leafOnlyDiagEdgeFeatsShader.SetBuffer(leafOnlyDiagWriteCh3Kernel, "leafDiagEdgeFeatsOut", leafOnlyDiagEdgeFeats);
        leafOnlyDiagEdgeFeatsShader.SetBuffer(leafOnlyDiagWriteCh3Kernel, "leafDiagA_Sum", leafOnlyDiagASumScratch);
        leafOnlyDiagEdgeFeatsShader.SetBuffer(leafOnlyDiagWriteCh3Kernel, "leafDiagA_Count", leafOnlyDiagACountScratch);
        leafOnlyDiagEdgeFeatsShader.Dispatch(leafOnlyDiagWriteCh3Kernel, LeafOnlyDiagGroups256(cells), 1, 1);

        int nOut = cells * 4;
        var data = new float[nOut];
        leafOnlyDiagEdgeFeats.GetData(data, 0, 0, nOut);
        LeafOnlyParitySummarize(phase, "diag_edge_feats", data, nOut);
        LeafOnlyParityHead(phase, "diag_edge_feats", data, nOut, 32);
    }

    private void ReleaseLeafOnlyDiagEdgeFeatsBuffers()
    {
        leafOnlyDiagEdgeFeats?.Release();
        leafOnlyDiagEdgeFeats = null;
        leafOnlyDiagASumScratch?.Release();
        leafOnlyDiagASumScratch = null;
        leafOnlyDiagACountScratch?.Release();
        leafOnlyDiagACountScratch = null;
        leafOnlyDiagCapCells = 0;
    }
}
