using UnityEngine;

/// <summary>
/// GPU dense off-diagonal H-tile <c>edge_feats</c> <c>(M, L, L, 4)</c> —
/// <c>build_hmatrix_off_dense_rpe_from_positions</c> with <c>with_block_key_column=False</c>.
/// Channels 0–2: mean column-strip pos minus mean row-strip pos per local <c>(i,j)</c>;
/// channel 3: mean normalized <c>A_ij</c> over CSR edges in the tile strips (÷ <c>scale_A</c>).
/// </summary>
public partial class FluidSimulator : MonoBehaviour
{
    [Tooltip("LeafOnlyOffEdgeFeats.compute — dense off H-tile grid; leave null to skip.")]
    public ComputeShader leafOnlyOffEdgeFeatsShader;

    private int leafOnlyOffGeomKernel;
    private int leafOnlyOffClearScratchKernel;
    private int leafOnlyOffAccumASerialKernel;
    private int leafOnlyOffWriteCh3Kernel;

    private ComputeBuffer leafOnlyOffEdgeFeats;
    private ComputeBuffer leafOnlyOffASumScratch;
    private ComputeBuffer leafOnlyOffACountScratch;
    private ComputeBuffer leafOnlyHMOffR0;
    private ComputeBuffer leafOnlyHMOffC0;
    private ComputeBuffer leafOnlyHMOffS;

    private int leafOnlyOffCapCells;
    private int leafOnlyOffCapM;

    public ComputeBuffer LeafOnlyOffEdgeFeatsBuffer => leafOnlyOffEdgeFeats;

    private void InitLeafOnlyOffEdgeFeatsKernels()
    {
        if (leafOnlyOffEdgeFeatsShader == null)
            return;
        leafOnlyOffGeomKernel = leafOnlyOffEdgeFeatsShader.FindKernel("LeafOnly_OffEdgeFeatGeom");
        leafOnlyOffClearScratchKernel = leafOnlyOffEdgeFeatsShader.FindKernel("LeafOnly_OffEdgeFeatClearScratch");
        leafOnlyOffAccumASerialKernel = leafOnlyOffEdgeFeatsShader.FindKernel("LeafOnly_OffEdgeFeatAccumA_Serial");
        leafOnlyOffWriteCh3Kernel = leafOnlyOffEdgeFeatsShader.FindKernel("LeafOnly_OffEdgeFeatWriteCh3");
    }

    private static int LeafOnlyOffGroups256(int total)
    {
        return Mathf.Max(1, Mathf.CeilToInt(total / 256f));
    }

    private void EnsureLeafOnlyOffEdgeFeatsBuffers(int numLeaves, int leafSize)
    {
        int mTiles = LeafOnlyHMatrixStatic.NumOffBlocks;
        int cells = mTiles * leafSize * leafSize;
        int outFloats = cells * 4;
        if (leafOnlyOffEdgeFeats != null && leafOnlyOffCapCells >= cells && leafOnlyOffCapM >= mTiles)
            return;

        leafOnlyOffEdgeFeats?.Release();
        leafOnlyOffASumScratch?.Release();
        leafOnlyOffACountScratch?.Release();
        leafOnlyHMOffR0?.Release();
        leafOnlyHMOffC0?.Release();
        leafOnlyHMOffS?.Release();

        leafOnlyOffEdgeFeats = new ComputeBuffer(outFloats, sizeof(float));
        leafOnlyOffASumScratch = new ComputeBuffer(cells, sizeof(float));
        leafOnlyOffACountScratch = new ComputeBuffer(cells, sizeof(uint));
        leafOnlyHMOffR0 = new ComputeBuffer(mTiles, sizeof(uint));
        leafOnlyHMOffC0 = new ComputeBuffer(mTiles, sizeof(uint));
        leafOnlyHMOffS = new ComputeBuffer(mTiles, sizeof(uint));

        var r = new uint[mTiles];
        var c = new uint[mTiles];
        var s = new uint[mTiles];
        var sr = LeafOnlyHMatrixStatic.OffR0;
        var sc = LeafOnlyHMatrixStatic.OffC0;
        var ss = LeafOnlyHMatrixStatic.OffS;
        for (int i = 0; i < mTiles; i++)
        {
            r[i] = (uint)sr[i];
            c[i] = (uint)sc[i];
            s[i] = (uint)ss[i];
        }

        leafOnlyHMOffR0.SetData(r);
        leafOnlyHMOffC0.SetData(c);
        leafOnlyHMOffS.SetData(s);

        leafOnlyOffCapCells = cells;
        leafOnlyOffCapM = mTiles;
    }

    private void BindLeafOnlyOffEdgeFeatsHm(int kernel)
    {
        leafOnlyOffEdgeFeatsShader.SetBuffer(kernel, "leafOffHMR0", leafOnlyHMOffR0);
        leafOnlyOffEdgeFeatsShader.SetBuffer(kernel, "leafOffHMC0", leafOnlyHMOffC0);
        leafOnlyOffEdgeFeatsShader.SetBuffer(kernel, "leafOffHMS", leafOnlyHMOffS);
    }

    /// <summary>Build off dense <c>edge_feats</c>; log parity (meta, full stats, head + mid + tail slices).</summary>
    private void DispatchLeafOnlyOffEdgeFeatsAndLogParity(int nPad, int csrNnz)
    {
        const string phase = "unity";
        if (leafOnlyOffEdgeFeatsShader == null)
            return;
        if (leafOnlyXPacked == null || csrRowIndices == null || csrColIndices == null || leafOnlyCsrValuesScaled == null)
            return;

        int L = LeafOnlyLeafSize;
        if (nPad <= 0 || (nPad % L) != 0)
            return;

        int K = nPad / L;
        int M = LeafOnlyHMatrixStatic.NumOffBlocks;
        EnsureLeafOnlyOffEdgeFeatsBuffers(K, L);

        leafOnlyOffEdgeFeatsShader.SetInt("leafNPadded", nPad);
        leafOnlyOffEdgeFeatsShader.SetInt("leafLeafSize", L);
        leafOnlyOffEdgeFeatsShader.SetInt("leafNumLeaves", K);
        leafOnlyOffEdgeFeatsShader.SetInt("leafNumOffTiles", M);
        leafOnlyOffEdgeFeatsShader.SetInt("leafCsrNnz", Mathf.Max(0, csrNnz));

        int cells = M * L * L;

        leafOnlyOffEdgeFeatsShader.SetBuffer(leafOnlyOffGeomKernel, "leafXPacked", leafOnlyXPacked);
        leafOnlyOffEdgeFeatsShader.SetBuffer(leafOnlyOffGeomKernel, "leafOffEdgeFeatsOut", leafOnlyOffEdgeFeats);
        BindLeafOnlyOffEdgeFeatsHm(leafOnlyOffGeomKernel);
        leafOnlyOffEdgeFeatsShader.Dispatch(leafOnlyOffGeomKernel, LeafOnlyOffGroups256(cells), 1, 1);

        leafOnlyOffEdgeFeatsShader.SetBuffer(leafOnlyOffClearScratchKernel, "leafOffA_Sum", leafOnlyOffASumScratch);
        leafOnlyOffEdgeFeatsShader.SetBuffer(leafOnlyOffClearScratchKernel, "leafOffA_Count", leafOnlyOffACountScratch);
        BindLeafOnlyOffEdgeFeatsHm(leafOnlyOffClearScratchKernel);
        leafOnlyOffEdgeFeatsShader.Dispatch(leafOnlyOffClearScratchKernel, LeafOnlyOffGroups256(cells), 1, 1);

        leafOnlyOffEdgeFeatsShader.SetBuffer(leafOnlyOffAccumASerialKernel, "leafOffCsrRows", csrRowIndices);
        leafOnlyOffEdgeFeatsShader.SetBuffer(leafOnlyOffAccumASerialKernel, "leafOffCsrCols", csrColIndices);
        leafOnlyOffEdgeFeatsShader.SetBuffer(leafOnlyOffAccumASerialKernel, "leafOffCsrVals", leafOnlyCsrValuesScaled);
        leafOnlyOffEdgeFeatsShader.SetBuffer(leafOnlyOffAccumASerialKernel, "leafOffA_Sum", leafOnlyOffASumScratch);
        leafOnlyOffEdgeFeatsShader.SetBuffer(leafOnlyOffAccumASerialKernel, "leafOffA_Count", leafOnlyOffACountScratch);
        BindLeafOnlyOffEdgeFeatsHm(leafOnlyOffAccumASerialKernel);
        leafOnlyOffEdgeFeatsShader.Dispatch(leafOnlyOffAccumASerialKernel, 1, 1, 1);

        leafOnlyOffEdgeFeatsShader.SetBuffer(leafOnlyOffWriteCh3Kernel, "leafOffEdgeFeatsOut", leafOnlyOffEdgeFeats);
        leafOnlyOffEdgeFeatsShader.SetBuffer(leafOnlyOffWriteCh3Kernel, "leafOffA_Sum", leafOnlyOffASumScratch);
        leafOnlyOffEdgeFeatsShader.SetBuffer(leafOnlyOffWriteCh3Kernel, "leafOffA_Count", leafOnlyOffACountScratch);
        BindLeafOnlyOffEdgeFeatsHm(leafOnlyOffWriteCh3Kernel);
        leafOnlyOffEdgeFeatsShader.Dispatch(leafOnlyOffWriteCh3Kernel, LeafOnlyOffGroups256(cells), 1, 1);

        int nOut = cells * 4;
        Debug.Log($"[LeafOnlyParity] phase={phase} tensor=off_edge_feats_meta M_off={M} L={L} n={nOut}");

        var data = new float[nOut];
        leafOnlyOffEdgeFeats.GetData(data, 0, 0, nOut);
        LeafOnlyParitySummarize(phase, "off_edge_feats", data, nOut);
        LeafOnlyParityHead(phase, "off_edge_feats", data, nOut, 32);
        int mid = (nOut / 2 / 4) * 4;
        LeafOnlyParityHeadAt(phase, "off_edge_feats", data, nOut, mid, 48);
        int tailStart = Mathf.Max(0, nOut - 32);
        LeafOnlyParityHeadAt(phase, "off_edge_feats", data, nOut, tailStart, 32);
    }

    private void ReleaseLeafOnlyOffEdgeFeatsBuffers()
    {
        leafOnlyOffEdgeFeats?.Release();
        leafOnlyOffEdgeFeats = null;
        leafOnlyOffASumScratch?.Release();
        leafOnlyOffASumScratch = null;
        leafOnlyOffACountScratch?.Release();
        leafOnlyOffACountScratch = null;
        leafOnlyHMOffR0?.Release();
        leafOnlyHMOffR0 = null;
        leafOnlyHMOffC0?.Release();
        leafOnlyHMOffC0 = null;
        leafOnlyHMOffS?.Release();
        leafOnlyHMOffS = null;
        leafOnlyOffCapCells = 0;
        leafOnlyOffCapM = 0;
    }
}
