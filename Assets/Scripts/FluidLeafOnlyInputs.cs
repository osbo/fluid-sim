using System.Globalization;
using UnityEngine;

/// <summary>
/// Builds LeafOnly neural inputs on the GPU from live nodes, Laplacian stencil, diffusion gradient, and CSR(A).
/// Matches <c>leafonly/data.py</c> feature layout (9-D x, 12 global features, CSR values / scale_A).
/// </summary>
public partial class FluidSimulator : MonoBehaviour
{
    public const int LeafOnlyMaxMixedSize = 8192;
    public const int LeafOnlyLeafSize = 128;

    [Tooltip("If assigned, rebuilds LeafOnly GPU inputs after each pressure CSR build (no .bin I/O).")]
    public ComputeShader leafOnlyInputsShader;

    [Tooltip("Disable to skip LeafOnly input dispatches (saves GPU time while debugging other paths).")]
    public bool buildLeafOnlyInputsEachPressureSolve = true;

    [Tooltip("After each LeafOnly GPU build, read back buffers and print [LeafOnlyParity] lines (matches InspectParity.py). GPU readback — costly if every frame.")]
    public bool debugLeafOnlyParityLog = true;

    [Tooltip("If true, parity lines print only once per play mode session (recommended). If false, prints every pressure solve while debugLeafOnlyParityLog is on.")]
    public bool debugLeafOnlyParityLogOnce = true;

    private static bool s_leafOnlyParityLoggedThisSession;

    private int leafOnlyClearKernel;
    private int leafOnlyRowAbsKernel;
    private int leafOnlyColAbsKernel;
    private int leafOnlyMomentsKernel;
    private int leafOnlyScaleCsrKernel;
    private int leafOnlyBuildXKernel;
    private int leafOnlyCopyNnzKernel;
    private int leafOnlyEdgeMaskKernel;
    private int leafOnlyCompactGcnEdgesKernel;
    private int leafOnlyWriteCompactEdgeCountKernel;

    private ComputeBuffer leafOnlyRowAbsSum;
    private ComputeBuffer leafOnlyColAbsFixed;
    private ComputeBuffer leafOnlyScaleA;
    private ComputeBuffer leafOnlyTotalNnz;
    private ComputeBuffer leafOnlyMoments;
    private ComputeBuffer leafOnlyGlobalFeatures;
    private ComputeBuffer leafOnlyXPacked;
    private ComputeBuffer leafOnlyCsrValuesScaled;
    private ComputeBuffer leafOnlyEdgeValid;
    private ComputeBuffer leafOnlyEdgeScanExclusive;
    private ComputeBuffer leafOnlyGcnEdgeRows;
    private ComputeBuffer leafOnlyGcnEdgeCols;
    private ComputeBuffer leafOnlyGcnEdgeVals;
    private ComputeBuffer leafOnlyCompactEdgeCount;
    private ComputeBuffer leafOnlyPrefixAux;
    private ComputeBuffer leafOnlyPrefixAux2;
    private ComputeBuffer leafOnlyPrefixAuxSmall;

    /// <summary>Last <c>n_pad</c> used (after ceil-to-leaf and MAX_MIXED cap).</summary>
    public int LeafOnlyLastNPadded { get; private set; }

    public ComputeBuffer LeafOnlyInputXBuffer => leafOnlyXPacked;
    public ComputeBuffer LeafOnlyGlobalFeaturesBuffer => leafOnlyGlobalFeatures;
    public ComputeBuffer LeafOnlyCsrValuesScaledBuffer => leafOnlyCsrValuesScaled;
    public ComputeBuffer LeafOnlyScaleABuffer => leafOnlyScaleA;
    /// <summary>Compact COO rows for GCN (<c>n_edge_active</c> mask), length upper bound <c>maxNnz</c>; use <see cref="LeafOnlyCompactEdgeCountBuffer"/> for active count.</summary>
    public ComputeBuffer LeafOnlyGcnEdgeRowsBuffer => leafOnlyGcnEdgeRows;
    public ComputeBuffer LeafOnlyGcnEdgeColsBuffer => leafOnlyGcnEdgeCols;
    public ComputeBuffer LeafOnlyGcnEdgeValsBuffer => leafOnlyGcnEdgeVals;
    public ComputeBuffer LeafOnlyCompactEdgeCountBuffer => leafOnlyCompactEdgeCount;

    private void InitLeafOnlyInputKernels()
    {
        if (leafOnlyInputsShader == null) return;
        leafOnlyClearKernel = leafOnlyInputsShader.FindKernel("LeafOnly_ClearAbsSums");
        leafOnlyRowAbsKernel = leafOnlyInputsShader.FindKernel("LeafOnly_RowAbsSumCSR");
        leafOnlyColAbsKernel = leafOnlyInputsShader.FindKernel("LeafOnly_ColAbsFixedFromCSR");
        leafOnlyMomentsKernel = leafOnlyInputsShader.FindKernel("LeafOnly_WriteScaleAndMoments");
        leafOnlyScaleCsrKernel = leafOnlyInputsShader.FindKernel("LeafOnly_ScaleCsrValues");
        leafOnlyBuildXKernel = leafOnlyInputsShader.FindKernel("LeafOnly_BuildX9");
        leafOnlyCopyNnzKernel = leafOnlyInputsShader.FindKernel("LeafOnly_CopyTotalNnz");
        leafOnlyEdgeMaskKernel = leafOnlyInputsShader.FindKernel("LeafOnly_EdgeActiveMask");
        leafOnlyCompactGcnEdgesKernel = leafOnlyInputsShader.FindKernel("LeafOnly_CompactGcnEdges");
        leafOnlyWriteCompactEdgeCountKernel = leafOnlyInputsShader.FindKernel("LeafOnly_WriteCompactEdgeCount");
    }

    private void EnsureLeafOnlyBuffers(int numNodes, int maxNnz)
    {
        int nodeCap = Mathf.NextPowerOfTwo(Mathf.Max(numNodes, 512));
        if (leafOnlyRowAbsSum == null || leafOnlyRowAbsSum.count < nodeCap)
        {
            leafOnlyRowAbsSum?.Release();
            leafOnlyRowAbsSum = new ComputeBuffer(nodeCap, sizeof(float));
        }
        if (leafOnlyColAbsFixed == null || leafOnlyColAbsFixed.count < nodeCap)
        {
            leafOnlyColAbsFixed?.Release();
            leafOnlyColAbsFixed = new ComputeBuffer(nodeCap, sizeof(uint));
        }
        if (leafOnlyScaleA == null)
            leafOnlyScaleA = new ComputeBuffer(1, sizeof(float));
        if (leafOnlyTotalNnz == null)
            leafOnlyTotalNnz = new ComputeBuffer(1, sizeof(uint));
        if (leafOnlyMoments == null)
            leafOnlyMoments = new ComputeBuffer(16, sizeof(float));
        if (leafOnlyGlobalFeatures == null)
            leafOnlyGlobalFeatures = new ComputeBuffer(12, sizeof(float));
        int xCount = LeafOnlyMaxMixedSize * 9;
        if (leafOnlyXPacked == null || leafOnlyXPacked.count < xCount)
        {
            leafOnlyXPacked?.Release();
            leafOnlyXPacked = new ComputeBuffer(xCount, sizeof(float));
        }
        if (leafOnlyCsrValuesScaled == null || leafOnlyCsrValuesScaled.count < maxNnz)
        {
            leafOnlyCsrValuesScaled?.Release();
            leafOnlyCsrValuesScaled = new ComputeBuffer(maxNnz, sizeof(float));
        }

        if (leafOnlyEdgeValid == null || leafOnlyEdgeValid.count < maxNnz)
        {
            leafOnlyEdgeValid?.Release();
            leafOnlyEdgeValid = new ComputeBuffer(maxNnz, sizeof(uint));
        }
        if (leafOnlyEdgeScanExclusive == null || leafOnlyEdgeScanExclusive.count < maxNnz)
        {
            leafOnlyEdgeScanExclusive?.Release();
            leafOnlyEdgeScanExclusive = new ComputeBuffer(maxNnz, sizeof(uint));
        }
        if (leafOnlyGcnEdgeRows == null || leafOnlyGcnEdgeRows.count < maxNnz)
        {
            leafOnlyGcnEdgeRows?.Release();
            leafOnlyGcnEdgeRows = new ComputeBuffer(maxNnz, sizeof(uint));
        }
        if (leafOnlyGcnEdgeCols == null || leafOnlyGcnEdgeCols.count < maxNnz)
        {
            leafOnlyGcnEdgeCols?.Release();
            leafOnlyGcnEdgeCols = new ComputeBuffer(maxNnz, sizeof(uint));
        }
        if (leafOnlyGcnEdgeVals == null || leafOnlyGcnEdgeVals.count < maxNnz)
        {
            leafOnlyGcnEdgeVals?.Release();
            leafOnlyGcnEdgeVals = new ComputeBuffer(maxNnz, sizeof(float));
        }
        if (leafOnlyCompactEdgeCount == null)
        {
            leafOnlyCompactEdgeCount = new ComputeBuffer(1, sizeof(uint));
        }

        const int tgDual = 1024;
        uint edgeTgs = (uint)((maxNnz + tgDual - 1) / tgDual);
        uint edgeAuxSize = (uint)Mathf.Max(1, (int)edgeTgs);
        if (leafOnlyPrefixAux == null || leafOnlyPrefixAux.count < edgeAuxSize)
        {
            leafOnlyPrefixAux?.Release();
            leafOnlyPrefixAux = new ComputeBuffer((int)edgeAuxSize, sizeof(uint));
        }
        if (leafOnlyPrefixAux2 == null || leafOnlyPrefixAux2.count < edgeAuxSize)
        {
            leafOnlyPrefixAux2?.Release();
            leafOnlyPrefixAux2 = new ComputeBuffer((int)edgeAuxSize, sizeof(uint));
        }
        if (leafOnlyPrefixAuxSmall == null)
            leafOnlyPrefixAuxSmall = new ComputeBuffer(1, sizeof(uint));
    }

    private void BindLeafOnlyCommon(int kernel)
    {
        leafOnlyInputsShader.SetBuffer(kernel, "leafNodesBuffer", nodesBuffer);
        leafOnlyInputsShader.SetBuffer(kernel, "leafMatrixABuffer", matrixABuffer);
        leafOnlyInputsShader.SetBuffer(kernel, "leafDiffusionGradientBuffer", diffusionGradientBuffer);
        leafOnlyInputsShader.SetBuffer(kernel, "leafCsrRowPtr", csrRowPtr);
        leafOnlyInputsShader.SetBuffer(kernel, "leafCsrColIndices", csrColIndices);
        leafOnlyInputsShader.SetBuffer(kernel, "leafCsrValues", csrValues);
        leafOnlyInputsShader.SetBuffer(kernel, "leafRowAbsSum", leafOnlyRowAbsSum);
        leafOnlyInputsShader.SetBuffer(kernel, "leafColAbsFixed", leafOnlyColAbsFixed);
        leafOnlyInputsShader.SetBuffer(kernel, "leafScaleA", leafOnlyScaleA);
        leafOnlyInputsShader.SetBuffer(kernel, "leafTotalNnz", leafOnlyTotalNnz);
        leafOnlyInputsShader.SetBuffer(kernel, "leafMoments", leafOnlyMoments);
        leafOnlyInputsShader.SetBuffer(kernel, "leafGlobalFeatures", leafOnlyGlobalFeatures);
        leafOnlyInputsShader.SetBuffer(kernel, "leafXPacked", leafOnlyXPacked);
        leafOnlyInputsShader.SetBuffer(kernel, "leafCsrValuesScaled", leafOnlyCsrValuesScaled);
    }

    /// <summary>Call after CSR fill (same frame as matrix A). Entirely GPU-side except dispatch bookkeeping.</summary>
    public void DispatchLeafOnlyGpuInputs(int numNodes, int maxNnz)
    {
        if (leafOnlyInputsShader == null || !buildLeafOnlyInputsEachPressureSolve || numNodes <= 0)
            return;
        if (diffusionGradientBuffer == null || nodesBuffer == null || matrixABuffer == null
            || csrRowPtr == null || csrColIndices == null || csrValues == null)
            return;

        int aligned = ((numNodes + LeafOnlyLeafSize - 1) / LeafOnlyLeafSize) * LeafOnlyLeafSize;
        int nPad = Mathf.Min(aligned, LeafOnlyMaxMixedSize);
        int nTake = Mathf.Min(numNodes, nPad);
        LeafOnlyLastNPadded = nPad;

        int nodeCap = Mathf.NextPowerOfTwo(Mathf.Max(numNodes, 512));
        EnsureLeafOnlyBuffers(numNodes, maxNnz);

        leafOnlyInputsShader.SetInt("leafNumNodes", numNodes);
        leafOnlyInputsShader.SetInt("leafNPadded", nPad);
        leafOnlyInputsShader.SetInt("leafNTake", nTake);
        leafOnlyInputsShader.SetInt("leafNEdgeActive", nTake);
        leafOnlyInputsShader.SetInt("leafMaxNnzBound", maxNnz);
        leafOnlyInputsShader.SetInt("leafBufferCapacity", nodeCap);

        int groupsCap = Mathf.CeilToInt(nodeCap / 256.0f);

        BindLeafOnlyCommon(leafOnlyClearKernel);
        leafOnlyInputsShader.Dispatch(leafOnlyClearKernel, groupsCap, 1, 1);

        BindLeafOnlyCommon(leafOnlyRowAbsKernel);
        leafOnlyInputsShader.Dispatch(leafOnlyRowAbsKernel, groupsCap, 1, 1);

        BindLeafOnlyCommon(leafOnlyColAbsKernel);
        leafOnlyInputsShader.Dispatch(leafOnlyColAbsKernel, groupsCap, 1, 1);

        BindLeafOnlyCommon(leafOnlyCopyNnzKernel);
        leafOnlyInputsShader.Dispatch(leafOnlyCopyNnzKernel, 1, 1, 1);

        BindLeafOnlyCommon(leafOnlyMomentsKernel);
        leafOnlyInputsShader.Dispatch(leafOnlyMomentsKernel, 1, 1, 1);

        BindLeafOnlyCommon(leafOnlyScaleCsrKernel);
        int nnzGroups = Mathf.CeilToInt(maxNnz / 256.0f);
        leafOnlyInputsShader.Dispatch(leafOnlyScaleCsrKernel, nnzGroups, 1, 1);

        DispatchLeafOnlyGcnEdgeCompaction(maxNnz);

        BindLeafOnlyCommon(leafOnlyBuildXKernel);
        int xGroups = Mathf.CeilToInt(nPad / 256.0f);
        leafOnlyInputsShader.Dispatch(leafOnlyBuildXKernel, xGroups, 1, 1);

        if (debugLeafOnlyParityLog)
        {
            if (!debugLeafOnlyParityLogOnce || !s_leafOnlyParityLoggedThisSession)
            {
                LogLeafOnlyParityToConsole(numNodes, nPad, nTake, maxNnz);
                if (debugLeafOnlyParityLogOnce)
                    s_leafOnlyParityLoggedThisSession = true;
            }
        }
    }

    /// <summary>Exclusive prefix sum over <paramref name="len"/> elements (zeroff=1), same as CSR row-pointer build.</summary>
    private void RunLeafOnlyExclusivePrefixScan(ComputeBuffer input, ComputeBuffer output, int len)
    {
        if (radixSortShader == null || len <= 0) return;

        uint tgSize = 512u;
        uint numThreadgroups = (uint)((len + (tgSize * 2) - 1) / (tgSize * 2));
        uint auxSize = (uint)Mathf.Max(1, (int)numThreadgroups);

        radixSortShader.SetBuffer(radixPrefixSumKernelId, "input", input);
        radixSortShader.SetBuffer(radixPrefixSumKernelId, "output", output);
        radixSortShader.SetBuffer(radixPrefixSumKernelId, "aux", leafOnlyPrefixAux);
        radixSortShader.SetInt("len", len);
        radixSortShader.SetInt("zeroff", 1);
        radixSortShader.Dispatch(radixPrefixSumKernelId, (int)numThreadgroups, 1, 1);

        if (numThreadgroups > 1)
        {
            radixSortShader.SetBuffer(radixPrefixSumKernelId, "input", leafOnlyPrefixAux);
            radixSortShader.SetBuffer(radixPrefixSumKernelId, "output", leafOnlyPrefixAux2);
            radixSortShader.SetBuffer(radixPrefixSumKernelId, "aux", leafOnlyPrefixAuxSmall);
            radixSortShader.SetInt("len", (int)auxSize);
            radixSortShader.SetInt("zeroff", 1);
            radixSortShader.Dispatch(radixPrefixSumKernelId, 1, 1, 1);

            radixSortShader.SetBuffer(radixPrefixFixupKernelId, "input", output);
            radixSortShader.SetBuffer(radixPrefixFixupKernelId, "aux", leafOnlyPrefixAux2);
            radixSortShader.SetInt("len", len);
            radixSortShader.Dispatch(radixPrefixFixupKernelId, (int)numThreadgroups, 1, 1);
        }
    }

    /// <summary>Stream-compact CSR edges where both endpoints &lt; n_edge_active; values use scaled CSR (GPU-only).</summary>
    private void DispatchLeafOnlyGcnEdgeCompaction(int maxNnz)
    {
        if (leafOnlyInputsShader == null || radixSortShader == null || maxNnz <= 0) return;
        if (csrRowIndices == null || csrColIndices == null) return;

        int nnzGroups = Mathf.CeilToInt(maxNnz / 256.0f);

        leafOnlyInputsShader.SetBuffer(leafOnlyEdgeMaskKernel, "leafTotalNnz", leafOnlyTotalNnz);
        leafOnlyInputsShader.SetBuffer(leafOnlyEdgeMaskKernel, "leafCsrEdgeRow", csrRowIndices);
        leafOnlyInputsShader.SetBuffer(leafOnlyEdgeMaskKernel, "leafCsrColIndices", csrColIndices);
        leafOnlyInputsShader.SetBuffer(leafOnlyEdgeMaskKernel, "leafEdgeValid", leafOnlyEdgeValid);
        leafOnlyInputsShader.Dispatch(leafOnlyEdgeMaskKernel, nnzGroups, 1, 1);

        RunLeafOnlyExclusivePrefixScan(leafOnlyEdgeValid, leafOnlyEdgeScanExclusive, maxNnz);

        leafOnlyInputsShader.SetBuffer(leafOnlyCompactGcnEdgesKernel, "leafTotalNnz", leafOnlyTotalNnz);
        leafOnlyInputsShader.SetBuffer(leafOnlyCompactGcnEdgesKernel, "leafEdgeValid", leafOnlyEdgeValid);
        leafOnlyInputsShader.SetBuffer(leafOnlyCompactGcnEdgesKernel, "leafEdgeScanExclusive", leafOnlyEdgeScanExclusive);
        leafOnlyInputsShader.SetBuffer(leafOnlyCompactGcnEdgesKernel, "leafGcnEdgeRows", leafOnlyGcnEdgeRows);
        leafOnlyInputsShader.SetBuffer(leafOnlyCompactGcnEdgesKernel, "leafGcnEdgeCols", leafOnlyGcnEdgeCols);
        leafOnlyInputsShader.SetBuffer(leafOnlyCompactGcnEdgesKernel, "leafGcnEdgeVals", leafOnlyGcnEdgeVals);
        leafOnlyInputsShader.SetBuffer(leafOnlyCompactGcnEdgesKernel, "leafCsrEdgeRow", csrRowIndices);
        leafOnlyInputsShader.SetBuffer(leafOnlyCompactGcnEdgesKernel, "leafCsrColIndices", csrColIndices);
        leafOnlyInputsShader.SetBuffer(leafOnlyCompactGcnEdgesKernel, "leafCsrValuesScaled", leafOnlyCsrValuesScaled);
        leafOnlyInputsShader.Dispatch(leafOnlyCompactGcnEdgesKernel, nnzGroups, 1, 1);

        leafOnlyInputsShader.SetBuffer(leafOnlyWriteCompactEdgeCountKernel, "leafTotalNnz", leafOnlyTotalNnz);
        leafOnlyInputsShader.SetBuffer(leafOnlyWriteCompactEdgeCountKernel, "leafEdgeScanExclusive", leafOnlyEdgeScanExclusive);
        leafOnlyInputsShader.SetBuffer(leafOnlyWriteCompactEdgeCountKernel, "leafEdgeValid", leafOnlyEdgeValid);
        leafOnlyInputsShader.SetBuffer(leafOnlyWriteCompactEdgeCountKernel, "leafCompactEdgeCount", leafOnlyCompactEdgeCount);
        leafOnlyInputsShader.Dispatch(leafOnlyWriteCompactEdgeCountKernel, 1, 1, 1);
    }

    /// <summary>Same numeric format as <c>InspectParity.py</c> (<c>phase=unity</c>).</summary>
    private void LogLeafOnlyParityToConsole(int numNodesReal, int nPad, int nTake, int maxNnzBound)
    {
        const string phase = "unity";
        int nFloor = (Mathf.Min(numNodesReal, LeafOnlyMaxMixedSize) / LeafOnlyLeafSize) * LeafOnlyLeafSize;
        Debug.Log(
            $"[LeafOnlyParity] phase={phase} num_nodes_real={numNodesReal} align=ceil n_pad={nPad} " +
            $"n_take={nTake} num_leaves={nPad / LeafOnlyLeafSize} problem_padded_floor={nFloor} MAX_MIXED_SIZE={LeafOnlyMaxMixedSize}");

        uint[] nnzArr = new uint[1];
        leafOnlyTotalNnz.GetData(nnzArr, 0, 0, 1);
        int nnz = (int)nnzArr[0];
        if (nnz < 0 || nnz > maxNnzBound)
            nnz = Mathf.Clamp(nnz, 0, maxNnzBound);

        int xCount = nPad * 9;
        float[] xFlat = new float[xCount];
        leafOnlyXPacked.GetData(xFlat, 0, 0, xCount);
        LeafOnlyParitySummarize(phase, "x_leaf", xFlat, xCount);
        LeafOnlyParityHead(phase, "x_leaf", xFlat, xCount, 16);

        float[] gf = new float[12];
        leafOnlyGlobalFeatures.GetData(gf, 0, 0, 12);
        LeafOnlyParitySummarize(phase, "global_features", gf, 12);
        LeafOnlyParityHead(phase, "global_features", gf, 12);

        float[] sc = new float[1];
        leafOnlyScaleA.GetData(sc, 0, 0, 1);
        LeafOnlyParitySummarize(phase, "scale_A", sc, 1);

        float[] moments = new float[16];
        leafOnlyMoments.GetData(moments, 0, 0, 16);
        LeafOnlyParitySummarize(phase, "leaf_moments", moments, 16);
        LeafOnlyParityHead(phase, "leaf_moments", moments, 16);

        if (nnz > 0)
        {
            float[] csr = new float[nnz];
            leafOnlyCsrValuesScaled.GetData(csr, 0, 0, nnz);
            LeafOnlyParitySummarize(phase, "csr_values_scaled", csr, nnz);
            LeafOnlyParityHead(phase, "csr_values_scaled", csr, nnz, 16);
        }
        else
        {
            Debug.Log($"[LeafOnlyParity] phase={phase} tensor=csr_values_scaled n=0 (empty)");
        }

        Debug.Log($"[LeafOnlyParity] phase={phase} tensor=meta nnz={nnz} (CSR nonzeros after build)");

        DispatchLeafOnlyDiagEdgeFeatsAndLogParity(nPad, nnz);
        DispatchLeafOnlyOffEdgeFeatsAndLogParity(nPad, nnz);

        if (leafOnlyCompactEdgeCount != null && leafOnlyGcnEdgeRows != null && leafOnlyGcnEdgeCols != null && leafOnlyGcnEdgeVals != null)
        {
            uint[] ec = new uint[1];
            leafOnlyCompactEdgeCount.GetData(ec, 0, 0, 1);
            int eCompact = (int)ec[0];
            Debug.Log($"[LeafOnlyParity] phase={phase} tensor=gcn_meta compact_nnz={eCompact} n_edge_active={Mathf.Min(numNodesReal, nPad)}");
            if (eCompact > 0)
            {
                int headE = Mathf.Min(16, eCompact);
                uint[] er = new uint[headE];
                uint[] ec2 = new uint[headE];
                float[] ev = new float[headE];
                leafOnlyGcnEdgeRows.GetData(er, 0, 0, headE);
                leafOnlyGcnEdgeCols.GetData(ec2, 0, 0, headE);
                leafOnlyGcnEdgeVals.GetData(ev, 0, 0, headE);
                var inv = CultureInfo.InvariantCulture;
                var sb = new System.Text.StringBuilder();
                for (int i = 0; i < headE; i++)
                {
                    if (i > 0) sb.Append(';');
                    sb.Append('(');
                    sb.Append(er[i].ToString(inv));
                    sb.Append(',');
                    sb.Append(ec2[i].ToString(inv));
                    sb.Append(',');
                    sb.Append(ev[i].ToString("G9", inv));
                    sb.Append(')');
                }
                Debug.Log($"[LeafOnlyParity] phase={phase} tensor=gcn_edges_head[{headE}]={sb}");
            }
        }

        int mOff = LeafOnlyHMatrixStatic.NumOffBlocks;
        var invHm = CultureInfo.InvariantCulture;
        Debug.Log(
            $"[LeafOnlyParity] phase={phase} tensor=hmatrix_meta M_off={mOff} " +
            $"MAX_NUM_LEAVES={LeafOnlyHMatrixStatic.MaxNumLeaves} ETA={LeafOnlyHMatrixStatic.Eta.ToString("G9", invHm)}");
        float[] hmFlat = LeafOnlyHMatrixStatic.GetFlatInterleavedFloats();
        LeafOnlyParitySummarize(phase, "hmatrix_off_r0_c0_s_flat", hmFlat, mOff * 3);
        LeafOnlyParityHead(phase, "hmatrix_off_r0_c0_s_flat", hmFlat, mOff * 3, 48);

        if (LeafOnlyWeightsLoadedSuccessfully && leafOnlyCompactEdgeCount != null)
        {
            uint[] ecTok = new uint[1];
            leafOnlyCompactEdgeCount.GetData(ecTok, 0, 0, 1);
            int eTok = (int)ecTok[0];
            DispatchLeafOnlyEmbedForwardAndLogParity(nPad, eTok);
            DispatchLeafOnlyCpuLayer1ParityAndLog(nPad);
        }
    }

    internal static void LeafOnlyParitySummarize(string phase, string name, float[] data, int n)
    {
        if (data == null || n <= 0)
        {
            Debug.Log($"[LeafOnlyParity] phase={phase} tensor={name} n=0 (empty)");
            return;
        }

        float xmin = float.PositiveInfinity, xmax = float.NegativeInfinity;
        double sum = 0.0, sumAbs = 0.0, sumSq = 0.0;
        for (int i = 0; i < n; i++)
        {
            float v = data[i];
            if (v < xmin) xmin = v;
            if (v > xmax) xmax = v;
            sum += v;
            sumAbs += Mathf.Abs(v);
            sumSq += (double)v * v;
        }

        float mean = (float)(sum / n);
        float l2 = (float)System.Math.Sqrt(sumSq);
        var inv = CultureInfo.InvariantCulture;
        Debug.Log(
            $"[LeafOnlyParity] phase={phase} tensor={name} n={n} min={xmin.ToString("G9", inv)} max={xmax.ToString("G9", inv)} " +
            $"mean={mean.ToString("G9", inv)} sum_abs={sumAbs.ToString("G9", inv)} l2={l2.ToString("G9", inv)}");
    }

    internal static void LeafOnlyParityHead(string phase, string name, float[] data, int n, int k = 16)
    {
        if (data == null || n <= 0)
        {
            Debug.Log($"[LeafOnlyParity] phase={phase} tensor={name}_head[0]=");
            return;
        }

        int c = Mathf.Min(k, n);
        var inv = CultureInfo.InvariantCulture;
        var parts = new System.Text.StringBuilder();
        for (int i = 0; i < c; i++)
        {
            if (i > 0) parts.Append(',');
            parts.Append(data[i].ToString("G9", inv));
        }

        Debug.Log($"[LeafOnlyParity] phase={phase} tensor={name}_head[{c}]={parts}");
    }

    /// <summary>Parity slice at <paramref name="start"/> (inclusive), same log pattern as <c>InspectParity._parity_head_phase_at</c>.</summary>
    internal static void LeafOnlyParityHeadAt(string phase, string name, float[] data, int nTotal, int start, int k)
    {
        if (data == null || nTotal <= 0 || start < 0 || start >= nTotal)
        {
            Debug.Log($"[LeafOnlyParity] phase={phase} tensor={name}_at{start}_head[0]=");
            return;
        }

        int c = Mathf.Min(k, nTotal - start);
        var inv = CultureInfo.InvariantCulture;
        var parts = new System.Text.StringBuilder();
        for (int i = 0; i < c; i++)
        {
            if (i > 0) parts.Append(',');
            parts.Append(data[start + i].ToString("G9", inv));
        }

        Debug.Log($"[LeafOnlyParity] phase={phase} tensor={name}_at{start}_head[{c}]={parts}");
    }

    private void ReleaseLeafOnlyBuffers()
    {
        leafOnlyRowAbsSum?.Release();
        leafOnlyRowAbsSum = null;
        leafOnlyColAbsFixed?.Release();
        leafOnlyColAbsFixed = null;
        leafOnlyScaleA?.Release();
        leafOnlyScaleA = null;
        leafOnlyTotalNnz?.Release();
        leafOnlyTotalNnz = null;
        leafOnlyMoments?.Release();
        leafOnlyMoments = null;
        leafOnlyGlobalFeatures?.Release();
        leafOnlyGlobalFeatures = null;
        leafOnlyXPacked?.Release();
        leafOnlyXPacked = null;
        leafOnlyCsrValuesScaled?.Release();
        leafOnlyCsrValuesScaled = null;
        leafOnlyEdgeValid?.Release();
        leafOnlyEdgeValid = null;
        leafOnlyEdgeScanExclusive?.Release();
        leafOnlyEdgeScanExclusive = null;
        leafOnlyGcnEdgeRows?.Release();
        leafOnlyGcnEdgeRows = null;
        leafOnlyGcnEdgeCols?.Release();
        leafOnlyGcnEdgeCols = null;
        leafOnlyGcnEdgeVals?.Release();
        leafOnlyGcnEdgeVals = null;
        leafOnlyCompactEdgeCount?.Release();
        leafOnlyCompactEdgeCount = null;
        leafOnlyPrefixAux?.Release();
        leafOnlyPrefixAux = null;
        leafOnlyPrefixAux2?.Release();
        leafOnlyPrefixAux2 = null;
        leafOnlyPrefixAuxSmall?.Release();
        leafOnlyPrefixAuxSmall = null;

        ReleaseLeafOnlyEmbedBuffers();
        ReleaseLeafOnlyDiagEdgeFeatsBuffers();
        ReleaseLeafOnlyOffEdgeFeatsBuffers();
    }
}
