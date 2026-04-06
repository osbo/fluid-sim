using System.Globalization;
using UnityEngine;

/// <summary>
/// GPU first transformer layer (diag + H-strip + off) via <c>LeafOnlyLayer1.compute</c>.
/// When <see cref="leafOnlyLayer1Shader"/> is assigned, parity logging uses this path instead of the CPU reference.
/// Requires same checkpoint constraints as <see cref="DispatchLeafOnlyCpuLayer1ParityAndLog"/> (highway=0, layout L×L dense, etc.).
/// </summary>
public partial class FluidSimulator
{
    public const int LeafOnlyLayer1DModelMax = 128;

    [Tooltip("Optional: LeafOnlyLayer1.compute — GPU layer 0 after embed. Leave null to keep CPU parity only.")]
    public ComputeShader leafOnlyLayer1Shader;

    private int leafL1KCopyFromEnc;
    private int leafL1KCopyAToSkipA;
    private int leafL1KCopyAToSkipB;
    private int leafL1KClear;
    private int leafL1KLayerNorm;
    private int leafL1KQkv;
    private int leafL1KAttnRow;
    private int leafL1KProjAddSkip;
    private int leafL1KFfn;
    private int leafL1KClearPad;
    private int leafL1KCopyPad;
    private int leafL1KStrip;
    private int leafL1KStripPool;
    private int leafL1KPoolOffEdge;

    private ComputeBuffer leafL1BufA;
    private ComputeBuffer leafL1BufB;
    private ComputeBuffer leafL1Qkv;
    private ComputeBuffer leafL1Comb;
    private ComputeBuffer leafL1SkipA;
    private ComputeBuffer leafL1SkipB;
    private ComputeBuffer leafL1HPad;
    private ComputeBuffer leafL1Strip;
    private ComputeBuffer leafL1HmWRow;
    private ComputeBuffer leafL1HmWCol;
    private ComputeBuffer leafL1OffEdgeLs;
    private ComputeBuffer leafL1OffBufA;
    private ComputeBuffer leafL1OffBufB;
    private ComputeBuffer leafL1OffQkv;
    private ComputeBuffer leafL1OffComb;
    private ComputeBuffer leafL1OffSkipA;
    private ComputeBuffer leafL1OffSkipB;

    private int leafL1CapNPad;
    private int leafL1CapDm;
    private int leafL1CapMOff;
    private int leafL1CapKMax;
    private int leafL1CapLFull;
    private static bool s_warnedLeafOnlyLayer1GpuMissing;
    private static bool s_warnedLeafOnlyLayer1GpuSkip;

    private void InitLeafOnlyLayer1GpuKernels()
    {
        if (leafOnlyLayer1Shader == null)
            return;
        leafL1KCopyFromEnc = leafOnlyLayer1Shader.FindKernel("LeafOnly_L1_CopyFromEnc");
        leafL1KCopyAToSkipA = leafOnlyLayer1Shader.FindKernel("LeafOnly_L1_CopyAToSkipA");
        leafL1KCopyAToSkipB = leafOnlyLayer1Shader.FindKernel("LeafOnly_L1_CopyAToSkipB");
        leafL1KClear = leafOnlyLayer1Shader.FindKernel("LeafOnly_L1_Clear");
        leafL1KLayerNorm = leafOnlyLayer1Shader.FindKernel("LeafOnly_L1_LayerNorm");
        leafL1KQkv = leafOnlyLayer1Shader.FindKernel("LeafOnly_L1_QKV");
        leafL1KAttnRow = leafOnlyLayer1Shader.FindKernel("LeafOnly_L1_AttnRow");
        leafL1KProjAddSkip = leafOnlyLayer1Shader.FindKernel("LeafOnly_L1_ProjAddSkip");
        leafL1KFfn = leafOnlyLayer1Shader.FindKernel("LeafOnly_L1_FFN");
        leafL1KClearPad = leafOnlyLayer1Shader.FindKernel("LeafOnly_L1_ClearPad");
        leafL1KCopyPad = leafOnlyLayer1Shader.FindKernel("LeafOnly_L1_CopyLeavesToPad");
        leafL1KStrip = leafOnlyLayer1Shader.FindKernel("LeafOnly_L1_StripAccum");
        leafL1KStripPool = leafOnlyLayer1Shader.FindKernel("LeafOnly_L1_StripPoolToOff");
        leafL1KPoolOffEdge = leafOnlyLayer1Shader.FindKernel("LeafOnly_L1_PoolOffEdgeFeats");
    }

    private void ReleaseLeafOnlyLayer1GpuBuffers()
    {
        leafL1BufA?.Release();
        leafL1BufA = null;
        leafL1BufB?.Release();
        leafL1BufB = null;
        leafL1Qkv?.Release();
        leafL1Qkv = null;
        leafL1Comb?.Release();
        leafL1Comb = null;
        leafL1SkipA?.Release();
        leafL1SkipA = null;
        leafL1SkipB?.Release();
        leafL1SkipB = null;
        leafL1HPad?.Release();
        leafL1HPad = null;
        leafL1Strip?.Release();
        leafL1Strip = null;
        leafL1HmWRow?.Release();
        leafL1HmWRow = null;
        leafL1HmWCol?.Release();
        leafL1HmWCol = null;
        leafL1OffEdgeLs?.Release();
        leafL1OffEdgeLs = null;
        leafL1OffBufA?.Release();
        leafL1OffBufA = null;
        leafL1OffBufB?.Release();
        leafL1OffBufB = null;
        leafL1OffQkv?.Release();
        leafL1OffQkv = null;
        leafL1OffComb?.Release();
        leafL1OffComb = null;
        leafL1OffSkipA?.Release();
        leafL1OffSkipA = null;
        leafL1OffSkipB?.Release();
        leafL1OffSkipB = null;
        leafL1CapNPad = 0;
        leafL1CapDm = 0;
        leafL1CapMOff = 0;
        leafL1CapKMax = 0;
        leafL1CapLFull = 0;
    }

    private void EnsureLeafOnlyLayer1GpuBuffers(int nPad, int d, int lFull, int kMax, int mOff, int lsOff)
    {
        if (leafL1BufA != null && leafL1CapNPad >= nPad && leafL1CapDm >= d
            && leafL1CapMOff >= mOff && leafL1CapKMax >= kMax && leafL1CapLFull == lFull)
            return;

        ReleaseLeafOnlyLayer1GpuBuffers();

        int nTok = nPad * d;
        int qkv = nPad * 3 * d;
        int kAct = nPad / lFull;
        int hPadSz = kMax * lFull * d;
        int stripSz = Mathf.Max(1, mOff * lFull * d);
        int nOffTokEl = Mathf.Max(1, mOff * lsOff * d);
        int offQkv = Mathf.Max(1, mOff * lsOff * 3 * d);
        int offLsSq = Mathf.Max(1, mOff * lsOff * lsOff * 4);
        int hmSz = Mathf.Max(1, mOff * kMax);

        leafL1BufA = new ComputeBuffer(nTok, sizeof(float));
        leafL1BufB = new ComputeBuffer(nTok, sizeof(float));
        leafL1Qkv = new ComputeBuffer(qkv, sizeof(float));
        leafL1Comb = new ComputeBuffer(nTok, sizeof(float));
        leafL1SkipA = new ComputeBuffer(nTok, sizeof(float));
        leafL1SkipB = new ComputeBuffer(nTok, sizeof(float));
        leafL1HPad = new ComputeBuffer(hPadSz, sizeof(float));
        leafL1Strip = new ComputeBuffer(stripSz, sizeof(float));
        leafL1HmWRow = new ComputeBuffer(hmSz, sizeof(float));
        leafL1HmWCol = new ComputeBuffer(hmSz, sizeof(float));
        leafL1OffEdgeLs = new ComputeBuffer(offLsSq, sizeof(float));
        leafL1OffBufA = new ComputeBuffer(nOffTokEl, sizeof(float));
        leafL1OffBufB = new ComputeBuffer(nOffTokEl, sizeof(float));
        leafL1OffQkv = new ComputeBuffer(offQkv, sizeof(float));
        leafL1OffComb = new ComputeBuffer(nOffTokEl, sizeof(float));
        leafL1OffSkipA = new ComputeBuffer(nOffTokEl, sizeof(float));
        leafL1OffSkipB = new ComputeBuffer(nOffTokEl, sizeof(float));
        leafL1CapNPad = nPad;
        leafL1CapDm = d;
        leafL1CapMOff = mOff;
        leafL1CapKMax = kMax;
        leafL1CapLFull = lFull;

        var r0 = LeafOnlyHMatrixStatic.OffR0;
        var c0 = LeafOnlyHMatrixStatic.OffC0;
        var sS = LeafOnlyHMatrixStatic.OffS;
        var wr = new float[hmSz];
        var wc = new float[hmSz];
        for (int m = 0; m < mOff; m++)
        {
            float invS = 1f / Mathf.Max(1, sS[m]);
            for (int k = r0[m]; k < r0[m] + sS[m] && k < kMax; k++)
                wr[m * kMax + k] = invS;
            for (int k = c0[m]; k < c0[m] + sS[m] && k < kMax; k++)
                wc[m * kMax + k] = invS;
        }

        leafL1HmWRow.SetData(wr);
        leafL1HmWCol.SetData(wc);
    }

    private static int LeafOnlyLayer1Groups256(int n) => Mathf.Max(1, Mathf.CeilToInt(n / 256f));

    private void LeafOnlyLayer1BindAll(ComputeBuffer edgeFeats)
    {
        int[] ks =
        {
            leafL1KCopyFromEnc, leafL1KCopyAToSkipA, leafL1KCopyAToSkipB, leafL1KClear, leafL1KLayerNorm,
            leafL1KQkv, leafL1KAttnRow, leafL1KProjAddSkip, leafL1KFfn, leafL1KClearPad, leafL1KCopyPad,
            leafL1KStrip, leafL1KStripPool, leafL1KPoolOffEdge,
        };
        foreach (int k in ks)
        {
            leafOnlyLayer1Shader.SetBuffer(k, "leafOnlyWeights", leafOnlyWeightsFloatBuffer);
            leafOnlyLayer1Shader.SetBuffer(k, "leafTokenAfterEnc", leafOnlyTokenAfterEnc);
            leafOnlyLayer1Shader.SetBuffer(k, "leafL1EdgeFeats", edgeFeats);
            leafOnlyLayer1Shader.SetBuffer(k, "leafOffEdgeFeatsRaw", LeafOnlyOffEdgeFeatsBuffer);
            leafOnlyLayer1Shader.SetBuffer(k, "leafL1HmWRow", leafL1HmWRow);
            leafOnlyLayer1Shader.SetBuffer(k, "leafL1HmWCol", leafL1HmWCol);
            leafOnlyLayer1Shader.SetBuffer(k, "leafL1BufA", leafL1BufA);
            leafOnlyLayer1Shader.SetBuffer(k, "leafL1BufB", leafL1BufB);
            leafOnlyLayer1Shader.SetBuffer(k, "leafL1Qkv", leafL1Qkv);
            leafOnlyLayer1Shader.SetBuffer(k, "leafL1Comb", leafL1Comb);
            leafOnlyLayer1Shader.SetBuffer(k, "leafL1SkipA", leafL1SkipA);
            leafOnlyLayer1Shader.SetBuffer(k, "leafL1SkipB", leafL1SkipB);
            leafOnlyLayer1Shader.SetBuffer(k, "leafL1HPad", leafL1HPad);
            leafOnlyLayer1Shader.SetBuffer(k, "leafL1Strip", leafL1Strip);
            leafOnlyLayer1Shader.SetBuffer(k, "leafL1OffEdgeLs", leafL1OffEdgeLs);
            leafOnlyLayer1Shader.SetBuffer(k, "leafL1OffBufA", leafL1OffBufA);
            leafOnlyLayer1Shader.SetBuffer(k, "leafL1OffBufB", leafL1OffBufB);
            leafOnlyLayer1Shader.SetBuffer(k, "leafL1OffQkv", leafL1OffQkv);
            leafOnlyLayer1Shader.SetBuffer(k, "leafL1OffComb", leafL1OffComb);
            leafOnlyLayer1Shader.SetBuffer(k, "leafL1OffSkipA", leafL1OffSkipA);
            leafOnlyLayer1Shader.SetBuffer(k, "leafL1OffSkipB", leafL1OffSkipB);
        }
    }

    private void LeafOnlyLayer1SetTbInts(in LeafOnlyTbWeightOffsets o)
    {
        leafOnlyLayer1Shader.SetInt("tbNorm1W", o.Norm1W);
        leafOnlyLayer1Shader.SetInt("tbNorm1B", o.Norm1B);
        leafOnlyLayer1Shader.SetInt("tbQkvW", o.QkvW);
        leafOnlyLayer1Shader.SetInt("tbQkvB", o.QkvB);
        leafOnlyLayer1Shader.SetInt("tbProjW", o.ProjW);
        leafOnlyLayer1Shader.SetInt("tbProjB", o.ProjB);
        leafOnlyLayer1Shader.SetInt("tbEg0W", o.Eg0W);
        leafOnlyLayer1Shader.SetInt("tbEg0B", o.Eg0B);
        leafOnlyLayer1Shader.SetInt("tbEg2W", o.Eg2W);
        leafOnlyLayer1Shader.SetInt("tbEg2B", o.Eg2B);
        leafOnlyLayer1Shader.SetInt("tbNorm2W", o.Norm2W);
        leafOnlyLayer1Shader.SetInt("tbNorm2B", o.Norm2B);
        leafOnlyLayer1Shader.SetInt("tbMlp0W", o.Mlp0W);
        leafOnlyLayer1Shader.SetInt("tbMlp0B", o.Mlp0B);
        leafOnlyLayer1Shader.SetInt("tbMlp2W", o.Mlp2W);
        leafOnlyLayer1Shader.SetInt("tbMlp2B", o.Mlp2B);
    }

    private void LeafOnlyLayer1RebindMainStreamToOff()
    {
        int[] ks =
        {
            leafL1KCopyAToSkipA, leafL1KCopyAToSkipB, leafL1KClear, leafL1KLayerNorm, leafL1KQkv, leafL1KAttnRow,
            leafL1KProjAddSkip, leafL1KFfn,
        };
        foreach (int k in ks)
        {
            leafOnlyLayer1Shader.SetBuffer(k, "leafL1BufA", leafL1OffBufA);
            leafOnlyLayer1Shader.SetBuffer(k, "leafL1BufB", leafL1OffBufB);
            leafOnlyLayer1Shader.SetBuffer(k, "leafL1Qkv", leafL1OffQkv);
            leafOnlyLayer1Shader.SetBuffer(k, "leafL1Comb", leafL1OffComb);
            leafOnlyLayer1Shader.SetBuffer(k, "leafL1SkipA", leafL1OffSkipA);
            leafOnlyLayer1Shader.SetBuffer(k, "leafL1SkipB", leafL1OffSkipB);
        }
    }

    private void LeafOnlyLayer1RebindMainStreamToDiag()
    {
        int[] ks =
        {
            leafL1KCopyAToSkipA, leafL1KCopyAToSkipB, leafL1KClear, leafL1KLayerNorm, leafL1KQkv, leafL1KAttnRow,
            leafL1KProjAddSkip, leafL1KFfn,
        };
        foreach (int k in ks)
        {
            leafOnlyLayer1Shader.SetBuffer(k, "leafL1BufA", leafL1BufA);
            leafOnlyLayer1Shader.SetBuffer(k, "leafL1BufB", leafL1BufB);
            leafOnlyLayer1Shader.SetBuffer(k, "leafL1Qkv", leafL1Qkv);
            leafOnlyLayer1Shader.SetBuffer(k, "leafL1Comb", leafL1Comb);
            leafOnlyLayer1Shader.SetBuffer(k, "leafL1SkipA", leafL1SkipA);
            leafOnlyLayer1Shader.SetBuffer(k, "leafL1SkipB", leafL1SkipB);
        }
    }

    /// <summary>Runs GPU layer 1 and logs <c>h_diag_after_layer0</c> / <c>off_stream_after_layer0</c> parity lines.</summary>
    private void DispatchLeafOnlyLayer1GpuForwardAndLogParity(int nPad)
    {
        const string phase = "unity";
        if (leafOnlyLayer1Shader == null)
        {
            if (!s_warnedLeafOnlyLayer1GpuMissing)
            {
                s_warnedLeafOnlyLayer1GpuMissing = true;
                Debug.Log(
                    "[LeafOnlyParity] GPU layer1 skipped (assign FluidSimulator.leafOnlyLayer1Shader = LeafOnlyLayer1.compute). Using CPU reference if enabled.");
            }

            DispatchLeafOnlyCpuLayer1ParityAndLog(nPad);
            return;
        }

        if (!LeafOnlyWeightsLoadedSuccessfully || leafOnlyWeightsFloatBuffer == null || leafOnlyTokenAfterEnc == null)
        {
            DispatchLeafOnlyCpuLayer1ParityAndLog(nPad);
            return;
        }

        LeafOnlyCheckpointHeader arch = LeafOnlyCheckpoint;
        if (arch.HighwayFfnMlp != 0 || arch.DecoupledRouteGates != 0 || arch.AttentionLayoutCode != 0 || arch.NumLayers < 1)
        {
            if (!s_warnedLeafOnlyLayer1GpuSkip)
            {
                s_warnedLeafOnlyLayer1GpuSkip = true;
                Debug.Log("[LeafOnlyParity] GPU layer1 skipped (needs highway=0, route_gates=0, attn_layout=0, num_layers>=1).");
            }

            DispatchLeafOnlyCpuLayer1ParityAndLog(nPad);
            return;
        }

        int d = arch.DModel;
        if (d > LeafOnlyLayer1DModelMax || arch.NumHeads < 1 || d % arch.NumHeads != 0)
        {
            DispatchLeafOnlyCpuLayer1ParityAndLog(nPad);
            return;
        }

        int lFull = LeafOnlyLeafSize;
        int lsOff = lFull / 4;
        if (nPad <= 0 || nPad % lFull != 0 || lFull != arch.LeafSize)
        {
            DispatchLeafOnlyCpuLayer1ParityAndLog(nPad);
            return;
        }

        int kAct = nPad / lFull;
        int nh = arch.NumHeads;
        int dh = d / nh;
        int mOff = LeafOnlyHMatrixStatic.NumOffBlocks;
        int kMax = LeafOnlyHMatrixStatic.MaxNumLeaves;
        if (LeafOnlyDiagEdgeFeatsBuffer == null || (mOff > 0 && LeafOnlyOffEdgeFeatsBuffer == null))
        {
            DispatchLeafOnlyCpuLayer1ParityAndLog(nPad);
            return;
        }

        int embedEnd = LeafOnlyEmbedPhaseFloatCount(in arch, LeafOnlyGlobalFeaturesDim);
        int tbFloats = LeafOnlyTransformerBlockFloatCount(in arch);
        if (tbFloats < 0 || leafOnlyWeightsFloatBuffer.count < embedEnd + 2 * tbFloats)
        {
            DispatchLeafOnlyCpuLayer1ParityAndLog(nPad);
            return;
        }

        LeafOnlyTbOffsetsFromBase(embedEnd, in arch, out LeafOnlyTbWeightOffsets oDiag);
        LeafOnlyTbOffsetsFromBase(embedEnd + tbFloats, in arch, out LeafOnlyTbWeightOffsets oOff);

        EnsureLeafOnlyLayer1GpuBuffers(nPad, d, lFull, kMax, mOff, lsOff);

        leafOnlyLayer1Shader.SetInt("leafDModel", d);
        leafOnlyLayer1Shader.SetInt("leafNumHeads", nh);
        leafOnlyLayer1Shader.SetInt("leafDHead", dh);
        leafOnlyLayer1Shader.SetInt("leafKActive", kAct);
        leafOnlyLayer1Shader.SetInt("leafKMax", kMax);
        leafOnlyLayer1Shader.SetInt("leafMOff", mOff);
        leafOnlyLayer1Shader.SetInt("leafLeafSize", lFull);
        leafOnlyLayer1Shader.SetInt("leafLsOff", lsOff);
        leafOnlyLayer1Shader.SetInt("leafOffPool", 4);
        leafOnlyLayer1Shader.SetInt("leafL1MlpHidden", d * 4);

        LeafOnlyLayer1BindAll(LeafOnlyDiagEdgeFeatsBuffer);

        int nTok = nPad * d;
        int attnDiag = kAct * nh * lFull;
        int edgeSliceDiag = lFull * lFull * 4;

        leafOnlyLayer1Shader.SetInt("leafL1NTok", nPad);
        leafOnlyLayer1Shader.SetInt("leafL1NSeq", kAct);
        leafOnlyLayer1Shader.SetInt("leafL1SeqLen", lFull);
        leafOnlyLayer1Shader.SetInt("leafL1AttnTotal", attnDiag);
        leafOnlyLayer1Shader.SetInt("leafL1EdgeFloatsPerSeq", edgeSliceDiag);

        LeafOnlyLayer1SetTbInts(in oDiag);
        LeafOnlyLayer1RebindMainStreamToDiag();

        leafOnlyLayer1Shader.Dispatch(leafL1KCopyFromEnc, LeafOnlyLayer1Groups256(nTok), 1, 1);
        leafOnlyLayer1Shader.Dispatch(leafL1KCopyAToSkipA, LeafOnlyLayer1Groups256(nTok), 1, 1);
        leafOnlyLayer1Shader.Dispatch(leafL1KLayerNorm, LeafOnlyLayer1Groups256(nPad), 1, 1);
        leafOnlyLayer1Shader.Dispatch(leafL1KQkv, LeafOnlyLayer1Groups256(nPad), 1, 1);
        leafOnlyLayer1Shader.SetInt("leafL1ClearCount", nTok);
        leafOnlyLayer1Shader.Dispatch(leafL1KClear, LeafOnlyLayer1Groups256(nTok), 1, 1);
        leafOnlyLayer1Shader.Dispatch(leafL1KAttnRow, LeafOnlyLayer1Groups256(attnDiag), 1, 1);
        leafOnlyLayer1Shader.Dispatch(leafL1KProjAddSkip, LeafOnlyLayer1Groups256(nPad), 1, 1);
        leafOnlyLayer1Shader.Dispatch(leafL1KCopyAToSkipB, LeafOnlyLayer1Groups256(nTok), 1, 1);
        leafOnlyLayer1Shader.Dispatch(leafL1KFfn, LeafOnlyLayer1Groups256(nPad), 1, 1);

        var hDiagHost = new float[nTok];
        leafL1BufB.GetData(hDiagHost, 0, 0, nTok);
        LeafOnlyParitySummarize(phase, "h_diag_after_layer0", hDiagHost, nTok);
        LeafOnlyParityHead(phase, "h_diag_after_layer0", hDiagHost, nTok, 16);
        LeafOnlyParityHead(phase, "h_diag_after_layer0", hDiagHost, nTok, 32);
        if (nTok > 64)
            LeafOnlyParityHeadAt(phase, "h_diag_after_layer0", hDiagHost, nTok, nTok / 2, 48);

        if (mOff <= 0)
        {
            var invC = CultureInfo.InvariantCulture;
            Debug.Log($"[LeafOnlyParity] phase={phase} tensor=layer1_gpu_meta path=gpu K={kAct.ToString(invC)} L_diag={lFull.ToString(invC)} M_off=0");
            return;
        }

        int hPadFloats = kMax * lFull * d;
        leafOnlyLayer1Shader.Dispatch(leafL1KClearPad, LeafOnlyLayer1Groups256(hPadFloats), 1, 1);
        leafOnlyLayer1Shader.Dispatch(leafL1KCopyPad, LeafOnlyLayer1Groups256(kAct * lFull * d), 1, 1);
        int stripFloats = mOff * lFull * d;
        leafOnlyLayer1Shader.Dispatch(leafL1KStrip, LeafOnlyLayer1Groups256(stripFloats), 1, 1);
        int nOffTok = mOff * lsOff * d;
        leafOnlyLayer1Shader.Dispatch(leafL1KStripPool, LeafOnlyLayer1Groups256(nOffTok), 1, 1);
        int poolCells = mOff * lsOff * lsOff;
        leafOnlyLayer1Shader.Dispatch(leafL1KPoolOffEdge, LeafOnlyLayer1Groups256(poolCells), 1, 1);

        int attnOff = mOff * nh * lsOff;
        int edgeSliceOff = lsOff * lsOff * 4;

        leafOnlyLayer1Shader.SetBuffer(leafL1KAttnRow, "leafL1EdgeFeats", leafL1OffEdgeLs);
        LeafOnlyLayer1SetTbInts(in oOff);
        LeafOnlyLayer1RebindMainStreamToOff();

        leafOnlyLayer1Shader.SetInt("leafL1NTok", mOff * lsOff);
        leafOnlyLayer1Shader.SetInt("leafL1NSeq", mOff);
        leafOnlyLayer1Shader.SetInt("leafL1SeqLen", lsOff);
        leafOnlyLayer1Shader.SetInt("leafL1AttnTotal", attnOff);
        leafOnlyLayer1Shader.SetInt("leafL1EdgeFloatsPerSeq", edgeSliceOff);

        leafOnlyLayer1Shader.Dispatch(leafL1KCopyAToSkipA, LeafOnlyLayer1Groups256(nOffTok), 1, 1);
        leafOnlyLayer1Shader.Dispatch(leafL1KLayerNorm, LeafOnlyLayer1Groups256(mOff * lsOff), 1, 1);
        leafOnlyLayer1Shader.Dispatch(leafL1KQkv, LeafOnlyLayer1Groups256(mOff * lsOff), 1, 1);
        leafOnlyLayer1Shader.SetInt("leafL1ClearCount", nOffTok);
        leafOnlyLayer1Shader.Dispatch(leafL1KClear, LeafOnlyLayer1Groups256(nOffTok), 1, 1);
        leafOnlyLayer1Shader.Dispatch(leafL1KAttnRow, LeafOnlyLayer1Groups256(attnOff), 1, 1);
        leafOnlyLayer1Shader.Dispatch(leafL1KProjAddSkip, LeafOnlyLayer1Groups256(mOff * lsOff), 1, 1);
        leafOnlyLayer1Shader.Dispatch(leafL1KCopyAToSkipB, LeafOnlyLayer1Groups256(nOffTok), 1, 1);
        leafOnlyLayer1Shader.Dispatch(leafL1KFfn, LeafOnlyLayer1Groups256(mOff * lsOff), 1, 1);

        var offHost = new float[nOffTok];
        leafL1OffBufB.GetData(offHost, 0, 0, nOffTok);
        LeafOnlyParitySummarize(phase, "off_stream_after_layer0", offHost, nOffTok);
        LeafOnlyParityHead(phase, "off_stream_after_layer0", offHost, nOffTok, 16);
        LeafOnlyParityHead(phase, "off_stream_after_layer0", offHost, nOffTok, 32);
        if (nOffTok > 64)
            LeafOnlyParityHeadAt(phase, "off_stream_after_layer0", offHost, nOffTok, nOffTok / 2, 48);

        LeafOnlyLayer1RebindMainStreamToDiag();
        leafOnlyLayer1Shader.SetBuffer(leafL1KAttnRow, "leafL1EdgeFeats", LeafOnlyDiagEdgeFeatsBuffer);

        var inv = CultureInfo.InvariantCulture;
        Debug.Log(
            $"[LeafOnlyParity] phase={phase} tensor=layer1_gpu_meta path=gpu K={kAct.ToString(inv)} L_diag={lFull.ToString(inv)} " +
            $"L_off={lsOff.ToString(inv)} M_off={mOff.ToString(inv)} d_model={d.ToString(inv)} heads={nh.ToString(inv)}");
    }
}
