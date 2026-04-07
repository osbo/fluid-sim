using System.Globalization;
using UnityEngine;

/// <summary>
/// GPU first transformer layer (diag + H-strip + off) via <c>LeafOnlyLayer1.compute</c> for
/// <c>[LeafOnlyParity]</c> logging. Needs highway=0, dense L×L layout, and compatible checkpoint shape;
/// otherwise layer 1 parity lines are skipped (no CPU reference path).
/// </summary>
public partial class FluidSimulator
{
    public const int LeafOnlyLayer1DModelMax = 128;

    [Tooltip("Optional: LeafOnlyLayer1.compute — GPU layer 0 after embed. If null, layer 1 parity logs are skipped.")]
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
    private int leafL1KBuildHeadUDiag;
    private int leafL1KPrecondDiagUUt;
    private int leafL1KBuildHeadUVOff;
    private int leafL1KPrecondOffUVt;
    private int leafL1KPrecondNodeU = -1;
    private int leafL1KPrecondNodeV = -1;
    private int leafL1KPrecondJacobi = -1;

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
    private ComputeBuffer leafL1HeadUDiag;
    private ComputeBuffer leafL1PrecondDiag;
    private ComputeBuffer leafL1HeadUOff;
    private ComputeBuffer leafL1HeadVOff;
    private ComputeBuffer leafL1PrecondOff;
    private ComputeBuffer leafL1PrecondU;
    private ComputeBuffer leafL1PrecondV;
    private ComputeBuffer leafL1PrecondJac;

    private int leafL1CapNPad;
    private int leafL1CapDm;
    private int leafL1CapMOff;
    private int leafL1CapKMax;
    private int leafL1CapLFull;
    private int leafL1CapLaD;
    private int leafL1CapLaO;
    private static bool s_warnedLeafOnlyLayer1GpuMissing;
    private static bool s_warnedLeafOnlyLayer1GpuSkip;
    private static bool s_warnedLeafOnlyLayer1GpuNoWeights;
    private static bool s_warnedLeafOnlyLayer1GpuDim;
    private static bool s_warnedLeafOnlyLayer1GpuPadLayout;
    private static bool s_warnedLeafOnlyLayer1GpuEdgeBuf;
    private static bool s_warnedLeafOnlyLayer1GpuWeightShort;
    private static bool s_warnedLeafOnlyLayer1GpuLeafApply;
    private static bool s_warnedLeafOnlyLayer1HeadKernelsMissing;
    private static bool s_warnedLeafOnlyLayer1TailKernelsMissing;
    private static bool s_warnedLeafOnlyPrecondPackedBufferSize;

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
        leafL1KBuildHeadUDiag = leafOnlyLayer1Shader.FindKernel("LeafOnly_L1_BuildHeadUDiag");
        leafL1KPrecondDiagUUt = leafOnlyLayer1Shader.FindKernel("LeafOnly_L1_PrecondDiagUUt");
        leafL1KBuildHeadUVOff = leafOnlyLayer1Shader.FindKernel("LeafOnly_L1_BuildHeadUVOff");
        leafL1KPrecondOffUVt = leafOnlyLayer1Shader.FindKernel("LeafOnly_L1_PrecondOffUVt");
        leafL1KPrecondNodeU = leafOnlyLayer1Shader.FindKernel("LeafOnly_L1_PrecondNodeU");
        leafL1KPrecondNodeV = leafOnlyLayer1Shader.FindKernel("LeafOnly_L1_PrecondNodeV");
        leafL1KPrecondJacobi = leafOnlyLayer1Shader.FindKernel("LeafOnly_L1_PrecondJacobi");
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
        leafL1HeadUDiag?.Release();
        leafL1HeadUDiag = null;
        leafL1PrecondDiag?.Release();
        leafL1PrecondDiag = null;
        leafL1HeadUOff?.Release();
        leafL1HeadUOff = null;
        leafL1HeadVOff?.Release();
        leafL1HeadVOff = null;
        leafL1PrecondOff?.Release();
        leafL1PrecondOff = null;
        leafL1PrecondU?.Release();
        leafL1PrecondU = null;
        leafL1PrecondV?.Release();
        leafL1PrecondV = null;
        leafL1PrecondJac?.Release();
        leafL1PrecondJac = null;
        leafL1CapNPad = 0;
        leafL1CapDm = 0;
        leafL1CapMOff = 0;
        leafL1CapKMax = 0;
        leafL1CapLFull = 0;
        leafL1CapLaD = 0;
        leafL1CapLaO = 0;
    }

    private void EnsureLeafOnlyLayer1GpuBuffers(int nPad, int d, int lFull, int kMax, int mOff, int lsOff, int laD, int laO)
    {
        if (leafL1BufA != null && leafL1CapNPad >= nPad && leafL1CapDm >= d
            && leafL1CapMOff >= mOff && leafL1CapKMax >= kMax && leafL1CapLFull == lFull
            && leafL1CapLaD >= laD && leafL1CapLaO >= laO)
            return;

        ReleaseLeafOnlyLayer1GpuBuffers();

        int kAct = nPad / lFull;
        int nDiagHead = kAct * laD * laD;
        int nOffSq = Mathf.Max(1, mOff * laO * laO);

        int nTok = nPad * d;
        int qkv = nPad * 3 * d;
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
        leafL1HeadUDiag = new ComputeBuffer(nDiagHead, sizeof(float));
        leafL1PrecondDiag = new ComputeBuffer(nDiagHead, sizeof(float));
        leafL1HeadUOff = new ComputeBuffer(nOffSq, sizeof(float));
        leafL1HeadVOff = new ComputeBuffer(nOffSq, sizeof(float));
        leafL1PrecondOff = new ComputeBuffer(nOffSq, sizeof(float));
        int nUv = nPad * laO;
        leafL1PrecondU = new ComputeBuffer(Mathf.Max(1, nUv), sizeof(float));
        leafL1PrecondV = new ComputeBuffer(Mathf.Max(1, nUv), sizeof(float));
        leafL1PrecondJac = new ComputeBuffer(Mathf.Max(1, nPad), sizeof(float));
        leafL1CapNPad = nPad;
        leafL1CapDm = d;
        leafL1CapMOff = mOff;
        leafL1CapKMax = kMax;
        leafL1CapLFull = lFull;
        leafL1CapLaD = laD;
        leafL1CapLaO = laO;

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

    private void LeafOnlyLayer1BindHeadKernelBuildUDiag()
    {
        leafOnlyLayer1Shader.SetBuffer(leafL1KBuildHeadUDiag, "leafOnlyWeights", leafOnlyWeightsFloatBuffer);
        leafOnlyLayer1Shader.SetBuffer(leafL1KBuildHeadUDiag, "leafL1BufB", leafL1BufB);
        leafOnlyLayer1Shader.SetBuffer(leafL1KBuildHeadUDiag, "leafL1HeadUDiag", leafL1HeadUDiag);
    }

    private void LeafOnlyLayer1BindHeadKernelPrecondDiag()
    {
        leafOnlyLayer1Shader.SetBuffer(leafL1KPrecondDiagUUt, "leafL1HeadUDiag", leafL1HeadUDiag);
        leafOnlyLayer1Shader.SetBuffer(leafL1KPrecondDiagUUt, "leafL1PrecondDiag", leafL1PrecondDiag);
    }

    private void LeafOnlyLayer1BindHeadKernelBuildUVOff()
    {
        leafOnlyLayer1Shader.SetBuffer(leafL1KBuildHeadUVOff, "leafOnlyWeights", leafOnlyWeightsFloatBuffer);
        leafOnlyLayer1Shader.SetBuffer(leafL1KBuildHeadUVOff, "leafL1OffBufB", leafL1OffBufB);
        leafOnlyLayer1Shader.SetBuffer(leafL1KBuildHeadUVOff, "leafL1HeadUOff", leafL1HeadUOff);
        leafOnlyLayer1Shader.SetBuffer(leafL1KBuildHeadUVOff, "leafL1HeadVOff", leafL1HeadVOff);
    }

    private void LeafOnlyLayer1BindHeadKernelPrecondOff()
    {
        leafOnlyLayer1Shader.SetBuffer(leafL1KPrecondOffUVt, "leafL1HeadUOff", leafL1HeadUOff);
        leafOnlyLayer1Shader.SetBuffer(leafL1KPrecondOffUVt, "leafL1HeadVOff", leafL1HeadVOff);
        leafOnlyLayer1Shader.SetBuffer(leafL1KPrecondOffUVt, "leafL1PrecondOff", leafL1PrecondOff);
    }

    private void LeafOnlyLayer1SetHeadWeightInts(
        int leafW0,
        int leafB0,
        int leafW1,
        int leafB1,
        int offUW0,
        int offUB0,
        int offUW1,
        int offUB1,
        int offVW0,
        int offVB0,
        int offVW1,
        int offVB1)
    {
        leafOnlyLayer1Shader.SetInt("headLeafW0", leafW0);
        leafOnlyLayer1Shader.SetInt("headLeafB0", leafB0);
        leafOnlyLayer1Shader.SetInt("headLeafW1", leafW1);
        leafOnlyLayer1Shader.SetInt("headLeafB1", leafB1);
        leafOnlyLayer1Shader.SetInt("headOffUW0", offUW0);
        leafOnlyLayer1Shader.SetInt("headOffUB0", offUB0);
        leafOnlyLayer1Shader.SetInt("headOffUW1", offUW1);
        leafOnlyLayer1Shader.SetInt("headOffUB1", offUB1);
        leafOnlyLayer1Shader.SetInt("headOffVW0", offVW0);
        leafOnlyLayer1Shader.SetInt("headOffVB0", offVB0);
        leafOnlyLayer1Shader.SetInt("headOffVW1", offVW1);
        leafOnlyLayer1Shader.SetInt("headOffVB1", offVB1);
    }

    private void LeafOnlyLayer1SetTailWeightInts(int nodeUW, int nodeUB, int nodeVW, int nodeVB, int jacW, int jacB)
    {
        leafOnlyLayer1Shader.SetInt("headNodeUW", nodeUW);
        leafOnlyLayer1Shader.SetInt("headNodeUB", nodeUB);
        leafOnlyLayer1Shader.SetInt("headNodeVW", nodeVW);
        leafOnlyLayer1Shader.SetInt("headNodeVB", nodeVB);
        leafOnlyLayer1Shader.SetInt("headJacobiW", jacW);
        leafOnlyLayer1Shader.SetInt("headJacobiB", jacB);
    }

    private void LeafOnlyLayer1BindPrecondTailKernels()
    {
        foreach (int k in new[] { leafL1KPrecondNodeU, leafL1KPrecondNodeV, leafL1KPrecondJacobi })
        {
            leafOnlyLayer1Shader.SetBuffer(k, "leafOnlyWeights", leafOnlyWeightsFloatBuffer);
            leafOnlyLayer1Shader.SetBuffer(k, "leafL1BufB", leafL1BufB);
        }

        leafOnlyLayer1Shader.SetBuffer(leafL1KPrecondNodeU, "leafL1PrecondU", leafL1PrecondU);
        leafOnlyLayer1Shader.SetBuffer(leafL1KPrecondNodeV, "leafL1PrecondV", leafL1PrecondV);
        leafOnlyLayer1Shader.SetBuffer(leafL1KPrecondJacobi, "leafL1PrecondJac", leafL1PrecondJac);
    }

    private void LeafOnlyLayer1LogPrecondNodeJacParity(string phase, int nPad, int laO, bool logParityTensors = true)
    {
        if (!logParityTensors)
            return;

        int nUv = nPad * laO;
        var u = new float[nUv];
        leafL1PrecondU.GetData(u, 0, 0, nUv);
        LeafOnlyParitySummarize(phase, "precondU", u, nUv);
        LeafOnlyParityHead(phase, "precondU", u, nUv, 16);
        if (nUv > 16)
            LeafOnlyParityHead(phase, "precondU", u, nUv, 32);
        if (nUv > 64)
            LeafOnlyParityHeadAt(phase, "precondU", u, nUv, nUv / 2, 48);

        var v = new float[nUv];
        leafL1PrecondV.GetData(v, 0, 0, nUv);
        LeafOnlyParitySummarize(phase, "precondV", v, nUv);
        LeafOnlyParityHead(phase, "precondV", v, nUv, 16);
        if (nUv > 16)
            LeafOnlyParityHead(phase, "precondV", v, nUv, 32);
        if (nUv > 64)
            LeafOnlyParityHeadAt(phase, "precondV", v, nUv, nUv / 2, 48);

        var jac = new float[nPad];
        leafL1PrecondJac.GetData(jac, 0, 0, nPad);
        LeafOnlyParitySummarize(phase, "precondJac", jac, nPad);
        LeafOnlyParityHead(phase, "precondJac", jac, nPad, 16);
        if (nPad > 16)
            LeafOnlyParityHead(phase, "precondJac", jac, nPad, 32);
        if (nPad > 64)
            LeafOnlyParityHeadAt(phase, "precondJac", jac, nPad, nPad / 2, 48);
    }

    /// <summary>
    /// Concatenates GPU precond tensors in PyTorch <c>LeafOnlyNet.forward</c> order (diag | off | U | V | jac),
    /// logs <c>packed_precond</c> for <c>InspectParity.py</c>, and uploads to <see cref="leafOnlyPrecondPackedBuffer"/> when its length matches.
    /// </summary>
    private void LeafOnlyLayer1PackLogAndUploadPackedPrecond(
        string phase,
        int kAct,
        int nPad,
        int laD,
        int laO,
        int mOff,
        bool canGpuHeads,
        bool canGpuTail,
        bool logParityTensors = true)
    {
        if (!canGpuHeads || !canGpuTail)
            return;

        int nPreD = kAct * laD * laD;
        int nPreO = mOff * laO * laO;
        int nUv = nPad * laO;
        int nJac = nPad;
        int total = nPreD + nPreO + nUv + nUv + nJac;

        var packed = new float[total];
        leafL1PrecondDiag.GetData(packed, 0, 0, nPreD);
        if (nPreO > 0)
            leafL1PrecondOff.GetData(packed, nPreD, 0, nPreO);
        leafL1PrecondU.GetData(packed, nPreD + nPreO, 0, nUv);
        leafL1PrecondV.GetData(packed, nPreD + nPreO + nUv, 0, nUv);
        leafL1PrecondJac.GetData(packed, nPreD + nPreO + nUv + nUv, 0, nJac);

        if (logParityTensors)
        {
            LeafOnlyParitySummarize(phase, "packed_precond", packed, total);
            LeafOnlyParityHead(phase, "packed_precond", packed, total, 16);
            if (total > 16)
                LeafOnlyParityHead(phase, "packed_precond", packed, total, 32);
        }

        bool usedPublic = leafOnlyPrecondPackedBuffer != null && leafOnlyPrecondPackedBuffer.count == total;
        LeafOnlyUploadPackedPrecondFromHost(packed, total);
        if (!usedPublic && leafOnlyPrecondPackedBuffer != null && leafOnlyPrecondPackedBuffer.count != total
            && !s_warnedLeafOnlyPrecondPackedBufferSize)
        {
            s_warnedLeafOnlyPrecondPackedBufferSize = true;
            Debug.Log(
                "[LeafOnlyParity] leafOnlyPrecondPackedBuffer length mismatch (expected " + total
                + " floats); GPU precond apply uses an internal packed buffer. Clear the field or call TryAllocLeafOnlyPrecondPackedBuffer() to assign a matching buffer.");
        }
    }

    /// <summary>Runs GPU layer 1. When <paramref name="logParityTensors"/>, logs <c>h_diag_after_layer0</c> / <c>off_stream_after_layer0</c> parity lines.</summary>
    private void DispatchLeafOnlyLayer1GpuForwardAndLogParity(int nPad, bool logParityTensors = true)
    {
        const string phase = "unity";
        if (leafOnlyLayer1Shader == null)
        {
            if (!s_warnedLeafOnlyLayer1GpuMissing)
            {
                s_warnedLeafOnlyLayer1GpuMissing = true;
                Debug.Log(
                    "[LeafOnlyParity] GPU layer1 skipped (assign FluidSimulator.leafOnlyLayer1Shader = LeafOnlyLayer1.compute). No layer 1 parity logs.");
            }

            return;
        }

        if (!LeafOnlyWeightsLoadedSuccessfully || leafOnlyWeightsFloatBuffer == null || leafOnlyTokenAfterEnc == null)
        {
            if (!s_warnedLeafOnlyLayer1GpuNoWeights)
            {
                s_warnedLeafOnlyLayer1GpuNoWeights = true;
                Debug.Log("[LeafOnlyParity] GPU layer1 skipped (weights not loaded or embed/token buffers missing). No layer 1 parity logs.");
            }

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

            return;
        }

        int d = arch.DModel;
        if (d > LeafOnlyLayer1DModelMax || arch.NumHeads < 1 || d % arch.NumHeads != 0)
        {
            if (!s_warnedLeafOnlyLayer1GpuDim)
            {
                s_warnedLeafOnlyLayer1GpuDim = true;
                Debug.Log(
                    "[LeafOnlyParity] GPU layer1 skipped (d_model / num_heads invalid or d_model > LeafOnlyLayer1DModelMax). No layer 1 parity logs.");
            }

            return;
        }

        int lFull = LeafOnlyLeafSize;
        int lsOff = lFull / 4;
        if (nPad <= 0 || nPad % lFull != 0 || lFull != arch.LeafSize)
        {
            if (!s_warnedLeafOnlyLayer1GpuPadLayout)
            {
                s_warnedLeafOnlyLayer1GpuPadLayout = true;
                Debug.Log("[LeafOnlyParity] GPU layer1 skipped (nPad vs LeafOnlyLeafSize / checkpoint LeafSize mismatch). No layer 1 parity logs.");
            }

            return;
        }

        int kAct = nPad / lFull;
        int nh = arch.NumHeads;
        int dh = d / nh;
        int mOff = LeafOnlyHMatrixStatic.NumOffBlocks;
        int kMax = LeafOnlyHMatrixStatic.MaxNumLeaves;
        if (LeafOnlyDiagEdgeFeatsBuffer == null || (mOff > 0 && LeafOnlyOffEdgeFeatsBuffer == null))
        {
            if (!s_warnedLeafOnlyLayer1GpuEdgeBuf)
            {
                s_warnedLeafOnlyLayer1GpuEdgeBuf = true;
                Debug.Log("[LeafOnlyParity] GPU layer1 skipped (diag/off edge feature buffers missing). No layer 1 parity logs.");
            }

            return;
        }

        int embedEnd = LeafOnlyEmbedPhaseFloatCount(in arch, LeafOnlyGlobalFeaturesDim);
        int tbFloats = LeafOnlyTransformerBlockFloatCount(in arch);
        if (tbFloats < 0 || leafOnlyWeightsFloatBuffer.count < embedEnd + 2 * tbFloats)
        {
            if (!s_warnedLeafOnlyLayer1GpuWeightShort)
            {
                s_warnedLeafOnlyLayer1GpuWeightShort = true;
                Debug.Log("[LeafOnlyParity] GPU layer1 skipped (weight float buffer too short for two transformer blocks). No layer 1 parity logs.");
            }

            return;
        }

        LeafOnlyTbOffsetsFromBase(embedEnd, in arch, out LeafOnlyTbWeightOffsets oDiag);
        LeafOnlyTbOffsetsFromBase(embedEnd + tbFloats, in arch, out LeafOnlyTbWeightOffsets oOff);

        int laD = arch.LeafApplyDiag;
        int laO = arch.LeafApplyOff;
        if (laD * LeafOnlyCpuDiagTokenPool != lFull || laO * LeafOnlyCpuOffTokenPool != lFull)
        {
            if (!s_warnedLeafOnlyLayer1GpuLeafApply)
            {
                s_warnedLeafOnlyLayer1GpuLeafApply = true;
                Debug.Log("[LeafOnlyParity] GPU layer1 skipped (LeafApplyDiag/Off vs leaf size inconsistent). No layer 1 parity logs.");
            }

            return;
        }

        int hLeafW0 = 0, hLeafB0 = 0, hLeafW1 = 0, hLeafB1 = 0;
        int hOffUW0 = 0, hOffUB0 = 0, hOffUW1 = 0, hOffUB1 = 0;
        int hOffVW0 = 0, hOffVB0 = 0, hOffVW1 = 0, hOffVB1 = 0;
        bool canGpuHeads = arch.MlpHeads == 1
            && leafL1KBuildHeadUDiag >= 0
            && leafL1KPrecondDiagUUt >= 0
            && (mOff <= 0 || (leafL1KBuildHeadUVOff >= 0 && leafL1KPrecondOffUVt >= 0))
            && LeafOnlyTryGetPrecondHeadWeightOffsets(
                in arch,
                LeafOnlyGlobalFeaturesDim,
                leafOnlyWeightsFloatBuffer.count,
                out hLeafW0,
                out hLeafB0,
                out hLeafW1,
                out hLeafB1,
                out hOffUW0,
                out hOffUB0,
                out hOffUW1,
                out hOffUB1,
                out hOffVW0,
                out hOffVB0,
                out hOffVW1,
                out hOffVB1);

        int tailNodeUW = 0, tailNodeUB = 0, tailNodeVW = 0, tailNodeVB = 0, tailJacW = 0, tailJacB = 0;
        bool tailWeightsOk = arch.MlpHeads == 1
            && LeafOnlyTryGetNodeJacobiWeightOffsets(
                in arch,
                LeafOnlyGlobalFeaturesDim,
                leafOnlyWeightsFloatBuffer.count,
                out tailNodeUW,
                out tailNodeUB,
                out tailNodeVW,
                out tailNodeVB,
                out tailJacW,
                out tailJacB);
        bool tailKernelsOk = leafL1KPrecondNodeU >= 0
            && leafL1KPrecondNodeV >= 0
            && leafL1KPrecondJacobi >= 0;
        bool canGpuTail = arch.MlpHeads == 1 && tailWeightsOk && tailKernelsOk;

        if (arch.MlpHeads == 1 && !canGpuHeads && !s_warnedLeafOnlyLayer1HeadKernelsMissing)
        {
            s_warnedLeafOnlyLayer1HeadKernelsMissing = true;
            Debug.Log(
                "[LeafOnlyParity] Skipping GPU precondDiag/precondOff parity logs (add head kernels in LeafOnlyLayer1.compute or fix checkpoint MlpHeads/weights).");
        }

        if (arch.MlpHeads == 1 && !canGpuTail && !s_warnedLeafOnlyLayer1TailKernelsMissing)
        {
            s_warnedLeafOnlyLayer1TailKernelsMissing = true;
            Debug.Log(
                "[LeafOnlyParity] Skipping GPU precondU/precondV/precondJac parity logs: " +
                $"tail_weights_ok={tailWeightsOk} (need room for node_u/node_v/jacobi_gate after leaf_head) " +
                $"tail_kernels_ok={tailKernelsOk} (kernel indices U={leafL1KPrecondNodeU} V={leafL1KPrecondNodeV} Jac={leafL1KPrecondJacobi}; " +
                "-1 means LeafOnlyLayer1.compute missing kernels or failed to compile — reimport the asset / check Console).");
        }

        EnsureLeafOnlyLayer1GpuBuffers(nPad, d, lFull, kMax, mOff, lsOff, laD, laO);

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

        if (canGpuTail)
        {
            leafOnlyLayer1Shader.SetInt("leafLaOff", laO);
            LeafOnlyLayer1SetTailWeightInts(tailNodeUW, tailNodeUB, tailNodeVW, tailNodeVB, tailJacW, tailJacB);
            LeafOnlyLayer1BindPrecondTailKernels();
            leafOnlyLayer1Shader.Dispatch(leafL1KPrecondNodeU, LeafOnlyLayer1Groups256(nPad * laO), 1, 1);
            leafOnlyLayer1Shader.Dispatch(leafL1KPrecondNodeV, LeafOnlyLayer1Groups256(nPad * laO), 1, 1);
            leafOnlyLayer1Shader.Dispatch(leafL1KPrecondJacobi, LeafOnlyLayer1Groups256(nPad), 1, 1);
        }

        if (canGpuHeads)
        {
            leafOnlyLayer1Shader.SetInt("leafLaDiag", laD);
            leafOnlyLayer1Shader.SetInt("leafL1DiagStridePerLeaf", lFull);
            LeafOnlyLayer1SetHeadWeightInts(
                hLeafW0, hLeafB0, hLeafW1, hLeafB1,
                hOffUW0, hOffUB0, hOffUW1, hOffUB1,
                hOffVW0, hOffVB0, hOffVW1, hOffVB1);
            LeafOnlyLayer1BindHeadKernelBuildUDiag();
            leafOnlyLayer1Shader.Dispatch(leafL1KBuildHeadUDiag, LeafOnlyLayer1Groups256(kAct * laD), 1, 1);
            LeafOnlyLayer1BindHeadKernelPrecondDiag();
            leafOnlyLayer1Shader.Dispatch(leafL1KPrecondDiagUUt, LeafOnlyLayer1Groups256(kAct * laD * laD), 1, 1);
        }

        if (logParityTensors)
        {
            var hDiagHost = new float[nTok];
            leafL1BufB.GetData(hDiagHost, 0, 0, nTok);
            LeafOnlyParitySummarize(phase, "h_diag_after_layer0", hDiagHost, nTok);
            LeafOnlyParityHead(phase, "h_diag_after_layer0", hDiagHost, nTok, 16);
            LeafOnlyParityHead(phase, "h_diag_after_layer0", hDiagHost, nTok, 32);
            if (nTok > 64)
                LeafOnlyParityHeadAt(phase, "h_diag_after_layer0", hDiagHost, nTok, nTok / 2, 48);
        }

        if (mOff <= 0)
        {
            if (canGpuHeads)
            {
                int nPreD0 = kAct * laD * laD;
                if (logParityTensors)
                {
                    var preD0 = new float[nPreD0];
                    leafL1PrecondDiag.GetData(preD0, 0, 0, nPreD0);
                    LeafOnlyParitySummarize(phase, "precondDiag", preD0, nPreD0);
                    LeafOnlyParityHead(phase, "precondDiag", preD0, nPreD0, 16);
                    LeafOnlyParityHead(phase, "precondDiag", preD0, nPreD0, 32);
                    if (nPreD0 > 64)
                        LeafOnlyParityHeadAt(phase, "precondDiag", preD0, nPreD0, nPreD0 / 2, 48);
                    Debug.Log(
                        $"[LeafOnlyParity] phase={phase} tensor=leaf_blocks_meta path=gpu K={kAct} La_diag={laD} M_off=0 (UUt)");
                }
            }

            if (canGpuTail)
                LeafOnlyLayer1LogPrecondNodeJacParity(phase, nPad, laO, logParityTensors);

            LeafOnlyLayer1PackLogAndUploadPackedPrecond(phase, kAct, nPad, laD, laO, 0, canGpuHeads, canGpuTail, logParityTensors);

            if (logParityTensors)
            {
                var invC = CultureInfo.InvariantCulture;
                Debug.Log($"[LeafOnlyParity] phase={phase} tensor=layer1_gpu_meta path=gpu K={kAct.ToString(invC)} L_diag={lFull.ToString(invC)} M_off=0");
            }

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

        if (canGpuHeads)
        {
            leafOnlyLayer1Shader.SetInt("leafLaOff", laO);
            LeafOnlyLayer1BindHeadKernelBuildUVOff();
            leafOnlyLayer1Shader.Dispatch(leafL1KBuildHeadUVOff, LeafOnlyLayer1Groups256(mOff * laO), 1, 1);
            LeafOnlyLayer1BindHeadKernelPrecondOff();
            leafOnlyLayer1Shader.Dispatch(leafL1KPrecondOffUVt, LeafOnlyLayer1Groups256(mOff * laO * laO), 1, 1);
        }

        if (logParityTensors)
        {
            var offHost = new float[nOffTok];
            leafL1OffBufB.GetData(offHost, 0, 0, nOffTok);
            LeafOnlyParitySummarize(phase, "off_stream_after_layer0", offHost, nOffTok);
            LeafOnlyParityHead(phase, "off_stream_after_layer0", offHost, nOffTok, 16);
            LeafOnlyParityHead(phase, "off_stream_after_layer0", offHost, nOffTok, 32);
            if (nOffTok > 64)
                LeafOnlyParityHeadAt(phase, "off_stream_after_layer0", offHost, nOffTok, nOffTok / 2, 48);
        }

        if (canGpuHeads)
        {
            int nPreD = kAct * laD * laD;
            int nPreO = mOff * laO * laO;
            if (logParityTensors)
            {
                var preDArr = new float[nPreD];
                leafL1PrecondDiag.GetData(preDArr, 0, 0, nPreD);
                LeafOnlyParitySummarize(phase, "precondDiag", preDArr, nPreD);
                LeafOnlyParityHead(phase, "precondDiag", preDArr, nPreD, 16);
                LeafOnlyParityHead(phase, "precondDiag", preDArr, nPreD, 32);
                if (nPreD > 64)
                    LeafOnlyParityHeadAt(phase, "precondDiag", preDArr, nPreD, nPreD / 2, 48);
                var preOArr = new float[nPreO];
                leafL1PrecondOff.GetData(preOArr, 0, 0, nPreO);
                LeafOnlyParitySummarize(phase, "precondOff", preOArr, nPreO);
                LeafOnlyParityHead(phase, "precondOff", preOArr, nPreO, 16);
                if (nPreO > 16)
                    LeafOnlyParityHead(phase, "precondOff", preOArr, nPreO, 32);
                if (nPreO > 64)
                    LeafOnlyParityHeadAt(phase, "precondOff", preOArr, nPreO, nPreO / 2, 48);
                Debug.Log(
                    $"[LeafOnlyParity] phase={phase} tensor=leaf_blocks_meta path=gpu K={kAct} La_diag={laD} La_off={laO} M_off={mOff} (UUt / UVt)");
            }
        }

        if (canGpuTail)
            LeafOnlyLayer1LogPrecondNodeJacParity(phase, nPad, laO, logParityTensors);

        LeafOnlyLayer1PackLogAndUploadPackedPrecond(phase, kAct, nPad, laD, laO, mOff, canGpuHeads, canGpuTail, logParityTensors);

        LeafOnlyLayer1RebindMainStreamToDiag();
        leafOnlyLayer1Shader.SetBuffer(leafL1KAttnRow, "leafL1EdgeFeats", LeafOnlyDiagEdgeFeatsBuffer);

        if (logParityTensors)
        {
            var inv = CultureInfo.InvariantCulture;
            Debug.Log(
                $"[LeafOnlyParity] phase={phase} tensor=layer1_gpu_meta path=gpu K={kAct.ToString(inv)} L_diag={lFull.ToString(inv)} " +
                $"L_off={lsOff.ToString(inv)} M_off={mOff.ToString(inv)} d_model={d.ToString(inv)} heads={nh.ToString(inv)}");
        }
    }
}
