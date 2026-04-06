using System;
using System.Globalization;
using UnityEngine;

/// <summary>
/// CPU mirror of the first transformer layer (diag block + H-off strip + off block) for
/// <c>[LeafOnlyParity]</c> vs <c>InspectParity.py</c>. Matches <c>leafonly/architecture.py</c>
/// <c>_apply_transformer_stacks</c> for <c>highway_ffn=False</c>, dense diag/off, layout L×L,
/// eager attention (manual softmax), checkpoint order in <c>checkpoint._write_transformer_block</c>.
/// </summary>
public partial class FluidSimulator
{
    private const int LeafOnlyCpuDiagTokenPool = 1;
    private const int LeafOnlyCpuOffTokenPool = 4;

    private static bool s_warnedLeafOnlyLayer1CpuSkip;

    internal struct LeafOnlyTbWeightOffsets
    {
        public int Norm1W, Norm1B, QkvW, QkvB, ProjW, ProjB, Eg0W, Eg0B, Eg2W, Eg2B, Norm2W, Norm2B, Mlp0W, Mlp0B, Mlp2W, Mlp2B;
    }

    internal static int LeafOnlyEmbedPhaseFloatCount(in LeafOnlyCheckpointHeader h, int globalFeatDim)
    {
        int dm = h.DModel;
        int liftIn = 6 + globalFeatDim;
        int idx = 0;
        idx += liftIn * dm + dm + dm * dm + dm;
        int nGcn = Mathf.Max(0, h.NumGcnLayers);
        for (int L = 0; L < nGcn; L++)
        {
            idx += dm * dm + dm + dm * dm + dm + (2 * dm) * dm + dm + dm * dm + dm;
        }

        idx += dm + dm + dm * dm + dm;
        return idx;
    }

    internal static int LeafOnlyTransformerBlockFloatCount(in LeafOnlyCheckpointHeader h)
    {
        if (h.DecoupledRouteGates != 0 || h.HighwayFfnMlp != 0)
            return -1;
        int d = h.DModel;
        int nh = h.NumHeads;
        int egH = Mathf.Max(1, h.EdgeGateHiddenDim);
        int fcw = Mathf.Max(1, h.FfnConcatWidth);
        int mlpIn = fcw * d;
        int hidden = d * 4;
        int n = 0;
        n += d + d;
        n += d * (3 * d) + 3 * d;
        n += d * d + d;
        n += 4 * egH + egH + egH * nh + nh;
        n += d + d;
        n += mlpIn * hidden + hidden + hidden * d + d;
        return n;
    }

    internal static void LeafOnlyTbOffsetsFromBase(int @base, in LeafOnlyCheckpointHeader h, out LeafOnlyTbWeightOffsets o)
    {
        int d = h.DModel;
        int nh = h.NumHeads;
        int egH = Mathf.Max(1, h.EdgeGateHiddenDim);
        int fcw = Mathf.Max(1, h.FfnConcatWidth);
        int mlpIn = fcw * d;
        int hidden = d * 4;
        int i = @base;
        o.Norm1W = i;
        i += d;
        o.Norm1B = i;
        i += d;
        o.QkvW = i;
        i += d * (3 * d);
        o.QkvB = i;
        i += 3 * d;
        o.ProjW = i;
        i += d * d;
        o.ProjB = i;
        i += d;
        o.Eg0W = i;
        i += 4 * egH;
        o.Eg0B = i;
        i += egH;
        o.Eg2W = i;
        i += egH * nh;
        o.Eg2B = i;
        i += nh;
        o.Norm2W = i;
        i += d;
        o.Norm2B = i;
        i += d;
        o.Mlp0W = i;
        i += mlpIn * hidden;
        o.Mlp0B = i;
        i += hidden;
        o.Mlp2W = i;
        i += hidden * d;
        o.Mlp2B = i;
        i += d;
    }

    private static float LeafOnlyCpuErfApprox(float x)
    {
        float sign = x >= 0f ? 1f : -1f;
        float ax = Mathf.Abs(x);
        float t = 1f / (1f + 0.3275911f * ax);
        float poly = (((((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t - 0.284496736f) * t) + 0.254829592f) * t;
        return sign * (1f - poly * Mathf.Exp(-ax * ax));
    }

    private static float LeafOnlyCpuGeluTorchLike(float x) => 0.5f * x * (1f + LeafOnlyCpuErfApprox(x * 0.70710678118f));

    private static void LeafOnlyCpuLayerNormTokens(float[] x, int nTok, int d, float[] w, int gw, int gb, float eps)
    {
        for (int t = 0; t < nTok; t++)
        {
            int o = t * d;
            float mean = 0f;
            for (int c = 0; c < d; c++)
                mean += x[o + c];
            mean /= d;
            float var = 0f;
            for (int c = 0; c < d; c++)
            {
                float z = x[o + c] - mean;
                var += z * z;
            }

            var /= d;
            float invStd = 1f / Mathf.Sqrt(var + eps);
            for (int c = 0; c < d; c++)
                x[o + c] = (x[o + c] - mean) * invStd * w[gw + c] + w[gb + c];
        }
    }

    private static void LeafOnlyCpuLinear(
        float[] xIn,
        int x0,
        int dIn,
        float[] w,
        int w0,
        int dOut,
        float[] b,
        int b0,
        float[] yOut,
        int y0)
    {
        for (int o = 0; o < dOut; o++)
        {
            double acc = b[b0 + o];
            for (int i = 0; i < dIn; i++)
                acc += xIn[x0 + i] * w[w0 + i * dOut + o];
            yOut[y0 + o] = (float)acc;
        }
    }

    private static void LeafOnlyCpuPoolEdgeFeatsLxL(
        float[] src,
        int mTiles,
        int lFull,
        int pool,
        float[] dst)
    {
        int ls = lFull / pool;
        for (int m = 0; m < mTiles; m++)
        for (int ip = 0; ip < ls; ip++)
        for (int jp = 0; jp < ls; jp++)
        for (int c = 0; c < 4; c++)
        {
            double acc = 0.0;
            for (int di = 0; di < pool; di++)
            for (int dj = 0; dj < pool; dj++)
            {
                int i = ip * pool + di;
                int j = jp * pool + dj;
                int idx = ((m * lFull + i) * lFull + j) * 4 + c;
                acc += src[idx];
            }

            dst[(((m * ls + ip) * ls + jp) * 4) + c] = (float)(acc / (pool * pool));
        }
    }

    /// <summary>One transformer block: residual attention + residual FFN. <paramref name="x"/> layout [numBlocks * L * d].</summary>
    private static void LeafOnlyCpuApplyTransformerBlock(
        float[] x,
        float[] edgeFeats,
        int numBlocks,
        int L,
        int d,
        int nh,
        int egHidden,
        int mlpInDim,
        float[] w,
        in LeafOnlyTbWeightOffsets o,
        float[] tmpNorm,
        float[] tmpQkv,
        float[] tmpRow,
        float[] tmpScores,
        float[] tmpComb,
        float[] tmpMlpH,
        float[] biasPhys,
        float[] lw,
        float[] egHid)
    {
        int dh = d / nh;
        if (dh * nh != d || egHidden < 1)
            return;
        float scale = 1f / Mathf.Sqrt(dh);
        int hidden = d * 4;

        for (int b = 0; b < numBlocks; b++)
        {
            int x0 = b * L * d;
            int e0 = b * L * L * 4;
            for (int i = 0; i < L * d; i++)
                tmpNorm[i] = x[x0 + i];

            LeafOnlyCpuLayerNormTokens(tmpNorm, L, d, w, o.Norm1W, o.Norm1B, 1e-5f);

            for (int t = 0; t < L; t++)
                LeafOnlyCpuLinear(tmpNorm, t * d, d, w, o.QkvW, 3 * d, w, o.QkvB, tmpQkv, t * 3 * d);

            for (int i = 0; i < L * L * 4; i++)
                biasPhys[i] = edgeFeats[e0 + i];
            for (int s = 0; s < L; s++)
            {
                int oDiag = (s * L + s) * 4;
                biasPhys[oDiag] = biasPhys[oDiag + 1] = biasPhys[oDiag + 2] = 0f;
                biasPhys[oDiag + 3] = 1f;
            }

            for (int q = 0; q < L; q++)
            for (int kk = 0; kk < L; kk++)
            {
                int bp = (q * L + kk) * 4;
                LeafOnlyCpuLinear(biasPhys, bp, 4, w, o.Eg0W, egHidden, w, o.Eg0B, egHid, 0);
                for (int z = 0; z < egHidden; z++)
                    egHid[z] = LeafOnlyCpuGeluTorchLike(egHid[z]);
                LeafOnlyCpuLinear(egHid, 0, egHidden, w, o.Eg2W, nh, w, o.Eg2B, lw, (q * L + kk) * nh);
            }

            for (int h = 0; h < nh; h++)
            {
                for (int q = 0; q < L; q++)
                {
                    float maxS = float.NegativeInfinity;
                    for (int kk = 0; kk < L; kk++)
                    {
                        double dot = 0.0;
                        int qOff = q * 3 * d + h * dh;
                        int kOff = kk * 3 * d + d + h * dh;
                        for (int u = 0; u < dh; u++)
                            dot += tmpQkv[qOff + u] * tmpQkv[kOff + u];
                        int b3 = (q * L + kk) * 4 + 3;
                        float s = (float)dot * scale + biasPhys[b3];
                        tmpScores[q * L + kk] = s;
                        if (s > maxS)
                            maxS = s;
                    }

                    double sumE = 0.0;
                    for (int kk = 0; kk < L; kk++)
                        sumE += System.Math.Exp(tmpScores[q * L + kk] - maxS);
                    for (int kk = 0; kk < L; kk++)
                        tmpScores[q * L + kk] = (float)(System.Math.Exp(tmpScores[q * L + kk] - maxS) / sumE);
                }

                for (int q = 0; q < L; q++)
                for (int u = 0; u < dh; u++)
                {
                    double accSoft = 0.0;
                    for (int kk = 0; kk < L; kk++)
                    {
                        float a = tmpScores[q * L + kk];
                        int vOff = kk * 3 * d + 2 * d + h * dh;
                        accSoft += a * tmpQkv[vOff + u];
                    }

                    double accEdge = 0.0;
                    for (int kk = 0; kk < L; kk++)
                    {
                        float lwQkh = lw[(q * L + kk) * nh + h];
                        int vOff = kk * 3 * d + 2 * d + h * dh;
                        accEdge += lwQkh * tmpQkv[vOff + u];
                    }

                    tmpComb[q * d + h * dh + u] = (float)(accSoft + accEdge);
                }
            }

            for (int t = 0; t < L; t++)
            {
                LeafOnlyCpuLinear(tmpComb, t * d, d, w, o.ProjW, d, w, o.ProjB, tmpRow, 0);
                for (int c = 0; c < d; c++)
                    x[x0 + t * d + c] += tmpRow[c];
            }

            for (int i = 0; i < L * d; i++)
                tmpNorm[i] = x[x0 + i];

            LeafOnlyCpuLayerNormTokens(tmpNorm, L, d, w, o.Norm2W, o.Norm2B, 1e-5f);
            for (int t = 0; t < L; t++)
            {
                LeafOnlyCpuLinear(tmpNorm, t * d, mlpInDim, w, o.Mlp0W, hidden, w, o.Mlp0B, tmpMlpH, 0);
                for (int z = 0; z < hidden; z++)
                    tmpMlpH[z] = LeafOnlyCpuGeluTorchLike(tmpMlpH[z]);
                LeafOnlyCpuLinear(tmpMlpH, 0, hidden, w, o.Mlp2W, d, w, o.Mlp2B, tmpRow, 0);
                for (int c = 0; c < d; c++)
                    x[x0 + t * d + c] += tmpRow[c];
            }
        }
    }

    private void DispatchLeafOnlyCpuLayer1ParityAndLog(int nPad)
    {
        const string phase = "unity";
        if (!LeafOnlyWeightsLoadedSuccessfully || leafOnlyWeightsFloatBuffer == null || leafOnlyTokenAfterEnc == null)
            return;

        LeafOnlyCheckpointHeader arch = LeafOnlyCheckpoint;
        if (arch.HighwayFfnMlp != 0 || arch.DecoupledRouteGates != 0 || arch.AttentionLayoutCode != 0 || arch.NumLayers < 1)
        {
            if (!s_warnedLeafOnlyLayer1CpuSkip)
            {
                s_warnedLeafOnlyLayer1CpuSkip = true;
                Debug.Log(
                    "[LeafOnlyParity] tensor=h_diag_after_layer0 skipped (CPU layer1 parity needs highway=0, route_gates=0, attn_layout=0, num_layers>=1).");
            }

            return;
        }

        int Lfull = LeafOnlyLeafSize;
        if (nPad <= 0 || nPad % Lfull != 0)
            return;

        int K = nPad / Lfull;
        int d = arch.DModel;
        int nh = arch.NumHeads;
        int laD = Lfull / LeafOnlyCpuDiagTokenPool;
        int laO = Lfull / LeafOnlyCpuOffTokenPool;
        if (laD * LeafOnlyCpuDiagTokenPool != Lfull || laO * LeafOnlyCpuOffTokenPool != Lfull)
            return;

        int mOff = LeafOnlyHMatrixStatic.NumOffBlocks;
        if (LeafOnlyDiagEdgeFeatsBuffer == null || (mOff > 0 && LeafOnlyOffEdgeFeatsBuffer == null))
            return;

        int embedEnd = LeafOnlyEmbedPhaseFloatCount(in arch, LeafOnlyGlobalFeaturesDim);
        int tbFloats = LeafOnlyTransformerBlockFloatCount(in arch);
        if (tbFloats < 0)
            return;
        int need = embedEnd + 2 * tbFloats;
        if (leafOnlyWeightsFloatBuffer.count < need)
        {
            Debug.LogError($"[LeafOnly] CPU layer1 parity: weight buffer too short (have {leafOnlyWeightsFloatBuffer.count}, need {need}).");
            return;
        }

        var wf = new float[leafOnlyWeightsFloatBuffer.count];
        leafOnlyWeightsFloatBuffer.GetData(wf);

        int nDiagEdge = K * laD * laD * 4;
        var edgeDiag = new float[nDiagEdge];
        LeafOnlyDiagEdgeFeatsBuffer.GetData(edgeDiag, 0, 0, nDiagEdge);

        float[] edgeOffPooled = null;
        int nOffPooled = 0;
        if (mOff > 0)
        {
            int nOffRaw = mOff * Lfull * Lfull * 4;
            var edgeOffRaw = new float[nOffRaw];
            LeafOnlyOffEdgeFeatsBuffer.GetData(edgeOffRaw, 0, 0, nOffRaw);
            nOffPooled = mOff * laO * laO * 4;
            edgeOffPooled = new float[nOffPooled];
            LeafOnlyCpuPoolEdgeFeatsLxL(edgeOffRaw, mOff, Lfull, LeafOnlyCpuOffTokenPool, edgeOffPooled);
        }

        int nTok = nPad * d;
        var hDiag = new float[nTok];
        leafOnlyTokenAfterEnc.GetData(hDiag, 0, 0, nTok);

        LeafOnlyTbOffsetsFromBase(embedEnd, in arch, out LeafOnlyTbWeightOffsets oDiag);
        LeafOnlyTbOffsetsFromBase(embedEnd + tbFloats, in arch, out LeafOnlyTbWeightOffsets oOff);

        int hidden = d * 4;
        int egH = Mathf.Max(1, arch.EdgeGateHiddenDim);
        int mlpInDim = Mathf.Max(1, arch.FfnConcatWidth) * d;
        var tmpNorm = new float[laD * d];
        var tmpQkv = new float[laD * 3 * d];
        var tmpRow = new float[d];
        var tmpScores = new float[laD * laD];
        var tmpComb = new float[laD * d];
        var tmpMlpH = new float[hidden];
        var biasPhys = new float[laD * laD * 4];
        var lw = new float[laD * laD * nh];
        var egHid = new float[egH];

        LeafOnlyCpuApplyTransformerBlock(
            hDiag,
            edgeDiag,
            K,
            laD,
            d,
            nh,
            egH,
            mlpInDim,
            wf,
            in oDiag,
            tmpNorm,
            tmpQkv,
            tmpRow,
            tmpScores,
            tmpComb,
            tmpMlpH,
            biasPhys,
            lw,
            egHid);

        LeafOnlyParitySummarize(phase, "h_diag_after_layer0", hDiag, nTok);
        LeafOnlyParityHead(phase, "h_diag_after_layer0", hDiag, nTok, 16);
        LeafOnlyParityHead(phase, "h_diag_after_layer0", hDiag, nTok, 32);
        if (nTok > 64)
            LeafOnlyParityHeadAt(phase, "h_diag_after_layer0", hDiag, nTok, nTok / 2, 48);

        if (mOff <= 0)
            return;

        int kMax = LeafOnlyHMatrixStatic.MaxNumLeaves;
        var wRow = new float[mOff * kMax];
        var wCol = new float[mOff * kMax];
        ReadOnlySpan<int> r0 = LeafOnlyHMatrixStatic.OffR0;
        ReadOnlySpan<int> c0 = LeafOnlyHMatrixStatic.OffC0;
        ReadOnlySpan<int> sS = LeafOnlyHMatrixStatic.OffS;
        for (int m = 0; m < mOff; m++)
        {
            float invS = 1f / Mathf.Max(1, sS[m]);
            for (int k = r0[m]; k < r0[m] + sS[m] && k < kMax; k++)
                wRow[m * kMax + k] = invS;
            for (int k = c0[m]; k < c0[m] + sS[m] && k < kMax; k++)
                wCol[m * kMax + k] = invS;
        }

        var hLeaves = new float[kMax * Lfull * d];
        for (int k = 0; k < K; k++)
        for (int l = 0; l < Lfull; l++)
        for (int c = 0; c < d; c++)
            hLeaves[(k * Lfull + l) * d + c] = hDiag[(k * Lfull + l) * d + c];

        var stripFull = new float[mOff * Lfull * d];
        for (int m = 0; m < mOff; m++)
        for (int l = 0; l < Lfull; l++)
        for (int c = 0; c < d; c++)
        {
            double rp = 0.0, cp = 0.0;
            for (int k = 0; k < kMax; k++)
            {
                float v = hLeaves[(k * Lfull + l) * d + c];
                rp += wRow[m * kMax + k] * v;
                cp += wCol[m * kMax + k] * v;
            }

            stripFull[(m * Lfull + l) * d + c] = (float)(rp + cp);
        }

        int bm = mOff;
        int nOffTok = bm * laO * d;
        var offIn = new float[nOffTok];
        for (int m = 0; m < mOff; m++)
        for (int ls = 0; ls < laO; ls++)
        {
            for (int c = 0; c < d; c++)
            {
                double acc = 0.0;
                for (int t = 0; t < LeafOnlyCpuOffTokenPool; t++)
                {
                    int l = ls * LeafOnlyCpuOffTokenPool + t;
                    acc += stripFull[(m * Lfull + l) * d + c];
                }

                offIn[(m * laO + ls) * d + c] = (float)(acc / LeafOnlyCpuOffTokenPool);
            }
        }

        var tmpNormO = new float[laO * d];
        var tmpQkvO = new float[laO * 3 * d];
        var tmpScoresO = new float[laO * laO];
        var tmpCombO = new float[laO * d];
        var biasPhysO = new float[laO * laO * 4];
        var lwO = new float[laO * laO * nh];

        for (int mb = 0; mb < bm; mb++)
        {
            int x0 = mb * laO * d;
            int e0 = mb * laO * laO * 4;
            var xBlk = new float[laO * d];
            var edgeBlk = new float[laO * laO * 4];
            for (int i = 0; i < laO * d; i++)
                xBlk[i] = offIn[x0 + i];
            for (int i = 0; i < laO * laO * 4; i++)
                edgeBlk[i] = edgeOffPooled[e0 + i];

            LeafOnlyCpuApplyTransformerBlock(
                xBlk,
                edgeBlk,
                1,
                laO,
                d,
                nh,
                egH,
                mlpInDim,
                wf,
                in oOff,
                tmpNormO,
                tmpQkvO,
                tmpRow,
                tmpScoresO,
                tmpCombO,
                tmpMlpH,
                biasPhysO,
                lwO,
                egHid);

            for (int i = 0; i < laO * d; i++)
                offIn[x0 + i] = xBlk[i];
        }

        LeafOnlyParitySummarize(phase, "off_stream_after_layer0", offIn, nOffTok);
        LeafOnlyParityHead(phase, "off_stream_after_layer0", offIn, nOffTok, 16);
        LeafOnlyParityHead(phase, "off_stream_after_layer0", offIn, nOffTok, 32);
        if (nOffTok > 64)
            LeafOnlyParityHeadAt(phase, "off_stream_after_layer0", offIn, nOffTok, nOffTok / 2, 48);

        var invC = CultureInfo.InvariantCulture;
        Debug.Log(
            $"[LeafOnlyParity] phase={phase} tensor=layer1_cpu_meta K={K.ToString(invC)} L_diag={laD.ToString(invC)} " +
            $"L_off={laO.ToString(invC)} M_off={mOff.ToString(invC)} d_model={d.ToString(invC)} heads={nh.ToString(invC)} eager_softmax=1");
    }
}
