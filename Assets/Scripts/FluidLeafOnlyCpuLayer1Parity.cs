using UnityEngine;

/// <summary>
/// Shared weight layout for the first transformer layer: float counts and per-block offsets
/// (<see cref="LeafOnlyTbWeightOffsets"/>) used by <c>LeafOnlyLayer1.compute</c> parity and embed setup.
/// Token-pool constants name the diag (1) and off (4) pooling factors expected by the GPU path.
/// </summary>
public partial class FluidSimulator
{
    internal const int LeafOnlyCpuDiagTokenPool = 1;
    internal const int LeafOnlyCpuOffTokenPool = 4;

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
}
