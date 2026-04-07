using UnityEngine;

/// <summary>
/// Checkpoint float indices for the two-layer <c>off_diag_head_U</c>, <c>off_diag_head_V</c>, and <c>leaf_head</c>
/// MLPs (after transformer blocks). Used by <see cref="FluidLeafOnlyLayer1Gpu"/> to set compute shader uniforms.
/// </summary>
public partial class FluidSimulator
{
    internal static int LeafOnlyTwoLayerHeadFloatCount(int dModel, int laOut) =>
        dModel * dModel + dModel + dModel * laOut + laOut;

    internal static bool LeafOnlyTryPostTransformerHeadWeightsBase(
        in LeafOnlyCheckpointHeader arch,
        int globalFeatDim,
        out int headBase)
    {
        headBase = 0;
        int embedEnd = LeafOnlyEmbedPhaseFloatCount(in arch, globalFeatDim);
        int tb = LeafOnlyTransformerBlockFloatCount(in arch);
        if (tb < 0 || arch.NumLayers < 1)
            return false;
        headBase = embedEnd + arch.NumLayers * tb * 2;
        return true;
    }

    private static void LeafOnlyHeadMlpOffsets(int wBase, int d, int laOut, out int w0, out int b0, out int w1, out int b1)
    {
        w0 = wBase;
        b0 = w0 + d * d;
        w1 = b0 + d;
        b1 = w1 + d * laOut;
    }

    internal static bool LeafOnlyTryGetPrecondHeadWeightOffsets(
        in LeafOnlyCheckpointHeader arch,
        int globalFeatDim,
        int weightFloatCount,
        out int leafW0,
        out int leafB0,
        out int leafW1,
        out int leafB1,
        out int offUW0,
        out int offUB0,
        out int offUW1,
        out int offUB1,
        out int offVW0,
        out int offVB0,
        out int offVW1,
        out int offVB1)
    {
        leafW0 = leafB0 = leafW1 = leafB1 = 0;
        offUW0 = offUB0 = offUW1 = offUB1 = 0;
        offVW0 = offVB0 = offVW1 = offVB1 = 0;
        if (arch.MlpHeads != 1)
            return false;
        int d = arch.DModel;
        int laD = arch.LeafApplyDiag;
        int laO = arch.LeafApplyOff;
        if (!LeafOnlyTryPostTransformerHeadWeightsBase(in arch, globalFeatDim, out int uBase))
            return false;
        int vBase = uBase + LeafOnlyTwoLayerHeadFloatCount(d, laO);
        int leafBase = vBase + LeafOnlyTwoLayerHeadFloatCount(d, laO);
        int end = leafBase + LeafOnlyTwoLayerHeadFloatCount(d, laD);
        if (weightFloatCount < end)
            return false;
        LeafOnlyHeadMlpOffsets(uBase, d, laO, out offUW0, out offUB0, out offUW1, out offUB1);
        LeafOnlyHeadMlpOffsets(vBase, d, laO, out offVW0, out offVB0, out offVW1, out offVB1);
        LeafOnlyHeadMlpOffsets(leafBase, d, laD, out leafW0, out leafB0, out leafW1, out leafB1);
        return true;
    }

    /// <summary>Float count for <c>node_u</c>, <c>node_v</c>, and <c>jacobi_gate</c> after MLP leaf heads (matches Python save order).</summary>
    internal static int LeafOnlyNodeJacobiWeightFloatCount(int dModel, int laOff) =>
        (dModel * laOff + laOff) * 2 + (dModel + 1);

    /// <summary>Checkpoint indices for <c>node_u</c>, <c>node_v</c>, <c>jacobi_gate</c> (after <c>leaf_head</c> MLP).</summary>
    internal static bool LeafOnlyTryGetNodeJacobiWeightOffsets(
        in LeafOnlyCheckpointHeader arch,
        int globalFeatDim,
        int weightFloatCount,
        out int nodeUW,
        out int nodeUB,
        out int nodeVW,
        out int nodeVB,
        out int jacobiW,
        out int jacobiB)
    {
        nodeUW = nodeUB = nodeVW = nodeVB = jacobiW = jacobiB = 0;
        if (arch.MlpHeads != 1)
            return false;
        int d = arch.DModel;
        int laD = arch.LeafApplyDiag;
        int laO = arch.LeafApplyOff;
        // Same layout as TryGetPrecondHeadWeightsBase: head0 = off_diag_head_U, then V, then leaf_head, then node_u.
        if (!LeafOnlyTryPostTransformerHeadWeightsBase(in arch, globalFeatDim, out int offUBase))
            return false;
        int offVBase = offUBase + LeafOnlyTwoLayerHeadFloatCount(d, laO);
        int leafBase = offVBase + LeafOnlyTwoLayerHeadFloatCount(d, laO);
        int tailBase = leafBase + LeafOnlyTwoLayerHeadFloatCount(d, laD);
        int need = tailBase + LeafOnlyNodeJacobiWeightFloatCount(d, laO);
        if (weightFloatCount < need)
            return false;
        nodeUW = tailBase;
        nodeUB = nodeUW + d * laO;
        nodeVW = nodeUB + laO;
        nodeVB = nodeVW + d * laO;
        jacobiW = nodeVB + laO;
        jacobiB = jacobiW + d;
        return true;
    }
}
