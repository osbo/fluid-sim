using UnityEngine;

/// <summary>
/// GPU path for LeafOnly <c>PhysicsAwareEmbedding</c> + <c>enc_input_proj</c> (checkpoint tensor order in <c>leafonly/checkpoint.py</c>).
/// Parity tensor name: <c>token_after_enc</c> (same stats as <c>InspectParity.py</c>).
/// </summary>
public partial class FluidSimulator : MonoBehaviour
{
    public const int LeafOnlyGlobalFeaturesDim = 12;
    public const int LeafOnlyEmbedDModelMax = 128;

    [Tooltip("LeafOnly_Embed* kernels. Editor loads Assets/Scripts/LeafOnlyEmbed.compute if unset.")]
    ComputeShader leafOnlyEmbedShader;

    private int leafOnlyEmbedLiftKernel;
    private int leafOnlyEmbedClearAggrKernel;
    private int leafOnlyEmbedNeighLinearKernel;
    private int leafOnlyEmbedScatterKernel;
    private int leafOnlyEmbedCombineKernel;
    private int leafOnlyEmbedLayerNormKernel;
    private int leafOnlyEmbedEncProjKernel;

    private ComputeBuffer leafOnlyEmbedPing;
    private ComputeBuffer leafOnlyEmbedPong;
    private ComputeBuffer leafOnlyEmbedNeighLin;
    private ComputeBuffer leafOnlyEmbedAggrFloat;
    private ComputeBuffer leafOnlyTokenAfterEnc;

    private int leafOnlyEmbedCapNodes;
    private int leafOnlyEmbedCapDm;
    private static bool s_warnedLeafOnlyEmbedMissing;
    private static bool s_warnedLeafOnlyEmbedDm;

    private struct LeafOnlyLiftWeightOffsets
    {
        public int Lift0W, Lift0B, Lift2W, Lift2B;
    }

    private struct LeafOnlyGcnLayerWeightOffsets
    {
        public int SelfW, SelfB, NeighW, NeighB, Gate0W, Gate0B, Gate2W, Gate2B;
    }

    /// <summary>Float indices into <see cref="LeafOnlyWeightsFloatBuffer"/> matching sequential checkpoint read through enc_input_proj.</summary>
    private static void LeafOnlyComputeEmbedWeightOffsets(
        in LeafOnlyCheckpointHeader h,
        int globalFeatDim,
        out LeafOnlyLiftWeightOffsets lift,
        out LeafOnlyGcnLayerWeightOffsets[] gcn,
        out int normW,
        out int normB,
        out int encW,
        out int encB,
        out int totalFloatsEmbedPhase)
    {
        int dm = h.DModel;
        int liftIn = 6 + globalFeatDim;
        int idx = 0;
        lift.Lift0W = idx;
        idx += liftIn * dm;
        lift.Lift0B = idx;
        idx += dm;
        lift.Lift2W = idx;
        idx += dm * dm;
        lift.Lift2B = idx;
        idx += dm;

        int nGcn = Mathf.Max(0, h.NumGcnLayers);
        gcn = new LeafOnlyGcnLayerWeightOffsets[nGcn];
        for (int L = 0; L < nGcn; L++)
        {
            ref var g = ref gcn[L];
            g.SelfW = idx;
            idx += dm * dm;
            g.SelfB = idx;
            idx += dm;
            g.NeighW = idx;
            idx += dm * dm;
            g.NeighB = idx;
            idx += dm;
            g.Gate0W = idx;
            idx += (2 * dm) * dm;
            g.Gate0B = idx;
            idx += dm;
            g.Gate2W = idx;
            idx += dm * dm;
            g.Gate2B = idx;
            idx += dm;
        }

        normW = idx;
        idx += dm;
        normB = idx;
        idx += dm;
        encW = idx;
        idx += dm * dm;
        encB = idx;
        idx += dm;
        totalFloatsEmbedPhase = idx;
    }

    private void InitLeafOnlyEmbedKernels()
    {
        if (leafOnlyEmbedShader == null)
            return;
        leafOnlyEmbedLiftKernel = leafOnlyEmbedShader.FindKernel("LeafOnly_EmbedLift");
        leafOnlyEmbedClearAggrKernel = leafOnlyEmbedShader.FindKernel("LeafOnly_EmbedClearAggr");
        leafOnlyEmbedNeighLinearKernel = leafOnlyEmbedShader.FindKernel("LeafOnly_EmbedNeighLinear");
        leafOnlyEmbedScatterKernel = leafOnlyEmbedShader.FindKernel("LeafOnly_EmbedGcnEdgeScatterSerial");
        leafOnlyEmbedCombineKernel = leafOnlyEmbedShader.FindKernel("LeafOnly_EmbedGcnCombine");
        leafOnlyEmbedLayerNormKernel = leafOnlyEmbedShader.FindKernel("LeafOnly_EmbedLayerNorm");
        leafOnlyEmbedEncProjKernel = leafOnlyEmbedShader.FindKernel("LeafOnly_EmbedEncProj");
    }

    private void EnsureLeafOnlyEmbedWorkBuffers(int nPad, int dModel)
    {
        int nFloat = nPad * dModel;
        if (leafOnlyEmbedPing != null && leafOnlyEmbedCapNodes >= nPad && leafOnlyEmbedCapDm >= dModel)
            return;

        leafOnlyEmbedPing?.Release();
        leafOnlyEmbedPong?.Release();
        leafOnlyEmbedNeighLin?.Release();
        leafOnlyEmbedAggrFloat?.Release();
        leafOnlyTokenAfterEnc?.Release();

        leafOnlyEmbedPing = new ComputeBuffer(nFloat, sizeof(float));
        leafOnlyEmbedPong = new ComputeBuffer(nFloat, sizeof(float));
        leafOnlyEmbedNeighLin = new ComputeBuffer(nFloat, sizeof(float));
        leafOnlyEmbedAggrFloat = new ComputeBuffer(nFloat, sizeof(float));
        leafOnlyTokenAfterEnc = new ComputeBuffer(nFloat, sizeof(float));
        leafOnlyEmbedCapNodes = nPad;
        leafOnlyEmbedCapDm = dModel;
    }

    private void BindLeafOnlyEmbedWeightsAndFeatures(int kernel)
    {
        leafOnlyEmbedShader.SetBuffer(kernel, "leafOnlyWeights", leafOnlyWeightsFloatBuffer);
        leafOnlyEmbedShader.SetBuffer(kernel, "leafXPacked", leafOnlyXPacked);
        leafOnlyEmbedShader.SetBuffer(kernel, "leafGlobalFeatures", leafOnlyGlobalFeatures);
    }

    private void SetLeafOnlyEmbedCommonInts(int nPad, int dModel, int liftIn)
    {
        leafOnlyEmbedShader.SetInt("leafNPadded", nPad);
        leafOnlyEmbedShader.SetInt("leafDModel", dModel);
        leafOnlyEmbedShader.SetInt("leafLiftIn", liftIn);
    }

    private void SetLeafOnlyLiftOffsets(in LeafOnlyLiftWeightOffsets lift)
    {
        leafOnlyEmbedShader.SetInt("offLift0W", lift.Lift0W);
        leafOnlyEmbedShader.SetInt("offLift0B", lift.Lift0B);
        leafOnlyEmbedShader.SetInt("offLift2W", lift.Lift2W);
        leafOnlyEmbedShader.SetInt("offLift2B", lift.Lift2B);
    }

    private void SetLeafOnlyGcnOffsets(in LeafOnlyGcnLayerWeightOffsets g)
    {
        leafOnlyEmbedShader.SetInt("offGcnSelfW", g.SelfW);
        leafOnlyEmbedShader.SetInt("offGcnSelfB", g.SelfB);
        leafOnlyEmbedShader.SetInt("offGcnNeighW", g.NeighW);
        leafOnlyEmbedShader.SetInt("offGcnNeighB", g.NeighB);
        leafOnlyEmbedShader.SetInt("offGcnGate0W", g.Gate0W);
        leafOnlyEmbedShader.SetInt("offGcnGate0B", g.Gate0B);
        leafOnlyEmbedShader.SetInt("offGcnGate2W", g.Gate2W);
        leafOnlyEmbedShader.SetInt("offGcnGate2B", g.Gate2B);
    }

    private void DispatchLeafOnlyEmbed1D(int kernel, int totalThreads)
    {
        int groups = Mathf.Max(1, Mathf.CeilToInt(totalThreads / 256f));
        leafOnlyEmbedShader.Dispatch(kernel, groups, 1, 1);
    }

    /// <summary>Runs embed + enc on GPU. When <paramref name="logParityTensors"/>, logs <c>[LeafOnlyParity] tensor=token_after_enc</c> lines.</summary>
    private void DispatchLeafOnlyEmbedForwardAndLogParity(int nPad, int compactEdgeCount, bool logParityTensors = true)
    {
        const string phase = "unity";
        if (leafOnlyEmbedShader == null)
        {
            if (!s_warnedLeafOnlyEmbedMissing)
            {
                s_warnedLeafOnlyEmbedMissing = true;
                Debug.Log(
                    "[LeafOnlyParity] phase=unity tensor=token_after_enc skipped (assign leafOnlyEmbedShader = LeafOnlyEmbed.compute to compare with InspectParity.py).");
            }
            return;
        }

        if (!LeafOnlyWeightsLoadedSuccessfully || leafOnlyWeightsFloatBuffer == null
            || leafOnlyXPacked == null || leafOnlyGlobalFeatures == null)
            return;

        LeafOnlyCheckpointHeader arch = LeafOnlyCheckpoint;
        int dm = arch.DModel;
        if (dm > LeafOnlyEmbedDModelMax)
        {
            if (!s_warnedLeafOnlyEmbedDm)
            {
                s_warnedLeafOnlyEmbedDm = true;
                Debug.LogWarning($"[LeafOnly] Embed GPU: d_model={dm} > {LeafOnlyEmbedDModelMax}; skipping token_after_enc.");
            }
            return;
        }

        LeafOnlyComputeEmbedWeightOffsets(
            in arch,
            LeafOnlyGlobalFeaturesDim,
            out LeafOnlyLiftWeightOffsets liftOff,
            out LeafOnlyGcnLayerWeightOffsets[] gcnOff,
            out int normW,
            out int normB,
            out int encW,
            out int encB,
            out int needFloats);

        if (leafOnlyWeightsFloatBuffer.count < needFloats)
        {
            Debug.LogError(
                $"[LeafOnly] Weight buffer too short for embed phase: have {leafOnlyWeightsFloatBuffer.count}, need {needFloats}.");
            return;
        }

        EnsureLeafOnlyEmbedWorkBuffers(nPad, dm);
        int liftIn = 6 + LeafOnlyGlobalFeaturesDim;

        // --- Lift -> ping
        SetLeafOnlyEmbedCommonInts(nPad, dm, liftIn);
        SetLeafOnlyLiftOffsets(in liftOff);
        leafOnlyEmbedShader.SetInt("leafCompactEdgeCount", compactEdgeCount);

        BindLeafOnlyEmbedWeightsAndFeatures(leafOnlyEmbedLiftKernel);
        leafOnlyEmbedShader.SetBuffer(leafOnlyEmbedLiftKernel, "leafEmbedHOut", leafOnlyEmbedPing);
        DispatchLeafOnlyEmbed1D(leafOnlyEmbedLiftKernel, nPad);

        ComputeBuffer hIn = leafOnlyEmbedPing;
        ComputeBuffer hOut = leafOnlyEmbedPong;

        for (int L = 0; L < gcnOff.Length; L++)
        {
            SetLeafOnlyGcnOffsets(in gcnOff[L]);

            BindLeafOnlyEmbedWeightsAndFeatures(leafOnlyEmbedNeighLinearKernel);
            leafOnlyEmbedShader.SetBuffer(leafOnlyEmbedNeighLinearKernel, "leafEmbedHIn", hIn);
            leafOnlyEmbedShader.SetBuffer(leafOnlyEmbedNeighLinearKernel, "leafEmbedNeighLin", leafOnlyEmbedNeighLin);
            DispatchLeafOnlyEmbed1D(leafOnlyEmbedNeighLinearKernel, nPad);

            BindLeafOnlyEmbedWeightsAndFeatures(leafOnlyEmbedClearAggrKernel);
            leafOnlyEmbedShader.SetBuffer(leafOnlyEmbedClearAggrKernel, "leafEmbedAggrFloat", leafOnlyEmbedAggrFloat);
            DispatchLeafOnlyEmbed1D(leafOnlyEmbedClearAggrKernel, nPad * dm);

            BindLeafOnlyEmbedWeightsAndFeatures(leafOnlyEmbedScatterKernel);
            leafOnlyEmbedShader.SetBuffer(leafOnlyEmbedScatterKernel, "leafEmbedNeighLin", leafOnlyEmbedNeighLin);
            leafOnlyEmbedShader.SetBuffer(leafOnlyEmbedScatterKernel, "leafEmbedAggrFloat", leafOnlyEmbedAggrFloat);
            leafOnlyEmbedShader.SetBuffer(leafOnlyEmbedScatterKernel, "leafGcnEdgeRows", leafOnlyGcnEdgeRows);
            leafOnlyEmbedShader.SetBuffer(leafOnlyEmbedScatterKernel, "leafGcnEdgeCols", leafOnlyGcnEdgeCols);
            leafOnlyEmbedShader.SetBuffer(leafOnlyEmbedScatterKernel, "leafGcnEdgeVals", leafOnlyGcnEdgeVals);
            leafOnlyEmbedShader.Dispatch(leafOnlyEmbedScatterKernel, 1, 1, 1);

            BindLeafOnlyEmbedWeightsAndFeatures(leafOnlyEmbedCombineKernel);
            leafOnlyEmbedShader.SetBuffer(leafOnlyEmbedCombineKernel, "leafEmbedHIn", hIn);
            leafOnlyEmbedShader.SetBuffer(leafOnlyEmbedCombineKernel, "leafEmbedHOut", hOut);
            leafOnlyEmbedShader.SetBuffer(leafOnlyEmbedCombineKernel, "leafEmbedAggrFloat", leafOnlyEmbedAggrFloat);
            DispatchLeafOnlyEmbed1D(leafOnlyEmbedCombineKernel, nPad);

            (hIn, hOut) = (hOut, hIn);
        }

        // After each GCN layer we swap; latest activations always live in hIn.
        ComputeBuffer hAfterGcn = hIn;

        leafOnlyEmbedShader.SetInt("offNormW", normW);
        leafOnlyEmbedShader.SetInt("offNormB", normB);
        BindLeafOnlyEmbedWeightsAndFeatures(leafOnlyEmbedLayerNormKernel);
        leafOnlyEmbedShader.SetBuffer(leafOnlyEmbedLayerNormKernel, "leafEmbedHIn", hAfterGcn);
        leafOnlyEmbedShader.SetBuffer(leafOnlyEmbedLayerNormKernel, "leafEmbedHOut", hOut);
        DispatchLeafOnlyEmbed1D(leafOnlyEmbedLayerNormKernel, nPad);

        leafOnlyEmbedShader.SetInt("offEncW", encW);
        leafOnlyEmbedShader.SetInt("offEncB", encB);
        BindLeafOnlyEmbedWeightsAndFeatures(leafOnlyEmbedEncProjKernel);
        leafOnlyEmbedShader.SetBuffer(leafOnlyEmbedEncProjKernel, "leafEmbedHIn", hOut);
        leafOnlyEmbedShader.SetBuffer(leafOnlyEmbedEncProjKernel, "leafTokenAfterEnc", leafOnlyTokenAfterEnc);
        DispatchLeafOnlyEmbed1D(leafOnlyEmbedEncProjKernel, nPad);

        if (!logParityTensors)
            return;

        int nTok = nPad * dm;
        var tok = new float[nTok];
        leafOnlyTokenAfterEnc.GetData(tok, 0, 0, nTok);
        LeafOnlyParitySummarize(phase, "token_after_enc", tok, nTok);
        LeafOnlyParityHead(phase, "token_after_enc", tok, nTok, 16);
        LeafOnlyParityHead(phase, "token_after_enc", tok, nTok, 32);
    }

    private void ReleaseLeafOnlyEmbedBuffers()
    {
        leafOnlyEmbedPing?.Release();
        leafOnlyEmbedPing = null;
        leafOnlyEmbedPong?.Release();
        leafOnlyEmbedPong = null;
        leafOnlyEmbedNeighLin?.Release();
        leafOnlyEmbedNeighLin = null;
        leafOnlyEmbedAggrFloat?.Release();
        leafOnlyEmbedAggrFloat = null;
        leafOnlyTokenAfterEnc?.Release();
        leafOnlyTokenAfterEnc = null;
        leafOnlyEmbedCapNodes = 0;
        leafOnlyEmbedCapDm = 0;
    }
}
