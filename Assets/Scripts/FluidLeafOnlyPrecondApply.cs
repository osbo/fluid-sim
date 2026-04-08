using UnityEngine;

/// <summary>
/// GPU apply <c>z += M r</c> for packed LeafOnly preconditioner coefficients (same layout as PyTorch
/// <c>LeafOnlyNet.forward</c> / <c>unpack_precond</c>). Packed data is uploaded from layer-1 inference
/// (<see cref="LeafOnlyUploadPackedPrecondFromHost"/>) into an assigned <see cref="leafOnlyPrecondPackedBuffer"/>
/// or an internal buffer when the assigned buffer is missing or the wrong size. In the Editor, the apply
/// shader is auto-loaded from <c>Assets/Scripts/LeafOnlyPrecondApply.compute</c> if unset.
/// </summary>
public partial class FluidSimulator : MonoBehaviour
{
    [Tooltip("LeafOnlyPrecondApply.compute — packed precond apply z += M r. Editor loads from Assets/Scripts if unset.")]
    ComputeShader leafOnlyPrecondApplyShader;

    [Tooltip("Optional packed preconditioner buffer (diag | off | U | V | optional jacobi). If null or wrong length, an internal buffer is used for GPU apply.")]
    public ComputeBuffer leafOnlyPrecondPackedBuffer;

    [Tooltip("Optional Jacobi inverse diagonal (length ≥ n_pad). If unset, Neural precond fills an internal buffer from matrix A diagonals each pressure solve.")]
    public ComputeBuffer leafOnlyJacobiInvDiagBuffer;

    private ComputeBuffer leafOnlyPrecondPackedInternal;
    private ComputeBuffer leafOnlyJacobiInvDiagInternal;
    private float[] leafOnlyMatrixADiagReadbackCache;

    private int kPrecondClearZ;
    private int kPrecondClearScratch;
    private int kPrecondPoolDiag;
    private int kPrecondDiag;
    private int kPrecondProjUv;
    private int kPrecondRowStrip;
    private int kPrecondColStrip;
    private int kPrecondBmmCol;
    private int kPrecondBmmRowT;
    private int kPrecondProlongRow;
    private int kPrecondProlongCol;
    private int kPrecondNodeAcc;
    private int kPrecondJacobi;

    private ComputeBuffer leafOnlyHmR0Buffer;
    private ComputeBuffer leafOnlyHmC0Buffer;
    private ComputeBuffer leafOnlyHmSBuffer;
    private ComputeBuffer leafOnlyPrecondApplyScratchBuffer;
    private int leafOnlyPrecondApplyScratchFloats;

    /// <summary>
    /// Allocates or reallocates <see cref="leafOnlyPrecondPackedBuffer"/> for the loaded checkpoint and current
    /// <see cref="LeafOnlyLastNPadded"/> (or inferred pad from <see cref="FluidSimulator.NumNodes"/> if inputs have not run yet).
    /// Jacobi tail size is inferred from weight layout (<see cref="LeafOnlyTryGetNodeJacobiWeightOffsets"/>).
    /// </summary>
    public bool TryAllocLeafOnlyPrecondPackedBuffer()
    {
        if (!LeafOnlyWeightsLoadedSuccessfully)
            return false;
        var arch = LeafOnlyCheckpoint;
        int l = arch.LeafSize;
        if (l <= 0)
            return false;
        int nPad = LeafOnlyLastNPadded;
        if (nPad <= 0)
        {
            int aligned = ((numNodes + l - 1) / l) * l;
            nPad = Mathf.Min(aligned, LeafOnlyMaxMixedSize);
        }
        if (nPad <= 0 || nPad % l != 0)
            return false;
        int kLeaves = nPad / l;
        bool jac = LeafOnlyTryGetNodeJacobiWeightOffsets(
            in arch,
            LeafOnlyGlobalFeaturesDim,
            LeafOnlyWeightsFloatCount,
            out _,
            out _,
            out _,
            out _,
            out _,
            out _);
        int n = LeafOnlyPrecondPackedFloatCount(in arch, kLeaves, jac);
        leafOnlyPrecondPackedBuffer?.Release();
        leafOnlyPrecondPackedBuffer = new ComputeBuffer(n, sizeof(float));
        return true;
    }

    /// <summary>Tight packed float count for <c>K</c> active leaves (n_pad / leaf_size).</summary>
    internal static int LeafOnlyPrecondPackedFloatCount(in LeafOnlyCheckpointHeader arch, int kLeaves, bool includeJacobi)
    {
        int laD = arch.LeafApplyDiag;
        int laO = arch.LeafApplyOff;
        int l = arch.LeafSize;
        int mh = LeafOnlyHMatrixStatic.NumOffBlocks;
        int diag = kLeaves * laD * laD;
        int off = mh * laO * laO;
        int node = kLeaves * l * laO;
        int n = diag + off + 2 * node;
        if (includeJacobi)
            n += kLeaves * l;
        return n;
    }

    internal static int LeafOnlyPrecondApplyScratchFloatCount(int kLeaves, int laD, int laO, int mOff)
    {
        int xPool = kLeaves * laD;
        int ku = kLeaves * laO;
        int strip = mOff * laO;
        return xPool + 2 * ku + 4 * strip + 2 * ku;
    }

    private void InitLeafOnlyPrecondApplyKernels()
    {
        if (leafOnlyPrecondApplyShader == null)
            return;
        kPrecondClearZ = leafOnlyPrecondApplyShader.FindKernel("LeafOnly_PrecondClearZ");
        kPrecondClearScratch = leafOnlyPrecondApplyShader.FindKernel("LeafOnly_PrecondClearScratch");
        kPrecondPoolDiag = leafOnlyPrecondApplyShader.FindKernel("LeafOnly_PrecondPoolDiag");
        kPrecondDiag = leafOnlyPrecondApplyShader.FindKernel("LeafOnly_PrecondDiag");
        kPrecondProjUv = leafOnlyPrecondApplyShader.FindKernel("LeafOnly_PrecondProjUV");
        kPrecondRowStrip = leafOnlyPrecondApplyShader.FindKernel("LeafOnly_PrecondRowStrip");
        kPrecondColStrip = leafOnlyPrecondApplyShader.FindKernel("LeafOnly_PrecondColStrip");
        kPrecondBmmCol = leafOnlyPrecondApplyShader.FindKernel("LeafOnly_PrecondBmmCol");
        kPrecondBmmRowT = leafOnlyPrecondApplyShader.FindKernel("LeafOnly_PrecondBmmRowT");
        kPrecondProlongRow = leafOnlyPrecondApplyShader.FindKernel("LeafOnly_PrecondProlongRow");
        kPrecondProlongCol = leafOnlyPrecondApplyShader.FindKernel("LeafOnly_PrecondProlongCol");
        kPrecondNodeAcc = leafOnlyPrecondApplyShader.FindKernel("LeafOnly_PrecondNodeAcc");
        kPrecondJacobi = leafOnlyPrecondApplyShader.FindKernel("LeafOnly_PrecondJacobi");
    }

    private void EnsureLeafOnlyHmStaticBuffers()
    {
        int m = LeafOnlyHMatrixStatic.NumOffBlocks;
        if (m <= 0)
            return;
        if (leafOnlyHmR0Buffer != null && leafOnlyHmR0Buffer.count >= m)
            return;
        leafOnlyHmR0Buffer?.Release();
        leafOnlyHmC0Buffer?.Release();
        leafOnlyHmSBuffer?.Release();
        var r0 = new int[m];
        var c0 = new int[m];
        var s = new int[m];
        LeafOnlyHMatrixStatic.OffR0.CopyTo(r0);
        LeafOnlyHMatrixStatic.OffC0.CopyTo(c0);
        LeafOnlyHMatrixStatic.OffS.CopyTo(s);
        leafOnlyHmR0Buffer = new ComputeBuffer(m, sizeof(int));
        leafOnlyHmC0Buffer = new ComputeBuffer(m, sizeof(int));
        leafOnlyHmSBuffer = new ComputeBuffer(m, sizeof(int));
        leafOnlyHmR0Buffer.SetData(r0);
        leafOnlyHmC0Buffer.SetData(c0);
        leafOnlyHmSBuffer.SetData(s);
    }

    private void EnsureLeafOnlyPrecondApplyScratch(in LeafOnlyCheckpointHeader arch, int kLeaves, int mOff)
    {
        int laD = arch.LeafApplyDiag;
        int laO = arch.LeafApplyOff;
        int need = LeafOnlyPrecondApplyScratchFloatCount(kLeaves, laD, laO, mOff);
        if (leafOnlyPrecondApplyScratchBuffer != null && leafOnlyPrecondApplyScratchBuffer.count >= need
            && leafOnlyPrecondApplyScratchFloats == need)
            return;
        leafOnlyPrecondApplyScratchBuffer?.Release();
        leafOnlyPrecondApplyScratchFloats = need;
        leafOnlyPrecondApplyScratchBuffer = new ComputeBuffer(Mathf.Max(need, 1), sizeof(float));
    }

    private static int LeafOnlyPrecondScratchOffsets(int kLeaves, int laD, int laO, int mOff, out int xPool, out int xu, out int xv, out int rowStrip, out int colStrip, out int yr, out int yc, out int yrPad, out int ycPad)
    {
        int o = 0;
        xPool = o;
        o += kLeaves * laD;
        xu = o;
        o += kLeaves * laO;
        xv = o;
        o += kLeaves * laO;
        rowStrip = o;
        o += mOff * laO;
        colStrip = o;
        o += mOff * laO;
        yr = o;
        o += mOff * laO;
        yc = o;
        o += mOff * laO;
        yrPad = o;
        o += kLeaves * laO;
        ycPad = o;
        o += kLeaves * laO;
        return o;
    }

    private void LeafOnlyPrecondApplyClearScratch(int floatCount)
    {
        if (floatCount <= 0 || leafOnlyPrecondApplyScratchBuffer == null || leafOnlyPrecondApplyShader == null)
            return;
        leafOnlyPrecondApplyShader.SetBuffer(kPrecondClearScratch, "leafPrecondScratch", leafOnlyPrecondApplyScratchBuffer);
        leafOnlyPrecondApplyShader.SetInt("leafScratchClearCount", floatCount);
        int groups = Mathf.Max(1, Mathf.CeilToInt(floatCount / 256f));
        leafOnlyPrecondApplyShader.Dispatch(kPrecondClearScratch, groups, 1, 1);
    }

    /// <summary>
    /// Uploads packed precond floats from CPU (parity / mirror of GPU tensors). Uses <see cref="leafOnlyPrecondPackedBuffer"/>
    /// when its element count matches <paramref name="total"/>; otherwise (re)allocates <see cref="leafOnlyPrecondPackedInternal"/>.
    /// </summary>
    private void LeafOnlyUploadPackedPrecondFromHost(float[] packed, int total)
    {
        if (packed == null || total <= 0 || packed.Length < total)
            return;
        if (leafOnlyPrecondPackedBuffer != null && leafOnlyPrecondPackedBuffer.count == total)
        {
            leafOnlyPrecondPackedBuffer.SetData(packed);
            return;
        }

        if (leafOnlyPrecondPackedInternal == null || leafOnlyPrecondPackedInternal.count != total)
        {
            leafOnlyPrecondPackedInternal?.Release();
            leafOnlyPrecondPackedInternal = new ComputeBuffer(total, sizeof(float));
        }

        leafOnlyPrecondPackedInternal.SetData(packed);
    }

    /// <summary>
    /// When the checkpoint includes a Jacobi tail and no <see cref="leafOnlyJacobiInvDiagBuffer"/> is assigned,
    /// builds <see cref="leafOnlyJacobiInvDiagInternal"/> from Laplacian diagonal (slot 0 of <see cref="matrixABuffer"/>).
    /// Call once per pressure solve after the matrix is built (and after <see cref="LeafOnlyLastNPadded"/> is known).
    /// </summary>
    private void LeafOnlyEnsureJacobiInvDiagFromMatrixA(int nPad)
    {
        if (!LeafOnlyWeightsLoadedSuccessfully || nPad <= 0 || matrixABuffer == null || numNodes <= 0)
            return;
        if (leafOnlyJacobiInvDiagBuffer != null)
            return;

        var arch = LeafOnlyCheckpoint;
        if (!LeafOnlyTryGetNodeJacobiWeightOffsets(
                in arch,
                LeafOnlyGlobalFeaturesDim,
                LeafOnlyWeightsFloatCount,
                out _,
                out _,
                out _,
                out _,
                out _,
                out _))
            return;

        if (leafOnlyJacobiInvDiagInternal == null || leafOnlyJacobiInvDiagInternal.count < nPad)
        {
            leafOnlyJacobiInvDiagInternal?.Release();
            leafOnlyJacobiInvDiagInternal = new ComputeBuffer(nPad, sizeof(float));
        }

        int mc = matrixABuffer.count;
        if (leafOnlyMatrixADiagReadbackCache == null || leafOnlyMatrixADiagReadbackCache.Length < mc)
            leafOnlyMatrixADiagReadbackCache = new float[mc];
        matrixABuffer.GetData(leafOnlyMatrixADiagReadbackCache);

        var inv = new float[nPad];
        const float eps = 1e-12f;
        for (int i = 0; i < nPad; i++)
        {
            if (i < numNodes)
            {
                float d = leafOnlyMatrixADiagReadbackCache[i];
                inv[i] = Mathf.Abs(d) > eps ? 1f / d : 0f;
            }
            else
            {
                inv[i] = 0f;
            }
        }

        leafOnlyJacobiInvDiagInternal.SetData(inv);
    }

    /// <summary>
    /// If Neural precond is enabled and GPU apply is configured, writes <paramref name="zOut"/> = M r (packed buffer must be filled).
    /// </summary>
    /// <returns>True if the GPU apply ran; false falls back to caller (e.g. Jacobi).</returns>
    private bool TryDispatchLeafOnlyPrecondPackedApply(ComputeBuffer rIn, ComputeBuffer zOut)
    {
        if (leafOnlyPrecondApplyShader == null || !LeafOnlyWeightsLoadedSuccessfully)
            return false;

        int nPad = LeafOnlyLastNPadded;
        if (nPad <= 0 || rIn == null || zOut == null)
            return false;
        if (rIn.count < nPad || zOut.count < nPad)
            return false;

        var arch = LeafOnlyCheckpoint;
        if (arch.HighwayFfnMlp != 0 || arch.DecoupledRouteGates != 0 || arch.AttentionLayoutCode != 0 || arch.NumLayers < 1)
            return false;

        int l = arch.LeafSize;
        if (l <= 0 || nPad % l != 0 || l != LeafOnlyLeafSize)
            return false;

        int kLeaves = nPad / l;
        int laD = arch.LeafApplyDiag;
        int laO = arch.LeafApplyOff;
        if (laD <= 0 || laO <= 0 || l % laD != 0 || l % laO != 0)
            return false;

        int poolD = l / laD;
        int mOff = LeafOnlyHMatrixStatic.NumOffBlocks;
        int diagSize = kLeaves * laD * laD;
        int offSz = mOff * laO * laO;
        int nodeBlock = kLeaves * l * laO;
        int minPackedNoJac = diagSize + offSz + 2 * nodeBlock;
        int minPackedJac = minPackedNoJac + kLeaves * l;

        ComputeBuffer pack = null;
        if (leafOnlyPrecondPackedBuffer != null && leafOnlyPrecondPackedBuffer.count >= minPackedNoJac)
            pack = leafOnlyPrecondPackedBuffer;
        else if (leafOnlyPrecondPackedInternal != null && leafOnlyPrecondPackedInternal.count >= minPackedNoJac)
            pack = leafOnlyPrecondPackedInternal;
        if (pack == null)
            return false;

        bool hasJacobiTail = pack.count >= minPackedJac;
        ComputeBuffer jacInv = null;
        if (hasJacobiTail)
        {
            jacInv = leafOnlyJacobiInvDiagBuffer;
            if (jacInv == null || jacInv.count < nPad)
                jacInv = leafOnlyJacobiInvDiagInternal;
            if (jacInv == null || jacInv.count < nPad)
                return false;
        }

        int offBase = diagSize;
        int uBase = offBase + mOff * laO * laO;
        int vBase = uBase + kLeaves * l * laO;
        int jacobiBase = vBase + kLeaves * l * laO;

        LeafOnlyPrecondScratchOffsets(kLeaves, laD, laO, mOff, out int oPool, out int oXu, out int oXv, out int oRs, out int oCs, out int oYr, out int oYc, out int oYrp, out int oYcp);
        EnsureLeafOnlyPrecondApplyScratch(in arch, kLeaves, mOff);
        EnsureLeafOnlyHmStaticBuffers();
        if (mOff > 0 && leafOnlyHmR0Buffer == null)
            return false;

        void BindCommon(int k)
        {
            leafOnlyPrecondApplyShader.SetBuffer(k, "leafPrecondPacked", pack);
            leafOnlyPrecondApplyShader.SetBuffer(k, "leafPrecondR", rIn);
            leafOnlyPrecondApplyShader.SetBuffer(k, "leafPrecondZ", zOut);
            leafOnlyPrecondApplyShader.SetBuffer(k, "leafPrecondScratch", leafOnlyPrecondApplyScratchBuffer);
            leafOnlyPrecondApplyShader.SetInt("leafNumPad", nPad);
            leafOnlyPrecondApplyShader.SetInt("leafNTake", Mathf.Min(numNodes, nPad));
            leafOnlyPrecondApplyShader.SetInt("leafKLeaves", kLeaves);
            leafOnlyPrecondApplyShader.SetInt("leafL", l);
            leafOnlyPrecondApplyShader.SetInt("leafLaD", laD);
            leafOnlyPrecondApplyShader.SetInt("leafLaO", laO);
            leafOnlyPrecondApplyShader.SetInt("leafPoolD", poolD);
            leafOnlyPrecondApplyShader.SetInt("leafMOff", mOff);
            leafOnlyPrecondApplyShader.SetInt("leafDiagSize", diagSize);
            leafOnlyPrecondApplyShader.SetInt("leafOffBase", offBase);
            leafOnlyPrecondApplyShader.SetInt("leafUBase", uBase);
            leafOnlyPrecondApplyShader.SetInt("leafVBase", vBase);
            leafOnlyPrecondApplyShader.SetInt("leafJacobiPackedBase", jacobiBase);
            leafOnlyPrecondApplyShader.SetInt("leafHasJacobi", hasJacobiTail ? 1 : 0);
            leafOnlyPrecondApplyShader.SetInt("leafScratchXPool", oPool);
            leafOnlyPrecondApplyShader.SetInt("leafScratchXu", oXu);
            leafOnlyPrecondApplyShader.SetInt("leafScratchXv", oXv);
            leafOnlyPrecondApplyShader.SetInt("leafScratchRowStrip", oRs);
            leafOnlyPrecondApplyShader.SetInt("leafScratchColStrip", oCs);
            leafOnlyPrecondApplyShader.SetInt("leafScratchYr", oYr);
            leafOnlyPrecondApplyShader.SetInt("leafScratchYc", oYc);
            leafOnlyPrecondApplyShader.SetInt("leafScratchYrPad", oYrp);
            leafOnlyPrecondApplyShader.SetInt("leafScratchYcPad", oYcp);
        }

        int g256(int n) => Mathf.CeilToInt(n / 256f);

        LeafOnlyPrecondApplyClearScratch(leafOnlyPrecondApplyScratchFloats);

        BindCommon(kPrecondClearZ);
        leafOnlyPrecondApplyShader.Dispatch(kPrecondClearZ, g256(nPad), 1, 1);

        BindCommon(kPrecondPoolDiag);
        leafOnlyPrecondApplyShader.Dispatch(kPrecondPoolDiag, g256(kLeaves * laD), 1, 1);

        BindCommon(kPrecondDiag);
        leafOnlyPrecondApplyShader.Dispatch(kPrecondDiag, g256(kLeaves * l), 1, 1);

        if (mOff > 0)
        {
            BindCommon(kPrecondProjUv);
            leafOnlyPrecondApplyShader.Dispatch(kPrecondProjUv, g256(kLeaves * laO), 1, 1);

            leafOnlyPrecondApplyShader.SetBuffer(kPrecondRowStrip, "leafHmR0", leafOnlyHmR0Buffer);
            leafOnlyPrecondApplyShader.SetBuffer(kPrecondRowStrip, "leafHmC0", leafOnlyHmC0Buffer);
            leafOnlyPrecondApplyShader.SetBuffer(kPrecondRowStrip, "leafHmS", leafOnlyHmSBuffer);
            BindCommon(kPrecondRowStrip);
            leafOnlyPrecondApplyShader.Dispatch(kPrecondRowStrip, g256(mOff * laO), 1, 1);

            leafOnlyPrecondApplyShader.SetBuffer(kPrecondColStrip, "leafHmR0", leafOnlyHmR0Buffer);
            leafOnlyPrecondApplyShader.SetBuffer(kPrecondColStrip, "leafHmC0", leafOnlyHmC0Buffer);
            leafOnlyPrecondApplyShader.SetBuffer(kPrecondColStrip, "leafHmS", leafOnlyHmSBuffer);
            BindCommon(kPrecondColStrip);
            leafOnlyPrecondApplyShader.Dispatch(kPrecondColStrip, g256(mOff * laO), 1, 1);

            leafOnlyPrecondApplyShader.SetBuffer(kPrecondBmmCol, "leafHmR0", leafOnlyHmR0Buffer);
            leafOnlyPrecondApplyShader.SetBuffer(kPrecondBmmCol, "leafHmC0", leafOnlyHmC0Buffer);
            leafOnlyPrecondApplyShader.SetBuffer(kPrecondBmmCol, "leafHmS", leafOnlyHmSBuffer);
            BindCommon(kPrecondBmmCol);
            leafOnlyPrecondApplyShader.Dispatch(kPrecondBmmCol, g256(mOff * laO), 1, 1);

            leafOnlyPrecondApplyShader.SetBuffer(kPrecondBmmRowT, "leafHmR0", leafOnlyHmR0Buffer);
            leafOnlyPrecondApplyShader.SetBuffer(kPrecondBmmRowT, "leafHmC0", leafOnlyHmC0Buffer);
            leafOnlyPrecondApplyShader.SetBuffer(kPrecondBmmRowT, "leafHmS", leafOnlyHmSBuffer);
            BindCommon(kPrecondBmmRowT);
            leafOnlyPrecondApplyShader.Dispatch(kPrecondBmmRowT, g256(mOff * laO), 1, 1);

            leafOnlyPrecondApplyShader.SetBuffer(kPrecondProlongRow, "leafHmR0", leafOnlyHmR0Buffer);
            leafOnlyPrecondApplyShader.SetBuffer(kPrecondProlongRow, "leafHmC0", leafOnlyHmC0Buffer);
            leafOnlyPrecondApplyShader.SetBuffer(kPrecondProlongRow, "leafHmS", leafOnlyHmSBuffer);
            BindCommon(kPrecondProlongRow);
            leafOnlyPrecondApplyShader.Dispatch(kPrecondProlongRow, g256(kLeaves * laO), 1, 1);

            leafOnlyPrecondApplyShader.SetBuffer(kPrecondProlongCol, "leafHmR0", leafOnlyHmR0Buffer);
            leafOnlyPrecondApplyShader.SetBuffer(kPrecondProlongCol, "leafHmC0", leafOnlyHmC0Buffer);
            leafOnlyPrecondApplyShader.SetBuffer(kPrecondProlongCol, "leafHmS", leafOnlyHmSBuffer);
            BindCommon(kPrecondProlongCol);
            leafOnlyPrecondApplyShader.Dispatch(kPrecondProlongCol, g256(kLeaves * laO), 1, 1);

            BindCommon(kPrecondNodeAcc);
            leafOnlyPrecondApplyShader.Dispatch(kPrecondNodeAcc, g256(kLeaves * l), 1, 1);
        }

        if (hasJacobiTail)
        {
            leafOnlyPrecondApplyShader.SetBuffer(kPrecondJacobi, "leafJacobiInvDiag", jacInv);
            BindCommon(kPrecondJacobi);
            leafOnlyPrecondApplyShader.Dispatch(kPrecondJacobi, g256(nPad), 1, 1);
        }

        return true;
    }

    /// <summary>
    /// Logs <c>apply_r</c> / <c>apply_z</c> for <c>r = ones(n_pad)</c> via the same GPU path as PCG
    /// (<see cref="TryDispatchLeafOnlyPrecondPackedApply"/>), matching <c>InspectParity.py</c> when <c>--r ones</c>.
    /// </summary>
    private void LeafOnlyParityLogGpuPackedApplyOnes(string phase)
    {
        if (leafOnlyPrecondApplyShader == null || !LeafOnlyWeightsLoadedSuccessfully)
        {
            Debug.Log($"[LeafOnlyParity] phase={phase} tensor=apply_z skipped (precond apply shader or weights)");
            return;
        }

        int nPad = LeafOnlyLastNPadded;
        if (nPad <= 0)
            return;

        LeafOnlyEnsureJacobiInvDiagFromMatrixA(nPad);

        var rOnes = new float[nPad];
        for (int i = 0; i < nPad; i++)
            rOnes[i] = 1f;
        var zOut = new float[nPad];

        var rBuf = new ComputeBuffer(nPad, sizeof(float));
        var zBuf = new ComputeBuffer(nPad, sizeof(float));
        try
        {
            rBuf.SetData(rOnes);
            zBuf.SetData(zOut);
            LeafOnlyParitySummarize(phase, "apply_r", rOnes, nPad);
            LeafOnlyParityHead(phase, "apply_r", rOnes, nPad, 16);

            if (!TryDispatchLeafOnlyPrecondPackedApply(rBuf, zBuf))
            {
                Debug.Log(
                    $"[LeafOnlyParity] phase={phase} tensor=apply_z gpu_apply_failed " +
                    "(packed buffer, jacobi tail, arch gating, or H-matrix buffers — see TryDispatchLeafOnlyPrecondPackedApply)");
                return;
            }

            zBuf.GetData(zOut, 0, 0, nPad);
            LeafOnlyParitySummarize(phase, "apply_z", zOut, nPad);
            LeafOnlyParityHead(phase, "apply_z", zOut, nPad, 16);
            Debug.Log($"[LeafOnlyParity] phase={phase} tensor=apply_meta source=gpu_packed_apply r=ones n_pad={nPad}");
        }
        finally
        {
            rBuf.Release();
            zBuf.Release();
        }
    }

    private void ReleaseLeafOnlyPrecondApplyBuffers()
    {
        leafOnlyHmR0Buffer?.Release();
        leafOnlyHmC0Buffer?.Release();
        leafOnlyHmSBuffer?.Release();
        leafOnlyPrecondApplyScratchBuffer?.Release();
        leafOnlyPrecondPackedInternal?.Release();
        leafOnlyJacobiInvDiagInternal?.Release();
        leafOnlyHmR0Buffer = null;
        leafOnlyHmC0Buffer = null;
        leafOnlyHmSBuffer = null;
        leafOnlyPrecondApplyScratchBuffer = null;
        leafOnlyPrecondPackedInternal = null;
        leafOnlyJacobiInvDiagInternal = null;
        leafOnlyPrecondApplyScratchFloats = 0;
    }
}
