using System;
using System.Text;
using UnityEngine;

public partial class FluidSimulator
{
    /// <summary>D3D11 (and Unity) cap on dispatch thread-group count per dimension.</summary>
    private const int MaxComputeThreadGroupsPerDimension = 65535;
    private const int UniformGridClearDenseThreadsPerGroup = 256;

    private int uniformGridClearNodeAccumKernel;
    private int uniformGridSplatParticlesKernel;
    private int uniformGridNormalizeNodesKernel;
    private int uniformGridFindNeighborsKernel;
    private int uniformGridApplyPressureKernel = -1;
    private int uniformGridApplyViscosityKernel = -1;
    private int uniformGridInterpolateFaceVelocitiesKernel = -1;
    private int uniformGridCopyFaceVelocitiesKernel = -1;
    private int uniformGridCalculateDensityGradientKernel = -1;

    private void InitUniformGridKernels()
    {
        if (uniformGridShader == null)
        {
            uniformGridApplyPressureKernel = -1;
            uniformGridApplyViscosityKernel = -1;
            uniformGridInterpolateFaceVelocitiesKernel = -1;
            uniformGridCopyFaceVelocitiesKernel = -1;
            uniformGridCalculateDensityGradientKernel = -1;
            Debug.LogWarning("FluidSimulator: UniformGrid.compute is not assigned; uniform grid dispatches are skipped.");
            return;
        }

        uniformGridClearDenseKernel = uniformGridShader.FindKernel("UniformGrid_ClearDense");
        uniformGridBinParticlesKernel = uniformGridShader.FindKernel("UniformGrid_BinParticles");
        uniformGridClearNodeAccumKernel = uniformGridShader.FindKernel("UniformGrid_ClearNodeAccum");
        uniformGridSplatParticlesKernel = uniformGridShader.FindKernel("UniformGrid_SplatParticles");
        uniformGridNormalizeNodesKernel = uniformGridShader.FindKernel("UniformGrid_NormalizeNodes");
        uniformGridFindNeighborsKernel = uniformGridShader.FindKernel("UniformGrid_FindNeighbors");
        uniformGridApplyPressureKernel = uniformGridShader.FindKernel("UniformGrid_ApplyPressureGradient");
        uniformGridApplyViscosityKernel = uniformGridShader.FindKernel("UniformGrid_ApplyViscosity");
        uniformGridInterpolateFaceVelocitiesKernel = uniformGridShader.FindKernel("UniformGrid_InterpolateFaceVelocities");
        uniformGridCopyFaceVelocitiesKernel = uniformGridShader.FindKernel("UniformGrid_CopyFaceVelocities");
        uniformGridCalculateDensityGradientKernel = uniformGridShader.FindKernel("UniformGrid_CalculateDensityGradient");
    }

    /// <summary>
    /// Clears N³ maps, bins particles, atomically splats into per-node accumulators, normalizes nodes.
    /// No sort or prefix-sum required.
    /// </summary>
    private void DispatchUniformGridClearAndBin()
    {
        numNodes = 0;

        if (uniformGridShader == null || particlesBuffer == null || uniformCellCountsBuffer == null)
            return;

        GetUniformGridDims(out _, out _, out int expectedCellCount);
        if (expectedCellCount <= 0)
            return;

        // N³ depends on uniformGridCellLayer; AllocateOctreeBuffersToCapacity() often skips realloc when only the layer changes.
        if (uniformCellCountsBuffer == null || uniformDenseIndexMapBuffer == null)
            AllocateOctreeBuffersToCapacity();
        else if (uniformCellCountsBuffer.count != expectedCellCount || uniformDenseIndexMapBuffer.count != expectedCellCount)
            AllocateUniformGridBuffers();

        int cellCount = UniformGridCellCount;
        if (cellCount <= 0)
            return;

        int k = UniformGridBinsPerAxisLog2;
        int layer = Mathf.Clamp(uniformGridCellLayer, 0, MortonAxisBits);

        uniformGridShader.SetInt("uniformGridLog2K", k);
        uniformGridShader.SetInt("uniformGridCellCount", cellCount);
        uniformGridShader.SetInt("uniformGridCellLayer", layer);
        uniformGridShader.SetInt("numParticles", numParticles);
        uniformGridShader.SetInt("maxUniformActiveCells", maxNodesCapacity);
        uniformGridShader.SetInt("numNodesCapacity", maxNodesCapacity);

        // 1. Clear dense maps (chunk if N³ needs more than MaxComputeThreadGroupsPerDimension groups)
        uniformGridShader.SetBuffer(uniformGridClearDenseKernel, "cellCounts", uniformCellCountsBuffer);
        uniformGridShader.SetBuffer(uniformGridClearDenseKernel, "denseIndexMap", uniformDenseIndexMapBuffer);
        uniformGridShader.SetBuffer(uniformGridClearDenseKernel, "activeNodeCount", uniformActiveNodeCountBuffer);
        int maxCellsPerClearPass = MaxComputeThreadGroupsPerDimension * UniformGridClearDenseThreadsPerGroup;
        int clearBase = 0;
        while (clearBase < cellCount)
        {
            int cellsThisPass = Mathf.Min(cellCount - clearBase, maxCellsPerClearPass);
            int clearGroups = Mathf.Max(1, (cellsThisPass + UniformGridClearDenseThreadsPerGroup - 1) / UniformGridClearDenseThreadsPerGroup);
            uniformGridShader.SetInt("uniformGridClearCellBase", clearBase);
            uniformGridShader.Dispatch(uniformGridClearDenseKernel, clearGroups, 1, 1);
            clearBase += cellsThisPass;
        }

        // 2. Bin particles: assign node IDs, populate denseIndexMap and activeMortonList
        uniformGridShader.SetBuffer(uniformGridBinParticlesKernel, "particlesBuffer", particlesBuffer);
        uniformGridShader.SetBuffer(uniformGridBinParticlesKernel, "cellCounts", uniformCellCountsBuffer);
        uniformGridShader.SetBuffer(uniformGridBinParticlesKernel, "denseIndexMap", uniformDenseIndexMapBuffer);
        uniformGridShader.SetBuffer(uniformGridBinParticlesKernel, "activeNodeCount", uniformActiveNodeCountBuffer);
        uniformGridShader.SetBuffer(uniformGridBinParticlesKernel, "activeMortonList", uniformActiveMortonListBuffer);
        int binGroups = Mathf.Max(1, (numParticles + 255) / 256);
        uniformGridShader.Dispatch(uniformGridBinParticlesKernel, binGroups, 1, 1);

        // Read back active node count (needed to size subsequent dispatches)
        uniformActiveNodeCountBuffer.GetData(gpuNodeCountReadback);
        int capped = Mathf.Min((int)gpuNodeCountReadback[0], maxNodesCapacity);
        if (capped <= 0)
        {
            gpuUintScratch1[0] = 0u;
            nodeCount.SetData(gpuUintScratch1);
            return;
        }

        uniformGridShader.SetInt("uniformGridBuildNodeCount", capped);

        // 3. Clear per-node accumulation buffer
        uniformGridShader.SetBuffer(uniformGridClearNodeAccumKernel, "activeNodeCount", uniformActiveNodeCountBuffer);
        uniformGridShader.SetBuffer(uniformGridClearNodeAccumKernel, "nodeAccumBuffer", uniformNodeAccumBuffer);
        int clearAccumGroups = Mathf.Max(1, (capped + 511) / 512);
        uniformGridShader.Dispatch(uniformGridClearNodeAccumKernel, clearAccumGroups, 1, 1);

        // 4. Splat particles into accumulation buffer
        uniformGridShader.SetBuffer(uniformGridSplatParticlesKernel, "particlesBuffer", particlesBuffer);
        uniformGridShader.SetBuffer(uniformGridSplatParticlesKernel, "denseIndexMap", uniformDenseIndexMapBuffer);
        uniformGridShader.SetBuffer(uniformGridSplatParticlesKernel, "nodeAccumBuffer", uniformNodeAccumBuffer);
        int splatGroups = Mathf.Max(1, (numParticles + 255) / 256);
        uniformGridShader.Dispatch(uniformGridSplatParticlesKernel, splatGroups, 1, 1);

        // 5. Normalize: read accumulators, write final Node structs
        uniformGridShader.SetBuffer(uniformGridNormalizeNodesKernel, "activeNodeCount", uniformActiveNodeCountBuffer);
        uniformGridShader.SetBuffer(uniformGridNormalizeNodesKernel, "nodeAccumBuffer", uniformNodeAccumBuffer);
        uniformGridShader.SetBuffer(uniformGridNormalizeNodesKernel, "activeMortonList", uniformActiveMortonListBuffer);
        uniformGridShader.SetBuffer(uniformGridNormalizeNodesKernel, "nodesBuffer", nodesBuffer);
        uniformGridShader.SetBuffer(uniformGridNormalizeNodesKernel, "mortonCodesBuffer", mortonCodesBuffer);
        uniformGridShader.Dispatch(uniformGridNormalizeNodesKernel, clearAccumGroups, 1, 1);

        gpuUintScratch1[0] = (uint)capped;
        nodeCount.SetData(gpuUintScratch1);
        numNodes = capped;

        DispatchUniformGridFindNeighbors(cellCount, k);
        if (debugValidateUniformNeighbors && gridMode == GridMode.Uniform)
            ValidateUniformGridNeighborsDebug(cellCount, k);
        DispatchUniformGridInterpolateFaceVelocities();
    }

    /// <summary>
    /// Fills <see cref="uniformNeighborsBuffer"/> from <see cref="uniformDenseIndexMapBuffer"/> (6 orthogonal faces, O(1) per node).
    /// Domain boundary uses numFluidNodes+1; empty cells use 0xFFFFFFFF (Dirichlet air).
    /// </summary>
    private void DispatchUniformGridFindNeighbors(int cellCount, int binsK)
    {
        if (uniformGridShader == null || uniformNeighborsBuffer == null || nodesBuffer == null || uniformDenseIndexMapBuffer == null)
            return;
        if (uniformGridFindNeighborsKernel < 0 || numNodes <= 0)
            return;

        uniformGridShader.SetInt("uniformGridLog2K", binsK);
        uniformGridShader.SetInt("uniformGridCellCount", cellCount);
        uniformGridShader.SetInt("numFluidNodes", numNodes);
        uniformGridShader.SetInt("numNodesCapacity", maxNodesCapacity);

        uniformGridShader.SetBuffer(uniformGridFindNeighborsKernel, "denseIndexMap", uniformDenseIndexMapBuffer);
        uniformGridShader.SetBuffer(uniformGridFindNeighborsKernel, "nodesBuffer", nodesBuffer);
        uniformGridShader.SetBuffer(uniformGridFindNeighborsKernel, "uniformNeighborsBuffer", uniformNeighborsBuffer);

        int groups = Mathf.Max(1, (numNodes + 511) / 512);
        uniformGridShader.Dispatch(uniformGridFindNeighborsKernel, groups, 1, 1);
    }

    /// <summary>
    /// Step 9: 1:1 face velocity matching across faces (no AMR T-junctions). Mirrors octree interpolateFaceVelocities + copyFaceVelocities.
    /// </summary>
    private void DispatchUniformGridInterpolateFaceVelocities()
    {
        if (uniformGridShader == null || numNodes <= 0 || nodesBuffer == null || tempNodesBuffer == null
            || uniformNeighborsBuffer == null || nodeCount == null)
            return;
        if (uniformGridInterpolateFaceVelocitiesKernel < 0 || uniformGridCopyFaceVelocitiesKernel < 0)
            return;

        uniformGridShader.SetInt("numNodesCapacity", maxNodesCapacity);
        uniformGridShader.SetBuffer(uniformGridInterpolateFaceVelocitiesKernel, "nodeCountBuffer", nodeCount);
        uniformGridShader.SetBuffer(uniformGridInterpolateFaceVelocitiesKernel, "nodesBuffer", nodesBuffer);
        uniformGridShader.SetBuffer(uniformGridInterpolateFaceVelocitiesKernel, "tempNodesBuffer", tempNodesBuffer);
        uniformGridShader.SetBuffer(uniformGridInterpolateFaceVelocitiesKernel, "uniformNeighborsBuffer", uniformNeighborsBuffer);

        uniformGridShader.SetBuffer(uniformGridCopyFaceVelocitiesKernel, "nodeCountBuffer", nodeCount);
        uniformGridShader.SetBuffer(uniformGridCopyFaceVelocitiesKernel, "nodesBuffer", nodesBuffer);
        uniformGridShader.SetBuffer(uniformGridCopyFaceVelocitiesKernel, "tempNodesBuffer", tempNodesBuffer);

        int groups = Mathf.Max(1, (numNodes + 511) / 512);
        uniformGridShader.Dispatch(uniformGridInterpolateFaceVelocitiesKernel, groups, 1, 1);
        uniformGridShader.Dispatch(uniformGridCopyFaceVelocitiesKernel, groups, 1, 1);
    }

    private static uint UniformCellMortonFromFineMorton(uint fineMorton, int k)
    {
        uint s = (uint)(MortonAxisBits - k);
        uint x = 0, y = 0, z = 0;
        for (int i = 0; i < MortonAxisBits; i++)
        {
            x |= ((fineMorton >> (3 * i + 0)) & 1u) << i;
            y |= ((fineMorton >> (3 * i + 1)) & 1u) << i;
            z |= ((fineMorton >> (3 * i + 2)) & 1u) << i;
        }

        x >>= (int)s;
        y >>= (int)s;
        z >>= (int)s;

        uint m = 0;
        for (uint i = 0; i < k; i++)
        {
            m |= (((x >> (int)i) & 1u) << (int)(3 * i + 0));
            m |= (((y >> (int)i) & 1u) << (int)(3 * i + 1));
            m |= (((z >> (int)i) & 1u) << (int)(3 * i + 2));
        }

        return m;
    }

    private static void FineMortonToUniformCell(uint fineMorton, int k, out int cx, out int cy, out int cz)
    {
        uint s = (uint)(MortonAxisBits - k);
        uint x = 0, y = 0, z = 0;
        for (int i = 0; i < MortonAxisBits; i++)
        {
            x |= ((fineMorton >> (3 * i + 0)) & 1u) << i;
            y |= ((fineMorton >> (3 * i + 1)) & 1u) << i;
            z |= ((fineMorton >> (3 * i + 2)) & 1u) << i;
        }

        cx = (int)(x >> (int)s);
        cy = (int)(y >> (int)s);
        cz = (int)(z >> (int)s);
    }

    private static bool UniformFaceExpectsDomainWall(int face, int cx, int cy, int cz, int gridDim)
    {
        return face switch
        {
            0 => cx == 0,
            1 => cx == gridDim - 1,
            2 => cy == 0,
            3 => cy == gridDim - 1,
            4 => cz == 0,
            5 => cz == gridDim - 1,
            _ => false
        };
    }

    private static readonly string[] UniformFaceNames = { "−x", "+x", "−y", "+y", "−z", "+z" };

    private static string UniformNeighborProbeLabel(uint v, int nAct, uint wallSentinel, uint airSentinel)
    {
        if (v == airSentinel) return "AIR";
        if (v == wallSentinel) return "WALL";
        if (v < (uint)nAct) return v.ToString();
        return $"?({v})";
    }

    /// <summary>
    /// After GPU FindNeighbors: read back stencil, report reciprocity / wall / dense map issues with per-face (+/− axis) counts and examples.
    /// </summary>
    private void ValidateUniformGridNeighborsDebug(int cellCount, int k)
    {
        int n = numNodes;
        if (n <= 0 || uniformNeighborsBuffer == null || mortonCodesBuffer == null || maxNodesCapacity <= 0)
            return;

        // Pull this many consecutive node slots from the GPU (must cover all fluid neighbors you want reciprocity for).
        int nRead = Mathf.Min(n, debugUniformNeighborsMaxNodes);
        if (nRead < n)
        {
            Debug.LogWarning(
                $"[UniformNeighbors] GPU readback capped at {nRead}/{n} nodes (raise debugUniformNeighborsMaxNodes). " +
                "Reciprocity is only checked when neighbor index < nRead.");
        }

        if (uniformNeighborDebugFacesScratch == null || uniformNeighborDebugFacesScratch.Length < 6 * nRead)
            uniformNeighborDebugFacesScratch = new uint[6 * nRead];
        if (uniformNeighborDebugMortonScratch == null || uniformNeighborDebugMortonScratch.Length < nRead)
            uniformNeighborDebugMortonScratch = new uint[nRead];

        var faces = uniformNeighborDebugFacesScratch;
        var morton = uniformNeighborDebugMortonScratch;

        for (int d = 0; d < 6; d++)
        {
            int gpuOffset = d * maxNodesCapacity;
            uniformNeighborsBuffer.GetData(faces, d * nRead, gpuOffset, nRead);
        }

        mortonCodesBuffer.GetData(morton, 0, 0, nRead);

        uint wall = (uint)(n + 1);
        const uint air = uint.MaxValue;
        int gridDim = 1 << k;

        const int ORecip = 0, OWallMiss = 6, OWallSpur = 12, OAirInt = 18, OFluidInt = 24;
        if (uniformNeighborsDebugFaceStats30 == null || uniformNeighborsDebugFaceStats30.Length != 30)
            uniformNeighborsDebugFaceStats30 = new int[30];
        int[] st = uniformNeighborsDebugFaceStats30;
        Array.Clear(st, 0, 30);

        int recipSkippedNbOutOfRange = 0;
        int denseMismatch = 0;
        int coordsOutOfRange = 0;

        if (uniformNeighborsDebugSbRecip == null) uniformNeighborsDebugSbRecip = new StringBuilder(256);
        else uniformNeighborsDebugSbRecip.Clear();
        if (uniformNeighborsDebugSbWall == null) uniformNeighborsDebugSbWall = new StringBuilder(256);
        else uniformNeighborsDebugSbWall.Clear();
        if (uniformNeighborsDebugSbDense == null) uniformNeighborsDebugSbDense = new StringBuilder(256);
        else uniformNeighborsDebugSbDense.Clear();
        StringBuilder exRecip = uniformNeighborsDebugSbRecip;
        StringBuilder exWall = uniformNeighborsDebugSbWall;
        StringBuilder exDense = uniformNeighborsDebugSbDense;
        int exRecipN = 0, exWallN = 0, exDenseN = 0;
        int maxEx = debugUniformNeighborsMaxExamples;

        bool haveDense = uniformDenseIndexMapBuffer != null && cellCount <= debugUniformNeighborsMaxDenseCells;
        if (haveDense)
        {
            if (uniformNeighborDebugDenseScratch == null || uniformNeighborDebugDenseScratch.Length < cellCount)
                uniformNeighborDebugDenseScratch = new uint[cellCount];
            uniformDenseIndexMapBuffer.GetData(uniformNeighborDebugDenseScratch, 0, 0, cellCount);
        }

        for (int i = 0; i < nRead; i++)
        {
            FineMortonToUniformCell(morton[i], k, out int cx, out int cy, out int cz);
            if (cx < 0 || cy < 0 || cz < 0 || cx >= gridDim || cy >= gridDim || cz >= gridDim)
            {
                coordsOutOfRange++;
                continue;
            }

            if (haveDense)
            {
                uint cm = UniformCellMortonFromFineMorton(morton[i], k);
                if (cm >= (uint)cellCount)
                {
                    if (exDenseN < maxEx)
                    {
                        exDense.AppendLine($"  node {i}: cellMorton {cm} >= cellCount {cellCount} (fine morton {morton[i]})");
                        exDenseN++;
                    }

                    denseMismatch++;
                }
                else
                {
                    uint mapped = uniformNeighborDebugDenseScratch[cm];
                    if (mapped != (uint)i)
                    {
                        denseMismatch++;
                        if (exDenseN < maxEx)
                        {
                            exDense.AppendLine(
                                $"  node {i}: cell ({cx},{cy},{cz}) cellMorton {cm} → denseIndexMap[]={mapped} (expected {i})");
                            exDenseN++;
                        }
                    }
                }
            }

            for (int d = 0; d < 6; d++)
            {
                uint nb = faces[d * nRead + i];
                bool expectWall = UniformFaceExpectsDomainWall(d, cx, cy, cz, gridDim);

                if (expectWall)
                {
                    if (nb != wall)
                    {
                        st[OWallMiss + d]++;
                        if (exWallN < maxEx)
                        {
                            exWall.AppendLine(
                                $"  node {i} cell ({cx},{cy},{cz}) face {UniformFaceNames[d]}: expected WALL ({wall}), got {UniformNeighborProbeLabel(nb, n, wall, air)}");
                            exWallN++;
                        }
                    }
                }
                else
                {
                    if (nb == air)
                        st[OAirInt + d]++;
                    else if (nb < (uint)n)
                        st[OFluidInt + d]++;

                    if (nb == wall)
                    {
                        st[OWallSpur + d]++;
                        if (exWallN < maxEx)
                        {
                            exWall.AppendLine(
                                $"  node {i} cell ({cx},{cy},{cz}) face {UniformFaceNames[d]}: unexpected WALL (interior/half-open)");
                            exWallN++;
                        }
                    }
                }

                if (nb < (uint)nRead)
                {
                    int opp = d ^ 1;
                    uint back = faces[opp * nRead + (int)nb];
                    if (back != (uint)i)
                    {
                        st[ORecip + d]++;
                        if (exRecipN < maxEx)
                        {
                            exRecip.AppendLine(
                                $"  node {i} cell ({cx},{cy},{cz}) → face {UniformFaceNames[d]} → node {nb}; " +
                                $"back face {UniformFaceNames[opp]} of {nb} = {UniformNeighborProbeLabel(back, n, wall, air)} (expected {i})");
                            exRecipN++;
                        }
                    }
                }
                else if (nb < (uint)n && nb >= (uint)nRead)
                    recipSkippedNbOutOfRange++;
            }
        }

        int sumRecip = 0;
        for (int d = 0; d < 6; d++) sumRecip += st[ORecip + d];
        int sumWallMiss = 0, sumWallSpur = 0;
        for (int d = 0; d < 6; d++)
        {
            sumWallMiss += st[OWallMiss + d];
            sumWallSpur += st[OWallSpur + d];
        }

        if (uniformNeighborsDebugSbSummary == null) uniformNeighborsDebugSbSummary = new StringBuilder(512);
        else uniformNeighborsDebugSbSummary.Clear();
        StringBuilder sb = uniformNeighborsDebugSbSummary;
        sb.Append("[UniformNeighbors] ");
        sb.Append($"nodes={n} read={nRead} k={k} gridDim={gridDim} wall={wall} air=0xFFFFFFFF | ");
        sb.Append($"recipFail={sumRecip} wallMissing={sumWallMiss} wallSpurious={sumWallSpur} denseMismatch={denseMismatch} coordsBad={coordsOutOfRange}");
        if (recipSkippedNbOutOfRange > 0)
            sb.Append($" recipSkipped(nb in [{nRead},{n}))={recipSkippedNbOutOfRange}");
        if (!haveDense)
            sb.Append($" | dense map skipped (N³={cellCount} > debugUniformNeighborsMaxDenseCells={debugUniformNeighborsMaxDenseCells})");

        Debug.Log(sb.ToString());

        void LogAxisPair(string axis, int negFace, int posFace)
        {
            Debug.Log(
                $"[UniformNeighbors] axis {axis}: " +
                $"recip −={st[ORecip + negFace]} +={st[ORecip + posFace]} | " +
                $"wallMissing −={st[OWallMiss + negFace]} +={st[OWallMiss + posFace]} | " +
                $"wallSpurious −={st[OWallSpur + negFace]} +={st[OWallSpur + posFace]} | " +
                $"interior AIR −={st[OAirInt + negFace]} +={st[OAirInt + posFace]} | " +
                $"interior FLUID −={st[OFluidInt + negFace]} +={st[OFluidInt + posFace]}");
        }

        LogAxisPair("x", 0, 1);
        LogAxisPair("y", 2, 3);
        LogAxisPair("z", 4, 5);

        if (sumRecip > 0 && exRecipN > 0)
            Debug.Log($"[UniformNeighbors] reciprocity examples (max {maxEx}):\n{exRecip}");
        if ((sumWallMiss > 0 || sumWallSpur > 0) && exWallN > 0)
            Debug.Log($"[UniformNeighbors] wall examples (max {maxEx}):\n{exWall}");
        if (denseMismatch > 0 && exDenseN > 0)
            Debug.Log($"[UniformNeighbors] denseIndexMap examples (max {maxEx}):\n{exDense}");
    }
}
