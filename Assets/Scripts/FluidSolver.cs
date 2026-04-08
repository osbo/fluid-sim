using UnityEngine;
using UnityEngine.Rendering;
using System;

// CG Solver component of FluidSimulator
public partial class FluidSimulator : MonoBehaviour
{
    // cgPcgIndirectArgsBuffer layout (uint triples): SpMV (256 th/g), 512-th/g kernels, 1-group kernels.
    private const int CgIndirectArgsOffsetSpmv = 0;
    private const int CgIndirectArgsOffsetVec512 = 12;
    private const int CgIndirectArgsOffsetSingleGroup = 24;

    // Timing variables for CG loop
    private System.Diagnostics.Stopwatch laplacianSw = new System.Diagnostics.Stopwatch();
    private System.Diagnostics.Stopwatch dotProductSw = new System.Diagnostics.Stopwatch();
    private System.Diagnostics.Stopwatch updateVectorSw = new System.Diagnostics.Stopwatch();

    // Cache all solver kernel IDs once in Start() so SolvePressure never calls FindKernel.
    private void InitSolverKernels()
    {
        if (cgSolverShader == null || csrBuilderShader == null || radixSortShader == null)
        {
            Debug.LogError("One or more solver shaders not assigned. Cannot cache kernel IDs.");
            return;
        }

        calculateDivergenceKernel   = cgSolverShader.FindKernel("CalculateDivergence");
        buildMatrixAKernelId        = cgSolverShader.FindKernel("BuildMatrixA");
        applyMatrixAndDotKernelId   = cgSolverShader.FindKernel("ApplyMatrixAndDot");
        spmvCsrKernelId             = cgSolverShader.FindKernel("SpMV_CSR");
        axpyKernel                  = cgSolverShader.FindKernel("Axpy");
        scaleKernelId               = cgSolverShader.FindKernel("Scale");
        dotProductKernel            = cgSolverShader.FindKernel("DotProduct");
        applyJacobiKernelId         = cgSolverShader.FindKernel("ApplyJacobi");
        globalReduceSumKernelId     = cgSolverShader.FindKernel("GlobalReduceSum");
        copyFloatKernelId           = cgSolverShader.FindKernel("CopyFloat");
        computeAlphaKernelId        = cgSolverShader.FindKernel("ComputeAlpha");
        computeBetaKernelId         = cgSolverShader.FindKernel("ComputeBeta");
        storeRhoFromDotKernelId     = cgSolverShader.FindKernel("StoreRhoFromDot");
        axpyAlphaKernelId           = cgSolverShader.FindKernel("AxpyAlpha");
        axpyNegAlphaKernelId        = cgSolverShader.FindKernel("AxpyNegAlpha");
        scaleAddBetaKernelId        = cgSolverShader.FindKernel("ScaleAddBeta");
        checkConvergenceKernelId    = cgSolverShader.FindKernel("CheckConvergence");

        csrCountNnzKernelId         = csrBuilderShader.FindKernel("CountNNZ");
        csrFinalizeRowPtrKernelId   = csrBuilderShader.FindKernel("FinalizeRowPtr");
        csrFillKernelId             = csrBuilderShader.FindKernel("FillCSR");

        radixPrefixSumKernelId      = radixSortShader.FindKernel("prefixSum");
        radixPrefixFixupKernelId    = radixSortShader.FindKernel("prefixFixup");

        InitLeafOnlyInputKernels();
        InitLeafOnlyEmbedKernels();
        InitLeafOnlyDiagEdgeFeatsKernels();
        InitLeafOnlyOffEdgeFeatsKernels();
        InitLeafOnlyLayer1GpuKernels();
        InitLeafOnlyPrecondApplyKernels();

        applyPressureGradientKernel = nodesShader.FindKernel("ApplyPressureGradient");
        applyExternalForcesKernel   = nodesShader.FindKernel("ApplyExternalForces");
        calculateDensityGradientKernel = nodesShader.FindKernel("CalculateDensityGradient");
    }

    private void SolvePressure()
    {
        if (cgSolverShader == null)
        {
            Debug.LogError("CGSolver compute shader is not assigned.");
            return;
        }

        // --- Buffer Init (Grow-only) ---
        int requiredSize = Mathf.NextPowerOfTwo(Mathf.Max(numNodes, 512));

        if (divergenceBuffer == null || divergenceBuffer.count < requiredSize) {
            divergenceBuffer?.Release(); divergenceBuffer = new ComputeBuffer(requiredSize, sizeof(float));
        }
        if (residualBuffer == null || residualBuffer.count < requiredSize) {
            residualBuffer?.Release(); residualBuffer = new ComputeBuffer(requiredSize, sizeof(float));
        }
        if (pBuffer == null || pBuffer.count < requiredSize) {
            pBuffer?.Release(); pBuffer = new ComputeBuffer(requiredSize, sizeof(float));
        }
        if (ApBuffer == null || ApBuffer.count < requiredSize) {
            ApBuffer?.Release(); ApBuffer = new ComputeBuffer(requiredSize, sizeof(float));
        }
        if (pressureBuffer == null || pressureBuffer.count < requiredSize) {
            pressureBuffer?.Release(); pressureBuffer = new ComputeBuffer(requiredSize, sizeof(float));
        }
        if (zVectorBuffer == null || zVectorBuffer.count < requiredSize) {
            zVectorBuffer?.Release(); zVectorBuffer = new ComputeBuffer(requiredSize, sizeof(float));
        }
        if (matrixABuffer == null || matrixABuffer.count < requiredSize * 25) {
            matrixABuffer?.Release();
            matrixABuffer = new ComputeBuffer(requiredSize * 25, sizeof(float));
        }

        // CSR helper buffers — pre-allocated to max stencil width (25 entries per node).
        // This avoids the per-frame GPU→CPU readback of totalNNZ.
        int maxNnz = requiredSize * 25;
        if (nnzPerNode == null || nnzPerNode.count < requiredSize) {
            nnzPerNode?.Release();
            nnzPerNode = new ComputeBuffer(requiredSize, sizeof(uint));
        }
        if (csrRowPtr == null || csrRowPtr.count < requiredSize + 1) {
            csrRowPtr?.Release();
            csrRowPtr = new ComputeBuffer(requiredSize + 1, sizeof(uint));
        }
        if (csrColIndices == null || csrColIndices.count < maxNnz) {
            csrColIndices?.Release();
            csrColIndices = new ComputeBuffer(maxNnz, sizeof(uint));
        }
        if (csrRowIndices == null || csrRowIndices.count < maxNnz) {
            csrRowIndices?.Release();
            csrRowIndices = new ComputeBuffer(maxNnz, sizeof(uint));
        }
        if (csrValues == null || csrValues.count < maxNnz) {
            csrValues?.Release();
            csrValues = new ComputeBuffer(maxNnz, sizeof(float));
        }
        if (cgAlphaBuffer == null) cgAlphaBuffer = new ComputeBuffer(1, sizeof(float));
        if (cgBetaBuffer == null) cgBetaBuffer = new ComputeBuffer(1, sizeof(float));
        if (cgRhoBuffer == null) cgRhoBuffer = new ComputeBuffer(1, sizeof(float));

        // neighborsBuffer uses SoA stride maxNodesCapacity (see Nodes.compute); matrixABuffer still uses numNodes stride.
        cgSolverShader.SetInt("numNodesCapacity", maxNodesCapacity);
        csrBuilderShader.SetInt("numNodesCapacity", maxNodesCapacity);

        // --- Step 1: Calculate Divergence (RHS 'b') ---
        cgSolverShader.SetBuffer(calculateDivergenceKernel, "nodesBuffer", nodesBuffer);
        cgSolverShader.SetBuffer(calculateDivergenceKernel, "divergenceBuffer", divergenceBuffer);
        cgSolverShader.SetBuffer(calculateDivergenceKernel, "neighborsBuffer", neighborsBuffer);
        cgSolverShader.SetInt("numNodes", numNodes);
        Dispatch(calculateDivergenceKernel, numNodes);

        // --- Step 1b: Build Matrix A ---
        float deltaTime = useRealTime ? Time.deltaTime : (1 / frameRate);
        cgSolverShader.SetBuffer(buildMatrixAKernelId, "nodesBuffer", nodesBuffer);
        cgSolverShader.SetBuffer(buildMatrixAKernelId, "neighborsBuffer", neighborsBuffer);
        cgSolverShader.SetBuffer(buildMatrixAKernelId, "matrixABuffer", matrixABuffer);
        cgSolverShader.SetInt("numNodes", numNodes);
        cgSolverShader.SetFloat("deltaTime", deltaTime);
        int groupsA = Mathf.CeilToInt(numNodes / 256.0f);
        int groups512 = Mathf.CeilToInt(numNodes / 512.0f);
        cgSolverShader.Dispatch(buildMatrixAKernelId, groupsA, 1, 1);

        // --- Step 1c: Build CSR representation ---
        if (csrBuilderShader == null)
        {
            Debug.LogError("CSRBuilder compute shader is not assigned.");
            return;
        }

        int csrGroups = Mathf.CeilToInt(numNodes / 256.0f);

        // A. Count nnz per row
        csrBuilderShader.SetBuffer(csrCountNnzKernelId, "neighborsBuffer", neighborsBuffer);
        csrBuilderShader.SetBuffer(csrCountNnzKernelId, "nnzPerNode", nnzPerNode);
        csrBuilderShader.SetInt("numNodes", numNodes);
        csrBuilderShader.Dispatch(csrCountNnzKernelId, csrGroups, 1, 1);

        // B. Exclusive prefix sum nnzPerNode -> csrRowPtr
        uint tgSize = 512u;
        uint numThreadgroups = (uint)((numNodes + (tgSize * 2) - 1) / (tgSize * 2));
        uint auxSize = (uint)Mathf.Max(1, (int)numThreadgroups);

        if (aux == null || aux.count < auxSize) {
            aux?.Release(); aux = new ComputeBuffer((int)auxSize, sizeof(uint));
        }
        if (aux2 == null || aux2.count < auxSize) {
            aux2?.Release(); aux2 = new ComputeBuffer((int)auxSize, sizeof(uint));
        }
        if (auxSmall == null) auxSmall = new ComputeBuffer(1, sizeof(uint));

        radixSortShader.SetBuffer(radixPrefixSumKernelId, "input", nnzPerNode);
        radixSortShader.SetBuffer(radixPrefixSumKernelId, "output", csrRowPtr);
        radixSortShader.SetBuffer(radixPrefixSumKernelId, "aux", aux);
        radixSortShader.SetInt("len", numNodes);
        radixSortShader.SetInt("zeroff", 1);
        radixSortShader.Dispatch(radixPrefixSumKernelId, (int)numThreadgroups, 1, 1);

        if (numThreadgroups > 1)
        {
            radixSortShader.SetBuffer(radixPrefixSumKernelId, "input", aux);
            radixSortShader.SetBuffer(radixPrefixSumKernelId, "output", aux2);
            radixSortShader.SetBuffer(radixPrefixSumKernelId, "aux", auxSmall);
            radixSortShader.SetInt("len", (int)auxSize);
            radixSortShader.SetInt("zeroff", 1);
            radixSortShader.Dispatch(radixPrefixSumKernelId, 1, 1, 1);

            radixSortShader.SetBuffer(radixPrefixFixupKernelId, "input", csrRowPtr);
            radixSortShader.SetBuffer(radixPrefixFixupKernelId, "aux", aux2);
            radixSortShader.SetInt("len", numNodes);
            radixSortShader.Dispatch(radixPrefixFixupKernelId, (int)numThreadgroups, 1, 1);
        }

        // C. Finalize csrRowPtr[numNodes] on GPU
        csrBuilderShader.SetBuffer(csrFinalizeRowPtrKernelId, "nnzPerNode", nnzPerNode);
        csrBuilderShader.SetBuffer(csrFinalizeRowPtrKernelId, "rowPtrBuffer", csrRowPtr);
        csrBuilderShader.SetInt("numNodes", numNodes);
        csrBuilderShader.Dispatch(csrFinalizeRowPtrKernelId, 1, 1, 1);

        // D. Fill CSR colIndices / values (buffers pre-allocated — no readback needed)
        csrBuilderShader.SetBuffer(csrFillKernelId, "neighborsBuffer", neighborsBuffer);
        csrBuilderShader.SetBuffer(csrFillKernelId, "matrixABuffer", matrixABuffer);
        csrBuilderShader.SetBuffer(csrFillKernelId, "rowPtrBuffer", csrRowPtr);
        csrBuilderShader.SetBuffer(csrFillKernelId, "colIndicesBuffer", csrColIndices);
        csrBuilderShader.SetBuffer(csrFillKernelId, "rowIndicesBuffer", csrRowIndices);
        csrBuilderShader.SetBuffer(csrFillKernelId, "valuesBuffer", csrValues);
        csrBuilderShader.SetInt("numNodes", numNodes);
        csrBuilderShader.Dispatch(csrFillKernelId, csrGroups, 1, 1);

        DispatchLeafOnlyGpuInputs(numNodes, maxNnz);

        if (preconditioner == PreconditionerType.Neural)
            LeafOnlyEnsureJacobiInvDiagFromMatrixA(LeafOnlyLastNPadded);

        // Init Pressure x=0 via GPU Scale(0) — no CPU allocation
        cgSolverShader.SetBuffer(scaleKernelId, "yBuffer", pressureBuffer);
        cgSolverShader.SetFloat("a", 0.0f);
        cgSolverShader.SetInt("numNodes", numNodes);
        Dispatch(scaleKernelId, numNodes);

        // r = b (GPU copy, no CPU round-trip)
        GpuCopyBuffer(divergenceBuffer, residualBuffer);

        // --- Step 2: Preconditioned Initialization (rho on GPU only) ---
        ApplyPreconditionerInitStoreRhoGpu(residualBuffer, zVectorBuffer, applyJacobiKernelId);

        // p = z (GPU copy)
        GpuCopyBuffer(zVectorBuffer, pBuffer);

        float initialResidual = GpuDotProduct(residualBuffer, residualBuffer);
        if (initialResidual < convergenceThreshold) return;

        laplacianSw.Reset();
        dotProductSw.Reset();
        updateVectorSw.Reset();

        int cgCheckInterval = Mathf.Max(1, cgConvergenceCheckInterval);

        // --- Step 3: PCG Loop ---
        // Neural preconditioner dispatches separate compute shaders; indirect kill only covers CGSolver kernels.
        bool pcgIndirectEarlyOut = preconditioner != PreconditionerType.Neural;
        if (pcgIndirectEarlyOut)
        {
            if (cgPcgIndirectArgsBuffer == null || cgPcgIndirectArgsBuffer.count != 9)
            {
                cgPcgIndirectArgsBuffer?.Release();
                cgPcgIndirectArgsBuffer = new ComputeBuffer(9, sizeof(uint), ComputeBufferType.IndirectArguments);
            }
            if (cgPcgIterationStatBuffer == null)
                cgPcgIterationStatBuffer = new ComputeBuffer(1, sizeof(uint));
            InitCgPcgIndirectArgs(groupsA, groups512);
            cgPcgIterationStatScratch[0] = 0u;
            cgPcgIterationStatBuffer.SetData(cgPcgIterationStatScratch);
        }

        int totalIterations = maxCgIterations;

        for (int k = 0; k < maxCgIterations; k++)
        {
            // 1. Ap = A*p, partial p·Ap → divergenceBuffer
            laplacianSw.Start();
            cgSolverShader.SetBuffer(spmvCsrKernelId, "pBuffer", pBuffer);
            cgSolverShader.SetBuffer(spmvCsrKernelId, "ApBuffer", ApBuffer);
            cgSolverShader.SetBuffer(spmvCsrKernelId, "csrRowPtr", csrRowPtr);
            cgSolverShader.SetBuffer(spmvCsrKernelId, "csrColIndices", csrColIndices);
            cgSolverShader.SetBuffer(spmvCsrKernelId, "csrValues", csrValues);
            cgSolverShader.SetBuffer(spmvCsrKernelId, "divergenceBuffer", divergenceBuffer);
            cgSolverShader.SetInt("numNodes", numNodes);
            if (pcgIndirectEarlyOut)
                cgSolverShader.DispatchIndirect(spmvCsrKernelId, cgPcgIndirectArgsBuffer, CgIndirectArgsOffsetSpmv);
            else
                cgSolverShader.Dispatch(spmvCsrKernelId, groupsA, 1, 1);
            laplacianSw.Stop();

            // 2. Sum p·Ap → divergence[0]; alpha = rho / (p·Ap) on GPU
            dotProductSw.Start();
            cgSolverShader.SetBuffer(globalReduceSumKernelId, "divergenceBuffer", divergenceBuffer);
            cgSolverShader.SetInt("reductionCount", groupsA);
            if (pcgIndirectEarlyOut)
                cgSolverShader.DispatchIndirect(globalReduceSumKernelId, cgPcgIndirectArgsBuffer, CgIndirectArgsOffsetSingleGroup);
            else
                GpuFinalizeReduction(groupsA);

            if (pcgIndirectEarlyOut)
            {
                cgSolverShader.SetBuffer(computeAlphaKernelId, "divergenceBuffer", divergenceBuffer);
                cgSolverShader.SetBuffer(computeAlphaKernelId, "rhoBuffer", cgRhoBuffer);
                cgSolverShader.SetBuffer(computeAlphaKernelId, "alphaBuffer", cgAlphaBuffer);
                cgSolverShader.DispatchIndirect(computeAlphaKernelId, cgPcgIndirectArgsBuffer, CgIndirectArgsOffsetSingleGroup);
            }
            else
                DispatchComputeAlpha();
            dotProductSw.Stop();

            // 3. x += alpha * p ; r -= alpha * Ap
            updateVectorSw.Start();
            cgSolverShader.SetBuffer(axpyAlphaKernelId, "xBuffer", pBuffer);
            cgSolverShader.SetBuffer(axpyAlphaKernelId, "yBuffer", pressureBuffer);
            cgSolverShader.SetBuffer(axpyAlphaKernelId, "alphaBuffer", cgAlphaBuffer);
            cgSolverShader.SetInt("numNodes", numNodes);
            if (pcgIndirectEarlyOut)
                cgSolverShader.DispatchIndirect(axpyAlphaKernelId, cgPcgIndirectArgsBuffer, CgIndirectArgsOffsetVec512);
            else
                Dispatch(axpyAlphaKernelId, numNodes);

            cgSolverShader.SetBuffer(axpyNegAlphaKernelId, "xBuffer", ApBuffer);
            cgSolverShader.SetBuffer(axpyNegAlphaKernelId, "yBuffer", residualBuffer);
            cgSolverShader.SetBuffer(axpyNegAlphaKernelId, "alphaBuffer", cgAlphaBuffer);
            cgSolverShader.SetInt("numNodes", numNodes);
            if (pcgIndirectEarlyOut)
                cgSolverShader.DispatchIndirect(axpyNegAlphaKernelId, cgPcgIndirectArgsBuffer, CgIndirectArgsOffsetVec512);
            else
                Dispatch(axpyNegAlphaKernelId, numNodes);
            updateVectorSw.Stop();

            if (k % cgCheckInterval == 0)
            {
                dotProductSw.Start();
                if (pcgIndirectEarlyOut)
                {
                    cgSolverShader.SetBuffer(dotProductKernel, "xBuffer", residualBuffer);
                    cgSolverShader.SetBuffer(dotProductKernel, "yBuffer", residualBuffer);
                    cgSolverShader.SetBuffer(dotProductKernel, "divergenceBuffer", divergenceBuffer);
                    cgSolverShader.SetInt("numNodes", numNodes);
                    cgSolverShader.DispatchIndirect(dotProductKernel, cgPcgIndirectArgsBuffer, CgIndirectArgsOffsetVec512);
                    cgSolverShader.SetBuffer(globalReduceSumKernelId, "divergenceBuffer", divergenceBuffer);
                    cgSolverShader.SetInt("reductionCount", groups512);
                    cgSolverShader.DispatchIndirect(globalReduceSumKernelId, cgPcgIndirectArgsBuffer, CgIndirectArgsOffsetSingleGroup);
                    cgSolverShader.SetFloat("convergenceThreshold", convergenceThreshold);
                    cgSolverShader.SetFloat("initialResidualSq", initialResidual);
                    cgSolverShader.SetInt("pcgLoopK", k);
                    cgSolverShader.SetBuffer(checkConvergenceKernelId, "divergenceBuffer", divergenceBuffer);
                    cgSolverShader.SetBuffer(checkConvergenceKernelId, "cgIndirectDispatchArgs", cgPcgIndirectArgsBuffer);
                    cgSolverShader.SetBuffer(checkConvergenceKernelId, "cgPcgIterationStat", cgPcgIterationStatBuffer);
                    cgSolverShader.Dispatch(checkConvergenceKernelId, 1, 1, 1);
                }
                else
                {
                    float r_dot_r = GpuDotProduct(residualBuffer, residualBuffer);
                    if (r_dot_r < convergenceThreshold || r_dot_r > 10.0f * initialResidual)
                    {
                        totalIterations = k + 1;
                        break;
                    }
                }
                dotProductSw.Stop();
            }

            // 4. z = M^-1 r, rho_new = r·z → divergence[0]; beta and rho on GPU
            dotProductSw.Start();
            if (pcgIndirectEarlyOut)
                ApplyPreconditionerPcgIterationGpuIndirect(residualBuffer, zVectorBuffer, applyJacobiKernelId, groups512);
            else
                ApplyPreconditionerPcgIterationGpu(residualBuffer, zVectorBuffer, applyJacobiKernelId);

            cgSolverShader.SetBuffer(computeBetaKernelId, "divergenceBuffer", divergenceBuffer);
            cgSolverShader.SetBuffer(computeBetaKernelId, "rhoBuffer", cgRhoBuffer);
            cgSolverShader.SetBuffer(computeBetaKernelId, "betaBuffer", cgBetaBuffer);
            if (pcgIndirectEarlyOut)
                cgSolverShader.DispatchIndirect(computeBetaKernelId, cgPcgIndirectArgsBuffer, CgIndirectArgsOffsetSingleGroup);
            else
                cgSolverShader.Dispatch(computeBetaKernelId, 1, 1, 1);
            dotProductSw.Stop();

            // 5. p = beta * p + z
            updateVectorSw.Start();
            cgSolverShader.SetBuffer(scaleAddBetaKernelId, "pBuffer", pBuffer);
            cgSolverShader.SetBuffer(scaleAddBetaKernelId, "yBuffer", zVectorBuffer);
            cgSolverShader.SetBuffer(scaleAddBetaKernelId, "betaBuffer", cgBetaBuffer);
            cgSolverShader.SetInt("numNodes", numNodes);
            if (pcgIndirectEarlyOut)
                cgSolverShader.DispatchIndirect(scaleAddBetaKernelId, cgPcgIndirectArgsBuffer, CgIndirectArgsOffsetVec512);
            else
                Dispatch(scaleAddBetaKernelId, numNodes);
            updateVectorSw.Stop();

            if (!pcgIndirectEarlyOut)
                totalIterations = k + 1;
        }

        if (pcgIndirectEarlyOut)
        {
            AsyncGPUReadback.Request(cgPcgIterationStatBuffer, RecordCgIterationStatsFromReadback);
        }
        else
        {
            cgSolveFrameCount++;
            cumulativeCgIterations += totalIterations;
            averageCgIterations = (float)(cumulativeCgIterations / Math.Max(1, cgSolveFrameCount));
            lastCgIterations = totalIterations;
        }

        // Save training data (totalNNZ read back only when recording)
        if (recorder != null && recorder.isRecording)
        {
            int totalNNZ = csrColIndices.count; // pre-allocated upper bound
            uint[] totalNnzArr = new uint[1];
            csrRowPtr.GetData(totalNnzArr, 0, numNodes, 1);
            totalNNZ = (int)totalNnzArr[0];

            Vector3 simBoundsMin = simulationBounds != null ? simulationBounds.bounds.min : Vector3.zero;
            Vector3 simBoundsMax = simulationBounds != null ? simulationBounds.bounds.max : Vector3.zero;
            Vector3 fluidBoundsMin = fluidInitialBounds != null ? fluidInitialBounds.bounds.min : Vector3.zero;
            Vector3 fluidBoundsMax = fluidInitialBounds != null ? fluidInitialBounds.bounds.max : Vector3.zero;

            recorder.SaveFrame(
                nodesBuffer,
                neighborsBuffer,
                divergenceBuffer,
                pressureBuffer,
                diffusionGradientBuffer,
                csrRowIndices,
                csrColIndices,
                csrValues,
                totalNNZ,
                numNodes,
                minLayer,
                maxLayer,
                gravity,
                numParticles,
                maxCgIterations,
                convergenceThreshold,
                frameRate,
                simBoundsMin,
                simBoundsMax,
                fluidBoundsMin,
                fluidBoundsMax
            );
        }

        // --- Step 4: Apply pressure to velocities ---
        ApplyPressureGradient();
    }

    private void ApplyPressureGradient()
    {
        if (nodesShader == null)
        {
            Debug.LogError("Nodes compute shader is not assigned.");
            return;
        }

        BindNodesOctreeCounts();
        nodesShader.SetBuffer(applyPressureGradientKernel, "nodesBuffer", nodesBuffer);
        nodesShader.SetBuffer(applyPressureGradientKernel, "tempNodesBuffer", tempNodesBuffer);
        nodesShader.SetBuffer(applyPressureGradientKernel, "neighborsBuffer", neighborsBuffer);
        nodesShader.SetBuffer(applyPressureGradientKernel, "pressureBuffer", pressureBuffer);
        float deltaTime = useRealTime ? Time.deltaTime : (1 / frameRate);
        nodesShader.SetFloat("deltaTime", deltaTime);
        nodesShader.SetFloat("maxDetailCellSize", maxDetailCellSize);
        nodesShader.SetInt("minLayer", minLayer);

        int threadGroups = Mathf.CeilToInt(numNodes / 256.0f);
        nodesShader.Dispatch(applyPressureGradientKernel, threadGroups, 1, 1);

        ComputeBuffer swap = nodesBuffer;
        nodesBuffer = tempNodesBuffer;
        tempNodesBuffer = swap;
    }

    private void InitCgPcgIndirectArgs(int groupsA256, int groups512Ct)
    {
        cgPcgIndirectArgsScratch[0] = (uint)groupsA256;
        cgPcgIndirectArgsScratch[1] = 1u;
        cgPcgIndirectArgsScratch[2] = 1u;
        cgPcgIndirectArgsScratch[3] = (uint)groups512Ct;
        cgPcgIndirectArgsScratch[4] = 1u;
        cgPcgIndirectArgsScratch[5] = 1u;
        cgPcgIndirectArgsScratch[6] = 1u;
        cgPcgIndirectArgsScratch[7] = 1u;
        cgPcgIndirectArgsScratch[8] = 1u;
        cgPcgIndirectArgsBuffer.SetData(cgPcgIndirectArgsScratch);
    }

    private void GpuCopyBufferIndirect(ComputeBuffer source, ComputeBuffer destination)
    {
        cgSolverShader.SetBuffer(copyFloatKernelId, "xBuffer", source);
        cgSolverShader.SetBuffer(copyFloatKernelId, "yBuffer", destination);
        cgSolverShader.SetInt("numNodes", numNodes);
        cgSolverShader.DispatchIndirect(copyFloatKernelId, cgPcgIndirectArgsBuffer, CgIndirectArgsOffsetVec512);
    }

    private void GpuDotProductReduceNoReadbackIndirect(ComputeBuffer bufferA, ComputeBuffer bufferB, int groups512Ct)
    {
        cgSolverShader.SetBuffer(dotProductKernel, "xBuffer", bufferA);
        cgSolverShader.SetBuffer(dotProductKernel, "yBuffer", bufferB);
        cgSolverShader.SetBuffer(dotProductKernel, "divergenceBuffer", divergenceBuffer);
        cgSolverShader.SetInt("numNodes", numNodes);
        cgSolverShader.DispatchIndirect(dotProductKernel, cgPcgIndirectArgsBuffer, CgIndirectArgsOffsetVec512);
        cgSolverShader.SetBuffer(globalReduceSumKernelId, "divergenceBuffer", divergenceBuffer);
        cgSolverShader.SetInt("reductionCount", groups512Ct);
        cgSolverShader.DispatchIndirect(globalReduceSumKernelId, cgPcgIndirectArgsBuffer, CgIndirectArgsOffsetSingleGroup);
    }

    private void Dispatch(int kernel, int count)
    {
        int threadGroups = Mathf.CeilToInt(count / 512.0f);
        cgSolverShader.Dispatch(kernel, threadGroups, 1, 1);
    }

    // Legacy name kept for FluidPreconditioner.cs callers.
    private void CopyBuffer(ComputeBuffer source, ComputeBuffer destination) => GpuCopyBuffer(source, destination);

    // GPU-only buffer copy: no CPU round-trip.
    private void GpuCopyBuffer(ComputeBuffer source, ComputeBuffer destination)
    {
        cgSolverShader.SetBuffer(copyFloatKernelId, "xBuffer", source);
        cgSolverShader.SetBuffer(copyFloatKernelId, "yBuffer", destination);
        cgSolverShader.SetInt("numNodes", numNodes);
        Dispatch(copyFloatKernelId, numNodes);
    }

    private void GpuFinalizeReduction(int groupCount)
    {
        cgSolverShader.SetBuffer(globalReduceSumKernelId, "divergenceBuffer", divergenceBuffer);
        cgSolverShader.SetInt("reductionCount", groupCount);
        cgSolverShader.Dispatch(globalReduceSumKernelId, 1, 1, 1);
    }

    // Finishes the reduction already started by SpMV (partial sums in divergenceBuffer[0..groupCount-1]).
    private float GpuReduceSum(int groupCount)
    {
        GpuFinalizeReduction(groupCount);
        divergenceBuffer.GetData(reductionResult, 0, 0, 1);
        return reductionResult[0];
    }

    private void GpuDotProductReduceNoReadback(ComputeBuffer bufferA, ComputeBuffer bufferB)
    {
        cgSolverShader.SetBuffer(dotProductKernel, "xBuffer", bufferA);
        cgSolverShader.SetBuffer(dotProductKernel, "yBuffer", bufferB);
        cgSolverShader.SetBuffer(dotProductKernel, "divergenceBuffer", divergenceBuffer);
        cgSolverShader.SetInt("numNodes", numNodes);
        int threadGroups = Mathf.CeilToInt(numNodes / 512.0f);
        cgSolverShader.Dispatch(dotProductKernel, threadGroups, 1, 1);
        GpuFinalizeReduction(threadGroups);
    }

    private void DispatchStoreRhoFromDot()
    {
        cgSolverShader.SetBuffer(storeRhoFromDotKernelId, "divergenceBuffer", divergenceBuffer);
        cgSolverShader.SetBuffer(storeRhoFromDotKernelId, "rhoBuffer", cgRhoBuffer);
        cgSolverShader.Dispatch(storeRhoFromDotKernelId, 1, 1, 1);
    }

    private void DispatchComputeAlpha()
    {
        cgSolverShader.SetBuffer(computeAlphaKernelId, "divergenceBuffer", divergenceBuffer);
        cgSolverShader.SetBuffer(computeAlphaKernelId, "rhoBuffer", cgRhoBuffer);
        cgSolverShader.SetBuffer(computeAlphaKernelId, "alphaBuffer", cgAlphaBuffer);
        cgSolverShader.Dispatch(computeAlphaKernelId, 1, 1, 1);
    }

    private void DispatchComputeBeta()
    {
        cgSolverShader.SetBuffer(computeBetaKernelId, "divergenceBuffer", divergenceBuffer);
        cgSolverShader.SetBuffer(computeBetaKernelId, "rhoBuffer", cgRhoBuffer);
        cgSolverShader.SetBuffer(computeBetaKernelId, "betaBuffer", cgBetaBuffer);
        cgSolverShader.Dispatch(computeBetaKernelId, 1, 1, 1);
    }

    private float GpuDotProduct(ComputeBuffer bufferA, ComputeBuffer bufferB)
    {
        if (cgSolverShader == null) return 0.0f;

        cgSolverShader.SetBuffer(dotProductKernel, "xBuffer", bufferA);
        cgSolverShader.SetBuffer(dotProductKernel, "yBuffer", bufferB);
        cgSolverShader.SetBuffer(dotProductKernel, "divergenceBuffer", divergenceBuffer);
        cgSolverShader.SetInt("numNodes", numNodes);

        int threadGroups = Mathf.CeilToInt(numNodes / 512.0f);
        cgSolverShader.Dispatch(dotProductKernel, threadGroups, 1, 1);

        return GpuReduceSum(threadGroups);
    }

    private void UpdateVector(ComputeBuffer yBuffer, ComputeBuffer xBuffer, float a)
    {
        if (cgSolverShader == null) return;

        cgSolverShader.SetBuffer(axpyKernel, "xBuffer", xBuffer);
        cgSolverShader.SetBuffer(axpyKernel, "yBuffer", yBuffer);
        cgSolverShader.SetFloat("a", a);
        cgSolverShader.SetInt("numNodes", numNodes);
        Dispatch(axpyKernel, numNodes);
    }

    private void UpdateVector(ComputeBuffer pBuf, ComputeBuffer rBuffer, float rCoeff, float pCoeff)
    {
        if (cgSolverShader == null) return;

        // p = beta * p
        cgSolverShader.SetBuffer(scaleKernelId, "yBuffer", pBuf);
        cgSolverShader.SetFloat("a", pCoeff);
        cgSolverShader.SetInt("numNodes", numNodes);
        Dispatch(scaleKernelId, numNodes);

        // p = r + p  (rCoeff is always 1.0)
        cgSolverShader.SetBuffer(axpyKernel, "xBuffer", rBuffer);
        cgSolverShader.SetBuffer(axpyKernel, "yBuffer", pBuf);
        cgSolverShader.SetFloat("a", rCoeff);
        cgSolverShader.SetInt("numNodes", numNodes);
        Dispatch(axpyKernel, numNodes);
    }

    /// <summary>Finishes after the GPU has written <see cref="cgPcgIterationStatBuffer"/>; updates CG profiling (typically next frame).</summary>
    private void RecordCgIterationStatsFromReadback(AsyncGPUReadbackRequest req)
    {
        if (req.hasError)
            return;
        var data = req.GetData<uint>();
        if (!data.IsCreated || data.Length == 0)
            return;
        uint v = data[0];
        int iters = v != 0u ? (int)v : maxCgIterations;
        lastCgIterations = iters;
        cgSolveFrameCount++;
        cumulativeCgIterations += iters;
        averageCgIterations = (float)(cumulativeCgIterations / Math.Max(1, cgSolveFrameCount));
    }

    private void ApplyExternalForces()
    {
        if (nodesShader == null) return;

        BindNodesOctreeCounts();
        WriteIndirectArgsFromCountBuffer(nodeCount);
        nodesShader.SetBuffer(applyExternalForcesKernel, "nodesBuffer", nodesBuffer);
        nodesShader.SetFloat("gravity", gravity);
        float deltaTime = useRealTime ? Time.deltaTime : (1 / frameRate);
        nodesShader.SetFloat("deltaTime", deltaTime);
        nodesShader.DispatchIndirect(applyExternalForcesKernel, dispatchArgsBuffer, 0);
    }
}
