using UnityEngine;
using System;

// CG Solver component of FluidSimulator
public partial class FluidSimulator : MonoBehaviour
{
    // solverIndirectArgsBuffer byte offsets (each triple = 3 uints = 12 bytes):
    private const int SolverArgsOffset256         = 0;   // ceil(n/256) groups — BuildMatrixA, Divergence, CSR, ApplyPressure
    private const int SolverArgsOffset512         = 12;  // ceil(n/512) groups — Scale, Axpy, Dot, Copy, Jacobi, AxpyAlpha/NegAlpha, ScaleAddBeta
    private const int SolverArgsOffsetSingleGroup = 24;  // 1 group — GlobalReduceSum, ComputeAlpha/Beta, StoreRho
    private const int SolverArgsOffsetPfx1        = 36;  // ceil(n/1024) groups — CSR prefix sum passes 1 & fixup
    // Aliases matching old CgIndirectArgs names so CG loop code is unchanged:
    private const int CgIndirectArgsOffsetSpmv        = SolverArgsOffset256;
    private const int CgIndirectArgsOffsetVec512       = SolverArgsOffset512;
    private const int CgIndirectArgsOffsetSingleGroup  = SolverArgsOffsetSingleGroup;

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
        csrCountNnzKernelId         = csrBuilderShader.FindKernel("CountNNZ");
        csrFinalizeRowPtrKernelId   = csrBuilderShader.FindKernel("FinalizeRowPtr");
        csrFillKernelId             = csrBuilderShader.FindKernel("FillCSR");

        radixPrefixSumKernelId           = radixSortShader.FindKernel("prefixSum");
        radixPrefixFixupKernelId         = radixSortShader.FindKernel("prefixFixup");
        radixPrefixSumSolverMainKernelId = radixSortShader.FindKernel("prefixSumSolverMain");
        radixPrefixSumSolverAuxKernelId  = radixSortShader.FindKernel("prefixSumSolverAux");
        radixPrefixFixupSolverKernelId   = radixSortShader.FindKernel("prefixFixupSolver");
        buildSolverIndirectArgsKernelId  = nodesPrefixSumsShader.FindKernel("BuildSolverIndirectArgs");

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

        // All solver buffers are pre-allocated to maxNodesCapacity in AllocateOctreeBuffersToCapacity.
        // Build all GPU-driven indirect dispatch args and scalar counts in one kernel.
        int maxNnz = matrixABuffer.count; // = maxNodesCapacity * 25
        if (cgAlphaBuffer == null) cgAlphaBuffer = new ComputeBuffer(1, sizeof(float));
        if (cgBetaBuffer == null) cgBetaBuffer = new ComputeBuffer(1, sizeof(float));
        if (cgRhoBuffer == null) cgRhoBuffer = new ComputeBuffer(1, sizeof(float));

        const bool buildCsr = true;

        // SoA strides are fixed at maxNodesCapacity (both neighborsBuffer and matrixABuffer).
        cgSolverShader.SetInt("numNodesCapacity", maxNodesCapacity);
        if (buildCsr)
        {
            if (csrBuilderShader == null)
            {
                Debug.LogError("CSR build requires CSRBuilder compute shader (assign or use Editor default path Assets/Scripts/CSRBuilder.compute).");
                return;
            }
            csrBuilderShader.SetInt("numNodesCapacity", maxNodesCapacity);
        }

        // Build all GPU-driven indirect dispatch args from nodeCount in a single kernel.
        // Writes solverIndirectArgsBuffer, reductionCount256/512 buffers, and solverPrefixLenBuffer.
        using (new GpuProfileSection(this, "FluidSim.SolvePressure.Build"))
        {
        nodesPrefixSumsShader.SetBuffer(buildSolverIndirectArgsKernelId, "nodeCount", nodeCount);
        nodesPrefixSumsShader.SetBuffer(buildSolverIndirectArgsKernelId, "solverIndirectArgsBuffer", solverIndirectArgsBuffer);
        nodesPrefixSumsShader.SetBuffer(buildSolverIndirectArgsKernelId, "solverReductionCount256Buffer", solverReductionCount256Buffer);
        nodesPrefixSumsShader.SetBuffer(buildSolverIndirectArgsKernelId, "solverReductionCount512Buffer", solverReductionCount512Buffer);
        nodesPrefixSumsShader.SetBuffer(buildSolverIndirectArgsKernelId, "solverPrefixLenBuffer", solverPrefixLenBuffer);
        GpuProfileDispatchCompute(nodesPrefixSumsShader, buildSolverIndirectArgsKernelId, 1, 1, 1);

        // Bind nodeCountBuffer to all shaders that replaced the numNodes uniform.
        cgSolverShader.SetBuffer(calculateDivergenceKernel, "nodeCountBuffer", nodeCount);
        cgSolverShader.SetBuffer(buildMatrixAKernelId,      "nodeCountBuffer", nodeCount);
        cgSolverShader.SetBuffer(scaleKernelId,             "nodeCountBuffer", nodeCount);
        cgSolverShader.SetBuffer(axpyKernel,                "nodeCountBuffer", nodeCount);
        cgSolverShader.SetBuffer(dotProductKernel,          "nodeCountBuffer", nodeCount);
        cgSolverShader.SetBuffer(applyJacobiKernelId,       "nodeCountBuffer", nodeCount);
        cgSolverShader.SetBuffer(globalReduceSumKernelId,   "reductionCountBuffer", solverReductionCount256Buffer); // default; overridden below as needed
        cgSolverShader.SetBuffer(spmvCsrKernelId,           "nodeCountBuffer", nodeCount);
        cgSolverShader.SetBuffer(copyFloatKernelId,         "nodeCountBuffer", nodeCount);
        cgSolverShader.SetBuffer(axpyAlphaKernelId,         "nodeCountBuffer", nodeCount);
        cgSolverShader.SetBuffer(axpyNegAlphaKernelId,      "nodeCountBuffer", nodeCount);
        cgSolverShader.SetBuffer(scaleAddBetaKernelId,      "nodeCountBuffer", nodeCount);
        if (buildCsr)
        {
            csrBuilderShader.SetBuffer(csrCountNnzKernelId,     "nodeCountBuffer", nodeCount);
            csrBuilderShader.SetBuffer(csrFinalizeRowPtrKernelId,"nodeCountBuffer", nodeCount);
            csrBuilderShader.SetBuffer(csrFillKernelId,         "nodeCountBuffer", nodeCount);
        }

        // --- Step 1: Calculate Divergence (RHS 'b') ---
        cgSolverShader.SetBuffer(calculateDivergenceKernel, "nodesBuffer", nodesBuffer);
        cgSolverShader.SetBuffer(calculateDivergenceKernel, "divergenceBuffer", divergenceBuffer);
        cgSolverShader.SetBuffer(calculateDivergenceKernel, "neighborsBuffer", neighborsBuffer);
        GpuProfileDispatchIndirect(cgSolverShader,calculateDivergenceKernel, solverIndirectArgsBuffer, SolverArgsOffset512);

        // --- Step 1b: Build Matrix A ---
        float deltaTime = useRealTime ? Time.deltaTime : (1 / frameRate);
        cgSolverShader.SetBuffer(buildMatrixAKernelId, "nodesBuffer", nodesBuffer);
        cgSolverShader.SetBuffer(buildMatrixAKernelId, "neighborsBuffer", neighborsBuffer);
        cgSolverShader.SetBuffer(buildMatrixAKernelId, "matrixABuffer", matrixABuffer);
        cgSolverShader.SetFloat("deltaTime", deltaTime);
        GpuProfileDispatchIndirect(cgSolverShader,buildMatrixAKernelId, solverIndirectArgsBuffer, SolverArgsOffset256);

        // --- Step 1c: Build CSR (only when PCG uses CSR matvec, neural leaf inputs, or training export) ---
        if (buildCsr)
        {
            // A. Count nnz per row
            csrBuilderShader.SetBuffer(csrCountNnzKernelId, "neighborsBuffer", neighborsBuffer);
            csrBuilderShader.SetBuffer(csrCountNnzKernelId, "nnzPerNode", nnzPerNode);
            GpuProfileDispatchIndirect(csrBuilderShader,csrCountNnzKernelId, solverIndirectArgsBuffer, SolverArgsOffset256);

            // B. Exclusive prefix sum nnzPerNode -> csrRowPtr (3-pass, all GPU-driven)
            radixSortShader.SetBuffer(radixPrefixSumSolverMainKernelId, "input",               nnzPerNode);
            radixSortShader.SetBuffer(radixPrefixSumSolverMainKernelId, "output",              csrRowPtr);
            radixSortShader.SetBuffer(radixPrefixSumSolverMainKernelId, "aux",                 aux);
            radixSortShader.SetBuffer(radixPrefixSumSolverMainKernelId, "solverPrefixLenBuffer", solverPrefixLenBuffer);
            radixSortShader.SetInt("zeroff", 1);
            GpuProfileDispatchIndirect(radixSortShader,radixPrefixSumSolverMainKernelId, solverIndirectArgsBuffer, SolverArgsOffsetPfx1);

            radixSortShader.SetBuffer(radixPrefixSumSolverAuxKernelId, "input",               aux);
            radixSortShader.SetBuffer(radixPrefixSumSolverAuxKernelId, "output",              aux2);
            radixSortShader.SetBuffer(radixPrefixSumSolverAuxKernelId, "aux",                 auxSmall);
            radixSortShader.SetBuffer(radixPrefixSumSolverAuxKernelId, "solverPrefixLenBuffer", solverPrefixLenBuffer);
            GpuProfileDispatchCompute(radixSortShader, radixPrefixSumSolverAuxKernelId, 1, 1, 1);

            radixSortShader.SetBuffer(radixPrefixFixupSolverKernelId, "input",               csrRowPtr);
            radixSortShader.SetBuffer(radixPrefixFixupSolverKernelId, "aux",                 aux2);
            radixSortShader.SetBuffer(radixPrefixFixupSolverKernelId, "solverPrefixLenBuffer", solverPrefixLenBuffer);
            GpuProfileDispatchIndirect(radixSortShader,radixPrefixFixupSolverKernelId, solverIndirectArgsBuffer, SolverArgsOffsetPfx1);

            csrBuilderShader.SetBuffer(csrFinalizeRowPtrKernelId, "nnzPerNode", nnzPerNode);
            csrBuilderShader.SetBuffer(csrFinalizeRowPtrKernelId, "rowPtrBuffer", csrRowPtr);
            GpuProfileDispatchCompute(csrBuilderShader, csrFinalizeRowPtrKernelId, 1, 1, 1);

            csrBuilderShader.SetBuffer(csrFillKernelId, "neighborsBuffer", neighborsBuffer);
            csrBuilderShader.SetBuffer(csrFillKernelId, "matrixABuffer", matrixABuffer);
            csrBuilderShader.SetBuffer(csrFillKernelId, "rowPtrBuffer", csrRowPtr);
            csrBuilderShader.SetBuffer(csrFillKernelId, "colIndicesBuffer", csrColIndices);
            csrBuilderShader.SetBuffer(csrFillKernelId, "rowIndicesBuffer", csrRowIndices);
            csrBuilderShader.SetBuffer(csrFillKernelId, "valuesBuffer", csrValues);
            GpuProfileDispatchIndirect(csrBuilderShader,csrFillKernelId, solverIndirectArgsBuffer, SolverArgsOffset256);
        }
        }

        if (buildCsr)
            DispatchLeafOnlyGpuInputs(numNodes, maxNnz);

        if (preconditioner == PreconditionerType.Neural)
            LeafOnlyEnsureJacobiInvDiagFromMatrixA(LeafOnlyLastNPadded);

        using (new GpuProfileSection(this, "FluidSim.SolvePressure.Init"))
        {
        // Init Pressure x=0 via GPU Scale(0)
        cgSolverShader.SetBuffer(scaleKernelId, "yBuffer", pressureBuffer);
        cgSolverShader.SetFloat("a", 0.0f);
        GpuProfileDispatchIndirect(cgSolverShader,scaleKernelId, solverIndirectArgsBuffer, SolverArgsOffset512);

        // r = b (GPU copy, no CPU round-trip)
        GpuCopyBuffer(divergenceBuffer, residualBuffer);

        // --- Step 2: Preconditioned Initialization (rho on GPU only) ---
        ApplyPreconditionerInitStoreRhoGpu(residualBuffer, zVectorBuffer, applyJacobiKernelId);

        // p = z (GPU copy)
        GpuCopyBuffer(zVectorBuffer, pBuffer);
        }

        float initialResidual = GpuDotProduct(residualBuffer, residualBuffer);
        if (initialResidual < convergenceThreshold) return;

        laplacianSw.Reset();
        dotProductSw.Reset();
        updateVectorSw.Reset();

        // --- Step 3: PCG Loop (always maxCgIterations; no in-loop convergence / indirect early-out) ---

        void RunPcgLoop()
        {
        for (int k = 0; k < maxCgIterations; k++)
        {
            // 1. Ap = A*p, partial p·Ap → divergenceBuffer (CSR SpMV)
            int matvecKernel = spmvCsrKernelId;
            laplacianSw.Start();
            cgSolverShader.SetBuffer(matvecKernel, "pBuffer", pBuffer);
            cgSolverShader.SetBuffer(matvecKernel, "ApBuffer", ApBuffer);
            cgSolverShader.SetBuffer(matvecKernel, "divergenceBuffer", divergenceBuffer);
            cgSolverShader.SetBuffer(matvecKernel, "csrRowPtr", csrRowPtr);
            cgSolverShader.SetBuffer(matvecKernel, "csrColIndices", csrColIndices);
            cgSolverShader.SetBuffer(matvecKernel, "csrValues", csrValues);
            GpuProfileDispatchIndirect(cgSolverShader, matvecKernel, solverIndirectArgsBuffer, CgIndirectArgsOffsetSpmv);
            laplacianSw.Stop();

            // 2. Sum p·Ap → divergence[0]; alpha = rho / (p·Ap) on GPU
            dotProductSw.Start();
            cgSolverShader.SetBuffer(globalReduceSumKernelId, "divergenceBuffer", divergenceBuffer);
            cgSolverShader.SetBuffer(globalReduceSumKernelId, "reductionCountBuffer", solverReductionCount256Buffer);
            GpuProfileDispatchIndirect(cgSolverShader,globalReduceSumKernelId, solverIndirectArgsBuffer, CgIndirectArgsOffsetSingleGroup);

            cgSolverShader.SetBuffer(computeAlphaKernelId, "divergenceBuffer", divergenceBuffer);
            cgSolverShader.SetBuffer(computeAlphaKernelId, "rhoBuffer", cgRhoBuffer);
            cgSolverShader.SetBuffer(computeAlphaKernelId, "alphaBuffer", cgAlphaBuffer);
            GpuProfileDispatchCompute(cgSolverShader,computeAlphaKernelId, 1, 1, 1);
            dotProductSw.Stop();

            // 3. x += alpha * p ; r -= alpha * Ap
            updateVectorSw.Start();
            cgSolverShader.SetBuffer(axpyAlphaKernelId, "xBuffer", pBuffer);
            cgSolverShader.SetBuffer(axpyAlphaKernelId, "yBuffer", pressureBuffer);
            cgSolverShader.SetBuffer(axpyAlphaKernelId, "alphaBuffer", cgAlphaBuffer);
            GpuProfileDispatchIndirect(cgSolverShader,axpyAlphaKernelId, solverIndirectArgsBuffer, CgIndirectArgsOffsetVec512);

            cgSolverShader.SetBuffer(axpyNegAlphaKernelId, "xBuffer", ApBuffer);
            cgSolverShader.SetBuffer(axpyNegAlphaKernelId, "yBuffer", residualBuffer);
            cgSolverShader.SetBuffer(axpyNegAlphaKernelId, "alphaBuffer", cgAlphaBuffer);
            GpuProfileDispatchIndirect(cgSolverShader,axpyNegAlphaKernelId, solverIndirectArgsBuffer, CgIndirectArgsOffsetVec512);
            updateVectorSw.Stop();

            // 4. z = M^-1 r, rho_new = r·z → divergence[0]; beta and rho on GPU
            dotProductSw.Start();
            ApplyPreconditionerPcgIterationGpu(residualBuffer, zVectorBuffer, applyJacobiKernelId);

            cgSolverShader.SetBuffer(computeBetaKernelId, "divergenceBuffer", divergenceBuffer);
            cgSolverShader.SetBuffer(computeBetaKernelId, "rhoBuffer", cgRhoBuffer);
            cgSolverShader.SetBuffer(computeBetaKernelId, "betaBuffer", cgBetaBuffer);
            GpuProfileDispatchCompute(cgSolverShader,computeBetaKernelId, 1, 1, 1);
            dotProductSw.Stop();

            // 5. p = beta * p + z
            updateVectorSw.Start();
            cgSolverShader.SetBuffer(scaleAddBetaKernelId, "pBuffer", pBuffer);
            cgSolverShader.SetBuffer(scaleAddBetaKernelId, "yBuffer", zVectorBuffer);
            cgSolverShader.SetBuffer(scaleAddBetaKernelId, "betaBuffer", cgBetaBuffer);
            GpuProfileDispatchIndirect(cgSolverShader,scaleAddBetaKernelId, solverIndirectArgsBuffer, CgIndirectArgsOffsetVec512);
            updateVectorSw.Stop();
        }
        }

        if (emitGpuDebugLabels)
        {
            using (new GpuProfileSection(this, "FluidSim.SolvePressure.PCG"))
                RunPcgLoop();
        }
        else
            RunPcgLoop();

        cgSolveFrameCount++;
        cumulativeCgIterations += maxCgIterations;
        averageCgIterations = (float)(cumulativeCgIterations / Math.Max(1, cgSolveFrameCount));
        lastCgIterations = maxCgIterations;

        if (enablePerFrameDebugTimingLog)
        {
            float rFinSq = GpuDotProduct(residualBuffer, residualBuffer);
            float rel = rFinSq / Mathf.Max(initialResidual, 1e-30f);
            Debug.Log($"[PCG] ||r||²={rFinSq:E6}  rel(||r||²/||r0||²)={rel:E6}  iters={maxCgIterations}");
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
        using (new GpuProfileSection(this, "FluidSim.SolvePressure.ApplyPressure"))
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
        GpuProfileDispatchIndirect(nodesShader,applyPressureGradientKernel, solverIndirectArgsBuffer, SolverArgsOffset256);

        ComputeBuffer swap = nodesBuffer;
        nodesBuffer = tempNodesBuffer;
        tempNodesBuffer = swap;
    }

    private void GpuCopyBufferIndirect(ComputeBuffer source, ComputeBuffer destination)
    {
        cgSolverShader.SetBuffer(copyFloatKernelId, "xBuffer", source);
        cgSolverShader.SetBuffer(copyFloatKernelId, "yBuffer", destination);
        GpuProfileDispatchIndirect(cgSolverShader,copyFloatKernelId, solverIndirectArgsBuffer, CgIndirectArgsOffsetVec512);
    }

    private void GpuDotProductReduceNoReadbackIndirect(ComputeBuffer bufferA, ComputeBuffer bufferB, int _unused)
    {
        cgSolverShader.SetBuffer(dotProductKernel, "xBuffer", bufferA);
        cgSolverShader.SetBuffer(dotProductKernel, "yBuffer", bufferB);
        cgSolverShader.SetBuffer(dotProductKernel, "divergenceBuffer", divergenceBuffer);
        GpuProfileDispatchIndirect(cgSolverShader,dotProductKernel, solverIndirectArgsBuffer, CgIndirectArgsOffsetVec512);
        cgSolverShader.SetBuffer(globalReduceSumKernelId, "divergenceBuffer", divergenceBuffer);
        cgSolverShader.SetBuffer(globalReduceSumKernelId, "reductionCountBuffer", solverReductionCount512Buffer);
        GpuProfileDispatchIndirect(cgSolverShader,globalReduceSumKernelId, solverIndirectArgsBuffer, CgIndirectArgsOffsetSingleGroup);
    }

    // Legacy name kept for FluidPreconditioner.cs callers.
    private void CopyBuffer(ComputeBuffer source, ComputeBuffer destination) => GpuCopyBuffer(source, destination);

    // GPU-only buffer copy: fully indirect, no CPU round-trip.
    private void GpuCopyBuffer(ComputeBuffer source, ComputeBuffer destination)
    {
        cgSolverShader.SetBuffer(copyFloatKernelId, "xBuffer", source);
        cgSolverShader.SetBuffer(copyFloatKernelId, "yBuffer", destination);
        GpuProfileDispatchIndirect(cgSolverShader,copyFloatKernelId, solverIndirectArgsBuffer, CgIndirectArgsOffsetVec512);
    }

    private void GpuDotProductReduceNoReadback(ComputeBuffer bufferA, ComputeBuffer bufferB)
    {
        cgSolverShader.SetBuffer(dotProductKernel, "xBuffer", bufferA);
        cgSolverShader.SetBuffer(dotProductKernel, "yBuffer", bufferB);
        cgSolverShader.SetBuffer(dotProductKernel, "divergenceBuffer", divergenceBuffer);
        GpuProfileDispatchIndirect(cgSolverShader,dotProductKernel, solverIndirectArgsBuffer, CgIndirectArgsOffsetVec512);
        cgSolverShader.SetBuffer(globalReduceSumKernelId, "divergenceBuffer", divergenceBuffer);
        cgSolverShader.SetBuffer(globalReduceSumKernelId, "reductionCountBuffer", solverReductionCount512Buffer);
        GpuProfileDispatchIndirect(cgSolverShader,globalReduceSumKernelId, solverIndirectArgsBuffer, CgIndirectArgsOffsetSingleGroup);
    }

    private void DispatchStoreRhoFromDot()
    {
        cgSolverShader.SetBuffer(storeRhoFromDotKernelId, "divergenceBuffer", divergenceBuffer);
        cgSolverShader.SetBuffer(storeRhoFromDotKernelId, "rhoBuffer", cgRhoBuffer);
        GpuProfileDispatchCompute(cgSolverShader,storeRhoFromDotKernelId, 1, 1, 1);
    }

    private void DispatchComputeAlpha()
    {
        cgSolverShader.SetBuffer(computeAlphaKernelId, "divergenceBuffer", divergenceBuffer);
        cgSolverShader.SetBuffer(computeAlphaKernelId, "rhoBuffer", cgRhoBuffer);
        cgSolverShader.SetBuffer(computeAlphaKernelId, "alphaBuffer", cgAlphaBuffer);
        GpuProfileDispatchCompute(cgSolverShader,computeAlphaKernelId, 1, 1, 1);
    }

    private void DispatchComputeBeta()
    {
        cgSolverShader.SetBuffer(computeBetaKernelId, "divergenceBuffer", divergenceBuffer);
        cgSolverShader.SetBuffer(computeBetaKernelId, "rhoBuffer", cgRhoBuffer);
        cgSolverShader.SetBuffer(computeBetaKernelId, "betaBuffer", cgBetaBuffer);
        GpuProfileDispatchCompute(cgSolverShader,computeBetaKernelId, 1, 1, 1);
    }

    private float GpuDotProduct(ComputeBuffer bufferA, ComputeBuffer bufferB)
    {
        if (cgSolverShader == null) return 0.0f;

        cgSolverShader.SetBuffer(dotProductKernel, "xBuffer", bufferA);
        cgSolverShader.SetBuffer(dotProductKernel, "yBuffer", bufferB);
        cgSolverShader.SetBuffer(dotProductKernel, "divergenceBuffer", divergenceBuffer);
        GpuProfileDispatchIndirect(cgSolverShader,dotProductKernel, solverIndirectArgsBuffer, CgIndirectArgsOffsetVec512);

        cgSolverShader.SetBuffer(globalReduceSumKernelId, "divergenceBuffer", divergenceBuffer);
        cgSolverShader.SetBuffer(globalReduceSumKernelId, "reductionCountBuffer", solverReductionCount512Buffer);
        GpuProfileDispatchIndirect(cgSolverShader,globalReduceSumKernelId, solverIndirectArgsBuffer, CgIndirectArgsOffsetSingleGroup);

        divergenceBuffer.GetData(reductionResult, 0, 0, 1);
        return reductionResult[0];
    }

    private void UpdateVector(ComputeBuffer yBuffer, ComputeBuffer xBuffer, float a)
    {
        if (cgSolverShader == null) return;
        cgSolverShader.SetBuffer(axpyKernel, "xBuffer", xBuffer);
        cgSolverShader.SetBuffer(axpyKernel, "yBuffer", yBuffer);
        cgSolverShader.SetFloat("a", a);
        GpuProfileDispatchIndirect(cgSolverShader,axpyKernel, solverIndirectArgsBuffer, CgIndirectArgsOffsetVec512);
    }

    private void UpdateVector(ComputeBuffer pBuf, ComputeBuffer rBuffer, float rCoeff, float pCoeff)
    {
        if (cgSolverShader == null) return;
        cgSolverShader.SetBuffer(scaleKernelId, "yBuffer", pBuf);
        cgSolverShader.SetFloat("a", pCoeff);
        GpuProfileDispatchIndirect(cgSolverShader,scaleKernelId, solverIndirectArgsBuffer, CgIndirectArgsOffsetVec512);

        cgSolverShader.SetBuffer(axpyKernel, "xBuffer", rBuffer);
        cgSolverShader.SetBuffer(axpyKernel, "yBuffer", pBuf);
        cgSolverShader.SetFloat("a", rCoeff);
        GpuProfileDispatchIndirect(cgSolverShader,axpyKernel, solverIndirectArgsBuffer, CgIndirectArgsOffsetVec512);
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
        GpuProfileDispatchIndirect(nodesShader,applyExternalForcesKernel, dispatchArgsBuffer, 0);
    }
}
