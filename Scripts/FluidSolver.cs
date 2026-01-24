using UnityEngine;
using System;

// CG Solver component of FluidSimulator
public partial class FluidSimulator : MonoBehaviour
{
    // Timing variables for CG loop
    private System.Diagnostics.Stopwatch laplacianSw = new System.Diagnostics.Stopwatch();
    private System.Diagnostics.Stopwatch dotProductSw = new System.Diagnostics.Stopwatch();
    private System.Diagnostics.Stopwatch updateVectorSw = new System.Diagnostics.Stopwatch();

    private void SolvePressure()
    {
        if (cgSolverShader == null)
        {
            Debug.LogError("CGSolver compute shader is not assigned.");
            return;
        }

        // --- Kernel Lookups ---
        int calculateDivergenceKernel = cgSolverShader.FindKernel("CalculateDivergence");
        // CHANGED: Use new matrix kernel
        int buildMatrixAKernel = cgSolverShader.FindKernel("BuildMatrixA");
        int applyMatrixAndDotKernel = cgSolverShader.FindKernel("ApplyMatrixAndDot");
        
        int axpyKernel = cgSolverShader.FindKernel("Axpy");
        int dotProductKernel = cgSolverShader.FindKernel("DotProduct");
        
        // Preconditioner Kernels
        int precomputeIndicesKernel = cgSolverShader.FindKernel("PrecomputeIndices");
        int applySparseGTKernel = cgSolverShader.FindKernel("ApplySparseGT"); 
        int applySparseGAndDotKernel = cgSolverShader.FindKernel("ApplySparseGAndDot");
        int clearBufferFloatKernel = cgSolverShader.FindKernel("ClearBufferFloat");
        int applyJacobiKernel = cgSolverShader.FindKernel("ApplyJacobi");
        
        if (calculateDivergenceKernel < 0 || buildMatrixAKernel < 0 || applyMatrixAndDotKernel < 0)
        {
            Debug.LogError("CG solver kernels (BuildMatrixA/ApplyMatrixAndDot) not found.");
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
        
        // --- NEW: Alloc Matrix A ---
        if (matrixABuffer == null || matrixABuffer.count < requiredSize * 25) {
            matrixABuffer?.Release();
            matrixABuffer = new ComputeBuffer(requiredSize * 25, 4); // stride 4 (float)
        }

        // Keep existing buffers for preconditioners
        if (scatterIndicesBuffer == null || scatterIndicesBuffer.count < requiredSize * 24) {
            scatterIndicesBuffer?.Release(); scatterIndicesBuffer = new ComputeBuffer(requiredSize * 24, 4);
        }

        // --- Step 1: Init (r = b) ---
        // 1a. Calculate Divergence (RHS 'b')
        cgSolverShader.SetBuffer(calculateDivergenceKernel, "nodesBuffer", nodesBuffer);
        cgSolverShader.SetBuffer(calculateDivergenceKernel, "divergenceBuffer", divergenceBuffer);
        cgSolverShader.SetBuffer(calculateDivergenceKernel, "neighborsBuffer", neighborsBuffer);
        cgSolverShader.SetInt("numNodes", numNodes);
        Dispatch(calculateDivergenceKernel, numNodes);

        // 1b. Build Matrix A (LHS 'A')
        // This bakes geometry and deltaTime into matrixABuffer
        float deltaTime = useRealTime ? Time.deltaTime : (1 / frameRate);
        cgSolverShader.SetBuffer(buildMatrixAKernel, "nodesBuffer", nodesBuffer);
        cgSolverShader.SetBuffer(buildMatrixAKernel, "neighborsBuffer", neighborsBuffer);
        cgSolverShader.SetBuffer(buildMatrixAKernel, "matrixABuffer", matrixABuffer);
        cgSolverShader.SetInt("numNodes", numNodes);
        cgSolverShader.SetFloat("deltaTime", deltaTime); // Passed here, so we don't need it in the loop
        
        int groupsA = Mathf.CeilToInt(numNodes / 256.0f);
        cgSolverShader.Dispatch(buildMatrixAKernel, groupsA, 1, 1);

        // Preconditioner setup (Neural)
        if (preconditioner == PreconditionerType.Neural && preconditionerShader != null) {
            RunNeuralPreconditioner(); 
            cgSolverShader.SetBuffer(precomputeIndicesKernel, "neighborsBuffer", neighborsBuffer);
            cgSolverShader.SetBuffer(precomputeIndicesKernel, "scatterIndicesBuffer", scatterIndicesBuffer);
            cgSolverShader.SetInt("numNodes", numNodes);
            Dispatch(precomputeIndicesKernel, numNodes);
        }

        // Init Pressure x=0
        float[] zeros = new float[numNodes];
        pressureBuffer.SetData(zeros);

        // r = b
        CopyBuffer(divergenceBuffer, residualBuffer);

        // --- Step 2: Preconditioned Initialization ---
        // z = M^-1 * r AND rho = r . z (Fused)
        float rho = ApplyPreconditionerAndDot(
            residualBuffer, zVectorBuffer, 
            applySparseGTKernel, applySparseGAndDotKernel, applySparseGAndDotKernel, 
            clearBufferFloatKernel, applyJacobiKernel
        );

        CopyBuffer(zVectorBuffer, pBuffer);

        float initialResidual = GpuDotProduct(residualBuffer, residualBuffer);
        if (initialResidual < convergenceThreshold) return;

        laplacianSw.Reset();
        dotProductSw.Reset();
        updateVectorSw.Reset();

        // --- Step 3: PCG Loop ---
        int totalIterations = 0;
        
        for (int k = 0; k < maxCgIterations; k++)
        {
            // 1. FUSED: Ap = A * p AND p · Ap
            // CHANGED: Use ApplyMatrixAndDot with matrixABuffer
            laplacianSw.Start();
            cgSolverShader.SetBuffer(applyMatrixAndDotKernel, "pBuffer", pBuffer);
            cgSolverShader.SetBuffer(applyMatrixAndDotKernel, "ApBuffer", ApBuffer);
            cgSolverShader.SetBuffer(applyMatrixAndDotKernel, "matrixABuffer", matrixABuffer); // New Input
            cgSolverShader.SetBuffer(applyMatrixAndDotKernel, "neighborsBuffer", neighborsBuffer);
            cgSolverShader.SetBuffer(applyMatrixAndDotKernel, "divergenceBuffer", divergenceBuffer); // Output for reduction
            cgSolverShader.SetInt("numNodes", numNodes);
            // Note: deltaTime is NOT passed here anymore; it's baked into matrixA
            
            cgSolverShader.Dispatch(applyMatrixAndDotKernel, groupsA, 1, 1);
            laplacianSw.Stop();

            // 2. CPU Side Reduction
            dotProductSw.Start();
            float[] result = new float[groupsA];
            divergenceBuffer.GetData(result); 
            float p_dot_Ap = 0.0f;
            for (int i = 0; i < groupsA; i++) p_dot_Ap += result[i];
            dotProductSw.Stop();
            
            if (p_dot_Ap <= 1e-12f) break; // Singular or non-positive-definite
            
            float alpha = rho / p_dot_Ap;
            
            // 3. x = x + alpha * p
            updateVectorSw.Start();
            UpdateVector(pressureBuffer, pBuffer, alpha);
            updateVectorSw.Stop();

            // 4. r = r - alpha * Ap
            updateVectorSw.Start();
            UpdateVector(residualBuffer, ApBuffer, -alpha);
            updateVectorSw.Stop();

            // Check Convergence
            if (k % 5 == 0)
            {
                dotProductSw.Start();
                float r_dot_r = GpuDotProduct(residualBuffer, residualBuffer);
                dotProductSw.Stop();
                if (r_dot_r < convergenceThreshold || r_dot_r > 10.0f * initialResidual) {
                    totalIterations = k + 1;
                    break;
                }
            }

            // --- PRECONDITIONER STEP ---
            // 5. z_new = M^-1 * r_new
            dotProductSw.Start();
            float rho_new = ApplyPreconditionerAndDot(
                residualBuffer, zVectorBuffer, 
                applySparseGTKernel, applySparseGAndDotKernel, applySparseGAndDotKernel, 
                clearBufferFloatKernel, applyJacobiKernel
            );
            dotProductSw.Stop();

            // 7. beta
            float beta = rho_new / rho;

            // 8. p = z_new + beta * p
            updateVectorSw.Start();
            UpdateVector(pBuffer, zVectorBuffer, 1.0f, beta);
            updateVectorSw.Stop();

            rho = rho_new;
            totalIterations = k + 1;
        }
        
        cgSolveFrameCount++;
        cumulativeCgIterations += totalIterations;
        averageCgIterations = (float)(cumulativeCgIterations / Math.Max(1, cgSolveFrameCount));
        lastCgIterations = totalIterations;
        
        // Save training data
        if (recorder != null && recorder.isRecording)
        {
            Vector3 simBoundsMin = simulationBounds != null ? simulationBounds.bounds.min : Vector3.zero;
            Vector3 simBoundsMax = simulationBounds != null ? simulationBounds.bounds.max : Vector3.zero;
            Vector3 fluidBoundsMin = fluidInitialBounds != null ? fluidInitialBounds.bounds.min : Vector3.zero;
            Vector3 fluidBoundsMax = fluidInitialBounds != null ? fluidInitialBounds.bounds.max : Vector3.zero;
            
            recorder.SaveFrame(
                nodesBuffer,
                neighborsBuffer,
                divergenceBuffer,
                pressureBuffer,
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

        applyPressureGradientKernel = nodesShader.FindKernel("ApplyPressureGradient");
        
        // Set the necessary buffers and parameters
        nodesShader.SetBuffer(applyPressureGradientKernel, "nodesBuffer", nodesBuffer);
        nodesShader.SetBuffer(applyPressureGradientKernel, "tempNodesBuffer", tempNodesBuffer);
        nodesShader.SetBuffer(applyPressureGradientKernel, "neighborsBuffer", neighborsBuffer);
        nodesShader.SetBuffer(applyPressureGradientKernel, "pressureBuffer", pressureBuffer); // The result from the CG solve!
        
        nodesShader.SetInt("numNodes", numNodes);
        float deltaTime = useRealTime ? Time.deltaTime : (1 / frameRate); // Use the same deltaTime as in advection
        nodesShader.SetFloat("deltaTime", deltaTime);
        nodesShader.SetFloat("maxDetailCellSize", maxDetailCellSize);
        nodesShader.SetInt("minLayer", minLayer);

        // Dispatch the kernel to update velocities
        int threadGroups = Mathf.CeilToInt(numNodes / 256.0f);
        nodesShader.Dispatch(applyPressureGradientKernel, threadGroups, 1, 1);

        // SWAP the C# references
        ComputeBuffer swap = nodesBuffer;
        nodesBuffer = tempNodesBuffer;
        tempNodesBuffer = swap;
    }
    private void Dispatch(int kernel, int count) 
    {
        int threadGroups = Mathf.CeilToInt(count / 512.0f);
        cgSolverShader.Dispatch(kernel, threadGroups, 1, 1);
    }
    private void CopyBuffer(ComputeBuffer source, ComputeBuffer destination)
    {
        if (source.count != destination.count) return;
        
        float[] data = new float[source.count];
        source.GetData(data);
        destination.SetData(data);
    }
    private float GpuDotProduct(ComputeBuffer bufferA, ComputeBuffer bufferB)
    {
        if (cgSolverShader == null) return 0.0f;

        int dotProductKernel = cgSolverShader.FindKernel("DotProduct");
        int reduceKernel = cgSolverShader.FindKernel("GlobalReduceSum");
        
        // 1. Run Standard Dot Product (Partial Sums)
        cgSolverShader.SetBuffer(dotProductKernel, "xBuffer", bufferA);
        cgSolverShader.SetBuffer(dotProductKernel, "yBuffer", bufferB);
        cgSolverShader.SetBuffer(dotProductKernel, "divergenceBuffer", divergenceBuffer);
        cgSolverShader.SetInt("numNodes", numNodes);
        
        int threadGroups = Mathf.CeilToInt(numNodes / 512.0f);
        cgSolverShader.Dispatch(dotProductKernel, threadGroups, 1, 1);
        
        // 2. Run Final Reduction on GPU (Sum partials to index 0)
        cgSolverShader.SetBuffer(reduceKernel, "divergenceBuffer", divergenceBuffer);
        cgSolverShader.SetInt("reductionCount", threadGroups);
        cgSolverShader.Dispatch(reduceKernel, 1, 1, 1);

        // 3. Read back ONLY the single float result
        // This is still blocking, but minimal data transfer
        divergenceBuffer.GetData(reductionResult, 0, 0, 1);
        
        return reductionResult[0];
    }
    private void UpdateVector(ComputeBuffer yBuffer, ComputeBuffer xBuffer, float a)
    {
        if (cgSolverShader == null) return;

        int axpyKernel = cgSolverShader.FindKernel("Axpy");
        
        cgSolverShader.SetBuffer(axpyKernel, "xBuffer", xBuffer);
        cgSolverShader.SetBuffer(axpyKernel, "yBuffer", yBuffer);
        cgSolverShader.SetFloat("a", a);
        cgSolverShader.SetInt("numNodes", numNodes);
        
        Dispatch(axpyKernel, numNodes);
    }
    private void UpdateVector(ComputeBuffer pBuffer, ComputeBuffer rBuffer, float rCoeff, float pCoeff)
    {
        if (cgSolverShader == null) return;

        // Find the new Scale kernel (and the existing Axpy kernel)
        int scaleKernel = cgSolverShader.FindKernel("Scale");
        int axpyKernel = cgSolverShader.FindKernel("Axpy");
        
        // --- CORRECTED LOGIC ---

        // First, scale p by beta: p_new = beta * p_old
        cgSolverShader.SetBuffer(scaleKernel, "yBuffer", pBuffer);
        cgSolverShader.SetFloat("a", pCoeff); // pCoeff is beta
        cgSolverShader.SetInt("numNodes", numNodes);
        Dispatch(scaleKernel, numNodes);
        
        // Then, add r: p_new = r_new + p_new (which is now beta * p_old)
        cgSolverShader.SetBuffer(axpyKernel, "xBuffer", rBuffer);
        cgSolverShader.SetBuffer(axpyKernel, "yBuffer", pBuffer);
        cgSolverShader.SetFloat("a", rCoeff); // rCoeff is 1.0
        cgSolverShader.SetInt("numNodes", numNodes);
        Dispatch(axpyKernel, numNodes);
    }
    private void ApplyExternalForces()
    {
        if (nodesShader == null) return;

        applyExternalForcesKernel = nodesShader.FindKernel("ApplyExternalForces");
        if (applyExternalForcesKernel < 0)
        {
            Debug.LogError("ApplyExternalForces kernel not found in Nodes.compute shader.");
            return;
        }

        nodesShader.SetBuffer(applyExternalForcesKernel, "nodesBuffer", nodesBuffer);
        nodesShader.SetInt("numNodes", numNodes);
        nodesShader.SetFloat("gravity", gravity);
        float deltaTime = useRealTime ? Time.deltaTime : (1 / frameRate);
        nodesShader.SetFloat("deltaTime", deltaTime);
        int threadGroups = Mathf.CeilToInt(numNodes / 512.0f);
        nodesShader.Dispatch(applyExternalForcesKernel, threadGroups, 1, 1);
    }
}
