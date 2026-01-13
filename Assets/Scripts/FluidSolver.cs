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
            Debug.LogError("CGSolver compute shader is not assigned. Please assign `cgSolverShader` in the inspector.");
            return;
        }

        // --- Kernel Lookups ---
        int calculateDivergenceKernel = cgSolverShader.FindKernel("CalculateDivergence");
        int applyLaplacianAndDotKernel = cgSolverShader.FindKernel("ApplyLaplacianAndDot");
        int axpyKernel = cgSolverShader.FindKernel("Axpy");
        int dotProductKernel = cgSolverShader.FindKernel("DotProduct");
        // Preconditioner Kernels from CGSolver.compute
        int precomputeIndicesKernel = cgSolverShader.FindKernel("PrecomputeIndices");
        int applySparseGTKernel = cgSolverShader.FindKernel("ApplySparseGT"); 
        int applySparseGAndDotKernel = cgSolverShader.FindKernel("ApplySparseGAndDot");
        int clearBufferFloatKernel = cgSolverShader.FindKernel("ClearBufferFloat");
        int applyJacobiKernel = cgSolverShader.FindKernel("ApplyJacobi");
        int computeDiagonalKernel = cgSolverShader.FindKernel("ComputeDiagonal");
        
        if (calculateDivergenceKernel < 0 || applyLaplacianAndDotKernel < 0 || axpyKernel < 0 || dotProductKernel < 0)
        {
            Debug.LogError("One or more CG solver kernels not found. Check CGSolver.compute shader compilation.");
            return;
        }

        // --- Buffer Init (Grow-only) ---
        int requiredSize = Mathf.NextPowerOfTwo(Mathf.Max(numNodes, 512));
        
        if (divergenceBuffer == null || divergenceBuffer.count < requiredSize)
        {
            divergenceBuffer?.Release(); 
            divergenceBuffer = new ComputeBuffer(requiredSize, sizeof(float));
        }
        if (residualBuffer == null || residualBuffer.count < requiredSize)
        {
            residualBuffer?.Release(); 
            residualBuffer = new ComputeBuffer(requiredSize, sizeof(float));
        }
        if (pBuffer == null || pBuffer.count < requiredSize)
        {
            pBuffer?.Release(); 
            pBuffer = new ComputeBuffer(requiredSize, sizeof(float));
        }
        if (ApBuffer == null || ApBuffer.count < requiredSize)
        {
            ApBuffer?.Release(); 
            ApBuffer = new ComputeBuffer(requiredSize, sizeof(float));
        }
        if (pressureBuffer == null || pressureBuffer.count < requiredSize)
        {
            pressureBuffer?.Release(); 
            pressureBuffer = new ComputeBuffer(requiredSize, sizeof(float));
        }
        if (zVectorBuffer == null || zVectorBuffer.count < requiredSize)
        {
            zVectorBuffer?.Release(); 
            zVectorBuffer = new ComputeBuffer(requiredSize, sizeof(float));
        }
        if (scatterIndicesBuffer == null || scatterIndicesBuffer.count < requiredSize * 24)
        {
            scatterIndicesBuffer?.Release();
            scatterIndicesBuffer = new ComputeBuffer(requiredSize * 24, 4); // SoA: stride is 4 bytes (uint)
        }
        // Always allocate diagonalBuffer (needed for Neural input OR Jacobi)
        if (diagonalBuffer == null || diagonalBuffer.count < requiredSize)
        {
            diagonalBuffer?.Release();
            diagonalBuffer = new ComputeBuffer(requiredSize, sizeof(float));
        }

        // --- Step 1: Init (r = b) ---
        cgSolverShader.SetBuffer(calculateDivergenceKernel, "nodesBuffer", nodesBuffer);
        cgSolverShader.SetBuffer(calculateDivergenceKernel, "divergenceBuffer", divergenceBuffer);
        cgSolverShader.SetBuffer(calculateDivergenceKernel, "neighborsBuffer", neighborsBuffer);
        cgSolverShader.SetInt("numNodes", numNodes);
        Dispatch(calculateDivergenceKernel, numNodes);

        // Compute Diagonal EARLY (Before Neural Inference or Jacobi)
        // This is needed for both Neural (as input feature) and Jacobi (as preconditioner)
        if (computeDiagonalKernel >= 0)
        {
            cgSolverShader.SetBuffer(computeDiagonalKernel, "nodesBuffer", nodesBuffer);
            cgSolverShader.SetBuffer(computeDiagonalKernel, "neighborsBuffer", neighborsBuffer);
            cgSolverShader.SetBuffer(computeDiagonalKernel, "diagonalBuffer", diagonalBuffer);
            cgSolverShader.SetInt("numNodes", numNodes);

            // CHANGED: Use 256.0f divisor to match [numthreads(256, 1, 1)]
            int groups = Mathf.CeilToInt(numNodes / 256.0f); 
            cgSolverShader.Dispatch(computeDiagonalKernel, groups, 1, 1);
        }

        // Generate Matrix G (Neural Inference) or Use Diagonal (Jacobi)
        if (preconditioner == PreconditionerType.Neural && preconditionerShader != null)
        {
            RunNeuralPreconditioner(); // Fills matrixGBuffer (now has access to ready-made diagonalBuffer)
            
            // --- Precompute Optimization ---
            cgSolverShader.SetBuffer(precomputeIndicesKernel, "neighborsBuffer", neighborsBuffer);
            cgSolverShader.SetBuffer(precomputeIndicesKernel, "scatterIndicesBuffer", scatterIndicesBuffer);
            cgSolverShader.SetInt("numNodes", numNodes);
            Dispatch(precomputeIndicesKernel, numNodes);
        }
        // Note: Jacobi preconditioner uses the diagonalBuffer computed above, no additional dispatch needed

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

        // p = z
        CopyBuffer(zVectorBuffer, pBuffer);

        float initialResidual = GpuDotProduct(residualBuffer, residualBuffer); // For convergence check only
        if (initialResidual < convergenceThreshold)
        {
            //totalSolveSw.Stop(); // This is no longer needed after removing totalSolveSw
            return;
        }

        // Reset CG loop timers
        laplacianSw.Reset();
        dotProductSw.Reset();
        updateVectorSw.Reset();

        // --- Step 3: PCG Loop ---
        int totalIterations = 0;
        
        for (int k = 0; k < maxCgIterations; k++)
        {
            // 1. FUSED: Ap = A * p AND p · Ap (in single kernel)
            laplacianSw.Start();
            cgSolverShader.SetBuffer(applyLaplacianAndDotKernel, "nodesBuffer", nodesBuffer);
            cgSolverShader.SetBuffer(applyLaplacianAndDotKernel, "neighborsBuffer", neighborsBuffer);
            cgSolverShader.SetBuffer(applyLaplacianAndDotKernel, "pBuffer", pBuffer);
            cgSolverShader.SetBuffer(applyLaplacianAndDotKernel, "ApBuffer", ApBuffer);
            cgSolverShader.SetBuffer(applyLaplacianAndDotKernel, "divergenceBuffer", divergenceBuffer); // For reduction output
            cgSolverShader.SetFloat("deltaTime", (1 / frameRate));
            cgSolverShader.SetInt("numNodes", numNodes);
            int groups = Mathf.CeilToInt(numNodes / 256.0f);
            cgSolverShader.Dispatch(applyLaplacianAndDotKernel, groups, 1, 1);
            laplacianSw.Stop();

            // 2. CPU Side Reduction (same as GpuDotProduct helper)
            dotProductSw.Start();
            float[] result = new float[groups];
            divergenceBuffer.GetData(result); // Only read back small buffer
            float p_dot_Ap = 0.0f;
            for (int i = 0; i < groups; i++) p_dot_Ap += result[i];
            dotProductSw.Stop();
            
            if (Mathf.Abs(p_dot_Ap) < 1e-12f)
            {
                Debug.LogError($"PCG Solver: Matrix is singular or ill-conditioned! p_dot_Ap = {p_dot_Ap:E6}");
                break;
            }
            
            if (p_dot_Ap <= 0.0f)
            {
                Debug.LogError($"PCG Solver: Matrix is not positive definite! p_dot_Ap = {p_dot_Ap:E6}");
                break;
            }
            
            float alpha = rho / p_dot_Ap;
            
            if (Mathf.Abs(alpha) > 1e10f)
            {
                Debug.LogError($"PCG Solver: Alpha is too large! alpha = {alpha:E6}, rho = {rho:E6}, p_dot_Ap = {p_dot_Ap:E6}");
                break;
            }

            // 3. x = x + alpha * p
            updateVectorSw.Start();
            UpdateVector(pressureBuffer, pBuffer, alpha);
            updateVectorSw.Stop();

            // 4. r = r - alpha * Ap
            updateVectorSw.Start();
            UpdateVector(residualBuffer, ApBuffer, -alpha);
            updateVectorSw.Stop();

            // Check Convergence (using true residual r.r)
            // Only check every 5 iterations to reduce CPU-GPU transfer
            if (k % 5 == 0)
            {
                dotProductSw.Start();
                float r_dot_r = GpuDotProduct(residualBuffer, residualBuffer);
                dotProductSw.Stop();
                
                if (r_dot_r > 10.0f * initialResidual)
                {
                    totalIterations = k + 1;
                    break;
                }
                
                if (r_dot_r < convergenceThreshold)
                {
                    totalIterations = k + 1;
                    break;
                }
            }

            // --- PRECONDITIONER STEP ---
            // 5. z_new = M^-1 * r_new AND rho_new = r_new . z_new (Fused)
            dotProductSw.Start();
            float rho_new = ApplyPreconditionerAndDot(
                residualBuffer, zVectorBuffer, 
                applySparseGTKernel, applySparseGAndDotKernel, applySparseGAndDotKernel, 
                clearBufferFloatKernel, applyJacobiKernel
            );
            dotProductSw.Stop();

            // 7. beta = rho_new / rho
            float beta = rho_new / rho;

            // 8. p = z_new + beta * p
            updateVectorSw.Start();
            UpdateVector(pBuffer, zVectorBuffer, 1.0f, beta);
            updateVectorSw.Stop();

            rho = rho_new;
            totalIterations = k + 1;
        }
        
        // Update running statistics
        cgSolveFrameCount++;
        cumulativeCgIterations += totalIterations;
        averageCgIterations = (float)(cumulativeCgIterations / Math.Max(1, cgSolveFrameCount));
        lastCgIterations = totalIterations;
        
        // Summary report

        // Save training data
        if (recorder != null && recorder.isRecording)
        {
            Vector3 simBoundsMin = simulationBounds != null ? simulationBounds.bounds.min : Vector3.zero;
            Vector3 simBoundsMax = simulationBounds != null ? simulationBounds.bounds.max : Vector3.zero;
            Vector3 fluidBoundsMin = fluidInitialBounds != null ? fluidInitialBounds.bounds.min : Vector3.zero;
            Vector3 fluidBoundsMax = fluidInitialBounds != null ? fluidInitialBounds.bounds.max : Vector3.zero;
            
            recorder.SaveFrame(nodesBuffer, neighborsBuffer, divergenceBuffer, pressureBuffer, numNodes,
                minLayer, maxLayer, gravity, numParticles, maxCgIterations, convergenceThreshold, frameRate,
                simBoundsMin, simBoundsMax, fluidBoundsMin, fluidBoundsMax);
        }

        // --- Step 4: Apply pressure to velocities ---
        ApplyPressureGradient();
        
        // Final timing summary
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
        nodesShader.SetFloat("deltaTime", (1 / frameRate)); // Use the same deltaTime as in advection
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
        nodesShader.SetFloat("deltaTime", (1 / frameRate));
        int threadGroups = Mathf.CeilToInt(numNodes / 512.0f);
        nodesShader.Dispatch(applyExternalForcesKernel, threadGroups, 1, 1);
    }
}
