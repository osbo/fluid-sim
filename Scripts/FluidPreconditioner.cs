using UnityEngine;
using System;
using System.IO;
using System.Collections.Generic;

// Preconditioner component of FluidSimulator
public partial class FluidSimulator : MonoBehaviour
{
    // Timing variables for preconditioner
    private System.Diagnostics.Stopwatch featSw = new System.Diagnostics.Stopwatch();
    private System.Diagnostics.Stopwatch qkvSw = new System.Diagnostics.Stopwatch();
    private System.Diagnostics.Stopwatch attnSw = new System.Diagnostics.Stopwatch();
    private System.Diagnostics.Stopwatch ffnSw = new System.Diagnostics.Stopwatch();
    private System.Diagnostics.Stopwatch headSw = new System.Diagnostics.Stopwatch();
    private void LoadModelMetadata()
    {
        if (modelWeightsAsset == null) return;

        byte[] fileBytes = modelWeightsAsset.bytes;
        
        // ... (Header reading code remains the same: p_mean, d_model, etc.) ...
        
        try
        {
            using (BinaryReader reader = new BinaryReader(new MemoryStream(fileBytes)))
            {
                // 1. Header (Standard 32-bit reads)
                p_mean = reader.ReadSingle();
                p_std = reader.ReadSingle();
                d_model = reader.ReadInt32();
                num_heads = reader.ReadInt32();
                num_layers = reader.ReadInt32();
                input_dim = reader.ReadInt32();

                // 2. NEW Helper: Read packed integers into ComputeBuffer
                // 'floatCount' is the number of weights the model expects.
                ComputeBuffer ReadPackedBuffer(int floatCount)
                {
                    // Calculate how many uints we need (ceil(count / 2))
                    int packedCount = Mathf.CeilToInt(floatCount / 2.0f);
                    
                    // Unity StructuredBuffers must have a stride divisible by 4.
                    // uint is 4 bytes, so this works perfectly.
                    ComputeBuffer buffer = new ComputeBuffer(packedCount, sizeof(uint));
                    
                    // Read the packed data directly as unsigned integers
                    uint[] data = new uint[packedCount];
                    for (int i = 0; i < packedCount; i++)
                    {
                        // Check end of stream
                        if (reader.BaseStream.Position >= reader.BaseStream.Length) break;
                        data[i] = reader.ReadUInt32();
                    }
                    
                    buffer.SetData(data);
                    return buffer;
                }

                // 3. Load weights (Logic same, but function changed)
                foreach(var b in weightBuffers.Values) b?.Release();
                weightBuffers.Clear();

                // Stem
                weightBuffers["feature_proj.weight"] = ReadPackedBuffer(input_dim * d_model);
                weightBuffers["feature_proj.bias"]   = ReadPackedBuffer(d_model);
                weightBuffers["layer_embed"]         = ReadPackedBuffer(12 * d_model);
                weightBuffers["window_pos_embed"]    = ReadPackedBuffer(WINDOW_SIZE * d_model);

                // Layers
                for (int i = 0; i < num_layers; i++)
                {
                    string p = $"layer_{i}.";
                    weightBuffers[p + "in_proj_w"]  = ReadPackedBuffer(d_model * 3 * d_model);
                    weightBuffers[p + "in_proj_b"]  = ReadPackedBuffer(3 * d_model);
                    weightBuffers[p + "out_proj_w"] = ReadPackedBuffer(d_model * d_model);
                    weightBuffers[p + "out_proj_b"] = ReadPackedBuffer(d_model);
                    weightBuffers[p + "norm1_w"]    = ReadPackedBuffer(d_model);
                    weightBuffers[p + "norm1_b"]    = ReadPackedBuffer(d_model);
                    weightBuffers[p + "linear1_w"]  = ReadPackedBuffer(d_model * 2 * d_model);
                    weightBuffers[p + "linear1_b"]  = ReadPackedBuffer(2 * d_model);
                    weightBuffers[p + "linear2_w"]  = ReadPackedBuffer(2 * d_model * d_model);
                    weightBuffers[p + "linear2_b"]  = ReadPackedBuffer(d_model);
                    weightBuffers[p + "norm2_w"]    = ReadPackedBuffer(d_model);
                    weightBuffers[p + "norm2_b"]    = ReadPackedBuffer(d_model);
                }

                // Head
                weightBuffers["norm_out.weight"] = ReadPackedBuffer(d_model);
                weightBuffers["norm_out.bias"]   = ReadPackedBuffer(d_model);
                weightBuffers["head.weight"]     = ReadPackedBuffer(d_model * 25);
                weightBuffers["head.bias"]       = ReadPackedBuffer(25);

                Debug.Log("Loaded packed 16-bit weights.");
            }
        }
        catch (Exception e) { Debug.LogError($"Load failed: {e.Message}"); }
    }
    private void ApplyPreconditioner(ComputeBuffer r, ComputeBuffer z_out, int kGT, int kG, int kClear, int kJacobi)
    {
        if (preconditioner == PreconditionerType.None)
        {
            // Identity fallback: z = r
            CopyBuffer(r, z_out);
            return;
        }
        else if (preconditioner == PreconditionerType.Jacobi)
        {
            // Jacobi preconditioning: z = D^-1 * r
            // where D is the diagonal of the Laplacian matrix A
            if (kJacobi >= 0 && diagonalBuffer != null)
            {
                cgSolverShader.SetBuffer(kJacobi, "xBuffer", r);
                cgSolverShader.SetBuffer(kJacobi, "yBuffer", z_out);
                cgSolverShader.SetBuffer(kJacobi, "diagonalBuffer", diagonalBuffer);
                cgSolverShader.SetInt("numNodes", numNodes);
                Dispatch(kJacobi, numNodes);
            }
            else
            {
                // Fallback to identity if kernel not found
                CopyBuffer(r, z_out);
            }
            return;
        }
        else if (preconditioner == PreconditionerType.Neural)
        {
            if (matrixGBuffer == null)
            {
                // Identity fallback: z = r
                CopyBuffer(r, z_out);
                return;
            }

            // Ensure zBuffer is allocated (used as intermediate 'u' in shader)
            int requiredSize = Mathf.NextPowerOfTwo(Mathf.Max(numNodes, 512));
            if (zBuffer == null || zBuffer.count < requiredSize)
            {
                zBuffer?.Release();
                zBuffer = new ComputeBuffer(requiredSize, 4);
            }

            // 1. Clear Intermediate 'zBuffer' (used as 'u' in Shader)
            cgSolverShader.SetBuffer(kClear, "zBuffer", zBuffer); 
            cgSolverShader.SetInt("numNodes", numNodes);
            Dispatch(kClear, numNodes);

            // 2. u = G^T * r  (Scatter)
            // CGSolver.compute: ApplySparseGT reads 'xBuffer' (r), writes 'zBuffer' (u)
            cgSolverShader.SetBuffer(kGT, "xBuffer", r);
            cgSolverShader.SetBuffer(kGT, "zBuffer", zBuffer); // Intermediate
            cgSolverShader.SetBuffer(kGT, "matrixGBuffer", matrixGBuffer);
            cgSolverShader.SetBuffer(kGT, "neighborsBuffer", neighborsBuffer);
            cgSolverShader.SetBuffer(kGT, "reverseNeighborsBuffer", reverseNeighborsBuffer);
            cgSolverShader.SetBuffer(kGT, "scatterIndicesBuffer", scatterIndicesBuffer); // Bind precomputed indices
            cgSolverShader.SetInt("numNodes", numNodes);
            Dispatch(kGT, numNodes);

            // 3. z = G * u + eps * r (Gather)
            // CGSolver.compute: fused ApplySparseGAndDot is used elsewhere; here we use the non-dot variant if desired.
            cgSolverShader.SetBuffer(kG, "zBuffer", zBuffer); // Intermediate input
            cgSolverShader.SetBuffer(kG, "xBuffer", r);       // For epsilon skip connection
            cgSolverShader.SetBuffer(kG, "yBuffer", z_out);   // Final Output
            cgSolverShader.SetBuffer(kG, "matrixGBuffer", matrixGBuffer);
            cgSolverShader.SetBuffer(kG, "neighborsBuffer", neighborsBuffer);
            cgSolverShader.SetInt("numNodes", numNodes);
            Dispatch(kG, numNodes);
        }
        else
        {
            // Unknown preconditioner type, fallback to identity
            CopyBuffer(r, z_out);
        }
    }
    private float ApplyPreconditionerAndDot(ComputeBuffer r, ComputeBuffer z_out, int kGT, int kG, int kG_Dot, int kClear, int kJacobi)
    {
        float dotResult = 0.0f;

        if (preconditioner == PreconditionerType.Neural && matrixGBuffer != null)
        {
            // 1. Clear Intermediate
            cgSolverShader.SetBuffer(kClear, "zBuffer", zBuffer);
            cgSolverShader.SetInt("numNodes", numNodes);
            Dispatch(kClear, numNodes);

            // 2. Step 1: u = G^T * r (Standard Scatter)
            cgSolverShader.SetBuffer(kGT, "xBuffer", r);
            cgSolverShader.SetBuffer(kGT, "zBuffer", zBuffer);
            cgSolverShader.SetBuffer(kGT, "matrixGBuffer", matrixGBuffer);
            cgSolverShader.SetBuffer(kGT, "neighborsBuffer", neighborsBuffer);
            cgSolverShader.SetBuffer(kGT, "reverseNeighborsBuffer", reverseNeighborsBuffer);
            cgSolverShader.SetBuffer(kGT, "scatterIndicesBuffer", scatterIndicesBuffer);
            cgSolverShader.SetInt("numNodes", numNodes);
            Dispatch(kGT, numNodes);

            // 3. Step 2 + Dot: z = G * u + eps * r AND rho = dot(r, z)
            cgSolverShader.SetBuffer(kG_Dot, "zBuffer", zBuffer);       // u (Input)
            cgSolverShader.SetBuffer(kG_Dot, "xBuffer", r);             // r (Input for eps & dot)
            cgSolverShader.SetBuffer(kG_Dot, "yBuffer", z_out);         // z (Output)
            cgSolverShader.SetBuffer(kG_Dot, "matrixGBuffer", matrixGBuffer);
            cgSolverShader.SetBuffer(kG_Dot, "neighborsBuffer", neighborsBuffer);
            cgSolverShader.SetBuffer(kG_Dot, "divergenceBuffer", divergenceBuffer); // Scratch for reduction
            cgSolverShader.SetInt("numNodes", numNodes);
            
                        int groups = Mathf.CeilToInt(numNodes / 256.0f);
            
            
            
                        // --- [FIX START] ADD THIS LINE ---
            
                        cgSolverShader.Dispatch(kG_Dot, groups, 1, 1);
            
                        // --- [FIX END] ---
            
            
            
                        // 4. CPU Reduction
            
                        // Only read back the partial sums (tiny read)
            
                        float[] partials = new float[groups];
            divergenceBuffer.GetData(partials);
            for(int i=0; i<groups; i++) dotResult += partials[i];
        }
        else
        {
            // Fallback for Jacobi / None: Run standard logic + separate dot product
            ApplyPreconditioner(r, z_out, kGT, kG, kClear, kJacobi);
            dotResult = GpuDotProduct(r, z_out);
        }

        return dotResult;
    }
    private void RunNeuralPreconditioner()
    {
        if (preconditionerShader == null)
        {
            return;
        }

        // --- NEURAL PRECONDITIONER INFERENCE ---
        // Align to window size for attention
        int paddedNodes = Mathf.CeilToInt(numNodes / (float)WINDOW_SIZE) * WINDOW_SIZE;
        int windowGroups = paddedNodes / WINDOW_SIZE;

        // 1. Resize Buffers (UPDATED SIZES)
        int requiredSize = Mathf.NextPowerOfTwo(Mathf.Max(numNodes, WINDOW_SIZE));
        int requiredPaddedSize = Mathf.CeilToInt(requiredSize / (float)WINDOW_SIZE) * WINDOW_SIZE;
        
        // MatrixG: [N, 25] -> SoA layout: [25 * N] with stride 4
        if (matrixGBuffer == null || matrixGBuffer.count < requiredSize * 25)
        {
            matrixGBuffer?.Release();
            matrixGBuffer = new ComputeBuffer(requiredSize * 25, 4); // SoA: stride is 4 bytes (float)
        }
        if (zBuffer == null || zBuffer.count < requiredSize)
        {
            zBuffer?.Release();
            zBuffer = new ComputeBuffer(requiredSize, 4);
        }
        
        // Internal buffers: Use d_model loaded from the .bytes file
        int current_d_model = this.d_model; 
        int stride = current_d_model * 4; 
        
        if (tokenBuffer == null || tokenBuffer.count < requiredPaddedSize * current_d_model)
        {
            tokenBuffer?.Release();
            tokenBuffer = new ComputeBuffer(requiredPaddedSize * current_d_model, 4);
        }
        if (tokenBufferOut == null || tokenBufferOut.count < requiredPaddedSize * current_d_model)
        {
            tokenBufferOut?.Release();
            tokenBufferOut = new ComputeBuffer(requiredPaddedSize * current_d_model, 4);
        }
        if (bufferQ == null || bufferQ.count < requiredPaddedSize * current_d_model)
        {
            bufferQ?.Release();
            bufferQ = new ComputeBuffer(requiredPaddedSize * current_d_model, 4);
        }
        if (bufferK == null || bufferK.count < requiredPaddedSize * current_d_model)
        {
            bufferK?.Release();
            bufferK = new ComputeBuffer(requiredPaddedSize * current_d_model, 4);
        }
        if (bufferV == null || bufferV.count < requiredPaddedSize * current_d_model)
        {
            bufferV?.Release();
            bufferV = new ComputeBuffer(requiredPaddedSize * current_d_model, 4);
        }
        if (bufferAttn == null || bufferAttn.count < requiredPaddedSize * current_d_model)
        {
            bufferAttn?.Release();
            bufferAttn = new ComputeBuffer(requiredPaddedSize * current_d_model, 4);
        }

        // Reset timers
        featSw.Reset();
        qkvSw.Reset();
        attnSw.Reset();
        ffnSw.Reset();
        headSw.Reset();

        // 1. Compute Features
        featSw.Start();
        int kFeat = preconditionerShader.FindKernel("ComputeFeatures");
        preconditionerShader.SetBuffer(kFeat, "nodesBuffer", nodesBuffer);
        preconditionerShader.SetBuffer(kFeat, "neighborsBuffer", neighborsBuffer);
        preconditionerShader.SetBuffer(kFeat, "tokenBuffer", tokenBuffer);
        preconditionerShader.SetBuffer(kFeat, "diagonalBuffer", diagonalBuffer); // Bind pre-computed diagonal buffer
        preconditionerShader.SetInt("numNodes", numNodes);
        preconditionerShader.SetInt("maxNodes", paddedNodes);
        preconditionerShader.SetInt("windowSize", WINDOW_SIZE); // Pass window size to shader
        
        // Set Shader Constants
        preconditionerShader.SetInt("d_model", current_d_model);
        preconditionerShader.SetInt("d_ffn", current_d_model * 2);
        preconditionerShader.SetInt("numHeads", NUM_HEADS); // Pass num heads to shader
        
        // Set Feature/Embed weights
        if (weightBuffers.ContainsKey("feature_proj.weight"))
            preconditionerShader.SetBuffer(kFeat, "weightsFeatureProj", weightBuffers["feature_proj.weight"]);
        if (weightBuffers.ContainsKey("feature_proj.bias"))
            preconditionerShader.SetBuffer(kFeat, "biasFeatureProj", weightBuffers["feature_proj.bias"]);
        if (weightBuffers.ContainsKey("layer_embed"))
            preconditionerShader.SetBuffer(kFeat, "weightsLayerEmbed", weightBuffers["layer_embed"]);
        if (weightBuffers.ContainsKey("window_pos_embed"))
            preconditionerShader.SetBuffer(kFeat, "weightsPosEmbed", weightBuffers["window_pos_embed"]);
        
        // Dispatch features using Window Groups
        preconditionerShader.Dispatch(kFeat, windowGroups, 1, 1);
        featSw.Stop();

        // Loop Layers (num_layers should be 2 from metadata)
        int kFused = preconditionerShader.FindKernel("FusedTransformerLayer");

        for (int i = 0; i < num_layers; i++)
        {
            string p = $"layer_{i}.";

            qkvSw.Start(); // reuse qkvSw to time the fused layer

            // Input / Output buffers (ping-pong)
            preconditionerShader.SetBuffer(kFused, "tokenBuffer", tokenBuffer);
            preconditionerShader.SetBuffer(kFused, "tokenBufferOut", tokenBufferOut);

            // Global uniforms
            preconditionerShader.SetInt("numNodes", numNodes);
            preconditionerShader.SetInt("maxNodes", paddedNodes);

            // Transformer weights for this layer
            if (weightBuffers.ContainsKey(p + "norm1_w"))
                preconditionerShader.SetBuffer(kFused, "w_norm1", weightBuffers[p + "norm1_w"]);
            if (weightBuffers.ContainsKey(p + "norm1_b"))
                preconditionerShader.SetBuffer(kFused, "b_norm1", weightBuffers[p + "norm1_b"]);

            if (weightBuffers.ContainsKey(p + "in_proj_w"))
                preconditionerShader.SetBuffer(kFused, "w_attn_in", weightBuffers[p + "in_proj_w"]);
            if (weightBuffers.ContainsKey(p + "in_proj_b"))
                preconditionerShader.SetBuffer(kFused, "b_attn_in", weightBuffers[p + "in_proj_b"]);

            if (weightBuffers.ContainsKey(p + "out_proj_w"))
                preconditionerShader.SetBuffer(kFused, "w_attn_out", weightBuffers[p + "out_proj_w"]);
            if (weightBuffers.ContainsKey(p + "out_proj_b"))
                preconditionerShader.SetBuffer(kFused, "b_attn_out", weightBuffers[p + "out_proj_b"]);

            if (weightBuffers.ContainsKey(p + "norm2_w"))
                preconditionerShader.SetBuffer(kFused, "w_norm2", weightBuffers[p + "norm2_w"]);
            if (weightBuffers.ContainsKey(p + "norm2_b"))
                preconditionerShader.SetBuffer(kFused, "b_norm2", weightBuffers[p + "norm2_b"]);

            if (weightBuffers.ContainsKey(p + "linear1_w"))
                preconditionerShader.SetBuffer(kFused, "w_ffn1", weightBuffers[p + "linear1_w"]);
            if (weightBuffers.ContainsKey(p + "linear1_b"))
                preconditionerShader.SetBuffer(kFused, "b_ffn1", weightBuffers[p + "linear1_b"]);
            if (weightBuffers.ContainsKey(p + "linear2_w"))
                preconditionerShader.SetBuffer(kFused, "w_ffn2", weightBuffers[p + "linear2_w"]);
            if (weightBuffers.ContainsKey(p + "linear2_b"))
                preconditionerShader.SetBuffer(kFused, "b_ffn2", weightBuffers[p + "linear2_b"]);

            // Dispatch fused kernel: 1 group per window, WINDOW_SIZE threads per group
            preconditionerShader.Dispatch(kFused, windowGroups, 1, 1);
            qkvSw.Stop();

            // Swap Buffers
            var temp = tokenBuffer;
            tokenBuffer = tokenBufferOut;
            tokenBufferOut = temp;
        }

        // 4. Head
        headSw.Start();
        int kHead = preconditionerShader.FindKernel("PredictHead");
        preconditionerShader.SetBuffer(kHead, "tokenBufferOut", tokenBuffer); // Note: swapped buffer is now input
        preconditionerShader.SetBuffer(kHead, "matrixGBuffer", matrixGBuffer);
        preconditionerShader.SetInt("numNodes", numNodes);
        preconditionerShader.SetInt("d_model", current_d_model);
        preconditionerShader.SetInt("d_ffn", current_d_model * 2);
        
        if (weightBuffers.ContainsKey("norm_out.weight"))
            preconditionerShader.SetBuffer(kHead, "w_normOut", weightBuffers["norm_out.weight"]);
        if (weightBuffers.ContainsKey("norm_out.bias"))
            preconditionerShader.SetBuffer(kHead, "b_normOut", weightBuffers["norm_out.bias"]);
        if (weightBuffers.ContainsKey("head.weight"))
            preconditionerShader.SetBuffer(kHead, "w_head", weightBuffers["head.weight"]);
        if (weightBuffers.ContainsKey("head.bias"))
            preconditionerShader.SetBuffer(kHead, "b_head", weightBuffers["head.bias"]);
        
        // Dispatch Head: 1 Group per Node (32 threads)
        // Handle large node counts by dispatching in batches (max 65535 thread groups per dimension)
        const int MAX_THREAD_GROUPS = 65535;
        int remainingNodes = paddedNodes;
        int offset = 0;
        while (remainingNodes > 0)
        {
            int batchSize = Mathf.Min(remainingNodes, MAX_THREAD_GROUPS);
            preconditionerShader.SetInt("headOffset", offset);
            preconditionerShader.Dispatch(kHead, batchSize, 1, 1);
            offset += batchSize;
            remainingNodes -= batchSize;
        }
        headSw.Stop();
    }
    private void ReleasePreconditionerBuffers()
    {
        bufferQ?.Release();
        bufferK?.Release();
        bufferV?.Release();
        bufferAttn?.Release();
        tokenBuffer?.Release();
        tokenBufferOut?.Release();
        matrixGBuffer?.Release();
        zBuffer?.Release();
        zVectorBuffer?.Release();
        scatterIndicesBuffer?.Release();
        diagonalBuffer?.Release();
    }
}
