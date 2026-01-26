using UnityEngine;
using System;
using System.IO;
using System.Collections.Generic;

// Preconditioner component of FluidSimulator
public partial class FluidSimulator : MonoBehaviour
{
    // Timing variables
    private System.Diagnostics.Stopwatch featSw = new System.Diagnostics.Stopwatch();
    private System.Diagnostics.Stopwatch downSw = new System.Diagnostics.Stopwatch();
    private System.Diagnostics.Stopwatch bottleSw = new System.Diagnostics.Stopwatch();
    private System.Diagnostics.Stopwatch upSw = new System.Diagnostics.Stopwatch();
    private System.Diagnostics.Stopwatch headSw = new System.Diagnostics.Stopwatch();

    // V-Cycle Hierarchy
    private List<ComputeBuffer> levelBuffers = new List<ComputeBuffer>(); // Buffers for each level (0=Finest)
    private const int K_BRANCH = 32; // Branching factor
    private const int MIN_LEAF_SIZE = 128; // Bottleneck size
    private int maxLevels = 0; // Loaded from model header

    // Weight Containers
    private struct LevelWeights
    {
        // Down
        public ComputeBuffer w_mixer0, b_mixer0;
        public ComputeBuffer w_mixer2, b_mixer2;
        public ComputeBuffer w_normDown, b_normDown;
        
        // Up
        public ComputeBuffer w_upProj, b_upProj;
        public ComputeBuffer w_fusion0, b_fusion0;
        public ComputeBuffer w_fusion2, b_fusion2;
        public ComputeBuffer w_normUp, b_normUp;
    }
    
    private List<LevelWeights> vCycleWeights = new List<LevelWeights>();
    
    // Bottleneck Weights
    private Dictionary<string, ComputeBuffer> bottleneckWeights = new Dictionary<string, ComputeBuffer>();

    private void LoadModelMetadata()
    {
        if (modelWeightsAsset == null) return;

        byte[] fileBytes = modelWeightsAsset.bytes;
        
        try
        {
            using (BinaryReader reader = new BinaryReader(new MemoryStream(fileBytes)))
            {
                // 1. Header 
                // [reserved1, reserved2, d_model, nhead, num_levels, input_dim]
                p_mean = reader.ReadSingle(); 
                p_std = reader.ReadSingle();
                d_model = reader.ReadInt32();
                num_heads = reader.ReadInt32();
                maxLevels = reader.ReadInt32(); // Replaces num_layers
                input_dim = reader.ReadInt32();

                // Helper: Read packed integers
                ComputeBuffer ReadPackedBuffer(int floatCount)
                {
                    int packedCount = Mathf.CeilToInt(floatCount / 2.0f);
                    ComputeBuffer buffer = new ComputeBuffer(packedCount, sizeof(uint));
                    uint[] data = new uint[packedCount];
                    for (int i = 0; i < packedCount; i++)
                    {
                        if (reader.BaseStream.Position >= reader.BaseStream.Length) break;
                        data[i] = reader.ReadUInt32();
                    }
                    buffer.SetData(data);
                    return buffer;
                }

                // Cleanup old
                foreach(var b in weightBuffers.Values) b?.Release();
                weightBuffers.Clear();
                foreach(var l in vCycleWeights) {
                    l.w_mixer0?.Release(); l.b_mixer0?.Release();
                    l.w_mixer2?.Release(); l.b_mixer2?.Release();
                    l.w_normDown?.Release(); l.b_normDown?.Release();
                    l.w_upProj?.Release(); l.b_upProj?.Release();
                    l.w_fusion0?.Release(); l.b_fusion0?.Release();
                    l.w_fusion2?.Release(); l.b_fusion2?.Release();
                    l.w_normUp?.Release(); l.b_normUp?.Release();
                }
                vCycleWeights.Clear();
                foreach(var b in bottleneckWeights.Values) b?.Release();
                bottleneckWeights.Clear();

                // --- LOAD WEIGHTS ---

                // 1. Feature Projection
                weightBuffers["feature_proj.weight"] = ReadPackedBuffer(input_dim * d_model);
                weightBuffers["feature_proj.bias"]   = ReadPackedBuffer(d_model);
                weightBuffers["layer_embed"]         = ReadPackedBuffer(12 * d_model); // Max octree depth 12

                // 2. Down Samplers (one per level)
                // We load them into a list, as we will iterate through them
                // Note: The file stores them sequentially: Level 0 Down, Level 1 Down...
                // But the Python export loop wrote: Down Samplers (all), Bottleneck, Up Samplers (all)
                
                // We will temporarily store Down weights to pair them with Up weights later if we want a struct,
                // but simpler to just store lists.
                // Let's stick to the struct LevelWeights for organization.
                
                for(int i=0; i<maxLevels; i++)
                {
                    LevelWeights lw = new LevelWeights();
                    lw.w_mixer0 = ReadPackedBuffer(d_model * (d_model * 2)); // Expansion=2
                    lw.b_mixer0 = ReadPackedBuffer(d_model * 2);
                    lw.w_mixer2 = ReadPackedBuffer((d_model * 2) * d_model);
                    lw.b_mixer2 = ReadPackedBuffer(d_model);
                    lw.w_normDown = ReadPackedBuffer(d_model);
                    lw.b_normDown = ReadPackedBuffer(d_model);
                    vCycleWeights.Add(lw);
                }

                // 3. Bottleneck
                bottleneckWeights["q_proj.w"] = ReadPackedBuffer(d_model * d_model);
                bottleneckWeights["q_proj.b"] = ReadPackedBuffer(d_model);
                bottleneckWeights["k_proj.w"] = ReadPackedBuffer(d_model * d_model);
                bottleneckWeights["k_proj.b"] = ReadPackedBuffer(d_model);
                bottleneckWeights["v_proj.w"] = ReadPackedBuffer(d_model * d_model);
                bottleneckWeights["v_proj.b"] = ReadPackedBuffer(d_model);
                bottleneckWeights["out_proj.w"] = ReadPackedBuffer(d_model * d_model);
                bottleneckWeights["out_proj.b"] = ReadPackedBuffer(d_model);
                bottleneckWeights["norm.w"] = ReadPackedBuffer(d_model);
                bottleneckWeights["norm.b"] = ReadPackedBuffer(d_model);

                // 4. Up Samplers (one per level)
                for(int i=0; i<maxLevels; i++)
                {
                    LevelWeights lw = vCycleWeights[i]; // Get existing struct
                    lw.w_upProj = ReadPackedBuffer(d_model * d_model);
                    lw.b_upProj = ReadPackedBuffer(d_model);
                    // Fusion input is 2 * d_model (Skip + Expanded)
                    lw.w_fusion0 = ReadPackedBuffer((d_model * 2) * (d_model * 2)); 
                    lw.b_fusion0 = ReadPackedBuffer(d_model * 2);
                    lw.w_fusion2 = ReadPackedBuffer((d_model * 2) * d_model);
                    lw.b_fusion2 = ReadPackedBuffer(d_model);
                    lw.w_normUp = ReadPackedBuffer(d_model);
                    lw.b_normUp = ReadPackedBuffer(d_model);
                    vCycleWeights[i] = lw; // Update struct
                }

                // 5. Head
                weightBuffers["norm_out.weight"] = ReadPackedBuffer(d_model);
                weightBuffers["norm_out.bias"]   = ReadPackedBuffer(d_model);
                weightBuffers["head.weight"]     = ReadPackedBuffer(d_model * 25);
                weightBuffers["head.bias"]       = ReadPackedBuffer(25);
                
                // Validate all loaded weights for NaNs/Infs
                ValidateAllWeights();
            }
        }
        catch (Exception e) { Debug.LogError($"Load failed: {e.Message}\n{e.StackTrace}"); }
    }

    // Validate weights for NaNs/Infs - indicates training divergence
    private void ValidateWeightBuffer(ComputeBuffer buffer, string name)
    {
        if (buffer == null) return;
        uint[] data = new uint[buffer.count];
        buffer.GetData(data);
        
        int nanCount = 0;
        int infCount = 0;
        for (int i = 0; i < data.Length; i++)
        {
            // Unpack manually on CPU to check for NaNs
            uint packed = data[i];
            
            // Low 16 bits
            float v1 = Mathf.HalfToFloat((ushort)(packed & 0xFFFF));
            // High 16 bits
            float v2 = Mathf.HalfToFloat((ushort)(packed >> 16));
            
            if (float.IsNaN(v1) || float.IsNaN(v2))
            {
                nanCount++;
            }
            if (float.IsInfinity(v1) || float.IsInfinity(v2))
            {
                infCount++;
            }
        }
        
        if (nanCount > 0 || infCount > 0)
        {
            Debug.LogError($"WEIGHT ERROR: Buffer '{name}' contains {nanCount} NaNs and {infCount} Infs out of {data.Length * 2} values! The model training likely diverged.");
        }
    }

    private void ValidateAllWeights()
    {
        // Validate feature projection weights
        ValidateWeightBuffer(weightBuffers.ContainsKey("feature_proj.weight") ? weightBuffers["feature_proj.weight"] : null, "feature_proj.weight");
        ValidateWeightBuffer(weightBuffers.ContainsKey("feature_proj.bias") ? weightBuffers["feature_proj.bias"] : null, "feature_proj.bias");
        ValidateWeightBuffer(weightBuffers.ContainsKey("layer_embed") ? weightBuffers["layer_embed"] : null, "layer_embed");
        
        // Validate V-Cycle weights
        for (int i = 0; i < vCycleWeights.Count; i++)
        {
            var lw = vCycleWeights[i];
            ValidateWeightBuffer(lw.w_mixer0, $"Level {i} w_mixer0");
            ValidateWeightBuffer(lw.b_mixer0, $"Level {i} b_mixer0");
            ValidateWeightBuffer(lw.w_mixer2, $"Level {i} w_mixer2");
            ValidateWeightBuffer(lw.b_mixer2, $"Level {i} b_mixer2");
            ValidateWeightBuffer(lw.w_normDown, $"Level {i} w_normDown");
            ValidateWeightBuffer(lw.b_normDown, $"Level {i} b_normDown");
            ValidateWeightBuffer(lw.w_upProj, $"Level {i} w_upProj");
            ValidateWeightBuffer(lw.b_upProj, $"Level {i} b_upProj");
            ValidateWeightBuffer(lw.w_fusion0, $"Level {i} w_fusion0");
            ValidateWeightBuffer(lw.b_fusion0, $"Level {i} b_fusion0");
            ValidateWeightBuffer(lw.w_fusion2, $"Level {i} w_fusion2");
            ValidateWeightBuffer(lw.b_fusion2, $"Level {i} b_fusion2");
            ValidateWeightBuffer(lw.w_normUp, $"Level {i} w_normUp");
            ValidateWeightBuffer(lw.b_normUp, $"Level {i} b_normUp");
        }
        
        // Validate bottleneck weights
        if (bottleneckWeights.ContainsKey("q_proj.w")) ValidateWeightBuffer(bottleneckWeights["q_proj.w"], "Bottleneck q_proj.w");
        if (bottleneckWeights.ContainsKey("q_proj.b")) ValidateWeightBuffer(bottleneckWeights["q_proj.b"], "Bottleneck q_proj.b");
        if (bottleneckWeights.ContainsKey("k_proj.w")) ValidateWeightBuffer(bottleneckWeights["k_proj.w"], "Bottleneck k_proj.w");
        if (bottleneckWeights.ContainsKey("k_proj.b")) ValidateWeightBuffer(bottleneckWeights["k_proj.b"], "Bottleneck k_proj.b");
        if (bottleneckWeights.ContainsKey("v_proj.w")) ValidateWeightBuffer(bottleneckWeights["v_proj.w"], "Bottleneck v_proj.w");
        if (bottleneckWeights.ContainsKey("v_proj.b")) ValidateWeightBuffer(bottleneckWeights["v_proj.b"], "Bottleneck v_proj.b");
        if (bottleneckWeights.ContainsKey("out_proj.w")) ValidateWeightBuffer(bottleneckWeights["out_proj.w"], "Bottleneck out_proj.w");
        if (bottleneckWeights.ContainsKey("out_proj.b")) ValidateWeightBuffer(bottleneckWeights["out_proj.b"], "Bottleneck out_proj.b");
        if (bottleneckWeights.ContainsKey("norm.w")) ValidateWeightBuffer(bottleneckWeights["norm.w"], "Bottleneck norm.w");
        if (bottleneckWeights.ContainsKey("norm.b")) ValidateWeightBuffer(bottleneckWeights["norm.b"], "Bottleneck norm.b");
        
        // Validate head weights
        ValidateWeightBuffer(weightBuffers.ContainsKey("norm_out.weight") ? weightBuffers["norm_out.weight"] : null, "norm_out.weight");
        ValidateWeightBuffer(weightBuffers.ContainsKey("norm_out.bias") ? weightBuffers["norm_out.bias"] : null, "norm_out.bias");
        ValidateWeightBuffer(weightBuffers.ContainsKey("head.weight") ? weightBuffers["head.weight"] : null, "head.weight");
        ValidateWeightBuffer(weightBuffers.ContainsKey("head.bias") ? weightBuffers["head.bias"] : null, "head.bias");
    }

    private void ApplyPreconditioner(ComputeBuffer r, ComputeBuffer z_out, int kGT, int kG, int kClear, int kJacobi)
    {
        if (preconditioner == PreconditionerType.None)
        {
            CopyBuffer(r, z_out);
            return;
        }
        else if (preconditioner == PreconditionerType.Jacobi)
        {
            if (kJacobi >= 0 && matrixABuffer != null)
            {
                cgSolverShader.SetBuffer(kJacobi, "xBuffer", r);
                cgSolverShader.SetBuffer(kJacobi, "yBuffer", z_out);
                cgSolverShader.SetBuffer(kJacobi, "matrixABuffer", matrixABuffer);
                cgSolverShader.SetInt("numNodes", numNodes);
                Dispatch(kJacobi, numNodes);
            }
            else CopyBuffer(r, z_out);
        }
        else if (preconditioner == PreconditionerType.Neural)
        {
            if (matrixGBuffer == null || reverseNeighborsBuffer == null) { CopyBuffer(r, z_out); return; }

            int requiredSize = Mathf.NextPowerOfTwo(Mathf.Max(numNodes, 512));
            if (zBuffer == null || zBuffer.count < requiredSize)
            {
                zBuffer?.Release();
                zBuffer = new ComputeBuffer(requiredSize, 4);
            }

            // 1. Clear Intermediate
            cgSolverShader.SetBuffer(kClear, "zBuffer", zBuffer); 
            cgSolverShader.SetInt("numNodes", numNodes);
            Dispatch(kClear, numNodes);

            // 2. u = G^T * r
            cgSolverShader.SetBuffer(kGT, "xBuffer", r);
            cgSolverShader.SetBuffer(kGT, "zBuffer", zBuffer);
            cgSolverShader.SetBuffer(kGT, "matrixGBuffer", matrixGBuffer);
            cgSolverShader.SetBuffer(kGT, "neighborsBuffer", neighborsBuffer);
            cgSolverShader.SetBuffer(kGT, "reverseNeighborsBuffer", reverseNeighborsBuffer);
            cgSolverShader.SetBuffer(kGT, "scatterIndicesBuffer", scatterIndicesBuffer);
            cgSolverShader.SetInt("numNodes", numNodes);
            Dispatch(kGT, numNodes);

            // 3. z = G * u + eps * r
            cgSolverShader.SetBuffer(kG, "zBuffer", zBuffer);
            cgSolverShader.SetBuffer(kG, "xBuffer", r);
            cgSolverShader.SetBuffer(kG, "yBuffer", z_out);
            cgSolverShader.SetBuffer(kG, "matrixGBuffer", matrixGBuffer);
            cgSolverShader.SetBuffer(kG, "neighborsBuffer", neighborsBuffer);
            cgSolverShader.SetInt("numNodes", numNodes);
            Dispatch(kG, numNodes);
        }
        else CopyBuffer(r, z_out);
    }

    private float ApplyPreconditionerAndDot(ComputeBuffer r, ComputeBuffer z_out, int kGT, int kG, int kG_Dot, int kClear, int kJacobi)
    {
        float dotResult = 0.0f;

        if (preconditioner == PreconditionerType.Neural && matrixGBuffer != null && reverseNeighborsBuffer != null)
        {
            // 1. Clear
            cgSolverShader.SetBuffer(kClear, "zBuffer", zBuffer);
            cgSolverShader.SetInt("numNodes", numNodes);
            Dispatch(kClear, numNodes);

            // 2. u = G^T * r
            cgSolverShader.SetBuffer(kGT, "xBuffer", r);
            cgSolverShader.SetBuffer(kGT, "zBuffer", zBuffer);
            cgSolverShader.SetBuffer(kGT, "matrixGBuffer", matrixGBuffer);
            cgSolverShader.SetBuffer(kGT, "neighborsBuffer", neighborsBuffer);
            cgSolverShader.SetBuffer(kGT, "reverseNeighborsBuffer", reverseNeighborsBuffer);
            cgSolverShader.SetBuffer(kGT, "scatterIndicesBuffer", scatterIndicesBuffer);
            cgSolverShader.SetInt("numNodes", numNodes);
            Dispatch(kGT, numNodes);

            // 3. z = G * u + eps * r AND dot
            cgSolverShader.SetBuffer(kG_Dot, "zBuffer", zBuffer);
            cgSolverShader.SetBuffer(kG_Dot, "xBuffer", r);
            cgSolverShader.SetBuffer(kG_Dot, "yBuffer", z_out);
            cgSolverShader.SetBuffer(kG_Dot, "matrixGBuffer", matrixGBuffer);
            cgSolverShader.SetBuffer(kG_Dot, "neighborsBuffer", neighborsBuffer);
            cgSolverShader.SetBuffer(kG_Dot, "divergenceBuffer", divergenceBuffer);
            cgSolverShader.SetInt("numNodes", numNodes);
            
            int groups = Mathf.CeilToInt(numNodes / 256.0f);
            cgSolverShader.Dispatch(kG_Dot, groups, 1, 1);

            // 4. Reduction
            float[] partials = new float[groups];
            divergenceBuffer.GetData(partials);
            for(int i=0; i<groups; i++) dotResult += partials[i];
        }
        else
        {
            ApplyPreconditioner(r, z_out, kGT, kG, kClear, kJacobi);
            dotResult = GpuDotProduct(r, z_out);
        }
        return dotResult;
    }

    private void RunNeuralPreconditioner()
    {
        if (preconditionerShader == null || weightBuffers.Count == 0) return;

        // Reset Timers
        featSw.Reset(); downSw.Reset(); bottleSw.Reset(); upSw.Reset(); headSw.Reset();

        // 1. Calculate Schedule (Depth & Padding)
        // We do this CPU side to manage buffer sizes
        List<int> paddingSchedule = new List<int>();
        int currentLen = numNodes;
        int depth = 0;

        // Ensure Level 0 Buffer exists
        int reqSize = Mathf.NextPowerOfTwo(Mathf.Max(numNodes, 256));
        EnsureBuffer(0, reqSize);

        // Calculate V-Cycle path
        while (currentLen > MIN_LEAF_SIZE)
        {
            int remainder = currentLen % K_BRANCH;
            int padding = (K_BRANCH - remainder) % K_BRANCH;
            paddingSchedule.Add(padding);
            
            currentLen = (currentLen + padding) / K_BRANCH;
            depth++;
            
            // Allocate next level buffer
            // Next level needs to hold 'currentLen' elements
            EnsureBuffer(depth, Mathf.NextPowerOfTwo(Mathf.Max(currentLen, 32)));
        }

        // Limit depth if model is smaller than data implies (rare)
        depth = Mathf.Min(depth, maxLevels);

        // --- PHASE 1: FEATURES ---
        featSw.Start();
        int kFeat = preconditionerShader.FindKernel("ComputeFeatures");
        
        preconditionerShader.SetBuffer(kFeat, "nodesBuffer", nodesBuffer);
        preconditionerShader.SetBuffer(kFeat, "neighborsBuffer", neighborsBuffer);
        preconditionerShader.SetBuffer(kFeat, "matrixABuffer", matrixABuffer);
        preconditionerShader.SetBuffer(kFeat, "tokenBufferOut", levelBuffers[0]); // Write to Level 0
        
        preconditionerShader.SetInt("numNodes", numNodes);
        preconditionerShader.SetInt("d_model", d_model);
        float deltaTime = useRealTime ? Time.deltaTime : (1 / frameRate);
        preconditionerShader.SetFloat("deltaTime", deltaTime);

        BindWeights(kFeat, "feature_proj.weight", "weightsFeatureProj");
        BindWeights(kFeat, "feature_proj.bias", "biasFeatureProj");
        BindWeights(kFeat, "layer_embed", "weightsLayerEmbed");
        BindWeights(kFeat, "window_pos_embed", "weightsPosEmbed"); // Not used in V-Cycle but good to have

        // Dispatch: One thread per node
        int groups = Mathf.CeilToInt(numNodes / 256.0f);
        preconditionerShader.Dispatch(kFeat, groups, 1, 1);
        featSw.Stop();

        // --- PHASE 2: DOWN-SAMPLING (Restriction) ---
        downSw.Start();
        int kDown = preconditionerShader.FindKernel("DownSample");
        int currentCount = numNodes;

        for (int i = 0; i < depth; i++)
        {
            int padding = paddingSchedule[i];
            int paddedCount = currentCount + padding;
            int nextCount = paddedCount / K_BRANCH;

            // Bind Buffers
            preconditionerShader.SetBuffer(kDown, "inputBuffer", levelBuffers[i]); // Fine (Read)
            preconditionerShader.SetBuffer(kDown, "outputBuffer", levelBuffers[i+1]); // Coarse (Write)
            
            // Uniforms
            // FIX: Send both valid count and padded count
            // "fineCount" = limit for reading memory (valid data)
            preconditionerShader.SetInt("fineCount", currentCount);
            // "coarseCount" = limit for writing (padded output)
            preconditionerShader.SetInt("coarseCount", nextCount);
            
            // Weights for Level i
            BindLevelWeights(kDown, i, true);

            // Dispatch covers the PADDED count
            // Each group reads 32 Fine tokens (some may be zero-padded)
            preconditionerShader.Dispatch(kDown, nextCount, 1, 1);

            currentCount = nextCount;
        }
        downSw.Stop();

        // --- PHASE 3: BOTTLENECK ---
        bottleSw.Start();
        int kBottle = preconditionerShader.FindKernel("Bottleneck");
        
        // This is an in-place modification of the coarsest buffer
        ComputeBuffer coarseBuf = levelBuffers[depth];
        
        preconditionerShader.SetBuffer(kBottle, "inputBuffer", coarseBuf);
        // We need a temp buffer or careful read/write? 
        // Our shader does: Load Global -> Shared -> Attn -> Write Global. 
        // RWStructuredBuffer is safe if we don't read other global indices in the same pass.
        // But MHA reads ALL indices. So we need a swap buffer or ping-pong.
        // However, N is small (~128). We can assume it fits in cache or just use levelBuffers[0] as scratch if needed.
        // Safer: Use zBuffer as temporary scratch since it's idle right now.
        int requiredBottleneckSize = currentCount * d_model;
        if (zBuffer == null || zBuffer.count < requiredBottleneckSize) 
        {
             zBuffer?.Release(); 
             zBuffer = new ComputeBuffer(Mathf.NextPowerOfTwo(requiredBottleneckSize), 4);
        }
        preconditionerShader.SetBuffer(kBottle, "outputBuffer", zBuffer); // Write to scratch

        preconditionerShader.SetInt("seqLen", currentCount);
        preconditionerShader.SetInt("d_model", d_model);
        preconditionerShader.SetInt("numHeads", num_heads);

        // Bind Bottleneck Weights
        if(bottleneckWeights.ContainsKey("q_proj.w")) preconditionerShader.SetBuffer(kBottle, "w_q", bottleneckWeights["q_proj.w"]);
        if(bottleneckWeights.ContainsKey("q_proj.b")) preconditionerShader.SetBuffer(kBottle, "b_q", bottleneckWeights["q_proj.b"]);
        if(bottleneckWeights.ContainsKey("k_proj.w")) preconditionerShader.SetBuffer(kBottle, "w_k", bottleneckWeights["k_proj.w"]);
        if(bottleneckWeights.ContainsKey("k_proj.b")) preconditionerShader.SetBuffer(kBottle, "b_k", bottleneckWeights["k_proj.b"]);
        if(bottleneckWeights.ContainsKey("v_proj.w")) preconditionerShader.SetBuffer(kBottle, "w_v", bottleneckWeights["v_proj.w"]);
        if(bottleneckWeights.ContainsKey("v_proj.b")) preconditionerShader.SetBuffer(kBottle, "b_v", bottleneckWeights["v_proj.b"]);
        if(bottleneckWeights.ContainsKey("out_proj.w")) preconditionerShader.SetBuffer(kBottle, "w_out", bottleneckWeights["out_proj.w"]);
        if(bottleneckWeights.ContainsKey("out_proj.b")) preconditionerShader.SetBuffer(kBottle, "b_out", bottleneckWeights["out_proj.b"]);
        if(bottleneckWeights.ContainsKey("norm.w")) preconditionerShader.SetBuffer(kBottle, "w_norm1", bottleneckWeights["norm.w"]);
        if(bottleneckWeights.ContainsKey("norm.b")) preconditionerShader.SetBuffer(kBottle, "b_norm1", bottleneckWeights["norm.b"]);

        // Dispatch: One group covers the whole sequence (if small) or tiled. 
        // Kernel uses [numthreads(32, 1, 1)].
        // We dispatch (currentCount, 1, 1). Each group handles one token's attention.
        preconditionerShader.Dispatch(kBottle, currentCount, 1, 1);
        
        // Copy Scratch back to Coarse Buffer (using a simple copy kernel or Swap)
        // Since we are moving UP next, we can just treat zBuffer as the source for the first Up step?
        // No, let's copy back to keep the "levelBuffers" logic clean.
        int kCopy = preconditionerShader.FindKernel("CopyBuffer");
        preconditionerShader.SetBuffer(kCopy, "inputBuffer", zBuffer);
        preconditionerShader.SetBuffer(kCopy, "outputBuffer", coarseBuf);
        preconditionerShader.SetInt("size", currentCount * d_model);
        preconditionerShader.Dispatch(kCopy, Mathf.CeilToInt((currentCount * d_model)/256.0f), 1, 1);
        
        bottleSw.Stop();

        // --- PHASE 4: UP-SAMPLING (Prolongation) ---
        upSw.Start();
        int kUp = preconditionerShader.FindKernel("UpSample");

        for (int i = depth - 1; i >= 0; i--)
        {
            int padding = paddingSchedule[i];
            int fineCount = (currentCount * K_BRANCH) - padding; // Logic reverse
            // Actually, we know the counts from the Down pass implicitly.
            // Fine Count (padded) was `currentCount * K`.
            // The shader handles `fineCount` (padded).
            
            // Buffers:
            // Coarse (Read): levelBuffers[i+1]
            // Fine/Skip (Read/Write): levelBuffers[i] 
            // Note: levelBuffers[i] contains the Skip connection (from Down pass).
            // We write the result back into levelBuffers[i] (Fuse(Coarse, Skip) + Residual).
            // Since Skip == Residual, it's: Fuse(Coarse, Old) + Old.
            
            preconditionerShader.SetBuffer(kUp, "coarseBuffer", levelBuffers[i+1]);
            preconditionerShader.SetBuffer(kUp, "fineBuffer", levelBuffers[i]); // RW
            
            preconditionerShader.SetInt("coarseCount", currentCount);
            
            // Bind Weights
            BindLevelWeights(kUp, i, false);

            // Dispatch: One group per Coarse token.
            // Each group generates 32 Fine tokens.
            preconditionerShader.Dispatch(kUp, currentCount, 1, 1);
            
            // Next iteration count
            int nextFineCount = currentCount * K_BRANCH - padding;
            
            currentCount = nextFineCount;
        }
        upSw.Stop();

        // --- PHASE 5: HEAD ---
        headSw.Start();
        int kHead = preconditionerShader.FindKernel("PredictHead");
        
        // Input is now in levelBuffers[0] (Full Resolution)
        // Ensure MatrixG is allocated with exact size (reallocate if size changed)
        int finalSize = numNodes;
        int requiredSize = finalSize * 25;
        if (matrixGBuffer == null || matrixGBuffer.count != requiredSize)
        {
            matrixGBuffer?.Release();
            matrixGBuffer = new ComputeBuffer(requiredSize, 4);
        }

        preconditionerShader.SetBuffer(kHead, "tokenBufferOut", levelBuffers[0]);
        preconditionerShader.SetBuffer(kHead, "matrixGBuffer", matrixGBuffer);
        
        BindWeights(kHead, "norm_out.weight", "w_normOut");
        BindWeights(kHead, "norm_out.bias", "b_normOut");
        BindWeights(kHead, "head.weight", "w_head");
        BindWeights(kHead, "head.bias", "b_head");
        
        preconditionerShader.SetInt("numNodes", finalSize);
        preconditionerShader.SetInt("d_model", d_model);

        // Batched Dispatch (max 65535 groups)
        int processed = 0;
        while(processed < finalSize)
        {
            int batch = Mathf.Min(finalSize - processed, 65535);
            preconditionerShader.SetInt("headOffset", processed);
            preconditionerShader.Dispatch(kHead, batch, 1, 1);
            processed += batch;
        }
        headSw.Stop();
    }

    private void EnsureBuffer(int level, int size)
    {
        while(levelBuffers.Count <= level) levelBuffers.Add(null);
        
        if (levelBuffers[level] == null || levelBuffers[level].count < size * d_model)
        {
            levelBuffers[level]?.Release();
            levelBuffers[level] = new ComputeBuffer(size * d_model, 4); // float
        }
    }

    private void BindWeights(int k, string dictKey, string shaderName)
    {
        if (weightBuffers.ContainsKey(dictKey))
            preconditionerShader.SetBuffer(k, shaderName, weightBuffers[dictKey]);
    }

    private void BindLevelWeights(int k, int level, bool isDown)
    {
        if (level >= vCycleWeights.Count) return;
        var lw = vCycleWeights[level];
        if (isDown)
        {
            preconditionerShader.SetBuffer(k, "w_mixer0", lw.w_mixer0);
            preconditionerShader.SetBuffer(k, "b_mixer0", lw.b_mixer0);
            preconditionerShader.SetBuffer(k, "w_mixer2", lw.w_mixer2);
            preconditionerShader.SetBuffer(k, "b_mixer2", lw.b_mixer2);
            preconditionerShader.SetBuffer(k, "w_normDown", lw.w_normDown);
            preconditionerShader.SetBuffer(k, "b_normDown", lw.b_normDown);
        }
        else
        {
            preconditionerShader.SetBuffer(k, "w_upProj", lw.w_upProj);
            preconditionerShader.SetBuffer(k, "b_upProj", lw.b_upProj);
            preconditionerShader.SetBuffer(k, "w_fusion0", lw.w_fusion0);
            preconditionerShader.SetBuffer(k, "b_fusion0", lw.b_fusion0);
            preconditionerShader.SetBuffer(k, "w_fusion2", lw.w_fusion2);
            preconditionerShader.SetBuffer(k, "b_fusion2", lw.b_fusion2);
            preconditionerShader.SetBuffer(k, "w_normUp", lw.w_normUp);
            preconditionerShader.SetBuffer(k, "b_normUp", lw.b_normUp);
        }
    }

    private void ReleasePreconditionerBuffers()
    {
        foreach(var b in levelBuffers) b?.Release();
        levelBuffers.Clear();
        zBuffer?.Release();
        matrixGBuffer?.Release();
        zVectorBuffer?.Release();
        scatterIndicesBuffer?.Release();
    }
}