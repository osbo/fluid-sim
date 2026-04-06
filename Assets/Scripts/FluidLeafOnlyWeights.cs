using System;
using System.IO;
using UnityEngine;

/// <summary>
/// Loads <c>leaf_only_weights.bytes</c> (same layout as <c>leafonly/checkpoint.py</c>): int header, fp16 tensor blob.
/// Weights are uploaded as float32 for upcoming staged inference / parity. Does not run the network yet.
/// </summary>
public partial class FluidSimulator : MonoBehaviour
{
    public const int LeafOnlyEdgeGateHiddenDim = 16;

    [Tooltip("Optional TextAsset for leaf_only_weights.bytes (binary). If null, StreamingAssets is tried.")]
    public TextAsset leafOnlyWeightsAsset;

    [Tooltip("When leafOnlyWeightsAsset is null, load from Application.streamingAssetsPath + this file name.")]
    public string leafOnlyWeightsStreamingAssetsName = "leaf_only_weights.bytes";

    [Tooltip("If true, also load when the StreamingAssets file exists (even if preconditioner is not Neural).")]
    public bool tryLoadLeafOnlyWeightsFromStreamingAssets = true;

    private ComputeBuffer leafOnlyWeightsFloatBuffer;
    private LeafOnlyCheckpointHeader leafOnlyCheckpointHeader;
    private bool leafOnlyWeightsLoadAttempted;
    private string leafOnlyWeightsLoadError;

    public bool LeafOnlyWeightsLoadedSuccessfully { get; private set; }
    public LeafOnlyCheckpointHeader LeafOnlyCheckpoint => leafOnlyCheckpointHeader;
    public ComputeBuffer LeafOnlyWeightsFloatBuffer => leafOnlyWeightsFloatBuffer;
    public int LeafOnlyWeightsFloatCount => leafOnlyWeightsFloatBuffer != null ? leafOnlyWeightsFloatBuffer.count : 0;

    private void TryLoadLeafOnlyWeightsFromDisk()
    {
        if (leafOnlyWeightsLoadAttempted)
            return;
        leafOnlyWeightsLoadAttempted = true;
        LeafOnlyWeightsLoadedSuccessfully = false;
        leafOnlyWeightsLoadError = null;

        bool wantLoad = preconditioner == PreconditionerType.Neural || leafOnlyWeightsAsset != null;
        if (!wantLoad && tryLoadLeafOnlyWeightsFromStreamingAssets && !string.IsNullOrEmpty(leafOnlyWeightsStreamingAssetsName))
        {
            string streamingPath = Path.Combine(Application.streamingAssetsPath, leafOnlyWeightsStreamingAssetsName);
            if (File.Exists(streamingPath))
                wantLoad = true;
        }

        if (!wantLoad)
            return;

        byte[] bytes = null;
        if (leafOnlyWeightsAsset != null && leafOnlyWeightsAsset.bytes != null && leafOnlyWeightsAsset.bytes.Length > 0)
            bytes = leafOnlyWeightsAsset.bytes;
        else if (!string.IsNullOrEmpty(leafOnlyWeightsStreamingAssetsName))
        {
            string path = Path.Combine(Application.streamingAssetsPath, leafOnlyWeightsStreamingAssetsName);
            if (File.Exists(path))
            {
                try
                {
                    bytes = File.ReadAllBytes(path);
                }
                catch (Exception e)
                {
                    leafOnlyWeightsLoadError = e.Message;
                    Debug.LogWarning($"[LeafOnly] Failed to read {path}: {e.Message}");
                    return;
                }
            }
        }

        if (bytes == null || bytes.Length < 32)
        {
            leafOnlyWeightsLoadError = "missing or too small";
            if (preconditioner == PreconditionerType.Neural)
                Debug.LogWarning("[LeafOnly] Neural preconditioner selected but no weights found (assign TextAsset or place bytes under StreamingAssets).");
            return;
        }

        if (!LeafOnlyCheckpointHeader.TryParse(bytes, out leafOnlyCheckpointHeader, out int headerBytes, out string parseErr))
        {
            leafOnlyWeightsLoadError = parseErr;
            Debug.LogError($"[LeafOnly] Checkpoint parse failed: {parseErr}");
            return;
        }

        int payloadBytes = bytes.Length - headerBytes;
        if (payloadBytes < 0 || (payloadBytes & 1) != 0)
        {
            leafOnlyWeightsLoadError = "bad payload size";
            Debug.LogError($"[LeafOnly] Invalid payload length after header ({payloadBytes} bytes).");
            return;
        }

        int numHalf = payloadBytes / 2;
        var floats = new float[numHalf];
        for (int i = 0; i < numHalf; i++)
        {
            ushort h = (ushort)(bytes[headerBytes + i * 2] | (bytes[headerBytes + i * 2 + 1] << 8));
            floats[i] = LeafOnlyCheckpointHeader.HalfBitsToFloat(h);
        }

        leafOnlyWeightsFloatBuffer?.Release();
        leafOnlyWeightsFloatBuffer = new ComputeBuffer(numHalf, sizeof(float));
        leafOnlyWeightsFloatBuffer.SetData(floats);

        LeafOnlyWeightsLoadedSuccessfully = true;
        Debug.Log(
            $"[LeafOnly] Loaded weights: d_model={leafOnlyCheckpointHeader.DModel} leaf_size={leafOnlyCheckpointHeader.LeafSize} " +
            $"layers={leafOnlyCheckpointHeader.NumLayers} heads={leafOnlyCheckpointHeader.NumHeads} gcn_layers={leafOnlyCheckpointHeader.NumGcnLayers} " +
            $"leaf_apply=({leafOnlyCheckpointHeader.LeafApplyDiag},{leafOnlyCheckpointHeader.LeafApplyOff}) attn_layout={leafOnlyCheckpointHeader.AttentionLayoutCode} " +
            $"route_gates={leafOnlyCheckpointHeader.DecoupledRouteGates} highway={leafOnlyCheckpointHeader.HighwayFfnMlp} ffn_concat={leafOnlyCheckpointHeader.FfnConcatWidth} " +
            $"edge_gate_h={leafOnlyCheckpointHeader.EdgeGateHiddenDim} mlp_heads={leafOnlyCheckpointHeader.MlpHeads} " +
            $"floats={numHalf} headerBytes={headerBytes}");
    }

    private void ReleaseLeafOnlyWeightsBuffers()
    {
        leafOnlyWeightsFloatBuffer?.Release();
        leafOnlyWeightsFloatBuffer = null;
        LeafOnlyWeightsLoadedSuccessfully = false;
        leafOnlyWeightsLoadAttempted = false;
    }
}

/// <summary>Architecture fields from the first bytes of leaf_only_weights.bytes (matches Python <c>leaf_only_arch_from_checkpoint</c>).</summary>
public struct LeafOnlyCheckpointHeader
{
    private const uint EdgeGateExtMagic = 0x4544474D; // "MDGE" LE
    private const int EdgeGateExtVer = 1;
    private const uint HeadExtMagic = 0x48454144; // "DAEH" LE
    private const int HeadExtVer = 1;

    public int DModel;
    public int LeafSize;
    public int InputDim;
    public int NumLayers;
    public int NumHeads;
    public int UseGcn;
    public int NumGcnLayers;
    public int LeafApplyDiag;
    public int LeafApplyOff;
    public int AttentionLayoutCode;
    public int DecoupledRouteGates;
    public int EdgeGateHiddenDim;
    public int MlpHeads;
    public int HighwayFfnMlp;
    public int FfnConcatWidth;
    public int HeaderByteSize;

    internal static bool TryParse(byte[] file, out LeafOnlyCheckpointHeader h, out int headerBytes, out string error)
    {
        h = default;
        headerBytes = 0;
        error = null;

        if (file == null || file.Length < 32)
        {
            error = "file too short";
            return false;
        }

        int prefixLen = Math.Min(file.Length, 52);
        var prefix = new byte[prefixLen];
        Buffer.BlockCopy(file, 0, prefix, 0, prefixLen);

        int dModel, leafSize, inputDim, numLayers, numHeads, useGcn, numGcnLayers;
        int leafApplyDiag, leafApplyOff, attentionLayoutCode = -1, decoupledRouteGates = 0;
        int baseHeaderEnd;
        int highwayFfnMlp = 0;
        int ffnConcatWidth = 1;

        if (prefixLen >= 44)
        {
            dModel = ReadI32(prefix, 0);
            leafSize = ReadI32(prefix, 4);
            inputDim = ReadI32(prefix, 8);
            numLayers = ReadI32(prefix, 12);
            numHeads = ReadI32(prefix, 16);
            useGcn = ReadI32(prefix, 20);
            numGcnLayers = ReadI32(prefix, 24);
            leafApplyDiag = ReadI32(prefix, 28);
            leafApplyOff = ReadI32(prefix, 32);
            attentionLayoutCode = ReadI32(prefix, 36);
            decoupledRouteGates = ReadI32(prefix, 40);

            int peek = ReadI32(prefix, 44);
            if (peek == unchecked((int)EdgeGateExtMagic))
            {
                baseHeaderEnd = 44;
                highwayFfnMlp = 0;
                ffnConcatWidth = 1;
            }
            else
            {
                highwayFfnMlp = peek != 0 ? 1 : 0;
                baseHeaderEnd = 48;
                if (prefixLen >= 52)
                {
                    int peek2 = ReadI32(prefix, 48);
                    if (peek2 == unchecked((int)EdgeGateExtMagic))
                        ffnConcatWidth = highwayFfnMlp != 0 ? 3 : 1;
                    else
                    {
                        baseHeaderEnd = 52;
                        int fw = peek2;
                        if (fw == 4)
                            ffnConcatWidth = 4;
                        else if (fw == 3)
                            ffnConcatWidth = 3;
                        else
                            ffnConcatWidth = highwayFfnMlp != 0 ? 4 : 1;
                    }
                }
                else
                    ffnConcatWidth = highwayFfnMlp != 0 ? 3 : 1;
            }
        }
        else if (prefixLen >= 40)
        {
            dModel = ReadI32(prefix, 0);
            leafSize = ReadI32(prefix, 4);
            inputDim = ReadI32(prefix, 8);
            numLayers = ReadI32(prefix, 12);
            numHeads = ReadI32(prefix, 16);
            useGcn = ReadI32(prefix, 20);
            numGcnLayers = ReadI32(prefix, 24);
            leafApplyDiag = ReadI32(prefix, 28);
            leafApplyOff = ReadI32(prefix, 32);
            attentionLayoutCode = ReadI32(prefix, 36);
            decoupledRouteGates = 0;
            baseHeaderEnd = 40;
        }
        else if (prefixLen >= 36)
        {
            dModel = ReadI32(prefix, 0);
            leafSize = ReadI32(prefix, 4);
            inputDim = ReadI32(prefix, 8);
            numLayers = ReadI32(prefix, 12);
            numHeads = ReadI32(prefix, 16);
            useGcn = ReadI32(prefix, 20);
            numGcnLayers = ReadI32(prefix, 24);
            leafApplyDiag = ReadI32(prefix, 28);
            leafApplyOff = ReadI32(prefix, 32);
            decoupledRouteGates = 0;
            baseHeaderEnd = 36;
            attentionLayoutCode = -1;
        }
        else
        {
            dModel = ReadI32(prefix, 0);
            leafSize = ReadI32(prefix, 4);
            inputDim = ReadI32(prefix, 8);
            numLayers = ReadI32(prefix, 12);
            numHeads = ReadI32(prefix, 16);
            useGcn = ReadI32(prefix, 20);
            numGcnLayers = ReadI32(prefix, 24);
            leafApplyDiag = ReadI32(prefix, 28);
            leafApplyOff = leafApplyDiag;
            decoupledRouteGates = 0;
            baseHeaderEnd = 32;
            attentionLayoutCode = -1;
        }

        if (leafApplyDiag <= 0)
            leafApplyDiag = leafSize;
        if (leafApplyOff <= 0)
            leafApplyOff = leafApplyDiag;

        int edgeGateHiddenDim = 0;
        int mlpHeads = 0;
        int pos = baseHeaderEnd;
        if ((pos is 44 or 48 or 52) && pos + 8 <= file.Length)
        {
            int mag = ReadI32(file, pos);
            int ver = ReadI32(file, pos + 4);
            if (mag == unchecked((int)EdgeGateExtMagic) && ver == EdgeGateExtVer)
            {
                pos += 8;
                edgeGateHiddenDim = FluidSimulator.LeafOnlyEdgeGateHiddenDim;
                if (pos + 8 <= file.Length)
                {
                    int mag2 = ReadI32(file, pos);
                    int ver2 = ReadI32(file, pos + 4);
                    if (mag2 == unchecked((int)HeadExtMagic) && ver2 == HeadExtVer)
                    {
                        pos += 8;
                        mlpHeads = 1;
                    }
                }
            }
        }

        headerBytes = pos;

        if (headerBytes < 40)
        {
            error = "checkpoint header too old (need v3+)";
            return false;
        }

        h = new LeafOnlyCheckpointHeader
        {
            DModel = dModel,
            LeafSize = leafSize,
            InputDim = inputDim,
            NumLayers = numLayers,
            NumHeads = numHeads,
            UseGcn = useGcn,
            NumGcnLayers = numGcnLayers,
            LeafApplyDiag = leafApplyDiag,
            LeafApplyOff = leafApplyOff,
            AttentionLayoutCode = attentionLayoutCode,
            DecoupledRouteGates = decoupledRouteGates,
            EdgeGateHiddenDim = edgeGateHiddenDim,
            MlpHeads = mlpHeads,
            HighwayFfnMlp = highwayFfnMlp,
            FfnConcatWidth = ffnConcatWidth,
            HeaderByteSize = headerBytes,
        };
        return true;
    }

    private static int ReadI32(byte[] b, int o) =>
        b[o] | (b[o + 1] << 8) | (b[o + 2] << 16) | (b[o + 3] << 24);

    /// <summary>Decode fp16 LE half word to float32 (matches NumPy float16 → float32 used in checkpoint load).</summary>
    internal static float HalfBitsToFloat(ushort h)
    {
        int sign = h & 0x8000;
        int exp = (h >> 10) & 0x1F;
        int mant = h & 0x3FF;

        if (exp == 0)
        {
            if (mant == 0)
            {
                uint bits = (uint)sign << 16;
                return BitConverter.ToSingle(BitConverter.GetBytes(bits), 0);
            }
            float v = (mant / 1024f) * (1f / 16384f);
            return sign != 0 ? -v : v;
        }

        if (exp == 31)
        {
            if (mant != 0)
                return float.NaN;
            return sign != 0 ? float.NegativeInfinity : float.PositiveInfinity;
        }

        int e = exp - 15 + 127;
        uint ubits = (uint)((sign << 16) | ((uint)e << 23) | ((uint)mant << 13));
        return BitConverter.ToSingle(BitConverter.GetBytes(ubits), 0);
    }
}
