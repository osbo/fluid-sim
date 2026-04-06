using System;
using System.Buffers.Binary;
using System.IO;
using UnityEngine;

/// <summary>
/// Loads <c>leaf_only_weights.bytes</c> (same layout as <c>leafonly/checkpoint.py</c>): int header, fp16 tensor blob.
/// Weights are uploaded as float32 for upcoming staged inference / parity. Does not run the network yet.
/// </summary>
public partial class FluidSimulator : MonoBehaviour
{
    public const int LeafOnlyEdgeGateHiddenDim = 16;

    [Tooltip("Fallback only: raw binary TextAsset. Prefer StreamingAssets — Unity may alter binary TextAssets.")]
    public TextAsset leafOnlyWeightsAsset;

    [Tooltip("File name only (not a path): read from Assets/StreamingAssets/<this name>. Example: model_weights_sim_8192.bytes")]
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

        // Try StreamingAssets first (raw File.ReadAllBytes, same as Python open(path,"rb")), then TextAsset.
        byte[] bytes = null;
        string bytesSource = null;
        if (!string.IsNullOrEmpty(leafOnlyWeightsStreamingAssetsName))
        {
            string path = Path.Combine(Application.streamingAssetsPath, leafOnlyWeightsStreamingAssetsName);
            if (File.Exists(path))
            {
                try
                {
                    bytes = File.ReadAllBytes(path);
                    bytesSource = path;
                }
                catch (Exception e)
                {
                    leafOnlyWeightsLoadError = e.Message;
                    Debug.LogWarning($"[LeafOnly] Failed to read {path}: {e.Message}");
                    return;
                }
            }
        }

        string parseErr = null;
        string firstErr = null;

        if (bytes != null)
        {
            if (LeafOnlyCheckpointHeader.TryParse(bytes, out leafOnlyCheckpointHeader, out int weightsByteOffset, out parseErr))
            {
                FinalizeLeafOnlyWeightUpload(bytes, bytesSource, weightsByteOffset);
                return;
            }

            firstErr = parseErr;
        }

        if (leafOnlyWeightsAsset != null && leafOnlyWeightsAsset.bytes != null && leafOnlyWeightsAsset.bytes.Length > 0)
        {
            byte[] ab = leafOnlyWeightsAsset.bytes;
            string an = leafOnlyWeightsAsset.name;
            if (LeafOnlyCheckpointHeader.TryParse(ab, out leafOnlyCheckpointHeader, out int weightsByteOffset, out parseErr))
            {
                Debug.LogWarning(
                    "[LeafOnly] Loaded weights from TextAsset '" + an + "' (StreamingAssets parse failed or missing). Prefer a raw .bytes file under StreamingAssets.");
                FinalizeLeafOnlyWeightUpload(ab, "TextAsset:" + an, weightsByteOffset);
                return;
            }

            if (bytes == null)
                firstErr = parseErr;
        }

        if (bytes == null && (leafOnlyWeightsAsset == null || leafOnlyWeightsAsset.bytes == null || leafOnlyWeightsAsset.bytes.Length == 0))
        {
            leafOnlyWeightsLoadError = "missing or too small";
            if (preconditioner == PreconditionerType.Neural)
                Debug.LogWarning("[LeafOnly] Neural preconditioner selected but no weights found (assign TextAsset or place file under StreamingAssets).");
            return;
        }

        leafOnlyWeightsLoadError = firstErr ?? parseErr ?? "checkpoint parse failed";
        Debug.LogError($"[LeafOnly] Checkpoint parse failed: {leafOnlyWeightsLoadError}");
    }

    private void FinalizeLeafOnlyWeightUpload(byte[] bytes, string sourceLabel, int weightsByteOffset)
    {

        int payloadBytes = bytes.Length - weightsByteOffset;
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
            int o = weightsByteOffset + i * 2;
            ushort h = (ushort)(bytes[o] | (bytes[o + 1] << 8));
            floats[i] = LeafOnlyCheckpointHeader.HalfBitsToFloat(h);
        }

        leafOnlyWeightsFloatBuffer?.Release();
        leafOnlyWeightsFloatBuffer = new ComputeBuffer(numHalf, sizeof(float));
        leafOnlyWeightsFloatBuffer.SetData(floats);

        LeafOnlyWeightsLoadedSuccessfully = true;
        string src = string.IsNullOrEmpty(sourceLabel) ? "(unknown)" : sourceLabel;
        Debug.Log(
            $"[LeafOnly] Loaded weights from {src}: d_model={leafOnlyCheckpointHeader.DModel} leaf_size={leafOnlyCheckpointHeader.LeafSize} " +
            $"layers={leafOnlyCheckpointHeader.NumLayers} heads={leafOnlyCheckpointHeader.NumHeads} gcn_layers={leafOnlyCheckpointHeader.NumGcnLayers} " +
            $"leaf_apply=({leafOnlyCheckpointHeader.LeafApplyDiag},{leafOnlyCheckpointHeader.LeafApplyOff}) attn_layout={leafOnlyCheckpointHeader.AttentionLayoutCode} " +
            $"route_gates={leafOnlyCheckpointHeader.DecoupledRouteGates} highway={leafOnlyCheckpointHeader.HighwayFfnMlp} ffn_concat={leafOnlyCheckpointHeader.FfnConcatWidth} " +
            $"edge_gate_h={leafOnlyCheckpointHeader.EdgeGateHiddenDim} mlp_heads={leafOnlyCheckpointHeader.MlpHeads} " +
            $"floats={numHalf} weightsStartByte={weightsByteOffset}");
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

    /// <summary>Byte offset in <paramref name="file"/> where fp16 weights begin (matches Python <c>read_leaf_only_header</c> end + BOM skip).</summary>
    internal static bool TryParse(byte[] file, out LeafOnlyCheckpointHeader h, out int weightsByteOffset, out string error)
    {
        h = default;
        weightsByteOffset = 0;
        error = null;
        if (file == null)
        {
            error = "null file";
            return false;
        }

        int bestStart = -1;
        LeafOnlyCheckpointHeader bestH = default;
        int bestWeightsOff = 0;

        void Consider(int start)
        {
            if (!TryParseFromOffset(file, start, out LeafOnlyCheckpointHeader ht, out int wt, out _))
                return;
            if (!ValidateArchitecture(in ht))
                return;
            if (bestStart < 0 || start < bestStart)
            {
                bestStart = start;
                bestH = ht;
                bestWeightsOff = wt;
            }
        }

        Consider(0);
        if (file.Length >= 3 && file[0] == 0xEF && file[1] == 0xBB && file[2] == 0xBF)
            Consider(3);

        int maxAligned = Math.Min(file.Length - 96, 8192);
        for (int s = 4; s <= maxAligned; s += 4)
            Consider(s);

        if (bestStart >= 0)
        {
            h = bestH;
            weightsByteOffset = bestWeightsOff;
            if (bestStart == 3)
                Debug.Log("[LeafOnly] Checkpoint: header starts after UTF-8 BOM (Python open(rb) has no BOM; file is still valid).");
            else if (bestStart > 0)
                Debug.LogWarning(
                    "[LeafOnly] Checkpoint header found at byte offset " + bestStart
                    + " (Python read_leaf_only_header expects int fields at offset 0). "
                    + "Copy the raw file produced by save_leaf_only_weights, or strip any preamble before the header.");
            return true;
        }

        bool p0 = TryParseFromOffset(file, 0, out LeafOnlyCheckpointHeader h0, out _, out string e0);
        string detail;
        if (!p0)
            detail = e0 ?? "parse failed at offset 0";
        else if (!ValidateArchitecture(in h0))
            detail = $"d_model={h0.DModel}, leaf_size={h0.LeafSize}, heads={h0.NumHeads}, layers={h0.NumLayers}";
        else
            detail = "internal: offset 0 valid but scan found nothing";

        string hex = HexPrefix(file, 64);
        error =
            "Invalid LeafOnly checkpoint header (" + detail + "). first64_hex=" + hex + ". "
            + "Use the binary from Python (save_leaf_only_weights / leaf_only_weights.bytes) under StreamingAssets; avoid TextAsset for raw .bytes.";
        h = default;
        weightsByteOffset = 0;
        return false;
    }

    private static string HexPrefix(byte[] file, int maxBytes)
    {
        int n = Math.Min(maxBytes, file.Length);
        var sb = new System.Text.StringBuilder(n * 2);
        for (int i = 0; i < n; i++)
            sb.AppendFormat("{0:X2}", file[i]);
        return sb.ToString();
    }

    private static bool ValidateArchitecture(in LeafOnlyCheckpointHeader h)
    {
        if (h.DModel < 8 || h.DModel > 4096)
            return false;
        if (h.LeafSize < 8 || h.LeafSize > 512)
            return false;
        if (h.NumLayers < 1 || h.NumLayers > 64)
            return false;
        if (h.NumHeads < 1 || h.NumHeads > 128)
            return false;
        if (h.InputDim != 9)
            return false;
        if (h.UseGcn != 1)
            return false;
        if (h.NumGcnLayers < 0 || h.NumGcnLayers > 32)
            return false;
        if (h.LeafApplyDiag < 1 || h.LeafApplyDiag > 512)
            return false;
        return true;
    }

    /// <param name="start">Offset into <paramref name="file"/> where the checkpoint begins (0 or 3 after BOM).</param>
    private static bool TryParseFromOffset(byte[] file, int start, out LeafOnlyCheckpointHeader h, out int weightsByteOffset, out string error)
    {
        h = default;
        weightsByteOffset = 0;
        error = null;

        int nAvail = file.Length - start;
        if (nAvail < 32)
        {
            error = "file too short";
            return false;
        }

        int prefixLen = Math.Min(nAvail, 52);

        int dModel, leafSize, inputDim, numLayers, numHeads, useGcn, numGcnLayers;
        int leafApplyDiag, leafApplyOff, attentionLayoutCode = -1, decoupledRouteGates = 0;
        int baseHeaderEnd;
        int highwayFfnMlp = 0;
        int ffnConcatWidth = 1;

        if (prefixLen >= 44)
        {
            dModel = ReadI32Le(file, start);
            leafSize = ReadI32Le(file, start + 4);
            inputDim = ReadI32Le(file, start + 8);
            numLayers = ReadI32Le(file, start + 12);
            numHeads = ReadI32Le(file, start + 16);
            useGcn = ReadI32Le(file, start + 20);
            numGcnLayers = ReadI32Le(file, start + 24);
            leafApplyDiag = ReadI32Le(file, start + 28);
            leafApplyOff = ReadI32Le(file, start + 32);
            attentionLayoutCode = ReadI32Le(file, start + 36);
            decoupledRouteGates = ReadI32Le(file, start + 40);

            if (prefixLen >= 48)
            {
                int peek = ReadI32Le(file, start + 44);
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
                        int peek2 = ReadI32Le(file, start + 48);
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
            else
            {
                baseHeaderEnd = 44;
                highwayFfnMlp = 0;
                ffnConcatWidth = 1;
            }
        }
        else if (prefixLen >= 40)
        {
            dModel = ReadI32Le(file, start);
            leafSize = ReadI32Le(file, start + 4);
            inputDim = ReadI32Le(file, start + 8);
            numLayers = ReadI32Le(file, start + 12);
            numHeads = ReadI32Le(file, start + 16);
            useGcn = ReadI32Le(file, start + 20);
            numGcnLayers = ReadI32Le(file, start + 24);
            leafApplyDiag = ReadI32Le(file, start + 28);
            leafApplyOff = ReadI32Le(file, start + 32);
            attentionLayoutCode = ReadI32Le(file, start + 36);
            decoupledRouteGates = 0;
            baseHeaderEnd = 40;
        }
        else if (prefixLen >= 36)
        {
            dModel = ReadI32Le(file, start);
            leafSize = ReadI32Le(file, start + 4);
            inputDim = ReadI32Le(file, start + 8);
            numLayers = ReadI32Le(file, start + 12);
            numHeads = ReadI32Le(file, start + 16);
            useGcn = ReadI32Le(file, start + 20);
            numGcnLayers = ReadI32Le(file, start + 24);
            leafApplyDiag = ReadI32Le(file, start + 28);
            leafApplyOff = ReadI32Le(file, start + 32);
            decoupledRouteGates = 0;
            baseHeaderEnd = 36;
            attentionLayoutCode = -1;
        }
        else
        {
            dModel = ReadI32Le(file, start);
            leafSize = ReadI32Le(file, start + 4);
            inputDim = ReadI32Le(file, start + 8);
            numLayers = ReadI32Le(file, start + 12);
            numHeads = ReadI32Le(file, start + 16);
            useGcn = ReadI32Le(file, start + 20);
            numGcnLayers = ReadI32Le(file, start + 24);
            leafApplyDiag = ReadI32Le(file, start + 28);
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
        if ((pos is 44 or 48 or 52) && start + pos + 8 <= file.Length)
        {
            int mag = ReadI32Le(file, start + pos);
            int ver = ReadI32Le(file, start + pos + 4);
            if (mag == unchecked((int)EdgeGateExtMagic) && ver == EdgeGateExtVer)
            {
                pos += 8;
                edgeGateHiddenDim = FluidSimulator.LeafOnlyEdgeGateHiddenDim;
                if (start + pos + 8 <= file.Length)
                {
                    int mag2 = ReadI32Le(file, start + pos);
                    int ver2 = ReadI32Le(file, start + pos + 4);
                    if (mag2 == unchecked((int)HeadExtMagic) && ver2 == HeadExtVer)
                    {
                        pos += 8;
                        mlpHeads = 1;
                    }
                }
            }
        }

        int headerRelEnd = pos;
        if (headerRelEnd < 40)
        {
            error = "checkpoint header too old (need v3+)";
            return false;
        }

        weightsByteOffset = start + pos;
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
            HeaderByteSize = weightsByteOffset,
        };
        return true;
    }

    private static int ReadI32Le(byte[] b, int o) => BinaryPrimitives.ReadInt32LittleEndian(b.AsSpan(o, 4));

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
