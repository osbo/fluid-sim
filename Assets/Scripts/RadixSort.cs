using UnityEngine;

public class RadixSort
{
    private ComputeShader sortShader;
    private int prefixSumKernel;
    private int prefixFixupKernel;
    private int splitPrepKernel;
    private int splitScatterKernel;
    private int extractKeysKernel;
    private int applySortKernel;
    private int copyParticlesKernel;

    private ComputeBuffer keysBufferA;
    private ComputeBuffer keysBufferB;
    private ComputeBuffer auxBuffer;
    private ComputeBuffer aux2Buffer;
    private ComputeBuffer auxSmallBuffer;
    private ComputeBuffer eBuffer;
    private ComputeBuffer fBuffer;
    private ComputeBuffer particleSortScratch;

    private uint maxLength;

    /// <param name="maxParticleCount">Max radix-sort particle count (keys / e / f buffers).</param>
    /// <param name="maxExclusiveScanLength">
    /// Max length for <see cref="ExclusivePrefixScan"/> (e.g. uniform-grid block-sum chain = ceil(N³/1024)).
    /// Aux buffers are sized for max(particles, this) so multi-block prefix + fixup matches octree <see cref="EncodeScan"/>.
    /// </param>
    public RadixSort(ComputeShader shader, uint maxParticleCount, uint maxExclusiveScanLength = 0)
    {
        maxLength = maxParticleCount;
        sortShader = shader;

        prefixSumKernel = sortShader.FindKernel("prefixSum");
        prefixFixupKernel = sortShader.FindKernel("prefixFixup");
        splitPrepKernel = sortShader.FindKernel("split_prep");
        splitScatterKernel = sortShader.FindKernel("split_scatter");
        extractKeysKernel = sortShader.FindKernel("ExtractKeys");
        applySortKernel = sortShader.FindKernel("ApplySort");
        copyParticlesKernel = sortShader.FindKernel("CopyParticles");

        int keySize = sizeof(uint) * 2;
        keysBufferA = new ComputeBuffer((int)maxParticleCount, keySize, ComputeBufferType.Default);
        keysBufferB = new ComputeBuffer((int)maxParticleCount, keySize, ComputeBufferType.Default);

        eBuffer = new ComputeBuffer((int)maxParticleCount, sizeof(uint), ComputeBufferType.Default);
        fBuffer = new ComputeBuffer((int)maxParticleCount, sizeof(uint), ComputeBufferType.Default);

        int particleStride = 3 * 4 + 3 * 4 + 4;
        particleSortScratch = new ComputeBuffer((int)maxParticleCount, particleStride, ComputeBufferType.Default);

        uint threadgroupSize = 512;
        uint ngParticles = (maxParticleCount + (threadgroupSize * 2) - 1) / (threadgroupSize * 2);
        uint ngScan = (maxExclusiveScanLength + (threadgroupSize * 2) - 1) / (threadgroupSize * 2);
        uint requiredAuxSize = System.Math.Max(1, System.Math.Max(ngParticles, ngScan));

        auxBuffer = new ComputeBuffer((int)requiredAuxSize, sizeof(uint), ComputeBufferType.Default);
        aux2Buffer = new ComputeBuffer((int)requiredAuxSize, sizeof(uint), ComputeBufferType.Default);
        auxSmallBuffer = new ComputeBuffer(1, sizeof(uint), ComputeBufferType.Default);
    }

    public void ReleaseBuffers()
    {
        keysBufferA?.Release();
        keysBufferB?.Release();
        auxBuffer?.Release();
        aux2Buffer?.Release();
        auxSmallBuffer?.Release();
        eBuffer?.Release();
        fBuffer?.Release();
        particleSortScratch?.Release();
    }

    /// <summary>Morton-code radix sort: 32 single-bit passes (stable on <c>uint2(morton, index)</c>).</summary>
    public void Sort(ComputeBuffer inputParticles, ComputeBuffer outputParticles, uint actualCount)
    {
        if (actualCount == 0) return;

        int threadGroupSize = 512;
        int threadGroups = (int)((actualCount + threadGroupSize - 1) / threadGroupSize);

        bool inPlace = ReferenceEquals(inputParticles, outputParticles);
        ComputeBuffer sortedParticlesOut = inPlace ? particleSortScratch : outputParticles;

        sortShader.SetBuffer(extractKeysKernel, "originalParticles", inputParticles);
        sortShader.SetBuffer(extractKeysKernel, "inputKeys", keysBufferA);
        sortShader.SetInt("count", (int)actualCount);
        sortShader.Dispatch(extractKeysKernel, threadGroups, 1, 1);

        ComputeBuffer keysIn = keysBufferA;
        ComputeBuffer keysOut = keysBufferB;

        for (int i = 0; i < 32; i++)
        {
            EncodeSplit(keysIn, keysOut, (uint)i, actualCount);
            (keysIn, keysOut) = (keysOut, keysIn);
        }

        sortShader.SetBuffer(applySortKernel, "originalParticles", inputParticles);
        sortShader.SetBuffer(applySortKernel, "inputKeys", keysIn);
        sortShader.SetBuffer(applySortKernel, "sortedParticlesOut", sortedParticlesOut);
        sortShader.SetInt("count", (int)actualCount);
        sortShader.Dispatch(applySortKernel, threadGroups, 1, 1);

        if (inPlace)
        {
            sortShader.SetBuffer(copyParticlesKernel, "copySrcParticles", particleSortScratch);
            sortShader.SetBuffer(copyParticlesKernel, "copyDstParticles", outputParticles);
            sortShader.SetInt("count", (int)actualCount);
            sortShader.Dispatch(copyParticlesKernel, threadGroups, 1, 1);
        }
    }

    /// <summary>
    /// Same exclusive prefix as each radix pass (<see cref="EncodeScan"/>): multi-group scan + block-sum scan + fixup.
    /// Used by uniform-grid cell-count prefix (block totals in <c>aux</c> after NodesPrefixSums pass 1).
    /// </summary>
    public void ExclusivePrefixScan(ComputeBuffer input, ComputeBuffer output, uint length)
    {
        if (length == 0) return;
        EncodeScan(input, output, length);
    }

    private void EncodeSplit(ComputeBuffer inputKeys, ComputeBuffer outputKeys, uint bit, uint count)
    {
        int threadGroups = (int)((count + 512 - 1) / 512);

        sortShader.SetBuffer(splitPrepKernel, "inputKeys", inputKeys);
        sortShader.SetInt("bit", (int)bit);
        sortShader.SetBuffer(splitPrepKernel, "e", eBuffer);
        sortShader.SetInt("count", (int)count);
        sortShader.Dispatch(splitPrepKernel, threadGroups, 1, 1);

        EncodeScan(eBuffer, fBuffer, count);

        sortShader.SetBuffer(splitScatterKernel, "inputKeys", inputKeys);
        sortShader.SetBuffer(splitScatterKernel, "outputKeys", outputKeys);
        sortShader.SetInt("bit", (int)bit);
        sortShader.SetBuffer(splitScatterKernel, "e", eBuffer);
        sortShader.SetBuffer(splitScatterKernel, "f", fBuffer);
        sortShader.SetInt("count", (int)count);
        sortShader.Dispatch(splitScatterKernel, threadGroups, 1, 1);
    }

    private void EncodeScan(ComputeBuffer input, ComputeBuffer output, uint length)
    {
        if (length == 0) return;

        uint threadgroupSize = 512;
        uint numThreadgroups = (length + (threadgroupSize * 2) - 1) / (threadgroupSize * 2);
        uint zeroff = 1;

        if (numThreadgroups <= 1)
        {
            sortShader.SetBuffer(prefixSumKernel, "input", input);
            sortShader.SetBuffer(prefixSumKernel, "output", output);
            sortShader.SetBuffer(prefixSumKernel, "aux", auxBuffer);
            sortShader.SetInt("len", (int)length);
            sortShader.SetInt("zeroff", (int)zeroff);
            sortShader.Dispatch(prefixSumKernel, 1, 1, 1);
        }
        else
        {
            sortShader.SetBuffer(prefixSumKernel, "input", input);
            sortShader.SetBuffer(prefixSumKernel, "output", output);
            sortShader.SetBuffer(prefixSumKernel, "aux", auxBuffer);
            sortShader.SetInt("len", (int)length);
            sortShader.SetInt("zeroff", (int)zeroff);
            sortShader.Dispatch(prefixSumKernel, (int)numThreadgroups, 1, 1);

            uint auxLength = numThreadgroups;
            sortShader.SetBuffer(prefixSumKernel, "input", auxBuffer);
            sortShader.SetBuffer(prefixSumKernel, "output", aux2Buffer);
            sortShader.SetBuffer(prefixSumKernel, "aux", auxSmallBuffer);
            sortShader.SetInt("len", (int)auxLength);
            sortShader.SetInt("zeroff", (int)zeroff);
            sortShader.Dispatch(prefixSumKernel, 1, 1, 1);

            sortShader.SetBuffer(prefixFixupKernel, "input", output);
            sortShader.SetBuffer(prefixFixupKernel, "aux", aux2Buffer);
            sortShader.SetInt("len", (int)length);
            sortShader.Dispatch(prefixFixupKernel, (int)numThreadgroups, 1, 1);
        }
    }
}
