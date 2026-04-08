using UnityEngine;
using UnityEngine.Rendering;

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

    public RadixSort(ComputeShader shader, uint maxLength)
    {
        this.maxLength = maxLength;
        sortShader = shader;

        prefixSumKernel = sortShader.FindKernel("prefixSum");
        prefixFixupKernel = sortShader.FindKernel("prefixFixup");
        splitPrepKernel = sortShader.FindKernel("split_prep");
        splitScatterKernel = sortShader.FindKernel("split_scatter");
        extractKeysKernel = sortShader.FindKernel("ExtractKeys");
        applySortKernel = sortShader.FindKernel("ApplySort");
        copyParticlesKernel = sortShader.FindKernel("CopyParticles");

        int keySize = sizeof(uint) * 2;
        keysBufferA = new ComputeBuffer((int)maxLength, keySize, ComputeBufferType.Default);
        keysBufferB = new ComputeBuffer((int)maxLength, keySize, ComputeBufferType.Default);

        eBuffer = new ComputeBuffer((int)maxLength, sizeof(uint), ComputeBufferType.Default);
        fBuffer = new ComputeBuffer((int)maxLength, sizeof(uint), ComputeBufferType.Default);

        int particleStride = 3 * 4 + 3 * 4 + 4;
        particleSortScratch = new ComputeBuffer((int)maxLength, particleStride, ComputeBufferType.Default);

        uint threadgroupSize = 512;
        uint numThreadgroups = (maxLength + (threadgroupSize * 2) - 1) / (threadgroupSize * 2);
        uint requiredAuxSize = System.Math.Max(1, numThreadgroups);

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
    public void Sort(ComputeBuffer inputParticles, ComputeBuffer outputParticles, uint actualCount, CommandBuffer gpuCmd = null)
    {
        if (actualCount == 0) return;

        int threadGroupSize = 512;
        int threadGroups = (int)((actualCount + threadGroupSize - 1) / threadGroupSize);

        bool inPlace = ReferenceEquals(inputParticles, outputParticles);
        ComputeBuffer sortedParticlesOut = inPlace ? particleSortScratch : outputParticles;

        sortShader.SetBuffer(extractKeysKernel, "originalParticles", inputParticles);
        sortShader.SetBuffer(extractKeysKernel, "inputKeys", keysBufferA);
        sortShader.SetInt("count", (int)actualCount);
        if (gpuCmd != null)
            gpuCmd.DispatchCompute(sortShader, extractKeysKernel, threadGroups, 1, 1);
        else
            sortShader.Dispatch(extractKeysKernel, threadGroups, 1, 1);

        ComputeBuffer keysIn = keysBufferA;
        ComputeBuffer keysOut = keysBufferB;

        for (int i = 0; i < 32; i++)
        {
            EncodeSplit(keysIn, keysOut, (uint)i, actualCount, gpuCmd);
            (keysIn, keysOut) = (keysOut, keysIn);
        }

        sortShader.SetBuffer(applySortKernel, "originalParticles", inputParticles);
        sortShader.SetBuffer(applySortKernel, "inputKeys", keysIn);
        sortShader.SetBuffer(applySortKernel, "sortedParticlesOut", sortedParticlesOut);
        sortShader.SetInt("count", (int)actualCount);
        if (gpuCmd != null)
            gpuCmd.DispatchCompute(sortShader, applySortKernel, threadGroups, 1, 1);
        else
            sortShader.Dispatch(applySortKernel, threadGroups, 1, 1);

        if (inPlace)
        {
            sortShader.SetBuffer(copyParticlesKernel, "copySrcParticles", particleSortScratch);
            sortShader.SetBuffer(copyParticlesKernel, "copyDstParticles", outputParticles);
            sortShader.SetInt("count", (int)actualCount);
            if (gpuCmd != null)
                gpuCmd.DispatchCompute(sortShader, copyParticlesKernel, threadGroups, 1, 1);
            else
                sortShader.Dispatch(copyParticlesKernel, threadGroups, 1, 1);
        }
    }

    private void EncodeSplit(ComputeBuffer inputKeys, ComputeBuffer outputKeys, uint bit, uint count, CommandBuffer gpuCmd)
    {
        int threadGroups = (int)((count + 512 - 1) / 512);

        sortShader.SetBuffer(splitPrepKernel, "inputKeys", inputKeys);
        sortShader.SetInt("bit", (int)bit);
        sortShader.SetBuffer(splitPrepKernel, "e", eBuffer);
        sortShader.SetInt("count", (int)count);
        if (gpuCmd != null)
            gpuCmd.DispatchCompute(sortShader, splitPrepKernel, threadGroups, 1, 1);
        else
            sortShader.Dispatch(splitPrepKernel, threadGroups, 1, 1);

        EncodeScan(eBuffer, fBuffer, count, gpuCmd);

        sortShader.SetBuffer(splitScatterKernel, "inputKeys", inputKeys);
        sortShader.SetBuffer(splitScatterKernel, "outputKeys", outputKeys);
        sortShader.SetInt("bit", (int)bit);
        sortShader.SetBuffer(splitScatterKernel, "e", eBuffer);
        sortShader.SetBuffer(splitScatterKernel, "f", fBuffer);
        sortShader.SetInt("count", (int)count);
        if (gpuCmd != null)
            gpuCmd.DispatchCompute(sortShader, splitScatterKernel, threadGroups, 1, 1);
        else
            sortShader.Dispatch(splitScatterKernel, threadGroups, 1, 1);
    }

    private void EncodeScan(ComputeBuffer input, ComputeBuffer output, uint length, CommandBuffer gpuCmd)
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
            if (gpuCmd != null)
                gpuCmd.DispatchCompute(sortShader, prefixSumKernel, 1, 1, 1);
            else
                sortShader.Dispatch(prefixSumKernel, 1, 1, 1);
        }
        else
        {
            sortShader.SetBuffer(prefixSumKernel, "input", input);
            sortShader.SetBuffer(prefixSumKernel, "output", output);
            sortShader.SetBuffer(prefixSumKernel, "aux", auxBuffer);
            sortShader.SetInt("len", (int)length);
            sortShader.SetInt("zeroff", (int)zeroff);
            if (gpuCmd != null)
                gpuCmd.DispatchCompute(sortShader, prefixSumKernel, (int)numThreadgroups, 1, 1);
            else
                sortShader.Dispatch(prefixSumKernel, (int)numThreadgroups, 1, 1);

            uint auxLength = numThreadgroups;
            sortShader.SetBuffer(prefixSumKernel, "input", auxBuffer);
            sortShader.SetBuffer(prefixSumKernel, "output", aux2Buffer);
            sortShader.SetBuffer(prefixSumKernel, "aux", auxSmallBuffer);
            sortShader.SetInt("len", (int)auxLength);
            sortShader.SetInt("zeroff", (int)zeroff);
            if (gpuCmd != null)
                gpuCmd.DispatchCompute(sortShader, prefixSumKernel, 1, 1, 1);
            else
                sortShader.Dispatch(prefixSumKernel, 1, 1, 1);

            sortShader.SetBuffer(prefixFixupKernel, "input", output);
            sortShader.SetBuffer(prefixFixupKernel, "aux", aux2Buffer);
            sortShader.SetInt("len", (int)length);
            if (gpuCmd != null)
                gpuCmd.DispatchCompute(sortShader, prefixFixupKernel, (int)numThreadgroups, 1, 1);
            else
                sortShader.Dispatch(prefixFixupKernel, (int)numThreadgroups, 1, 1);
        }
    }
}
