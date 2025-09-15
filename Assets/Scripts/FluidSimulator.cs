using UnityEngine;

public class FluidSimulator : MonoBehaviour
{
    public ComputeShader radixSortShader;
    public ComputeShader fluidKernelsShader;
    public int numParticles = 10000;
    private RadixSort radixSort;
    private ComputeBuffer mortonCodes;
    private ComputeBuffer particleIndices;
    private ComputeBuffer sortedMortonCodes;
    private ComputeBuffer sortedParticleIndices;

    void Start()
    {
        radixSort = new RadixSort(radixSortShader, (uint)numParticles);
        mortonCodes = new ComputeBuffer(numParticles, sizeof(uint));
        particleIndices = new ComputeBuffer(numParticles, sizeof(uint));
        sortedMortonCodes = new ComputeBuffer(numParticles, sizeof(uint));
        sortedParticleIndices = new ComputeBuffer(numParticles, sizeof(uint));

        uint[] mortonCodesArray = new uint[numParticles];
        uint[] particleIndicesArray = new uint[numParticles];
        for (uint i = 0; i < numParticles; i++)
        {
            mortonCodesArray[i] = (uint)Random.Range(0, int.MaxValue);
            particleIndicesArray[i] = i;
        }

        // Debug: Log unsorted codes
        // string unsorted_output = "Unsorted Morton Codes (first 100): ";
        // for (int i = 0; i < Mathf.Min(100, numParticles); i++)
        // {
        //     unsorted_output += mortonCodesArray[i] + " ";
        // }
        // Debug.Log(unsorted_output);

        mortonCodes.SetData(mortonCodesArray);
        particleIndices.SetData(particleIndicesArray);

        radixSort.Sort(mortonCodes, particleIndices, sortedMortonCodes, sortedParticleIndices, (uint)numParticles);

        uint[] sortedMortonCodesArray = new uint[numParticles];
        sortedMortonCodes.GetData(sortedMortonCodesArray);

        // Debug: Log sorted codes
        // string sorted_output = "Sorted Morton Codes (first 100): ";
        // for (int i = 0; i < Mathf.Min(100, numParticles); i++)
        // {
        //     sorted_output += sortedMortonCodesArray[i] + " ";
        // }
        // Debug.Log(sorted_output);
    }

    void OnDestroy()
    {
        radixSort.ReleaseBuffers();
        mortonCodes.Release();
        particleIndices.Release();
        sortedMortonCodes.Release();
        sortedParticleIndices.Release();
    }
}

public class RadixSort
{
    private ComputeShader sortShader;
    private int prefixSumKernel;
    private int prefixFixupKernel;
    private int splitPrepKernel;
    private int splitScatterKernel;
    private int copyBufferKernel;
    private int copyIndicesKernel;
    private int clearBuffer32Kernel;

    private ComputeBuffer tempKeys;
    private ComputeBuffer tempIndices;
    private ComputeBuffer tempKeysB;
    private ComputeBuffer tempIndicesB;
    private ComputeBuffer auxBuffer;
    private ComputeBuffer aux2Buffer;
    private ComputeBuffer auxSmallBuffer;
    private ComputeBuffer eBuffer;
    private ComputeBuffer fBuffer;

    private uint maxLength;

    public RadixSort(ComputeShader shader, uint maxLength)
    {
        this.maxLength = maxLength;
        this.sortShader = shader;

        prefixSumKernel = sortShader.FindKernel("prefixSum");
        prefixFixupKernel = sortShader.FindKernel("prefixFixup");
        splitPrepKernel = sortShader.FindKernel("split_prep");
        splitScatterKernel = sortShader.FindKernel("split_scatter");
        copyBufferKernel = sortShader.FindKernel("copyBuffer");
        copyIndicesKernel = sortShader.FindKernel("copyIndices");
        clearBuffer32Kernel = sortShader.FindKernel("clearBuffer32");

        tempKeys = new ComputeBuffer((int)maxLength, sizeof(uint), ComputeBufferType.Default);
        tempIndices = new ComputeBuffer((int)maxLength, sizeof(uint), ComputeBufferType.Default);
        tempKeysB = new ComputeBuffer((int)maxLength, sizeof(uint), ComputeBufferType.Default);
        tempIndicesB = new ComputeBuffer((int)maxLength, sizeof(uint), ComputeBufferType.Default);
        eBuffer = new ComputeBuffer((int)maxLength, sizeof(uint), ComputeBufferType.Default);
        fBuffer = new ComputeBuffer((int)maxLength, sizeof(uint), ComputeBufferType.Default);

        uint threadgroupSize = 512;
        uint numThreadgroups = (maxLength + (threadgroupSize * 2) - 1) / (threadgroupSize * 2);
        uint requiredAuxSize = System.Math.Max(1, numThreadgroups);
        auxBuffer = new ComputeBuffer((int)requiredAuxSize, sizeof(uint), ComputeBufferType.Default);
        aux2Buffer = new ComputeBuffer((int)requiredAuxSize, sizeof(uint), ComputeBufferType.Default);
        auxSmallBuffer = new ComputeBuffer(1, sizeof(uint), ComputeBufferType.Default);
    }

    public void ReleaseBuffers()
    {
        tempKeys.Release();
        tempIndices.Release();
        tempKeysB.Release();
        tempIndicesB.Release();
        auxBuffer.Release();
        aux2Buffer.Release();
        auxSmallBuffer.Release();
        eBuffer.Release();
        fBuffer.Release();
    }

    public void Sort(ComputeBuffer inputKeys, ComputeBuffer inputIndices, ComputeBuffer outputKeys, ComputeBuffer outputIndices, uint actualCount)
    {
        if (actualCount == 0) return;

        ClearBuffer(tempKeys, (uint)tempKeys.count);
        ClearBuffer(tempIndices, (uint)tempIndices.count);
        ClearBuffer(tempKeysB, (uint)tempKeysB.count);
        ClearBuffer(tempIndicesB, (uint)tempIndicesB.count);

        int threadGroupSize = 512;
        int threadGroups = (int)((actualCount + threadGroupSize - 1) / threadGroupSize);

        sortShader.SetBuffer(copyBufferKernel, "input", inputKeys);
        sortShader.SetBuffer(copyBufferKernel, "output", tempKeys);
        sortShader.SetInt("count", (int)actualCount);
        sortShader.Dispatch(copyBufferKernel, threadGroups, 1, 1);

        sortShader.SetBuffer(copyIndicesKernel, "inputIndices", inputIndices);
        sortShader.SetBuffer(copyIndicesKernel, "outputIndices", tempIndices);
        sortShader.SetInt("count", (int)actualCount);
        sortShader.Dispatch(copyIndicesKernel, threadGroups, 1, 1);

        ComputeBuffer keysIn = tempKeys;
        ComputeBuffer keysOut = tempKeysB;
        ComputeBuffer indicesIn = tempIndices;
        ComputeBuffer indicesOut = tempIndicesB;

        for (int i = 0; i < 32; i++)
        {
            EncodeSplit(keysIn, indicesIn, keysOut, indicesOut, (uint)i, actualCount);

            (keysIn, keysOut) = (keysOut, keysIn);
            (indicesIn, indicesOut) = (indicesOut, indicesIn);
        }

        sortShader.SetBuffer(copyBufferKernel, "input", keysIn);
        sortShader.SetBuffer(copyBufferKernel, "output", outputKeys);
        sortShader.SetInt("count", (int)actualCount);
        sortShader.Dispatch(copyBufferKernel, threadGroups, 1, 1);

        sortShader.SetBuffer(copyIndicesKernel, "inputIndices", indicesIn);
        sortShader.SetBuffer(copyIndicesKernel, "outputIndices", outputIndices);
        sortShader.SetInt("count", (int)actualCount);
        sortShader.Dispatch(copyIndicesKernel, threadGroups, 1, 1);
    }

    private void EncodeSplit(ComputeBuffer input, ComputeBuffer inputIndices, ComputeBuffer output, ComputeBuffer outputIndices, uint bit, uint count)
    {
        sortShader.SetBuffer(splitPrepKernel, "input", input);
        sortShader.SetInt("bit", (int)bit);
        sortShader.SetBuffer(splitPrepKernel, "e", eBuffer);
        sortShader.SetInt("count", (int)count);
        int threadGroups = (int)((count + 512 - 1) / 512);
        sortShader.Dispatch(splitPrepKernel, threadGroups, 1, 1);

        EncodeScan(eBuffer, fBuffer, count, bit);

        sortShader.SetBuffer(splitScatterKernel, "input", input);
        sortShader.SetBuffer(splitScatterKernel, "inputIndices", inputIndices);
        sortShader.SetBuffer(splitScatterKernel, "output", output);
        sortShader.SetBuffer(splitScatterKernel, "outputIndices", outputIndices);
        sortShader.SetInt("bit", (int)bit);
        sortShader.SetBuffer(splitScatterKernel, "e", eBuffer);
        sortShader.SetBuffer(splitScatterKernel, "f", fBuffer);
        sortShader.SetInt("count", (int)count);
        sortShader.Dispatch(splitScatterKernel, threadGroups, 1, 1);
    }

    private void EncodeScan(ComputeBuffer input, ComputeBuffer output, uint length, uint bit)
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
    
    // Debug method - can be removed or commented out
    // private void VerifyPrefixSum(ComputeBuffer prefixSumBuffer, uint N, uint bit)
    // {
    //     uint[] data = new uint[N];
    //     prefixSumBuffer.GetData(data);
    //
    //     Debug.Log($"Prefix Sum Verification for bit {bit}:");
    //     string output = "Prefix Sum (first 100): ";
    //     for (int i = 0; i < Mathf.Min(100, N); i++)
    //     {
    //         output += data[i] + " ";
    //     }
    //     Debug.Log(output);
    // }


    private void ClearBuffer(ComputeBuffer buffer, uint count)
    {
        sortShader.SetBuffer(clearBuffer32Kernel, "output", buffer);
        sortShader.SetInt("count", (int)count);
        int threadGroups = (int)((count + 63) / 64);
        sortShader.Dispatch(clearBuffer32Kernel, threadGroups, 1, 1);
    }
}