#if UNITY_EDITOR
using UnityEditor;
#endif
using UnityEngine;

/// <summary>Editor: fills null compute references from fixed project paths (same idea as <see cref="FluidLeafOnlyPrecondApply"/> auto-load).</summary>
public partial class FluidSimulator
{
    void LoadEditorDefaultComputeShaders()
    {
#if UNITY_EDITOR
        void Ens(ref ComputeShader s, string path) { if (s == null) s = AssetDatabase.LoadAssetAtPath<ComputeShader>(path); }

        Ens(ref radixSortShader, "Assets/Scripts/RadixSort.compute");
        Ens(ref particlesShader, "Assets/Scripts/Particles.compute");
        Ens(ref nodesPrefixSumsShader, "Assets/Scripts/NodesPrefixSums.compute");
        Ens(ref nodesShader, "Assets/Scripts/Nodes.compute");
        Ens(ref cgSolverShader, "Assets/Scripts/CGSolver.compute");
        Ens(ref csrBuilderShader, "Assets/Scripts/CSRBuilder.compute");

        Ens(ref leafOnlyInputsShader, "Assets/Scripts/LeafOnlyInputs.compute");
        Ens(ref leafOnlyEmbedShader, "Assets/Scripts/LeafOnlyEmbed.compute");
        Ens(ref leafOnlyDiagEdgeFeatsShader, "Assets/Scripts/LeafOnlyDiagEdgeFeats.compute");
        Ens(ref leafOnlyOffEdgeFeatsShader, "Assets/Scripts/LeafOnlyOffEdgeFeats.compute");
        Ens(ref leafOnlyLayer1Shader, "Assets/Scripts/LeafOnlyLayer1.compute");
        if (leafOnlyPrecondApplyShader == null)
            leafOnlyPrecondApplyShader = AssetDatabase.LoadAssetAtPath<ComputeShader>("Assets/Scripts/LeafOnlyPrecondApply.compute");
#endif
    }
}
