using UnityEngine;
using UnityEditor;

[CustomEditor(typeof(FluidSimulator))]
public class FluidSimulatorEditor : Editor
{
    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();

        var sim = (FluidSimulator)target;
        EditorGUILayout.Space();
        using (new EditorGUI.DisabledScope(sim.NumNodesForUI <= 0))
        {
            int maxIndex = Mathf.Max(0, sim.NumNodesForUI - 1);
            int newIndex = EditorGUILayout.IntSlider("Selected Node", sim.selectedNode, 0, maxIndex);
            if (newIndex != sim.selectedNode)
            {
                Undo.RecordObject(sim, "Change Selected Node");
                sim.selectedNode = newIndex;
                EditorUtility.SetDirty(sim);
            }

            // Display divergence for selected node
            if (sim.TryGetSelectedNodeDivergence(out float divergence))
            {
                EditorGUILayout.LabelField("Selected Node Divergence", divergence.ToString("F6"));
            }

            // Display expected divergence based on neighbors and nodesCPU_before
            if (sim.TryGetSelectedNodeExpectedDivergence(out float expected))
            {
                EditorGUILayout.LabelField("Expected Divergence", expected.ToString("F6"));
            }
        }
    }
}


