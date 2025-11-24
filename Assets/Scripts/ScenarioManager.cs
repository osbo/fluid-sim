using UnityEngine;

public class ScenarioManager : MonoBehaviour
{
    public FluidSimulator simulator;
    public TrainingDataRecorder recorder;
    public int maxFramesPerRun = 600;
    
    private int currentFrame = 0;

    void Start()
    {
        if (simulator == null)
        {
            simulator = FindObjectOfType<FluidSimulator>();
            if (simulator == null)
            {
                Debug.LogError("ScenarioManager: FluidSimulator not found!");
                enabled = false;
                return;
            }
        }

        if (recorder == null)
        {
            recorder = FindObjectOfType<TrainingDataRecorder>();
            if (recorder == null)
            {
                Debug.LogError("ScenarioManager: TrainingDataRecorder not found!");
                enabled = false;
                return;
            }
        }
    }

    void Update()
    {
        currentFrame++;

        if (currentFrame >= maxFramesPerRun)
        {
            // Stop current recording
            if (recorder != null)
            {
                recorder.StopRecording();
            }
            
            // Disable this component so it doesn't keep checking
            enabled = false;
            
            Debug.Log($"ScenarioManager: Stopped recording after {currentFrame} frames.");
        }
    }
}

