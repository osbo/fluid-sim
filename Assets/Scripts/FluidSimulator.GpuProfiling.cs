using System;
using UnityEngine;
using UnityEngine.Rendering;

// Optional CommandBuffer BeginSample/EndSample scopes for Frame Debugger / Xcode GPU capture.
// Does not enable Unity Editor "GPU Usage" on Metal (that module is unsupported there).
public partial class FluidSimulator : MonoBehaviour
{
    [Tooltip("Nest simulation compute in one CommandBuffer per major phase (safer on Metal than many ExecuteCommandBuffer calls per frame). Labels show in Frame Debugger / Xcode. Off by default — enable only when profiling; Unity 6 + Metal may glitch if misused.")]
    public bool emitGpuDebugLabels = false;

    private CommandBuffer fluidSimGpuProfileCmd;
    private int gpuProfileCmdDepth;

    private void EnsureFluidSimGpuProfileCmd()
    {
        if (fluidSimGpuProfileCmd != null)
            return;
        fluidSimGpuProfileCmd = new CommandBuffer { name = "FluidSim GPU Labels" };
    }

    private void ReleaseFluidSimGpuProfileCmd()
    {
        fluidSimGpuProfileCmd?.Dispose();
        fluidSimGpuProfileCmd = null;
        gpuProfileCmdDepth = 0;
    }

    /// <summary>Record a compute dispatch; uses the active GPU profile CommandBuffer when <see cref="emitGpuDebugLabels"/> and a <see cref="GpuProfileSection"/> is active.</summary>
    public void GpuProfileDispatchCompute(ComputeShader shader, int kernel, int threadGroupsX, int threadGroupsY, int threadGroupsZ)
    {
        if (emitGpuDebugLabels && gpuProfileCmdDepth > 0 && fluidSimGpuProfileCmd != null)
            fluidSimGpuProfileCmd.DispatchCompute(shader, kernel, threadGroupsX, threadGroupsY, threadGroupsZ);
        else
            shader.Dispatch(kernel, threadGroupsX, threadGroupsY, threadGroupsZ);
    }

    public void GpuProfileDispatchIndirect(ComputeShader shader, int kernel, ComputeBuffer indirectBuffer, uint argsOffset)
    {
        if (emitGpuDebugLabels && gpuProfileCmdDepth > 0 && fluidSimGpuProfileCmd != null)
            fluidSimGpuProfileCmd.DispatchCompute(shader, kernel, indirectBuffer, argsOffset);
        else
            shader.DispatchIndirect(kernel, indirectBuffer, argsOffset);
    }

    private readonly struct GpuProfileSection : IDisposable
    {
        private readonly FluidSimulator owner;
        private readonly string name;
        private readonly bool active;

        public GpuProfileSection(FluidSimulator owner, string name)
        {
            this.owner = owner;
            this.name = name;
            if (!owner.emitGpuDebugLabels)
            {
                active = false;
                return;
            }
            owner.EnsureFluidSimGpuProfileCmd();
            owner.fluidSimGpuProfileCmd.BeginSample(name);
            owner.gpuProfileCmdDepth++;
            active = true;
        }

        public void Dispose()
        {
            if (!active)
                return;
            owner.gpuProfileCmdDepth--;
            owner.fluidSimGpuProfileCmd.EndSample(name);
            if (owner.gpuProfileCmdDepth == 0)
            {
                Graphics.ExecuteCommandBuffer(owner.fluidSimGpuProfileCmd);
                owner.fluidSimGpuProfileCmd.Clear();
            }
        }
    }
}
