using UnityEngine;
using UnityEngine.Rendering;
using Unity.Profiling;

/// <summary>
/// Fluid visualization: camera callback, materials, and fullscreen composite. Keeps <see cref="FluidSimulator"/> focused on simulation parameters and GPU solve.
/// </summary>
[RequireComponent(typeof(FluidSimulator))]
[DefaultExecutionOrder(50)]
public class FluidRenderer : MonoBehaviour
{
    FluidSimulator _sim;

    [Header("Rendering")]
    public RenderingMode renderingMode = RenderingMode.Thickness;
    public ThicknessSource thicknessSource = ThicknessSource.Nodes;
    public Light mainLight;

    public ComputeShader blurCompute;
    public Material particlesMaterial;
    public Material particleDepthMaterial;
    public Material nodeThicknessMaterial;
    public Material nodeWireframeMaterial;
    public Material uniformGridWireframeMaterial;
    public Material voxelWireframeMaterial;
    public Material particleThicknessMaterial;
    public Material normalMaterial;
    public Material compositeMaterial;
    public Material debugDisplayMaterial;

    public Color fluidColor = new Color(0.2f, 0.6f, 1.0f);
    public bool useSkybox = true;
    public Cubemap skyboxTexture;
    public Color skyHorizonColor = new Color(0.0f, 0.0f, 0.0f);
    public Color skyTopColor = new Color(1.0f, 1.0f, 1.0f);
    [Range(-20.0f, 20.0f)] public float sunIntensity = -2.3f;
    [Range(0.0f, 500.0f)] public float sunSharpness = 53.0f;

    private float calculatedDepthDisplayScale;
    private float calculatedDepthMinValue;
    private float calculatedMinDepth;
    private float calculatedMaxDepth;

    [Range(-20.0f, 20.0f)] public float thicknessScaleNodes = -8.9f;
    [Range(-20.0f, 20.0f)] public float thicknessScaleParticles = -9.6f;

    [Range(0, 100)] public int depthBlurRadius = 6;
    [Range(0.0001f, 10.0f)] public float depthBlurThreshold = 1.24f;
    [Range(0, 100)] public int thicknessBlurRadius = 22;
    [Range(0.0001f, 1.0f)] public float particleRadius = 0.066f;
    [Range(0.25f, 4f)] public float particleDepthShadeGamma = 1.2f;
    [Range(0f, 0.6f)] public float particleDepthLumaFloor = 0.15f;
    public float particleVelocityColorReference = 10f;
    public bool particleVelocityNormalizePerFrame;
    [Range(1, 120)] public int particleVelocityNormalizeSmoothingFrames = 10;
    public ComputeShader particleVelocityReduceShader;
    [Range(0.0001f, 1.0f)] public float depthRadius = 0.279f;
    [Range(0.0001f, 1.0f)] public float thicknessRadius = 0.776f;
    [Range(0, 20)] public float absorptionStrength = 5.8f;
    [Range(0.0f, 1.0f)] public float depthOfFieldStrength = 0.076f;
    [Range(0.0f, 1.0f)] public float reflectionStrength = 0.633f;
    [Range(0.0f, 1.0f)] public float reflectionTint = 0.418f;
    [Range(0.0f, 1.0f)] public float fresnelClamp = 0.8f;
    [Range(0, 2)] public float refractionScale = 0.57f;

    static readonly ProfilerMarker s_RenderMarker = new ProfilerMarker("FluidSim.Render");
    static readonly ProfilerMarker s_RenderThicknessMarker = new ProfilerMarker("FluidSim.Render.Thickness");
    static readonly ProfilerMarker s_ThicknessBlurMarker = new ProfilerMarker("FluidSim.Render.ThicknessBlur");
    static readonly ProfilerMarker s_RenderNodesMarker = new ProfilerMarker("FluidSim.Render.Nodes");
    static readonly ProfilerMarker s_RenderUniformGridNodesMarker = new ProfilerMarker("FluidSim.Render.UniformGridNodes");
    static readonly ProfilerMarker s_RenderColliderVoxelsMarker = new ProfilerMarker("FluidSim.Render.ColliderVoxels");
    static readonly ProfilerMarker s_DrawParticlesMarker = new ProfilerMarker("FluidSim.Render.DrawParticles");
    static readonly ProfilerMarker s_DepthBlurMarker = new ProfilerMarker("FluidSim.Render.DepthBlur");
    static readonly ProfilerMarker s_NormalsMarker = new ProfilerMarker("FluidSim.Render.Normals");
    static readonly ProfilerMarker s_CompositeMarker = new ProfilerMarker("FluidSim.Render.Composite");

    Mesh quadMesh;
    ComputeBuffer argsBuffer;
    ComputeBuffer particleArgsBuffer;
    ComputeBuffer voxelArgsBuffer;
    int cachedVoxelArgsResolution = -1;
    static bool s_warnedVoxelInstanceBudget;
    static bool s_warnedVoxelBufferMissing;
    /// <summary>Thickness blit scale: node-based vs particle-based thickness pass.</summary>
    float ThicknessBlitDisplayScale =>
        thicknessSource == ThicknessSource.Nodes ? thicknessScaleNodes : thicknessScaleParticles;
    int particleVelocityBlockReduceKernel = -1;
    int particleVelocityReduceLevelKernel = -1;
    int particleVelocitySmoothMinMaxKernel = -1;
    ComputeBuffer particleVelocityReduceBufferA;
    ComputeBuffer particleVelocityReduceBufferB;
    ComputeBuffer particleVelocitySmoothedMinMaxBuffer;
    ComputeBuffer particleVelocityMinMaxDummyBuffer;
    CommandBuffer thicknessCmd;
    CommandBuffer depthCmd;
    readonly uint[] indirectDrawArgsScratch = new uint[5];
    readonly uint[] indirectInstanceCountOnlyScratch = new uint[1];
    readonly float[] float2SetDataScratch = new float[2];
    RenderTexture thicknessTexture;
    RenderTexture fluidDepthTexture;
    RenderTexture rawDepthTexture;
    RenderTexture fluidNormalTexture;
    RenderTexture fluidThicknessTexture;
    RenderTexture nodesTexture;

    System.Diagnostics.Stopwatch renderSw = new System.Diagnostics.Stopwatch();

    void Awake()
    {
        _sim = GetComponent<FluidSimulator>();
        if (_sim == null)
            _sim = GetComponentInParent<FluidSimulator>();
    }

    void OnEnable()
    {
        if (thicknessCmd == null)
        {
            thicknessCmd = new CommandBuffer();
            thicknessCmd.name = "Node Thickness Render";
        }
        if (depthCmd == null)
        {
            depthCmd = new CommandBuffer();
            depthCmd.name = "Particle Depth Render";
        }
    }

    void Start()
    {
        LoadDefaultRenderAssetsIfNeeded();
        ResolveMainLightIfNeeded();
        if (_sim == null)
        {
            Debug.LogError("FluidRenderer requires a FluidSimulator on the same GameObject or parent.");
            return;
        }
        if (particleVelocityReduceShader != null)
        {
            particleVelocityBlockReduceKernel = particleVelocityReduceShader.FindKernel("BlockReduce");
            particleVelocityReduceLevelKernel = particleVelocityReduceShader.FindKernel("ReduceLevel");
            particleVelocitySmoothMinMaxKernel = particleVelocityReduceShader.FindKernel("SmoothVelocityMinMax");
        }
        RenderPipelineManager.endCameraRendering += OnEndCameraRendering;
    }

    void LoadDefaultRenderAssetsIfNeeded()
    {
#if UNITY_EDITOR
        if (blurCompute == null)
            blurCompute = UnityEditor.AssetDatabase.LoadAssetAtPath<ComputeShader>("Assets/Scripts/BilateralBlur.compute");
        if (particleVelocityReduceShader == null)
            particleVelocityReduceShader = UnityEditor.AssetDatabase.LoadAssetAtPath<ComputeShader>("Assets/Scripts/ParticleVelocityReduce.compute");
#endif
        EnsureMaterial(ref particlesMaterial, ParticlesPointsShaderName);
        EnsureMaterial(ref particleDepthMaterial, "Custom/ParticleDepth");
        EnsureMaterial(ref nodeThicknessMaterial, "Custom/NodeThickness");
        EnsureMaterial(ref nodeWireframeMaterial, "Custom/NodeWireframe");
        EnsureMaterial(ref uniformGridWireframeMaterial, "Custom/UniformGridWireframe");
        EnsureMaterial(ref voxelWireframeMaterial, "Custom/VoxelWireframe");
        EnsureMaterial(ref particleThicknessMaterial, "Custom/ParticleThickness");
        EnsureMaterial(ref normalMaterial, "Custom/NormalsFromDepth");
        EnsureMaterial(ref compositeMaterial, "Custom/FluidRender");
        EnsureMaterial(ref debugDisplayMaterial, "Custom/DebugTextureDisplay");
    }

    static void EnsureMaterial(ref Material mat, string shaderName)
    {
        if (mat != null)
            return;
        var sh = Shader.Find(shaderName);
        if (sh != null)
            mat = new Material(sh);
    }

    const string ParticlesPointsShaderName = "Custom/ParticlesPoints";

    /// <summary>
    /// Metal requires every <see cref="StructuredBuffer{T}"/> declared in the shader to be bound for each draw.
    /// <c>Custom/ParticlesPoints</c> always declares <c>_ParticleVelocityMinMax</c> (slot 1); bind a dummy before any path
    /// that might use this shader (including if <see cref="particleDepthMaterial"/> was reassigned to ParticlesPoints in the Inspector).
    /// </summary>
    void EnsureParticleVelocityMinMaxDummyBuffer()
    {
        if (particleVelocityMinMaxDummyBuffer != null)
            return;
        particleVelocityMinMaxDummyBuffer = new ComputeBuffer(1, sizeof(float) * 2);
        float2SetDataScratch[0] = 0f;
        float2SetDataScratch[1] = 1f;
        particleVelocityMinMaxDummyBuffer.SetData(float2SetDataScratch);
    }

    void BindParticlesPointsMinMaxBufferIfNeeded(Material m)
    {
        if (m == null || m.shader == null || m.shader.name != ParticlesPointsShaderName)
            return;
        EnsureParticleVelocityMinMaxDummyBuffer();
        m.SetBuffer("_ParticleVelocityMinMax", particleVelocityMinMaxDummyBuffer);
    }

    void ResolveMainLightIfNeeded()
    {
        if (mainLight != null)
            return;
        var lights = FindObjectsByType<Light>(FindObjectsSortMode.None);
        foreach (var L in lights)
        {
            if (L.type == LightType.Directional)
            {
                mainLight = L;
                return;
            }
        }
        if (lights.Length > 0)
            mainLight = lights[0];
    }

    void OnDisable()
    {
        RenderPipelineManager.endCameraRendering -= OnEndCameraRendering;
        if (argsBuffer != null)
        {
            argsBuffer.Release();
            argsBuffer = null;
        }
        if (particleArgsBuffer != null)
        {
            particleArgsBuffer.Release();
            particleArgsBuffer = null;
        }
        if (voxelArgsBuffer != null)
        {
            voxelArgsBuffer.Release();
            voxelArgsBuffer = null;
        }
        cachedVoxelArgsResolution = -1;
        if (thicknessCmd != null)
        {
            thicknessCmd.Release();
            thicknessCmd = null;
        }
        if (depthCmd != null)
        {
            depthCmd.Release();
            depthCmd = null;
        }
        if (thicknessTexture != null)
        {
            thicknessTexture.Release();
            thicknessTexture = null;
        }
        if (rawDepthTexture != null)
        {
            rawDepthTexture.Release();
            rawDepthTexture = null;
        }
        if (nodesTexture != null)
        {
            nodesTexture.Release();
            nodesTexture = null;
        }
        particleVelocityReduceBufferA?.Release();
        particleVelocityReduceBufferA = null;
        particleVelocityReduceBufferB?.Release();
        particleVelocityReduceBufferB = null;
        particleVelocityMinMaxDummyBuffer?.Release();
        particleVelocityMinMaxDummyBuffer = null;
        particleVelocitySmoothedMinMaxBuffer?.Release();
        particleVelocitySmoothedMinMaxBuffer = null;
        if (fluidDepthTexture != null)
        {
            fluidDepthTexture.Release();
            fluidDepthTexture = null;
        }
        if (fluidNormalTexture != null)
        {
            fluidNormalTexture.Release();
            fluidNormalTexture = null;
        }
        if (fluidThicknessTexture != null)
        {
            fluidThicknessTexture.Release();
            fluidThicknessTexture = null;
        }
    }

    Bounds SimBounds
    {
        get
        {
            if (_sim == null)
                return new Bounds(Vector3.zero, Vector3.one);
            return _sim.WorldSimulationBounds;
        }
    }

    static bool IsNodeOrColliderVoxelView(RenderingMode mode) =>
        mode == RenderingMode.Nodes || mode == RenderingMode.UniformGridNodes || mode == RenderingMode.ColliderVoxels;

    /// <returns>False if min/max was not written (caller should keep dummy buffer + _VelocityAutoNormalize 0).</returns>
    bool DispatchParticleVelocityMinMaxForRendering(Material particleMat)
    {
        if (particleVelocityReduceShader == null
            || particleVelocityBlockReduceKernel < 0
            || _sim.ParticlesBuffer == null
            || _sim.NumParticlesForRender <= 0
            || particleMat == null)
            return false;

        int smoothFrames = Mathf.Max(1, particleVelocityNormalizeSmoothingFrames);
        int numParticles = _sim.NumParticlesForRender;

        int blocks = (numParticles + 255) / 256;
        int cap = Mathf.NextPowerOfTwo(Mathf.Max(blocks, 512));
        if (particleVelocityReduceBufferA == null || particleVelocityReduceBufferA.count < cap)
        {
            particleVelocityReduceBufferA?.Release();
            particleVelocityReduceBufferB?.Release();
            particleVelocityReduceBufferA = new ComputeBuffer(cap, sizeof(float) * 2);
            particleVelocityReduceBufferB = new ComputeBuffer(cap, sizeof(float) * 2);
        }

        particleVelocityReduceShader.SetBuffer(particleVelocityBlockReduceKernel, "_Particles", _sim.ParticlesBuffer);
        particleVelocityReduceShader.SetInt("_ParticleCount", numParticles);
        particleVelocityReduceShader.SetBuffer(particleVelocityBlockReduceKernel, "_BlockMinMaxOut", particleVelocityReduceBufferA);
        particleVelocityReduceShader.Dispatch(particleVelocityBlockReduceKernel, blocks, 1, 1);

        bool dataInA = true;
        int count = blocks;
        while (count > 1)
        {
            int outCount = (count + 255) / 256;
            ComputeBuffer src = dataInA ? particleVelocityReduceBufferA : particleVelocityReduceBufferB;
            ComputeBuffer dst = dataInA ? particleVelocityReduceBufferB : particleVelocityReduceBufferA;
            particleVelocityReduceShader.SetBuffer(particleVelocityReduceLevelKernel, "_LevelIn", src);
            particleVelocityReduceShader.SetBuffer(particleVelocityReduceLevelKernel, "_LevelOut", dst);
            particleVelocityReduceShader.SetInt("_LevelCount", count);
            particleVelocityReduceShader.Dispatch(particleVelocityReduceLevelKernel, outCount, 1, 1);
            dataInA = !dataInA;
            count = outCount;
        }

        ComputeBuffer minMaxSrc = dataInA ? particleVelocityReduceBufferA : particleVelocityReduceBufferB;

        if (smoothFrames <= 1 || particleVelocitySmoothMinMaxKernel < 0)
        {
            particleMat.SetBuffer("_ParticleVelocityMinMax", minMaxSrc);
            return true;
        }

        if (particleVelocitySmoothedMinMaxBuffer == null)
        {
            particleVelocitySmoothedMinMaxBuffer = new ComputeBuffer(1, sizeof(float) * 2);
            float2SetDataScratch[0] = float.MaxValue;
            float2SetDataScratch[1] = float.MinValue;
            particleVelocitySmoothedMinMaxBuffer.SetData(float2SetDataScratch);
        }

        float alpha = 1f / smoothFrames;
        particleVelocityReduceShader.SetBuffer(particleVelocitySmoothMinMaxKernel, "_RawMinMax", minMaxSrc);
        particleVelocityReduceShader.SetBuffer(particleVelocitySmoothMinMaxKernel, "_SmoothedMinMax", particleVelocitySmoothedMinMaxBuffer);
        particleVelocityReduceShader.SetFloat("_VelocityMinMaxSmoothAlpha", alpha);
        particleVelocityReduceShader.Dispatch(particleVelocitySmoothMinMaxKernel, 1, 1, 1);
        particleMat.SetBuffer("_ParticleVelocityMinMax", particleVelocitySmoothedMinMaxBuffer);
        return true;
    }

    void OnEndCameraRendering(ScriptableRenderContext ctx, Camera cam)
    {
        if (_sim == null)
            return;

        renderSw.Restart();
        using var _render = s_RenderMarker.Auto();

        if (renderingMode == RenderingMode.Thickness || renderingMode == RenderingMode.BlurredThickness || renderingMode == RenderingMode.Composite)
        {
            using (s_RenderThicknessMarker.Auto()) { RenderThickness(cam); }

            if (renderingMode == RenderingMode.Composite || renderingMode == RenderingMode.BlurredThickness)
            {
                if (fluidThicknessTexture == null || fluidThicknessTexture.width != cam.pixelWidth)
                {
                    if (fluidThicknessTexture != null) fluidThicknessTexture.Release();
                    RenderTextureDescriptor desc = new RenderTextureDescriptor(cam.pixelWidth, cam.pixelHeight, RenderTextureFormat.RFloat, 0);
                    desc.enableRandomWrite = true;
                    fluidThicknessTexture = new RenderTexture(desc);
                    fluidThicknessTexture.Create();
                }

                using (s_ThicknessBlurMarker.Auto()) { RunThicknessBlur(cam, thicknessTexture, fluidThicknessTexture); }
            }
        }

        if (renderingMode == RenderingMode.Nodes)
        {
            using (s_RenderNodesMarker.Auto()) { RenderNodes(cam); }
        }
        if (renderingMode == RenderingMode.UniformGridNodes)
        {
            using (s_RenderUniformGridNodesMarker.Auto()) { RenderUniformGridNodes(cam); }
        }
        if (renderingMode == RenderingMode.ColliderVoxels)
        {
            using (s_RenderColliderVoxelsMarker.Auto()) { RenderColliderVoxels(cam); }
        }

        // Particles / ParticlesVelocity: URP has already drawn opaque scene geometry (colliders, bunny, etc.) into the
        // camera target. DrawParticles only adds particle splats on top with depth test — no fullscreen blit, unlike
        // Thickness, Nodes, ColliderVoxels, Depth, Normal, or Composite (those replace the framebuffer).
        if (renderingMode != RenderingMode.Thickness && renderingMode != RenderingMode.BlurredThickness && !IsNodeOrColliderVoxelView(renderingMode))
        {
            using (s_DrawParticlesMarker.Auto()) { DrawParticles(cam, ctx); }
        }

        if (renderingMode == RenderingMode.BlurredDepth || renderingMode == RenderingMode.Normal || renderingMode == RenderingMode.Composite)
        {
            if (fluidDepthTexture == null || fluidDepthTexture.width != cam.pixelWidth || fluidDepthTexture.height != cam.pixelHeight)
            {
                if (fluidDepthTexture != null) fluidDepthTexture.Release();
                RenderTextureDescriptor desc = new RenderTextureDescriptor(cam.pixelWidth, cam.pixelHeight, RenderTextureFormat.RFloat, 0);
                desc.enableRandomWrite = true;
                fluidDepthTexture = new RenderTexture(desc);
                fluidDepthTexture.Create();
            }

            using (s_DepthBlurMarker.Auto()) { RunBilateralBlur(cam, rawDepthTexture, fluidDepthTexture); }
            using (s_NormalsMarker.Auto()) { RenderNormals(cam); }
        }

        if (renderingMode == RenderingMode.Nodes || renderingMode == RenderingMode.UniformGridNodes || renderingMode == RenderingMode.ColliderVoxels)
        {
            Graphics.Blit(nodesTexture, (RenderTexture)null);
        }
        else if (renderingMode == RenderingMode.Thickness)
        {
            DebugBlit(thicknessTexture, ThicknessBlitDisplayScale, 0.0f, false);
        }
        else if (renderingMode == RenderingMode.BlurredThickness)
        {
            DebugBlit(fluidThicknessTexture, ThicknessBlitDisplayScale, 0.0f, false);
        }
        else if (renderingMode == RenderingMode.Depth)
        {
            if (cam != null && SimBounds.size.sqrMagnitude > 1e-6f)
            {
                Vector3 cameraPos = cam.transform.position;
                Bounds bounds = SimBounds;
                float boundsDiagonal = bounds.size.magnitude;
                CalculateDepthParameters(cameraPos, bounds, boundsDiagonal);
            }
            DebugBlit(rawDepthTexture, calculatedDepthDisplayScale, calculatedDepthMinValue);
        }
        else if (renderingMode == RenderingMode.BlurredDepth)
        {
            if (cam != null && SimBounds.size.sqrMagnitude > 1e-6f)
            {
                Vector3 cameraPos = cam.transform.position;
                Bounds bounds = SimBounds;
                float boundsDiagonal = bounds.size.magnitude;
                CalculateDepthParameters(cameraPos, bounds, boundsDiagonal);
            }
            DebugBlit(fluidDepthTexture, calculatedDepthDisplayScale, calculatedDepthMinValue);
        }
        else if (renderingMode == RenderingMode.Normal)
        {
            Graphics.Blit(fluidNormalTexture, (RenderTexture)null);
        }
        else if (renderingMode == RenderingMode.Composite)
        {
            using (s_CompositeMarker.Auto()) { RenderComposite(cam); }
        }

        renderSw.Stop();
        _sim.SetLastRenderTimeMs(renderSw.Elapsed.TotalMilliseconds);
    }

    void DrawParticles(Camera cam, ScriptableRenderContext ctx = default)
    {
        if (_sim.ParticlesBuffer == null || _sim.NumParticlesForRender <= 0) return;
        if (cam == null) return;

        Material currentMaterial = null;
        bool useBlur = false;

        switch (renderingMode)
        {
            case RenderingMode.Particles:
            case RenderingMode.ParticlesVelocity:
                currentMaterial = particlesMaterial;
                break;
            case RenderingMode.Depth:
                currentMaterial = particleDepthMaterial;
                break;
            case RenderingMode.BlurredDepth:
            case RenderingMode.Normal:
            case RenderingMode.Composite:
                currentMaterial = particleDepthMaterial;
                useBlur = true;
                break;
        }

        if (currentMaterial == null) return;

        if (quadMesh == null)
            CreateQuadMesh();

        if (particleArgsBuffer == null || particleArgsBuffer.count != 5)
            CreateParticleArgsBuffer();
        else
            PatchIndirectDrawInstanceCount(particleArgsBuffer, (uint)_sim.NumParticlesForRender);

        currentMaterial.SetBuffer("_Particles", _sim.ParticlesBuffer);
        BindParticlesPointsMinMaxBufferIfNeeded(currentMaterial);

        int numParticles = _sim.NumParticlesForRender;

        if (renderingMode == RenderingMode.Particles || renderingMode == RenderingMode.ParticlesVelocity)
            currentMaterial.SetFloat("_Radius", particleRadius);
        else if (renderingMode == RenderingMode.Depth || renderingMode == RenderingMode.BlurredDepth || renderingMode == RenderingMode.Normal || renderingMode == RenderingMode.Composite)
            currentMaterial.SetFloat("_Radius", depthRadius);

        currentMaterial.SetVector("_SimulationBoundsMin", SimBounds.min);
        currentMaterial.SetVector("_SimulationBoundsMax", SimBounds.max);

        Vector3 cameraPos = cam.transform.position;
        Bounds bounds = SimBounds;
        Vector3 boundsCenter = bounds.center;
        Vector3 boundsSize = bounds.size;
        Vector3 nearestPoint = new Vector3(
            Mathf.Clamp(cameraPos.x, bounds.min.x, bounds.max.x),
            Mathf.Clamp(cameraPos.y, bounds.min.y, bounds.max.y),
            Mathf.Clamp(cameraPos.z, bounds.min.z, bounds.max.z)
        );
        float distanceToNearest = Vector3.Distance(cameraPos, nearestPoint);
        float distanceToCenter = Vector3.Distance(cameraPos, boundsCenter);
        float boundsDiagonal = boundsSize.magnitude;

        CalculateDepthParameters(cameraPos, bounds, boundsDiagonal);

        if (renderingMode == RenderingMode.Particles || renderingMode == RenderingMode.ParticlesVelocity)
        {
            // Dummy min/max already bound in BindParticlesPointsMinMaxBufferIfNeeded; Dispatch may replace with GPU-reduced buffer.
            float calculatedMinValue = Mathf.Exp(calculatedDepthMinValue);
            float calculatedScale = Mathf.Exp(calculatedDepthDisplayScale);
            currentMaterial.SetFloat("_Scale", calculatedScale);
            currentMaterial.SetFloat("_MinValue", calculatedMinValue);
            currentMaterial.SetFloat("_DepthShadeGamma", Mathf.Max(0.01f, particleDepthShadeGamma));
            currentMaterial.SetFloat("_DepthLumaFloor", Mathf.Clamp01(particleDepthLumaFloor));
            currentMaterial.SetFloat("_UseVelocityColor", renderingMode == RenderingMode.ParticlesVelocity ? 1f : 0f);
            currentMaterial.SetFloat("_VelocityReferenceSpeed", Mathf.Max(particleVelocityColorReference, 1e-4f));
            bool autoNorm = renderingMode == RenderingMode.ParticlesVelocity
                && particleVelocityNormalizePerFrame
                && particleVelocityReduceShader != null
                && particleVelocityBlockReduceKernel >= 0;
            if (autoNorm)
            {
                bool minMaxOk = DispatchParticleVelocityMinMaxForRendering(currentMaterial);
                currentMaterial.SetFloat("_VelocityAutoNormalize", minMaxOk ? 1f : 0f);
            }
            else
            {
                currentMaterial.SetFloat("_VelocityAutoNormalize", 0f);
            }
        }

        if (renderingMode != RenderingMode.Particles && renderingMode != RenderingMode.ParticlesVelocity)
        {
            float fadeStart = Mathf.Max(0.0f, distanceToNearest - boundsDiagonal * 0.3f);
            float fadeEnd = distanceToCenter + boundsDiagonal * 0.8f;
            if (fadeEnd <= fadeStart)
                fadeEnd = fadeStart + boundsDiagonal * 0.5f;
            currentMaterial.SetFloat("_DepthFadeStart", fadeStart);
            currentMaterial.SetFloat("_DepthFadeEnd", fadeEnd);
        }

        if (renderingMode == RenderingMode.Depth || renderingMode == RenderingMode.BlurredDepth || renderingMode == RenderingMode.Normal || renderingMode == RenderingMode.Composite)
        {
            currentMaterial.SetFloat("_DepthMin", calculatedMinDepth);
            currentMaterial.SetFloat("_DepthMax", calculatedMaxDepth);
        }

        if (useBlur && blurCompute != null && renderingMode != RenderingMode.Composite)
        {
            RenderBlurredDepth(cam, currentMaterial);
        }
        else
        {
            if (renderingMode == RenderingMode.Depth ||
                renderingMode == RenderingMode.BlurredDepth ||
                renderingMode == RenderingMode.Normal ||
                renderingMode == RenderingMode.Composite)
            {
                if (rawDepthTexture == null || rawDepthTexture.width != cam.pixelWidth || rawDepthTexture.height != cam.pixelHeight)
                {
                    if (rawDepthTexture != null)
                        rawDepthTexture.Release();
                    rawDepthTexture = new RenderTexture(cam.pixelWidth, cam.pixelHeight, 24, RenderTextureFormat.RFloat);
                    rawDepthTexture.enableRandomWrite = true;
                    rawDepthTexture.Create();
                }

                depthCmd.Clear();
                depthCmd.SetRenderTarget(rawDepthTexture);
                depthCmd.ClearRenderTarget(true, true, new Color(10000.0f, 10000.0f, 10000.0f, 1.0f));
                depthCmd.DrawMeshInstancedIndirect(quadMesh, 0, currentMaterial, 0, particleArgsBuffer);
                Graphics.ExecuteCommandBuffer(depthCmd);

                if (renderingMode == RenderingMode.Depth)
                    DebugBlit(rawDepthTexture, calculatedDepthDisplayScale, calculatedDepthMinValue);
            }
            else
            {
                if (_sim.ParticlesBuffer == null || numParticles <= 0) return;

                if (particleArgsBuffer == null || particleArgsBuffer.count != 5)
                    CreateParticleArgsBuffer();
                else
                    PatchIndirectDrawInstanceCount(particleArgsBuffer, (uint)numParticles);

                depthCmd.Clear();
                depthCmd.SetRenderTarget(BuiltinRenderTextureType.CameraTarget);
                depthCmd.DrawMeshInstancedIndirect(quadMesh, 0, currentMaterial, 0, particleArgsBuffer);
                Graphics.ExecuteCommandBuffer(depthCmd);
            }
        }
    }

    void RenderThickness(Camera cam)
    {
        if (cam == null) return;

        if (quadMesh == null)
            CreateQuadMesh();

        if (thicknessTexture == null || thicknessTexture.width != cam.pixelWidth || thicknessTexture.height != cam.pixelHeight)
        {
            if (thicknessTexture != null)
                thicknessTexture.Release();
            thicknessTexture = new RenderTexture(cam.pixelWidth, cam.pixelHeight, 0, RenderTextureFormat.RFloat);
            thicknessTexture.enableRandomWrite = true;
            thicknessTexture.Create();
        }

        thicknessCmd.Clear();
        thicknessCmd.SetRenderTarget(thicknessTexture);
        thicknessCmd.ClearRenderTarget(true, true, Color.black);

        if (thicknessSource == ThicknessSource.Nodes)
        {
            if (_sim.NodesBuffer == null || _sim.NumNodes <= 0) return;
            if (nodeThicknessMaterial == null) return;

            if (argsBuffer == null || argsBuffer.count != 5)
                CreateArgsBuffer();
            else
                PatchIndirectDrawInstanceCount(argsBuffer, (uint)_sim.NumNodes);

            nodeThicknessMaterial.SetBuffer("_Nodes", _sim.NodesBuffer);
            nodeThicknessMaterial.SetVector("_SimulationBoundsMin", SimBounds.min);
            nodeThicknessMaterial.SetVector("_SimulationBoundsMax", SimBounds.max);
            nodeThicknessMaterial.SetFloat("_PointSize", _sim.MaxDetailCellSize);
            nodeThicknessMaterial.SetColor("_Color", new Color(0.0f, 0.5f, 1.0f, 1.0f));

            thicknessCmd.DrawMeshInstancedIndirect(quadMesh, 0, nodeThicknessMaterial, 0, argsBuffer);
        }
        else if (thicknessSource == ThicknessSource.Particles)
        {
            if (_sim.ParticlesBuffer == null || _sim.NumParticlesForRender <= 0) return;
            if (particleThicknessMaterial == null) return;

            if (particleArgsBuffer == null || particleArgsBuffer.count != 5)
                CreateParticleArgsBuffer();
            else
                PatchIndirectDrawInstanceCount(particleArgsBuffer, (uint)_sim.NumParticlesForRender);

            particleThicknessMaterial.SetBuffer("_Particles", _sim.ParticlesBuffer);
            particleThicknessMaterial.SetVector("_SimulationBoundsMin", SimBounds.min);
            particleThicknessMaterial.SetVector("_SimulationBoundsMax", SimBounds.max);
            particleThicknessMaterial.SetFloat("_Radius", thicknessRadius);

            thicknessCmd.DrawMeshInstancedIndirect(quadMesh, 0, particleThicknessMaterial, 0, particleArgsBuffer);
        }

        Graphics.ExecuteCommandBuffer(thicknessCmd);
    }

    void RenderNodes(Camera cam)
    {
        if (cam == null) return;

        if (quadMesh == null)
            CreateQuadMesh();

        if (nodesTexture == null || nodesTexture.width != cam.pixelWidth || nodesTexture.height != cam.pixelHeight)
        {
            if (nodesTexture != null)
                nodesTexture.Release();
            nodesTexture = new RenderTexture(cam.pixelWidth, cam.pixelHeight, 24, RenderTextureFormat.ARGBFloat);
            nodesTexture.enableRandomWrite = true;
            nodesTexture.Create();
        }

        thicknessCmd.Clear();
        thicknessCmd.SetRenderTarget(nodesTexture);
        thicknessCmd.ClearRenderTarget(true, true, Color.clear);

        if (_sim.NodesBuffer == null || _sim.NumNodes <= 0) return;
        if (nodeWireframeMaterial == null) return;

        if (argsBuffer == null || argsBuffer.count != 5)
            CreateArgsBuffer();
        else
            PatchIndirectDrawInstanceCount(argsBuffer, (uint)_sim.NumNodes);

        nodeWireframeMaterial.SetBuffer("_Nodes", _sim.NodesBuffer);
        nodeWireframeMaterial.SetVector("_SimulationBoundsMin", SimBounds.min);
        nodeWireframeMaterial.SetVector("_SimulationBoundsMax", SimBounds.max);
        nodeWireframeMaterial.SetFloat("_PointSize", _sim.MaxDetailCellSize);

        if (cam != null && SimBounds.size.sqrMagnitude > 1e-6f)
        {
            Vector3 cameraPos = cam.transform.position;
            Bounds bounds = SimBounds;
            float boundsDiagonal = bounds.size.magnitude;
            CalculateDepthParameters(cameraPos, bounds, boundsDiagonal);
            float calculatedMinValue = Mathf.Exp(calculatedDepthMinValue);
            float calculatedScale = Mathf.Exp(calculatedDepthDisplayScale);
            nodeWireframeMaterial.SetFloat("_Scale", calculatedScale);
            nodeWireframeMaterial.SetFloat("_MinValue", calculatedMinValue);
        }

        thicknessCmd.DrawMeshInstancedIndirect(quadMesh, 0, nodeWireframeMaterial, 0, argsBuffer);
        Graphics.ExecuteCommandBuffer(thicknessCmd);
    }

    static bool s_warnedUniformGridNodesWrongMode;
    static bool s_warnedUniformGridNodesBuffers;

    void RenderUniformGridNodes(Camera cam)
    {
        if (cam == null) return;

        if (quadMesh == null)
            CreateQuadMesh();

        if (nodesTexture == null || nodesTexture.width != cam.pixelWidth || nodesTexture.height != cam.pixelHeight)
        {
            if (nodesTexture != null)
                nodesTexture.Release();
            nodesTexture = new RenderTexture(cam.pixelWidth, cam.pixelHeight, 24, RenderTextureFormat.ARGBFloat);
            nodesTexture.enableRandomWrite = true;
            nodesTexture.Create();
        }

        thicknessCmd.Clear();
        thicknessCmd.SetRenderTarget(nodesTexture);
        thicknessCmd.ClearRenderTarget(true, true, Color.clear);

        if (_sim.gridMode != GridMode.Uniform)
        {
            if (!s_warnedUniformGridNodesWrongMode)
            {
                s_warnedUniformGridNodesWrongMode = true;
                Debug.LogWarning(
                    "FluidRenderer: RenderingMode.UniformGridNodes is meant for FluidSimulator.GridMode.Uniform (octree leaves use RenderingMode.Nodes).");
            }
            Graphics.ExecuteCommandBuffer(thicknessCmd);
            return;
        }

        if (_sim.NumNodes <= 0
            || uniformGridWireframeMaterial == null
            || _sim.UniformActiveMortonListBuffer == null
            || _sim.UniformCellCountsBuffer == null)
        {
            if (!s_warnedUniformGridNodesBuffers && _sim.gridMode == GridMode.Uniform)
            {
                s_warnedUniformGridNodesBuffers = true;
                Debug.LogWarning(
                    "FluidRenderer: UniformGridNodes needs active uniform-grid buffers and NumNodes > 0 (run simulation in Uniform mode).");
            }
            Graphics.ExecuteCommandBuffer(thicknessCmd);
            return;
        }

        if (argsBuffer == null || argsBuffer.count != 5)
            CreateArgsBuffer();
        else
            PatchIndirectDrawInstanceCount(argsBuffer, (uint)_sim.NumNodes);

        uniformGridWireframeMaterial.SetBuffer("_ActiveMortonList", _sim.UniformActiveMortonListBuffer);
        uniformGridWireframeMaterial.SetBuffer("_CellCounts", _sim.UniformCellCountsBuffer);
        uniformGridWireframeMaterial.SetInt("_UniformGridLog2K", _sim.UniformGridBinsPerAxisLog2);
        uniformGridWireframeMaterial.SetInt("_UniformGridCellCount", _sim.UniformGridCellCount);
        uniformGridWireframeMaterial.SetVector("_SimulationBoundsMin", SimBounds.min);
        uniformGridWireframeMaterial.SetVector("_SimulationBoundsMax", SimBounds.max);

        if (cam != null && SimBounds.size.sqrMagnitude > 1e-6f)
        {
            Vector3 cameraPos = cam.transform.position;
            Bounds bounds = SimBounds;
            float boundsDiagonal = bounds.size.magnitude;
            CalculateDepthParameters(cameraPos, bounds, boundsDiagonal);
            float calculatedMinValue = Mathf.Exp(calculatedDepthMinValue);
            float calculatedScale = Mathf.Exp(calculatedDepthDisplayScale);
            uniformGridWireframeMaterial.SetFloat("_Scale", calculatedScale);
            uniformGridWireframeMaterial.SetFloat("_MinValue", calculatedMinValue);
        }

        thicknessCmd.DrawMeshInstancedIndirect(quadMesh, 0, uniformGridWireframeMaterial, 0, argsBuffer);
        Graphics.ExecuteCommandBuffer(thicknessCmd);
    }

    void RenderColliderVoxels(Camera cam)
    {
        if (cam == null || _sim == null || !_sim.useColliders) return;

        if (quadMesh == null)
            CreateQuadMesh();

        int R = _sim.SolidVoxelResolution;
        long voxelCountLong = (long)R * R * R;
        const long maxInstances = 6_000_000;
        if (voxelCountLong > maxInstances)
        {
            if (!s_warnedVoxelInstanceBudget)
            {
                s_warnedVoxelInstanceBudget = true;
                Debug.LogWarning(
                    $"FluidRenderer: ColliderVoxels view skipped — R³={voxelCountLong} exceeds budget ({maxInstances}). Raise minLayer on Fluid Simulator to reduce R (current R={R}).");
            }
            return;
        }

        uint voxelInstances = (uint)voxelCountLong;
        if (_sim.SolidVoxelsBuffer == null || _sim.SolidVoxelsBuffer.count != voxelInstances)
        {
            if (!s_warnedVoxelBufferMissing)
            {
                s_warnedVoxelBufferMissing = true;
                Debug.LogWarning("FluidRenderer: ColliderVoxels needs a baked solid voxel grid (Fluid Simulator → colliders / resolution).");
            }
            return;
        }

        if (voxelWireframeMaterial == null) return;

        if (nodesTexture == null || nodesTexture.width != cam.pixelWidth || nodesTexture.height != cam.pixelHeight)
        {
            if (nodesTexture != null)
                nodesTexture.Release();
            nodesTexture = new RenderTexture(cam.pixelWidth, cam.pixelHeight, 24, RenderTextureFormat.ARGBFloat);
            nodesTexture.enableRandomWrite = true;
            nodesTexture.Create();
        }

        EnsureVoxelIndirectArgs(R, voxelInstances);

        thicknessCmd.Clear();
        thicknessCmd.SetRenderTarget(nodesTexture);
        thicknessCmd.ClearRenderTarget(true, true, Color.clear);

        voxelWireframeMaterial.SetBuffer("_SolidVoxels", _sim.SolidVoxelsBuffer);
        voxelWireframeMaterial.SetInt("_SolidVoxelResolution", R);
        voxelWireframeMaterial.SetVector("_SimulationBoundsMin", SimBounds.min);
        voxelWireframeMaterial.SetVector("_SimulationBoundsMax", SimBounds.max);

        if (SimBounds.size.sqrMagnitude > 1e-6f)
        {
            Vector3 cameraPos = cam.transform.position;
            Bounds bounds = SimBounds;
            float boundsDiagonal = bounds.size.magnitude;
            CalculateDepthParameters(cameraPos, bounds, boundsDiagonal);
            voxelWireframeMaterial.SetFloat("_Scale", Mathf.Exp(calculatedDepthDisplayScale));
            voxelWireframeMaterial.SetFloat("_MinValue", Mathf.Exp(calculatedDepthMinValue));
        }

        thicknessCmd.DrawMeshInstancedIndirect(quadMesh, 0, voxelWireframeMaterial, 0, voxelArgsBuffer);
        Graphics.ExecuteCommandBuffer(thicknessCmd);
    }

    void EnsureVoxelIndirectArgs(int R, uint instanceCount)
    {
        if (quadMesh == null)
            CreateQuadMesh();
        if (voxelArgsBuffer != null && cachedVoxelArgsResolution == R)
            return;

        const int subMeshIndex = 0;
        indirectDrawArgsScratch[0] = (uint)quadMesh.GetIndexCount(subMeshIndex);
        indirectDrawArgsScratch[1] = instanceCount;
        indirectDrawArgsScratch[2] = (uint)quadMesh.GetIndexStart(subMeshIndex);
        indirectDrawArgsScratch[3] = (uint)quadMesh.GetBaseVertex(subMeshIndex);
        indirectDrawArgsScratch[4] = 0;

        voxelArgsBuffer?.Release();
        voxelArgsBuffer = new ComputeBuffer(5, sizeof(uint), ComputeBufferType.IndirectArguments);
        voxelArgsBuffer.SetData(indirectDrawArgsScratch);
        cachedVoxelArgsResolution = R;
    }

    void RenderNormals(Camera cam)
    {
        int width = cam.pixelWidth;
        int height = cam.pixelHeight;

        if (fluidNormalTexture == null || fluidNormalTexture.width != width || fluidNormalTexture.height != height)
        {
            if (fluidNormalTexture != null) fluidNormalTexture.Release();
            RenderTextureDescriptor desc = new RenderTextureDescriptor(width, height, RenderTextureFormat.ARGBHalf, 0);
            fluidNormalTexture = new RenderTexture(desc);
            fluidNormalTexture.Create();
        }

        if (normalMaterial != null && fluidDepthTexture != null)
        {
            normalMaterial.SetMatrix("_CameraInvViewMatrix", cam.cameraToWorldMatrix);
            float fovY = Mathf.Deg2Rad * cam.fieldOfView;
            float aspect = cam.aspect;
            float tanHalfFovY = Mathf.Tan(fovY * 0.5f);
            normalMaterial.SetVector("_CameraParams", new Vector4(aspect * tanHalfFovY, tanHalfFovY, 0, 0));
            normalMaterial.SetTexture("_MainTex", fluidDepthTexture);
            Graphics.Blit(fluidDepthTexture, fluidNormalTexture, normalMaterial);
        }
    }

    void DebugBlit(RenderTexture source, float scaleExponent, float minValueExponent = float.MinValue, bool useExpForMin = true)
    {
        if (debugDisplayMaterial == null || source == null) return;

        Graphics.SetRenderTarget(null);
        float scale = Mathf.Exp(scaleExponent);
        debugDisplayMaterial.SetFloat("_Scale", scale);
        float minValue;
        if (minValueExponent == float.MinValue)
            minValue = 0.0f;
        else if (useExpForMin)
            minValue = Mathf.Exp(minValueExponent);
        else
            minValue = minValueExponent;
        debugDisplayMaterial.SetFloat("_MinValue", minValue);
        Graphics.Blit(source, (RenderTexture)null, debugDisplayMaterial);
    }

    void CalculateDepthParameters(Vector3 cameraPos, Bounds bounds, float boundsDiagonal)
    {
        Vector3 nearestPoint = new Vector3(
            Mathf.Clamp(cameraPos.x, bounds.min.x, bounds.max.x),
            Mathf.Clamp(cameraPos.y, bounds.min.y, bounds.max.y),
            Mathf.Clamp(cameraPos.z, bounds.min.z, bounds.max.z)
        );
        float minDepth = Vector3.Distance(cameraPos, nearestPoint);

        Vector3[] corners = new Vector3[]
        {
            bounds.min,
            new Vector3(bounds.max.x, bounds.min.y, bounds.min.z),
            new Vector3(bounds.min.x, bounds.max.y, bounds.min.z),
            new Vector3(bounds.max.x, bounds.max.y, bounds.min.z),
            new Vector3(bounds.min.x, bounds.min.y, bounds.max.z),
            new Vector3(bounds.max.x, bounds.min.y, bounds.max.z),
            new Vector3(bounds.min.x, bounds.max.y, bounds.max.z),
            bounds.max
        };

        float maxDepth = 0.0f;
        foreach (Vector3 corner in corners)
        {
            float distance = Vector3.Distance(cameraPos, corner);
            maxDepth = Mathf.Max(maxDepth, distance);
        }

        if (maxDepth <= minDepth)
            maxDepth = minDepth + boundsDiagonal * 0.5f;

        calculatedMinDepth = minDepth;
        calculatedMaxDepth = maxDepth;

        float calculatedMinValue = Mathf.Max(0.001f, minDepth);
        float depthRange = maxDepth - minDepth;
        float calculatedScale = depthRange > 0.001f ? (1.0f / depthRange) : 1.0f;

        calculatedDepthMinValue = Mathf.Log(calculatedMinValue);
        calculatedDepthDisplayScale = Mathf.Log(calculatedScale);
    }

    void CreateQuadMesh()
    {
        Vector3[] vertices = new Vector3[4]
        {
            new Vector3(-0.5f, -0.5f, 0),
            new Vector3(0.5f, -0.5f, 0),
            new Vector3(-0.5f, 0.5f, 0),
            new Vector3(0.5f, 0.5f, 0)
        };

        int[] triangles = new int[6] { 0, 2, 1, 2, 3, 1 };

        Vector2[] uv = new Vector2[4]
        {
            new Vector2(0, 0),
            new Vector2(1, 0),
            new Vector2(0, 1),
            new Vector2(1, 1)
        };

        quadMesh = new Mesh();
        quadMesh.vertices = vertices;
        quadMesh.triangles = triangles;
        quadMesh.uv = uv;
        quadMesh.RecalculateNormals();
    }

    /// <summary>Updates only indirect args element [1] (instance count). Avoids GetData+SetData of the full 5×uint buffer.</summary>
    void PatchIndirectDrawInstanceCount(ComputeBuffer indirectArgs, uint instanceCount)
    {
        indirectInstanceCountOnlyScratch[0] = instanceCount;
        indirectArgs.SetData(indirectInstanceCountOnlyScratch, 0, 1, 1);
    }

    void CreateArgsBuffer()
    {
        if (_sim == null) return;
        if (quadMesh == null) CreateQuadMesh();

        const int stride = sizeof(uint);
        const int numArgs = 5;
        const int subMeshIndex = 0;

        uint[] args = new uint[numArgs];
        args[0] = (uint)quadMesh.GetIndexCount(subMeshIndex);
        args[1] = (uint)_sim.NumNodes;
        args[2] = (uint)quadMesh.GetIndexStart(subMeshIndex);
        args[3] = (uint)quadMesh.GetBaseVertex(subMeshIndex);
        args[4] = 0;

        argsBuffer?.Release();
        argsBuffer = new ComputeBuffer(numArgs, stride, ComputeBufferType.IndirectArguments);
        argsBuffer.SetData(args);
    }

    void CreateParticleArgsBuffer()
    {
        if (_sim == null) return;
        if (quadMesh == null) CreateQuadMesh();

        const int stride = sizeof(uint);
        const int numArgs = 5;
        const int subMeshIndex = 0;

        uint[] args = new uint[numArgs];
        args[0] = (uint)quadMesh.GetIndexCount(subMeshIndex);
        args[1] = (uint)_sim.NumParticlesForRender;
        args[2] = (uint)quadMesh.GetIndexStart(subMeshIndex);
        args[3] = (uint)quadMesh.GetBaseVertex(subMeshIndex);
        args[4] = 0;

        particleArgsBuffer?.Release();
        particleArgsBuffer = new ComputeBuffer(numArgs, stride, ComputeBufferType.IndirectArguments);
        particleArgsBuffer.SetData(args);
    }

    void RenderBlurredDepth(Camera cam, Material mat)
    {
        int width = cam.pixelWidth;
        int height = cam.pixelHeight;

        if (fluidDepthTexture == null || fluidDepthTexture.width != width || fluidDepthTexture.height != height)
        {
            if (fluidDepthTexture != null) fluidDepthTexture.Release();
            RenderTextureDescriptor desc = new RenderTextureDescriptor(width, height, RenderTextureFormat.RFloat, 0);
            desc.enableRandomWrite = true;
            fluidDepthTexture = new RenderTexture(desc);
            fluidDepthTexture.Create();
        }

        RenderTextureDescriptor rawDesc = new RenderTextureDescriptor(width, height, RenderTextureFormat.RFloat, 24);
        rawDesc.enableRandomWrite = false;
        RenderTexture rawDepth = RenderTexture.GetTemporary(rawDesc);

        RenderTextureDescriptor blurDesc = new RenderTextureDescriptor(width, height, RenderTextureFormat.RFloat, 0);
        blurDesc.enableRandomWrite = true;
        RenderTexture tempBlur = RenderTexture.GetTemporary(blurDesc);

        depthCmd.Clear();
        depthCmd.SetRenderTarget(rawDepth);
        depthCmd.ClearRenderTarget(true, true, Color.black);
        depthCmd.DrawMeshInstancedIndirect(quadMesh, 0, mat, 0, particleArgsBuffer);
        Graphics.ExecuteCommandBuffer(depthCmd);

        int kernelHandleH = blurCompute.FindKernel("DepthBlurHorizontal");
        blurCompute.SetTexture(kernelHandleH, "_SourceTexture", rawDepth);
        blurCompute.SetTexture(kernelHandleH, "_DestinationTexture", tempBlur);
        blurCompute.SetFloat("_BlurRadius", depthBlurRadius);
        blurCompute.SetFloat("_BlurDepthFalloff", depthBlurThreshold);
        blurCompute.SetVector("_Resolution", new Vector2(width, height));
        int groupsX = Mathf.CeilToInt(width / 8.0f);
        int groupsY = Mathf.CeilToInt(height / 8.0f);
        blurCompute.Dispatch(kernelHandleH, groupsX, groupsY, 1);

        int kernelHandleV = blurCompute.FindKernel("DepthBlurVertical");
        blurCompute.SetTexture(kernelHandleV, "_SourceTexture", tempBlur);
        blurCompute.SetTexture(kernelHandleV, "_DestinationTexture", fluidDepthTexture);
        blurCompute.SetFloat("_BlurRadius", depthBlurRadius);
        blurCompute.SetFloat("_BlurDepthFalloff", depthBlurThreshold);
        blurCompute.SetVector("_Resolution", new Vector2(width, height));
        blurCompute.Dispatch(kernelHandleV, groupsX, groupsY, 1);

        if (renderingMode == RenderingMode.BlurredDepth)
        {
            if (cam != null && SimBounds.size.sqrMagnitude > 1e-6f)
            {
                Vector3 cameraPos = cam.transform.position;
                Bounds bounds = SimBounds;
                float boundsDiagonal = bounds.size.magnitude;
                CalculateDepthParameters(cameraPos, bounds, boundsDiagonal);
            }
            DebugBlit(fluidDepthTexture, calculatedDepthDisplayScale, calculatedDepthMinValue);
        }

        RenderTexture.ReleaseTemporary(rawDepth);
        RenderTexture.ReleaseTemporary(tempBlur);
    }

    void RunBilateralBlur(Camera cam, RenderTexture source, RenderTexture dest)
    {
        if (blurCompute == null || source == null || dest == null) return;

        int width = cam.pixelWidth;
        int height = cam.pixelHeight;

        RenderTextureDescriptor blurDesc = new RenderTextureDescriptor(width, height, RenderTextureFormat.RFloat, 0);
        blurDesc.enableRandomWrite = true;
        RenderTexture tempBlur = RenderTexture.GetTemporary(blurDesc);

        int kH = blurCompute.FindKernel("DepthBlurHorizontal");
        blurCompute.SetTexture(kH, "_SourceTexture", source);
        blurCompute.SetTexture(kH, "_DestinationTexture", tempBlur);
        blurCompute.SetFloat("_BlurRadius", depthBlurRadius);
        blurCompute.SetFloat("_BlurDepthFalloff", depthBlurThreshold);
        blurCompute.SetVector("_Resolution", new Vector2(width, height));
        blurCompute.Dispatch(kH, Mathf.CeilToInt(width / 8.0f), Mathf.CeilToInt(height / 8.0f), 1);

        int kV = blurCompute.FindKernel("DepthBlurVertical");
        blurCompute.SetTexture(kV, "_SourceTexture", tempBlur);
        blurCompute.SetTexture(kV, "_DestinationTexture", dest);
        blurCompute.SetFloat("_BlurRadius", depthBlurRadius);
        blurCompute.SetFloat("_BlurDepthFalloff", depthBlurThreshold);
        blurCompute.SetVector("_Resolution", new Vector2(width, height));
        blurCompute.Dispatch(kV, Mathf.CeilToInt(width / 8.0f), Mathf.CeilToInt(height / 8.0f), 1);

        RenderTexture.ReleaseTemporary(tempBlur);
    }

    void RunThicknessBlur(Camera cam, RenderTexture source, RenderTexture dest)
    {
        if (blurCompute == null || source == null || dest == null) return;

        int width = cam.pixelWidth;
        int height = cam.pixelHeight;

        RenderTextureDescriptor blurDesc = new RenderTextureDescriptor(width, height, RenderTextureFormat.RFloat, 0);
        blurDesc.enableRandomWrite = true;
        RenderTexture tempBlur = RenderTexture.GetTemporary(blurDesc);

        int kH = blurCompute.FindKernel("ThicknessBlurHorizontal");
        blurCompute.SetTexture(kH, "_SourceTexture", source);
        blurCompute.SetTexture(kH, "_DestinationTexture", tempBlur);
        blurCompute.SetFloat("_BlurRadius", thicknessBlurRadius);
        blurCompute.SetVector("_Resolution", new Vector2(width, height));
        blurCompute.Dispatch(kH, Mathf.CeilToInt(width / 8.0f), Mathf.CeilToInt(height / 8.0f), 1);

        int kV = blurCompute.FindKernel("ThicknessBlurVertical");
        blurCompute.SetTexture(kV, "_SourceTexture", tempBlur);
        blurCompute.SetTexture(kV, "_DestinationTexture", dest);
        blurCompute.SetFloat("_BlurRadius", thicknessBlurRadius);
        blurCompute.SetVector("_Resolution", new Vector2(width, height));
        blurCompute.Dispatch(kV, Mathf.CeilToInt(width / 8.0f), Mathf.CeilToInt(height / 8.0f), 1);

        RenderTexture.ReleaseTemporary(tempBlur);
    }

    void RenderComposite(Camera cam)
    {
        if (compositeMaterial == null) return;

        compositeMaterial.SetTexture("_MainTex", null);
        compositeMaterial.SetTexture("_NormalTex", fluidNormalTexture);
        compositeMaterial.SetTexture("_DepthTex", fluidDepthTexture);
        compositeMaterial.SetTexture("_DepthRawTex", rawDepthTexture);
        compositeMaterial.SetTexture("_ThicknessTex", fluidThicknessTexture);

        compositeMaterial.SetInt("_UseSkybox", useSkybox ? 1 : 0);
        if (useSkybox && skyboxTexture != null)
            compositeMaterial.SetTexture("_SkyboxTex", skyboxTexture);

        Vector3 sunDir = mainLight != null ? -mainLight.transform.forward : Vector3.up;
        float sunInt = Mathf.Exp(sunIntensity);

        Vector3 extinction = new Vector3(
            -Mathf.Log(Mathf.Max(1e-6f, fluidColor.r)),
            -Mathf.Log(Mathf.Max(1e-6f, fluidColor.g)),
            -Mathf.Log(Mathf.Max(1e-6f, fluidColor.b))
        ) * absorptionStrength;

        compositeMaterial.SetVector("extinctionCoefficients", extinction);
        compositeMaterial.SetVector("dirToSun", sunDir);
        compositeMaterial.SetVector("boundsSize", SimBounds.size);
        compositeMaterial.SetFloat("refractionMultiplier", refractionScale);
        compositeMaterial.SetFloat("sunIntensity", sunInt);
        compositeMaterial.SetFloat("sunSharpness", sunSharpness);
        compositeMaterial.SetVector("skyHorizonColor", skyHorizonColor);
        compositeMaterial.SetVector("skyTopColor", skyTopColor);
        compositeMaterial.SetFloat("reflectionStrength", reflectionStrength);
        compositeMaterial.SetFloat("reflectionTint", reflectionTint);
        compositeMaterial.SetFloat("fresnelClamp", fresnelClamp);

        Vector3 cameraPos = cam.transform.position;
        Bounds bounds = SimBounds;
        float boundsDiagonal = bounds.size.magnitude;
        CalculateDepthParameters(cameraPos, bounds, boundsDiagonal);

        compositeMaterial.SetFloat("depthDisplayScale", calculatedDepthDisplayScale);
        compositeMaterial.SetFloat("depthMinValue", calculatedDepthMinValue);
        compositeMaterial.SetFloat("thicknessDisplayScale", ThicknessBlitDisplayScale);
        compositeMaterial.SetFloat("depthOfFieldStrength", depthOfFieldStrength);

        compositeMaterial.SetVector("floorPos", new Vector3(0, -2.5f, 0));
        compositeMaterial.SetVector("floorSize", new Vector3(45.0f, 0.1f, 45.0f));
        compositeMaterial.SetColor("tileCol1", new Color(0.8f, 0.8f, 0.8f));
        compositeMaterial.SetColor("tileCol2", new Color(0.6f, 0.6f, 0.6f));

        Graphics.Blit(null, (RenderTexture)null, compositeMaterial);
    }
}
