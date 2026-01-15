using UnityEngine;
using UnityEngine.Rendering;

// Rendering component of FluidSimulator
public partial class FluidSimulator : MonoBehaviour
{
    void OnEnable()
    {
        RenderPipelineManager.endCameraRendering += OnEndCameraRendering;
        
        // Create command buffers for thickness and depth rendering (like working project)
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

    void OnDisable()
    {
        RenderPipelineManager.endCameraRendering -= OnEndCameraRendering;
        
        // Release args buffers
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
        
        // Release command buffers
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
        
        // Release render textures
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
    }
    private void OnEndCameraRendering(ScriptableRenderContext ctx, Camera cam)
    {
        // Start rendering timing
        renderSw.Restart();
        
        // --- 1. Render Thickness (Raw) ---
        // Needed for: Thickness, BlurredThickness, Composite
        if (renderingMode == RenderingMode.Thickness || renderingMode == RenderingMode.BlurredThickness || renderingMode == RenderingMode.Composite)
        {
            RenderThickness(cam);
            
            // If Composite or BlurredThickness, we also need to BLUR the thickness
            if (renderingMode == RenderingMode.Composite || renderingMode == RenderingMode.BlurredThickness)
            {
                 // Create smooth thickness texture if needed
                 if (fluidThicknessTexture == null || fluidThicknessTexture.width != cam.pixelWidth)
                 {
                     if (fluidThicknessTexture != null) fluidThicknessTexture.Release();
                     RenderTextureDescriptor desc = new RenderTextureDescriptor(cam.pixelWidth, cam.pixelHeight, RenderTextureFormat.RFloat, 0);
                     desc.enableRandomWrite = true;
                     fluidThicknessTexture = new RenderTexture(desc);
                     fluidThicknessTexture.Create();
                 }
                 
                 // Use thickness blur (doesn't blur with 0 thickness)
                 RunThicknessBlur(cam, thicknessTexture, fluidThicknessTexture);
            }
        }
        
        // --- 2. Render Nodes (Wireframe) ---
        // Needed for: Nodes
        if (renderingMode == RenderingMode.Nodes)
        {
            RenderNodes(cam);
        }
        
        // --- 3. Render Depth (Raw) ---
        // Needed for: Depth, BlurredDepth, Normal, Composite
        if (renderingMode != RenderingMode.Thickness && renderingMode != RenderingMode.BlurredThickness && renderingMode != RenderingMode.Nodes) 
        {
            DrawParticles(cam, ctx); // Renders to rawDepthTexture or screen
        }
        
        // --- 4. Blur Depth & Gen Normals ---
        // Needed for: BlurredDepth, Normal, Composite
        if (renderingMode == RenderingMode.BlurredDepth || renderingMode == RenderingMode.Normal || renderingMode == RenderingMode.Composite)
        {
            // Initialize fluidDepthTexture if needed
            if (fluidDepthTexture == null || fluidDepthTexture.width != cam.pixelWidth || fluidDepthTexture.height != cam.pixelHeight)
            {
                if (fluidDepthTexture != null) fluidDepthTexture.Release();
                RenderTextureDescriptor desc = new RenderTextureDescriptor(cam.pixelWidth, cam.pixelHeight, RenderTextureFormat.RFloat, 0);
                desc.enableRandomWrite = true;
                fluidDepthTexture = new RenderTexture(desc);
                fluidDepthTexture.Create();
            }
            
            // 1. Blur Depth (Raw -> FluidDepth)
            RunBilateralBlur(cam, rawDepthTexture, fluidDepthTexture);
            
            // 2. Generate Normals (FluidDepth -> FluidNormal)
            RenderNormals(cam);
        }
        
        // --- 5. Final Display / Composite ---
        if (renderingMode == RenderingMode.Nodes)
        {
            // Display nodes texture directly (it's already in RGB format with colors)
            Graphics.Blit(nodesTexture, (RenderTexture)null);
        }
        else if (renderingMode == RenderingMode.Thickness)
        {
            float thicknessScale = thicknessSource == ThicknessSource.Nodes 
                ? thicknessScaleNodes 
                : thicknessScaleParticles;
            DebugBlit(thicknessTexture, thicknessScale, 0.0f, false);
        }
        else if (renderingMode == RenderingMode.BlurredThickness)
        {
            float thicknessScale = thicknessSource == ThicknessSource.Nodes 
                ? thicknessScaleNodes 
                : thicknessScaleParticles;
            DebugBlit(fluidThicknessTexture, thicknessScale, 0.0f, false);
        }
        else if (renderingMode == RenderingMode.Depth)
        {
            // Calculate depth parameters for debug visualization
            if (cam != null && simulationBounds != null)
            {
                Vector3 cameraPos = cam.transform.position;
                Bounds bounds = simulationBounds.bounds;
                float boundsDiagonal = bounds.size.magnitude;
                CalculateDepthParameters(cameraPos, bounds, boundsDiagonal);
            }
            DebugBlit(rawDepthTexture, calculatedDepthDisplayScale, calculatedDepthMinValue);
        }
        else if (renderingMode == RenderingMode.BlurredDepth)
        {
            // Calculate depth parameters for debug visualization
            if (cam != null && simulationBounds != null)
            {
                Vector3 cameraPos = cam.transform.position;
                Bounds bounds = simulationBounds.bounds;
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
            RenderComposite(cam);
        }
        
        // Stop rendering timing and store the result
        renderSw.Stop();
        lastRenderTimeMs = renderSw.Elapsed.TotalMilliseconds;
    }
    private void DrawParticles(Camera cam, ScriptableRenderContext ctx = default)
    {
        if (particlesBuffer == null || numParticles <= 0) return;
        if (cam == null) return;

        Material currentMaterial = null;
        bool useBlur = false;

        switch (renderingMode)
        {
            case RenderingMode.Particles:
                currentMaterial = particlesMaterial;
                break;
            case RenderingMode.Depth:
                currentMaterial = particleDepthMaterial;
                break;
            // FIX: Handle Normal mode here so it generates the depth texture required!
            case RenderingMode.BlurredDepth:
            case RenderingMode.Normal: 
            case RenderingMode.Composite: // [FIX] Add this
                currentMaterial = particleDepthMaterial; // Use depth shader as base
                useBlur = true;
                break;
        }

        if (currentMaterial == null) return;

        // Create quad mesh and args buffer if they don't exist (needed for Depth mode)
        if (quadMesh == null)
        {
            CreateQuadMesh();
        }
        
        // Update particle args buffer with current particle count (needed for Depth mode)
        if (particleArgsBuffer == null || particleArgsBuffer.count != 5)
        {
            CreateParticleArgsBuffer();
        }
        else
        {
            // Update instance count in args buffer
            uint[] args = new uint[5];
            particleArgsBuffer.GetData(args);
            args[1] = (uint)numParticles; // Update instance count
            particleArgsBuffer.SetData(args);
        }

        // --- Common Material Setup ---
        currentMaterial.SetBuffer("_Particles", particlesBuffer);
        
        // Set radius for Particles mode or Depth modes
        if (renderingMode == RenderingMode.Particles)
        {
            currentMaterial.SetFloat("_Radius", particleRadius);
        }
        else if (renderingMode == RenderingMode.Depth || renderingMode == RenderingMode.BlurredDepth || renderingMode == RenderingMode.Normal || renderingMode == RenderingMode.Composite)
        {
            currentMaterial.SetFloat("_Radius", depthRadius);
        }
        currentMaterial.SetVector("_SimulationBoundsMin", simulationBounds.bounds.min);
        currentMaterial.SetVector("_SimulationBoundsMax", simulationBounds.bounds.max);
        
        // Calculate depth fade parameters based on simulation bounds and camera position
        Vector3 cameraPos = cam.transform.position;
        Bounds bounds = simulationBounds.bounds;
        
        // Calculate distance from camera to nearest and farthest points of the bounds
        Vector3 boundsCenter = bounds.center;
        Vector3 boundsSize = bounds.size;
        
        // Find the nearest point on the bounds to the camera
        Vector3 nearestPoint = new Vector3(
            Mathf.Clamp(cameraPos.x, bounds.min.x, bounds.max.x),
            Mathf.Clamp(cameraPos.y, bounds.min.y, bounds.max.y),
            Mathf.Clamp(cameraPos.z, bounds.min.z, bounds.max.z)
        );
        
        // Calculate distancesx
        float distanceToNearest = Vector3.Distance(cameraPos, nearestPoint);
        float distanceToCenter = Vector3.Distance(cameraPos, boundsCenter);
        float boundsDiagonal = boundsSize.magnitude; // Diagonal length of the bounds
        
        // Calculate depth min/max for normalizing particles inside the simulation bounds to 0-1 range
        CalculateDepthParameters(cameraPos, bounds, boundsDiagonal);
        
        // Set depth scale and min value for Particles mode (calculated dynamically)
        if (renderingMode == RenderingMode.Particles)
        {
            // Get the calculated values (convert from exponents to actual values)
            float calculatedMinValue = Mathf.Exp(calculatedDepthMinValue);
            float calculatedScale = Mathf.Exp(calculatedDepthDisplayScale);
            currentMaterial.SetFloat("_Scale", calculatedScale);
            currentMaterial.SetFloat("_MinValue", calculatedMinValue);
        }
        
        // Calculate fade start: start fading slightly before the nearest point
        // Subtract half the diagonal to account for particles that might be at the edge
        float fadeStart = Mathf.Max(0.0f, distanceToNearest - boundsDiagonal * 0.3f);
        
        // Calculate fade end: end fading at the farthest point plus some margin
        // Use the diagonal to estimate the farthest possible distance
        float fadeEnd = distanceToCenter + boundsDiagonal * 0.8f;
        
        // Ensure fade end is always greater than fade start
        if (fadeEnd <= fadeStart)
        {
            fadeEnd = fadeStart + boundsDiagonal * 0.5f;
        }
        
        currentMaterial.SetFloat("_DepthFadeStart", fadeStart);
        currentMaterial.SetFloat("_DepthFadeEnd", fadeEnd);
        
        // Calculate depth min/max for depth visualization (for depth and blurred depth materials)
        // Reuse the already calculated minDepth and maxDepth values
        if (renderingMode == RenderingMode.Depth || renderingMode == RenderingMode.BlurredDepth || renderingMode == RenderingMode.Normal || renderingMode == RenderingMode.Composite)
        {
            currentMaterial.SetFloat("_DepthMin", calculatedMinDepth);
            currentMaterial.SetFloat("_DepthMax", calculatedMaxDepth);
        }

        // For Composite mode, we render to rawDepthTexture directly (blur happens in OnEndCameraRendering)
        // For BlurredDepth mode, we use RenderBlurredDepth which handles its own blur
        if (useBlur && blurCompute != null && renderingMode != RenderingMode.Composite)
        {
            RenderBlurredDepth(cam, currentMaterial);
        }
        else
        {
            // [FIX] Update condition to include all modes that need the depth texture
            if (renderingMode == RenderingMode.Depth || 
                renderingMode == RenderingMode.BlurredDepth || 
                renderingMode == RenderingMode.Normal || 
                renderingMode == RenderingMode.Composite)
            {
                // Ensure raw depth texture exists and matches screen size
                if (rawDepthTexture == null || rawDepthTexture.width != cam.pixelWidth || rawDepthTexture.height != cam.pixelHeight)
                {
                    if (rawDepthTexture != null)
                    {
                        rawDepthTexture.Release();
                    }
                    rawDepthTexture = new RenderTexture(cam.pixelWidth, cam.pixelHeight, 24, RenderTextureFormat.RFloat);
                    rawDepthTexture.enableRandomWrite = true;
                    rawDepthTexture.Create();
                }
                
                depthCmd.Clear();
                depthCmd.SetRenderTarget(rawDepthTexture);
                // Clear to 'Far' (10,000)
                depthCmd.ClearRenderTarget(true, true, new Color(10000.0f, 10000.0f, 10000.0f, 1.0f));
                depthCmd.DrawMeshInstancedIndirect(quadMesh, 0, currentMaterial, 0, particleArgsBuffer);
                Graphics.ExecuteCommandBuffer(depthCmd);
                
                // Only debug blit if we are strictly in Depth mode
                if (renderingMode == RenderingMode.Depth)
                {
                    DebugBlit(rawDepthTexture, calculatedDepthDisplayScale, calculatedDepthMinValue);
                }
            }
            else
            {
                // For Particles mode, use instanced rendering with quad mesh
                if (particlesBuffer == null || numParticles <= 0) return;
                
                // Update particle args buffer with current particle count
                if (particleArgsBuffer == null || particleArgsBuffer.count != 5)
                {
                    CreateParticleArgsBuffer();
                }
                else
                {
                    // Update instance count in args buffer
                    uint[] args = new uint[5];
                    particleArgsBuffer.GetData(args);
                    args[1] = (uint)numParticles; // Update instance count
                    particleArgsBuffer.SetData(args);
                }
                
                // Use CommandBuffer to ensure correct RenderTarget and avoid strict bounds culling
                depthCmd.Clear();
                
                // Target the Camera's active buffer (Screen)
                depthCmd.SetRenderTarget(BuiltinRenderTextureType.CameraTarget);
                
                // Draw mesh without strict bounds checks
                depthCmd.DrawMeshInstancedIndirect(quadMesh, 0, currentMaterial, 0, particleArgsBuffer);
                
                Graphics.ExecuteCommandBuffer(depthCmd);
            }
        }
    }
    private void RenderThickness(Camera cam)
    {
        if (cam == null) return;

        // Create quad mesh if it doesn't exist
        if (quadMesh == null)
        {
            CreateQuadMesh();
        }

        // Ensure thickness texture exists and matches screen size
        if (thicknessTexture == null || thicknessTexture.width != cam.pixelWidth || thicknessTexture.height != cam.pixelHeight)
        {
            if (thicknessTexture != null)
            {
                thicknessTexture.Release();
            }
            thicknessTexture = new RenderTexture(cam.pixelWidth, cam.pixelHeight, 0, RenderTextureFormat.RFloat);
            thicknessTexture.enableRandomWrite = true;
            thicknessTexture.Create();
        }

        // Use CommandBuffer like working project (this may fix Metal binding issues)
        thicknessCmd.Clear();
        
        // Set render target to thickness texture and clear to black (0 thickness)
        thicknessCmd.SetRenderTarget(thicknessTexture);
        thicknessCmd.ClearRenderTarget(true, true, Color.black);

        // Switch between nodes and particles based on thicknessSource enum
        if (thicknessSource == ThicknessSource.Nodes)
        {
            // Render using nodes
            if (nodesBuffer == null || numNodes <= 0) return;
            if (nodeThicknessMaterial == null) return;

            // Update args buffer with current node count
            if (argsBuffer == null || argsBuffer.count != 5)
            {
                CreateArgsBuffer();
            }
            else
            {
                // Update instance count in args buffer
                uint[] args = new uint[5];
                argsBuffer.GetData(args);
                args[1] = (uint)numNodes; // Update instance count
                argsBuffer.SetData(args);
            }

            // Set up material properties for NodeThickness shader
            nodeThicknessMaterial.SetBuffer("_Nodes", nodesBuffer);
            nodeThicknessMaterial.SetVector("_SimulationBoundsMin", simulationBounds.bounds.min);
            nodeThicknessMaterial.SetVector("_SimulationBoundsMax", simulationBounds.bounds.max);
            nodeThicknessMaterial.SetFloat("_PointSize", maxDetailCellSize); // Use maxDetailCellSize as base size
            nodeThicknessMaterial.SetColor("_Color", new Color(0.0f, 0.5f, 1.0f, 1.0f)); // Blue color

            // Draw nodes to thickness texture
            thicknessCmd.DrawMeshInstancedIndirect(
                quadMesh,
                0,
                nodeThicknessMaterial,
                0,  // shader pass index
                argsBuffer
            );
        }
        else // ThicknessSource.Particles
        {
            // Render using particles
            if (particlesBuffer == null || numParticles <= 0) return;
            if (particleThicknessMaterial == null) return;

            // Update particle args buffer with current particle count
            if (particleArgsBuffer == null || particleArgsBuffer.count != 5)
            {
                CreateParticleArgsBuffer();
            }
            else
            {
                // Update instance count in args buffer
                uint[] args = new uint[5];
                particleArgsBuffer.GetData(args);
                args[1] = (uint)numParticles; // Update instance count
                particleArgsBuffer.SetData(args);
            }

            // Set up material properties for ParticleThickness shader
            particleThicknessMaterial.SetBuffer("_Particles", particlesBuffer);
            particleThicknessMaterial.SetVector("_SimulationBoundsMin", simulationBounds.bounds.min);
            particleThicknessMaterial.SetVector("_SimulationBoundsMax", simulationBounds.bounds.max);
            particleThicknessMaterial.SetFloat("_Radius", thicknessRadius);

            // Draw particles to thickness texture
            thicknessCmd.DrawMeshInstancedIndirect(
                quadMesh,
                0,
                particleThicknessMaterial,
                0,  // shader pass index
                particleArgsBuffer
            );
        }
        
        // Execute command buffer
        Graphics.ExecuteCommandBuffer(thicknessCmd);

        // Visualization is handled in OnEndCameraRendering
    }
    private void RenderNodes(Camera cam)
    {
        if (cam == null) return;

        // Create quad mesh if it doesn't exist
        if (quadMesh == null)
        {
            CreateQuadMesh();
        }

        // Ensure nodes texture exists and matches screen size
        if (nodesTexture == null || nodesTexture.width != cam.pixelWidth || nodesTexture.height != cam.pixelHeight)
        {
            if (nodesTexture != null)
            {
                nodesTexture.Release();
            }
            nodesTexture = new RenderTexture(cam.pixelWidth, cam.pixelHeight, 24, RenderTextureFormat.ARGBFloat);
            nodesTexture.enableRandomWrite = true;
            nodesTexture.Create();
        }

        // Use CommandBuffer for rendering
        thicknessCmd.Clear();
        
        // Set render target to nodes texture and clear to transparent
        thicknessCmd.SetRenderTarget(nodesTexture);
        thicknessCmd.ClearRenderTarget(true, true, Color.clear);

        // Render using nodes
        if (nodesBuffer == null || numNodes <= 0) return;
        if (nodeWireframeMaterial == null) return;

        // Update args buffer with current node count
        if (argsBuffer == null || argsBuffer.count != 5)
        {
            CreateArgsBuffer();
        }
        else
        {
            // Update instance count in args buffer
            uint[] args = new uint[5];
            argsBuffer.GetData(args);
            args[1] = (uint)numNodes; // Update instance count
            argsBuffer.SetData(args);
        }

        // Set up material properties for NodeWireframe shader
        nodeWireframeMaterial.SetBuffer("_Nodes", nodesBuffer);
        nodeWireframeMaterial.SetVector("_SimulationBoundsMin", simulationBounds.bounds.min);
        nodeWireframeMaterial.SetVector("_SimulationBoundsMax", simulationBounds.bounds.max);
        nodeWireframeMaterial.SetFloat("_PointSize", maxDetailCellSize); // Use maxDetailCellSize as base size
        // _WireframeWidth and _DensityScale are controlled from Material Inspector, not set here
        
        // Set depth parameters (same as particles shader - automatic depth shading)
        if (cam != null && simulationBounds != null)
        {
            Vector3 cameraPos = cam.transform.position;
            Bounds bounds = simulationBounds.bounds;
            float boundsDiagonal = bounds.size.magnitude;
            
            // Calculate depth parameters for automatic depth shading
            CalculateDepthParameters(cameraPos, bounds, boundsDiagonal);
            
            // Get the calculated values (convert from exponents to actual values)
            float calculatedMinValue = Mathf.Exp(calculatedDepthMinValue);
            float calculatedScale = Mathf.Exp(calculatedDepthDisplayScale);
            nodeWireframeMaterial.SetFloat("_Scale", calculatedScale);
            nodeWireframeMaterial.SetFloat("_MinValue", calculatedMinValue);
        }

        // Draw nodes to nodes texture
        thicknessCmd.DrawMeshInstancedIndirect(
            quadMesh,
            0,
            nodeWireframeMaterial,
            0,  // shader pass index
            argsBuffer
        );
        
        // Execute command buffer
        Graphics.ExecuteCommandBuffer(thicknessCmd);

        // Visualization is handled in OnEndCameraRendering
    }
    private void RenderNormals(Camera cam)
    {
        int width = cam.pixelWidth;
        int height = cam.pixelHeight;

        // 1. Ensure Texture Exists (Same as before)
        if (fluidNormalTexture == null || fluidNormalTexture.width != width || fluidNormalTexture.height != height)
        {
            if (fluidNormalTexture != null) fluidNormalTexture.Release();
            RenderTextureDescriptor desc = new RenderTextureDescriptor(width, height, RenderTextureFormat.ARGBHalf, 0);
            fluidNormalTexture = new RenderTexture(desc);
            fluidNormalTexture.Create();
        }

        // 2. Setup Material
        if (normalMaterial != null && fluidDepthTexture != null)
        {
            // Keep this: View -> World
            normalMaterial.SetMatrix("_CameraInvViewMatrix", cam.cameraToWorldMatrix);
            
            // FIX: Remove _CameraInvProjMatrix. 
            // Instead, pass the intrinsic camera parameters needed to unproject linear depth.
            // Unity's built-in projection params (zBufferParams, etc) handle part of this, 
            // but for manual linear depth, we just need FOV and Aspect.
            
            float fovY = Mathf.Deg2Rad * cam.fieldOfView;
            float aspect = cam.aspect;
            float tanHalfFovY = Mathf.Tan(fovY * 0.5f);
            
            // Pass these to the shader
            normalMaterial.SetVector("_CameraParams", new Vector4(aspect * tanHalfFovY, tanHalfFovY, 0, 0));
            normalMaterial.SetTexture("_MainTex", fluidDepthTexture);
            
            Graphics.Blit(fluidDepthTexture, fluidNormalTexture, normalMaterial);
        }
    }
    private void DebugBlit(RenderTexture source, float scaleExponent, float minValueExponent = float.MinValue, bool useExpForMin = true)
    {
        if (debugDisplayMaterial == null || source == null) return;

        // Restore the screen as the render target
        Graphics.SetRenderTarget(null);
        
        // Convert exponent to actual scale: exp(scaleExponent)
        float scale = Mathf.Exp(scaleExponent);
        debugDisplayMaterial.SetFloat("_Scale", scale);
        
        // Set min value
        float minValue;
        if (minValueExponent == float.MinValue)
        {
            minValue = 0.0f; // Default to 0 if not provided
        }
        else if (useExpForMin)
        {
            minValue = Mathf.Exp(minValueExponent); // Use exp for depth
        }
        else
        {
            minValue = minValueExponent; // Use directly for thickness (should be 0.0f)
        }
        debugDisplayMaterial.SetFloat("_MinValue", minValue);
        
        // Draw to screen
        Graphics.Blit(source, (RenderTexture)null, debugDisplayMaterial);
    }
    private void CalculateDepthParameters(Vector3 cameraPos, Bounds bounds, float boundsDiagonal)
    {
        // Calculate depth min/max for normalizing particles inside the simulation bounds to 0-1 range
        // Min depth: distance to nearest point on bounds (particles can't be closer)
        Vector3 nearestPoint = new Vector3(
            Mathf.Clamp(cameraPos.x, bounds.min.x, bounds.max.x),
            Mathf.Clamp(cameraPos.y, bounds.min.y, bounds.max.y),
            Mathf.Clamp(cameraPos.z, bounds.min.z, bounds.max.z)
        );
        float minDepth = Vector3.Distance(cameraPos, nearestPoint);
        
        // Max depth: find the farthest corner of the bounds (farthest point where particles can be)
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
        
        // Ensure max is greater than min
        if (maxDepth <= minDepth)
        {
            maxDepth = minDepth + boundsDiagonal * 0.5f;
        }
        
        // Store the actual depth values for use in depth visualization
        calculatedMinDepth = minDepth;
        calculatedMaxDepth = maxDepth;
        
        // Calculate depth scale and min value dynamically to map depth to 0-1 range
        // Formula: displayVal = (depth - minDepth) * (1.0 / (maxDepth - minDepth))
        // So: _MinValue = minDepth, _Scale = 1.0 / (maxDepth - minDepth)
        // Store as exponents: depthMinValue = ln(minDepth), depthDisplayScale = ln(1.0 / (maxDepth - minDepth))
        float calculatedMinValue = Mathf.Max(0.001f, minDepth); // Ensure > 0 for log
        float depthRange = maxDepth - minDepth;
        float calculatedScale = depthRange > 0.001f ? (1.0f / depthRange) : 1.0f; // Avoid division by zero
        
        // Store as exponents (for compatibility with shaders that use exp())
        calculatedDepthMinValue = Mathf.Log(calculatedMinValue);
        calculatedDepthDisplayScale = Mathf.Log(calculatedScale);
    }
    private void CreateQuadMesh()
    {
        // Create a simple quad mesh for billboard rendering
        Vector3[] vertices = new Vector3[4]
        {
            new Vector3(-0.5f, -0.5f, 0),
            new Vector3(0.5f, -0.5f, 0),
            new Vector3(-0.5f, 0.5f, 0),
            new Vector3(0.5f, 0.5f, 0)
        };
        
        int[] triangles = new int[6]
        {
            0, 2, 1,
            2, 3, 1
        };
        
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
    private void CreateArgsBuffer()
    {
        if (quadMesh == null) CreateQuadMesh();
        
        const int stride = sizeof(uint);
        const int numArgs = 5;
        const int subMeshIndex = 0;
        
        uint[] args = new uint[numArgs];
        args[0] = (uint)quadMesh.GetIndexCount(subMeshIndex);
        args[1] = (uint)numNodes;
        args[2] = (uint)quadMesh.GetIndexStart(subMeshIndex);
        args[3] = (uint)quadMesh.GetBaseVertex(subMeshIndex);
        args[4] = 0; // start instance location
        
        if (argsBuffer != null)
        {
            argsBuffer.Release();
        }
        argsBuffer = new ComputeBuffer(numArgs, stride, ComputeBufferType.IndirectArguments);
        argsBuffer.SetData(args);
    }
    private void CreateParticleArgsBuffer()
    {
        if (quadMesh == null) CreateQuadMesh();
        
        const int stride = sizeof(uint);
        const int numArgs = 5;
        const int subMeshIndex = 0;
        
        uint[] args = new uint[numArgs];
        args[0] = (uint)quadMesh.GetIndexCount(subMeshIndex);
        args[1] = (uint)numParticles;
        args[2] = (uint)quadMesh.GetIndexStart(subMeshIndex);
        args[3] = (uint)quadMesh.GetBaseVertex(subMeshIndex);
        args[4] = 0; // start instance location
        
        if (particleArgsBuffer != null)
        {
            particleArgsBuffer.Release();
        }
        particleArgsBuffer = new ComputeBuffer(numArgs, stride, ComputeBufferType.IndirectArguments);
        particleArgsBuffer.SetData(args);
    }
    private void RenderBlurredDepth(Camera cam, Material mat)
    {
        int width = cam.pixelWidth;
        int height = cam.pixelHeight;

        // 1. Initialize persistent texture (fluidDepthTexture)
        // Check if null or size changed
        if (fluidDepthTexture == null || fluidDepthTexture.width != width || fluidDepthTexture.height != height)
        {
            if (fluidDepthTexture != null) fluidDepthTexture.Release();
            
            // Use descriptor to properly ensure UAV flag is set BEFORE creation
            RenderTextureDescriptor desc = new RenderTextureDescriptor(width, height, RenderTextureFormat.RFloat, 0);
            desc.enableRandomWrite = true; // REQUIRED for Compute Shader output
            
            fluidDepthTexture = new RenderTexture(desc);
            fluidDepthTexture.Create();
        }

        // 2. Create rawDepth (Input for blur)
        // IMPORTANT: enableRandomWrite MUST be FALSE here because this texture has a Depth Buffer (24).
        // It is only read by the Compute Shader (as _SourceTexture), so it doesn't need UAV.
        RenderTextureDescriptor rawDesc = new RenderTextureDescriptor(width, height, RenderTextureFormat.RFloat, 24);
        rawDesc.enableRandomWrite = false; 
        RenderTexture rawDepth = RenderTexture.GetTemporary(rawDesc);

        // 3. Create tempBlur (Intermediate Ping-Pong buffer)
        // This needs UAV because the Compute Shader writes to it. No Depth Buffer needed (0).
        RenderTextureDescriptor blurDesc = new RenderTextureDescriptor(width, height, RenderTextureFormat.RFloat, 0);
        blurDesc.enableRandomWrite = true; 
        RenderTexture tempBlur = RenderTexture.GetTemporary(blurDesc);
        
        // --- RENDERING ---
        
        // A. Render Particles to rawDepth
        depthCmd.Clear();
        depthCmd.SetRenderTarget(rawDepth);
        depthCmd.ClearRenderTarget(true, true, Color.black); // Clear to "far away"
        depthCmd.DrawMeshInstancedIndirect(quadMesh, 0, mat, 0, particleArgsBuffer);
        Graphics.ExecuteCommandBuffer(depthCmd);

        // B. Horizontal Blur (rawDepth -> tempBlur)
        int kernelHandleH = blurCompute.FindKernel("DepthBlurHorizontal");
        blurCompute.SetTexture(kernelHandleH, "_SourceTexture", rawDepth);
        blurCompute.SetTexture(kernelHandleH, "_DestinationTexture", tempBlur);
        blurCompute.SetFloat("_BlurRadius", depthBlurRadius);
        blurCompute.SetFloat("_BlurDepthFalloff", depthBlurThreshold);
        blurCompute.SetVector("_Resolution", new Vector2(width, height));
        
        int groupsX = Mathf.CeilToInt(width / 8.0f);
        int groupsY = Mathf.CeilToInt(height / 8.0f);
        blurCompute.Dispatch(kernelHandleH, groupsX, groupsY, 1);

        // C. Vertical Blur (tempBlur -> fluidDepthTexture)
        int kernelHandleV = blurCompute.FindKernel("DepthBlurVertical");
        blurCompute.SetTexture(kernelHandleV, "_SourceTexture", tempBlur);
        blurCompute.SetTexture(kernelHandleV, "_DestinationTexture", fluidDepthTexture);
        blurCompute.SetFloat("_BlurRadius", depthBlurRadius);
        blurCompute.SetFloat("_BlurDepthFalloff", depthBlurThreshold);
        blurCompute.SetVector("_Resolution", new Vector2(width, height));
        
        blurCompute.Dispatch(kernelHandleV, groupsX, groupsY, 1);

        // D. Visualize
        if (renderingMode == RenderingMode.BlurredDepth)
        {
            // Calculate depth parameters for debug visualization
            if (cam != null && simulationBounds != null)
            {
                Vector3 cameraPos = cam.transform.position;
                Bounds bounds = simulationBounds.bounds;
                float boundsDiagonal = bounds.size.magnitude;
                CalculateDepthParameters(cameraPos, bounds, boundsDiagonal);
            }
            DebugBlit(fluidDepthTexture, calculatedDepthDisplayScale, calculatedDepthMinValue);
        }

        // E. Cleanup
        RenderTexture.ReleaseTemporary(rawDepth);
        RenderTexture.ReleaseTemporary(tempBlur);
    }
    private void RunBilateralBlur(Camera cam, RenderTexture source, RenderTexture dest)
    {
        if (blurCompute == null || source == null || dest == null) return;
        
        int width = cam.pixelWidth;
        int height = cam.pixelHeight;
        
        // Temp buffer
        RenderTextureDescriptor blurDesc = new RenderTextureDescriptor(width, height, RenderTextureFormat.RFloat, 0);
        blurDesc.enableRandomWrite = true; 
        RenderTexture tempBlur = RenderTexture.GetTemporary(blurDesc);

        // H Blur
        int kH = blurCompute.FindKernel("DepthBlurHorizontal");
        blurCompute.SetTexture(kH, "_SourceTexture", source);
        blurCompute.SetTexture(kH, "_DestinationTexture", tempBlur);
        blurCompute.SetFloat("_BlurRadius", depthBlurRadius);
        blurCompute.SetFloat("_BlurDepthFalloff", depthBlurThreshold);
        blurCompute.SetVector("_Resolution", new Vector2(width, height));
        blurCompute.Dispatch(kH, Mathf.CeilToInt(width/8.0f), Mathf.CeilToInt(height/8.0f), 1);
        
        // V Blur
        int kV = blurCompute.FindKernel("DepthBlurVertical");
        blurCompute.SetTexture(kV, "_SourceTexture", tempBlur);
        blurCompute.SetTexture(kV, "_DestinationTexture", dest);
        blurCompute.SetFloat("_BlurRadius", depthBlurRadius);
        blurCompute.SetFloat("_BlurDepthFalloff", depthBlurThreshold);
        blurCompute.SetVector("_Resolution", new Vector2(width, height));
        blurCompute.Dispatch(kV, Mathf.CeilToInt(width/8.0f), Mathf.CeilToInt(height/8.0f), 1);
        
        RenderTexture.ReleaseTemporary(tempBlur);
    }
    private void RunThicknessBlur(Camera cam, RenderTexture source, RenderTexture dest)
    {
        if (blurCompute == null || source == null || dest == null) return;
        
        int width = cam.pixelWidth;
        int height = cam.pixelHeight;
        
        // Temp buffer
        RenderTextureDescriptor blurDesc = new RenderTextureDescriptor(width, height, RenderTextureFormat.RFloat, 0);
        blurDesc.enableRandomWrite = true; 
        RenderTexture tempBlur = RenderTexture.GetTemporary(blurDesc);

        // H Blur
        int kH = blurCompute.FindKernel("ThicknessBlurHorizontal");
        blurCompute.SetTexture(kH, "_SourceTexture", source);
        blurCompute.SetTexture(kH, "_DestinationTexture", tempBlur);
        blurCompute.SetFloat("_BlurRadius", thicknessBlurRadius);
        blurCompute.SetVector("_Resolution", new Vector2(width, height));
        blurCompute.Dispatch(kH, Mathf.CeilToInt(width/8.0f), Mathf.CeilToInt(height/8.0f), 1);
        
        // V Blur
        int kV = blurCompute.FindKernel("ThicknessBlurVertical");
        blurCompute.SetTexture(kV, "_SourceTexture", tempBlur);
        blurCompute.SetTexture(kV, "_DestinationTexture", dest);
        blurCompute.SetFloat("_BlurRadius", thicknessBlurRadius);
        blurCompute.SetVector("_Resolution", new Vector2(width, height));
        blurCompute.Dispatch(kV, Mathf.CeilToInt(width/8.0f), Mathf.CeilToInt(height/8.0f), 1);
        
        RenderTexture.ReleaseTemporary(tempBlur);
    }
    private void RenderComposite(Camera cam)
    {
        if (compositeMaterial == null) return;

        // 1. Bind Textures
        compositeMaterial.SetTexture("_MainTex", null); // Or camera target if you grab it
        compositeMaterial.SetTexture("_NormalTex", fluidNormalTexture);
        compositeMaterial.SetTexture("_DepthTex", fluidDepthTexture); // Smooth
        compositeMaterial.SetTexture("_DepthRawTex", rawDepthTexture); // Raw
        compositeMaterial.SetTexture("_ThicknessTex", fluidThicknessTexture); // Smooth
        
        // --- NEW: Bind Skybox (only if useSkybox is enabled) ---
        compositeMaterial.SetInt("_UseSkybox", useSkybox ? 1 : 0);
        if (useSkybox && skyboxTexture != null)
        {
            compositeMaterial.SetTexture("_SkyboxTex", skyboxTexture);
        }

        // 2. Bind Parameters
        Vector3 sunDir = mainLight != null ? -mainLight.transform.forward : Vector3.up;
        float sunInt = Mathf.Exp(sunIntensity);
        
        // Calculate extinction from color (Logarithm of inverse color)
        Vector3 extinction = new Vector3(
            -Mathf.Log(fluidColor.r), 
            -Mathf.Log(fluidColor.g), 
            -Mathf.Log(fluidColor.b)
        ) * absorptionStrength;

        compositeMaterial.SetVector("extinctionCoefficients", extinction);
        compositeMaterial.SetVector("dirToSun", sunDir);
        compositeMaterial.SetVector("boundsSize", simulationBounds.bounds.size);
        compositeMaterial.SetFloat("refractionMultiplier", refractionScale);
        compositeMaterial.SetFloat("sunIntensity", sunInt);
        compositeMaterial.SetFloat("sunSharpness", sunSharpness);
        
        // Set sky colors (used when useSkybox is false)
        compositeMaterial.SetVector("skyHorizonColor", skyHorizonColor);
        compositeMaterial.SetVector("skyTopColor", skyTopColor);
        
        // Set reflection tuning parameters
        compositeMaterial.SetFloat("reflectionStrength", reflectionStrength);
        compositeMaterial.SetFloat("reflectionTint", reflectionTint);
        compositeMaterial.SetFloat("fresnelClamp", fresnelClamp);

        // Calculate depth parameters for composite rendering
        Vector3 cameraPos = cam.transform.position;
        Bounds bounds = simulationBounds.bounds;
        float boundsDiagonal = bounds.size.magnitude;
        CalculateDepthParameters(cameraPos, bounds, boundsDiagonal);
        
        // Use the calculated values (already stored as exponents)
        compositeMaterial.SetFloat("depthDisplayScale", calculatedDepthDisplayScale);
        compositeMaterial.SetFloat("depthMinValue", calculatedDepthMinValue);
        float thicknessScale = thicknessSource == ThicknessSource.Nodes 
            ? thicknessScaleNodes 
            : thicknessScaleParticles;
        compositeMaterial.SetFloat("thicknessDisplayScale", thicknessScale);
        compositeMaterial.SetFloat("depthOfFieldStrength", depthOfFieldStrength);
        
        // Environment (Simple checkerboard settings)
        compositeMaterial.SetVector("floorPos", new Vector3(0, -2.5f, 0)); // Adjust as needed
        compositeMaterial.SetVector("floorSize", new Vector3(45.0f, 0.1f, 45.0f));
        compositeMaterial.SetColor("tileCol1", new Color(0.8f, 0.8f, 0.8f));
        compositeMaterial.SetColor("tileCol2", new Color(0.6f, 0.6f, 0.6f));

        // 3. Blit to Screen
        Graphics.Blit(null, (RenderTexture)null, compositeMaterial);
    }
}
