Shader "Custom/NodeWireframe"
{
    Properties
    {
        _PointSize ("Point Size", Float) = 5.0
        _WireframeWidth ("Wireframe Width (Pixels)", Float) = 2.0
        _Transparency ("Transparency", Range(0.0, 1.0)) = 0.8
        _DensityScale ("Density Scale", Range(-20.0, 20.0)) = -8.9
    }
    SubShader
    {
        Tags { "RenderType"="Transparent" "Queue"="Transparent" "RenderPipeline"="UniversalRenderPipeline" }
        Pass
        {
            Tags { "LightMode" = "SRPDefaultUnlit" }
     
            ZWrite Off
            ZTest LEqual
            Cull Off
            // Additive blending for "glowing" wires that don't occlude each other
            Blend SrcAlpha One

            HLSLINCLUDE
            #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Common.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/SpaceTransforms.hlsl"
 
            struct faceVelocities { float left, right, bottom, top, front, back; };
            struct Node {
                float3 position;
                float3 velocity; faceVelocities v;
                float mass; uint layer; uint mortonCode; uint active;
            };

            StructuredBuffer<Node> _Nodes;
            float3 _SimulationBoundsMin;
            float3 _SimulationBoundsMax;
            float _PointSize;
            float _WireframeWidth;
            float _Transparency;
            float _DensityScale; // Exponent: actual scale = exp(value)
            float _Scale; // Depth scale (passed from C#)
            float _MinValue; // Depth min value (passed from C#)

            struct Attributes
            {
                float4 positionOS : POSITION;
                uint instanceID : SV_InstanceID;
            };

            struct VSOut
            {
                float4 pos : SV_POSITION;
                float3 centerWS : TEXCOORD0;
                float nodeSize : TEXCOORD1;
                float3 worldPos : TEXCOORD2;
                uint layer : TEXCOORD3;
                float2 screenPos : TEXCOORD4; // Screen position for pixel-space calculations
                float depth : TEXCOORD5; // Distance from camera for depth fade
                float density : TEXCOORD6; // Density for transparency
            };

            // Deinterleave morton code to get 3D grid coordinates
            uint3 DeinterleaveMorton(uint m)
            {
                uint x = 0, y = 0, z = 0;
                for (uint i = 0; i < 10; i++)
                {
                    x |= ((m >> (3 * i + 0)) & 1) << i;
                    y |= ((m >> (3 * i + 1)) & 1) << i;
                    z |= ((m >> (3 * i + 2)) & 1) << i;
                }
                return uint3(x, y, z);
            }

            float3 DecodeMorton3D(Node node)
            {
                // Calculate cell center from morton code at the node's actual layer
                // This matches the logic in UpdateParticles.compute (lines 254-263)
                uint shift = node.layer * 3;
                uint cellMortonCode = node.mortonCode & ~((1u << shift) - 1);
                uint3 cellGridMin = DeinterleaveMorton(cellMortonCode);
                
                // Calculate cell center in grid space (0-1023)
                float cellSideLength = exp2((float)node.layer);
                float3 cellCenter = float3(cellGridMin) + cellSideLength * 0.5;
                
                // Convert from grid space (0-1023) to world space
                float3 simulationSize = _SimulationBoundsMax - _SimulationBoundsMin;
                return _SimulationBoundsMin + float3(
                    cellCenter.x / 1024.0 * simulationSize.x,
                    cellCenter.y / 1024.0 * simulationSize.y,
                    cellCenter.z / 1024.0 * simulationSize.z
                );
            }

            // Layer colors matching OnGizmos
            float3 GetLayerColor(uint layer)
            {
                layer = clamp(layer, 0, 10);
                float3 colors[11] = {
                    float3(1.0, 0.0, 0.0),      // Red
                    float3(1.0, 0.3, 0.0),      // Orange-red
                    float3(1.0, 0.6, 0.0),      // Orange
                    float3(1.0, 1.0, 0.0),      // Yellow
                    float3(0.5, 1.0, 0.0),      // Yellow-green
                    float3(0.0, 1.0, 0.0),      // Green
                    float3(0.0, 1.0, 0.5),      // Blue-green
                    float3(0.0, 1.0, 1.0),      // Cyan
                    float3(0.0, 0.5, 1.0),      // Light blue
                    float3(0.0, 0.0, 1.0),      // Blue
                    float3(0.5, 0.0, 1.0)       // Violet
                };
                return colors[layer];
            }

            // Helper to calculate distance to the nearest edge of a cube
            // width is now in local space units (converted from pixels)
            float GetEdgeIntensity(float3 hitPointLocal, float width)
            {
                // hitPointLocal is -0.5 to 0.5. Convert to 0..1 for easier math
                float3 uv = abs(hitPointLocal); // 0 to 0.5
                
                // We are on an edge if TWO coordinates are close to 0.5
                // Distance to face boundaries
                float3 distToEdge = 0.5 - uv; // 0 at edge, 0.5 at center
                
                // Sort distances to find the two smallest ones (closest to edges)
                float3 s = distToEdge;
                float min1 = min(s.x, min(s.y, s.z));
                float max1 = max(s.x, max(s.y, s.z));
                float mid = s.x + s.y + s.z - min1 - max1; // The second smallest value

                // If the second smallest distance is within width, we are on an edge
                return 1.0 - smoothstep(0.0, width, mid);
            }

            VSOut VS(Attributes input)
            {
                VSOut o;
                Node node = _Nodes[input.instanceID];
                
                float3 centerWS = DecodeMorton3D(node);
                float nodeSize = max(_PointSize * exp2((float)node.layer), 0.01);
                
                // --- Calculate Density (same as NodeThickness) ---
                // Volume of a cube = size^3
                float volume = nodeSize * nodeSize * nodeSize;
                // Density = Mass / Volume
                float density = (node.mass / max(volume, 0.0001));

                // --- Billboard Calculation ---
                // Scale up canvas by 2.0 to ensure the rotated cube fits inside the quad
                float3 cameraRight = unity_CameraToWorld._m00_m10_m20;
                float3 cameraUp = unity_CameraToWorld._m01_m11_m21;
                float drawSize = nodeSize * 2.0;
                
                float3 positionWS = centerWS 
                                  + (cameraRight * input.positionOS.x * drawSize) 
                                  + (cameraUp * input.positionOS.y * drawSize);
                
                o.pos = TransformWorldToHClip(positionWS);
                o.centerWS = centerWS;
                o.nodeSize = nodeSize; 
                o.worldPos = positionWS;
                o.layer = node.layer;
                
                // Store screen position for pixel-space calculations
                o.screenPos = o.pos.xy / o.pos.w;
                
                // Calculate distance from camera for depth fade (same as ParticlesPoints)
                o.depth = distance(centerWS, _WorldSpaceCameraPos.xyz);
                
                // Pass density to pixel shader
                o.density = density;
                
                return o;
            }

            float4 PS(VSOut i) : SV_Target
            {
                // Ray-Box Intersection
                float3 rayOrigin = _WorldSpaceCameraPos;
                float3 rayDir = normalize(i.worldPos - _WorldSpaceCameraPos);
                
                float3 boxMin = i.centerWS - (i.nodeSize * 0.5);
                float3 boxMax = i.centerWS + (i.nodeSize * 0.5);

                float3 invDir = 1.0 / (rayDir + 1e-5);
                float3 t1 = (boxMin - rayOrigin) * invDir;
                float3 t2 = (boxMax - rayOrigin) * invDir;
                float3 tMin = min(t1, t2);
                float3 tMax = max(t1, t2);
                
                float tEnter = max(max(tMin.x, tMin.y), tMin.z);
                float tExit = min(min(tMax.x, tMax.y), tMax.z);

                // If ray missed box, discard
                if (tExit < tEnter || tExit < 0.0) discard;

                // --- Wireframe Logic ---
                // Calculate hit positions in Local Space (-0.5 to 0.5)
                float3 hitEnter = rayOrigin + rayDir * tEnter;
                float3 hitExit = rayOrigin + rayDir * tExit;
                
                float3 localEnter = (hitEnter - i.centerWS) / i.nodeSize;
                float3 localExit = (hitExit - i.centerWS) / i.nodeSize;

                // --- Convert pixel width to local space ---
                // Calculate how many pixels one unit of local space represents on screen
                // Local space: 1 unit = nodeSize in world space
                // We need the screen-space derivative of world position to get pixels per world unit
                
                // Get the change in world position per pixel (using screen-space derivatives)
                float3 worldPosDx = ddx(i.worldPos);
                float3 worldPosDy = ddy(i.worldPos);
                
                // Calculate the screen-space projection of world-space units
                // The magnitude of the derivative gives us world-space units per pixel
                float worldUnitsPerPixel = max(length(worldPosDx), length(worldPosDy));
                
                // Convert to pixels per world unit (invert)
                float pixelsPerWorldUnit = 1.0 / max(worldUnitsPerPixel, 0.0001);
                
                // Convert to pixels per local unit
                // 1 local unit = nodeSize world units
                float pixelsPerLocalUnit = pixelsPerWorldUnit * i.nodeSize;
                
                // Convert pixel width to local space units
                // _WireframeWidth is in pixels, pixelsPerLocalUnit is pixels per local unit
                float wireframeWidthLocal = _WireframeWidth / max(pixelsPerLocalUnit, 0.001);

                // Calculate intensity for front faces and back faces
                float edgeFront = GetEdgeIntensity(localEnter, wireframeWidthLocal);
                float edgeBack = GetEdgeIntensity(localExit, wireframeWidthLocal);
                
                // Combine them so we see the full 3D structure
                float totalEdge = max(edgeFront, edgeBack * 0.5); // Dim back edges slightly

                if (totalEdge < 0.05) discard;

                // --- Depth Shading (same as ParticlesPoints) ---
                // Automatic depth-based shading using _Scale and _MinValue from C#
                float val = i.depth;
                float depthShade = 1.0;
                if (val > 0.0 && val < 10000.0)
                {
                    float adjustedVal = val - _MinValue;
                    float displayVal = adjustedVal * _Scale;
                    depthShade = saturate(displayVal);
                }
                
                // --- Density-based Transparency (same as NodeThickness) ---
                // Apply exp-based density scale: actual scale = exp(_DensityScale)
                float densityScaleActual = exp(_DensityScale);
                float densityAlpha = saturate(i.density * densityScaleActual);
                
                // Combine all transparency factors
                float3 color = GetLayerColor(i.layer);
                float finalAlpha = totalEdge * _Transparency * depthShade * densityAlpha;
                
                return float4(color, finalAlpha);
            }
            ENDHLSL

            HLSLPROGRAM
            #pragma vertex VS
            #pragma fragment PS
            #pragma target 4.5
            ENDHLSL
        }
    }
}
