Shader "Custom/NodeThickness"
{
    Properties
    {
        _PointSize ("Point Size", Float) = 5.0
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
            
            // Additive Blending: Adds up the total "optical depth" of all nodes
            Blend One One 

            HLSLINCLUDE
            #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Common.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/SpaceTransforms.hlsl"
 
            struct faceVelocities { float left, right, bottom, top, front, back; };
            struct Node {
                float3 position; float3 velocity; faceVelocities v;
                float mass; uint layer; uint mortonCode; uint active;
            };

            StructuredBuffer<Node> _Nodes;
            float3 _SimulationBoundsMin;
            float3 _SimulationBoundsMax;
            float _PointSize;

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
                float density : TEXCOORD3; // Passing density to Pixel Shader
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

            VSOut VS(Attributes input)
            {
                VSOut o;
                Node node = _Nodes[input.instanceID];
                
                float3 centerWS = DecodeMorton3D(node);
                float nodeSize = max(_PointSize * exp2((float)node.layer), 0.01);
                
                // --- 1. Calculate Density ---
                // Volume of a cube = size^3
                float volume = nodeSize * nodeSize * nodeSize;
                // Density = Mass / Volume
                // We assume 'active' nodes have mass. 
                float density = (node.mass / max(volume, 0.0001));

                // --- 2. Billboard Calculation ---
                float3 cameraRight = unity_CameraToWorld._m00_m10_m20;
                float3 cameraUp = unity_CameraToWorld._m01_m11_m21;

                // Scale up canvas by 2.0 to fit the rotated cube inside
                float drawSize = nodeSize * 2.0;

                float3 positionWS = centerWS 
                                  + (cameraRight * input.positionOS.x * drawSize) 
                                  + (cameraUp * input.positionOS.y * drawSize);

                o.pos = TransformWorldToHClip(positionWS);
                
                o.centerWS = centerWS;
                o.nodeSize = nodeSize; 
                o.worldPos = positionWS;
                o.density = density; // Pass calculated density

                return o;
            }

            float4 PS(VSOut i) : SV_Target
            {
                // Ray-Box Intersection (Slab Method)
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

                // Distance traveled through THIS specific cube
                float dist = max(0.0, tExit - max(0.0, tEnter));

                // --- 3. Return Optical Depth ---
                // Instead of just returning distance, we multiply by density.
                // High Density Node = Large value added to texture
                // Low Density Node = Small value added to texture
                return float4(dist * i.density, 0, 0, 1);
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