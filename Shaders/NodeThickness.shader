Shader "Custom/NodeThickness"
{
    Properties
    {
        _PointSize ("Point Size", Float) = 5.0
        // No Color or Absorption here. This shader only outputs raw depth data.
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
            
            // CRITICAL CHANGE: Additive Blending
            // We want to SUM the thickness of all cubes overlapping this pixel.
            Blend One One 

            HLSLINCLUDE
            #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Common.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/SpaceTransforms.hlsl"
 
            // --- Structs (Same as before) ---
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
                float3 centerWS : TEXCOORD3;
                float nodeSize : TEXCOORD4;
                float3 worldPos : TEXCOORD5; 
            };

            // (Helper: DecodeMorton3D same as previous)
            float3 DecodeMorton3D(Node node)
            {
                int gridResolution = (int)exp2(10.0 - (float)node.layer);
                float cellSize = 1024.0 / (float)gridResolution;
                float3 quantizedPos = float3(
                    floor(node.position.x / cellSize) * cellSize + cellSize * 0.5,
                    floor(node.position.y / cellSize) * cellSize + cellSize * 0.5,
                    floor(node.position.z / cellSize) * cellSize + cellSize * 0.5
                );
                float3 simulationSize = _SimulationBoundsMax - _SimulationBoundsMin;
                return _SimulationBoundsMin + float3(
                    quantizedPos.x / 1024.0 * simulationSize.x,
                    quantizedPos.y / 1024.0 * simulationSize.y,
                    quantizedPos.z / 1024.0 * simulationSize.z
                );
            }

            VSOut VS(Attributes input)
            {
                VSOut o;
                Node node = _Nodes[input.instanceID];
                
                float3 centerWS = DecodeMorton3D(node);
                float nodeSize = max(_PointSize * exp2((float)node.layer), 0.01);
                float3 positionWS = centerWS + (input.positionOS.xyz * nodeSize);

                o.pos = TransformWorldToHClip(positionWS);
                
                // Pass bounds info
                o.centerWS = centerWS;
                o.nodeSize = nodeSize;
                o.worldPos = positionWS;
                return o;
            }

            float4 PS(VSOut i) : SV_Target
            {
                // --- Ray-Box Intersection (Slab Method) ---
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

                // Calculate distance through volume
                float dist = max(0.0, tExit - max(0.0, tEnter));

                // RETURN THE THICKNESS
                // We return 'dist' in the Red channel.
                // Since blend is One One, this adds 'dist' to the existing pixel value.
                return float4(dist, 0, 0, 1);
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