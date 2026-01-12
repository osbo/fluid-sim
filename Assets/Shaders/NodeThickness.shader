Shader "Custom/NodeThickness"
{
    Properties
    {
        _PointSize ("Point Size", Float) = 5.0
        _Color ("Color", Color) = (0, 0.5, 1, 1)
        _Absorption ("Absorption Coefficient", Float) = 10.0 // NEW: Controls how fast it gets opaque
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
            Blend SrcAlpha OneMinusSrcAlpha

            HLSLINCLUDE
            #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Common.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/SpaceTransforms.hlsl"
 
            // _WorldSpaceCameraPos is already defined in URP includes

            // --- Struct Definitions (Match C# exactly) ---
            struct faceVelocities {
                float left, right, bottom, top, front, back;
            };

            struct Node {
                float3 position;
                float3 velocity;
                faceVelocities v;
                float mass;
                uint layer;
                uint mortonCode;
                uint active;
            };

            StructuredBuffer<Node> _Nodes;
            float3 _SimulationBoundsMin;
            float3 _SimulationBoundsMax;
            float _PointSize;
            float4 _Color;
            float _Absorption; // NEW

            struct Attributes
            {
                float4 positionOS : POSITION;
                float2 uv : TEXCOORD0;
                uint instanceID : SV_InstanceID;
            };

            struct VSOut
            {
                float4 pos : SV_POSITION;
                float2 uv : TEXCOORD0;
                float4 col : COLOR0;
                float density : TEXCOORD1;
                
                // NEW: Data needed for ray-box intersection
                float3 centerWS : TEXCOORD3;
                float nodeSize : TEXCOORD4;
                float3 worldPos : TEXCOORD5; 
            };

            // Decode morton code position (matching DecodeMorton3D from C#)
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
                float3 quantizedWorldPos = _SimulationBoundsMin + float3(
                    quantizedPos.x / 1024.0 * simulationSize.x,
                    quantizedPos.y / 1024.0 * simulationSize.y,
                    quantizedPos.z / 1024.0 * simulationSize.z
                );
                return quantizedWorldPos;
            }

            VSOut VS(Attributes input)
            {
                VSOut o;
                
                Node node = _Nodes[input.instanceID];
                float3 centerWS = DecodeMorton3D(node);
                
                // Calculate size based on layer
                float nodeSize = _PointSize * exp2((float)node.layer);
                nodeSize = max(nodeSize, 0.01);

                // Calculate volume
                float volume = nodeSize * nodeSize * nodeSize;
                
                // Calculate density
                float density = (node.mass / max(volume, 0.0001));

                // Vertex Position in World Space
                float3 positionWS = centerWS + (input.positionOS.xyz * nodeSize);

                o.pos = TransformWorldToHClip(positionWS);
                o.uv = input.uv;
                o.col = _Color;
                o.density = density;

                // Pass bounds info to pixel shader
                o.centerWS = centerWS;
                o.nodeSize = nodeSize;
                o.worldPos = positionWS;

                return o;
            }

            float4 PS(VSOut i) : SV_Target
            {
                // 1. Setup Ray
                // Ray Origin = Camera Position
                // Ray Direction = Normalized vector from camera to the pixel's world position
                float3 rayOrigin = _WorldSpaceCameraPos;
                float3 rayDir = normalize(i.worldPos - _WorldSpaceCameraPos);

                // 2. Define Box Bounds (Axis Aligned in World Space)
                float3 boxMin = i.centerWS - (i.nodeSize * 0.5);
                float3 boxMax = i.centerWS + (i.nodeSize * 0.5);

                // 3. Ray-Box Intersection (Slab Method)
                // We add a tiny epsilon to rayDir to avoid division by zero
                float3 invDir = 1.0 / (rayDir + 1e-5);
                
                float3 t1 = (boxMin - rayOrigin) * invDir;
                float3 t2 = (boxMax - rayOrigin) * invDir;

                float3 tMin = min(t1, t2);
                float3 tMax = max(t1, t2);

                // The largest of the mins is the entry point
                float tEnter = max(max(tMin.x, tMin.y), tMin.z);
                // The smallest of the maxs is the exit point
                float tExit = min(min(tMax.x, tMax.y), tMax.z);

                // 4. Calculate Distance through Volume
                // max(0, tEnter) handles the camera being INSIDE the cube 
                // (if inside, tEnter is negative, so we start measuring from 0)
                float dist = max(0.0, tExit - max(0.0, tEnter));

                // If dist is 0 (ray missed or grazed), we discard or alpha is 0
                // (Though rasterization usually ensures we only process valid fragments)
                
                // 5. Calculate Volumetric Alpha
                // Beer-Lambert Law approximation: Alpha = 1 - e^(-density * distance)
                // Multiplied by the node density and a tunable absorption coefficient
                float alpha = 1.0 - exp(-dist * i.density * _Absorption);
                
                return float4(i.col.rgb, saturate(alpha));
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