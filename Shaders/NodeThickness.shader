Shader "Custom/NodeThickness"
{
    Properties
    {
        _PointSize ("Point Size", Float) = 5.0
        _Color ("Color", Color) = (0, 0.5, 1, 1)
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" "Queue"="Geometry" "RenderPipeline"="UniversalRenderPipeline" }
        Pass
        {
            Tags { "LightMode" = "SRPDefaultUnlit" }
            ZWrite On
            ZTest LEqual
            Cull Off

            HLSLINCLUDE
            #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Common.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/SpaceTransforms.hlsl"
            
            // _WorldSpaceCameraPos is already defined in URP includes, no need to redeclare

            // --- Struct Definitions (Match C# exactly) ---
            struct faceVelocities {
                float left, right, bottom, top, front, back;
            };

            struct Node {
                float3 position;    // 12
                float3 velocity;    // 12
                faceVelocities v;   // 24
                float mass;         // 4
                uint layer;         // 4
                uint mortonCode;    // 4
                uint active;        // 4
            };

            StructuredBuffer<Node> _Nodes;
            float3 _SimulationBoundsMin;
            float3 _SimulationBoundsMax;
            float _PointSize;
            float4 _Color;

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
            };

            VSOut VS(Attributes input)
            {
                VSOut o;
                
                // Get node data using instance ID
                Node node = _Nodes[input.instanceID];
                
                // Convert normalized position (0-1024 range) back to world coordinates
                float3 simulationSize = _SimulationBoundsMax - _SimulationBoundsMin;
                float3 centerWS = _SimulationBoundsMin + (node.position * simulationSize / 1024.0);
                
                // Create billboard quad - face the camera (matching working project approach)
                // Get camera basis vectors from view matrix (like unity_CameraToWorld in Built-in)
                // In URP, we can get this from GetViewToHClipMatrix or construct from camera
                float3 camPos = _WorldSpaceCameraPos;
                float3 viewDir = normalize(camPos - centerWS);
                
                // Get camera right and up from view matrix
                // Alternative: use GetCameraPositionWS and GetViewMatrix() if available
                // For now, construct from view direction (fallback)
                float3 worldUp = float3(0, 1, 0);
                
                // Handle edge case where viewDir is parallel to worldUp
                if (abs(dot(viewDir, worldUp)) > 0.99)
                {
                    worldUp = float3(0, 0, 1);
                }
                
                float3 camRight = normalize(cross(worldUp, viewDir));
                float3 camUp = cross(viewDir, camRight);
                
                // Scale quad by point size (input.positionOS is -0.5 to 0.5 for quad)
                // Match working project: v.vertex * scale * 2
                float size = _PointSize * 0.01f; // Convert to world units
                float3 vertOffset = input.positionOS.xyz * size * 2.0;
                float3 positionWS = centerWS + camRight * vertOffset.x + camUp * vertOffset.y;
                
                o.pos = TransformWorldToHClip(positionWS);
                o.uv = input.uv;
                o.col = _Color;
                
                return o;
            }

            float4 PS(VSOut i) : SV_Target
            {
                return i.col;
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
