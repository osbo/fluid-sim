Shader "Fluid/ParticleDepth"
{
    Properties
    {
        _PointSize ("Point Size", Float) = 2.0
        _DepthMin ("Depth Min", Float) = 0.0
        _DepthMax ("Depth Max", Float) = 100.0
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

            struct Particle
            {
                float3 position;
                float3 velocity;
                uint layer;
                uint mortonCode;
            };

            StructuredBuffer<Particle> _Particles;
            float _PointSize;
            float3 _SimulationBoundsMin;
            float3 _SimulationBoundsMax;
            float _DepthMin;
            float _DepthMax;

            struct VSOut
            {
                float4 pos : SV_POSITION;
                float3 posWorld : TEXCOORD0;
                float  psize : PSIZE;
            };

            VSOut VS(uint id : SV_VertexID)
            {
                VSOut o;
                Particle p = _Particles[id];
                
                // Convert normalized position (0-1024 range) back to world coordinates
                float3 simulationSize = _SimulationBoundsMax - _SimulationBoundsMin;
                float3 positionWS = _SimulationBoundsMin + (p.position * simulationSize / 1024.0);
                
                o.pos = TransformWorldToHClip(positionWS);
                o.posWorld = positionWS;
                o.psize = max(_PointSize, 1.0);
                
                return o;
            }

            float LinearDepthToUnityDepth(float linearDepth)
            {
                float depth01 = (linearDepth - _ProjectionParams.y) / (_ProjectionParams.z - _ProjectionParams.y);
                return (1.0 - (depth01 * _ZBufferParams.y)) / (depth01 * _ZBufferParams.x);
            }

            float4 PS(VSOut i, out float Depth : SV_Depth) : SV_Target
            {
                // Calculate linear depth from camera to particle
                float3 viewPos = TransformWorldToView(i.posWorld);
                float linearDepth = -viewPos.z;
                
                // Convert to Unity depth buffer format
                Depth = LinearDepthToUnityDepth(linearDepth);
                
                // Normalize depth to 0-1 range using min/max
                float depthRange = _DepthMax - _DepthMin;
                float normalizedDepth = depthRange > 0.0001 ? saturate((linearDepth - _DepthMin) / depthRange) : 0.0;
                
                // Return normalized depth as grayscale color
                return float4(normalizedDepth, normalizedDepth, normalizedDepth, 1.0);
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
