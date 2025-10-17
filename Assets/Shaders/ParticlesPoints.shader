Shader "Custom/ParticlesPoints"
{
    Properties
    {
        _PointSize ("Point Size", Float) = 2.0
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

            float4 LayerToColor(uint layer)
            {
                float t = saturate(layer / 10.0);
                float3 c = lerp(float3(1,0,0), float3(0,0,1), t);
                return float4(c, 1);
            }

            struct VSOut
            {
                float4 pos : SV_POSITION;
                float4 col : COLOR0;
                float  psize : PSIZE;
            };

            VSOut VS(uint id : SV_VertexID)
            {
                VSOut o;
                Particle p = _Particles[id];
                float3 positionWS = p.position; // already in world-space
                o.pos = TransformWorldToHClip(positionWS);
                o.col = LayerToColor(p.layer);
                o.psize = max(_PointSize, 1.0);
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
