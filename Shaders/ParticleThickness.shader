Shader "Custom/ParticleThickness"
{
    SubShader
    {
        // Change Queue to Transparent so it renders after opaque geometry
        Tags { "RenderType"="Transparent" "Queue"="Transparent" "RenderPipeline"="UniversalRenderPipeline" }
        Pass
        {
            Tags { "LightMode" = "SRPDefaultUnlit" }
            
            // 1. ADDITIVE BLENDING: Add this pixel's value to the buffer
            Blend One One
            
            // 2. DISABLE DEPTH WRITE: Allow accumulation without occlusion
            ZWrite Off
            
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
            float3 _SimulationBoundsMin;
            float3 _SimulationBoundsMax;
            float _ThicknessContribution;

            struct VSOut
            {
                float4 pos : SV_POSITION;
                float psize : PSIZE;
            };

            VSOut VS(uint id : SV_VertexID)
            {
                VSOut o;
                Particle p = _Particles[id];
                
                float3 simulationSize = _SimulationBoundsMax - _SimulationBoundsMin;
                float3 positionWS = _SimulationBoundsMin + (p.position * simulationSize / 1024.0);

                o.pos = TransformWorldToHClip(positionWS);

                // 3. FIXED SIZE: Force 4 pixel screen size
                o.psize = 4.0;

                return o;
            }

            float4 PS(VSOut i) : SV_Target
            {
                // Output thickness contribution per particle
                // With additive blending (Blend One One), overlapping particles will accumulate
                // Result: transparent (black) -> white based on number of overlapping particles
                float val = _ThicknessContribution;
                return float4(val, val, val, 1.0);
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

