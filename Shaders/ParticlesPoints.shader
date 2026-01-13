Shader "Custom/ParticlesPoints"
{
    Properties
    {
        _Radius ("Particle Radius", Float) = 0.01
        _DepthFadeStart ("Depth Fade Start", Float) = 0.0
        _DepthFadeEnd ("Depth Fade End", Float) = 100.0
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
                uint mortonCode;
            };

            StructuredBuffer<Particle> _Particles;
            float _Radius;
            float3 _SimulationBoundsMin;
            float3 _SimulationBoundsMax;
            float _DepthFadeStart;
            float _DepthFadeEnd;
            float _Scale;
            float _MinValue;

            struct Attributes
            {
                float4 positionOS : POSITION; // Standard Quad Vertex (-0.5 to 0.5)
                float2 uv : TEXCOORD0;
                uint instanceID : SV_InstanceID;
            };

            struct VSOut
            {
                float4 pos : SV_POSITION;
                float4 col : COLOR0;
                float2 uv : TEXCOORD0;
                float depth : TEXCOORD1;
            };

            // Helper to decode position (Same as your other shaders)
            float3 GetParticlePos(Particle p)
            {
                 float3 simulationSize = _SimulationBoundsMax - _SimulationBoundsMin;
                 return _SimulationBoundsMin + (p.position * simulationSize / 1024.0);
            }

            VSOut VS(Attributes input)
            {
                VSOut o;
                Particle p = _Particles[input.instanceID];
                
                float3 centerWorld = GetParticlePos(p);
                
                // Billboard Math: Always face camera
                float3 cameraRight = unity_CameraToWorld._m00_m10_m20;
                float3 cameraUp = unity_CameraToWorld._m01_m11_m21;
                
                // Scale up by 2*radius to get full diameter
                float3 positionWS = centerWorld 
                                  + (cameraRight * input.positionOS.x * _Radius * 2.0) 
                                  + (cameraUp * input.positionOS.y * _Radius * 2.0);
                
                o.pos = TransformWorldToHClip(positionWS);
                o.col = float4(0.0, 0.5, 1.0, 1.0);
                o.uv = input.uv;
                
                // Calculate distance from camera to particle
                o.depth = distance(centerWorld, _WorldSpaceCameraPos.xyz);
                
                return o;
            }

            float4 PS(VSOut i) : SV_Target
            {
                // Circle trick: discard pixels outside the circle
                float2 uv = (i.uv - 0.5) * 2.0;
                float distSq = dot(uv, uv);
                if (distSq > 1.0) discard;
                
                // Use the same depth scaling as DebugTextureDisplay
                float val = i.depth;
                
                // Early return: if depth is truly 0 or very large, output black
                if (val == 0.0 || val >= 10000.0) return float4(0, 0, 0, 1);
                
                // Subtract min value before scaling (same as DebugTextureDisplay)
                float adjustedVal = val - _MinValue;
                
                // Scale the adjusted value (same as DebugTextureDisplay)
                float displayVal = adjustedVal * _Scale;
                
                // Clamp to 0-1 range for color intensity
                displayVal = saturate(displayVal);
                
                // Use the scaled depth value to modulate the particle color
                // Near particles (low depth, high displayVal) are brighter
                // Far particles (high depth, low displayVal) are darker
                float4 finalColor = i.col * displayVal;
                
                return finalColor;
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
