Shader "Custom/ParticlesPoints"
{
    Properties
    {
        _PointSize ("Point Size", Float) = 2.0
        _MinLayer ("Min Layer", Int) = 0
        _MaxLayer ("Max Layer", Int) = 10
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
            int _MinLayer;
            int _MaxLayer;
            float3 _SimulationBoundsMin;
            float3 _SimulationBoundsMax;

            float4 LayerToColor(uint layer)
            {
                float t = saturate((float(layer) - float(_MinLayer)) / max(1.0, float(_MaxLayer) - float(_MinLayer)));
                float hue = t * 0.8; // 0.8 gives us red->yellow->green->cyan->blue (not full circle)
                float3 hsv = float3(hue, 1.0, 1.0);
                
                // Convert HSV to RGB
                float3 c = float3(0, 0, 0);
                if (hsv.y == 0.0) {
                    c = float3(hsv.z, hsv.z, hsv.z);
                } else {
                    float h = hsv.x * 6.0;
                    float i = floor(h);
                    float f = h - i;
                    float p = hsv.z * (1.0 - hsv.y);
                    float q = hsv.z * (1.0 - hsv.y * f);
                    float t_val = hsv.z * (1.0 - hsv.y * (1.0 - f));
                    
                    if (i == 0.0) c = float3(hsv.z, t_val, p);
                    else if (i == 1.0) c = float3(q, hsv.z, p);
                    else if (i == 2.0) c = float3(p, hsv.z, t_val);
                    else if (i == 3.0) c = float3(p, q, hsv.z);
                    else if (i == 4.0) c = float3(t_val, p, hsv.z);
                    else c = float3(hsv.z, p, q);
                }
                
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
                
                // Convert normalized position (0-1024 range) back to world coordinates
                float3 simulationSize = _SimulationBoundsMax - _SimulationBoundsMin;
                float3 positionWS = _SimulationBoundsMin + (p.position * simulationSize / 1024.0);
                
                o.pos = TransformWorldToHClip(positionWS);
                o.col = LayerToColor(p.layer);
                o.psize = max(_PointSize, 1.0);
                return o;
            }

            float4 PS(VSOut i) : SV_Target
            {
                return i.col;
                // return float4(0, 0, 0, 0);
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
