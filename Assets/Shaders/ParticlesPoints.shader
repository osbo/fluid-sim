Shader "Custom/ParticlesPoints"
{
    Properties
    {
        _Radius ("Particle Radius", Float) = 0.01
        _DepthShadeGamma ("Depth Shade Gamma (bounds)", Float) = 1.2
        _DepthLumaFloor ("Depth Luma Floor (HSV V)", Float) = 0.15
        _UseVelocityColor ("Use Velocity Color", Float) = 0
        _VelocityReferenceSpeed ("Velocity Reference Speed", Float) = 10
        _VelocityAutoNormalize ("Velocity Auto Normalize", Float) = 0
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" "Queue"="Geometry" "RenderPipeline"="UniversalPipeline" }
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
            StructuredBuffer<float2> _ParticleVelocityMinMax; // [0] = (minSpeed, maxSpeed) for this frame (GPU)
            float _Radius;
            float3 _SimulationBoundsMin;
            float3 _SimulationBoundsMax;
            float _Scale;
            float _MinValue;
            float _DepthShadeGamma;
            float _DepthLumaFloor;
            float _UseVelocityColor;
            float _VelocityReferenceSpeed;
            float _VelocityAutoNormalize;

            float3 RgbToHsv(float3 rgb)
            {
                float maxc = max(rgb.r, max(rgb.g, rgb.b));
                float minc = min(rgb.r, min(rgb.g, rgb.b));
                float d = maxc - minc;
                float h = 0.0;
                if (d > 1e-6)
                {
                    if (maxc == rgb.r) h = (rgb.g - rgb.b) / d + (rgb.g < rgb.b ? 6.0 : 0.0);
                    else if (maxc == rgb.g) h = (rgb.b - rgb.r) / d + 2.0;
                    else h = (rgb.r - rgb.g) / d + 4.0;
                    h /= 6.0;
                }
                float s = (maxc > 1e-6) ? (d / maxc) : 0.0;
                return float3(h, s, maxc);
            }

            float3 HsvToRgb(float3 c)
            {
                float4 K = float4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
                float3 p = abs(frac(c.xxx + K.xyz) * 6.0 - K.www);
                return c.z * lerp(K.xxx, saturate(p - K.xxx), c.y);
            }

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

                // col.rgb carries HSV (not RGB) so the PS only needs one HsvToRgb call.
                float3 hsv;
                if (_UseVelocityColor > 0.5)
                {
                    float speed = length(p.velocity);
                    float t;
                    if (_VelocityAutoNormalize > 0.5)
                    {
                        float2 mm = _ParticleVelocityMinMax[0];
                        float lo = mm.x;
                        float hi = mm.y;
                        float range = max(hi - lo, 1e-6);
                        t = saturate((speed - lo) / range);
                    }
                    else
                    {
                        float denom = max(_VelocityReferenceSpeed, 1e-5);
                        t = saturate(speed / denom);
                    }
                    float hue = (2.0 / 3.0) * (1.0 - t);
                    hsv = float3(hue, 0.92, 1.0);
                }
                else
                {
                    hsv = float3(0.583, 1.0, 1.0); // RgbToHsv(float3(0, 0.5, 1)) — constant
                }
                o.col = float4(hsv, 1.0);
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
                
                float val = i.depth;
                if (!isfinite(val)) return float4(0, 0, 0, 1);
                
                // Bounds-normalized depth: 0 = near (closest in range), 1 = far
                float tFar = saturate((val - _MinValue) * _Scale);
                float wNear = 1.0 - tFar;
                wNear = pow(max(wNear, 1e-5), max(_DepthShadeGamma, 0.01));
                float floorV = saturate(_DepthLumaFloor);
                float vScale = floorV + (1.0 - floorV) * wNear;
                float3 hsv = i.col.rgb; // VS passes HSV directly
                hsv.z *= vScale;
                float3 outRgb = HsvToRgb(hsv);
                return float4(outRgb, i.col.a);
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
