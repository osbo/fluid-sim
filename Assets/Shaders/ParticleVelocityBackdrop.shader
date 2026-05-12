Shader "Custom/ParticleVelocityBackdrop"
{
    Properties
    {
        _BackgroundTopColor ("Background Top Color", Color) = (0.95, 0.95, 0.95, 1.0)
        _BackgroundBottomColor ("Background Bottom Color", Color) = (0.90, 0.90, 0.90, 1.0)
        _TileColorA ("Tile Color A", Color) = (0.96, 0.96, 0.96, 1.0)
        _TileColorB ("Tile Color B", Color) = (0.88, 0.88, 0.88, 1.0)
        _FloorY ("Floor Y", Float) = -2.5
        _FloorMinXZ ("Floor Min XZ", Vector) = (-1, -1, 0, 0)
        _FloorMaxXZ ("Floor Max XZ", Vector) = (1, 1, 0, 0)
        _CellSize ("Cell Size", Float) = 1.0
        _VerticalGradientStrength ("Vertical Gradient Strength", Float) = 1.0
        _DirectionalBias ("Directional Bias", Float) = 0.06
    }

    SubShader
    {
        Tags { "RenderType"="Opaque" "Queue"="Background" "RenderPipeline"="UniversalPipeline" }
        Cull Off
        ZWrite Off
        ZTest Always

        Pass
        {
            Name "ParticleVelocityBackdrop"

            HLSLINCLUDE
            #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Common.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            struct Attributes
            {
                float4 positionOS : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float2 uv : TEXCOORD0;
            };

            float4 _BackgroundTopColor;
            float4 _BackgroundBottomColor;
            float4 _TileColorA;
            float4 _TileColorB;
            float _FloorY;
            float2 _FloorMinXZ;
            float2 _FloorMaxXZ;
            float _CellSize;
            float _VerticalGradientStrength;
            float _DirectionalBias;

            Varyings vert(Attributes input)
            {
                Varyings o;
                o.positionCS = TransformObjectToHClip(input.positionOS.xyz);
                o.uv = input.uv;
                return o;
            }

            float3 WorldViewDir(float2 uv)
            {
                float4 viewVector = mul(unity_CameraInvProjection, float4(uv * 2.0 - 1.0, 0.0, -1.0));
                return normalize(mul((float3x3)unity_CameraToWorld, viewVector.xyz));
            }

            float Checker(float2 pos, float cellSize)
            {
                float2 c = floor(pos / max(cellSize, 1e-4));
                return fmod(abs(c.x + c.y), 2.0);
            }

            float4 frag(Varyings i) : SV_Target
            {
                float3 viewDir = WorldViewDir(i.uv);
                float3 cameraPos = _WorldSpaceCameraPos;
                float verticalT = saturate((viewDir.y * 0.5 + 0.5) * _VerticalGradientStrength);
                float3 baseCol = lerp(_BackgroundBottomColor.rgb, _BackgroundTopColor.rgb, verticalT);
                float directional = (viewDir.x * 0.5 + 0.5) * _DirectionalBias;
                baseCol = saturate(baseCol + directional.xxx);

                if (viewDir.y < -1e-5)
                {
                    float t = (_FloorY - cameraPos.y) / viewDir.y;
                    if (t > 0.0)
                    {
                        float3 hit = cameraPos + viewDir * t;
                        float2 hitXZ = hit.xz;
                        bool insideX = hitXZ.x >= _FloorMinXZ.x && hitXZ.x <= _FloorMaxXZ.x;
                        bool insideZ = hitXZ.y >= _FloorMinXZ.y && hitXZ.y <= _FloorMaxXZ.y;
                        if (insideX && insideZ)
                        {
                            float checker = Checker(hitXZ, _CellSize);
                            baseCol = lerp(_TileColorA.rgb, _TileColorB.rgb, checker);
                        }
                    }
                }

                return float4(baseCol, 1.0);
            }
            ENDHLSL

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma target 4.5
            ENDHLSL
        }
    }
}
