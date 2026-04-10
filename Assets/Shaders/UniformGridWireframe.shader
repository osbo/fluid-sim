Shader "Custom/UniformGridWireframe"
{
    Properties
    {
        _WireframeWidth ("Wireframe Width (Pixels)", Float) = 2.0
        _Transparency ("Transparency", Range(0.0, 1.0)) = 0.8
        _DensityScale ("Density Scale", Range(-20.0, 20.0)) = -8.9
    }
    SubShader
    {
        Tags { "RenderType"="Transparent" "Queue"="Transparent" "RenderPipeline"="UniversalPipeline" }
        Pass
        {
            Tags { "LightMode" = "SRPDefaultUnlit" }

            ZWrite Off
            ZTest LEqual
            Cull Off
            Blend SrcAlpha One

            HLSLINCLUDE
            #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Common.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/SpaceTransforms.hlsl"

            static const uint kMaxAxisBits = 10u;

            StructuredBuffer<uint> _ActiveMortonList;
            StructuredBuffer<uint> _CellCounts;

            uint _UniformGridLog2K;
            uint _UniformGridCellCount;

            float3 _SimulationBoundsMin;
            float3 _SimulationBoundsMax;
            float _WireframeWidth;
            float _Transparency;
            float _DensityScale;
            float _Scale;
            float _MinValue;

            uint3 Deinterleave10(uint m)
            {
                uint x = 0, y = 0, z = 0;
                for (uint i = 0; i < kMaxAxisBits; i++)
                {
                    x |= ((m >> (3 * i + 0)) & 1u) << i;
                    y |= ((m >> (3 * i + 1)) & 1u) << i;
                    z |= ((m >> (3 * i + 2)) & 1u) << i;
                }
                return uint3(x, y, z);
            }

            uint InterleaveK(uint3 c, uint k)
            {
                uint m = 0;
                for (uint i = 0; i < k; i++)
                {
                    m |= (((c.x >> i) & 1u) << (3 * i + 0));
                    m |= (((c.y >> i) & 1u) << (3 * i + 1));
                    m |= (((c.z >> i) & 1u) << (3 * i + 2));
                }
                return m;
            }

            uint UniformCellMorton(uint fineMorton, uint k)
            {
                uint s = kMaxAxisBits - k;
                uint3 p = Deinterleave10(fineMorton);
                uint3 c = uint3(p.x >> s, p.y >> s, p.z >> s);
                return InterleaveK(c, k);
            }

            struct Attributes
            {
                float4 positionOS : POSITION;
                uint instanceID : SV_InstanceID;
            };

            struct VSOut
            {
                float4 pos : SV_POSITION;
                float3 centerWS : TEXCOORD0;
                float3 nodeSize : TEXCOORD1;
                float3 worldPos : TEXCOORD2;
                float2 screenPos : TEXCOORD4;
                float depth : TEXCOORD5;
                float density : TEXCOORD6;
            };

            float GetEdgeIntensity(float3 hitPointLocal, float width)
            {
                float3 uv = abs(hitPointLocal);
                float3 distToEdge = 0.5 - uv;
                float3 s = distToEdge;
                float min1 = min(s.x, min(s.y, s.z));
                float max1 = max(s.x, max(s.y, s.z));
                float mid = s.x + s.y + s.z - min1 - max1;
                return 1.0 - smoothstep(0.0, width, mid);
            }

            VSOut VS(Attributes input)
            {
                VSOut o;
                uint morton = _ActiveMortonList[input.instanceID];
                uint k = _UniformGridLog2K;
                uint idx = UniformCellMorton(morton, k);
                uint cnt = (idx < _UniformGridCellCount) ? _CellCounts[idx] : 0u;

                uint3 p = Deinterleave10(morton);
                uint s = kMaxAxisBits - k;
                uint3 c = uint3(p.x >> s, p.y >> s, p.z >> s);
                float N = (float)(1u << k);
                float3 centerSim = (float3(c) + 0.5) * (1024.0 / N);

                float3 simulationSize = _SimulationBoundsMax - _SimulationBoundsMin;
                float3 centerWS = _SimulationBoundsMin + centerSim / 1024.0 * simulationSize;
                float3 nodeSize = simulationSize / N;

                float volume = nodeSize.x * nodeSize.y * nodeSize.z;
                float density = (float)cnt / max(volume, 0.0001);

                float3 cameraRight = unity_CameraToWorld._m00_m10_m20;
                float3 cameraUp = unity_CameraToWorld._m01_m11_m21;
                float drawSize = max(nodeSize.x, max(nodeSize.y, nodeSize.z)) * 2.0;

                float3 positionWS = centerWS
                    + (cameraRight * input.positionOS.x * drawSize)
                    + (cameraUp * input.positionOS.y * drawSize);

                o.pos = TransformWorldToHClip(positionWS);
                o.centerWS = centerWS;
                o.nodeSize = nodeSize;
                o.worldPos = positionWS;
                o.screenPos = o.pos.xy / o.pos.w;
                o.depth = distance(centerWS, _WorldSpaceCameraPos.xyz);
                o.density = density;
                return o;
            }

            float4 PS(VSOut i) : SV_Target
            {
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

                if (tExit < tEnter || tExit < 0.0) discard;

                float3 hitEnter = rayOrigin + rayDir * tEnter;
                float3 hitExit = rayOrigin + rayDir * tExit;
                float3 localEnter = (hitEnter - i.centerWS) / i.nodeSize;
                float3 localExit = (hitExit - i.centerWS) / i.nodeSize;

                float3 worldPosDx = ddx(i.worldPos);
                float3 worldPosDy = ddy(i.worldPos);
                float worldUnitsPerPixel = max(length(worldPosDx), length(worldPosDy));
                float pixelsPerWorldUnit = 1.0 / max(worldUnitsPerPixel, 0.0001);
                float pixelsPerLocalUnit = pixelsPerWorldUnit * max(i.nodeSize.x, max(i.nodeSize.y, i.nodeSize.z));
                float wireframeWidthLocal = _WireframeWidth / max(pixelsPerLocalUnit, 0.001);

                float edgeFront = GetEdgeIntensity(localEnter, wireframeWidthLocal);
                float edgeBack = GetEdgeIntensity(localExit, wireframeWidthLocal);
                float totalEdge = max(edgeFront, edgeBack * 0.5);

                if (totalEdge < 0.05) discard;

                float val = i.depth;
                float depthShade = 1.0;
                if (val > 0.0 && val < 10000.0)
                {
                    float adjustedVal = val - _MinValue;
                    float displayVal = adjustedVal * _Scale;
                    depthShade = saturate(displayVal);
                }

                float densityScaleActual = exp(_DensityScale);
                float densityAlpha = saturate(i.density * densityScaleActual);

                float3 color = float3(0.2, 0.85, 1.0);
                float finalAlpha = totalEdge * _Transparency * depthShade * densityAlpha;

                return float4(color, finalAlpha);
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
