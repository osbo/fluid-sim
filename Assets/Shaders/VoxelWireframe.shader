Shader "Custom/VoxelWireframe"
{
    Properties
    {
        _WireframeWidth ("Wireframe Width (Pixels)", Float) = 2.0
        _Transparency ("Transparency", Range(0.0, 1.0)) = 0.85
        _DensityScale ("Density Scale", Range(-20.0, 20.0)) = 0.0
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

            StructuredBuffer<uint> _SolidVoxels;
            uint _SolidVoxelResolution;
            float3 _SimulationBoundsMin;
            float3 _SimulationBoundsMax;
            float _WireframeWidth;
            float _Transparency;
            float _DensityScale;
            float _Scale;
            float _MinValue;

            struct Attributes
            {
                float4 positionOS : POSITION;
                uint instanceID : SV_InstanceID;
            };

            struct VSOut
            {
                float4 pos : SV_POSITION;
                float3 centerWS : TEXCOORD0;
                float3 voxelSize : TEXCOORD1;
                float3 worldPos : TEXCOORD2;
                float2 screenPos : TEXCOORD3;
                float depth : TEXCOORD4;
                float density : TEXCOORD5;
                float solid : TEXCOORD6;
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
                uint R = _SolidVoxelResolution;
                uint idx = input.instanceID;
                uint vol = R * R * R;
                o.solid = 0.0;

                if (R == 0u || idx >= vol || _SolidVoxels[idx] == 0u)
                {
                    o.pos = float4(0, 0, -1, 1);
                    o.centerWS = float3(0, 0, 0);
                    o.voxelSize = float3(0, 0, 0);
                    o.worldPos = float3(0, 0, 0);
                    o.screenPos = float2(0, 0);
                    o.depth = 0;
                    o.density = 0;
                    return o;
                }

                o.solid = 1.0;
                uint iz = idx / (R * R);
                uint t = idx % (R * R);
                uint iy = t / R;
                uint ix = t % R;

                float3 bmin = _SimulationBoundsMin;
                float3 bmax = _SimulationBoundsMax;
                float3 bsize = bmax - bmin;
                float rf = (float)R;
                float3 cellSize = bsize / rf;
                float3 centerWS = bmin + (float3(ix, iy, iz) + 0.5) * cellSize;

                float3 cameraRight = unity_CameraToWorld._m00_m10_m20;
                float3 cameraUp = unity_CameraToWorld._m01_m11_m21;
                float drawSize = max(cellSize.x, max(cellSize.y, cellSize.z)) * 2.0;

                float3 positionWS = centerWS
                    + (cameraRight * input.positionOS.x * drawSize)
                    + (cameraUp * input.positionOS.y * drawSize);

                o.pos = TransformWorldToHClip(positionWS);
                o.centerWS = centerWS;
                o.voxelSize = cellSize;
                o.worldPos = positionWS;
                o.screenPos = o.pos.xy / o.pos.w;
                o.depth = distance(centerWS, _WorldSpaceCameraPos.xyz);
                o.density = 1.0;
                return o;
            }

            float4 PS(VSOut i) : SV_Target
            {
                if (i.solid < 0.5)
                    discard;

                float3 rayOrigin = _WorldSpaceCameraPos;
                float3 rayDir = normalize(i.worldPos - _WorldSpaceCameraPos);

                float3 boxMin = i.centerWS - (i.voxelSize * 0.5);
                float3 boxMax = i.centerWS + (i.voxelSize * 0.5);

                float3 invDir = 1.0 / (rayDir + 1e-5);
                float3 t1 = (boxMin - rayOrigin) * invDir;
                float3 t2 = (boxMax - rayOrigin) * invDir;
                float3 tMin = min(t1, t2);
                float3 tMax = max(t1, t2);

                float tEnter = max(max(tMin.x, tMin.y), tMin.z);
                float tExit = min(min(tMax.x, tMax.y), tMax.z);

                if (tExit < tEnter || tExit < 0.0)
                    discard;

                float3 hitEnter = rayOrigin + rayDir * tEnter;
                float3 hitExit = rayOrigin + rayDir * tExit;

                float3 localEnter = (hitEnter - i.centerWS) / i.voxelSize;
                float3 localExit = (hitExit - i.centerWS) / i.voxelSize;

                float3 worldPosDx = ddx(i.worldPos);
                float3 worldPosDy = ddy(i.worldPos);
                float worldUnitsPerPixel = max(length(worldPosDx), length(worldPosDy));
                float pixelsPerWorldUnit = 1.0 / max(worldUnitsPerPixel, 0.0001);
                float pixelsPerLocalUnit = pixelsPerWorldUnit * max(i.voxelSize.x, max(i.voxelSize.y, i.voxelSize.z));
                float wireframeWidthLocal = _WireframeWidth / max(pixelsPerLocalUnit, 0.001);

                float edgeFront = GetEdgeIntensity(localEnter, wireframeWidthLocal);
                float edgeBack = GetEdgeIntensity(localExit, wireframeWidthLocal);
                float totalEdge = max(edgeFront, edgeBack * 0.5);

                if (totalEdge < 0.05)
                    discard;

                float depthShade = 1.0;
                float val = i.depth;
                if (val > 0.0 && val < 10000.0)
                {
                    float adjustedVal = val - _MinValue;
                    float displayVal = adjustedVal * _Scale;
                    depthShade = saturate(displayVal);
                }

                float densityScaleActual = exp(_DensityScale);
                float densityAlpha = saturate(i.density * densityScaleActual);

                // Silver metal read for solid collider voxels (axis-aligned cells): face normal + view fresnel.
                float3 absL = abs(localEnter);
                float3 nLocal;
                if (absL.x >= absL.y && absL.x >= absL.z)
                    nLocal = float3(sign(localEnter.x), 0.0, 0.0);
                else if (absL.y >= absL.z)
                    nLocal = float3(0.0, sign(localEnter.y), 0.0);
                else
                    nLocal = float3(0.0, 0.0, sign(localEnter.z));
                float3 N = normalize(nLocal);
                float3 V = normalize(_WorldSpaceCameraPos - hitEnter);
                float NdotV = saturate(dot(N, V));
                float fresnel = pow(1.0 - NdotV, 4.0);
                float3 silverBase = float3(0.72, 0.74, 0.78);
                float3 specular = float3(0.95, 0.96, 1.0) * fresnel;
                float3 color = silverBase * (0.35 + 0.65 * depthShade) + specular * 0.9;
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
