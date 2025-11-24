Shader "Custom/Raymarching"
{
    Properties
    {
        _BaseStepSize ("Step Size", Float) = 0.039
        _BaseMaxSteps ("Max Steps", Int) = 2048
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

            struct Node
            {
                float3 position;
                float3 velocity;
                uint layer;
                uint mortonCode;
                uint active;
            };

            StructuredBuffer<Node> _NodesBuffer;
            StructuredBuffer<uint> _NeighborsBuffer;
            int _NumNodes;
            float3 _SimulationBoundsMin;
            float3 _SimulationBoundsMax;
            float _BaseStepSize;
            int _BaseMaxSteps;
            int _MinLayer;

            static const uint kNeighborsPerNode = 24;

            float GetStepSize()
            {
                return _BaseStepSize * pow(2.0, (float)_MinLayer);
            }

            int GetMaxSteps()
            {
                return (int)round(_BaseMaxSteps / max(pow(2.0, (float)_MinLayer), 1.0));
            }

            float3 NodeToWorld(float3 nodePosition)
            {
                float3 simulationSize = _SimulationBoundsMax - _SimulationBoundsMin;
                return _SimulationBoundsMin + (nodePosition * simulationSize / 1024.0);
            }

            float3 WorldToNode(float3 worldPosition)
            {
                float3 simulationSize = _SimulationBoundsMax - _SimulationBoundsMin;
                return (worldPosition - _SimulationBoundsMin) * 1024.0 / simulationSize;
            }

            Node GetNode(uint nodeIndex)
            {
                return _NodesBuffer[nodeIndex];
            }

            uint GetNeighbor(uint nodeIndex, uint neighborSlot)
            {
                uint baseIndex = nodeIndex * kNeighborsPerNode;
                return _NeighborsBuffer[baseIndex + neighborSlot];
            }

            struct Attributes
            {
                uint vertexID : SV_VertexID;
            };

            struct Varyings
            {
                float4 positionCS : SV_POSITION;
                float2 uv : TEXCOORD0;
            };

            Varyings Vert(Attributes input)
            {
                Varyings output;
                uint vertexID = input.vertexID;

                output.uv = float2(
                    (vertexID == 1) ? 2.0 : 0.0,
                    (vertexID == 2) ? 2.0 : 0.0
                );

                output.positionCS = float4(
                    (vertexID == 1) ? 3.0 : -1.0,
                    (vertexID == 2) ? 3.0 : -1.0,
                    0.0,
                    1.0
                );

                #if UNITY_UV_STARTS_AT_TOP
                output.uv.y = 1.0 - output.uv.y;
                #endif

                return output;
            }

            float4 Frag(Varyings input) : SV_Target
            {
                float2 screenUV = input.uv;
                float4x4 invVP = mul(UNITY_MATRIX_I_P, UNITY_MATRIX_I_V);

                float4 nearPos = float4(screenUV * 2.0 - 1.0, 0.0, 1.0);
                float4 farPos = float4(screenUV * 2.0 - 1.0, 1.0, 1.0);

                float4 nearWS = mul(invVP, nearPos);
                nearWS /= nearWS.w;
                float4 farWS = mul(invVP, farPos);
                farWS /= farWS.w;

                float3 rayOrigin = nearWS.xyz;
                float3 rayDir = normalize(farWS.xyz - nearWS.xyz);

                float3 boundsMin = _SimulationBoundsMin;
                float3 boundsMax = _SimulationBoundsMax;

                float3 invDir = 1.0 / max(abs(rayDir), 1e-6) * sign(rayDir);
                float3 t0 = (boundsMin - rayOrigin) * invDir;
                float3 t1 = (boundsMax - rayOrigin) * invDir;
                float3 tmin = min(t0, t1);
                float3 tmax = max(t0, t1);

                float t_entry = max(max(tmin.x, tmin.y), tmin.z);
                float t_exit = min(min(tmax.x, tmax.y), tmax.z);

                if (t_entry > t_exit || t_exit < 0.0)
                {
                    discard;
                }

                float t = max(t_entry, 0.0);
                float steps = 0.0;

                const float stepSize = max(GetStepSize(), 1e-4);
                const int maxSteps = GetMaxSteps();

                for (int i = 0; i < maxSteps; ++i)
                {
                    if (t > t_exit)
                        break;

                    float3 worldPos = rayOrigin + rayDir * t;
                    float3 nodePos = WorldToNode(worldPos);
                    float3 reconstructedWorldPos = NodeToWorld(nodePos);

                    // Touch buffers so they stay wired into the pipeline
                    if (_NumNodes > 0)
                    {
                        Node n = GetNode(0);
                        n.layer += 0; // no-op to keep compiler quiet
                        uint neighborIdx = GetNeighbor(0, 0);
                        neighborIdx += 0;
                    }

                    t += stepSize;
                    steps += 1.0;

                    worldPos += reconstructedWorldPos * 0.0; // placeholder use
                }

                float marchProgress = saturate((t - t_entry) / max(t_exit - t_entry, 1e-3));
                return float4(marchProgress.xxx, 1.0);
            }
            ENDHLSL

            HLSLPROGRAM
            #pragma vertex Vert
            #pragma fragment Frag
            #pragma target 4.5
            ENDHLSL
        }
    }
}

