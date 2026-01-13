Shader "Custom/ParticleThickness"
{
    Properties
    {
        _Radius ("Particle Radius", Float) = 0.01
    }
    SubShader
    {
        Tags { "RenderType"="Transparent" "Queue"="Transparent" "RenderPipeline"="UniversalRenderPipeline" }
        Pass
        {
            Tags { "LightMode" = "SRPDefaultUnlit" }
     
            ZWrite Off
            ZTest LEqual
            Cull Off
            
            // Additive Blending: Adds up the total "optical depth" of all particles
            Blend One One 

            HLSLINCLUDE
            #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Common.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.core/ShaderLibrary/SpaceTransforms.hlsl"

            // --- Structs match ParticleDepth ---
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

            struct Attributes
            {
                float4 positionOS : POSITION; // Standard Quad Vertex (-0.5 to 0.5)
                float2 uv : TEXCOORD0;
                uint instanceID : SV_InstanceID;
            };

            struct VSOut
            {
                float4 pos : SV_POSITION;
                float3 centerWS : TEXCOORD0; // World space center
                float2 uv : TEXCOORD1;      // UVs to calculate circle shape
                float radius : TEXCOORD2;    // World space radius
                float3 worldPos : TEXCOORD3; // World space position of this vertex
            };

            // Helper to decode position (Same as ParticleDepth)
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
                o.centerWS = centerWorld;
                o.uv = input.uv;        // 0 to 1
                o.radius = _Radius;
                o.worldPos = positionWS;

                return o;
            }

            float4 PS(VSOut i) : SV_Target
            {
                // 1. Calculate XY coordinates relative to center (-1 to 1)
                float2 uv = (i.uv - 0.5) * 2.0;

                // 2. Discard pixels outside the circle
                float distSq = dot(uv, uv);
                if (distSq > 1.0) discard;

                // 3. Ray-Sphere Intersection to calculate distance through sphere
                float3 rayOrigin = _WorldSpaceCameraPos;
                float3 rayDir = normalize(i.worldPos - _WorldSpaceCameraPos);
                
                // Ray: P(t) = rayOrigin + t * rayDir
                // Sphere: |P - center| = radius
                // Solve: |rayOrigin + t*rayDir - center|^2 = radius^2
                float3 oc = rayOrigin - i.centerWS;
                float a = dot(rayDir, rayDir); // Should be 1.0 if normalized
                float b = 2.0 * dot(oc, rayDir);
                float c = dot(oc, oc) - i.radius * i.radius;
                float discriminant = b * b - 4.0 * a * c;
                
                // If no intersection, discard
                if (discriminant < 0.0) discard;
                
                float sqrtDiscriminant = sqrt(discriminant);
                float t1 = (-b - sqrtDiscriminant) / (2.0 * a);
                float t2 = (-b + sqrtDiscriminant) / (2.0 * a);
                
                // Ensure t1 < t2
                float tEnter = min(t1, t2);
                float tExit = max(t1, t2);
                
                // Distance traveled through the sphere
                float dist = max(0.0, tExit - max(0.0, tEnter));
                
                // 4. Return optical depth (distance through sphere)
                // For particles, we can optionally multiply by a density factor if needed
                // For now, just return the distance
                return float4(dist, 0, 0, 1);
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
