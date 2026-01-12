Shader "Custom/ParticleDepth"
{
    Properties
    {
        _Radius ("Depth Radius", Float) = 0.01
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

            // --- Structs match your C# ---
            struct Particle
            {
                float3 position;
                float3 velocity;
                uint layer;
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
                float3 viewPos : TEXCOORD0; // View space position of the center
                float2 uv : TEXCOORD1;      // UVs to calculate circle shape
                float radius : TEXCOORD2;   // World space radius
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
                float3 centerView = TransformWorldToView(centerWorld);
                
                // Use the radius directly (world space, converted to view space for billboard)
                // Billboard Math: Always face camera
                // In View Space, the camera is at (0,0,0) looking down -Z.
                // We just add the vertex offset (xy) to the center position (xyz).
                // input.positionOS.xy is -0.5 to 0.5, so we multiply by 2*radius to get full diameter
                float3 quadPosView = centerView + float3(input.positionOS.xy * _Radius * 2.0, 0);

                o.pos = TransformWViewToHClip(quadPosView);
                o.viewPos = centerView; // Pass the center, not the vertex!
                o.uv = input.uv;        // 0 to 1
                o.radius = _Radius;     // Use radius directly

                return o;
            }

            float4 PS(VSOut i, out float depthOut : SV_Depth) : SV_Target
            {
                // 1. Calculate XY coordinates relative to center (-1 to 1)
                float2 uv = (i.uv - 0.5) * 2.0;

                // 2. Discard pixels outside the circle
                float distSq = dot(uv, uv);
                if (distSq > 1.0) discard;

                // 3. Calculate Sphere "Thickness" (Z) at this pixel
                // x^2 + y^2 + z^2 = r^2  ->  z = sqrt(r^2 - (x^2 + y^2))
                // We use the UVs to represent x/y relative to radius
                float zOffset = sqrt(1.0 - distSq) * i.radius;

                // 4. Modify Depth
                // The surface of the sphere is CLOSER to the camera than the center.
                // In View Space, "closer" means a LARGER Z value (because Z is negative).
                // Wait, Unity View Space: Forward is -Z. 
                // So Closer = More Positive (e.g. -5 is closer than -10).
                float3 pixelViewPos = i.viewPos;
                pixelViewPos.z += zOffset; 

                // 5. Convert View Space Z to Depth Buffer Value
                float4 clipPos = TransformWViewToHClip(pixelViewPos);
                depthOut = clipPos.z / clipPos.w;

                // Output linear depth for the blur shader
                // (Optional: You can just output the calculated linear depth)
                return -pixelViewPos.z; 
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