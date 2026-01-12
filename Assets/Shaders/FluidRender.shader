Shader "Custom/FluidRender"
{
    Properties
    {
        _MainTex ("Background Texture", 2D) = "white" {}
    }
    SubShader
    {
        Cull Off ZWrite Off ZTest Always

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            // ... (Keep Structs and Vertex Shader the same) ...
            
            struct appdata {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            v2f vert(appdata v) {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }

            // --- INPUT TEXTURES ---
            sampler2D _MainTex;      
            sampler2D _NormalTex;    
            sampler2D _DepthTex;     
            sampler2D _ThicknessTex; 
            sampler2D _DepthRawTex;  
            
            // --- PARAMETERS ---
            float3 extinctionCoefficients;
            float3 dirToSun;
            float3 boundsSize;
            float refractionMultiplier;
            
            // Scaling Params (New!)
            float depthDisplayScale;     // Exponent for Depth Scale
            float depthMinValue;         // Exponent for Depth Min
            float thicknessDisplayScale; // Exponent for Thickness Scale
            
            // Environment / Tile settings
            float4 tileCol1;
            float4 tileCol2;
            float3 floorPos;
            float3 floorSize;
            float sunIntensity;

            // ... (Keep Helper Functions: HitInfo, RayBox, SampleEnvironment, Refract, etc.) ...
            
            struct HitInfo {
                bool didHit;
                float dst;
                float3 hitPoint;
            };

            struct LightResponse {
                float3 reflectDir;
                float3 refractDir;
                float reflectWeight;
                float refractWeight;
            };

            float3 WorldViewDir(float2 uv) {
                float3 viewVector = mul(unity_CameraInvProjection, float4(uv.xy * 2 - 1, 0, -1));
                return normalize(mul(unity_CameraToWorld, viewVector));
            }

            HitInfo RayBox(float3 rayPos, float3 rayDir, float3 centre, float3 size) {
                float3 boxMin = centre - size * 0.5;
                float3 boxMax = centre + size * 0.5;
                float3 invDir = 1.0 / rayDir;
                float3 tMin = (boxMin - rayPos) * invDir;
                float3 tMax = (boxMax - rayPos) * invDir;
                float3 t1 = min(tMin, tMax);
                float3 t2 = max(tMin, tMax);
                float tNear = max(max(t1.x, t1.y), t1.z);
                float tFar = min(min(t2.x, t2.y), t2.z);
                
                HitInfo hit;
                hit.didHit = tFar >= tNear && tFar > 0;
                hit.dst = hit.didHit ? (tNear > 0 ? tNear : tFar) : 1e30;
                hit.hitPoint = rayPos + rayDir * hit.dst;
                return hit;
            }

            float3 SampleEnvironment(float3 pos, float3 dir) {
                // HitInfo floorHit = RayBox(pos, dir, floorPos, floorSize);
                // if (floorHit.didHit) {
                //     float2 tile = floor(floorHit.hitPoint.xz * 2.0);
                //     bool isDark = fmod(abs(tile.x + tile.y), 2.0) > 0.5;
                //     return isDark ? tileCol1.rgb : tileCol2.rgb;
                // }
                float sun = pow(max(0, dot(dir, dirToSun)), 500.0) * sunIntensity;
                return lerp(float3(0.1, 0.1, 0.15), float3(0.4, 0.6, 0.9), dir.y * 0.5 + 0.5) + sun;
            }

            float CalculateReflectance(float3 inDir, float3 normal, float iorA, float iorB) {
                float r0 = (iorA - iorB) / (iorA + iorB);
                r0 *= r0;
                float cosX = -dot(inDir, normal);
                if (iorA > iorB) {
                    float n = iorA / iorB;
                    float sinT2 = n * n * (1.0 - cosX * cosX);
                    if (sinT2 > 1.0) return 1.0; 
                    cosX = sqrt(1.0 - sinT2);
                }
                float x = 1.0 - cosX;
                return r0 + (1.0 - r0) * pow(x, 5.0);
            }

            float3 Refract(float3 inDir, float3 normal, float iorA, float iorB) {
                float eta = iorA / iorB;
                float cosI = -dot(inDir, normal);
                float k = 1.0 - eta * eta * (1.0 - cosI * cosI);
                return k < 0.0 ? 0.0 : eta * inDir + (eta * cosI - sqrt(k)) * normal;
            }

            LightResponse CalculateReflectionAndRefraction(float3 inDir, float3 normal, float iorA, float iorB) {
                LightResponse r;
                r.reflectWeight = CalculateReflectance(inDir, normal, iorA, iorB);
                r.refractWeight = 1.0 - r.reflectWeight;
                r.reflectDir = reflect(inDir, normal);
                r.refractDir = Refract(inDir, normal, iorA, iorB);
                return r;
            }

            float4 frag(v2f i) : SV_Target {
                // 1. Sample Raw Values
                float3 normal = tex2D(_NormalTex, i.uv).xyz * 2.0 - 1.0; // Unpack Normal (0..1 -> -1..1)
                float rawDepth = tex2D(_DepthTex, i.uv).r;
                float rawThickness = tex2D(_ThicknessTex, i.uv).r;
                float3 bgCol = tex2D(_MainTex, i.uv).rgb;

                // 2. Early Exit (No Fluid)
                if (rawDepth > 9000.0) return float4(bgCol, 1);

                // --- APPLY SCALING (Using your tuned params) ---
                // Convert stored exponents to actual multipliers
                float dMin = exp(depthMinValue);
                float dScale = exp(depthDisplayScale);
                float tScale = exp(thicknessDisplayScale);

                // Apply logic: (Val - Min) * Scale
                // Note: For physics, we usually want real world units. 
                // If your 'tuned' look is 0-1, we might need to be careful here.
                // Assuming your debug view showed "Correct Physics" when mapped to 0-1:
                
                // Thickness needs to be positive and non-zero for absorption
                float thickness = max(0, rawThickness * tScale);
                // float displayVal = thickness;
                
                // Depth is distance from camera.
                float depthSmooth = rawDepth;
                // float depthSmooth = (rawDepth - dMin) * dScale;
                // float displayVal = depthSmooth;

                // 3. View Ray
                float3 viewDir = WorldViewDir(i.uv);
                float3 hitPos = _WorldSpaceCameraPos + viewDir * depthSmooth;

                // 4. Lighting & Shading
                LightResponse lr = CalculateReflectionAndRefraction(viewDir, normal, 1.0, 1.33);
                
                // Refraction (Beer's Law)
                float3 refractPos = hitPos + lr.refractDir * thickness * refractionMultiplier;
                float3 refractCol = SampleEnvironment(refractPos, viewDir); 
                
                float3 transmission = exp(-thickness * extinctionCoefficients);
                refractCol *= transmission;

                // Reflection
                float3 reflectCol = SampleEnvironment(hitPos, lr.reflectDir);
                float specular = pow(max(0, dot(lr.reflectDir, dirToSun)), 100.0) * sunIntensity;
                reflectCol += specular;

                // 5. Composite
                float3 finalCol = lerp(refractCol, reflectCol, lr.reflectWeight);

                // return float4(displayVal, displayVal, displayVal, 1);
                return float4(finalCol, 1);
            }
            ENDCG
        }
    }
}