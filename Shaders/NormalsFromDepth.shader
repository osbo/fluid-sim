Shader "Custom/NormalsFromDepth"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
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

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 pos : SV_POSITION;
            };

            v2f vert (appdata v)
            {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }

            sampler2D _MainTex;
            float4 _MainTex_TexelSize;
            
            // Properties set from C#
            float4 _CameraParams; // x: aspect * tanHalfFovY, y: tanHalfFovY
            float4x4 _CameraInvViewMatrix;

            // 1. Reconstruct View Position from Linear Depth
            float3 ReconstructViewPos(float2 uv, float linearDepth)
            {
                // Remap UV from [0,1] to [-1,1]
                float2 p11 = uv * 2.0 - 1.0;
                
                // Calculate View Position components
                // _CameraParams contains the scale factors for the view frustum at z=1
                float3 viewPos;
                viewPos.z = linearDepth; // Linear depth is simply Z
                viewPos.xy = p11 * _CameraParams.xy * linearDepth;
                
                return viewPos;
            }

            float4 frag (v2f i) : SV_Target
            {
                float depth = tex2D(_MainTex, i.uv).r;
                
                // Ignore background (the large clear value we set earlier)
                if (depth > 9000.0) return float4(0, 0, 0, 0); 

                // 2. Reconstruct Position
                float3 viewPos = ReconstructViewPos(i.uv, depth);

                // 3. Compute Derivatives (ddx/ddy) to get face tangent vectors
                float3 ddxPos = ddx(viewPos);
                float3 ddyPos = ddy(viewPos);

                // 4. Cross Product for Normal
                // Note: Cross product order depends on Coordinate System (Left/Right handed)
                // For Unity (Left Handed, Z+ forward): cross(ddy, ddx) usually points OUT.
                float3 viewNormal = normalize(cross(ddxPos, ddyPos));

                // 5. Transform to World Space (Optional, if you need World Normals)
                // Use the 3x3 portion of Inverse View Matrix (rotation only)
                float3 worldNormal = mul((float3x3)_CameraInvViewMatrix, viewNormal);

                return float4(worldNormal * 0.5 + 0.5, 1); // Pack to display
            }
            ENDCG
        }
    }
}