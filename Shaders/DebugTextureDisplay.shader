Shader "Custom/DebugTextureDisplay"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _Scale ("Display Scale", Float) = 1.0
        _MinValue ("Min Value", Float) = 0.0
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        Cull Off ZWrite Off ZTest Always

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            struct appdata {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            v2f vert (appdata v) {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }

            sampler2D _MainTex;
            float _Scale;
            float _MinValue;

            float4 frag (v2f i) : SV_Target {
                float val = tex2D(_MainTex, i.uv).r;
                
                // Early return: if depth is truly 0 (before anything) or very large (cleared value), output black
                if (val == 0.0 || val >= 10000.0) return float4(0, 0, 0, 1);
                
                // Subtract min value before scaling
                float adjustedVal = val - _MinValue;
                
                // Visualize: Scale the adjusted value to 0-1 range
                float displayVal = adjustedVal * _Scale;
                
                // Below min (negative after adjustment): green
                if (displayVal < 0.0) return float4(0, 1, 0, 1);
                
                // Above max (clipping): red
                if (displayVal > 1.0) return float4(1, 0, 0, 1);

                // Normal range: grayscale
                return float4(displayVal, displayVal, displayVal, 1);
            }
            ENDCG
        }
    }
}
