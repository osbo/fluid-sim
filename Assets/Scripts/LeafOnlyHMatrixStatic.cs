using System;
using System.Collections.Generic;

/// <summary>
/// Static weak-admissibility H-matrix off-block index triples <c>(r0, c0, S)</c> in leaf space.
/// Matches <c>leafonly/hmatrix.py</c> (<c>precompute_hmatrix_off_buffers</c> + <c>off_blocks_strict_upper</c>)
/// with <c>MAX_NUM_LEAVES</c> and <c>HMATRIX_ETA</c> from <c>leafonly/config.py</c>.
/// </summary>
public static class LeafOnlyHMatrixStatic
{
    /// <summary>Same as <c>FluidLeafOnlyInputs.LeafOnlyMaxMixedSize / LeafOnlyLeafSize</c>.</summary>
    public const int MaxNumLeaves = 8192 / 128;

    /// <summary>Weak admissibility η; must match <c>leafonly.config.HMATRIX_ETA</c>.</summary>
    public const float Eta = 1f;

    private static int[] _r0 = Array.Empty<int>();
    private static int[] _c0 = Array.Empty<int>();
    private static int[] _s = Array.Empty<int>();
    private static int[] _flatInterleaved = Array.Empty<int>();
    private static float[] _flatFloats = Array.Empty<float>();
    private static int _m;
    private static bool _ready;

    public static int NumOffBlocks
    {
        get
        {
            Ensure();
            return _m;
        }
    }

    /// <summary>Length <c>NumOffBlocks</c>; call after <see cref="NumOffBlocks"/> (triggers init).</summary>
    public static ReadOnlySpan<int> OffR0
    {
        get
        {
            Ensure();
            return _r0.AsSpan(0, _m);
        }
    }

    public static ReadOnlySpan<int> OffC0
    {
        get
        {
            Ensure();
            return _c0.AsSpan(0, _m);
        }
    }

    public static ReadOnlySpan<int> OffS
    {
        get
        {
            Ensure();
            return _s.AsSpan(0, _m);
        }
    }

    /// <summary>Length <c>3 * NumOffBlocks</c>: r0,c0,S per block, lex order (same as Python <c>HM_R0_LIST</c> chain).</summary>
    public static ReadOnlySpan<int> FlatInterleavedInt
    {
        get
        {
            Ensure();
            return _flatInterleaved.AsSpan(0, 3 * _m);
        }
    }

    private static void Ensure()
    {
        if (_ready)
            return;
        lock (typeof(LeafOnlyHMatrixStatic))
        {
            if (_ready)
                return;
            Compute();
            _ready = true;
        }
    }

    private static int BitLengthPositive(int n)
    {
        int b = 0;
        while (n > 0)
        {
            b++;
            n >>= 1;
        }
        return b;
    }

    private static void Compute()
    {
        int nu = MaxNumLeaves;
        if (nu <= 0)
        {
            _m = 0;
            _r0 = _c0 = _s = _flatInterleaved = Array.Empty<int>();
            _flatFloats = Array.Empty<float>();
            return;
        }

        int maxLevel = nu > 1 ? BitLengthPositive(nu) - 1 : 0;
        var sMax = new int[nu, nu];
        for (int i = 0; i < nu; i++)
        for (int j = 0; j < nu; j++)
            sMax[i, j] = 1;

        for (int lv = 1; lv <= maxLevel; lv++)
        {
            int S = 1 << lv;
            for (int i = 0; i < nu; i++)
            for (int j = 0; j < nu; j++)
            {
                int r0Cell = i / S * S;
                int c0Cell = j / S * S;
                float d = Math.Abs(r0Cell - c0Cell) - S;
                bool admissible = d > 0f && S <= Eta * d;
                if (admissible)
                    sMax[i, j] = S;
            }
        }

        var triples = new List<(int fr0, int fc0, int fs)>(nu * nu);
        for (int i = 0; i < nu; i++)
        for (int j = 0; j < nu; j++)
        {
            int sm = sMax[i, j];
            int fr0 = i / sm * sm;
            int fc0 = j / sm * sm;
            triples.Add((fr0, fc0, sm));
        }

        triples.Sort(static (a, b) =>
        {
            int c = a.fr0.CompareTo(b.fr0);
            if (c != 0) return c;
            c = a.fc0.CompareTo(b.fc0);
            if (c != 0) return c;
            return a.fs.CompareTo(b.fs);
        });

        var uniq = new List<(int fr0, int fc0, int fs)>(triples.Count);
        for (int k = 0; k < triples.Count; k++)
        {
            var t = triples[k];
            if (uniq.Count == 0 || uniq[uniq.Count - 1] != t)
                uniq.Add(t);
        }

        var off = new List<(int fr0, int fc0, int fs)>();
        foreach (var t in uniq)
        {
            if (t.fr0 == t.fc0 && t.fs == 1)
                continue;
            if (t.fr0 > t.fc0)
                continue;
            off.Add(t);
        }

        _m = off.Count;
        _r0 = new int[_m];
        _c0 = new int[_m];
        _s = new int[_m];
        _flatInterleaved = new int[_m * 3];
        _flatFloats = new float[_m * 3];
        for (int i = 0; i < _m; i++)
        {
            var t = off[i];
            _r0[i] = t.fr0;
            _c0[i] = t.fc0;
            _s[i] = t.fs;
            _flatInterleaved[i * 3] = t.fr0;
            _flatInterleaved[i * 3 + 1] = t.fc0;
            _flatInterleaved[i * 3 + 2] = t.fs;
            _flatFloats[i * 3] = t.fr0;
            _flatFloats[i * 3 + 1] = t.fc0;
            _flatFloats[i * 3 + 2] = t.fs;
        }
    }

    /// <summary>Parity helper: same values as <see cref="FlatInterleavedInt"/> as floats for <c>LeafOnlyParitySummarize</c>.</summary>
    public static float[] GetFlatInterleavedFloats()
    {
        Ensure();
        var copy = new float[_m * 3];
        Array.Copy(_flatFloats, copy, _m * 3);
        return copy;
    }
}
