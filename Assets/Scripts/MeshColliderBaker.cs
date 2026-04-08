using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using UnityEngine;

// Accurate mesh voxelization: triangle–AABB tests over the simulation bounds grid, then a
// 6-connected flood fill from the grid boundary marks exterior air. Solid = surface voxels
// (touching geometry) or interior cells not reachable from the boundary (closed volumes).
// Results are cached under Application.persistentDataPath when the scene collider setup matches.
public static class MeshColliderBaker
{
    // File header magic bytes: 'S','O','L','1'
    static readonly byte[] FileMagicBytes = { 0x53, 0x4F, 0x4C, 0x31 };

    /// <summary>Linear index [iz*R*R + iy*R + ix]. Values 1 = solid, 0 = free.</summary>
    public static uint[] Bake(GameObject collidersRoot, Bounds simWorldBounds, int resolution, bool useDiskCache = true)
    {
        int R = resolution;
        int total = R * R * R;
        if (collidersRoot == null)
            return new uint[total];

        if (useDiskCache && TryLoadCache(collidersRoot, simWorldBounds, R, total, out uint[] cached))
        {
            Debug.Log($"[MeshColliderBaker] Loaded cached {R}³ solid voxel grid ({total} cells).");
            return cached;
        }

        var triangles = new List<Triangle>(4096);
        CollectWorldTriangles(collidersRoot.transform, triangles);
        if (triangles.Count == 0)
        {
            Debug.LogWarning("[MeshColliderBaker] No mesh geometry under colliders root; grid is empty.");
            var empty = new uint[total];
            if (useDiskCache)
                SaveCache(collidersRoot, simWorldBounds, R, empty);
            return empty;
        }

        var blocked = new BitArray(total);
        Vector3 bMin = simWorldBounds.min;
        Vector3 bSize = simWorldBounds.size;
        Vector3 cell = new Vector3(bSize.x / R, bSize.y / R, bSize.z / R);
        Vector3 half = cell * 0.5f;

        foreach (var tri in triangles)
        {
            Bounds tb = new Bounds(tri.a, Vector3.zero);
            tb.Encapsulate(tri.b);
            tb.Encapsulate(tri.c);

            int ix0 = Mathf.Clamp(Mathf.FloorToInt((tb.min.x - bMin.x) / cell.x), 0, R - 1);
            int ix1 = Mathf.Clamp(Mathf.FloorToInt((tb.max.x - bMin.x) / cell.x), 0, R - 1);
            int iy0 = Mathf.Clamp(Mathf.FloorToInt((tb.min.y - bMin.y) / cell.y), 0, R - 1);
            int iy1 = Mathf.Clamp(Mathf.FloorToInt((tb.max.y - bMin.y) / cell.y), 0, R - 1);
            int iz0 = Mathf.Clamp(Mathf.FloorToInt((tb.min.z - bMin.z) / cell.z), 0, R - 1);
            int iz1 = Mathf.Clamp(Mathf.FloorToInt((tb.max.z - bMin.z) / cell.z), 0, R - 1);

            for (int iz = iz0; iz <= iz1; iz++)
            for (int iy = iy0; iy <= iy1; iy++)
            for (int ix = ix0; ix <= ix1; ix++)
            {
                Vector3 center = new Vector3(
                    bMin.x + (ix + 0.5f) * cell.x,
                    bMin.y + (iy + 0.5f) * cell.y,
                    bMin.z + (iz + 0.5f) * cell.z);
                if (TriangleIntersectsAABB(center, half, tri.a, tri.b, tri.c))
                    blocked.Set(iz * R * R + iy * R + ix, true);
            }
        }

        var outside = new BitArray(total);
        var q = new Queue<int>(Mathf.Min(total, 65536));

        void TryEnqueueBoundary(int ix, int iy, int iz)
        {
            int idx = iz * R * R + iy * R + ix;
            if (blocked.Get(idx) || outside.Get(idx))
                return;
            outside.Set(idx, true);
            q.Enqueue(idx);
        }

        for (int iy = 0; iy < R; iy++)
        for (int ix = 0; ix < R; ix++)
        {
            TryEnqueueBoundary(ix, iy, 0);
            TryEnqueueBoundary(ix, iy, R - 1);
        }
        for (int iz = 0; iz < R; iz++)
        for (int ix = 0; ix < R; ix++)
        {
            TryEnqueueBoundary(ix, 0, iz);
            TryEnqueueBoundary(ix, R - 1, iz);
        }
        for (int iz = 0; iz < R; iz++)
        for (int iy = 0; iy < R; iy++)
        {
            TryEnqueueBoundary(0, iy, iz);
            TryEnqueueBoundary(R - 1, iy, iz);
        }

        while (q.Count > 0)
        {
            int idx = q.Dequeue();
            int ix = idx % R;
            int t = idx / R;
            int iy = t % R;
            int iz = t / R;

            void TryN(int nix, int niy, int niz)
            {
                if ((uint)nix >= (uint)R || (uint)niy >= (uint)R || (uint)niz >= (uint)R)
                    return;
                int n = niz * R * R + niy * R + nix;
                if (blocked.Get(n) || outside.Get(n))
                    return;
                outside.Set(n, true);
                q.Enqueue(n);
            }

            TryN(ix - 1, iy, iz);
            TryN(ix + 1, iy, iz);
            TryN(ix, iy - 1, iz);
            TryN(ix, iy + 1, iz);
            TryN(ix, iy, iz - 1);
            TryN(ix, iy, iz + 1);
        }

        var solid = new uint[total];
        int solidCount = 0;
        for (int i = 0; i < total; i++)
        {
            bool s = blocked.Get(i) || !outside.Get(i);
            if (s)
            {
                solid[i] = 1u;
                solidCount++;
            }
        }

        Debug.Log($"[MeshColliderBaker] Baked {R}³ grid: {solidCount} solid voxels ({100f * solidCount / total:F1}%), {triangles.Count} triangles.");

        if (useDiskCache)
            SaveCache(collidersRoot, simWorldBounds, R, solid);

        return solid;
    }

    struct Triangle
    {
        public Vector3 a, b, c;
    }

    static void CollectWorldTriangles(Transform root, List<Triangle> outTris)
    {
        var filters = root.GetComponentsInChildren<MeshFilter>(includeInactive: false);
        foreach (var mf in filters)
        {
            Mesh mesh = mf.sharedMesh;
            if (mesh == null)
                continue;
            Matrix4x4 M = mf.transform.localToWorldMatrix;
            var verts = mesh.vertices;
            var tris = mesh.triangles;
            for (int i = 0; i < tris.Length; i += 3)
            {
                outTris.Add(new Triangle
                {
                    a = M.MultiplyPoint3x4(verts[tris[i]]),
                    b = M.MultiplyPoint3x4(verts[tris[i + 1]]),
                    c = M.MultiplyPoint3x4(verts[tris[i + 2]])
                });
            }
        }
    }

    /// <summary>SAT triangle vs axis-aligned box (box center, half extents; triangle in same space).</summary>
    static bool TriangleIntersectsAABB(Vector3 boxCenter, Vector3 boxHalf, Vector3 v0, Vector3 v1, Vector3 v2)
    {
        if (boxHalf.x <= 0f || boxHalf.y <= 0f || boxHalf.z <= 0f)
            return false;

        v0 -= boxCenter;
        v1 -= boxCenter;
        v2 -= boxCenter;

        float r = boxHalf.x;
        float t0 = v0.x, t1 = v1.x, t2 = v2.x;
        float mn = Mathf.Min(t0, Mathf.Min(t1, t2));
        float mx = Mathf.Max(t0, Mathf.Max(t1, t2));
        if (mx < -r || mn > r)
            return false;

        r = boxHalf.y;
        t0 = v0.y;
        t1 = v1.y;
        t2 = v2.y;
        mn = Mathf.Min(t0, Mathf.Min(t1, t2));
        mx = Mathf.Max(t0, Mathf.Max(t1, t2));
        if (mx < -r || mn > r)
            return false;

        r = boxHalf.z;
        t0 = v0.z;
        t1 = v1.z;
        t2 = v2.z;
        mn = Mathf.Min(t0, Mathf.Min(t1, t2));
        mx = Mathf.Max(t0, Mathf.Max(t1, t2));
        if (mx < -r || mn > r)
            return false;

        Vector3 e0 = v1 - v0;
        Vector3 e1 = v2 - v1;
        Vector3 e2 = v0 - v2;
        Vector3 n = Vector3.Cross(e0, v2 - v0);
        const float nEps2 = 1e-12f;
        if (n.sqrMagnitude >= nEps2)
        {
            if (!AxisIntervalsOverlap(n, v0, v1, v2, boxHalf))
                return false;
        }

        for (int i = 0; i < 3; i++)
        {
            Vector3 edge = i == 0 ? e0 : (i == 1 ? e1 : e2);
            for (int j = 0; j < 3; j++)
            {
                Vector3 basis = j == 0 ? Vector3.right : (j == 1 ? Vector3.up : Vector3.forward);
                Vector3 ax = Vector3.Cross(edge, basis);
                if (!AxisIntervalsOverlap(ax, v0, v1, v2, boxHalf))
                    return false;
            }
        }

        return true;
    }

    static bool AxisIntervalsOverlap(Vector3 axis, Vector3 v0, Vector3 v1, Vector3 v2, Vector3 h)
    {
        if (axis.sqrMagnitude < 1e-18f)
            return true;
        float p0 = Vector3.Dot(v0, axis);
        float p1 = Vector3.Dot(v1, axis);
        float p2 = Vector3.Dot(v2, axis);
        float triMin = Mathf.Min(p0, Mathf.Min(p1, p2));
        float triMax = Mathf.Max(p0, Mathf.Max(p1, p2));
        float boxRadius = h.x * Mathf.Abs(axis.x) + h.y * Mathf.Abs(axis.y) + h.z * Mathf.Abs(axis.z);
        return !(triMax < -boxRadius || triMin > boxRadius);
    }

    static string CacheDirectory()
    {
        string dir = Path.Combine(Application.persistentDataPath, "FluidSimSolidVoxelCache");
        Directory.CreateDirectory(dir);
        return dir;
    }

    static byte[] BuildKeyBytes(GameObject collidersRoot, Bounds simWorldBounds, int R)
    {
        var mfs = collidersRoot.GetComponentsInChildren<MeshFilter>(includeInactive: false);
        Array.Sort(mfs, (a, b) => a.GetInstanceID().CompareTo(b.GetInstanceID()));

        var sb = new StringBuilder(512);
        IFormatProvider inv = System.Globalization.CultureInfo.InvariantCulture;
        foreach (var mf in mfs)
        {
            if (mf.sharedMesh == null)
                continue;
            Mesh m = mf.sharedMesh;
            sb.Append(m.GetInstanceID());
            sb.Append(',');
            sb.Append(m.vertexCount);
            sb.Append(',');
            sb.Append(m.triangles.Length);
            sb.Append(',');
            Matrix4x4 M = mf.transform.localToWorldMatrix;
            for (int k = 0; k < 16; k++)
                sb.Append(M[k].ToString("R", inv));
            sb.Append(';');
        }

        sb.Append('R').Append(R).Append('|');
        Vector3 mn = simWorldBounds.min, mx = simWorldBounds.max;
        sb.Append(mn.x.ToString("R", inv)).Append(',').Append(mn.y.ToString("R", inv)).Append(',').Append(mn.z.ToString("R", inv)).Append('|');
        sb.Append(mx.x.ToString("R", inv)).Append(',').Append(mx.y.ToString("R", inv)).Append(',').Append(mx.z.ToString("R", inv));

        return Encoding.UTF8.GetBytes(sb.ToString());
    }

    static string CachePath(GameObject collidersRoot, Bounds simWorldBounds, int R)
    {
        byte[] key = BuildKeyBytes(collidersRoot, simWorldBounds, R);
        using (var sha = SHA256.Create())
        {
            byte[] hash = sha.ComputeHash(key);
            var hex = new StringBuilder(hash.Length * 2);
            foreach (byte b in hash)
                hex.Append(b.ToString("x2"));
            return Path.Combine(CacheDirectory(), hex.ToString() + ".vox");
        }
    }

    static bool TryLoadCache(GameObject collidersRoot, Bounds simWorldBounds, int R, int total, out uint[] data)
    {
        data = null;
        string path = CachePath(collidersRoot, simWorldBounds, R);
        if (!File.Exists(path))
            return false;
        try
        {
            byte[] raw = File.ReadAllBytes(path);
            if (raw.Length < 8 + total * 4)
                return false;
            if (raw.Length < 8 || raw[0] != FileMagicBytes[0] || raw[1] != FileMagicBytes[1] ||
                raw[2] != FileMagicBytes[2] || raw[3] != FileMagicBytes[3])
                return false;
            int fileR = BitConverter.ToInt32(raw, 4);
            if (fileR != R)
                return false;
            var arr = new uint[total];
            Buffer.BlockCopy(raw, 8, arr, 0, total * 4);
            data = arr;
            return true;
        }
        catch
        {
            return false;
        }
    }

    static void SaveCache(GameObject collidersRoot, Bounds simWorldBounds, int R, uint[] solid)
    {
        try
        {
            string path = CachePath(collidersRoot, simWorldBounds, R);
            int total = solid.Length;
            var raw = new byte[8 + total * 4];
            FileMagicBytes.CopyTo(raw, 0);
            BitConverter.GetBytes(R).CopyTo(raw, 4);
            Buffer.BlockCopy(solid, 0, raw, 8, total * 4);
            File.WriteAllBytes(path, raw);
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[MeshColliderBaker] Could not write voxel cache: {e.Message}");
        }
    }
}
