using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using UnityEngine;

// Accurate mesh voxelization: triangle–AABB tests over the simulation bounds grid, then a
// 6-connected flood fill from the grid boundary marks exterior air. Solid = any cell not
// reachable through empty space from the domain boundary (closed mesh interiors) plus
// full volumes for Box/Sphere/Capsule colliders (flood-fill obstacles, SDF seeds on their surface).
// Per-voxel normals: area-weighted mesh normals on geometry hits, oriented toward exterior
// air; interior-only voxels use a sum of directions to air neighbors, then optional
// neighbor dilation. Final normals are scaled by 1/boundsSize and normalized so they match
// particle motion in 0–1023 simulation space (axis-aligned box).
// Cache: Application.persistentDataPath/FluidSimSolidVoxelCache/*.vox (magic SOL4).
public static class MeshColliderBaker
{
    // File header magic bytes: 'S','O','L','4' (solid + normals + SDF; v4 = prim volume + split SDF shell)
    static readonly byte[] FileMagicBytes = { 0x53, 0x4F, 0x4C, 0x34 };
    const int HeaderSize = 8;

    public struct BakeResult
    {
        public uint[]    Solid;
        public Vector3[] Normals;
        public float[]   SDF;   // signed distance in sim units (0-1023 space); negative = inside solid
    }

    /// <summary>Linear index [iz*R*R + iy*R + ix]. Solid 1 = obstacle; normals for air are zero.</summary>
    public static BakeResult Bake(GameObject collidersRoot, Bounds simWorldBounds, int resolution, bool useDiskCache = true)
    {
        int R = resolution;
        int total = R * R * R;
        var empty = new BakeResult
        {
            Solid   = new uint[total],
            Normals = new Vector3[total],
            SDF     = new float[total]   // all zeros → treated as surface everywhere (safe default)
        };

        if (collidersRoot == null)
            return empty;

        if (useDiskCache && TryLoadCache(collidersRoot, simWorldBounds, R, total, out BakeResult cached))
        {
            Debug.Log($"[MeshColliderBaker] Loaded cached {R}³ solid + normals ({total} cells).");
            return cached;
        }

        var triangles = new List<Triangle>(4096);
        CollectWorldTriangles(collidersRoot.transform, triangles);

        Vector3 bMin = simWorldBounds.min;
        Vector3 bSize = simWorldBounds.size;
        Vector3 cell = new Vector3(bSize.x / R, bSize.y / R, bSize.z / R);
        Vector3 half = cell * 0.5f;

        bool anyPrimitive = HasPrimitiveColliders(collidersRoot.transform);
        if (triangles.Count == 0 && !anyPrimitive)
        {
            Debug.LogWarning("[MeshColliderBaker] No mesh geometry or primitive colliders under colliders root; grid is empty.");
            if (useDiskCache)
                SaveCache(collidersRoot, simWorldBounds, R, empty.Solid, empty.Normals, empty.SDF);
            return empty;
        }

        var triHit = new BitArray(total);
        var normalAccum = new Vector3[total];

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

            Vector3 ab = tri.b - tri.a;
            Vector3 ac = tri.c - tri.a;
            Vector3 triCross = Vector3.Cross(ab, ac);
            float len2 = triCross.sqrMagnitude;
            Vector3 triN = Vector3.zero;
            float areaW = 0f;
            if (len2 > 1e-20f)
            {
                float len = Mathf.Sqrt(len2);
                triN = triCross / len;
                areaW = 0.5f * len;
            }

            for (int iz = iz0; iz <= iz1; iz++)
            for (int iy = iy0; iy <= iy1; iy++)
            for (int ix = ix0; ix <= ix1; ix++)
            {
                Vector3 center = new Vector3(
                    bMin.x + (ix + 0.5f) * cell.x,
                    bMin.y + (iy + 0.5f) * cell.y,
                    bMin.z + (iz + 0.5f) * cell.z);
                if (!TriangleIntersectsAABB(center, half, tri.a, tri.b, tri.c))
                    continue;
                int idx = iz * R * R + iy * R + ix;
                triHit.Set(idx, true);
                if (areaW > 0f)
                    normalAccum[idx] += triN * areaW;
            }
        }

        // Flood-fill cannot traverse primitive volumes or mesh shell voxels; SDF seeds use triHit ∪ ∂(flood).
        var floodObstacle = (BitArray)triHit.Clone();
        MarkPrimitiveColliderVolumes(collidersRoot.transform, simWorldBounds, cell, R, floodObstacle);

        var sdfSurface = (BitArray)triHit.Clone();
        AddFloodObstacleBoundaryToSdfSurface(R, total, floodObstacle, sdfSurface);

        var outside = new BitArray(total);
        var q = new Queue<int>(Mathf.Min(total, 65536));

        void TryEnqueueBoundary(int ix, int iy, int iz)
        {
            int idx = iz * R * R + iy * R + ix;
            if (floodObstacle.Get(idx) || outside.Get(idx))
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
                if (floodObstacle.Get(n) || outside.Get(n))
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
            bool s = floodObstacle.Get(i) || !outside.Get(i);
            if (s)
            {
                solid[i] = 1u;
                solidCount++;
            }
        }

        // BFS signed distance from sdfSurface; sign from solid mask (negative = inside obstacle).
        float[] sdf = BuildSDF(R, total, sdfSurface, solid, 1024f / R);

        Vector3[] normals = BuildSolidNormals(R, total, solid, normalAccum, bMin, cell);

        // Map world normals to simulation-space directions (uniform dot product with sim velocity).
        Vector3 invSize = new Vector3(
            bSize.x > 1e-8f ? 1f / bSize.x : 1f,
            bSize.y > 1e-8f ? 1f / bSize.y : 1f,
            bSize.z > 1e-8f ? 1f / bSize.z : 1f);
        for (int i = 0; i < total; i++)
        {
            if (solid[i] == 0u)
                continue;
            Vector3 n = normals[i];
            if (n.sqrMagnitude < 1e-14f)
                continue;
            Vector3 nSim = new Vector3(n.x * invSize.x, n.y * invSize.y, n.z * invSize.z);
            if (nSim.sqrMagnitude > 1e-14f)
                normals[i] = nSim.normalized;
        }

        Debug.Log($"[MeshColliderBaker] Baked {R}³ grid: {solidCount} solid voxels ({100f * solidCount / total:F1}%), {triangles.Count} triangles.");

        if (useDiskCache)
            SaveCache(collidersRoot, simWorldBounds, R, solid, normals, sdf);

        return new BakeResult { Solid = solid, Normals = normals, SDF = sdf };
    }

    // BFS distance transform: spread from sdfSurface seeds outward in all directions.
    // Returns signed distance in sim units: positive = outside solid, negative = inside solid.
    static float[] BuildSDF(int R, int total, BitArray sdfSurface, uint[] solid, float voxelSimSize)
    {
        var dist = new int[total];
        for (int i = 0; i < total; i++) dist[i] = int.MaxValue / 2;

        var bfsQ = new Queue<int>(total / 4 + 64);
        for (int i = 0; i < total; i++)
        {
            if (sdfSurface.Get(i)) { dist[i] = 0; bfsQ.Enqueue(i); }
        }

        while (bfsQ.Count > 0)
        {
            int idx = bfsQ.Dequeue();
            int ix = idx % R, t = idx / R, iy = t % R, iz = t / R;
            int d1 = dist[idx] + 1;

            void TryN(int nix, int niy, int niz)
            {
                if ((uint)nix >= (uint)R || (uint)niy >= (uint)R || (uint)niz >= (uint)R) return;
                int n = niz * R * R + niy * R + nix;
                if (dist[n] <= d1) return;
                dist[n] = d1;
                bfsQ.Enqueue(n);
            }
            TryN(ix - 1, iy, iz); TryN(ix + 1, iy, iz);
            TryN(ix, iy - 1, iz); TryN(ix, iy + 1, iz);
            TryN(ix, iy, iz - 1); TryN(ix, iy, iz + 1);
        }

        var sdf = new float[total];
        for (int i = 0; i < total; i++)
        {
            float d = dist[i] * voxelSimSize;
            sdf[i] = solid[i] != 0u ? -d : d;
        }
        return sdf;
    }

    static Vector3 EgressWorld(int idx, int R, uint[] solid, Vector3 bMin, Vector3 cell)
    {
        IdxToIJK(idx, R, out int ix, out int iy, out int iz);
        Vector3 c = CellCenter(ix, iy, iz, bMin, cell);
        Vector3 sum = Vector3.zero;

        void AddAirNeighbor(int nix, int niy, int niz)
        {
            if ((uint)nix >= (uint)R || (uint)niy >= (uint)R || (uint)niz >= (uint)R)
                return;
            int n = niz * R * R + niy * R + nix;
            if (solid[n] != 0u)
                return;
            sum += CellCenter(nix, niy, niz, bMin, cell) - c;
        }

        AddAirNeighbor(ix - 1, iy, iz);
        AddAirNeighbor(ix + 1, iy, iz);
        AddAirNeighbor(ix, iy - 1, iz);
        AddAirNeighbor(ix, iy + 1, iz);
        AddAirNeighbor(ix, iy, iz - 1);
        AddAirNeighbor(ix, iy, iz + 1);

        return sum;
    }

    static Vector3[] BuildSolidNormals(int R, int total, uint[] solid, Vector3[] normalAccum, Vector3 bMin, Vector3 cell)
    {
        var normals = new Vector3[total];
        const float eps2 = 1e-12f;

        for (int i = 0; i < total; i++)
        {
            if (solid[i] == 0u)
                continue;
            Vector3 egress = EgressWorld(i, R, solid, bMin, cell);
            if (normalAccum[i].sqrMagnitude > eps2)
            {
                Vector3 n = normalAccum[i].normalized;
                if (egress.sqrMagnitude > eps2 && Vector3.Dot(n, egress) < 0f)
                    n = -n;
                normals[i] = n;
            }
            else if (egress.sqrMagnitude > eps2)
                normals[i] = egress.normalized;
        }

        // Dilate from labeled surface into any remaining solid voxels (deep interior).
        bool changed = true;
        int guard = 0;
        while (changed && guard < R + 2)
        {
            guard++;
            changed = false;
            for (int i = 0; i < total; i++)
            {
                if (solid[i] == 0u || normals[i].sqrMagnitude > eps2)
                    continue;
                Vector3 avg = Vector3.zero;
                int count = 0;
                IdxToIJK(i, R, out int ix, out int iy, out int iz);
                void TryNbor(int nix, int niy, int niz)
                {
                    if ((uint)nix >= (uint)R || (uint)niy >= (uint)R || (uint)niz >= (uint)R)
                        return;
                    int j = niz * R * R + niy * R + nix;
                    if (solid[j] == 0u || normals[j].sqrMagnitude < eps2)
                        return;
                    avg += normals[j];
                    count++;
                }
                TryNbor(ix - 1, iy, iz);
                TryNbor(ix + 1, iy, iz);
                TryNbor(ix, iy - 1, iz);
                TryNbor(ix, iy + 1, iz);
                TryNbor(ix, iy, iz - 1);
                TryNbor(ix, iy, iz + 1);
                if (count > 0)
                {
                    normals[i] = (avg / count).normalized;
                    changed = true;
                }
            }
        }

        for (int i = 0; i < total; i++)
        {
            if (solid[i] == 0u)
                continue;
            if (normals[i].sqrMagnitude < eps2)
                normals[i] = Vector3.up;
        }

        return normals;
    }

    static void IdxToIJK(int idx, int R, out int ix, out int iy, out int iz)
    {
        ix = idx % R;
        int t = idx / R;
        iy = t % R;
        iz = t / R;
    }

    static Vector3 CellCenter(int ix, int iy, int iz, Vector3 bMin, Vector3 cell)
    {
        return new Vector3(
            bMin.x + (ix + 0.5f) * cell.x,
            bMin.y + (iy + 0.5f) * cell.y,
            bMin.z + (iz + 0.5f) * cell.z);
    }

    struct Triangle
    {
        public Vector3 a, b, c;
    }

    static bool HasPrimitiveColliders(Transform root)
    {
        if (root.GetComponentsInChildren<BoxCollider>(includeInactive: false).Length > 0) return true;
        if (root.GetComponentsInChildren<SphereCollider>(includeInactive: false).Length > 0) return true;
        if (root.GetComponentsInChildren<CapsuleCollider>(includeInactive: false).Length > 0) return true;
        return false;
    }

    static bool WorldPointInsideBoxCollider(BoxCollider bc, Vector3 world)
    {
        Vector3 local = bc.transform.InverseTransformPoint(world) - bc.center;
        Vector3 half = bc.size * 0.5f;
        const float eps = 1e-5f;
        return Mathf.Abs(local.x) <= half.x + eps && Mathf.Abs(local.y) <= half.y + eps && Mathf.Abs(local.z) <= half.z + eps;
    }

    static bool WorldPointInsideSphereCollider(SphereCollider sc, Vector3 world)
    {
        Vector3 local = sc.transform.InverseTransformPoint(world) - sc.center;
        float r = sc.radius;
        return local.sqrMagnitude <= r * r + 1e-8f;
    }

    static bool WorldPointInsideCapsuleCollider(CapsuleCollider c, Vector3 world)
    {
        Transform t = c.transform;
        Vector3 ctr = t.TransformPoint(c.center);
        Vector3 ax = c.direction == 0 ? t.right : (c.direction == 1 ? t.up : t.forward);
        float cylHalf = Mathf.Max(0f, c.height * 0.5f - c.radius);
        Vector3 p0 = ctr - ax * cylHalf;
        Vector3 p1 = ctr + ax * cylHalf;
        Vector3 ab = p1 - p0;
        float den = ab.sqrMagnitude + 1e-12f;
        float u = Mathf.Clamp01(Vector3.Dot(world - p0, ab) / den);
        Vector3 closest = p0 + u * ab;
        return (world - closest).sqrMagnitude <= c.radius * c.radius + 1e-8f;
    }

    static void MarkCellsInWorldBoundsOverlap(Vector3 simMin, Vector3 cell, int R, BitArray flood,
        Bounds objWorldBounds, Func<Vector3, bool> insideWorld)
    {
        Bounds wb = objWorldBounds;
        float pad = Mathf.Max(cell.x, Mathf.Max(cell.y, cell.z)) * 0.51f;
        wb.Expand(pad);

        int ix0 = Mathf.Clamp(Mathf.FloorToInt((wb.min.x - simMin.x) / cell.x), 0, R - 1);
        int ix1 = Mathf.Clamp(Mathf.CeilToInt((wb.max.x - simMin.x) / cell.x) - 1, 0, R - 1);
        int iy0 = Mathf.Clamp(Mathf.FloorToInt((wb.min.y - simMin.y) / cell.y), 0, R - 1);
        int iy1 = Mathf.Clamp(Mathf.CeilToInt((wb.max.y - simMin.y) / cell.y) - 1, 0, R - 1);
        int iz0 = Mathf.Clamp(Mathf.FloorToInt((wb.min.z - simMin.z) / cell.z), 0, R - 1);
        int iz1 = Mathf.Clamp(Mathf.CeilToInt((wb.max.z - simMin.z) / cell.z) - 1, 0, R - 1);

        for (int iz = iz0; iz <= iz1; iz++)
        for (int iy = iy0; iy <= iy1; iy++)
        for (int ix = ix0; ix <= ix1; ix++)
        {
            Vector3 center = new Vector3(
                simMin.x + (ix + 0.5f) * cell.x,
                simMin.y + (iy + 0.5f) * cell.y,
                simMin.z + (iz + 0.5f) * cell.z);
            if (!insideWorld(center))
                continue;
            int idx = iz * R * R + iy * R + ix;
            flood.Set(idx, true);
        }
    }

    static void MarkPrimitiveColliderVolumes(Transform root, Bounds simWorldBounds, Vector3 cell, int R, BitArray flood)
    {
        Vector3 simMin = simWorldBounds.min;

        foreach (var bc in root.GetComponentsInChildren<BoxCollider>(includeInactive: false))
        {
            if (!bc.enabled || !bc.gameObject.activeInHierarchy) continue;
            MarkCellsInWorldBoundsOverlap(simMin, cell, R, flood, bc.bounds, p => WorldPointInsideBoxCollider(bc, p));
        }
        foreach (var sc in root.GetComponentsInChildren<SphereCollider>(includeInactive: false))
        {
            if (!sc.enabled || !sc.gameObject.activeInHierarchy) continue;
            MarkCellsInWorldBoundsOverlap(simMin, cell, R, flood, sc.bounds, p => WorldPointInsideSphereCollider(sc, p));
        }
        foreach (var cap in root.GetComponentsInChildren<CapsuleCollider>(includeInactive: false))
        {
            if (!cap.enabled || !cap.gameObject.activeInHierarchy) continue;
            MarkCellsInWorldBoundsOverlap(simMin, cell, R, flood, cap.bounds, p => WorldPointInsideCapsuleCollider(cap, p));
        }
    }

    static void AddFloodObstacleBoundaryToSdfSurface(int R, int total, BitArray floodObstacle, BitArray sdfSurface)
    {
        for (int i = 0; i < total; i++)
        {
            if (!floodObstacle[i])
                continue;
            int ix = i % R;
            int t = i / R;
            int iy = t % R;
            int iz = t / R;

            bool NeighborIsAirOrOob(int nx, int ny, int nz)
            {
                if ((uint)nx >= (uint)R || (uint)ny >= (uint)R || (uint)nz >= (uint)R)
                    return true;
                int ni = nz * R * R + ny * R + nx;
                return !floodObstacle[ni];
            }

            if (NeighborIsAirOrOob(ix - 1, iy, iz) || NeighborIsAirOrOob(ix + 1, iy, iz) ||
                NeighborIsAirOrOob(ix, iy - 1, iz) || NeighborIsAirOrOob(ix, iy + 1, iz) ||
                NeighborIsAirOrOob(ix, iy, iz - 1) || NeighborIsAirOrOob(ix, iy, iz + 1))
                sdfSurface.Set(i, true);
        }
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

        var meshColliders = root.GetComponentsInChildren<MeshCollider>(includeInactive: false);
        foreach (var mc in meshColliders)
        {
            if (!mc.enabled || !mc.gameObject.activeInHierarchy || mc.sharedMesh == null)
                continue;
            Mesh mesh = mc.sharedMesh;
            Matrix4x4 M = mc.transform.localToWorldMatrix;
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

        var mcols = collidersRoot.GetComponentsInChildren<MeshCollider>(includeInactive: false);
        Array.Sort(mcols, (a, b) => a.GetInstanceID().CompareTo(b.GetInstanceID()));
        foreach (var mc in mcols)
        {
            if (mc.sharedMesh == null)
                continue;
            Mesh m = mc.sharedMesh;
            sb.Append("MC|");
            sb.Append(mc.enabled ? '1' : '0');
            sb.Append(m.GetInstanceID());
            sb.Append(',');
            sb.Append(m.vertexCount);
            sb.Append(',');
            sb.Append(m.triangles.Length);
            sb.Append(',');
            Matrix4x4 M = mc.transform.localToWorldMatrix;
            for (int k = 0; k < 16; k++)
                sb.Append(M[k].ToString("R", inv));
            sb.Append(';');
        }

        void AppendBox(BoxCollider bc)
        {
            sb.Append("Bx|");
            sb.Append(bc.enabled ? '1' : '0');
            Matrix4x4 M = bc.transform.localToWorldMatrix;
            for (int k = 0; k < 16; k++)
                sb.Append(M[k].ToString("R", inv));
            sb.Append('|');
            sb.Append(bc.center.x.ToString("R", inv)).Append(',').Append(bc.center.y.ToString("R", inv)).Append(',').Append(bc.center.z.ToString("R", inv)).Append('|');
            sb.Append(bc.size.x.ToString("R", inv)).Append(',').Append(bc.size.y.ToString("R", inv)).Append(',').Append(bc.size.z.ToString("R", inv)).Append(';');
        }
        void AppendSphere(SphereCollider sc)
        {
            sb.Append("Sp|");
            sb.Append(sc.enabled ? '1' : '0');
            Matrix4x4 M = sc.transform.localToWorldMatrix;
            for (int k = 0; k < 16; k++)
                sb.Append(M[k].ToString("R", inv));
            sb.Append('|');
            sb.Append(sc.center.x.ToString("R", inv)).Append(',').Append(sc.center.y.ToString("R", inv)).Append(',').Append(sc.center.z.ToString("R", inv)).Append('|');
            sb.Append(sc.radius.ToString("R", inv)).Append(';');
        }
        void AppendCapsule(CapsuleCollider c)
        {
            sb.Append("Cp|");
            sb.Append(c.enabled ? '1' : '0');
            Matrix4x4 M = c.transform.localToWorldMatrix;
            for (int k = 0; k < 16; k++)
                sb.Append(M[k].ToString("R", inv));
            sb.Append('|');
            sb.Append(c.center.x.ToString("R", inv)).Append(',').Append(c.center.y.ToString("R", inv)).Append(',').Append(c.center.z.ToString("R", inv)).Append('|');
            sb.Append(c.direction.ToString(inv)).Append('|');
            sb.Append(c.height.ToString("R", inv)).Append('|');
            sb.Append(c.radius.ToString("R", inv)).Append(';');
        }

        var boxes = collidersRoot.GetComponentsInChildren<BoxCollider>(includeInactive: false);
        Array.Sort(boxes, (a, b) => a.GetInstanceID().CompareTo(b.GetInstanceID()));
        foreach (var bc in boxes) AppendBox(bc);
        var spheres = collidersRoot.GetComponentsInChildren<SphereCollider>(includeInactive: false);
        Array.Sort(spheres, (a, b) => a.GetInstanceID().CompareTo(b.GetInstanceID()));
        foreach (var sc in spheres) AppendSphere(sc);
        var caps = collidersRoot.GetComponentsInChildren<CapsuleCollider>(includeInactive: false);
        Array.Sort(caps, (a, b) => a.GetInstanceID().CompareTo(b.GetInstanceID()));
        foreach (var c in caps) AppendCapsule(c);

        sb.Append('R').Append(R).Append('|');
        Vector3 mn = simWorldBounds.min, mx = simWorldBounds.max;
        sb.Append(mn.x.ToString("R", inv)).Append(',').Append(mn.y.ToString("R", inv)).Append(',').Append(mn.z.ToString("R", inv)).Append('|');
        sb.Append(mx.x.ToString("R", inv)).Append(',').Append(mx.y.ToString("R", inv)).Append(',').Append(mx.z.ToString("R", inv));
        sb.Append("|SOL4");

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

    // Layout: header(8) + solid(total*4) + normals(total*12) + sdf(total*4)
    static int ExpectedFileLength(int total) => HeaderSize + total * 4 + total * 12 + total * 4;

    /// <summary>Buffer.BlockCopy only accepts primitive arrays; pack Vector3 as float3.</summary>
    static void PackNormalsForCache(Vector3[] normals, float[] dstFloat3)
    {
        int n = normals.Length;
        for (int i = 0; i < n; i++)
        {
            int b = i * 3;
            dstFloat3[b]     = normals[i].x;
            dstFloat3[b + 1] = normals[i].y;
            dstFloat3[b + 2] = normals[i].z;
        }
    }

    static void UnpackNormalsFromCache(float[] srcFloat3, Vector3[] normals)
    {
        int n = normals.Length;
        for (int i = 0; i < n; i++)
        {
            int b = i * 3;
            normals[i] = new Vector3(srcFloat3[b], srcFloat3[b + 1], srcFloat3[b + 2]);
        }
    }

    static bool TryLoadCache(GameObject collidersRoot, Bounds simWorldBounds, int R, int total, out BakeResult result)
    {
        result = default;
        string path = CachePath(collidersRoot, simWorldBounds, R);
        if (!File.Exists(path)) return false;
        try
        {
            byte[] raw = File.ReadAllBytes(path);
            if (raw.Length < ExpectedFileLength(total)) return false;
            if (raw[0] != FileMagicBytes[0] || raw[1] != FileMagicBytes[1] ||
                raw[2] != FileMagicBytes[2] || raw[3] != FileMagicBytes[3])
                return false;
            if (BitConverter.ToInt32(raw, 4) != R) return false;

            var solid = new uint[total];
            Buffer.BlockCopy(raw, HeaderSize, solid, 0, total * 4);

            var normFlat = new float[total * 3];
            Buffer.BlockCopy(raw, HeaderSize + total * 4, normFlat, 0, total * 12);
            var normals = new Vector3[total];
            UnpackNormalsFromCache(normFlat, normals);

            var sdf = new float[total];
            Buffer.BlockCopy(raw, HeaderSize + total * 4 + total * 12, sdf, 0, total * 4);
            result = new BakeResult { Solid = solid, Normals = normals, SDF = sdf };
            return true;
        }
        catch { return false; }
    }

    static void SaveCache(GameObject collidersRoot, Bounds simWorldBounds, int R,
                          uint[] solid, Vector3[] normals, float[] sdf)
    {
        try
        {
            string path = CachePath(collidersRoot, simWorldBounds, R);
            int total = solid.Length;
            var raw = new byte[ExpectedFileLength(total)];
            FileMagicBytes.CopyTo(raw, 0);
            BitConverter.GetBytes(R).CopyTo(raw, 4);
            Buffer.BlockCopy(solid, 0, raw, HeaderSize, total * 4);

            var normFlat = new float[total * 3];
            PackNormalsForCache(normals, normFlat);
            Buffer.BlockCopy(normFlat, 0, raw, HeaderSize + total * 4, total * 12);

            Buffer.BlockCopy(sdf, 0, raw, HeaderSize + total * 4 + total * 12, total * 4);
            File.WriteAllBytes(path, raw);
        }
        catch (Exception e)
        {
            Debug.LogWarning($"[MeshColliderBaker] Could not write voxel cache: {e.Message}");
        }
    }
}
