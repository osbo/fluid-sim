# Repo layout and what's included

## Current setup

- **Git repo root:** `Fluid Sim/Assets` (only the Assets folder is in the repo)
- **Unity project root:** `Fluid Sim` (parent of Assets)

## What’s in the repo (under Assets)

| Item | Location | Status |
|------|----------|--------|
| **Scripts** | `Scripts/` | Tracked (C#, compute, Python) |
| **Scene** | `Scenes/SampleScene.unity` | Tracked |
| **Custom materials & shaders** | `Shaders/*.shader`, `Shaders/*.mat` | Tracked |
| **URP / lighting (SSAO, etc.)** | `Settings/` (PC_Renderer, PC_RPAsset, DefaultVolumeProfile, etc.) | Tracked |
| **Recorded training data** | `StreamingAssets/TestData/Run_*/` | Tracked (one Run ≈ 2.1GB) |
| **Videos** | Not in Assets (they’re in `Fluid Sim/Recordings/`) | Not in repo |

## What’s not in the repo (outside Assets)

- **ProjectSettings** – lives in `Fluid Sim/ProjectSettings/` (quality, URP project settings, etc.). To have it in git, the repo root needs to be `Fluid Sim` (see below).
- **Recordings (videos)** – `Fluid Sim/Recordings/*.mp4`. Correctly excluded; `.gitignore` also ignores `*.mp4` and `Recordings/` if you ever add that folder under the repo.

## Optional: include ProjectSettings (repo root = whole project)

If you want ProjectSettings (and the rest of the Unity project except generated stuff) in the repo:

1. Move the repo root from `Assets` to `Fluid Sim` (e.g. move `.git` and `.gitignore` up one level, or re-init at `Fluid Sim` and re-add; the latter loses history unless you use `git filter-repo` / subtree).
2. Add a root-level `.gitignore` that ignores:
   - `Library/`
   - `Logs/`
   - `UserSettings/`
   - `Recordings/`
   - `*.mp4`
   - `MemoryCaptures/`
   - Optional: `Packages/` (or track it if you pin URP versions).

Then you can track `ProjectSettings/`, `Assets/`, and optionally `Packages/`.

## How to adjust the repo (cleanup steps)

Run these from `Fluid Sim/Assets` (your repo root):

1. **Stop tracking junk** (they’ll be ignored from now on; files stay on disk):
   ```bash
   git rm -r --cached Scripts/__pycache__ 2>/dev/null || true
   git rm --cached Scripts/.DS_Store Shaders/.DS_Store StreamingAssets/.DS_Store "StreamingAssets/TestData/.DS_Store" 2>/dev/null || true
   ```
   If you have `__pycache__.meta` tracked, remove it too:
   ```bash
   git rm --cached Scripts/__pycache__.meta 2>/dev/null || true
   ```

2. **Commit the new .gitignore and cleanup:**
   ```bash
   git add .gitignore REPO_LAYOUT.md
   git status   # review
   git commit -m "chore: add .gitignore, repo layout doc, stop tracking __pycache__ and .DS_Store"
   ```

3. **Optional – stop tracking a large Run** (to shrink the repo; you can keep one small Run):
   ```bash
   git rm -r --cached "StreamingAssets/TestData/Run_2026-01-26_00-04-12"
   git commit -m "chore: stop tracking large TestData run (add a small sample Run if needed)"
   ```
   Then add a single small recording (e.g. a few frames) and commit that if you want a sample in the repo.

## Recording data size

`StreamingAssets/TestData` for one Run is ~2.1GB (many small `.bin` files). To avoid the repo growing too much:

- Keep one small Run (e.g. a few frames) as a sample and ignore the rest, or
- Uncomment in `.gitignore`: `StreamingAssets/TestData/Run_*/` and only add a single small Run by not matching that pattern, or
- Use [Git LFS](https://git-lfs.com/) for `*.bin` if you want to track more runs.
