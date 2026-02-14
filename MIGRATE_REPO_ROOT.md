# Migrate repo root to Unity project (Fluid Sim)

This moves the git repo root from `Assets` to `Fluid Sim` so you can track **ProjectSettings** and keep one repo for the whole Unity project. History is preserved.

## Prerequisites

- **Clean working tree** in Assets: commit or stash any changes.
- **git-filter-repo** installed:
  ```bash
  pip install git-filter-repo
  ```
  or: `brew install git-filter-repo`

## Steps (run from Terminal)

**1. Go to the Unity project folder**
```bash
cd "/Users/carlosborne/Documents/MIT/UROP/Fluid Sim"
```

**2. Go into Assets (current repo root) and rewrite history so all paths are under `Assets/`**
```bash
cd Assets
git filter-repo --to-subdirectory-filter Assets --force
```
You may see “Aborting: refs/heads/main is not the same as refs/heads/master” or similar; if so, use `--refs refs/heads/main` (or your branch name). Example:
```bash
git filter-repo --to-subdirectory-filter Assets --force --refs refs/heads/main
```

**3. Fix the folder layout**  
After step 2, you have `Fluid Sim/Assets/Assets/Scripts/...`. Move the inner `Assets` content up:
```bash
mv Assets/* .
rmdir Assets
```

**4. Move the repo root to the Unity project folder**
```bash
mv .git ..
cd ..
```
You should now be in `Fluid Sim` and `git status` should list all your files under `Assets/`.

**5. Add ProjectSettings and root .gitignore**
```bash
git add ProjectSettings .gitignore
git status   # confirm ProjectSettings and .gitignore are added
git commit -m "chore: repo root is Unity project; add ProjectSettings and root .gitignore"
```

**6. Optional: track Packages (URP versions)**  
If you want to pin package versions:
```bash
git add Packages/manifest.json Packages/packages-lock.json
git commit -m "chore: add Packages manifest and lock"
```

**7. Set upstream again (remote is removed by filter-repo)**  
Re-add your remote and push:
```bash
git remote add origin https://github.com/osbo/fluid-sim.git
git push -u origin main --force
```
Use `--force` only if you’re replacing the remote history with the new root (required after a history rewrite).

---

## If you don’t care about history (simpler, no filter-repo)

1. Copy your current `Assets` folder and `ProjectSettings` somewhere safe (backup).
2. In `Fluid Sim`, remove the old repo and start fresh:
   ```bash
   cd "/Users/carlosborne/Documents/MIT/UROP/Fluid Sim"
   rm -rf Assets/.git
   git init
   git add .gitignore
   git add Assets ProjectSettings
   git commit -m "Initial commit: Unity project and ProjectSettings"
   ```
3. Add remote and push as usual (no `--force` needed if it’s a new repo).

Use this only if you’re okay losing the previous git history in `Assets`.
