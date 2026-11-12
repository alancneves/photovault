# Build from Source

Smriti builds natively on Linux, Windows, and macOS.

## Prerequisites

### Linux

- Rust toolchain (stable, MSRV 1.88+) via `rustup`
- Node 20.19+ and npm (the frontend is Vite 8 + Svelte 5)
- Tauri build deps. On Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y \
  libwebkit2gtk-4.1-dev \
  libgtk-3-dev \
  libayatana-appindicator3-dev \
  librsvg2-dev \
  libsoup-3.0-dev \
  libjavascriptcoregtk-4.1-dev \
  libdbus-1-dev \
  pkg-config \
  libheif-dev          # only if you want HEIC decoding
```

### Windows

- Rust toolchain via `rustup`
- Node 20.19+ via the official installer or `winget install OpenJS.NodeJS`
- Visual Studio Build Tools (Desktop development with C++) — provides
  the MSVC linker that `cargo build` and Tauri's bundler both need.
- WebView2 Runtime — preinstalled on Windows 11; auto-downloaded by
  Tauri on Windows 10 if missing.

### macOS

- Xcode command line tools (`xcode-select --install`)
- Rust toolchain via `rustup`
- Node 20.19+ via Homebrew (`brew install node`) or the official installer

### One-time tooling

```bash
cargo install tauri-cli --version "^2" --locked
```

## Asset setup (optional but recommended)

Smriti now ships a small core application.
Face recognition and offline geocoding use an optional asset pack
(ONNX runtime + models + geonames DB) that can be installed in-app with one click.

Manual setup scripts are still available:

Linux/macOS:

```bash
./scripts/setup_assets.sh
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_assets.ps1
```

Both scripts are idempotent and useful for local development/testing.

## Build and run

Smriti is a Tauri 2 app: the Rust engine + Tauri shell live in
`src/` and `src-tauri/`, the Svelte 5 frontend lives in `src-ui/`.
The `cargo tauri` CLI orchestrates both halves.

Dev mode (hot reload via Vite, opens a native window):

```bash
cd src-ui && npm install && cd ..   # one-time
./scripts/dev.sh                    # Linux / macOS
# .\scripts\dev.ps1                 # Windows PowerShell
```

Production bundle (`.deb` + AppImage on Linux, `.msi` on Windows,
`.dmg` / tar.gz on macOS):

```bash
cargo tauri build
```

Engine-only build (useful when iterating on `src/` without a window):

```bash
cargo build -p smriti -p smriti-tauri
```

## Test and lint

```bash
cargo fmt --check
cargo clippy --all-targets -p smriti -p smriti-tauri -- -D warnings
cargo test -p smriti -p smriti-tauri
(cd src-ui && npm run check && npm run build)
```

## Docker

`Dockerfile` and `docker-compose.yml` build, test and run Smriti without
installing Rust, Node or the Tauri system libraries on the host. Linux only.

```bash
docker compose run --rm assets     # one-time: face models + GeoNames into ./models ./data
docker compose run --rm ci         # fmt, clippy, tests and frontend checks (same as scripts/ci_local.sh ci)
docker compose run --rm dev        # shell in the toolchain image, repo mounted at /workspace

xhost +local:docker                # let the container open a window on your X11 display
PHOTOS_DIR=~/Pictures docker compose run --rm app       # GPU first, CPU fallback
PHOTOS_DIR=~/Pictures docker compose run --rm app-cpu   # no GPU request (no NVIDIA needed)
```

- Pass `UID=$(id -u) GID=$(id -g)` if your user is not 1000, so files written
  to the repo are yours.
- The image comes in two sizes, picked at build time with the `GPU` build arg:
  - **heavy** (`GPU=true`, service `app`, image `smriti-app:gpu`): CUDA 12 +
    cuDNN 9 base and the GPU build of ONNX Runtime 1.23. Needs the NVIDIA
    driver and the NVIDIA Container Toolkit on the host. If the GPU can't be
    used, face detection and embedding fall back to the CPU (see
    `gpu_inference` in Settings).
  - **light** (`GPU=false`, service `app-cpu`, image `smriti-app:cpu`): plain
    Ubuntu base and the CPU build of ONNX Runtime. No NVIDIA needed, and the
    image is several GB smaller.
  - Without compose: `docker build --target runtime --build-arg GPU=false -t smriti-app:cpu .`
- Photos: set `PHOTOS_DIR` to a host folder and open it in the app at
  `PHOTOS_MOUNT` (default `/photos`). Set both to the same absolute path, for
  example `/media/me/Drive/Photos`, to keep one path on the host and in the
  container. Put them in a git-ignored `.env` next to `docker-compose.yml` to
  make them permanent. The folder must be writable by your user, because Smriti
  keeps its index in a `.photovault` folder inside the library. Don't open the
  same library in the host app and the container at the same time.
- Settings and downloaded packs (`~/.config/smriti`, `~/.local/share/smriti` on
  the host) are kept in Docker volumes by default, separate from a Smriti
  installed on the host. To share them instead, set `SMRITI_CONFIG_DIR` and
  `SMRITI_DATA_DIR` to those host folders (absolute paths, folders must exist).
  Sharing avoids re-downloading the packs and keeps one set of settings, but
  don't run the host app and the container at the same time.
- Build output and the cargo registry live in the `smriti-cache` volume, not
  in `./target`. `docker compose down -v` deletes both volumes.
- `CARGO_FEATURES=` (empty) builds without HEIC support.
- `docker compose run` reuses an image that already exists. After changing the
  code or the Dockerfile, rebuild with `docker compose build app` (or add
  `--build` to the `run` command).

## Linux packaging (.deb + AppImage)

The official path is `cargo tauri build` — it produces both
`.deb` and AppImage in `target/release/bundle/`. Tag-driven
CI (`.github/workflows/release.yml`) runs the same command on a
fresh Ubuntu runner for every `v*` tag.

For a local `.deb` build through the same Tauri path as CI:

```bash
./scripts/release_local.sh ubuntu
```

Output:
- `target/release/bundle/deb/*.deb`

Install test:

```bash
sudo dpkg -i target/release/bundle/deb/*.deb
smriti-tauri
```

Uninstall test:

```bash
sudo apt remove smriti
```

### Linux AppImage

AppImage is built in CI from release tags (`v*`) via `.github/workflows/release.yml`.
Output directory:
- `target/release/bundle/appimage/`

Run:

```bash
chmod +x target/release/bundle/appimage/*.AppImage
target/release/bundle/appimage/*.AppImage
```

## Windows packaging (MSI + ZIP)

MSI packaging should run on native Windows with the MSVC build tools.

### Prerequisites

- Tauri CLI:

```powershell
cargo install tauri-cli --version "^2" --locked
```

### Build MSI installer

From repository root:

```powershell
cd src-tauri
cargo tauri build --ci --bundles msi
```

Output:
- `target\release\bundle\msi\*.msi`

MSI installs the core app. Optional assets are installed separately.

## Troubleshooting

- Optional assets not detected: install via startup prompt, or run setup script and restart app.
- GUI linking errors on Linux: install missing Wayland/X11 dev packages listed above.
- Slow Windows builds on UNC path: expected; primary development should happen in WSL.

## Local installer verification scripts

Linux/macOS shell:

```bash
./scripts/verify_installers_local.sh
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\release_local.ps1 -Mode full
```

These scripts summarize installer packaging status locally before you create release tags.

Mandatory push gate before any `git push`:

```bash
cargo fmt --all --check
cargo clippy --all-targets
cargo test --no-run
```

## Automated release orchestration (draft-only publish flow)

From Windows PowerShell, once local checks are green:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\release_publish.ps1 -Version v0.1.0-rc3 -RunLocalChecks -Wait
```

Behavior:

- validates repo state and branch sync
- optionally runs local verification first
- creates and pushes the release tag
- waits for `release.yml` workflow
- verifies expected assets in the GitHub draft release
- prints draft release URL for manual publish

By design, this script does **not** auto-publish the release.

## Asset installer URL behavior

In-app optional asset installation resolves in this order:

1. `SMRITI_ASSET_PACK_PATH` (local zip path override; legacy
   `PHOTOVAULT_ASSET_PACK_PATH` is also accepted)
2. `SMRITI_ASSET_PACK_URL` (custom URL override; legacy
   `PHOTOVAULT_ASSET_PACK_URL` is also accepted)
3. version-pinned release URL (`.../releases/download/v<app-version>/Smriti-Assets.zip`)
4. latest published release fallback (`.../releases/latest/download/Smriti-Assets.zip`)

If you are testing locally before publishing a release, set
`SMRITI_ASSET_PACK_PATH`.
