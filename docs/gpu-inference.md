# GPU acceleration for on-device AI

Smriti runs several ONNX models on your machine. This page covers how they use a
GPU, what happens when there isn't one or it fails, and the optional remote GPU
bridge for face embedding:

1. [Which models use the GPU](#which-models-use-the-gpu)
2. [Local GPU](#local-gpu): used automatically when the machine has a usable GPU.
3. [CPU fallback](#cpu-fallback): what happens when there is no GPU or it fails.
4. [Remote GPU bridge](#remote-gpu-bridge): optional, offloads face embedding to a
   GPU machine you control.

Everything here is additive. With no GPU and no bridge, Smriti runs on the CPU
exactly as it always has.

## Which models use the GPU

| Model | Used for | Runs on |
|---|---|---|
| SCRFD | Face detection | GPU, else CPU |
| AdaFace | Face embedding | Remote bridge if configured, else GPU, else CPU |
| Visual encoder | Indexing photos for visual search | GPU, else CPU |
| Text encoder | Embedding a search query | CPU, always |

The text encoder is the one deliberate exception. A search query is a single
tiny input, and building a GPU session for it costs far more than the query
itself (on Windows, DirectML compiled a ~1.1 GB graph at every launch and kept
Search unavailable for close to a minute). The CPU is faster there.

All of the GPU models share one policy, implemented once in `ManagedSession`
(see [For contributors](#for-contributors)): prefer the GPU, and on any GPU
failure move to the CPU and carry on.

## How Smriti picks where to run

For face recognition, per step:

| Step | Runs on | Condition |
|---|---|---|
| Detection | Local GPU | "Use the GPU" is on and a GPU provider starts |
| Detection | Local CPU | Otherwise, or after a GPU failure |
| Embedding | Remote bridge | A bridge is enabled, reachable, reports a GPU and serves the same model |
| Embedding | Local GPU | No bridge configured, "Use the GPU" is on and a GPU provider starts |
| Embedding | Local CPU | Otherwise, or after a GPU or bridge failure |

Visual-search indexing follows the detection rows: GPU when the setting is on
and a provider starts, else CPU.

Decided once per run (a face run, or a visual-search indexing run), when it
starts. Changing a setting applies to the next run.

When a bridge is configured, Smriti does not open a local GPU session for the
face embedder (the bridge replaces that step, and the session would just hold
VRAM). Detection can still use the local GPU. The local embedder that backs up a
bridge is therefore a CPU one.

## Local GPU

Smriti tries accelerated ONNX Runtime providers in this order and uses the
first one that initializes for the model:

| Platform | Providers tried |
|---|---|
| Windows | DirectML (NVIDIA, AMD, Intel, Qualcomm through D3D12) |
| Linux | CUDA (NVIDIA), then OpenVINO on a GPU device |
| macOS | CoreML |

**One session per model, never one per worker.** A face run builds one GPU
session for the detector and one for the embedder, shared by all workers.
Workers still decode, resize, align and post-process in parallel and only take a
lock around the inference call. Building one GPU session per worker was tried
first and caused severe start-up and throughput regressions on integrated GPUs,
so it is avoided on purpose. Visual-search indexing runs on a single worker, so
it simply owns one GPU session.

The sessions are created when a run starts and dropped when it ends, so VRAM is
not held while Smriti is idle.

### Requirements

- **Windows**: nothing extra. The ONNX Runtime that `setup_assets.ps1` installs
  includes DirectML.
- **macOS**: CoreML works with the stock ONNX Runtime.
- **Linux**: the stock ONNX Runtime that `scripts/setup_assets.sh` downloads is
  CPU-only, so CUDA fails to register and Smriti stays on the CPU. For CUDA you
  need the GPU build of ONNX Runtime 1.23 (CUDA 12 and cuDNN 9 libraries on the
  library path), pointed to with `ORT_DYLIB_PATH`. The easiest route is the
  heavy Docker image, which has all of this; see
  [Docker](BUILD.md#docker).
- An NVIDIA GPU also needs a current driver.

### Turning it off

Settings → AI features → **Use the GPU for on-device AI (faces, visual search)**.
It is on by default and stored as `gpu_inference` in `config.json`. Turn it off
if the GPU is slower than your CPU, which can happen on some integrated GPUs. It
does not affect the remote bridge, which has its own switch.

## CPU fallback

There are three layers, each independent:

1. **No usable GPU.** If every provider fails to initialize (no driver, a
   CPU-only ONNX Runtime, unsupported model) or the setting is off, each worker
   builds its own CPU session. This is identical to the behaviour before GPU
   support was added.
2. **GPU fails during a run.** The first inference error on a GPU session
   disables it for the rest of the run and logs a warning. Each worker that was
   using it then loads its own CPU session and retries the failing call once. The
   run finishes, only slower. The next run tries the GPU again. A CPU session
   that errors is not retried, because a second CPU attempt would not change
   anything.
3. **Bridge fails.** See [Automatic fallback](#automatic-fallback) below.

Providers register with errors turned on, so a failed GPU registration is a
visible warning, not a silent CPU session.

### How to tell what ran

Look at the Smriti logs (Help → Open logs folder):

| Log line | Meaning |
|---|---|
| `Face inference device` with `detector` and `embedder` fields | Provider used for each face model this run, or `CPU` |
| `GPU inference session initialized` | A GPU session started for `model`; `provider` says which |
| `GPU provider initialization failed; trying next` | A provider could not start; Smriti tries the next, then CPU |
| `Inference failed on the GPU; retrying once on CPU` | A model's GPU run failed and it moved to CPU |
| `GPU inference failed; workers will continue on CPU` | The shared session was disabled for the rest of the run |
| `Face pipeline: N workers, M intra-threads per session` | CPU worker budget (unchanged) |

You should see one GPU session line per model, not one per worker.

### Known limits

- "Provider initialized" does not prove every node ran on the GPU. Operators a
  provider does not support can stay on the CPU. Smriti logs the provider but
  does not verify placement.
- GPU and CPU runs have not been benchmarked against each other for speed or
  for embedding agreement. A visual-search index built partly on the GPU and
  partly on the CPU (for example after a mid-run fallback) contains vectors that
  may differ slightly in the last digits. If you switch an existing face library
  between devices, spot-check a few people before relying on it.

## Remote GPU bridge

Smriti can also offload embedding to a GPU machine you own. Default behaviour is
unchanged: local ONNX Runtime stays primary. The bridge is strictly additive and
**never required**.

### What it sends

Only 112×112 aligned face JPEG crops (~5 KB each, quality 85). No photos,
no metadata, no telemetry. You control the endpoint URL.

### Model must match

The bridge's `/health` advertises which face-embedding model it has loaded.
Smriti refuses to send work when the bridge's model differs from the one
configured in Settings, because embeddings from different models live in
incompatible metric spaces and mixing them would corrupt clustering. Smriti also
ignores a bridge whose `/health` does not report a GPU provider.

Smriti uses **AdaFace** (`adaface_ir101_webface12m`) as its face recognizer.
The notebook and the script download the same ONNX model as `scripts/setup_assets.*`;
nothing to change.

### Setup paths

#### Kaggle (free, 30 hrs GPU/week, T4×2)

1. Open [kaggle.com](https://kaggle.com), create an account (free).
2. Create a new Notebook. Choose **GPU T4×2** as the accelerator.
3. Upload `notebooks/face_bridge.ipynb` from the Smriti repo.
4. Sign up for [ngrok](https://ngrok.com) (free). Copy your auth token from
   the dashboard.
5. Paste your ngrok token in the notebook's cell 5.
6. Run all cells. The last cell prints a public URL; copy it.

#### Colab (free with limits / Pro $10/mo for A100)

1. Open [colab.research.google.com](https://colab.research.google.com).
2. File → Upload notebook → `notebooks/face_bridge.ipynb`.
3. Runtime → Change runtime type → **T4 GPU** (or **A100** on Pro).
4. Set up ngrok as above (or use the cloudflared alternative commented out).
5. Run all cells. Copy the printed URL.

#### Local GPU server (no internet)

1. Run the bridge on a machine with an NVIDIA GPU, either the notebook
   (`notebooks/face_bridge.ipynb`, skipping the tunnel cells) or the
   [Python script](#python-script-no-notebook).
2. The server listens on port 8000.
3. In Smriti Settings, enter `http://127.0.0.1:8000` if the bridge runs on the
   same machine. For another machine on your network, forward the port over SSH
   (see [Connecting to a bridge on another machine](#connecting-to-a-bridge-on-another-machine)),
   because Smriti does not accept a plain `http://<lan-ip>` URL.

### Python script (no notebook)

`notebooks/face_bridge.py` is a bridge as a single Python file, with no
Jupyter, Kaggle or Colab involved. It serves the API Smriti expects
(`POST /embed`, `GET /health`) with the AdaFace model and the same
preprocessing as Smriti's local embedder, so Smriti treats it like the
notebook. Use it on any machine you control that has an NVIDIA GPU: your own PC,
a rented GPU server, or a lab box.

Smriti never starts it. You run it yourself and give Smriti its URL.

**Requirements**

- [`uv`](https://docs.astral.sh/uv/). The script declares its own dependencies
  (FastAPI, uvicorn, `onnxruntime-gpu`, Pillow, pyngrok and the CUDA 13 / cuDNN 9
  runtime wheels), so there is nothing to `pip install` and no CUDA toolkit to
  set up.
- An NVIDIA driver that supports CUDA 13.

**Run it**

```bash
uv run notebooks/face_bridge.py                         # 0.0.0.0:8000
uv run notebooks/face_bridge.py --host 127.0.0.1        # this machine only
NGROK_AUTHTOKEN=... uv run notebooks/face_bridge.py --tunnel ngrok
```

The first run downloads the AdaFace model next to the script
(`notebooks/models/`). The log then shows `Active providers: [...]`, which
should start with `CUDAExecutionProvider`.

| Option | Default | Meaning |
|---|---|---|
| `--host` | `0.0.0.0` | Address to listen on |
| `--port` | `8000` | Port to listen on |
| `--model-path` | `notebooks/models/adaface_ir101_webface12m.onnx` | Where to find or download the model |
| `--tunnel ngrok` | off | Open an ngrok tunnel and print an `https://` URL (needs `NGROK_AUTHTOKEN`) |
| `--allow-cpu` | off | Start even without a CUDA provider |

By default the script refuses to start when CUDA is not active. `--allow-cpu`
is only useful for testing the server: a bridge that does not report a GPU
provider is ignored by Smriti, so a CPU-only bridge is never used.

### Connecting to a bridge on another machine

Smriti accepts an `https://` URL, or plain `http://` only for `localhost` and
`127.0.0.1`. A bare `http://192.168.x.x:8000` is rejected. Two ways to reach a
bridge running elsewhere:

- **SSH tunnel (recommended on a LAN).** On the machine running Smriti:

  ```bash
  ssh -N -L 8000:127.0.0.1:8000 user@gpu-machine
  ```

  Then use `http://127.0.0.1:8000` as the bridge URL. Start the script on the
  GPU machine with `--host 127.0.0.1` so the port is not exposed otherwise.
- **ngrok.** Use `--tunnel ngrok` and paste the `https://` URL it prints.

The bridge has no authentication. Anyone who can reach the port can ask it for
embeddings and use your GPU, so keep it on loopback or behind a tunnel rather
than on an open network.

The notebook and the script implement the same contract. If you change the model
or the preprocessing in one, change the other.

### How to connect Smriti

1. Open Smriti → Settings → **Cloud face acceleration (advanced)**.
2. Toggle "Use a remote GPU for face embedding" on.
3. Paste the bridge URL (e.g. `https://abc.ngrok.io`).
4. Click **Test connection**. It should show "✓ CUDAExecutionProvider @ XXms".
5. Run face detection. Embedding goes remote; detection stays local.

### ngrok auth token (60-second setup)

1. Sign up at [ngrok.com](https://ngrok.com) (free, no credit card).
2. Copy your authtoken from [dashboard.ngrok.com/get-started/your-authtoken](https://dashboard.ngrok.com/get-started/your-authtoken).
3. Paste it in the notebook's `ngrok.set_auth_token("YOUR_TOKEN_HERE")` cell, or
   export it as `NGROK_AUTHTOKEN` for the script.

### cloudflared alternative (no account needed)

The notebook includes a commented-out cloudflared section. It doesn't need an
account but requires fetching the binary and parsing the tunnel URL from its
stdout.

### Automatic fallback

If the bridge is unreachable or returns errors on 3 consecutive batches, Smriti
falls back to local CPU embedding for the remainder of the job. A single failed
batch is recomputed locally. The job completes normally, just slower.

The local fallback needs the matching local embedding model installed. Without
it, Smriti runs on the bridge alone and has nothing to fall back to.

### Expected speed

| Library size | Local (i7-7567U) | T4 GPU | Speed-up |
|---|---|---|---|
| 10k photos (~30k faces) | ~5 min | ~30 s | 10× |
| 90k photos (~200k faces) | ~3 hrs | ~2.5 min | 70× |

### Cost

All paths are free or nominal:
- **Kaggle**: Free (30 hrs/week GPU quota, resets weekly)
- **Colab free**: Free (limits apply; sessions may throttle)
- **Colab Pro**: $10/mo (A100 GPU, unmetered)
- **ngrok**: Free (1 online tunnel, 1 GB/month)
- **cloudflared**: Free (no account needed)

## Troubleshooting

**The local GPU is never used (Linux)**
- Look for `GPU provider initialization failed` in the logs. On Linux with the
  stock ONNX Runtime this is expected; see [Requirements](#requirements).

**Face processing got slower with the GPU on**
- Turn off "Use the GPU for face detection & embedding". Some integrated GPUs
  are slower than a multi-core CPU for these models.

**Test connection says "Unreachable"**
- Check that the bridge is still running. Colab/Kaggle sessions time out
  after 30–90 min of inactivity.
- Verify the ngrok URL hasn't changed (free ngrok URLs rotate on restart).
- If you typed an `http://` address that is not `localhost` or `127.0.0.1`,
  Smriti rejects it. Use `https://` or an SSH tunnel.

**Embedding is slow, still on CPU**
- Check the Smriti logs. You should see either
  "Remote GPU bridge at <url> unhealthy or wrong model" or "Remote bridge
  model mismatch" if the bridge failed.
- The most common cause is a model mismatch; see
  [Model must match](#model-must-match).
- Restart the bridge. For ngrok, get a fresh URL and update it in Settings.

**ngrok free plan limits**
- 1 online tunnel at a time, 40 connections/min, 1 GB/month bandwidth.
  Face crops at 5 KB each × 200k faces = ~1 GB. For a 90k-photo library,
  this fits within the free plan but is near the limit.
- If you hit the limit, use cloudflared (no limits) or Colab Pro ($10/mo).

## Privacy

- Smriti sends **only** 112×112 face crops to a bridge, not the original photos.
- No metadata (location, filenames, EXIF) is sent.
- No telemetry or analytics from Smriti itself.
- The bridge URL is set by you; Smriti never phones home.
- The notebook runs on your Kaggle/Colab account, and the script on your own
  machine. Neither is a shared service.
- Local GPU inference sends nothing anywhere.

## For contributors

| Piece | Where |
|---|---|
| Provider list, GPU-only loader, `SharedGpuSession`, `ManagedSession` | `src/ml/runtime.rs` |
| Face detector / embedder (use `ManagedSession`) | `src/ml/face_detector.rs`, `src/ml/face_embedder.rs` |
| Visual-search image encoder (uses `ManagedSession`) | `src/services/semantic.rs` |
| Bridge client: health check, batching, 3-strike unhealthy logic | `src/ml/remote_embedder.rs` |
| Deciding GPU vs CPU vs bridge per face run | `src/services/face_processor.rs` |
| `gpu_inference`, bridge URL validation | `src/config/mod.rs` |
| Settings wiring | `src-tauri/src/commands/settings.rs`, `src-ui/src/routes/Settings.svelte` |
| Bridge server | `notebooks/face_bridge.py`, `notebooks/face_bridge.ipynb` |
| Test connection command | `src-tauri/src/commands/system.rs` |

### Adding a model

Use `ManagedSession`; do not call `Session::builder()` directly.

```rust
// One owner (a single worker): GPU if enabled and available, else CPU.
let use_gpu = AppConfig::load().gpu_inference;
let mut model = ManagedSession::auto(&path, use_gpu, cpu_threads)?;

// Many workers: build ONE SharedGpuSession up front, hand each worker a handle.
let gpu = SharedGpuSession::try_new(&path);             // None = no usable GPU
let mut model = match &gpu {
    Some(g) => ManagedSession::shared(g.clone(), &path, cpu_threads),
    None => ManagedSession::cpu(&path, cpu_threads)?,
};

// Inference. The closure may run twice (GPU attempt, then CPU retry), so it
// must not consume its inputs, and it must return owned data.
let out = model.run(|session| {
    let outputs = session.run(ort::inputs![input_tensor])?;
    /* extract and copy what you need */
    Ok(vec)
})?;
```

Decide per model whether the GPU is worth it. Large models that run over many
inputs (batch indexing) are. A tiny model run once per user action is not; keep
it on the CPU like the search text encoder.

### Rules to keep

- Never build one GPU session per worker. Share one per model.
- A GPU failure must degrade to CPU, never fail the run.
- Read `AppConfig::gpu_inference` before choosing the GPU, so the user's switch
  is honoured everywhere.
- The bridge and the local embedder must produce vectors in the same metric
  space, which is why the model name must match.
