#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy>=1.26",
#     "fastapi>=0.110",
#     "uvicorn>=0.29",
#     "python-multipart>=0.0.9",
#     "pillow>=10",
#     "onnxruntime-gpu>=1.24",
#     # CUDA 13 runtime libs for onnxruntime-gpu, so no system toolkit is needed
#     # (only the NVIDIA driver). Loaded by ort.preload_dlls() below.
#     "nvidia-cuda-runtime>=13",
#     "nvidia-cublas>=13",
#     "nvidia-cufft>=12",
#     "nvidia-curand>=10",
#     "nvidia-cudnn-cu13>=9",
#     "pyngrok>=7",
# ]
# ///
"""Smriti face embedding bridge.

Serves a face-embedding ONNX model over HTTP so the desktop app can offload
embedding to a GPU. Receives batches of 112x112 face crops and returns
L2-normalised 512-d embeddings.

Run:
    uv run face_bridge.py                       # LAN only, 0.0.0.0:8000
    NGROK_AUTHTOKEN=... uv run face_bridge.py --tunnel ngrok

The model MUST match the one Smriti uses locally (AdaFace). `/health`
advertises the model name and execution provider; Smriti refuses to use the
bridge when the model differs or when no GPU provider is active.
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import onnxruntime as ort
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image

log = logging.getLogger("face_bridge")

MODEL_NAME = "adaface_ir101_webface12m"
MODEL_URL = (
    "https://drive.usercontent.google.com/download"
    "?id=1dgMFOASKnaujQcCL4sSYkKOkBrmXUUU1&export=download&confirm=t"
)
MIN_MODEL_BYTES = 100_000_000  # a smaller file is an HTML error page, not the model
FACE_SIZE = 112
EMBED_DIM = 512


def ensure_model(path: Path) -> None:
    """Download the model to `path` unless it is already there."""
    if path.exists() and path.stat().st_size > MIN_MODEL_BYTES:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s -> %s", MODEL_NAME, path)
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(MODEL_URL, tmp)
    if tmp.stat().st_size <= MIN_MODEL_BYTES:
        tmp.unlink()
        raise RuntimeError("model download failed or returned an HTML error page")
    tmp.replace(path)


def create_session(path: Path, require_gpu: bool) -> ort.InferenceSession:
    ort.preload_dlls()  # pick up the CUDA 13 / cuDNN 9 wheels from site-packages
    session = ort.InferenceSession(
        str(path),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    active = session.get_providers()
    log.info("Active providers: %s", active)
    if require_gpu and "CUDAExecutionProvider" not in active:
        raise RuntimeError(
            f"CUDAExecutionProvider is not active (got {active}). Check the NVIDIA "
            "driver and that the driver supports CUDA 13, or pass "
            "--allow-cpu. Smriti ignores bridges that do not report a GPU provider."
        )
    return session


def preprocess(img_bytes: bytes) -> np.ndarray:
    """Decode an image, resize to 112x112, normalise to [-1, 1], return CHW."""
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((FACE_SIZE, FACE_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    arr = (arr - 127.5) / 127.5
    return np.transpose(arr, (2, 0, 1))


def l2_normalize(batch: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(batch, axis=1, keepdims=True)
    return batch / np.where(norms > 0, norms, 1.0)


def create_app(session: ort.InferenceSession) -> FastAPI:
    input_name = session.get_inputs()[0].name

    app = FastAPI(title="Smriti face bridge")

    def run_batch(face_bytes: list[bytes]) -> list[list[float]]:
        batch = np.stack([preprocess(b) for b in face_bytes]).astype(np.float32)
        output = session.run(None, {input_name: batch})[0]
        return l2_normalize(output).tolist()

    @app.post("/embed")
    async def embed(files: list[UploadFile] = File(...)):
        """Receive N face crops, return N 512-d embeddings."""
        if not files:
            return {"embeddings": []}
        face_bytes = [await f.read() for f in files]
        try:
            embeddings = await run_in_threadpool(run_batch, face_bytes)
        except OSError as e:  # PIL could not decode an upload
            raise HTTPException(status_code=400, detail=f"bad image: {e}") from e
        return {"embeddings": embeddings}

    @app.get("/health")
    async def health():
        providers = session.get_providers()
        return {
            "status": "ok",
            "provider": providers[0] if providers else "unknown",
            "model": MODEL_NAME,
        }

    return app


def start_ngrok(port: int) -> None:
    from pyngrok import ngrok

    token = os.environ.get("NGROK_AUTHTOKEN")
    if not token:
        sys.exit("--tunnel ngrok needs the NGROK_AUTHTOKEN environment variable")
    ngrok.set_auth_token(token)
    url = ngrok.connect(port, "http").public_url
    print(f"\nBRIDGE URL -> {url}")
    print("Paste this URL into Smriti Settings -> Cloud face acceleration\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--model-path",
        type=Path,
        default=Path(__file__).resolve().parent / "models" / f"{MODEL_NAME}.onnx",
        help="where to find / download the ONNX model",
    )
    p.add_argument("--tunnel", choices=["none", "ngrok"], default="none")
    p.add_argument(
        "--allow-cpu",
        action="store_true",
        help="start even if the CUDA provider is unavailable (Smriti will ignore it)",
    )
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    ensure_model(args.model_path)
    session = create_session(args.model_path, require_gpu=not args.allow_cpu)

    if args.tunnel == "ngrok":
        start_ngrok(args.port)

    log.info("Serving model %r on http://%s:%d", MODEL_NAME, args.host, args.port)
    uvicorn.run(create_app(session), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
