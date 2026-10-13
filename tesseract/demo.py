"""Local 4D Mind platform. The Tauri 2 shell opens this viewer in a native window."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from tesseract.character import Resident, voice_from_env
from tesseract.codeblock import SCULPTURES, file_at, hold_sculpture, repo_probe_sentence, rasterize_repo
from tesseract.coder import CoderError, ask as ask_coder
from tesseract.ollama_cloud import api_key, default_model, fallback_models, list_local_models, list_models
from tesseract.life import SMEAR_KINDS, load_iris, load_npz, media_probe_sentence, rasterize, rasterize_smear
from tesseract.platform import catalog
from tesseract.viewdata import probe_channels, sample_field_at_points, sample_points, slice_image
from tesseract.voice_text import describe, probe_sentence
from tesseract.weightblock import hold_weights, is_weight_source, weights_probe_sentence

WEB_ROOT = Path(__file__).resolve().parents[1] / "web"
VIEW_SHAPE = (16, 32, 32, 32)
LIFE_LABELS = {"occupancy": "body", "tissue": "tissue", "affect": "affect", "scar": "scar", "other": "other"}


def _risk_model():
    path = os.environ.get("FOURD_MIND_RISK_CKPT")
    if not path:
        return None
    try:
        from tesseract.train import load_checkpoint, pick_device

        model, _cfg, _blob = load_checkpoint(path, pick_device("auto"))
        model.eval()
        return model
    except Exception as exc:  # noqa: BLE001
        print(f"risk checkpoint unavailable ({exc})", flush=True)
        return None


class BlockWorld:
    def __init__(
        self,
        npz: str | None = None,
        shape: tuple[int, int, int, int] = VIEW_SHAPE,
        repo: str | None = None,
        sculpture: str | None = None,
        weights: str | None = None,
    ) -> None:
        self.shape = shape
        self.lock = threading.Lock()
        self.model = _risk_model()
        self.resident = Resident(voice_from_env())
        self.chat: list[dict] = []
        self.coder_chat: list[dict] = []
        self.risk_cache: dict | None = None
        self.source: Path | None = None
        self.ollama_key = api_key()
        self.load(npz=npz, repo=repo, sculpture=sculpture, weights=weights)

    def load(
        self,
        npz: str | None = None,
        repo: str | None = None,
        sculpture: str | None = None,
        smear: str | None = None,
        smear_amount: float = 0.16,
        wav: str | None = None,
        weights: str | None = None,
    ) -> None:
        if weights:
            volume, invariants, meta = hold_weights(weights, self.shape)
        elif smear:
            volume, invariants, meta = rasterize_smear(self.shape, smear, smear=smear_amount, wav=wav)
        elif sculpture:
            volume, invariants, meta = hold_sculpture(sculpture, self.shape, repo=repo)
        elif repo:
            volume, invariants, meta = rasterize_repo(repo, self.shape)
        elif npz:
            volume, invariants, meta = load_npz(npz)
        else:
            volume, invariants, meta = rasterize(load_iris(), self.shape)
        points = sample_points(volume)
        reading = meta.get("reading") or describe(meta)
        root = meta.get("root")
        source = Path(root) if root and Path(root).is_dir() else None
        with self.lock:
            self.volume, self.invariants, self.meta = volume, invariants, meta
            self.points = points
            self.reading = reading
            self.source = source
            self.chat = []
            self.coder_chat = []
            self.risk_cache = None

    @property
    def kind(self) -> str:
        return str(self.meta.get("kind", "life"))

    @property
    def is_repo(self) -> bool:
        return self.kind == "repo"

    def block_payload(self) -> dict:
        kind = self.kind
        if kind == "repo":
            subtitle = (
                "Each dot is a line of this repository, placed by file, line, and commit. "
                "The text stays on disk. Click a dot to name the file."
            )
        elif kind == "sculpture":
            subtitle = f"{self.meta['name']} as an ambient sculpture. Time is an axis you can turn."
        elif kind == "weights":
            subtitle = (
                f"{self.meta['name']}: every tensor as a slice of t, output unit on x, input unit on y, "
                "magnitude up z. Click a dot to read the layer."
            )
        elif kind == "media":
            subtitle = f"{self.meta['name']}, smeared into one solid. Time is thick."
        else:
            subtitle = f"{self.meta['name']}, held as one solid. Time is an axis you can turn."
        return {
            "name": self.meta["name"],
            "kind": kind,
            "subtitle": subtitle,
            "reading": self.reading,
            "labels": self.meta.get("labels", LIFE_LABELS),
            "invariants": self.meta["invariants"],
            "invariant_names": self.meta["invariant_names"],
            "points": self.points.tolist(),
            "shape": list(self.volume.shape),
            "theme": self.meta.get("theme"),
            "voice": self.resident.backend,
            "has_source": self.source is not None,
            "tensors": self.meta.get("tensors") if kind == "weights" else None,
            "studio": kind if kind in {"sculpture", "media"} else ("scan" if kind == "repo" else "resident"),
        }

    def probe_payload(self, x: float, y: float, z: float, t: float) -> dict:
        channels = probe_channels(self.volume, x, y, z, t)
        if self.is_repo:
            sentence = repo_probe_sentence(self.meta, channels, x, y, z, t)
        elif self.kind == "sculpture":
            from tesseract.codeblock import sculpture_probe_sentence

            sentence = sculpture_probe_sentence(self.meta, channels, x, y, z, t)
        elif self.kind == "weights":
            sentence = weights_probe_sentence(self.meta, channels, x, y, z, t)
        elif self.kind == "media":
            sentence = media_probe_sentence(channels, x, y, z, t)
        else:
            sentence = probe_sentence(channels, x, y, z, t)
        payload = {"x": x, "y": y, "z": z, "t": t, "channels": channels, "sentence": sentence}
        if self.is_repo:
            payload["file"] = file_at(self.meta, x)
        return payload

    def slice_payload(self, axis: str, value: float) -> dict:
        image = slice_image(self.volume, axis, value)
        height, width, _rgb = image.shape
        return {"width": width, "height": height, "rgb": image.reshape(-1).tolist(), "axis": axis, "value": value}

    def risk_payload(self) -> dict:
        if self.risk_cache is not None:
            return self.risk_cache
        from tesseract.risk import report, report_text

        raw = report(self.volume, self.meta, self.model)
        field = raw.pop("risk")
        payload = {
            "source": raw["source"],
            "backtest": raw["backtest"],
            "weights": raw["weights"],
            "files": raw.get("files", []),
            "hotspots": raw.get("hotspots", []),
            "points": sample_field_at_points(field, self.points),
            "reading": report_text(raw, self.meta.get("name", "this block")),
        }
        self.risk_cache = payload
        return payload

    def talk(self, query: str, t: float) -> dict:
        reply = self.resident.reply(self.meta, t, query, self.chat, self.volume)
        self.chat.append({"role": "user", "content": query})
        self.chat.append({"role": "assistant", "content": reply["text"]})
        self.chat = self.chat[-16:]
        return reply

    def models_payload(self) -> dict:
        local_models: list[dict] = [{"name": "grounded:4d-lens", "coder": True}]
        if self.resident.backend == "lora":
            local_models.insert(0, {"name": "lora:4d-coder", "coder": True})
        try:
            local_models.extend(list_local_models())
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        try:
            cloud_models = list_models()
            error = ""
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            cloud_models = fallback_models()
            error = str(exc)
        models = local_models + cloud_models
        has_local_daemon = any(row["name"].startswith("local:") for row in local_models)
        if self.resident.backend == "lora":
            preferred = "lora:4d-coder"
        elif has_local_daemon:
            preferred = next((row["name"] for row in local_models if row["name"].startswith("local:")), "grounded:4d-lens")
        elif self.ollama_key:
            preferred = default_model(cloud_models) or "grounded:4d-lens"
        else:
            preferred = "grounded:4d-lens"
        return {
            "models": models,
            "default": preferred,
            "key_set": bool(self.ollama_key),
            "local_available": True,
            "error": error,
        }

    def set_ollama_key(self, key: str) -> None:
        self.ollama_key = key.strip()

    def code(self, query: str, model: str, focus: str | None) -> dict:
        with self.lock:
            meta = self.meta
            source = self.source
            volume = self.volume
            key = self.ollama_key
            risk_model = self.model
            voice = self.resident.voice
        hotspots: list[dict] = []
        if source is not None and meta.get("kind") == "repo":
            from tesseract.risk import report

            raw = report(volume, meta, risk_model)
            raw.pop("risk", None)
            hotspots = raw.get("hotspots") or []
        chosen = model.strip() or (self.models_payload().get("default") or "grounded:4d-lens")
        reply = ask_coder(
            meta,
            source,
            query,
            chosen,
            key,
            hotspots=hotspots,
            focus=focus,
            volume=volume,
            voice=voice,
        )
        self.coder_chat.append({"role": "user", "content": query})
        self.coder_chat.append({"role": "assistant", "content": reply["text"]})
        self.coder_chat = self.coder_chat[-12:]
        return reply


def _read_json(handler: SimpleHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", "0"))
    return json.loads(handler.rfile.read(length) or b"{}")


def make_handler(world: BlockWorld):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

        def log_message(self, fmt: str, *args) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/api/platform":
                self._json(catalog())
                return
            if parsed.path == "/api/sculptures":
                self._json({"sculptures": catalog()["sculptures"]})
                return
            if parsed.path == "/api/block":
                self._json(world.block_payload())
                return
            if parsed.path == "/api/risk":
                self._json(world.risk_payload())
                return
            if parsed.path == "/api/models":
                self._json(world.models_payload())
                return
            if parsed.path == "/api/probe":
                query = parse_qs(parsed.query)
                try:
                    payload = world.probe_payload(
                        float(query.get("x", ["0.5"])[0]),
                        float(query.get("y", ["0.5"])[0]),
                        float(query.get("z", ["0.5"])[0]),
                        float(query.get("t", ["0.5"])[0]),
                    )
                except (TypeError, ValueError):
                    self._json({"error": "probe expects numeric x, y, z, t"}, status=400)
                    return
                self._json(payload)
                return
            if parsed.path == "/api/slice":
                query = parse_qs(parsed.query)
                axis = query.get("axis", ["t"])[0]
                try:
                    value = float(query.get("value", ["0.5"])[0])
                    payload = world.slice_payload(axis, value)
                except (TypeError, ValueError) as exc:
                    self._json({"error": str(exc)}, status=400)
                    return
                self._json(payload)
                return
            if parsed.path == "/":
                self.path = "/index.html"
            super().do_GET()

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
                self._json({"error": "send application/json"}, status=415)
                return
            if path == "/api/open":
                self._open()
                return
            if path == "/api/resident":
                try:
                    request = _read_json(self)
                    payload = world.talk(str(request.get("query", "")), float(request.get("t", 0.5)))
                except (TypeError, ValueError) as exc:
                    self._json({"error": str(exc)}, status=400)
                    return
                self._json(payload)
                return
            if path == "/api/coder":
                try:
                    request = _read_json(self)
                    payload = world.code(
                        str(request.get("query", "")),
                        str(request.get("model", "")),
                        str(request.get("file") or "").strip() or None,
                    )
                except CoderError as exc:
                    self._json({"error": str(exc)}, status=400)
                    return
                except RuntimeError as exc:
                    self._json({"error": str(exc)}, status=502)
                    return
                self._json(payload)
                return
            if path == "/api/ollama/key":
                request = _read_json(self)
                world.set_ollama_key(str(request.get("key", "")))
                self._json({"key_set": bool(world.ollama_key)})
                return
            if path == "/api/smear":
                try:
                    request = _read_json(self)
                    kind = str(request.get("kind", "pulse")).strip() or "pulse"
                    if kind not in SMEAR_KINDS:
                        raise ValueError(f"unknown smear {kind!r}")
                    smear = float(request.get("smear", 0.16))
                    wav = str(request.get("wav") or "").strip() or None
                    world.load(smear=kind, smear_amount=smear, wav=wav)
                except (OSError, ValueError, KeyError) as exc:
                    self._json({"error": str(exc)}, status=400)
                    return
                self._json({"name": world.meta["name"], "kind": world.kind})
                return
            self._json({"error": "not found"}, status=404)

        def _open(self) -> None:
            source = ""
            try:
                request = _read_json(self)
                source = str(request.get("path", "")).strip()
                sculpture = str(request.get("sculpture") or "").strip() or None
                smear = str(request.get("smear") or "").strip() or None
                if smear:
                    world.load(smear=smear, smear_amount=float(request.get("smear_amount", 0.16)))
                elif sculpture:
                    world.load(sculpture=sculpture, repo=source or None)
                elif not source:
                    world.load()
                elif is_weight_source(source):
                    world.load(weights=source)
                elif source.lower() in SCULPTURES:
                    world.load(sculpture=source.lower())
                elif source.endswith(".npz"):
                    world.load(npz=source)
                else:
                    world.load(repo=source)
            except (OSError, ValueError, KeyError, ImportError, subprocess.CalledProcessError) as exc:
                detail = getattr(exc, "stderr", None) or str(exc)
                self._json({"error": f"could not hold {source or 'Iris'}: {str(detail).strip()}"}, status=400)
                return
            self._json({"name": world.meta["name"], "kind": world.kind})

        def _json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    return Handler


def make_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    npz: str | None = None,
    shape=VIEW_SHAPE,
    repo: str | None = None,
    sculpture: str | None = None,
    weights: str | None = None,
    base: str | None = None,
):
    if weights and base:
        weights = f"{weights} - {base}"
    world = BlockWorld(npz=npz, shape=shape, repo=repo, sculpture=sculpture, weights=weights)
    server = ThreadingHTTPServer((host, port), make_handler(world))
    return server, world


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the 4D Mind platform")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--npz", default=None, help="Exported block from the Colab notebook or tesseract.codeblock")
    parser.add_argument("--repo", default=None, help="Hold a git repository's history instead of Iris")
    parser.add_argument("--sculpture", default=None, choices=sorted(SCULPTURES), help="Ambient sculpture of a famous tree")
    parser.add_argument("--weights", default=None, help="Hold model weights: .safetensors, .pt/.pth/.bin, a sharded folder, or 'tesseract'")
    parser.add_argument("--base", default=None, help="With --weights, show what changed since this checkpoint")
    parser.add_argument("--shape", type=int, nargs=4, default=list(VIEW_SHAPE), metavar=("T", "D", "H", "W"))
    args = parser.parse_args()
    shape = tuple(args.shape)
    _server, world = make_server(
        args.host, args.port, npz=args.npz, shape=shape, repo=args.repo, sculpture=args.sculpture, weights=args.weights, base=args.base
    )
    print(
        f"4D Mind {catalog()['version']}  {world.meta['name']} block {tuple(world.volume.shape)}  "
        f"points {len(world.points)}  http://{args.host}:{args.port}",
        flush=True,
    )
    try:
        _server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped", flush=True)


if __name__ == "__main__":
    main()
