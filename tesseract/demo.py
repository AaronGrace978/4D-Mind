"""Local worldtube viewer. Speech stays on the Colab GPU; this page shows the solid."""

from __future__ import annotations

import argparse
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from tesseract.life import load_iris, load_npz, rasterize
from tesseract.viewdata import probe_channels, sample_points, slice_image
from tesseract.voice_text import describe, probe_sentence

WEB_ROOT = Path(__file__).resolve().parents[1] / "web"
VIEW_SHAPE = (16, 32, 32, 32)


class BlockWorld:
    def __init__(self, npz: str | None = None, shape: tuple[int, int, int, int] = VIEW_SHAPE) -> None:
        if npz:
            self.volume, self.invariants, self.meta = load_npz(npz)
        else:
            self.volume, self.invariants, self.meta = rasterize(load_iris(), shape)
        self.points = sample_points(self.volume)
        self.reading = describe(self.meta)

    def block_payload(self) -> dict:
        return {
            "name": self.meta["name"],
            "reading": self.reading,
            "invariants": self.meta["invariants"],
            "invariant_names": self.meta["invariant_names"],
            "points": self.points.tolist(),
            "shape": list(self.volume.shape),
        }

    def probe_payload(self, x: float, y: float, z: float, t: float) -> dict:
        channels = probe_channels(self.volume, x, y, z, t)
        return {
            "x": x,
            "y": y,
            "z": z,
            "t": t,
            "channels": channels,
            "sentence": probe_sentence(channels, x, y, z, t),
        }

    def slice_payload(self, axis: str, value: float) -> dict:
        image = slice_image(self.volume, axis, value)
        height, width, _rgb = image.shape
        return {"width": width, "height": height, "rgb": image.reshape(-1).tolist(), "axis": axis, "value": value}


def make_handler(world: BlockWorld):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

        def log_message(self, fmt: str, *args) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/api/block":
                self._json(world.block_payload())
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

        def _json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    return Handler


def make_server(host: str = "127.0.0.1", port: int = 8765, npz: str | None = None, shape=VIEW_SHAPE):
    world = BlockWorld(npz=npz, shape=shape)
    server = ThreadingHTTPServer((host, port), make_handler(world))
    return server, world


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the Iris worldtube viewer")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--npz", default=None, help="Exported block from the Colab notebook")
    parser.add_argument("--shape", type=int, nargs=4, default=list(VIEW_SHAPE), metavar=("T", "D", "H", "W"))
    args = parser.parse_args()
    shape = tuple(args.shape)
    _server, world = make_server(args.host, args.port, npz=args.npz, shape=shape)
    print(
        f"Iris block {tuple(world.volume.shape)}  points {len(world.points)}  http://{args.host}:{args.port}",
        flush=True,
    )
    try:
        _server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped", flush=True)


if __name__ == "__main__":
    main()
