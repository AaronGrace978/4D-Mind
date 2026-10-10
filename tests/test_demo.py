import json
import threading
from urllib.request import urlopen

from tesseract.demo import make_server


def test_viewer_api_serves_iris_block_slice_and_probe():
    server, world = make_server("127.0.0.1", 0, shape=(12, 24, 24, 24))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        with urlopen(f"http://127.0.0.1:{port}/api/block") as response:
            payload = json.load(response)
        assert payload["name"] == "Iris"
        assert "solid" in payload["reading"].lower() or "Iris" in payload["reading"]
        assert len(payload["points"]) > 100
        assert len(payload["points"][0]) == 9

        with urlopen(f"http://127.0.0.1:{port}/") as response:
            page = response.read().decode("utf-8")
        assert "Rotate time into x" in page
        assert "/app.js" in page

        point = payload["points"][0]
        query = f"x={point[0]}&y={point[1]}&z={point[2]}&t={point[3]}"
        with urlopen(f"http://127.0.0.1:{port}/api/probe?{query}") as response:
            probe = json.load(response)
        assert "Point" in probe["sentence"]
        assert set(probe["channels"]) == {"occupancy", "tissue", "affect", "scar", "other"}

        with urlopen("http://127.0.0.1:%d/api/slice?axis=t&value=0.2" % port) as response:
            plane = json.load(response)
        assert len(plane["rgb"]) == plane["width"] * plane["height"] * 3
        assert world.meta["name"] == "Iris"
    finally:
        server.shutdown()
