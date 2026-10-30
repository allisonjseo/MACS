"""Read-only Nutrition5k ingredient overlap review and image browser.

Run: python3 image_browser.py --host 0.0.0.0 --port 8000
"""

import argparse
import csv
import json
import shutil
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parent


def load_dataset(source: Path):
    """Index official dishes and optional photos; never download or modify data."""
    labels = {}
    for path in sorted((source / "metadata").glob("dish_metadata_cafe*.csv")):
        with path.open(newline="") as stream:
            for row in csv.reader(stream):
                if not row or not row[0].startswith("dish_"):
                    continue
                labels[row[0]] = {
                    "calories": float(row[1]),
                    "mass": float(row[2]),
                    "fat": float(row[3]),
                    "carbs": float(row[4]),
                    "protein": float(row[5]),
                    "ingredients": [
                        {"id": row[i], "name": row[i + 1], "grams": float(row[i + 2])}
                        for i in range(6, len(row) - 2, 7)
                        if row[i].startswith("ingr_")
                    ],
                }
    split_ids = {
        split: set(
            (source / "dish_ids/splits" / f"rgb_{split}_ids.txt").read_text().split()
        )
        for split in ("train", "test")
    }
    overlap = sorted(split_ids["train"] & split_ids["test"])
    items, images, counts = [], {}, {}
    for split, ids in split_ids.items():
        available = 0
        for dish in sorted(ids):
            candidates = (
                source / "imagery/realsense_overhead" / dish / "rgb.png",
                source / "references" / f"{dish}.png",
                source / "images" / f"{dish}.png",
            )
            photo = next((path for path in candidates if path.is_file()), None)
            if photo is not None:
                images[dish] = photo
                available += 1
            items.append(
                {
                    "id": dish,
                    "split": split,
                    "has_image": photo is not None,
                    **labels.get(dish, {}),
                }
            )
        counts[split] = {"available": available, "total": len(ids)}
    return {"items": items, "counts": counts, "id_overlap": overlap}, images


def make_handler(dataset, images):
    payload = json.dumps(dataset, separators=(",", ":"), allow_nan=False).encode()
    from ingredient_overlap import build_report

    report = json.dumps(
        build_report(dataset), separators=(",", ":"), allow_nan=False
    ).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_HEAD(self):
            self.respond(head=True)

        def do_GET(self):
            self.respond()

        def respond(self, head=False):
            try:
                self.route(head)
            except (BrokenPipeError, ConnectionResetError):
                pass  # A browser can cancel lazy image requests during navigation.

        def route(self, head):
            path = unquote(urlsplit(self.path).path)
            if path in ("/", "/index.html"):
                index = (ROOT / "image_browser/index.html").read_bytes()
                self.send_bytes(index, "text/html; charset=utf-8", head)
            elif path == "/api/dataset":
                self.send_bytes(payload, "application/json; charset=utf-8", head)
            elif path == "/api/ingredients":
                self.send_bytes(report, "application/json; charset=utf-8", head)
            elif path.startswith("/image/"):
                dish = path.removeprefix("/image/")
                photo = images.get(dish)
                if photo is None:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                try:
                    stream = photo.open("rb")
                except FileNotFoundError:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                with stream:
                    stat = photo.stat()
                    etag = f'"{stat.st_mtime_ns}-{stat.st_size}"'
                    if self.headers.get("If-None-Match") == etag:
                        self.send_response(HTTPStatus.NOT_MODIFIED)
                        self.send_header("ETag", etag)
                        self.end_headers()
                        return
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", str(stat.st_size))
                    self.send_header("Cache-Control", "private, max-age=3600")
                    self.send_header("ETag", etag)
                    self.end_headers()
                    if not head:
                        shutil.copyfileobj(stream, self.wfile)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def send_bytes(self, body, content_type, head):
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            if not head:
                self.wfile.write(body)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/nutrition5k")
    args = parser.parse_args()
    dataset, images = load_dataset(args.data_dir.resolve())
    server = ThreadingHTTPServer((args.host, args.port), make_handler(dataset, images))
    for split, counts in dataset["counts"].items():
        print(
            f"{split}: {counts['available']:,} photos / {counts['total']:,} official IDs"
        )
    print(
        f"Image browser listening on http://{args.host}:{server.server_port}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
