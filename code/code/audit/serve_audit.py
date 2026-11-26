#!/usr/bin/env python3
"""A local review UI for the audit queues. No dependencies - stdlib only.

    python code/audit/build_queues.py     # once, to build the queues
    python code/audit/serve_audit.py      # then open the browser it points at

Every choice is saved the moment it is clicked, to
`training_data/evidence/audit_queues/<queue>.labels.json`. Closing the browser
loses nothing; reopening resumes at the first unlabelled row.

The server binds to 127.0.0.1 only. It serves images from the dataset's image
root and refuses any path that resolves outside it, so a bad queue file cannot
turn this into a file browser.
"""
from __future__ import annotations

import argparse
import datetime
import http.server
import json
import pathlib
import socketserver
import threading
import urllib.parse
import webbrowser

ROOT = pathlib.Path(__file__).resolve().parents[2]
QUEUES = ROOT / "training_data" / "evidence" / "audit_queues"
STATIC = pathlib.Path(__file__).resolve().parent / "static"

MEDIA = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
         ".gif": "image/gif", ".webp": "image/webp", ".svg": "image/svg+xml"}

_write_lock = threading.Lock()


def queue_path(name: str) -> pathlib.Path:
    """Queue files live directly in QUEUES; a name with a separator is a bug or
    an attack, and either way is refused."""
    if not name or "/" in name or "\\" in name or name.startswith("."):
        raise ValueError(f"bad queue name: {name!r}")
    return QUEUES / f"{name}.json"


def labels_path(name: str) -> pathlib.Path:
    return QUEUES / f"{name}.labels.json"


def load_labels(name: str) -> dict:
    path = labels_path(name)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("items", {})
    except (json.JSONDecodeError, OSError):
        return {}


def save_label(name: str, item_id: str, payload: dict) -> dict:
    """Read-modify-write under a lock, through a temp file, so an interrupted
    write cannot leave a half-written labels file behind."""
    with _write_lock:
        items = load_labels(name)
        if payload.get("choice") is None and not payload.get("note") \
                and not payload.get("extra"):
            items.pop(item_id, None)
        else:
            entry = items.get(item_id, {})
            entry.update({k: v for k, v in payload.items() if v is not None})
            entry["ts"] = datetime.datetime.now().isoformat(timespec="seconds")
            items[item_id] = entry
        blob = {"queue": name,
                "updated": datetime.datetime.now().isoformat(timespec="seconds"),
                "items": items}
        path = labels_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(blob, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        tmp.replace(path)
        return items


def list_queues():
    out = []
    for path in sorted(QUEUES.glob("*.json")):
        if path.name.endswith(".labels.json") or path.name.endswith(".json.tmp"):
            continue
        try:
            blob = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        labels = load_labels(blob["name"])
        done = sum(1 for v in labels.values() if v.get("choice"))
        out.append({"name": blob["name"], "title": blob.get("title", blob["name"]),
                    "subtitle": blob.get("subtitle", ""),
                    "total": len(blob.get("items", [])), "labeled": done})
    return out


def summarize(name: str) -> dict:
    blob = json.loads(queue_path(name).read_text(encoding="utf-8"))
    labels = load_labels(name)
    counts, extra_counts = {}, {}
    for item in blob["items"]:
        entry = labels.get(item["id"]) or {}
        choice = entry.get("choice")
        if choice:
            counts[choice] = counts.get(choice, 0) + 1
        for key, value in (entry.get("extra") or {}).items():
            extra_counts.setdefault(key, {})
            extra_counts[key][value] = extra_counts[key].get(value, 0) + 1
    return {"queue": name, "total": len(blob["items"]),
            "labeled": sum(counts.values()), "counts": counts,
            "extra_counts": extra_counts,
            "rows": [{"id": it["id"], **(labels.get(it["id"]) or {})}
                     for it in blob["items"]]}


class Handler(http.server.SimpleHTTPRequestHandler):
    image_root: pathlib.Path = ROOT / "training_data" / "normalized"

    # keep the terminal readable - one line per request is noise here
    def log_message(self, fmt, *args):
        pass

    # ---------------- helpers ----------------

    def _send(self, code, body: bytes, ctype="application/json; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def _error(self, code, message):
        self._json({"error": message}, code)

    # ---------------- GET ----------------

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)
        try:
            if path in ("/", "/index.html"):
                return self._static("index.html")
            if path.startswith("/static/"):
                return self._static(path[len("/static/"):])
            if path == "/api/queues":
                return self._json(list_queues())
            if path.startswith("/api/queue/"):
                name = path[len("/api/queue/"):]
                blob = json.loads(queue_path(name).read_text(encoding="utf-8"))
                blob["labels"] = load_labels(name)
                return self._json(blob)
            if path.startswith("/api/summary/"):
                return self._json(summarize(path[len("/api/summary/"):]))
            if path == "/img":
                return self._image(query.get("p", [""])[0])
        except FileNotFoundError:
            return self._error(404, "not found")
        except ValueError as exc:
            return self._error(400, str(exc))
        return self._error(404, "not found")

    def _static(self, rel):
        target = (STATIC / rel).resolve()
        if not target.is_relative_to(STATIC.resolve()) or not target.is_file():
            return self._error(404, "not found")
        ctype = MEDIA.get(target.suffix.lower())
        if ctype is None:
            ctype = ("text/html; charset=utf-8" if target.suffix == ".html"
                     else "text/plain; charset=utf-8")
        return self._send(200, target.read_bytes(), ctype)

    def _image(self, rel):
        if not rel:
            return self._error(400, "missing p")
        root = self.image_root.resolve()
        target = (root / rel).resolve()
        # The queue file is data, not code. A path that climbs out of the image
        # root is refused rather than trusted.
        if not target.is_relative_to(root):
            return self._error(403, "outside the image root")
        if not target.is_file():
            return self._error(404, f"no such image: {rel}")
        ctype = MEDIA.get(target.suffix.lower(), "application/octet-stream")
        return self._send(200, target.read_bytes(), ctype)

    # ---------------- POST ----------------

    def do_POST(self):
        if urllib.parse.urlparse(self.path).path != "/api/label":
            return self._error(404, "not found")
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self._error(400, "bad json")
        name, item_id = body.get("queue"), body.get("id")
        if not name or not item_id:
            return self._error(400, "queue and id are required")
        try:
            queue_path(name)
        except ValueError as exc:
            return self._error(400, str(exc))
        items = save_label(name, item_id, {
            "choice": body.get("choice"),
            "extra": body.get("extra"),
            "note": body.get("note"),
        })
        done = sum(1 for v in items.values() if v.get("choice"))
        return self._json({"ok": True, "labeled": done})


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--image-root", type=pathlib.Path,
                        default=ROOT / "training_data" / "normalized",
                        help="images inside the queues are relative to this")
    parser.add_argument("--no-open", action="store_true",
                        help="do not open a browser")
    args = parser.parse_args()

    if not QUEUES.is_dir() or not any(list_queues()):
        print(f"no queues in {QUEUES}")
        print("build them first:  python code/audit/build_queues.py")
        return 1

    Handler.image_root = args.image_root.resolve()
    url = f"http://127.0.0.1:{args.port}/"

    print("MMDL audit UI")
    for q in list_queues():
        print(f"  {q['name']:<18} {q['labeled']:>3}/{q['total']:<4} {q['title']}")
    print(f"\n  images  {Handler.image_root}")
    print(f"  labels  {QUEUES}")
    print(f"\n  open    {url}")
    print("\n  Ctrl+C to stop. Every click is saved immediately.")

    if not args.no_open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    with Server(("127.0.0.1", args.port), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
