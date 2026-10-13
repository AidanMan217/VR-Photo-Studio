"""A fake ComfyUI server for the tests (standard library only; no torch, no GPU).

Two ways to use it:

* in-process: ``with FakeComfyServer(...) as fake: ComfyClient(fake.url)...`` - an
  ``http.server`` thread plus a thread that "executes" queued prompts;
* as a stand-in for ComfyUI's ``main.py`` (the qwenedit worker tests put a tiny ``main.py`` that
  calls :func:`main` into a fake ComfyUI folder): it parses ComfyUI's command line (``--listen``,
  ``--port``, ``--output-directory``, ``--novram``, ...), prints a tqdm bar per sampling step to
  stderr like ComfyUI's KSampler and serves until it is killed.

Implemented routes: ``GET /system_stats``, ``/object_info``, ``/object_info/{class}``, ``/queue``,
``/history/{id}``, ``/view``; ``POST /prompt``, ``/queue``, ``/interrupt``, ``/upload/image``,
``/free``.  Executing a prompt writes a small PNG for every ``SaveImage`` node (its size from an
``ImageScale`` node when there is one), so the client's ``outputs`` / ``download`` work.

Behaviour switches (keyword arguments, or ``FAKE_COMFY_*`` environment variables in script mode):

* ``step_s`` (``FAKE_COMFY_STEP_S``): seconds per sampling step;
* ``oom`` (``FAKE_COMFY_OOM``): ``"always"``, ``"unless_novram"`` (out of memory unless the server
  was started with ``--novram``) or ``"above:<pixels>"`` (out of memory for larger outputs);
* ``fail`` (``FAKE_COMFY_FAIL``): an execution error message for every prompt;
* ``black`` (``FAKE_COMFY_BLACK``): black pictures unless started with ``--force-fp32``;
* ``reject`` (``FAKE_COMFY_REJECT``): answer ``/prompt`` with HTTP 400;
* ``missing_nodes`` (``FAKE_COMFY_MISSING_NODES``, comma-separated): node types not installed;
* ``crash_after`` (``FAKE_COMFY_CRASH_AFTER``): script mode only - the process exits while
  running prompt number N (1-based) unless started with ``--novram``;

Script mode logs like ComfyUI v0.39.2 (``app/logger.py`` colours every console line, e.g.
``\x1b[1m\x1b[31m[ERROR]\x1b[0m ...``) and, when its port is taken, prints ComfyUI's own line ("Port N
is already in use on address ...") and exits with code 1 (``server.py`` ``start_multi_address``).
* ``record`` (``FAKE_COMFY_RECORD``): a file receiving one JSON line per start / prompt.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import re
import struct
import sys
import threading
import time
import urllib.parse
import uuid
import zlib
from email.parser import BytesParser
from email.policy import HTTP
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

VERSION = "0.39.2-fake"

#: Node types the fake knows, with the combo lists the qwenedit worker checks.
BASE_NODES = (
    "UnetLoaderGGUF", "CLIPLoaderGGUF", "VAELoader", "LoraLoaderModelOnly", "ModelSamplingAuraFlow", "CFGNorm",
    "LoadImage", "ImageScale", "TextEncodeQwenImageEditPlus", "ReferenceLatent",
    "FluxKontextMultiReferenceLatentMethod", "VAEEncode", "KSampler", "VAEDecode", "SaveImage",
    "CheckpointLoaderSimple", "CLIPTextEncode", "EmptyLatentImage",
)

ANSI_RED_BOLD, ANSI_RESET = "\x1b[1m\x1b[31m", "\x1b[0m"


def log_error(message: str) -> None:
    """An ERROR line as ComfyUI's ColoredFormatter writes it to the console."""
    sys.stderr.write(f"{ANSI_RED_BOLD}[ERROR]{ANSI_RESET} {message}\n")
    sys.stderr.flush()


# --------------------------------------------------------------------------- tiny PNG writer
def png_bytes(width: int, height: int, rgb: tuple[int, int, int] = (128, 128, 128),
              blob: tuple[int, int, int] | None = (40, 60, 90)) -> bytes:
    """An RGB PNG of ``width`` x ``height``: a flat colour with a darker ellipse in the middle."""
    w, h = max(1, int(width)), max(1, int(height))
    rows = []
    cx, cy, rx, ry = w / 2, h / 2, max(1.0, w * 0.25), max(1.0, h * 0.4)
    for y in range(h):
        row = bytearray([0])
        dy = ((y + 0.5 - cy) / ry) ** 2
        for x in range(w):
            inside = blob is not None and ((x + 0.5 - cx) / rx) ** 2 + dy <= 1.0
            row += bytes(blob if inside else rgb)
        rows.append(bytes(row))
    raw = zlib.compress(b"".join(rows), 6)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", raw) + chunk(b"IEND", b""))


def tqdm_line(n: int, total: int, *, elapsed: float = 0.0) -> str:
    """A tqdm progress bar line as ComfyUI's samplers print it."""
    frac = n / total if total else 0.0
    bar = ("█" * int(frac * 10)).ljust(10)
    return f"{int(frac * 100):3d}%|{bar}| {n}/{total} [00:{int(elapsed):02d}<00:00,  1.00s/it]"


# --------------------------------------------------------------------------- the fake server state
class FakeComfy:
    """State and prompt execution of the fake (shared by the HTTP handler threads)."""

    def __init__(self, *, output_dir: str | os.PathLike[str], input_dir: str | os.PathLike[str], argv: list[str] | None = None,
                 step_s: float = 0.0, oom: str = "", fail: str = "", black: bool = False, reject: bool = False,
                 missing_nodes: tuple[str, ...] = (), crash_after: int = 0, record: str | None = None,
                 model_lists: dict[str, list[str]] | None = None, emit_tqdm: bool = False,
                 exit_fn: Any = None):
        self.output_dir = Path(output_dir)
        self.input_dir = Path(input_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.argv = list(argv or [])
        self.step_s = float(step_s)
        self.oom = oom
        self.fail = fail
        self.black = black
        self.reject = reject
        self.missing_nodes = set(missing_nodes)
        self.crash_after = int(crash_after)
        self.record = record
        self.model_lists = model_lists or {}
        self.emit_tqdm = emit_tqdm
        self.exit_fn = exit_fn or (lambda code: os._exit(code))
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.pending: list[tuple[float, str, dict]] = []
        self.running: tuple[float, str, dict] | None = None
        self.history: dict[str, dict] = {}
        self.prompts: list[dict] = []  # every accepted graph, in order
        self.interrupts: list[dict] = []
        self.deleted: list[str] = []
        self.uploads: list[dict] = []
        self.frees: list[dict] = []
        self.number = 0
        self._interrupt = threading.Event()
        self._stop = threading.Event()
        self._counter = 0
        self.executor = threading.Thread(target=self._execute_loop, name="fake-comfy-exec", daemon=True)
        self.executor.start()
        self._record({"event": "start", "argv": self.argv})

    # ----------------------------------------------------------------- helpers
    def _record(self, obj: dict) -> None:
        if not self.record:
            return
        with open(self.record, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(obj) + "\n")

    def flag(self, name: str) -> bool:
        return name in self.argv

    def stop(self) -> None:
        self._stop.set()
        with self.cond:
            self.cond.notify_all()

    def node_info(self, name: str) -> dict:
        lists = {
            "UnetLoaderGGUF": ("unet_name", self.model_lists.get("unet", [])),
            "CLIPLoaderGGUF": ("clip_name", self.model_lists.get("clip", [])),
            "VAELoader": ("vae_name", self.model_lists.get("vae", [])),
            "LoraLoaderModelOnly": ("lora_name", self.model_lists.get("loras", [])),
        }
        required: dict[str, Any] = {}
        if name in lists:
            key, values = lists[name]
            required[key] = [list(values)]
        return {"input": {"required": required}, "output": [], "name": name, "display_name": name,
                "category": "fake", "output_node": name == "SaveImage"}

    def object_info(self, name: str | None = None) -> dict:
        names = [n for n in BASE_NODES if n not in self.missing_nodes]
        if name is not None:
            return {name: self.node_info(name)} if name in names else {}
        return {n: self.node_info(n) for n in names}

    # ----------------------------------------------------------------- queue
    def queue_prompt(self, payload: dict) -> tuple[int, dict]:
        graph = payload.get("prompt")
        if not isinstance(graph, dict) or not graph:
            return 400, {"error": {"type": "no_prompt", "message": "No prompt provided", "details": ""},
                         "node_errors": {}}
        missing = sorted({v.get("class_type") for v in graph.values() if isinstance(v, dict)}
                         - set(self.object_info()))
        if missing:
            return 400, {"error": {"type": "invalid_prompt", "message": f"Cannot execute because node {missing[0]} "
                                   "does not exist.", "details": f"Node ID '#{missing[0]}'"}, "node_errors": {}}
        if self.reject:
            nid = next(iter(graph))
            return 400, {"error": {"type": "prompt_outputs_failed_validation",
                                   "message": "Prompt outputs failed validation", "details": ""},
                         "node_errors": {nid: {"errors": [{"type": "value_not_in_list", "message": "Value not in list",
                                                           "details": "unet_name: 'x.gguf' not in []"}],
                                               "class_type": graph[nid].get("class_type", "?")}}}
        pid = str(payload.get("prompt_id") or uuid.uuid4())
        with self.cond:
            self.number += 1
            item = (float(self.number), pid, graph)
            self.pending.append(item)
            self.prompts.append(graph)
            self.cond.notify_all()
        self._record({"event": "prompt", "prompt_id": pid, "prompt": graph})
        return 200, {"prompt_id": pid, "number": self.number, "node_errors": {}}

    def queue_info(self) -> dict:
        with self.lock:
            running = [[n, pid, g, {}, []] for n, pid, g in ([self.running] if self.running else [])]
            pending = [[n, pid, g, {}, []] for n, pid, g in self.pending]
        return {"queue_running": running, "queue_pending": pending}

    def delete(self, ids: list[str]) -> None:
        with self.lock:
            self.deleted.extend(ids)
            self.pending = [p for p in self.pending if p[1] not in ids]

    def interrupt(self, payload: dict) -> None:
        with self.lock:
            self.interrupts.append(dict(payload))
            pid = payload.get("prompt_id")
            if not pid or (self.running and self.running[1] == pid):
                self._interrupt.set()

    # ----------------------------------------------------------------- execution
    def _execute_loop(self) -> None:
        while not self._stop.is_set():
            with self.cond:
                while not self.pending and not self._stop.is_set():
                    self.cond.wait(0.05)
                if self._stop.is_set():
                    return
                item = self.pending.pop(0)
                self.running = item
                self._interrupt.clear()
            entry = self._execute(item)
            with self.lock:
                self.history[item[1]] = entry
                self.running = None

    def _size(self, graph: dict) -> tuple[int, int]:
        """The output size: the ``ImageScale`` behind the KSampler's ``latent_image`` (through a
        ``VAEEncode``), else the first ``ImageScale``, else 64 x 64."""

        def link(node: Any, key: str) -> dict | None:
            v = ((node or {}).get("inputs") or {}).get(key)
            return graph.get(str(v[0])) if isinstance(v, list) and v else None

        scale = None
        for node in graph.values():
            if node.get("class_type") == "KSampler":
                enc = link(node, "latent_image")
                if enc is not None and enc.get("class_type") == "VAEEncode":
                    scale = link(enc, "pixels")
                break
        if scale is None or scale.get("class_type") != "ImageScale":
            scale = next((n for n in graph.values() if n.get("class_type") == "ImageScale"), None)
        if scale is not None:
            ins = scale.get("inputs") or {}
            try:
                return int(ins.get("width") or 64), int(ins.get("height") or 64)
            except (TypeError, ValueError):
                pass
        return 64, 64

    def _error(self, item: tuple, node_type: str, message: str, exc_type: str = "RuntimeError") -> dict:
        data = {"prompt_id": item[1], "node_id": "0", "node_type": node_type, "executed": [],
                "exception_message": message, "exception_type": exc_type, "traceback": [], "current_inputs": {},
                "current_outputs": []}
        return {"prompt": list(item), "outputs": {},
                "status": {"status_str": "error", "completed": False,
                           "messages": [["execution_start", {"prompt_id": item[1]}], ["execution_error", data]]}}

    def _execute(self, item: tuple) -> dict:
        self._counter += 1
        graph = item[2]
        width, height = self._size(graph)
        steps = 1
        for node in graph.values():
            if node.get("class_type") == "KSampler":
                try:
                    steps = max(1, int((node.get("inputs") or {}).get("steps") or 1))
                except (TypeError, ValueError):
                    steps = 1
        if self.crash_after and self._counter >= self.crash_after and not self.flag("--novram"):
            log_error("Fake ComfyUI: simulated crash")
            self.exit_fn(3)
            return self._error(item, "KSampler", "crashed")
        oom = False
        if self.oom == "always":
            oom = True
        elif self.oom == "unless_novram":
            oom = not self.flag("--novram")
        elif self.oom.startswith("above:"):
            oom = width * height > int(self.oom.split(":", 1)[1])
        if oom:
            return self._error(item, "KSampler", "CUDA out of memory. Tried to allocate 2.00 GiB\nThis error means "
                               "you ran out of memory on your GPU.", "torch.OutOfMemoryError")
        if self.fail:
            return self._error(item, "KSampler", self.fail)
        t0 = time.monotonic()
        for n in range(1, steps + 1):
            deadline = time.monotonic() + self.step_s
            while time.monotonic() < deadline:
                if self._interrupt.is_set() or self._stop.is_set():
                    break
                time.sleep(0.01)
            if self._interrupt.is_set() or self._stop.is_set():
                data = {"prompt_id": item[1], "node_id": "0", "node_type": "KSampler", "executed": []}
                return {"prompt": list(item), "outputs": {},
                        "status": {"status_str": "error", "completed": False,
                                   "messages": [["execution_interrupted", data]]}}
            if self.emit_tqdm:
                sys.stderr.write("\r" + tqdm_line(n, steps, elapsed=time.monotonic() - t0))
                sys.stderr.flush()
        if self.emit_tqdm:
            sys.stderr.write("\nPrompt executed in 0.01 seconds\n")
            sys.stderr.flush()
        outputs: dict[str, Any] = {}
        black = self.black and not self.flag("--force-fp32")
        for node_id, node in graph.items():
            if node.get("class_type") != "SaveImage":
                continue
            prefix = str((node.get("inputs") or {}).get("filename_prefix") or "ComfyUI")
            sub, _, base = prefix.rpartition("/")
            folder = self.output_dir / sub
            folder.mkdir(parents=True, exist_ok=True)
            k = len(list(folder.glob(f"{base}_*.png"))) + 1
            name = f"{base}_{k:05d}_.png"
            data = png_bytes(width, height, (0, 0, 0), None) if black else png_bytes(width, height)
            (folder / name).write_bytes(data)
            outputs[node_id] = {"images": [{"filename": name, "subfolder": sub, "type": "output"}]}
        return {"prompt": list(item), "outputs": outputs,
                "status": {"status_str": "success", "completed": True,
                           "messages": [["execution_start", {"prompt_id": item[1]}],
                                        ["execution_success", {"prompt_id": item[1]}]]}}

    def system_stats(self) -> dict:
        return {"system": {"os": sys.platform, "ram_total": 32 * 1024**3, "ram_free": 20 * 1024**3,
                           "comfyui_version": VERSION, "python_version": sys.version, "pytorch_version": "fake",
                           "embedded_python": False, "argv": ["main.py", *self.argv]},
                "devices": [{"name": "cuda:0 NVIDIA GeForce GTX 1080 : cudaMallocAsync", "type": "cuda", "index": 0,
                             "vram_total": 8 * 1024**3, "vram_free": 7 * 1024**3,
                             "torch_vram_total": 0, "torch_vram_free": 0}]}


# --------------------------------------------------------------------------- HTTP
def _handler(fake: FakeComfy) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:  # quiet
            pass

        def _send(self, code: int, obj: Any = None, *, raw: bytes | None = None, ctype: str = "application/json") -> None:
            body = raw if raw is not None else (b"" if obj is None else json.dumps(obj).encode("utf-8"))
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> bytes:
            n = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(n) if n else b""

        def _json_body(self) -> dict:
            try:
                data = json.loads(self._body().decode("utf-8") or "{}")
            except ValueError:
                return {}
            return data if isinstance(data, dict) else {}

        def do_GET(self) -> None:  # noqa: N802 - http.server API
            url = urllib.parse.urlsplit(self.path)
            path, query = url.path, urllib.parse.parse_qs(url.query)
            if path == "/system_stats":
                return self._send(200, fake.system_stats())
            if path == "/object_info":
                return self._send(200, fake.object_info())
            if path.startswith("/object_info/"):
                return self._send(200, fake.object_info(urllib.parse.unquote(path.split("/", 2)[2])))
            if path == "/queue":
                return self._send(200, fake.queue_info())
            m = re.fullmatch(r"/history/(.+)", path)
            if m:
                pid = urllib.parse.unquote(m.group(1))
                with fake.lock:
                    entry = fake.history.get(pid)
                return self._send(200, {pid: entry} if entry is not None else {})
            if path == "/view":
                name = (query.get("filename") or [""])[0]
                sub = (query.get("subfolder") or [""])[0]
                ftype = (query.get("type") or ["output"])[0]
                if not name or ".." in name or name.startswith("/"):
                    return self._send(400)
                base = fake.output_dir if ftype == "output" else fake.input_dir
                f = base / sub / name
                if not f.is_file():
                    return self._send(404)
                return self._send(200, raw=f.read_bytes(), ctype="image/png")
            return self._send(404)

        def do_POST(self) -> None:  # noqa: N802 - http.server API
            path = urllib.parse.urlsplit(self.path).path
            if path == "/prompt":
                code, obj = fake.queue_prompt(self._json_body())
                return self._send(code, obj)
            if path == "/interrupt":
                fake.interrupt(self._json_body())
                return self._send(200)
            if path == "/queue":
                data = self._json_body()
                if data.get("delete"):
                    fake.delete([str(x) for x in data["delete"]])
                if data.get("clear"):
                    with fake.lock:
                        fake.pending.clear()
                return self._send(200)
            if path == "/free":
                fake.frees.append(self._json_body())
                return self._send(200)
            if path == "/upload/image":
                body = self._body()
                msg = BytesParser(policy=HTTP).parsebytes(
                    b"Content-Type: " + self.headers.get("Content-Type", "").encode() + b"\r\n\r\n" + body)
                fields: dict[str, str] = {}
                image: tuple[str, bytes] | None = None
                for part in msg.iter_parts():
                    name = part.get_param("name", header="content-disposition")
                    filename = part.get_filename()
                    data = part.get_payload(decode=True) or b""
                    if filename is not None and name == "image":
                        image = (filename, data)
                    elif name:
                        fields[name] = data.decode("utf-8")
                if image is None or not image[0]:
                    return self._send(400)
                sub = fields.get("subfolder", "")
                folder = fake.input_dir / sub
                folder.mkdir(parents=True, exist_ok=True)
                (folder / image[0]).write_bytes(image[1])
                fake.uploads.append({"name": image[0], "subfolder": sub, "size": len(image[1]), "fields": fields})
                return self._send(200, {"name": image[0], "subfolder": sub, "type": "input"})
            return self._send(404)

    return Handler


class FakeComfyServer:
    """``with FakeComfyServer(tmp_path) as fake: ...`` - serves on ``fake.url`` (a free port)."""

    def __init__(self, root: str | os.PathLike[str], *, port: int = 0, **kw: Any):
        root = Path(root)
        self.fake = FakeComfy(output_dir=root / "output", input_dir=root / "input", **kw)
        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), _handler(self.fake))
        self.httpd.daemon_threads = True
        self.thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.02},
                                       name="fake-comfy-http", daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def __getattr__(self, name: str) -> Any:
        return getattr(self.fake, name)

    def __enter__(self) -> FakeComfyServer:
        self.thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Stop serving and executing (also safe when the server was never entered)."""
        self.fake.stop()
        if self.thread.is_alive():  # shutdown() waits for serve_forever, which only __enter__ starts
            self.httpd.shutdown()
        self.httpd.server_close()
        if self.thread.ident is not None:  # started
            self.thread.join(timeout=5)
        self.fake.executor.join(timeout=5)


# --------------------------------------------------------------------------- script mode (ComfyUI's main.py)
def _parse_argv(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--listen", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8188)
    p.add_argument("--output-directory", default=None)
    p.add_argument("--input-directory", default=None)
    p.add_argument("--temp-directory", default=None)
    p.add_argument("--user-directory", default=None)
    p.add_argument("--extra-model-paths-config", action="append", default=[])
    args, _unknown = p.parse_known_args(argv)
    return args


def _model_lists(yaml_paths: list[str]) -> dict[str, list[str]]:
    """File names in the folders of a ComfyUI ``extra_model_paths.yaml`` (the subset the worker writes:
    ``key: |`` followed by one folder per indented line)."""
    out: dict[str, list[str]] = {}
    key_map = {"diffusion_models": "unet", "text_encoders": "clip", "vae": "vae", "loras": "loras"}
    for yp in yaml_paths:
        key = None
        for line in Path(yp).read_text(encoding="utf-8").splitlines():
            m = re.match(r"^\s{2}([a-z_]+):", line)
            if m:
                key = key_map.get(m.group(1))
                continue
            if key and line.startswith("    ") and line.strip():
                folder = Path(line.strip())
                if folder.is_dir():
                    out.setdefault(key, []).extend(sorted(p.name for p in folder.iterdir() if p.is_file()))
    return out


class _ScriptServer(ThreadingHTTPServer):
    """Binds like ComfyUI's aiohttp / asyncio server: SO_REUSEADDR on POSIX only (on Windows it
    would let the fake share a port that another program holds)."""

    allow_reuse_address = os.name == "posix"


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = _parse_argv(argv)
    env = os.environ
    out_dir = args.output_directory or os.path.join(os.getcwd(), "output")
    in_dir = args.input_directory or os.path.join(os.getcwd(), "input")
    sys.stderr.write(f"Fake ComfyUI {VERSION} starting\n")
    sys.stderr.flush()
    time.sleep(float(env.get("FAKE_COMFY_START_S", "0") or 0))
    fake = FakeComfy(output_dir=out_dir, input_dir=in_dir, argv=argv,
                     step_s=float(env.get("FAKE_COMFY_STEP_S", "0") or 0),
                     oom=env.get("FAKE_COMFY_OOM", ""), fail=env.get("FAKE_COMFY_FAIL", ""),
                     black=bool(env.get("FAKE_COMFY_BLACK")), reject=bool(env.get("FAKE_COMFY_REJECT")),
                     missing_nodes=tuple(x for x in env.get("FAKE_COMFY_MISSING_NODES", "").split(",") if x),
                     crash_after=int(env.get("FAKE_COMFY_CRASH_AFTER", "0") or 0),
                     record=env.get("FAKE_COMFY_RECORD") or None,
                     model_lists=_model_lists([p for group in args.extra_model_paths_config for p in
                                               ([group] if isinstance(group, str) else group)]),
                     emit_tqdm=True)
    try:
        httpd = _ScriptServer((args.listen, args.port), _handler(fake))
    except OSError as e:
        if e.errno not in (errno.EADDRINUSE, 10048):  # 10048: WSAEADDRINUSE
            raise
        fake.stop()
        log_error(f"Port {args.port} is already in use on address {args.listen}. Please close the other "
                  "application or use a different port with --port.")
        raise SystemExit(1) from None
    httpd.daemon_threads = True
    sys.stderr.write(f"Starting server\nTo see the GUI go to: http://{args.listen}:{args.port}\n")
    sys.stderr.flush()
    try:
        httpd.serve_forever(poll_interval=0.05)
    finally:
        fake.stop()
        httpd.server_close()


if __name__ == "__main__":
    main()
