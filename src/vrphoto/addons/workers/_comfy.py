"""A small ComfyUI HTTP client and helpers for ComfyUI's API-format graphs (standard library only).

Used in two places:

* inside the ``qwenedit`` add-on worker (``qwenedit_worker.py``), which drives a private,
  headless ComfyUI it starts itself on ``127.0.0.1``;
* in the app (``vrphoto.addons.workers._comfy``) by the expert 'Your ComfyUI server' back source,
  which drives the user's own ComfyUI with the user's own workflow.

It must stay importable both ways: no ``vrphoto`` imports, no ``_common`` import, only the
standard library (``tests/addons/test_comfy_client.py`` checks this).

ComfyUI's HTTP API (checked against ComfyUI v0.39.2, ``server.py``):

* ``POST /prompt`` ``{"prompt": graph, "client_id"}`` -> ``{"prompt_id", "number", "node_errors"}``;
  an invalid graph -> HTTP 400 ``{"error": {"type", "message", "details"}, "node_errors": {id:
  {"errors": [{"message", "details"}], "class_type"}}}``.
* ``GET /history/{id}`` -> ``{id: {"prompt", "outputs": {node: {"images": [{"filename",
  "subfolder", "type"}]}}, "status": {"status_str": "success" | "error", "completed",
  "messages": [[event, data], ...]}}}`` once the prompt has finished (``{}`` before);
  ``execution_error`` data hold ``node_type`` and ``exception_message``.
* ``GET /queue`` -> ``{"queue_running": [[number, id, ...]], "queue_pending": [...]}``;
  ``POST /queue`` ``{"delete": [ids]}``; ``POST /interrupt`` (optionally ``{"prompt_id"}``).
* ``POST /upload/image`` (multipart ``image``, ``overwrite``, ``subfolder``, ``type``) ->
  ``{"name", "subfolder", "type"}``; ``GET /view?filename=&subfolder=&type=`` -> the bytes.
* ``GET /object_info`` / ``GET /object_info/{class}``; ``GET /system_stats`` ->
  ``{"system": {"comfyui_version", "ram_total", ...}, "devices": [{"name", "type", "vram_total",
  ...}]}``; ``POST /free``.
"""

from __future__ import annotations

import copy
import http.client
import json
import mimetypes
import os
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

#: Placeholders the app fills in an expert workflow (design 1.2, ``comfyui_workflow``).
PLACEHOLDERS = ("IMAGE", "PROMPT", "SEED", "AZIMUTH", "WIDTH", "HEIGHT")
_PLACEHOLDER_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")
DEFAULT_URL = "http://127.0.0.1:8188"
NOT_RUNNING_HINT = ("Start ComfyUI and check the address (for example http://127.0.0.1:8188) in "
                    "'Your ComfyUI server'.")
API_FORMAT_HINT = ("In ComfyUI, open the workflow and save it with 'Export (API)' (Workflow menu), then "
                   "pick that file.")
#: How many polls in a row a queued prompt may be missing from both the queue and the history
#: before :meth:`ComfyClient.wait` gives up (ComfyUI was restarted or the queue was cleared).
_LOST_POLLS = 6
#: How many polls in a row may time out (ComfyUI busy) before :meth:`ComfyClient.wait` gives up.
_SLOW_POLLS = 10


class ComfyError(Exception):
    """A plain-language ComfyUI failure.

    ``kind``: ``"connection"`` (not running / unreachable), ``"http"``, ``"rejected"`` (the graph
    did not validate), ``"execution"`` (a node failed), ``"oom"`` (out of memory while running),
    ``"interrupted"``, ``"cancelled"``, ``"timeout"`` or ``"protocol"`` (an unexpected answer).
    ``details`` keeps ComfyUI's own data (node errors, the failing node's message, ...).
    """

    def __init__(self, message: str, hint: str | None = None, *, kind: str = "error",
                 details: dict[str, Any] | None = None):
        super().__init__(message)
        self.hint = hint
        self.kind = kind
        self.details = details or {}

    @property
    def oom(self) -> bool:
        return self.kind == "oom"


class ComfyCancelled(ComfyError):
    """The caller cancelled a prompt (it was interrupted / removed from ComfyUI's queue)."""


class ComfyTimeout(ComfyError):
    """A prompt or the server start took longer than allowed (the prompt was interrupted)."""


# --------------------------------------------------------------------------- small helpers
def is_oom_text(text: str | None) -> bool:
    """True for the out-of-memory messages of PyTorch / ComfyUI."""
    t = str(text or "")
    low = t.lower()
    return ("out of memory" in low or "outofmemoryerror" in low or "cuda_error_out_of_memory" in low
            or "cublas_status_alloc_failed" in low or "allocation on device" in low
            or "cudaerrormemoryallocation" in low)


def _is_cancelled(cancel: Any) -> bool:
    """``cancel`` may be None, a callable returning bool, a ``threading.Event`` or an object with a
    ``cancelled`` attribute (the app's ``CancelToken``)."""
    if cancel is None:
        return False
    if callable(cancel) and not hasattr(cancel, "is_set") and not hasattr(cancel, "cancelled"):
        return bool(cancel())
    if hasattr(cancel, "cancelled"):
        return bool(cancel.cancelled)
    if hasattr(cancel, "is_set"):
        return bool(cancel.is_set())
    return False


def free_port(host: str = "127.0.0.1") -> int:
    """A TCP port that is free on ``host`` right now (the OS picks it)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def normalize_url(url: str | None) -> str:
    """``"127.0.0.1:8188/"`` -> ``"http://127.0.0.1:8188"``; raises :class:`ComfyError` for
    something that is not an http(s) address."""
    u = str(url or "").strip() or DEFAULT_URL
    if "://" not in u:
        u = "http://" + u
    bad = ComfyError(f"'{url}' is not a ComfyUI address.", hint="Use an address like http://127.0.0.1:8188.",
                     kind="connection")
    try:
        parts = urllib.parse.urlsplit(u)
        port = parts.port  # ValueError for a port that is not a number from 0 to 65535
    except ValueError as e:  # e.g. "http://[::1" (an unclosed IPv6 address)
        raise bad from e
    if parts.scheme not in ("http", "https") or not parts.hostname or port == 0:
        raise bad
    path = parts.path.rstrip("/")
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def _first_line(text: Any, limit: int = 300) -> str:
    s = str(text or "").strip()
    return (s.splitlines()[0] if s else "")[:limit]


# --------------------------------------------------------------------------- graphs
def is_api_graph(graph: Any) -> bool:
    """True for ComfyUI's API format: ``{node_id: {"class_type": str, "inputs": {...}}, ...}``."""
    if not isinstance(graph, dict) or not graph:
        return False
    return all(isinstance(v, dict) and isinstance(v.get("class_type"), str) and isinstance(v.get("inputs", {}), dict)
               for v in graph.values())


def load_workflow(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read an API-format workflow file (``{"prompt": graph}`` wrappers are unwrapped).  A workflow
    saved in ComfyUI's normal UI format (``nodes`` + ``links``) is refused with a hint."""
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8-sig"))
    except OSError as e:
        raise ComfyError(f"Cannot read the ComfyUI workflow {p.name}: {e}", kind="protocol") from e
    except ValueError as e:
        raise ComfyError(f"The ComfyUI workflow {p.name} is not a valid JSON file ({e}).", hint=API_FORMAT_HINT,
                         kind="protocol") from e
    if isinstance(data, dict) and is_api_graph(data.get("prompt")):
        data = data["prompt"]
    if is_api_graph(data):
        return data
    if isinstance(data, dict) and "nodes" in data and "links" in data:
        raise ComfyError(f"The ComfyUI workflow {p.name} is saved in ComfyUI's normal format, not the API format.",
                         hint=API_FORMAT_HINT, kind="protocol")
    raise ComfyError(f"{p.name} does not look like a ComfyUI workflow.", hint=API_FORMAT_HINT, kind="protocol")


def placeholders(graph: Any) -> set[str]:
    """Names of the ``{{NAME}}`` placeholders used in the string values of ``graph``."""
    found: set[str] = set()

    def walk(v: Any) -> None:
        if isinstance(v, str):
            found.update(m.group(1).upper() for m in _PLACEHOLDER_RE.finditer(v))
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)

    walk(graph)
    return found


def _norm_key(k: str) -> str:
    m = _PLACEHOLDER_RE.fullmatch(str(k).strip())
    return (m.group(1) if m else str(k).strip()).upper()


def _text(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def substitute(graph: Any, mapping: Mapping[str, Any], *, strict: bool = False) -> Any:
    """A copy of ``graph`` with ``{{IMAGE}}``-style placeholders replaced from ``mapping``.

    Keys may be given as ``"SEED"`` or ``"{{SEED}}"`` (any case).  A string that is exactly one
    placeholder becomes the mapped value itself (so ``"{{SEED}}"`` -> ``42``, an int, as ComfyUI's
    INT inputs need); a placeholder inside a longer string is replaced by its text
    (``"a {{PROMPT}}"`` -> ``"a red coat"``).  Dictionary keys are never changed.  Placeholders
    without a value stay as they are, unless ``strict`` (then :class:`ComfyError` names them)."""
    values = {_norm_key(k): v for k, v in dict(mapping or {}).items()}
    unknown: set[str] = set()

    def sub_str(s: str) -> Any:
        whole = _PLACEHOLDER_RE.fullmatch(s.strip())
        if whole:
            name = whole.group(1).upper()
            if name in values:
                return copy.deepcopy(values[name])
            unknown.add(name)
            return s

        def rep(m: re.Match[str]) -> str:
            name = m.group(1).upper()
            if name in values:
                return _text(values[name])
            unknown.add(name)
            return m.group(0)

        return _PLACEHOLDER_RE.sub(rep, s)

    def walk(v: Any) -> Any:
        if isinstance(v, str):
            return sub_str(v)
        if isinstance(v, dict):
            return {k: walk(x) for k, x in v.items()}
        if isinstance(v, list):
            return [walk(x) for x in v]
        if isinstance(v, tuple):
            return [walk(x) for x in v]
        return copy.deepcopy(v)

    out = walk(graph)
    if strict and unknown:
        names = ", ".join("{{" + n + "}}" for n in sorted(unknown))
        known = " ".join("{{" + n + "}}" for n in PLACEHOLDERS)
        raise ComfyError(f"The workflow uses {names}, which has no value (VR Photo Studio fills in {known}).",
                         kind="protocol")
    return out


def _multipart(fields: Mapping[str, str], files: Iterable[tuple[str, str, bytes, str]]) -> tuple[bytes, str]:
    """``multipart/form-data`` body and its content type."""
    boundary = "----vrphoto" + uuid.uuid4().hex
    out: list[bytes] = []
    for name, value in fields.items():
        out += [f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                str(value).encode("utf-8"), b"\r\n"]
    for name, filename, data, ctype in files:
        safe = filename.replace('"', "_").replace("\r", "_").replace("\n", "_")
        out += [f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"; filename="{safe}"\r\n'.encode(),
                f"Content-Type: {ctype}\r\n\r\n".encode(), data, b"\r\n"]
    out.append(f"--{boundary}--\r\n".encode())
    return b"".join(out), f"multipart/form-data; boundary={boundary}"


def _node_sort_key(node_id: Any) -> tuple[int, Any]:
    s = str(node_id)
    return (0, int(s)) if s.isdigit() else (1, s)


# --------------------------------------------------------------------------- the client
class ComfyClient:
    """Talks to one ComfyUI server over HTTP (``urllib``; proxies are bypassed: ComfyUI runs on this
    PC or the local network).  Every failure is a :class:`ComfyError` with a plain message."""

    def __init__(self, base_url: str = DEFAULT_URL, *, timeout: float = 30, client_id: str | None = None):
        self.base_url = normalize_url(base_url)
        self.timeout = float(timeout)
        self.client_id = client_id or uuid.uuid4().hex
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def __repr__(self) -> str:
        return f"ComfyClient({self.base_url!r})"

    # ----------------------------------------------------------------- transport
    def _request(self, method: str, path: str, *, query: Mapping[str, Any] | None = None,
                 body: bytes | None = None, headers: Mapping[str, str] | None = None,
                 timeout: float | None = None, raw: bool = False) -> Any:
        url = self.base_url + path
        if query:
            url += "?" + urllib.parse.urlencode({k: v for k, v in query.items() if v is not None})
        req = urllib.request.Request(url, data=body, method=method, headers=dict(headers or {}))
        t = self.timeout if timeout is None else float(timeout)
        try:
            with self._opener.open(req, timeout=t) as resp:
                data = resp.read()
        except urllib.error.HTTPError as e:
            try:
                payload = e.read()
            except Exception:
                payload = b""
            raise self._http_error(e.code, payload, path) from e
        except http.client.HTTPException as e:  # the server went away in the middle of an answer
            raise ComfyError(f"ComfyUI at {self.base_url} stopped answering ({type(e).__name__}).",
                             hint=NOT_RUNNING_HINT, kind="connection", details={"reason": "disconnected"}) from e
        except (urllib.error.URLError, OSError) as e:
            reason = getattr(e, "reason", e)
            if isinstance(reason, ConnectionRefusedError) or "refused" in str(reason).lower():
                raise ComfyError(f"ComfyUI is not running at {self.base_url}.", hint=NOT_RUNNING_HINT,
                                 kind="connection", details={"reason": "refused"}) from e
            if isinstance(reason, (TimeoutError, socket.timeout)) or "timed out" in str(reason).lower():
                raise ComfyError(f"ComfyUI at {self.base_url} did not answer within {t:g} s.",
                                 hint="ComfyUI may be busy or stuck; restart it and try again.",
                                 kind="connection", details={"reason": "timeout"}) from e
            raise ComfyError(f"Cannot reach ComfyUI at {self.base_url} ({reason}).", hint=NOT_RUNNING_HINT,
                             kind="connection", details={"reason": str(reason)[:200]}) from e
        if raw:
            return data
        if not data.strip():
            return None
        try:
            return json.loads(data.decode("utf-8"))
        except ValueError as e:
            raise ComfyError(f"ComfyUI at {self.base_url} sent an answer that is not JSON ({path}).",
                             hint="Check that the address points to ComfyUI.", kind="protocol") from e

    def _json(self, method: str, path: str, payload: Any = None, **kw: Any) -> Any:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"} if body is not None else None
        return self._request(method, path, body=body, headers=headers, **kw)

    def _http_error(self, code: int, payload: bytes, path: str) -> ComfyError:
        try:
            data = json.loads(payload.decode("utf-8")) if payload else None
        except ValueError:
            data = None
        if path == "/prompt" and code == 400 and isinstance(data, dict):
            return ComfyError(describe_rejection(data), hint="Check the workflow: the models and custom nodes it "
                              "uses must be installed in this ComfyUI.", kind="rejected", details=data)
        text = _first_line(payload.decode("utf-8", "replace")) if payload else ""
        return ComfyError(f"ComfyUI at {self.base_url} answered HTTP {code} for {path}" + (f": {text}" if text else "."),
                          kind="http", details={"status": code, "body": text})

    # ----------------------------------------------------------------- endpoints
    def system_stats(self) -> dict[str, Any]:
        data = self._request("GET", "/system_stats")
        if not isinstance(data, dict) or "system" not in data:
            raise ComfyError(f"The server at {self.base_url} does not look like ComfyUI.",
                             hint="Check the address and port.", kind="protocol")
        return data

    def version(self) -> str:
        """ComfyUI's version (``""`` when it does not say)."""
        return str((self.system_stats().get("system") or {}).get("comfyui_version") or "")

    def object_info(self, node_class: str | None = None) -> dict[str, Any]:
        """All node definitions, or one (``{}`` when ``node_class`` is not installed)."""
        path = "/object_info" if node_class is None else "/object_info/" + urllib.parse.quote(node_class, safe="")
        data = self._request("GET", path, timeout=max(self.timeout, 120.0))
        return data if isinstance(data, dict) else {}

    def missing_nodes(self, names: Iterable[str]) -> list[str]:
        """The node types in ``names`` this ComfyUI does not have (asked one by one)."""
        missing = []
        for name in dict.fromkeys(names):
            try:
                info = self.object_info(name)
            except ComfyError as e:
                if e.kind != "http":
                    raise
                info = self.object_info()  # very old servers: no per-node route
            if name not in info:
                missing.append(name)
        return missing

    def choices(self, node_class: str, input_name: str) -> list[str]:
        """The list of allowed values of a combo input (e.g. the model files a loader can see)."""
        info = self.object_info(node_class).get(node_class) or {}
        for group in ("required", "optional"):
            spec = ((info.get("input") or {}).get(group) or {}).get(input_name)
            if isinstance(spec, list) and spec:
                first = spec[0]
                if isinstance(first, list):
                    return [str(x) for x in first]
                if first == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
                    return [str(x) for x in spec[1].get("options") or []]
        return []

    def upload_image(self, path: str | os.PathLike[str], *, subfolder: str = "", overwrite: bool = True,
                     name: str | None = None) -> str:
        """Upload a picture to ComfyUI's input folder; returns the value for a ``LoadImage`` node
        (``"name"`` or ``"subfolder/name"``)."""
        p = Path(path)
        try:
            data = p.read_bytes()
        except OSError as e:
            raise ComfyError(f"Cannot read the picture {p.name}: {e}", kind="protocol") from e
        filename = name or p.name
        ctype = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        fields = {"overwrite": "true" if overwrite else "false", "type": "input"}
        if subfolder:
            fields["subfolder"] = subfolder
        body, content_type = _multipart(fields, [("image", filename, data, ctype)])
        res = self._request("POST", "/upload/image", body=body, headers={"Content-Type": content_type},
                            timeout=max(self.timeout, 120.0))
        if not isinstance(res, dict) or not res.get("name"):
            raise ComfyError(f"ComfyUI did not accept the picture {p.name}.", kind="protocol", details={"answer": res})
        sub = str(res.get("subfolder") or "")
        return f"{sub}/{res['name']}" if sub else str(res["name"])

    def queue(self, graph: Mapping[str, Any], *, prompt_id: str | None = None,
              extra_data: Mapping[str, Any] | None = None) -> str:
        """Queue an API-format graph; returns its prompt id.  An invalid graph raises
        :class:`ComfyError` (``kind="rejected"``) naming the failing nodes."""
        if not is_api_graph(dict(graph)):
            raise ComfyError("The workflow is not in ComfyUI's API format.", hint=API_FORMAT_HINT, kind="protocol")
        payload: dict[str, Any] = {"prompt": dict(graph), "client_id": self.client_id}
        if prompt_id:
            payload["prompt_id"] = prompt_id
        if extra_data:
            payload["extra_data"] = dict(extra_data)
        res = self._json("POST", "/prompt", payload)
        pid = res.get("prompt_id") if isinstance(res, dict) else None
        if not pid:
            raise ComfyError("ComfyUI did not queue the workflow.", kind="protocol", details={"answer": res})
        return str(pid)

    def queue_state(self) -> dict[str, Any]:
        data = self._request("GET", "/queue")
        return data if isinstance(data, dict) else {}

    def position(self, prompt_id: str) -> tuple[str | None, int]:
        """``("running", 0)``, ``("pending", ahead)`` or ``(None, 0)`` when it is in neither list."""
        q = self.queue_state()
        if any(len(item) > 1 and item[1] == prompt_id for item in q.get("queue_running") or []):
            return "running", 0
        pending = sorted((item for item in q.get("queue_pending") or [] if len(item) > 1),
                         key=lambda item: item[0] if isinstance(item[0], (int, float)) else 0)
        for k, item in enumerate(pending):
            if item[1] == prompt_id:
                return "pending", k
        return None, 0

    def history(self, prompt_id: str) -> dict[str, Any] | None:
        data = self._request("GET", "/history/" + urllib.parse.quote(prompt_id, safe=""))
        if isinstance(data, dict):
            entry = data.get(prompt_id)
            return entry if isinstance(entry, dict) else None
        return None

    def interrupt(self, prompt_id: str | None = None) -> None:
        """Stop the running prompt (only ``prompt_id`` when given)."""
        self._json("POST", "/interrupt", {"prompt_id": prompt_id} if prompt_id else {})

    def delete_queued(self, prompt_ids: Iterable[str]) -> None:
        ids = list(prompt_ids)
        if ids:
            self._json("POST", "/queue", {"delete": ids})

    def cancel_prompt(self, prompt_id: str) -> None:
        """Remove ``prompt_id`` from the queue and interrupt it if it runs (best effort)."""
        for call in (lambda: self.delete_queued([prompt_id]), lambda: self.interrupt(prompt_id)):
            try:
                call()
            except ComfyError:
                pass

    def free(self, *, unload_models: bool = True, free_memory: bool = True) -> None:
        self._json("POST", "/free", {"unload_models": unload_models, "free_memory": free_memory})

    # ----------------------------------------------------------------- waiting
    def _listening(self, timeout: float = 0.25) -> bool:
        """For a server on this PC: is anything accepting connections on its port yet?  A short TCP
        probe (Windows takes about 2 s to refuse a connection to a closed local port, so polling
        with HTTP requests would wait that long each time).  Other hosts: always True."""
        parts = urllib.parse.urlsplit(self.base_url)
        host = parts.hostname or ""
        if not (host in ("localhost", "::1") or host.startswith("127.")):
            return True
        try:
            port = parts.port or (443 if parts.scheme == "https" else 80)
        except ValueError:
            return True
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            return False

    def wait_ready(self, timeout_s: float = 120.0, *, cancel: Any = None,
                   alive: Callable[[], bool] | None = None, poll_s: float = 0.5) -> dict[str, Any]:
        """Poll ``/system_stats`` until ComfyUI answers; returns it.  ``alive()`` returning False (the
        server process ended) raises :class:`ComfyError` at once."""
        t0 = time.monotonic()
        last: ComfyError | None = None
        while True:
            if _is_cancelled(cancel):
                raise ComfyCancelled("Cancelled while waiting for ComfyUI to start.", kind="cancelled")
            if alive is not None and not alive():
                raise ComfyError("ComfyUI stopped while starting.", kind="connection",
                                 details={"last_error": str(last) if last else ""})
            if self._listening():
                try:
                    return self.system_stats()
                except ComfyError as e:
                    if e.kind not in ("connection", "http", "protocol"):
                        raise
                    last = e
            if time.monotonic() - t0 > timeout_s:
                raise ComfyTimeout(f"ComfyUI did not start within {timeout_s:.0f} s.", kind="timeout",
                                   details={"last_error": str(last) if last else ""})
            _sleep(poll_s, cancel)

    def wait(self, prompt_id: str, *, cancel: Any = None, poll_s: float = 0.5, timeout_s: float | None = None,
             on_progress: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
        """Wait for a queued prompt and return its history entry.

        Raises :class:`ComfyError` when it failed (``kind="oom"`` for out of memory, else
        ``"execution"``; ``"interrupted"`` when someone stopped it), :class:`ComfyCancelled` when
        ``cancel`` fired and :class:`ComfyTimeout` after ``timeout_s`` - in both cases the prompt is
        first removed from the queue / interrupted (``POST /interrupt``).  ``on_progress`` gets
        ``{"state": "pending" | "running", "ahead", "elapsed"}`` while it waits."""
        t0 = time.monotonic()
        missing = 0
        slow = 0
        last_pos = -1e9
        while True:
            if _is_cancelled(cancel):
                self.cancel_prompt(prompt_id)
                raise ComfyCancelled("Cancelled; ComfyUI was asked to stop.", kind="cancelled",
                                     details={"prompt_id": prompt_id})
            elapsed = time.monotonic() - t0
            if timeout_s is not None and elapsed > timeout_s:
                self.cancel_prompt(prompt_id)
                raise ComfyTimeout(f"ComfyUI did not finish within {timeout_s:.0f} s; it was asked to stop.",
                                   kind="timeout", details={"prompt_id": prompt_id})
            asked = False
            state, ahead = None, 0
            try:
                entry = self.history(prompt_id)
                if entry is None and (on_progress is not None or time.monotonic() - last_pos >= 2.0):
                    last_pos, asked = time.monotonic(), True
                    state, ahead = self.position(prompt_id)
                    if state is None:
                        entry = self.history(prompt_id)  # it may have finished between the two calls
                slow = 0
            except ComfyError as e:
                # a busy ComfyUI (loading a model with the GIL held) may miss an answer: tolerate a few
                # slow polls, never a refused connection (the server is gone)
                if e.kind != "connection" or e.details.get("reason") != "timeout" or slow >= _SLOW_POLLS:
                    raise
                slow += 1
                _sleep(poll_s, cancel)
                continue
            if entry is not None:
                status = entry.get("status") or {}
                if status.get("completed") or status.get("status_str") in ("success", "error") or entry.get("outputs"):
                    check_entry(entry)
                    return entry
            elif asked:
                if state is None:
                    missing += 1
                    if missing >= _LOST_POLLS:
                        raise ComfyError("ComfyUI lost the job (it was restarted or its queue was cleared).",
                                         kind="protocol", details={"prompt_id": prompt_id})
                else:
                    missing = 0
                    if on_progress is not None:
                        on_progress({"state": state, "ahead": ahead, "elapsed": elapsed})
            _sleep(poll_s, cancel)

    # ----------------------------------------------------------------- results
    @staticmethod
    def outputs(entry: Mapping[str, Any], *, keys: Iterable[str] = ("images",)) -> list[tuple[str, str, str]]:
        """``(filename, subfolder, type)`` of every output picture of a history entry, by node id."""
        out: list[tuple[str, str, str]] = []
        nodes = entry.get("outputs") or {}
        for node_id in sorted(nodes, key=_node_sort_key):
            res = nodes[node_id] or {}
            for key in keys:
                for item in res.get(key) or []:
                    if isinstance(item, dict) and item.get("filename"):
                        out.append((str(item["filename"]), str(item.get("subfolder") or ""),
                                    str(item.get("type") or "output")))
        return out

    def download(self, file_ref: tuple[str, str, str] | Mapping[str, str], dest: str | os.PathLike[str]) -> Path:
        """Fetch one output picture (``/view``) to ``dest`` (a file, or a folder for its own name);
        written atomically."""
        if isinstance(file_ref, Mapping):
            filename, subfolder, ftype = (str(file_ref.get("filename") or ""), str(file_ref.get("subfolder") or ""),
                                          str(file_ref.get("type") or "output"))
        else:
            filename, subfolder, ftype = (str(x) for x in file_ref)
        if not filename:
            raise ComfyError("No file name to download.", kind="protocol")
        data = self._request("GET", "/view", query={"filename": filename, "subfolder": subfolder, "type": ftype},
                             raw=True, timeout=max(self.timeout, 120.0))
        d = Path(dest)
        target = d / Path(filename).name if d.is_dir() else d
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".part")
        tmp.write_bytes(data)
        os.replace(tmp, target)
        return target


def _sleep(seconds: float, cancel: Any) -> None:
    end = time.monotonic() + max(0.0, seconds)
    while True:
        left = end - time.monotonic()
        if left <= 0 or _is_cancelled(cancel):
            return
        time.sleep(min(left, 0.05))


def describe_rejection(data: Mapping[str, Any]) -> str:
    """One plain sentence for ComfyUI's HTTP 400 answer to ``/prompt``."""
    err = data.get("error") or {}
    if isinstance(err, str):
        head = err
    else:
        head = str(err.get("message") or "the workflow is not valid").rstrip(".")
        det = str(err.get("details") or "").strip()
        if det and det not in head:
            head += f": {det}"
    parts = []
    for node_id, info in sorted((data.get("node_errors") or {}).items(), key=lambda kv: _node_sort_key(kv[0])):
        if not isinstance(info, dict):
            continue
        msgs = []
        for e in info.get("errors") or []:
            if isinstance(e, dict):
                m = str(e.get("message") or "")
                d = str(e.get("details") or "")
                msgs.append(f"{m}: {d}" if d and d not in m else m)
        if msgs:
            parts.append(f"node {node_id} ({info.get('class_type', '?')}): " + "; ".join(msgs))
    text = f"ComfyUI rejected the workflow ({head})"
    if parts:
        text += " - " + " | ".join(parts[:4])
    return text[:1500] + "."


def check_entry(entry: Mapping[str, Any]) -> None:
    """Raise :class:`ComfyError` when a history entry says the prompt failed or was interrupted."""
    status = entry.get("status") or {}
    if status.get("status_str") != "error":
        return
    error: dict[str, Any] | None = None
    interrupted: dict[str, Any] | None = None
    for msg in status.get("messages") or []:
        if not isinstance(msg, (list, tuple)) or len(msg) < 2:
            continue
        event, data = msg[0], msg[1] if isinstance(msg[1], dict) else {}
        if event == "execution_error":
            error = data
        elif event == "execution_interrupted":
            interrupted = data
    if error is not None:
        node = error.get("node_type") or "a node"
        text = str(error.get("exception_message") or error.get("exception_type") or "unknown error")
        oom = is_oom_text(text) or is_oom_text(error.get("exception_type"))
        if oom:
            raise ComfyError(f"The graphics card ran out of memory in ComfyUI ({node}).", kind="oom", details=error)
        raise ComfyError(f"ComfyUI could not run the workflow: {node} failed ({_first_line(text)}).",
                         kind="execution", details=error)
    if interrupted is not None:
        raise ComfyError("The ComfyUI job was interrupted.", kind="interrupted", details=interrupted)
    raise ComfyError("ComfyUI could not run the workflow.", kind="execution", details=dict(status))


__all__ = [
    "DEFAULT_URL",
    "PLACEHOLDERS",
    "ComfyCancelled",
    "ComfyClient",
    "ComfyError",
    "ComfyTimeout",
    "check_entry",
    "describe_rejection",
    "free_port",
    "is_api_graph",
    "is_oom_text",
    "load_workflow",
    "normalize_url",
    "placeholders",
    "substitute",
]
