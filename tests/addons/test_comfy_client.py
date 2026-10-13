"""The stdlib-only ComfyUI client (``vrphoto.addons.workers._comfy``) against a fake ComfyUI server
(``fakes/fake_comfyui.py``): every endpoint, plain errors, timeouts and cancel -> ``/interrupt``,
plus the API-graph helpers (placeholders, ``substitute``, ``load_workflow``)."""

from __future__ import annotations

import ast
import json
import sys
import threading
import time
from pathlib import Path

import pytest

from vrphoto.addons.manifest import workers_dir
from vrphoto.addons.workers import _comfy
from vrphoto.addons.workers._comfy import ComfyCancelled, ComfyClient, ComfyError, ComfyTimeout

sys.path.insert(0, str(Path(__file__).parent / "fakes"))
from fake_comfyui import FakeComfyServer, png_bytes, tqdm_line  # noqa: E402

MODELS = {"unet": ["qwen-image-edit-2511-Q4_K_M.gguf"], "clip": ["Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"],
          "vae": ["qwen_image_vae.safetensors"], "loras": ["a.safetensors", "b.safetensors"]}


def _graph(steps: int = 2, width: int = 48, height: int = 32, prefix: str = "test") -> dict:
    return {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "qwen-image-edit-2511-Q4_K_M.gguf"}},
        "10": {"class_type": "LoadImage", "inputs": {"image": "in.png"}},
        "11": {"class_type": "ImageScale", "inputs": {"image": ["10", 0], "upscale_method": "lanczos",
                                                      "width": width, "height": height, "crop": "disabled"}},
        "31": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "seed": 1, "steps": steps, "cfg": 1.0}},
        "32": {"class_type": "VAEDecode", "inputs": {"samples": ["31", 0]}},
        "33": {"class_type": "SaveImage", "inputs": {"images": ["32", 0], "filename_prefix": prefix}},
    }


@pytest.fixture
def fake(tmp_path):
    with FakeComfyServer(tmp_path / "comfy", model_lists=MODELS) as srv:
        yield srv


def _server(tmp_path, **kw) -> FakeComfyServer:
    srv = FakeComfyServer(tmp_path / "comfy", model_lists=MODELS, **kw)
    srv.__enter__()
    return srv


# ------------------------------------------------------------------ module rules
def test_comfy_module_is_stdlib_only():
    """Imported by the add-on worker (venv) and by the app: standard library only, no _common."""
    tree = ast.parse((workers_dir() / "_comfy.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "no relative imports"
            mods = [(node.module or "").split(".")[0]]
        else:
            continue
        for mod in mods:
            assert mod in set(sys.stdlib_module_names) | {"__future__"}, mod


def test_free_port_and_url_normalisation():
    port = _comfy.free_port()
    assert 0 < port < 65536
    assert _comfy.normalize_url("127.0.0.1:8188/") == "http://127.0.0.1:8188"
    assert _comfy.normalize_url("https://box.local:8443/comfy/") == "https://box.local:8443/comfy"
    assert _comfy.normalize_url("") == "http://127.0.0.1:8188"
    with pytest.raises(ComfyError, match="not a ComfyUI address"):
        _comfy.normalize_url("ftp://x")
    assert _comfy.normalize_url("http://[::1]:8188") == "http://[::1]:8188"
    for bad in ("http://[::1", "http://127.0.0.1:abc", "127.0.0.1:70000", "http://127.0.0.1:0"):
        with pytest.raises(ComfyError, match="not a ComfyUI address"):  # never urllib's raw ValueError
            _comfy.normalize_url(bad)


def test_is_oom_text():
    assert _comfy.is_oom_text("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2 GiB")
    assert _comfy.is_oom_text("This error means you ran out of memory on your GPU.")
    assert _comfy.is_oom_text("Allocation on device")
    assert not _comfy.is_oom_text("mat1 and mat2 shapes cannot be multiplied")
    assert not _comfy.is_oom_text(None)


# ------------------------------------------------------------------ endpoints
def test_system_stats_object_info_and_choices(fake):
    c = ComfyClient(fake.url)
    stats = c.system_stats()
    assert stats["devices"][0]["type"] == "cuda" and c.version() == "0.39.2-fake"
    info = c.object_info()
    assert "KSampler" in info and "UnetLoaderGGUF" in info
    assert set(c.object_info("KSampler")) == {"KSampler"} and c.object_info("Nope") == {}
    assert c.missing_nodes(["KSampler", "Nope", "KSampler", "Nada"]) == ["Nope", "Nada"]
    assert c.choices("UnetLoaderGGUF", "unet_name") == MODELS["unet"]
    assert c.choices("LoraLoaderModelOnly", "lora_name") == MODELS["loras"]
    assert c.choices("KSampler", "sampler_name") == []


def test_choices_understands_the_combo_format(fake, monkeypatch):
    c = ComfyClient(fake.url)
    monkeypatch.setattr(c, "object_info", lambda name=None: {
        "X": {"input": {"required": {"a": ["COMBO", {"options": ["p", "q"]}]}, "optional": {"b": [["r"], {}]}}}})
    assert c.choices("X", "a") == ["p", "q"] and c.choices("X", "b") == ["r"]


def test_missing_nodes_reports_uninstalled_custom_nodes(tmp_path):
    srv = _server(tmp_path, missing_nodes=("UnetLoaderGGUF", "CLIPLoaderGGUF"))
    try:
        assert ComfyClient(srv.url).missing_nodes(["UnetLoaderGGUF", "CLIPLoaderGGUF", "KSampler"]) == [
            "UnetLoaderGGUF", "CLIPLoaderGGUF"]
    finally:
        srv.close()


def test_upload_queue_wait_outputs_download(fake, tmp_path):
    c = ComfyClient(fake.url)
    pic = tmp_path / "in.png"
    pic.write_bytes(png_bytes(8, 6))
    assert c.upload_image(pic) == "in.png"
    assert (fake.input_dir / "in.png").read_bytes() == pic.read_bytes()
    assert c.upload_image(pic, subfolder="vrphoto", name="x.png") == "vrphoto/x.png"
    assert fake.uploads[0]["fields"]["overwrite"] == "true" and fake.uploads[1]["subfolder"] == "vrphoto"
    pid = c.queue(_graph(width=40, height=24))
    assert pid and fake.prompts[-1]["11"]["inputs"]["width"] == 40
    entry = c.wait(pid, poll_s=0.02, timeout_s=10)
    assert entry["status"]["status_str"] == "success"
    refs = c.outputs(entry)
    assert refs == [("test_00001_.png", "", "output")]
    out = c.download(refs[0], tmp_path / "dl" / "view.png")
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" and not (tmp_path / "dl" / "view.png.part").exists()
    from PIL import Image

    with Image.open(out) as im:
        assert im.size == (40, 24)
    folder = tmp_path / "dl2"
    folder.mkdir()
    assert c.download({"filename": refs[0][0], "subfolder": "", "type": "output"}, folder) == folder / refs[0][0]
    # a client-chosen prompt id is passed through
    pid2 = c.queue(_graph(), prompt_id="2b0e6f43-77d6-4e8a-9b1e-6ad0d4f7e7a1")
    assert pid2 == "2b0e6f43-77d6-4e8a-9b1e-6ad0d4f7e7a1"
    c.wait(pid2, poll_s=0.02, timeout_s=10)


def test_outputs_are_ordered_by_node_id():
    entry = {"outputs": {"10": {"images": [{"filename": "b.png", "subfolder": "s", "type": "temp"}]},
                         "9": {"images": [{"filename": "a.png"}], "text": ["x"]},
                         "x": {"images": [{"no": "name"}]}}}
    assert ComfyClient.outputs(entry) == [("a.png", "", "output"), ("b.png", "s", "temp")]
    assert ComfyClient.outputs({}) == []


def test_free_and_queue_state(fake):
    c = ComfyClient(fake.url)
    c.free()
    assert fake.frees == [{"unload_models": True, "free_memory": True}]
    assert c.queue_state() == {"queue_running": [], "queue_pending": []}
    assert c.position("nothing") == (None, 0)


# ------------------------------------------------------------------ errors
def test_connection_refused_is_a_plain_message():
    port = _comfy.free_port()
    c = ComfyClient(f"127.0.0.1:{port}", timeout=5)
    with pytest.raises(ComfyError) as ei:
        c.system_stats()
    assert str(ei.value) == f"ComfyUI is not running at http://127.0.0.1:{port}."
    assert ei.value.kind == "connection" and ei.value.details["reason"] == "refused"
    assert "Start ComfyUI" in ei.value.hint


def test_unknown_route_and_not_comfyui(fake):
    c = ComfyClient(fake.url)
    with pytest.raises(ComfyError) as ei:
        c._request("GET", "/nope")
    assert ei.value.kind == "http" and "HTTP 404" in str(ei.value)


def test_rejected_graph_names_the_node(tmp_path):
    srv = _server(tmp_path, reject=True)
    try:
        with pytest.raises(ComfyError) as ei:
            ComfyClient(srv.url).queue(_graph())
        e = ei.value
        assert e.kind == "rejected" and "node 1 (UnetLoaderGGUF)" in str(e) and "Value not in list" in str(e)
        assert "unet_name" in str(e) and e.hint and e.details["node_errors"]
    finally:
        srv.close()


def test_graph_with_unknown_node_is_rejected(fake):
    g = _graph()
    g["99"] = {"class_type": "NotInstalledNode", "inputs": {}}
    with pytest.raises(ComfyError, match="NotInstalledNode does not exist") as ei:
        ComfyClient(fake.url).queue(g)
    assert ei.value.kind == "rejected"
    with pytest.raises(ComfyError, match="API format"):
        ComfyClient(fake.url).queue({"nodes": [], "links": []})


@pytest.mark.parametrize("kw, kind, text", [
    ({"fail": "mat1 and mat2 shapes cannot be multiplied"}, "execution", "KSampler failed (mat1 and mat2"),
    ({"oom": "always"}, "oom", "ran out of memory in ComfyUI (KSampler)"),
])
def test_execution_errors(tmp_path, kw, kind, text):
    srv = _server(tmp_path, **kw)
    try:
        c = ComfyClient(srv.url)
        pid = c.queue(_graph())
        with pytest.raises(ComfyError) as ei:
            c.wait(pid, poll_s=0.02, timeout_s=10)
        assert ei.value.kind == kind and text in str(ei.value)
        assert ei.value.oom == (kind == "oom") and ei.value.details["node_type"] == "KSampler"
    finally:
        srv.close()


def test_interrupted_elsewhere(tmp_path):
    srv = _server(tmp_path, step_s=0.2)
    try:
        c = ComfyClient(srv.url)
        pid = c.queue(_graph(steps=50))
        threading.Timer(0.3, lambda: ComfyClient(srv.url).interrupt()).start()
        with pytest.raises(ComfyError) as ei:
            c.wait(pid, poll_s=0.02, timeout_s=20)
        assert ei.value.kind == "interrupted" and srv.interrupts == [{}]
    finally:
        srv.close()


# ------------------------------------------------------------------ timeout / cancel -> /interrupt
def test_wait_timeout_interrupts_the_prompt(tmp_path):
    srv = _server(tmp_path, step_s=0.2)
    try:
        c = ComfyClient(srv.url)
        pid = c.queue(_graph(steps=100))
        t0 = time.monotonic()
        with pytest.raises(ComfyTimeout) as ei:
            c.wait(pid, poll_s=0.02, timeout_s=0.4)
        assert time.monotonic() - t0 < 5
        assert ei.value.kind == "timeout" and "did not finish within" in str(ei.value)
        assert {"prompt_id": pid} in srv.interrupts
        deadline = time.monotonic() + 5  # the fake stops at its next step
        while pid not in srv.history and time.monotonic() < deadline:
            time.sleep(0.02)
        assert srv.history[pid]["status"]["messages"][0][0] == "execution_interrupted"
    finally:
        srv.close()


@pytest.mark.parametrize("token", ["event", "callable", "attr"])
def test_cancel_interrupts_the_running_prompt(tmp_path, token):
    srv = _server(tmp_path, step_s=0.2)
    try:
        c = ComfyClient(srv.url)
        pid = c.queue(_graph(steps=100))
        ev = threading.Event()
        threading.Timer(0.3, ev.set).start()
        cancel = {"event": ev, "callable": ev.is_set,
                  "attr": type("Token", (), {"cancelled": property(lambda self: ev.is_set())})()}[token]
        with pytest.raises(ComfyCancelled) as ei:
            c.wait(pid, cancel=cancel, poll_s=0.02, timeout_s=20)
        assert ei.value.kind == "cancelled"
        assert {"prompt_id": pid} in srv.interrupts and pid in srv.deleted
    finally:
        srv.close()


def _wait_running(c: ComfyClient, prompt_id: str, timeout_s: float = 10.0) -> None:
    """Until the fake's executor thread has taken ``prompt_id`` (it runs a moment after queueing)."""
    deadline = time.monotonic() + timeout_s
    while c.position(prompt_id) != ("running", 0):
        assert time.monotonic() < deadline, "the fake ComfyUI never started the prompt"
        time.sleep(0.01)


def test_cancel_removes_a_pending_prompt_and_reports_progress(tmp_path):
    srv = _server(tmp_path, step_s=0.2)
    try:
        c = ComfyClient(srv.url)
        first = c.queue(_graph(steps=100))
        _wait_running(c, first)  # else a loaded machine may still show it pending: "1 ahead"
        second = c.queue(_graph(steps=1))
        seen: list[dict] = []
        polls = {"n": 0}

        def on_progress(info: dict) -> None:
            seen.append(info)
            polls["n"] += 1

        with pytest.raises(ComfyCancelled):
            c.wait(second, cancel=lambda: polls["n"] >= 3, poll_s=0.02, on_progress=on_progress)
        assert seen and all(s["state"] == "pending" and s["ahead"] == 0 for s in seen)
        assert second in srv.deleted and all(p[1] != second for p in srv.pending)
        c.cancel_prompt(first)
    finally:
        srv.close()


def test_the_fake_server_closes_without_being_entered(tmp_path):
    """close() must not wait for a serve_forever loop that __enter__ never started."""
    srv = FakeComfyServer(tmp_path / "comfy")
    errors: list[BaseException] = []

    def close() -> None:
        try:
            srv.close()
        except BaseException as e:  # noqa: BLE001 - reported below
            errors.append(e)

    t = threading.Thread(target=close, daemon=True)
    t.start()
    t.join(10)
    assert not t.is_alive() and not errors and not srv.fake.executor.is_alive()


def test_a_lost_prompt_ends_the_wait(tmp_path):
    srv = _server(tmp_path, step_s=0.2)
    try:
        c = ComfyClient(srv.url)
        first = c.queue(_graph(steps=100))
        second = c.queue(_graph(steps=1))
        c.delete_queued([second])  # e.g. someone cleared ComfyUI's queue
        with pytest.raises(ComfyError, match="lost the job") as ei:
            c.wait(second, poll_s=0.01, on_progress=lambda info: None, timeout_s=20)
        assert ei.value.kind == "protocol"
        c.cancel_prompt(first)
    finally:
        srv.close()


def test_slow_polls_are_tolerated_but_not_forever(fake, monkeypatch):
    c = ComfyClient(fake.url)
    pid = c.queue(_graph())
    real = c.history
    calls = {"n": 0}

    def flaky(prompt_id):
        calls["n"] += 1
        if calls["n"] <= 3:
            raise ComfyError("slow", kind="connection", details={"reason": "timeout"})
        return real(prompt_id)

    monkeypatch.setattr(c, "history", flaky)
    assert c.wait(pid, poll_s=0.01, timeout_s=10)["status"]["status_str"] == "success"

    def always_slow(prompt_id):
        raise ComfyError("slow", kind="connection", details={"reason": "timeout"})

    monkeypatch.setattr(c, "history", always_slow)
    with pytest.raises(ComfyError, match="slow"):
        c.wait(pid, poll_s=0.001, timeout_s=10)

    def refused(prompt_id):
        raise ComfyError("gone", kind="connection", details={"reason": "refused"})

    monkeypatch.setattr(c, "history", refused)
    with pytest.raises(ComfyError, match="gone"):
        c.wait(pid, poll_s=0.001, timeout_s=10)


def test_wait_ready(fake):
    assert ComfyClient(fake.url).wait_ready(5, poll_s=0.05)["system"]["comfyui_version"]
    dead = ComfyClient(f"http://127.0.0.1:{_comfy.free_port()}", timeout=2)
    with pytest.raises(ComfyError, match="stopped while starting"):
        dead.wait_ready(5, alive=lambda: False, poll_s=0.05)
    with pytest.raises(ComfyTimeout, match="did not start within"):
        dead.wait_ready(0.3, poll_s=0.05)
    with pytest.raises(ComfyCancelled):
        dead.wait_ready(5, cancel=lambda: True)


# ------------------------------------------------------------------ graph helpers
def test_placeholders_and_substitute():
    g = {"1": {"class_type": "LoadImage", "inputs": {"image": "{{IMAGE}}"}},
         "2": {"class_type": "KSampler", "inputs": {"seed": "{{ SEED }}", "steps": 8, "cfg": 1.0,
                                                    "list": ["{{width}}", ["{{HEIGHT}}"]]}},
         "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "{{PROMPT}}, seen from {{AZIMUTH}} degrees {{X}}"}}}
    assert _comfy.placeholders(g) == {"IMAGE", "SEED", "WIDTH", "HEIGHT", "PROMPT", "AZIMUTH", "X"}
    before = json.dumps(g, sort_keys=True)
    out = _comfy.substitute(g, {"IMAGE": "in.png", "{{SEED}}": 42, "width": 512, "HEIGHT": 768,
                                "PROMPT": "red coat", "AZIMUTH": 180.0})
    assert json.dumps(g, sort_keys=True) == before  # the input is not changed
    assert out["1"]["inputs"]["image"] == "in.png"
    assert out["2"]["inputs"]["seed"] == 42 and isinstance(out["2"]["inputs"]["seed"], int)
    assert out["2"]["inputs"]["list"] == [512, [768]]
    assert out["3"]["inputs"]["text"] == "red coat, seen from 180 degrees {{X}}"  # unknown stays
    with pytest.raises(ComfyError, match=r"\{\{X\}\}"):
        _comfy.substitute(g, {"IMAGE": "a"}, strict=True)
    # dictionary keys are never substituted
    assert _comfy.substitute({"{{IMAGE}}": "{{IMAGE}}"}, {"IMAGE": "x"}) == {"{{IMAGE}}": "x"}


def test_is_api_graph_and_load_workflow(tmp_path):
    api = _graph()
    assert _comfy.is_api_graph(api)
    assert not _comfy.is_api_graph({}) and not _comfy.is_api_graph({"1": {"inputs": {}}}) and not _comfy.is_api_graph([])
    p = tmp_path / "wf.json"
    p.write_text(json.dumps(api), encoding="utf-8")
    assert _comfy.load_workflow(p) == api
    p.write_text(json.dumps({"prompt": api, "extra": 1}), encoding="utf-8")
    assert _comfy.load_workflow(p) == api
    p.write_text(json.dumps({"nodes": [], "links": [], "version": 0.4}), encoding="utf-8")
    with pytest.raises(ComfyError, match="normal format") as ei:
        _comfy.load_workflow(p)
    assert "Export (API)" in ei.value.hint
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(ComfyError, match="not a valid JSON"):
        _comfy.load_workflow(p)
    with pytest.raises(ComfyError, match="Cannot read"):
        _comfy.load_workflow(tmp_path / "missing.json")


def test_check_entry_and_describe_rejection():
    _comfy.check_entry({"status": {"status_str": "success", "completed": True}})
    with pytest.raises(ComfyError, match="could not run the workflow") as ei:
        _comfy.check_entry({"status": {"status_str": "error", "messages": []}})
    assert ei.value.kind == "execution"
    text = _comfy.describe_rejection({"error": {"message": "Prompt outputs failed validation.", "details": ""},
                                      "node_errors": {"12": {"class_type": "VAELoader", "errors": [
                                          {"message": "Value not in list", "details": "vae_name: 'v' not in []"}]},
                                          "3": {"class_type": "KSampler", "errors": [{"message": "Required input is "
                                                                                      "missing", "details": "model"}]}}})
    assert text.startswith("ComfyUI rejected the workflow (Prompt outputs failed validation)")
    assert text.index("node 3 (KSampler)") < text.index("node 12 (VAELoader)")
    assert _comfy.describe_rejection({"error": "plain"}) == "ComfyUI rejected the workflow (plain)."


def test_fake_tqdm_line_matches_comfyui():
    assert tqdm_line(3, 8).strip().startswith("37%|") and "| 3/8 [" in tqdm_line(3, 8)
