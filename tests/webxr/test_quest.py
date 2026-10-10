"""Tests for vrphoto.webxr.quest (adb / platform-tools / MTP), without real devices."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

import pytest

from vrphoto.core.errors import Cancelled, DownloadError
from vrphoto.core.types import CancelToken
from vrphoto.webxr import quest

ADB = "adb.exe" if sys.platform == "win32" else "adb"


# --------------------------------------------------------------------------- environment isolation
@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    """No adb anywhere: empty PATH, no SDK variables, HOME in a temp dir."""
    empty = tmp_path / "emptybin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    for var in ("VRPHOTO_ADB", "ANDROID_HOME", "ANDROID_SDK_ROOT", "LOCALAPPDATA", "APPDATA", "ProgramFiles"):
        monkeypatch.delenv(var, raising=False)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    return tmp_path


def _make_exe(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


# --------------------------------------------------------------------------- find_adb
def test_find_adb_none_when_absent(clean_env):
    assert quest.find_adb() is None


def test_find_adb_in_data_root_tools(clean_env):
    exe = _make_exe(quest.tools_dir() / "platform-tools" / ADB)
    assert quest.find_adb() == str(exe)


def test_find_adb_android_home(clean_env, monkeypatch, tmp_path):
    sdk = tmp_path / "sdk"
    exe = _make_exe(sdk / "platform-tools" / ADB)
    monkeypatch.setenv("ANDROID_HOME", str(sdk))
    assert quest.find_adb() == str(exe)


def test_find_adb_path_wins_over_tools(clean_env, monkeypatch, tmp_path):
    _make_exe(quest.tools_dir() / "platform-tools" / ADB)
    bindir = tmp_path / "bin"
    exe = _make_exe(bindir / ADB)
    monkeypatch.setenv("PATH", str(bindir))
    assert Path(quest.find_adb()) == exe


def test_find_adb_env_override(clean_env, monkeypatch, tmp_path):
    exe = _make_exe(tmp_path / "custom" / ADB)
    monkeypatch.setenv("VRPHOTO_ADB", str(exe))
    assert quest.find_adb() == str(exe)


def test_find_adb_localappdata_on_windows(clean_env, monkeypatch, tmp_path):
    monkeypatch.setattr(quest, "_is_windows", lambda: True)
    local = tmp_path / "LocalAppData"
    exe = local / "Android" / "Sdk" / "platform-tools" / "adb.exe"
    _make_exe(exe)
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    assert quest.find_adb() == str(exe)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX exec bit")
def test_find_adb_ignores_non_executable(clean_env):
    p = quest.tools_dir() / "platform-tools" / "adb"
    p.parent.mkdir(parents=True)
    p.write_text("x", encoding="utf-8")
    p.chmod(0o644)
    assert quest.find_adb() is None


# --------------------------------------------------------------------------- graceful without adb
def test_functions_degrade_without_adb(monkeypatch, tmp_path):
    monkeypatch.setattr(quest, "find_adb", lambda: None)

    def boom(*a, **k):  # pragma: no cover - must not be reached
        raise AssertionError("subprocess must not run without adb")

    monkeypatch.setattr(quest._net, "_run", boom)
    f = tmp_path / "x.mp4"
    f.write_bytes(b"x")
    assert quest.adb_devices() == []
    assert quest.adb_device_states() == {}
    assert quest.adb_reverse(8080) is False
    assert quest.adb_reverse_remove(8080) is False
    assert quest.adb_push(f) is False
    assert quest.open_url_on_quest("http://localhost:8080/") is False


# --------------------------------------------------------------------------- fake adb
class FakeAdb:
    """Records adb invocations and answers like the real tool."""

    DEVICES = (
        "* daemon not running; starting now at tcp:5037\n"
        "List of devices attached\n"
        "2G0YC5ZG8L0123  device usb:1-1 product:eureka model:Quest_3 device:eureka transport_id:3\n"
        "R58N12345      device usb:1-2 product:a52 model:SM_A525F device:a52q transport_id:4\n"
        "1WMHH000000000 unauthorized usb:1-3 transport_id:5\n"
        "ZZ0000         no permissions (missing udev rules? user is in the plugdev group); "
        "see [http://developer.android.com/tools/device.html] usb:1-4 transport_id:6\n"
        "\n"
    )

    def __init__(self, devices: str | None = None):
        self.calls: list[list[str]] = []
        self.devices = self.DEVICES if devices is None else devices
        self.fail: dict[str, tuple[int, str, str]] = {}

    def __call__(self, args, timeout=10.0):
        self.calls.append(list(args))
        sub = args[3] if len(args) > 3 and args[1] == "-s" else args[1]
        if sub in self.fail:
            rc, out, err = self.fail[sub]
            return subprocess.CompletedProcess(args, rc, out, err)
        if sub == "devices":
            return subprocess.CompletedProcess(args, 0, self.devices, "")
        if sub == "shell" and "am start" in args[-1]:
            return subprocess.CompletedProcess(args, 0, "Starting: Intent { act=android.intent.action.VIEW }\n", "")
        return subprocess.CompletedProcess(args, 0, "", "")


@pytest.fixture
def fake_adb(monkeypatch):
    fake = FakeAdb()
    monkeypatch.setattr(quest, "find_adb", lambda: "/fake/adb")
    monkeypatch.setattr(quest, "_ensure_server", lambda adb: None)
    monkeypatch.setattr(quest._net, "_run", fake)
    return fake


def test_adb_devices_parsing(fake_adb):
    assert quest.adb_devices() == ["2G0YC5ZG8L0123", "R58N12345"]
    states = quest.adb_device_states()
    assert states["1WMHH000000000"] == "unauthorized"
    assert states["ZZ0000"].startswith("no permissions")
    assert fake_adb.calls[0] == ["/fake/adb", "devices", "-l"]


def test_adb_devices_failure_is_empty(fake_adb):
    fake_adb.fail["devices"] = (1, "", "error: cannot connect to daemon")
    assert quest.adb_devices() == []


def test_adb_reverse_picks_quest_among_devices(fake_adb):
    assert quest.adb_reverse(8080) is True
    assert fake_adb.calls[-1] == ["/fake/adb", "-s", "2G0YC5ZG8L0123", "reverse", "tcp:8080", "tcp:8080"]


def test_adb_reverse_explicit_serial_and_failure(fake_adb):
    assert quest.adb_reverse(9000, serial="R58N12345") is True
    assert fake_adb.calls[-1][:3] == ["/fake/adb", "-s", "R58N12345"]
    fake_adb.fail["reverse"] = (1, "", "error: device unauthorized")
    assert quest.adb_reverse(9000, serial="R58N12345") is False
    assert quest.adb_reverse_remove(9000, serial="R58N12345") is False


@pytest.mark.parametrize("port", [0, 70000, "abc", None])
def test_adb_reverse_invalid_port(fake_adb, port):
    assert quest.adb_reverse(port) is False


def test_no_authorised_device(monkeypatch):
    fake = FakeAdb(devices="List of devices attached\n1WMHH000000000\tunauthorized\n")
    monkeypatch.setattr(quest, "find_adb", lambda: "/fake/adb")
    monkeypatch.setattr(quest, "_ensure_server", lambda adb: None)
    monkeypatch.setattr(quest._net, "_run", fake)
    assert quest.adb_devices() == []
    assert quest.adb_reverse(8080) is False
    assert quest.open_url_on_quest("https://192.168.1.5:8443/") is False


def test_open_url_on_quest_quotes_url(fake_adb):
    url = "https://192.168.1.5:8443/view.html?scene=beach-01&x=1"
    assert quest.open_url_on_quest(url) is True
    call = fake_adb.calls[-1]
    assert call[:4] == ["/fake/adb", "-s", "2G0YC5ZG8L0123", "shell"]
    assert call[4] == (
        "am start -a android.intent.action.VIEW -d "
        "'https://192.168.1.5:8443/view.html?scene=beach-01&x=1' com.oculus.browser"
    )


def test_open_url_falls_back_to_default_browser(fake_adb, monkeypatch):
    def run(args, timeout=10.0):
        fake_adb.calls.append(list(args))
        if args[1] == "devices":
            return subprocess.CompletedProcess(args, 0, FakeAdb.DEVICES, "")
        if "com.oculus.browser" in args[-1]:
            return subprocess.CompletedProcess(args, 0, "Error: Activity not started, unable to resolve Intent", "")
        return subprocess.CompletedProcess(args, 0, "Starting: Intent", "")

    monkeypatch.setattr(quest._net, "_run", run)
    assert quest.open_url_on_quest("http://localhost:8080/") is True
    assert not fake_adb.calls[-1][-1].endswith("com.oculus.browser")


@pytest.mark.parametrize("url", ["file:///sdcard/x", "javascript:alert(1)", "", "http://a b/"])
def test_open_url_rejects_bad_urls(fake_adb, url):
    assert quest.open_url_on_quest(url) is False
    assert not fake_adb.calls


def test_adb_push_success(fake_adb, monkeypatch, tmp_path):
    f = tmp_path / "beach_LR.mp4"
    f.write_bytes(b"v" * 1000)
    pushed = {}

    def fake_cancellable(cmd, timeout, cancel, on_line=None):
        pushed["cmd"] = cmd
        pushed["timeout"] = timeout
        return 0, "1 file pushed"

    monkeypatch.setattr(quest, "_run_cancellable", fake_cancellable)
    prog = []
    assert quest.adb_push(f, progress=lambda fr, msg: prog.append(fr)) is True
    assert pushed["cmd"] == ["/fake/adb", "-s", "2G0YC5ZG8L0123", "push", str(f), "/sdcard/Movies/beach_LR.mp4"]
    assert pushed["timeout"] >= 120
    shell_cmds = [c[-1] for c in fake_adb.calls if "shell" in c]
    assert "mkdir -p /sdcard/Movies" in shell_cmds
    assert any("MEDIA_SCANNER_SCAN_FILE" in c and "file:///sdcard/Movies/beach_LR.mp4" in c for c in shell_cmds)
    assert prog[0] == 0.0 and prog[-1] == 1.0


def test_adb_push_custom_dir_with_spaces(fake_adb, monkeypatch, tmp_path):
    f = tmp_path / "my photo.jpg"
    f.write_bytes(b"j")
    seen = {}
    monkeypatch.setattr(quest, "_run_cancellable",
                        lambda cmd, timeout, cancel, on_line=None: seen.setdefault("cmd", cmd) and (0, ""))
    assert quest.adb_push(f, "/sdcard/Pictures/VR Photos/", serial="R58N12345") is True
    assert seen["cmd"][-1] == "/sdcard/Pictures/VR Photos/my photo.jpg"
    assert "mkdir -p '/sdcard/Pictures/VR Photos'" in [c[-1] for c in fake_adb.calls if "shell" in c]


def test_adb_push_failures(fake_adb, monkeypatch, tmp_path):
    assert quest.adb_push(tmp_path / "missing.mp4") is False
    f = tmp_path / "a.mp4"
    f.write_bytes(b"x")
    monkeypatch.setattr(quest, "_run_cancellable", lambda cmd, timeout, cancel, on_line=None: (1, "error: no space"))
    assert quest.adb_push(f) is False
    monkeypatch.setattr(quest, "_run_cancellable", lambda cmd, timeout, cancel, on_line=None: None)
    assert quest.adb_push(f) is False


# --------------------------------------------------------------------------- subprocess runner
def test_run_cancellable_collects_lines():
    lines = []
    res = quest._run_cancellable([sys.executable, "-c", "print('a'); print('b'); import sys; sys.exit(3)"],
                                 30, None, lines.append)
    assert res == (3, "a\nb") and lines == ["a", "b"]


def test_run_cancellable_cancel_kills_process():
    tok = CancelToken()
    threading.Timer(0.3, tok.cancel).start()
    t0 = time.monotonic()
    with pytest.raises(Cancelled):
        quest._run_cancellable([sys.executable, "-c", "import time; time.sleep(30)"], 60, tok)
    assert time.monotonic() - t0 < 10


def test_run_cancellable_timeout_and_missing_program():
    assert quest._run_cancellable([sys.executable, "-c", "import time; time.sleep(30)"], 0.5, None) is None
    assert quest._run_cancellable(["definitely-not-a-program-xyz"], 5, None) is None


# --------------------------------------------------------------------------- platform-tools install
def _zip(path: Path, entries: dict[str, bytes], modes: dict[str, int] | None = None) -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries.items():
            info = zipfile.ZipInfo(name)
            mode = (modes or {}).get(name, 0o100755 if name.endswith(("adb", "adb.exe")) else 0o100644)
            info.external_attr = mode << 16
            zf.writestr(info, data)
    return path


def _pt_zip(tmp_path: Path, payload: bytes = b"ADB-BINARY") -> Path:
    return _zip(
        tmp_path / "pt.zip",
        {
            f"platform-tools/{ADB}": payload,
            "platform-tools/fastboot": b"FB",
            "platform-tools/lib64/libc++.so": b"lib",
            "platform-tools/NOTICE.txt": b"notice",
        },
    )


def test_platform_tools_url_per_os(monkeypatch):
    monkeypatch.setattr(quest, "_is_windows", lambda: True)
    assert quest.platform_tools_url().endswith("platform-tools-latest-windows.zip")
    monkeypatch.setattr(quest, "_is_windows", lambda: False)
    monkeypatch.setattr(quest.sys, "platform", "linux")
    assert quest.platform_tools_url().endswith("platform-tools-latest-linux.zip")
    monkeypatch.setattr(quest.sys, "platform", "darwin")
    assert quest.platform_tools_url().startswith("https://dl.google.com/android/repository/")
    assert quest.platform_tools_url().endswith("-darwin.zip")


def test_install_platform_tools_from_archive(clean_env, monkeypatch, tmp_path):
    z = _pt_zip(tmp_path)
    monkeypatch.setattr(quest, "platform_tools_url", lambda: z.as_uri())
    prog: list[float] = []
    adb = quest.install_platform_tools(lambda f, m: prog.append(f))
    p = Path(adb)
    assert p == quest.tools_dir() / "platform-tools" / ADB
    assert p.read_bytes() == b"ADB-BINARY"
    assert (p.parent / "lib64" / "libc++.so").is_file()
    if sys.platform != "win32":
        assert os.stat(p).st_mode & stat.S_IXUSR
        assert not os.stat(p.parent / "NOTICE.txt").st_mode & stat.S_IXUSR
    assert prog and prog[-1] == 1.0
    assert all(b >= a - 1e-9 for a, b in zip(prog, prog[1:]))
    assert quest.find_adb() == adb
    leftovers = [x.name for x in quest.tools_dir().iterdir() if x.name != "platform-tools"]
    assert leftovers == []


def test_install_replaces_existing_copy(clean_env, monkeypatch, tmp_path):
    old = _make_exe(quest.tools_dir() / "platform-tools" / ADB)
    (old.parent / "stale.txt").write_text("old", encoding="utf-8")
    calls = []
    monkeypatch.setattr(quest._net, "_run", lambda args, timeout=10.0: calls.append(args))
    z = _pt_zip(tmp_path, b"NEW")
    monkeypatch.setattr(quest, "platform_tools_url", lambda: z.as_uri())
    adb = quest.install_platform_tools()
    assert Path(adb).read_bytes() == b"NEW"
    assert not (Path(adb).parent / "stale.txt").exists()
    assert calls and calls[0][-1] == "kill-server"


def test_install_cancel_leaves_nothing(clean_env, monkeypatch, tmp_path):
    z = _pt_zip(tmp_path)
    monkeypatch.setattr(quest, "platform_tools_url", lambda: z.as_uri())
    tok = CancelToken()
    tok.cancel()
    with pytest.raises(Cancelled):
        quest.install_platform_tools(cancel=tok)
    assert not (quest.tools_dir() / "platform-tools").exists()
    assert [x.name for x in quest.tools_dir().iterdir()] == []


@pytest.mark.parametrize(
    "kind",
    ["corrupt", "no_platform_tools", "no_adb", "zip_slip", "missing_url"],
)
def test_install_bad_archives_raise_download_error(clean_env, monkeypatch, tmp_path, kind):
    z = tmp_path / "bad.zip"
    if kind == "corrupt":
        z.write_bytes(b"PK\x03\x04 this is not really a zip")
    elif kind == "no_platform_tools":
        _zip(z, {"other/adb": b"x"})
    elif kind == "no_adb":
        _zip(z, {"platform-tools/fastboot": b"x"})
    elif kind == "zip_slip":
        _zip(z, {f"platform-tools/{ADB}": b"x", "platform-tools/../../evil.txt": b"evil"})
    url = (tmp_path / "does-not-exist.zip").as_uri() if kind == "missing_url" else z.as_uri()
    monkeypatch.setattr(quest, "platform_tools_url", lambda: url)
    with pytest.raises(DownloadError) as ei:
        quest.install_platform_tools()
    assert str(ei.value)
    if kind == "missing_url":
        assert ei.value.hint
    assert not (quest.tools_dir() / "platform-tools").exists()
    assert not (quest.tools_dir().parent / "evil.txt").exists()
    assert not (tmp_path / "evil.txt").exists()


# --------------------------------------------------------------------------- MTP
def test_mtp_copy_false_off_windows(monkeypatch, tmp_path):
    monkeypatch.setattr(quest, "_is_windows", lambda: False)
    f = tmp_path / "a.mp4"
    f.write_bytes(b"x")
    assert quest.mtp_copy([f]) is False


def test_mtp_script_escaping(tmp_path):
    files = [Path(r"C:\Users\O'Brien\Videos\beach_LR.mp4"), Path("C:/x/ü photo.jpg")]
    s = quest._mtp_script(files, ["Movies", "VR's"], "Quest")
    assert r"'C:\Users\O''Brien\Videos\beach_LR.mp4'" in s
    assert "'Movies', 'VR''s'" in s
    assert "$hint = 'Quest'" in s
    assert "NameSpace(17)" in s and "CopyHere" in s and "Internal" in s
    assert "__FILES__" not in s and "__SUB__" not in s and "__HINT__" not in s


def _fake_ps(monkeypatch, code: str):
    seen = {}

    def ps_args(script, *, sta=False):
        seen["script"] = Path(script).read_text(encoding="utf-8-sig")
        seen["sta"] = sta
        return [sys.executable, "-c", code]

    monkeypatch.setattr(quest, "_is_windows", lambda: True)
    monkeypatch.setattr(quest._net, "_ps_args", ps_args)
    return seen


def test_mtp_copy_success_simulated(monkeypatch, tmp_path):
    files = []
    for name in ("a_LR.mp4", "b_360.jpg"):
        f = tmp_path / name
        f.write_bytes(b"x" * 10)
        files.append(f)
    code = "print('START 1 a_LR.mp4'); print('OK 1'); print('START 2 b_360.jpg'); print('OK 2'); print('DONE')"
    seen = _fake_ps(monkeypatch, code)
    prog = []
    assert quest.mtp_copy(files, "Movies/VR", progress=lambda f, m: prog.append((f, m))) is True
    assert seen["sta"] is True
    assert "'Movies', 'VR'" in seen["script"]
    assert str(files[0].resolve()).replace("'", "''") in seen["script"]
    assert prog[-1][0] == 1.0 and prog[0][0] == 0.0
    assert any("b_360.jpg" in m for _, m in prog)


def test_mtp_copy_no_device_simulated(monkeypatch, tmp_path):
    f = tmp_path / "a.mp4"
    f.write_bytes(b"x")
    _fake_ps(monkeypatch, "import sys; print('ERR:NODEVICE:No Quest headset was found under This PC.'); sys.exit(2)")
    assert quest.mtp_copy([f]) is False


def test_mtp_copy_input_validation(monkeypatch, tmp_path):
    seen = _fake_ps(monkeypatch, "print('DONE')")
    f = tmp_path / "a.mp4"
    f.write_bytes(b"x")
    assert quest.mtp_copy([]) is False
    assert quest.mtp_copy([tmp_path / "missing.mp4"]) is False
    assert quest.mtp_copy([f], "../Movies") is False
    assert quest.mtp_copy([f], "C:/Movies") is False
    assert "script" not in seen


def test_mtp_copy_cancel(monkeypatch, tmp_path):
    f = tmp_path / "a.mp4"
    f.write_bytes(b"x")
    _fake_ps(monkeypatch, "import time; print('START 1 a.mp4', flush=True); time.sleep(30)")
    tok = CancelToken()
    threading.Timer(0.4, tok.cancel).start()
    with pytest.raises(Cancelled):
        quest.mtp_copy([f], cancel=tok)


def test_mtp_copy_without_powershell(monkeypatch, tmp_path):
    f = tmp_path / "a.mp4"
    f.write_bytes(b"x")
    monkeypatch.setattr(quest, "_is_windows", lambda: True)
    monkeypatch.setattr(quest._net, "_ps_args", lambda script, *, sta=False: None)
    assert quest.mtp_copy([f]) is False


# --------------------------------------------------------------------------- review additions
def test_find_adb_on_windows_ignores_batch_shims(clean_env, monkeypatch):
    """A PATH shim adb.cmd/adb.bat would run through cmd.exe, which re-parses '&' in URLs."""
    asked = []
    monkeypatch.setattr(quest, "_is_windows", lambda: True)
    monkeypatch.setattr(quest.shutil, "which", lambda name: asked.append(name) or None)
    assert quest.find_adb() is None
    assert asked == ["adb.exe"]


def test_adb_push_folder_goes_into_remote_dir(fake_adb, monkeypatch, tmp_path):
    album = tmp_path / "Beach album"
    album.mkdir()
    (album / "a_LR.jpg").write_bytes(b"a")
    seen = {}
    monkeypatch.setattr(quest, "_run_cancellable",
                        lambda cmd, timeout, cancel, on_line=None: seen.setdefault("cmd", cmd) and (0, ""))
    assert quest.adb_push(album, "/sdcard/Pictures") is True
    # pushing to /sdcard/Pictures/Beach album would nest a second copy on a repeated push
    assert seen["cmd"][-2:] == [str(album), "/sdcard/Pictures"]


@pytest.mark.parametrize("url", ['http://x/"&calc&"', "http://x/`id`", "http://x/<script>", "https://x/\u00e9"])
def test_open_url_rejects_characters_outside_rfc3986(fake_adb, url):
    assert quest.open_url_on_quest(url) is False
    assert not fake_adb.calls
    assert quest.last_error()


def test_last_error_explains_failures_and_clears_on_success(fake_adb, monkeypatch, tmp_path):
    monkeypatch.setattr(quest, "find_adb", lambda: None)
    assert quest.adb_reverse(8080) is False
    assert "adb" in quest.last_error() and "Install" in quest.last_error()

    monkeypatch.setattr(quest, "find_adb", lambda: "/fake/adb")
    fake_adb.devices = "List of devices attached\n1WMHH000000000\tunauthorized\n"
    assert quest.adb_reverse(8080) is False
    assert "Allow" in quest.last_error()

    fake_adb.devices = "List of devices attached\nZZ00\tno permissions (missing udev rules?)\n"
    assert quest.open_url_on_quest("http://localhost:8080/") is False
    assert "udev" in quest.last_error()

    fake_adb.devices = "List of devices attached\n\n"
    assert quest.adb_reverse(8080) is False
    assert "developer mode" in quest.last_error()

    fake_adb.devices = FakeAdb.DEVICES
    fake_adb.fail["reverse"] = (1, "", "error: closed")
    assert quest.adb_reverse(8080) is False
    assert "error: closed" in quest.last_error()

    del fake_adb.fail["reverse"]
    assert quest.adb_reverse(8080) is True
    assert quest.last_error() == ""

    f = tmp_path / "a.mp4"
    f.write_bytes(b"x")
    monkeypatch.setattr(quest, "_run_cancellable",
                        lambda cmd, timeout, cancel, on_line=None: (1, "adb: error: failed to copy\nNo space left"))
    assert quest.adb_push(f) is False
    assert "No space left" in quest.last_error() and "free space" in quest.last_error()


def test_last_error_is_per_thread(fake_adb, monkeypatch):
    monkeypatch.setattr(quest, "find_adb", lambda: None)
    assert quest.adb_reverse(8080) is False
    other: list[str] = []
    t = threading.Thread(target=lambda: other.append(quest.last_error()))
    t.start()
    t.join()
    assert other == [""] and quest.last_error()


def test_install_interrupted_chunked_download_is_download_error(clean_env, monkeypatch):
    import http.client

    class Resp:
        headers: dict = {}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, n):
            raise http.client.IncompleteRead(b"partial")

    monkeypatch.setattr(quest.urllib.request, "urlopen", lambda req, timeout=30: Resp())
    with pytest.raises(DownloadError) as ei:
        quest.install_platform_tools()
    assert ei.value.hint
    assert [x.name for x in quest.tools_dir().iterdir()] == []


def test_install_rejects_zip_bomb(clean_env, monkeypatch, tmp_path):
    z = _pt_zip(tmp_path)
    monkeypatch.setattr(quest, "platform_tools_url", lambda: z.as_uri())
    monkeypatch.setattr(quest, "_MAX_UNPACKED", 10)
    with pytest.raises(DownloadError):
        quest.install_platform_tools()
    assert not (quest.tools_dir() / "platform-tools").exists()


def test_install_unwritable_tools_dir_is_download_error(clean_env, monkeypatch, tmp_path):
    blocker = tmp_path / "file-not-dir"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.setattr(quest, "tools_dir", lambda: blocker / "tools")
    with pytest.raises(DownloadError) as ei:
        quest.install_platform_tools()
    assert ei.value.hint


def test_mtp_script_handles_existing_files_and_large_sizes():
    s = quest._mtp_script([Path("C:/v/a.mp4")], ["Movies"])
    assert "ExtendedProperty('System.Size')" in s  # FolderItem.Size is 32-bit
    assert "SKIP $i $leaf" in s  # identical file already on the headset
    assert "$cur -eq $size" in s and "-ge $size" not in s


def test_mtp_timeout_outlasts_script_deadlines(monkeypatch, tmp_path):
    files = []
    for i in range(10):
        f = tmp_path / f"f{i}.jpg"
        f.write_bytes(b"x" * 100)
        files.append(f)
    seen = {}
    monkeypatch.setattr(quest, "_is_windows", lambda: True)
    monkeypatch.setattr(quest._net, "_ps_args", lambda script, *, sta=False: ["ps", str(script)])
    monkeypatch.setattr(quest, "_run_cancellable",
                        lambda cmd, timeout, cancel, on_line=None: seen.setdefault("t", timeout) and (0, "DONE"))
    assert quest.mtp_copy(files) is True
    script_worst_case = sum(120 + 100 / (2 * 1024 * 1024) for _ in files) + 0.5 * len(files) + 2
    assert seen["t"] > script_worst_case


def test_mtp_failure_sets_last_error_with_hint(monkeypatch, tmp_path):
    f = tmp_path / "a.mp4"
    f.write_bytes(b"x")
    _fake_ps(monkeypatch, "import sys; print('ERR:NODEVICE:No Quest headset was found under This PC.'); sys.exit(2)")
    assert quest.mtp_copy([f]) is False
    msg = quest.last_error()
    assert "No Quest headset" in msg and "USB cable" in msg
    _fake_ps(monkeypatch, "print('START 1 a.mp4'); print('OK 1'); print('DONE')")
    assert quest.mtp_copy([f]) is True and quest.last_error() == ""


def _find_powershell() -> str | None:
    import shutil

    return os.environ.get("VRPHOTO_PWSH") or shutil.which("pwsh") or (
        shutil.which("powershell") if sys.platform == "win32" else None)


_SHELL_MOCK = r"""
class MockItem {
  [string]$Name; [bool]$IsFolder; [object]$GetFolder; [object]$SizeVal; [int64]$Size
  MockItem([string]$n, [bool]$f, [object]$folder, [object]$sz) {
    $this.Name = $n; $this.IsFolder = $f; $this.GetFolder = $folder; $this.SizeVal = $sz
    if ($null -ne $sz) { $this.Size = [int64]$sz }
  }
  [object] ExtendedProperty([string]$p) { return $this.SizeVal }
}
class MockFolder {
  [System.Collections.ArrayList]$List = (New-Object System.Collections.ArrayList)
  [object[]] Items() { return $this.List.ToArray() }
  [void] NewFolder([string]$n) { [void]$this.List.Add([MockItem]::new($n, $true, [MockFolder]::new(), $null)) }
  [void] CopyHere([object]$f, [int]$flags) {
    $leaf = [System.IO.Path]::GetFileName($f)
    [Console]::Error.WriteLine("COPY $leaf")
    foreach ($o in @($this.List | Where-Object { $_.Name -eq $leaf })) { $this.List.Remove($o) }
    [void]$this.List.Add([MockItem]::new($leaf, $false, $null, (Get-Item -LiteralPath $f).Length))
  }
}
$movies = [MockFolder]::new()
[void]$movies.List.Add([MockItem]::new('same.jpg', $false, $null, 1000))
[void]$movies.List.Add([MockItem]::new('changed.jpg', $false, $null, 7))
$root = [MockFolder]::new(); [void]$root.List.Add([MockItem]::new('Movies', $true, $movies, $null))
$dev = [MockFolder]::new(); [void]$dev.List.Add([MockItem]::new('Internal shared storage', $true, $root, $null))
$pcf = [MockFolder]::new()
[void]$pcf.List.Add([MockItem]::new('Local Disk (C:)', $true, $null, $null))
[void]$pcf.List.Add([MockItem]::new('Quest 3', $true, $dev, $null))
$global:MockShell = [pscustomobject]@{ PC = $pcf }
$global:MockShell | Add-Member ScriptMethod NameSpace { param($n) if ($n -eq 17) { return $this.PC } }
"""


def test_mtp_script_runs_in_real_powershell_with_mocked_shell(tmp_path):
    """Executes the generated MTP script (mocked Shell.Application): copies new files, waits
    for the exact size, skips an identical existing file, re-copies a changed one, and
    survives typographic quotes in file names.  Skipped without PowerShell."""
    ps = _find_powershell()
    if not ps:
        pytest.skip("PowerShell not available")
    src = tmp_path / "src"
    src.mkdir()
    (src / "Mom\u2019s \u2018clip\u2019 $(x).mp4").write_bytes(b"v" * 3000)
    (src / "same.jpg").write_bytes(b"s" * 1000)
    (src / "changed.jpg").write_bytes(b"c" * 500)
    files = sorted(src.iterdir())
    script = quest._mtp_script(files, ["Movies"], "Quest")
    script = script.replace("$shell = New-Object -ComObject Shell.Application", "$shell = $global:MockShell")
    script = script.replace("Start-Sleep -Seconds 2", "Start-Sleep -Milliseconds 10")
    path = tmp_path / "mtp.ps1"
    path.write_text(_SHELL_MOCK + script, encoding="utf-8-sig")
    cp = subprocess.run([ps, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(path)],
                        capture_output=True, text=True, encoding="utf-8", timeout=90)
    out = cp.stdout.splitlines()
    assert cp.returncode == 0 and out[-1] == "DONE", (cp.stdout, cp.stderr)
    copied = [line[5:] for line in cp.stderr.splitlines() if line.startswith("COPY ")]
    assert sorted(copied) == sorted(["changed.jpg", "Mom\u2019s \u2018clip\u2019 $(x).mp4"])
    assert any(line.startswith("SKIP") and line.endswith("same.jpg") for line in out)
