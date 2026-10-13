"""The cache of AI-generated back views (pipeline/view_cache.py, design 3.5.3 "Cache")."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from vrphoto.pipeline import view_cache as vc


def _set(folder: Path, seed: int = 0, n: int = 3, size: int = 1000, *, cameras: bool = True) -> list[Path]:
    """A worker-like set: ``view_<k>.png`` files and a ``cameras.json`` naming them."""
    folder.mkdir(parents=True, exist_ok=True)
    files = []
    for k in range(n):
        p = folder / f"view_{k}.png"
        p.write_bytes(bytes([k % 256]) * size)
        files.append(p)
    cams = folder / "cameras.json"
    if cameras:
        doc = {"seed": seed, "views": [{"file": p.name, "phi_deg": 60.0 * k} for k, p in enumerate(files)]}
        cams.write_text(json.dumps(doc), encoding="utf-8")
    return [cams, *files]


def _key(seed: int = 0, **kw) -> str:
    params = {"steps": 30, "guidance": 3.0, "prompt": "a person", "seed": seed, **kw}
    return vc.make_key("ab" * 32, source="mvadapter", params=params, worker="1.0.0+abc")


def test_keys_name_everything_the_views_depend_on():
    k = _key()
    assert len(k) == 40 and all(c in "0123456789abcdef" for c in k)
    assert k == vc.make_key(
        "ab" * 32,
        source="mvadapter",
        params={"seed": 0, "prompt": "a person", "guidance": 3.0, "steps": 30},  # order does not matter
        worker="1.0.0+abc",
    )
    others = {
        _key(seed=1),
        _key(steps=50),
        _key(prompt="grey hoodie"),
        _key(guidance=4.0),
        vc.make_key("cd" * 32, source="mvadapter", params={"seed": 0}, worker="1.0.0+abc"),
        vc.make_key("ab" * 32, source="mvadapter", params={"steps": 30, "guidance": 3.0, "prompt": "a person",
                                                          "seed": 0}, worker="1.0.1+abc"),
        vc.make_key("ab" * 32, source="qwen", params={"steps": 30, "guidance": 3.0, "prompt": "a person",
                                                     "seed": 0}, worker="1.0.0+abc"),
    }  # fmt: skip
    assert k not in others and len(others) == 7


def test_worker_version_hashes_the_worker_script(tmp_path):
    script = tmp_path / "w.py"
    script.write_text("print(1)", encoding="utf-8")
    m = SimpleNamespace(version="1.0.0", worker_path=lambda: script)
    v1 = vc.worker_version(m)
    assert v1.startswith("1.0.0+") and len(v1) == len("1.0.0+") + 12
    script.write_text("print(2)", encoding="utf-8")
    assert vc.worker_version(m) != v1
    assert vc.worker_version(SimpleNamespace(version="2", worker_path=lambda: None)) == "2"
    assert vc.worker_version(SimpleNamespace(version="3", worker_path=lambda: tmp_path / "gone.py")) == "3"


def test_put_and_get_round_trip(tmp_path):
    cache = vc.ViewCache(tmp_path / "cache")
    src = _set(tmp_path / "worker" / "views_0")
    key = _key()
    assert cache.get(key) is None and cache.misses == 1
    entry = cache.put(key, src, meta={"seed": 0})
    assert entry == cache.root / key
    got = cache.get(key)
    assert got == entry and cache.hits == 1 and key in cache and len(cache) == 1
    for p in src:  # byte-identical copies under their own names
        assert (got / p.name).read_bytes() == p.read_bytes()
    rec = json.loads((got / vc.ENTRY_FILE).read_text(encoding="utf-8"))
    assert rec["files"] == [p.name for p in src] and rec["meta"] == {"seed": 0} and rec["key"] == key
    # the worker's folder can go away: the cache has its own copies
    for p in src:
        p.unlink()
    assert cache.get(key) == entry
    # module-level shortcuts on the same folder
    assert vc.get(key, root=cache.root) == entry
    assert vc.put(_key(seed=5), _set(tmp_path / "w2"), root=cache.root) == cache.root / _key(seed=5)


def test_a_damaged_entry_can_be_discarded(tmp_path):
    cache = vc.ViewCache(tmp_path / "cache")
    key = _key()
    cache.put(key, _set(tmp_path / "w"))
    assert cache.discard(key) is True and cache.get(key) is None and len(cache) == 0
    assert cache.discard(key) is False and cache.discard("../evil") is False


def test_default_root_is_in_the_cache_folder(tmp_path):
    settings = SimpleNamespace(cache_dir=tmp_path / "c")
    assert vc.default_root(settings) == tmp_path / "c" / "backviews"
    from vrphoto.core.config import get_settings

    assert vc.default_root() == Path(get_settings().cache_dir) / vc.SUBDIR


def test_a_new_put_replaces_the_entry(tmp_path):
    cache = vc.ViewCache(tmp_path / "cache")
    key = _key()
    cache.put(key, _set(tmp_path / "a", size=10))
    cache.put(key, _set(tmp_path / "b", n=2, size=20))
    files = sorted(p.name for p in cache.get(key).iterdir())
    assert files == ["cameras.json", vc.ENTRY_FILE, "view_0.png", "view_1.png"]
    assert (cache.root / key / "view_0.png").stat().st_size == 20


def test_incomplete_or_foreign_entries_are_discarded(tmp_path):
    cache = vc.ViewCache(tmp_path / "cache")
    key = _key()
    entry = cache.put(key, _set(tmp_path / "a"))
    (entry / "view_1.png").unlink()  # a picture went missing
    assert cache.get(key) is None and not entry.exists()
    entry = cache.put(key, _set(tmp_path / "a"))
    (entry / vc.ENTRY_FILE).write_text("{not json", encoding="utf-8")
    assert key not in cache and cache.get(key) is None
    # a half-written folder without an entry record (a crash before the rename) is no hit
    (cache.root / _key(seed=9)).mkdir(parents=True)
    assert cache.get(_key(seed=9)) is None


def test_bad_keys_and_bad_file_lists_are_refused_quietly(tmp_path):
    cache = vc.ViewCache(tmp_path / "cache")
    src = _set(tmp_path / "a")
    for key in ("", "../x", "a/b", "x" * 200, None):
        assert cache.get(key) is None  # type: ignore[arg-type]
        assert cache.put(key, src) is None  # type: ignore[arg-type]
    other = tmp_path / "b" / "view_0.png"
    other.parent.mkdir()
    other.write_bytes(b"x")
    assert cache.put(_key(), [*src, other]) is None  # two files of one name
    assert cache.put(_key(), []) is None
    assert cache.put(_key(), [tmp_path / "missing.png"]) is None
    assert not cache.root.exists() or not any(cache.root.iterdir())
    with pytest.raises(ValueError):
        cache.path_for("../x")


def test_storing_can_be_disabled(tmp_path):
    cache = vc.ViewCache(tmp_path / "cache", max_bytes=0)
    assert cache.put(_key(), _set(tmp_path / "a")) is None and cache.get(_key()) is None


def test_least_recently_used_sets_go_first(tmp_path):
    cache = vc.ViewCache(tmp_path / "cache", max_bytes=11_000)
    keys = [_key(seed=i) for i in range(3)]
    t = time.time() - 100
    for i, key in enumerate(keys):
        cache.put(key, _set(tmp_path / f"s{i}", seed=i, n=3, size=1000))  # ~3 kB each
        os.utime(cache.root / key / vc.ENTRY_FILE, (t + i, t + i))
    assert cache.get(keys[0]) is not None  # used again: now the newest
    cache.put(_key(seed=3), _set(tmp_path / "s3", seed=3, n=3, size=1000))  # about 14 kB > 11 kB
    assert keys[1] not in cache  # the least recently used went first
    assert keys[0] in cache and keys[2] in cache and _key(seed=3) in cache
    assert cache.size_bytes() <= 11_000
    # a set larger than the whole cap is not stored (and evicts nothing)
    assert cache.put(_key(seed=4), _set(tmp_path / "big", n=1, size=20_000)) is None
    assert len(cache) == 3
    assert cache.prune(0) > 0 and len(cache) == 0


def test_old_temp_folders_of_a_crashed_write_are_removed(tmp_path):
    cache = vc.ViewCache(tmp_path / "cache")
    cache.root.mkdir(parents=True)
    old = cache.root / ".deadbeef.1234.tmp"
    new = cache.root / ".cafe.5678.tmp"
    for d in (old, new):
        d.mkdir()
        (d / "view_0.png").write_bytes(b"x")
    t = time.time() - 2 * 3600
    os.utime(old, (t, t))
    cache.prune()
    assert not old.exists() and new.exists()  # a write in progress is left alone
    assert len(cache) == 0  # temp folders are never entries
    cache.clear()
    assert not any(cache.root.iterdir())


def test_salvage_keeps_finished_sets_of_an_interrupted_run(tmp_path):
    cache = vc.ViewCache(tmp_path / "cache")
    out = tmp_path / "work" / "mvadapter"
    _set(out / "views_7", seed=7)
    _set(out / "views_8", seed=8)
    (out / "views_8" / "view_2.png").unlink()  # unfinished: a picture is missing
    _set(out / "views_9", seed=9)  # not asked for (another request)
    _set(out / "junk", seed=7, cameras=False)
    (out / "broken").mkdir()
    (out / "broken" / "cameras.json").write_text("{", encoding="utf-8")
    keys = {7: _key(seed=7), 8: _key(seed=8)}
    got = cache.salvage(out, keys)
    assert set(got) == {7} and got[7] == cache.root / keys[7]
    assert keys[7] in cache and keys[8] not in cache and len(cache) == 1
    assert json.loads((got[7] / vc.ENTRY_FILE).read_text(encoding="utf-8"))["meta"] == {"salvaged": True}
    # again: already cached, nothing copied twice; a missing folder salvages nothing
    assert cache.salvage(out, keys) == got
    assert cache.salvage(tmp_path / "nowhere", keys) == {}
    assert vc.salvage(out, {8: keys[8]}, root=cache.root) == {}


def test_parallel_puts_of_one_key_leave_one_complete_entry(tmp_path):
    cache = vc.ViewCache(tmp_path / "cache")
    key = _key()
    srcs = [_set(tmp_path / f"w{i}", size=500 + i) for i in range(6)]
    errors: list[BaseException] = []

    def put(src):
        try:
            cache.put(key, src)
        except BaseException as e:  # pragma: no cover - reported below
            errors.append(e)

    threads = [threading.Thread(target=put, args=(s,)) for s in srcs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert cache.get(key) is not None and len(cache) == 1
    assert not [p for p in cache.root.iterdir() if p.name.endswith(".tmp")]
