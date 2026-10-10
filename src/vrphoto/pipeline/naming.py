"""Output file names: Quest media-player tokens, Windows-safe sanitising, unique paths.

Meta Quest players (the built-in Files/Media app, Skybox, DeoVR, Pigasus, ...) recognise the
projection of a file from tokens at the end of its name, e.g. ``beach_180_LR.jpg`` is a
VR180 side-by-side image and ``beach_360_TB.mp4`` a stereo 360 top-bottom video.
:func:`output_name` appends the right token for each output kind; :func:`unique_path` never
overwrites an earlier result (``beach (2)_LR.jpg``, keeping the token at the end so players
still recognise the file).

Names are made safe for Windows (the primary platform): characters ``<>:"/\\|?*`` and
control characters are replaced, trailing dots/spaces removed, reserved device names
(``CON``, ``NUL``, ``COM1``...) escaped and long names shortened.
"""

from __future__ import annotations

import os
import re
import threading
import unicodedata
from collections.abc import Iterable
from pathlib import Path

from ..core.errors import InputError

#: Output kind -> filename token (appended to the base name, before the extension).
KIND_TOKENS: dict[str, str] = {
    # flat stereo photos / videos ("full" and "half" SBS share _LR: players infer from aspect)
    "sbs": "_LR",
    "half_sbs": "_LR",
    "lr": "_LR",
    "tb": "_TB",
    "half_tb": "_TB",
    "cross": "_cross",
    "anaglyph": "_anaglyph",
    "anaglyph_gray": "_anaglyph_gray",
    "left": "_left",
    "right": "_right",
    "mono": "",
    "flat": "",
    # VR180 / 360
    "vr180": "_180_LR",
    "vr180_sbs": "_180_LR",
    "vr180_tb": "_180_TB",
    "360_tb": "_360_TB",
    "pano_stereo": "_360_TB",
    "360_lr": "_360_LR",
    "360": "_360",
    "pano_mono": "_360",
    # parallax ("living photo") videos
    "parallax": "_parallax",
    "parallax_flat": "_parallax",
    "parallax_sbs": "_parallax_LR",
    "parallax_vr180": "_parallax_180_LR",
    # depth maps and previews
    "depth": "_depth",
    "depth16": "_depth",
    "depth_color": "_depth_color",
    "wiggle": "_wiggle",
    # 3D outputs
    "scene": "_3d",
    "object": "_object",
    "cutout": "_cutout",
    "splat": "_splat",
    "capture": "_capture",
}

#: Tokens a Quest player looks for at the END of a name (longest first for matching).
PLAYER_TOKENS: tuple[str, ...] = tuple(sorted({t for t in KIND_TOKENS.values() if t}, key=len, reverse=True))

#: Longest file name (base + token + extension) we create.  Windows allows 255 characters
#: per component but paths are limited to 260 characters by default, so stay well below.
MAX_NAME_CHARS = 120

_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f\x7f]')
_RESERVED = (
    {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$", "CLOCK$"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
    | {f"COM{c}" for c in "¹²³"}
    | {f"LPT{c}" for c in "¹²³"}
)
_COUNTER = re.compile(r"^(?P<stem>.*?) \((?P<n>\d{1,3})\)$")

_reserve_lock = threading.Lock()
_reserved: set[str] = set()


def sanitize_filename(name: str, *, max_chars: int = MAX_NAME_CHARS, fallback: str = "untitled") -> str:
    """Make ``name`` (a single path component, no folders) safe on Windows and Linux.

    Replaces illegal and control characters with ``_``, collapses whitespace, strips
    leading/trailing spaces and trailing dots, escapes reserved device names (``CON`` ->
    ``_CON``, also with an extension: ``nul.txt`` -> ``_nul.txt``) and shortens the name to
    ``max_chars`` characters (keeping the extension).  Never returns an empty string.
    """
    text = unicodedata.normalize("NFC", str(name or ""))
    text = re.sub(r"[\t\n\r\v\f]+", " ", text)  # whitespace controls become spaces
    text = _ILLEGAL.sub("_", text)
    text = re.sub(r"\s+", " ", text).strip().rstrip(". ")
    if not text or set(text) <= {".", "_", " "}:
        text = fallback
    stem, dot, ext = text.rpartition(".")
    if not dot or not stem or len(ext) > 10 or " " in ext:
        stem, ext = text, ""
    if stem.split(".")[0].upper().rstrip(" ") in _RESERVED:
        stem = "_" + stem
    max_chars = max(8, int(max_chars))
    suffix = f".{ext}" if ext else ""
    if len(stem) + len(suffix) > max_chars:
        stem = stem[: max(1, max_chars - len(suffix))].rstrip(". ")
        if not stem:
            stem = fallback[: max(1, max_chars - len(suffix))]
    return stem + suffix


def safe_base(base: str, *, max_chars: int = MAX_NAME_CHARS) -> str:
    """A sanitised base name (no extension handling: dots in ``base`` are kept)."""
    out = sanitize_filename(f"{base}.x", max_chars=max_chars + 2)
    return out[:-2] if out.endswith(".x") else out


def token_for(kind: str) -> str:
    """Filename token for an output ``kind`` (see :data:`KIND_TOKENS`).

    A ``kind`` starting with ``_`` is used literally as the token.  Raises
    :class:`InputError` for unknown kinds.
    """
    k = str(kind or "").strip()
    if k.startswith("_"):
        return safe_base(k)
    key = k.lower()
    if key not in KIND_TOKENS:
        raise InputError(
            f"Unknown output kind {kind!r}.", hint=f"Known kinds: {', '.join(sorted(KIND_TOKENS))}."
        )
    return KIND_TOKENS[key]


def output_name(base: str, kind: str, ext: str) -> str:
    """File name for an output: ``<base><token>.<ext>``, safe for Windows.

    ``kind`` selects the Quest player token (``"sbs"`` -> ``_LR``, ``"vr180"`` -> ``_180_LR``,
    ``"360_tb"`` -> ``_360_TB``, ``"360"`` -> ``_360``, ``"anaglyph"`` -> ``_anaglyph``,
    ``"parallax"`` -> ``_parallax``...).  ``ext`` may be given with or without the dot.  Long
    base names are shortened so the token and extension always survive.
    """
    token = token_for(kind)
    e = str(ext or "").strip().lstrip(".").lower()
    e = re.sub(r"[^a-z0-9]+", "", e)
    suffix = token + (f".{e}" if e else "")
    room = max(1, MAX_NAME_CHARS - len(suffix))
    stem = safe_base(base or "untitled", max_chars=room)
    stem = stem[:room].rstrip(". ") or "untitled"[:room]
    return stem + suffix


def split_token(stem: str) -> tuple[str, str]:
    """Split a file stem into ``(base, trailing player token)`` (token may be ``""``)."""
    for tok in PLAYER_TOKENS:
        if stem.endswith(tok) and len(stem) > len(tok):
            return stem[: -len(tok)], tok
    return stem, ""


def _taken(p: Path) -> bool:
    return os.path.lexists(p) or os.path.normcase(str(p)) in _reserved


def unique_path(path: str | os.PathLike[str], *, reserve: bool = True) -> Path:
    """``path`` if it is free, else the first free ``name (2).ext``, ``name (3).ext``...

    The counter goes before a trailing Quest token (``beach_180_LR.jpg`` ->
    ``beach (2)_180_LR.jpg``) so media players still recognise the format.  With
    ``reserve`` (default) the returned name is remembered for the rest of the process, so
    jobs running in parallel never pick the same file even before it is written.
    """
    p = Path(path)
    parent = p.parent
    ext = "".join(p.suffixes[-1:]) if p.suffix else ""
    stem = p.name[: len(p.name) - len(ext)] if ext else p.name
    base, token = split_token(stem)
    m = _COUNTER.match(base)
    if m:
        base = m.group("stem")
    with _reserve_lock:
        cand = p
        n = 2
        while _taken(cand):
            cand = parent / f"{base} ({n}){token}{ext}"
            n += 1
            if n > 100_000:  # pragma: no cover - pathological
                raise InputError(f"Could not find a free file name for {p.name} in {parent}.")
        if reserve:
            _reserved.add(os.path.normcase(str(cand)))
    return cand


def release_reserved(paths: Iterable[str | os.PathLike[str]] | None = None) -> None:
    """Forget reserved names: the given ``paths`` (e.g. when a job ends - written files then
    protect their names themselves), or all of them when ``paths`` is None."""
    with _reserve_lock:
        if paths is None:
            _reserved.clear()
            return
        for p in paths:
            _reserved.discard(os.path.normcase(str(Path(p))))


__all__ = [
    "KIND_TOKENS",
    "PLAYER_TOKENS",
    "MAX_NAME_CHARS",
    "sanitize_filename",
    "safe_base",
    "token_for",
    "output_name",
    "split_token",
    "unique_path",
    "release_reserved",
]
